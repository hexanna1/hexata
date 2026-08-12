#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from typing import Iterator, Optional

from board import (
    MAX_BOARD_SIZE,
    MIN_BOARD_SIZE,
    Board,
    GameType,
    Move,
    MoveKind,
    Side,
    coord_to_human,
)
from engine import (
    AnalysisMove,
    KataHexEngine,
    board_to_engine_vertex,
    engine_swap_transform_active,
    map_coords_to_engine,
    map_side_to_engine,
    parse_analysis_move_token,
)
from formats import hexworld
from formats.hexworld import cell_to_col_row

STARTUP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.02
_FILTER_MOVE_RE = re.compile(r"[a-z]+[0-9]+")
_ANALYSIS_CONCURRENCY = 64


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _fail(msg: str) -> int:
    _emit({"ok": False, "error": msg})
    return 1


def _side_to_text(side: Side) -> str:
    return "red" if side == Side.RED else "blue"


def _round6(x: Optional[float]) -> Optional[float]:
    if x is None or not math.isfinite(x):
        return None
    return round(x, 6)


def _side_winrate_to_red(winrate: Optional[float], side: Side) -> Optional[float]:
    if winrate is None:
        return None
    return _round6(winrate if side == Side.RED else (1.0 - winrate))


def _parse_filter_move_stream(raw: str) -> tuple[tuple[int, int], ...]:
    if raw == "":
        raise ValueError("Move filter must include at least one move")
    toks = [m.group(0) for m in _FILTER_MOVE_RE.finditer(raw)]
    if "".join(toks) != raw:
        raise ValueError(f"Invalid move filter: {raw!r}")

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for tok in toks:
        key = cell_to_col_row(tok)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _parse_analyze_position_spec(raw: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    if "::" not in raw:
        return raw, ()
    position, moves = raw.rsplit("::", 1)
    return position, _parse_filter_move_stream(moves)


def _to_output_row(r: AnalysisMove, *, side_to_play: Side) -> dict:
    move = r.move.lower()
    if r.col is not None and r.row is not None:
        move = coord_to_human(r.col, r.row)
    return {
        "move": move,
        "rank": None if r.order is None else r.order + 1,
        "red_winrate": _side_winrate_to_red(r.winrate, side_to_play),
        "visits": r.visits,
        "prior": _round6(r.prior),
    }


def _move_to_text(mv) -> str:
    if mv.kind == MoveKind.PASS:
        return "pass"
    if mv.kind == MoveKind.SWAP:
        return "swap"
    return coord_to_human(mv.col, mv.row)


def _opening_line_to_text(line: tuple[tuple[int, int], ...]) -> str:
    return ",".join(coord_to_human(col, row) for col, row in line)


def _match_hexworld_url(board: Board) -> str:
    return hexworld.build_position_url(board.game_type, board.n, board.history)


def _input_hexworld_url(text: str, game_type: GameType) -> str:
    output_type = hexworld.url_game_type(text) or game_type
    return hexworld.position_url_from_hash(output_type, hexworld.extract_hash(text))


def _search_failure_error(status: str) -> str:
    if status == "engine_exited":
        return "Engine exited before analysis completed"
    return "No analysis update received from engine"


def _search_payload_from_moves(
    recs: list[AnalysisMove],
    *,
    side_to_play: Side,
    top_n: Optional[int],
) -> dict:
    moves = [_to_output_row(r, side_to_play=side_to_play) for r in recs]

    best = None
    if moves:
        m0 = moves[0]
        best = {
            "move": m0["move"],
            "red_winrate": m0["red_winrate"],
            "visits": m0["visits"],
        }
    if top_n is not None:
        moves = moves[:top_n]

    return {
        "method": "search",
        "best": best,
        "total_visits": _analysis_total_visits(recs),
        "moves": moves,
    }


def _final_analysis_payload(*, side: Side, analyze: dict) -> dict:
    return {
        "side": _side_to_text(side),
        "analyze": analyze,
    }


def _parse_openings(raw: str) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        line = (cell_to_col_row(tok),)
        if line in out:
            raise ValueError(f"Duplicate opening move: {tok}")
        out.append(line)
    return out


def _sample_weighted_match_move(
    moves: list[tuple[int, int]],
    weights: list[float],
    *,
    temp: float,
    rng: random.Random,
) -> Optional[tuple[int, int]]:
    if not moves:
        return None
    if temp == 0.0:
        best = max(weights)
        picks = [move for move, weight in zip(moves, weights) if weight == best]
        return picks[rng.randrange(len(picks))]
    scaled = [weight ** (1.0 / temp) for weight in weights]
    total = sum(scaled)
    if total <= 0.0:
        return moves[rng.randrange(len(moves))]
    target = rng.random() * total
    upto = 0.0
    for move, weight in zip(moves, scaled):
        upto += weight
        if upto >= target:
            return move
    return moves[-1]


def _sample_match_search_move(
    board: Board,
    recs: list[AnalysisMove],
    *,
    temp: float,
    rng: random.Random,
) -> Optional[tuple[int, int]]:
    legal: list[tuple[int, int]] = []
    weights: list[float] = []
    seen: set[tuple[int, int]] = set()
    for r in recs:
        if r.col is None or r.row is None:
            # Match games sample placement moves only.
            continue
        key = (r.col, r.row)
        if key in seen or not board.is_empty(r.col, r.row):
            continue
        seen.add(key)
        legal.append(key)
        weights.append(float(0 if r.visits is None else max(0, r.visits)))
    return _sample_weighted_match_move(legal, weights, temp=temp, rng=rng)


def _analysis_total_visits(recs: list[AnalysisMove]) -> int:
    return sum(0 if r.visits is None else max(0, r.visits) for r in recs)


def _run_match_search_for_seconds(
    engine: KataHexEngine,
    side: Side,
    seconds: float,
    *,
    interval_cs: int = 15,
) -> tuple[str, Optional[list[AnalysisMove]]]:
    engine.clear_analysis()
    engine.start_analysis(side, interval_cs)
    first_update_at: Optional[float] = None
    wait_started_at = time.monotonic()
    try:
        while True:
            recs = engine.get_analysis()
            if first_update_at is None and recs:
                first_update_at = time.monotonic()
            if first_update_at is not None and time.monotonic() - first_update_at >= seconds:
                return "completed", recs
            if engine.proc.poll() is not None:
                return "engine_exited", None
            if first_update_at is None and time.monotonic() - wait_started_at >= STARTUP_TIMEOUT_SECONDS:
                return "startup_timeout", None
            time.sleep(POLL_SECONDS)
    finally:
        engine.stop_analysis()
        engine.clear_analysis()


def _match_visits_temp(
    *,
    base_visits_temp: float,
    visits_temp_decay: float,
    opening_len: int,
    played_len: int,
) -> float:
    if base_visits_temp == 0.0:
        return 0.0
    moves_after_opening = max(0, played_len - opening_len)
    pair_index = 1 + (moves_after_opening // 2)
    return base_visits_temp / (pair_index**visits_temp_decay)


def _match_payload(
    *,
    game_index: int,
    round_num: int,
    opening: tuple[tuple[int, int], ...],
    red_name: str,
    blue_name: str,
    game_plies: list[dict],
    winner: Optional[str] = None,
    result: Optional[str] = None,
    final: Optional[dict] = None,
) -> dict:
    payload = {
        "round": round_num,
        "game_index": game_index,
        "opening": _opening_line_to_text(opening),
        "red": red_name,
        "blue": blue_name,
    }
    if winner is not None:
        payload["winner"] = winner
    if result is not None:
        payload["result"] = result
    payload["plies"] = game_plies
    if final is not None:
        payload["final"] = final
    return payload


def _emit_match_record(
    *,
    ok: bool,
    error: Optional[str],
    board: Board,
    started_at: float,
    match: dict,
) -> None:
    _emit(
        {
            "hexworld": _match_hexworld_url(board),
            "ok": ok,
            "error": error,
            "match": match,
            "meta": {"elapsed_ms": int(round((time.monotonic() - started_at) * 1000))},
        }
    )


def _validate_match_args(
    args: argparse.Namespace,
    game_type: GameType,
) -> tuple[Optional[list[tuple[tuple[int, int], ...]]], Optional[str]]:
    if args.size < MIN_BOARD_SIZE or args.size > MAX_BOARD_SIZE:
        return None, f"--size must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}"
    if args.rounds < 1:
        return None, "--rounds must be >= 1"
    if not math.isfinite(args.visits_temp) or args.visits_temp < 0.0:
        return None, "--visits-temp must be finite and >= 0"
    if not math.isfinite(args.visits_temp_decay) or args.visits_temp_decay < 0.0:
        return None, "--visits-temp-decay must be finite and >= 0"
    if not math.isfinite(args.resign_winrate) or not (0.0 <= args.resign_winrate <= 1.0):
        return None, "--resign-winrate must be between 0 and 1"
    if not math.isfinite(args.search_seconds) or args.search_seconds < 0.0:
        return None, "--search-seconds must be finite and >= 0"
    try:
        openings = _parse_openings(args.openings)
    except Exception as exc:
        return None, f"Invalid --openings: {exc}"
    if not openings:
        return None, "--openings must include at least one coordinate"

    for line in openings:
        probe = Board(args.size, game_type)
        side = Side.RED
        for col, row in line:
            if not probe.place(side, col, row):
                return None, f"Illegal opening line: {_opening_line_to_text(line)}"
            side = Side.BLUE if side == Side.RED else Side.RED
    return openings, None


def _start_match_engines(
    board_n: int,
    engine_a_cmd: list[str],
    engine_b_cmd: list[str],
    game_type: GameType,
) -> tuple[Optional[tuple[KataHexEngine, KataHexEngine]], Optional[str]]:
    engine_a: Optional[KataHexEngine] = None
    try:
        engine_a = KataHexEngine(
            board_size=board_n,
            cmd=engine_a_cmd,
            game_type=game_type,
            engine_echo=False,
            suppress_stderr=True,
        )
        engine_b = KataHexEngine(
            board_size=board_n,
            cmd=engine_b_cmd,
            game_type=game_type,
            engine_echo=False,
            suppress_stderr=True,
        )
    except OSError:
        if engine_a is not None:
            engine_a.close()
        return None, "Engine executable not found"
    except RuntimeError as exc:
        if engine_a is not None:
            engine_a.close()
        return None, str(exc)
    return (engine_a, engine_b), None


def _run_match_game(
    args: argparse.Namespace,
    *,
    engine_a: KataHexEngine,
    engine_b: KataHexEngine,
    rng: random.Random,
    game_index: int,
    round_num: int,
    opening: tuple[tuple[int, int], ...],
    engine_a_is_red: bool,
    game_type: GameType,
) -> Optional[str]:
    started_at = time.monotonic()
    board = Board(args.size, game_type)
    engine_a.clear_board()
    engine_b.clear_board()
    engine_a.clear_cache()
    engine_b.clear_cache()
    red_engine, blue_engine = (
        (engine_a, engine_b) if engine_a_is_red else (engine_b, engine_a)
    )
    red_name, blue_name = (
        (args.engine_a, args.engine_b) if engine_a_is_red else (args.engine_b, args.engine_a)
    )
    side = Side.RED
    game_plies: list[dict] = []

    def finish(
        recs: list[AnalysisMove],
        *,
        winner: Optional[str] = None,
        result: Optional[str] = None,
    ) -> None:
        _emit_match_record(
            ok=True,
            error=None,
            board=board,
            started_at=started_at,
            match=_match_payload(
                game_index=game_index,
                round_num=round_num,
                opening=opening,
                red_name=red_name,
                blue_name=blue_name,
                game_plies=game_plies,
                winner=winner,
                result=result,
                final=_final_analysis_payload(
                    side=side,
                    analyze=_search_payload_from_moves(
                        recs,
                        side_to_play=side,
                        top_n=None,
                    ),
                ),
            ),
        )

    def fail(error: str) -> str:
        _emit_match_record(
            ok=False,
            error=error,
            board=board,
            started_at=started_at,
            match=_match_payload(
                game_index=game_index,
                round_num=round_num,
                opening=opening,
                red_name=red_name,
                blue_name=blue_name,
                game_plies=game_plies,
            ),
        )
        return error

    for col, row in opening:
        if not board.place(side, col, row):
            return fail(f"Failed to apply opening move: {coord_to_human(col, row)}")
        game_plies.append(
            {
                "ply": len(board.history),
                "side": _side_to_text(side),
                "played": coord_to_human(col, row),
            }
        )
        engine_a.play(side, col, row)
        engine_b.play(side, col, row)
        side = Side.BLUE if side == Side.RED else Side.RED

    while True:
        actor_engine = red_engine if side == Side.RED else blue_engine
        actor_name = red_name if side == Side.RED else blue_name
        visits_temp = _match_visits_temp(
            base_visits_temp=args.visits_temp,
            visits_temp_decay=args.visits_temp_decay,
            opening_len=len(opening),
            played_len=len(board.history),
        )
        status, recs = _run_match_search_for_seconds(
            actor_engine,
            side,
            args.search_seconds,
        )
        if status != "completed":
            return fail(_search_failure_error(status))
        top = recs[0] if recs else None
        side_wr = top.winrate if top is not None else None
        move = _sample_match_search_move(
            board,
            recs,
            temp=visits_temp,
            rng=rng,
        )
        if side_wr is not None and side_wr < args.resign_winrate:
            finish(
                recs,
                winner=blue_name if side == Side.RED else red_name,
                result="red_resigned" if side == Side.RED else "blue_resigned",
            )
            return None
        if move is None:
            # Search exhaustion is reported without inferring a winner.
            finish(recs)
            return None
        col, row = move
        if not board.place(side, col, row):
            return fail(f"Illegal sampled move: {coord_to_human(col, row)}")
        game_plies.append(
            {
                "ply": len(board.history),
                "side": _side_to_text(side),
                "engine": actor_name,
                "visits_temp": _round6(visits_temp),
                "played": coord_to_human(col, row),
                "analyze": _search_payload_from_moves(
                    recs,
                    side_to_play=side,
                    top_n=None,
                ),
            }
        )
        engine_a.play(side, col, row)
        engine_b.play(side, col, row)
        side = Side.BLUE if side == Side.RED else Side.RED


def _run_cli_match(
    args: argparse.Namespace,
    *,
    engine_a_cmd: list[str],
    engine_b_cmd: list[str],
    game_type: GameType = GameType.HEX,
) -> tuple[bool, dict]:
    openings, error = _validate_match_args(args, game_type)
    if error is not None or openings is None:
        return False, {"error": error}
    engines, error = _start_match_engines(
        args.size, engine_a_cmd, engine_b_cmd, game_type
    )
    if error is not None or engines is None:
        return False, {"error": error}
    engine_a, engine_b = engines
    rng = random.SystemRandom()
    jobs = [(round_idx + 1, line) for round_idx in range(args.rounds) for line in openings]
    try:
        engine_a.kata_set_param("analysisWideRootNoise", 0.0)
        engine_b.kata_set_param("analysisWideRootNoise", 0.0)
        game_index = 0
        for round_num, opening in jobs:
            for engine_a_is_red in (True, False):
                game_index += 1
                error = _run_match_game(
                    args,
                    engine_a=engine_a,
                    engine_b=engine_b,
                    rng=rng,
                    game_index=game_index,
                    round_num=round_num,
                    opening=opening,
                    engine_a_is_red=engine_a_is_red,
                    game_type=game_type,
                )
                if error is not None:
                    return False, {"error": error, "already_emitted": True}

        return True, {}
    finally:
        engine_a.close()
        engine_b.close()


def _iter_cli_positions(positions: list[str]) -> Iterator[str]:
    for position in positions:
        if position != "-":
            yield position
            continue
        for line in sys.stdin:
            position = line.strip()
            if position:
                yield position


def _analysis_engine_command(engine_cmd: list[str], *, raw_nn: bool) -> list[str]:
    if len(engine_cmd) < 2 or engine_cmd[1] != "gtp":
        raise ValueError("Engine command must use the KataGo gtp subcommand")
    cmd = list(engine_cmd)
    cmd[1] = "analysis"
    overrides = {
        "nnMaxBatchSize": str(_ANALYSIS_CONCURRENCY),
        "numAnalysisThreads": str(_ANALYSIS_CONCURRENCY),
        "numSearchThreads": "1",
        "reportAnalysisWinratesAs": "BLACK",
    }
    if raw_nn:
        overrides.update(
            nnForcedSymmetry="0",
            nnPolicyTemperature="1.0",
            wideRootNoise="0",
        )
    try:
        override_idx = cmd.index("-override-config")
    except ValueError:
        cmd.extend(["-override-config", ""])
        override_idx = len(cmd) - 2
    existing = [part for part in cmd[override_idx + 1].split(",") if part]
    existing = [part for part in existing if part.split("=", 1)[0].strip() not in overrides]
    cmd[override_idx + 1] = ",".join(
        [*existing, *(f"{key}={value}" for key, value in overrides.items())]
    )
    return cmd


def _parse_analysis_position(
    position: str,
    *,
    game_type: GameType,
) -> tuple[Board, list[Move], list[Move], str]:
    url_game_type = hexworld.url_game_type(position)
    if url_game_type is not None and url_game_type != game_type:
        raise ValueError(
            f"Position is for {url_game_type.value}; current game is {game_type.value}"
        )
    size, past_moves, future_moves, _next_side = hexworld.parse_hexworld_position(
        position,
        game_type=game_type,
    )
    if size < MIN_BOARD_SIZE or size > MAX_BOARD_SIZE:
        raise ValueError(f"Board size {size} must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}")
    board = Board(size, game_type)
    for move in past_moves:
        if not board.apply_move(move):
            raise ValueError(f"Illegal move: {_move_to_text(move)}")
    probe = board.copy()
    for move in future_moves:
        if not probe.apply_move(move):
            raise ValueError(f"Illegal move: {_move_to_text(move)}")
    canonical_url = hexworld.build_position_url(game_type, size, past_moves, future_moves)
    return board, past_moves, future_moves, canonical_url


def _analysis_request(
    board: Board,
    *,
    request_id: str,
    visits: Optional[int],
    filter_moves: tuple[tuple[int, int], ...] = (),
    analysis_wide_root_noise: Optional[float] = None,
) -> tuple[dict, bool]:
    swap_transform = engine_swap_transform_active(board.game_type, board.history)

    engine_moves: list[list[str]] = []
    for move in board.history:
        if move.kind == MoveKind.SWAP:
            continue
        side = map_side_to_engine(move.side, swap_transform)
        if move.kind == MoveKind.PASS:
            vertex = "pass"
        else:
            col, row = map_coords_to_engine(int(move.col), int(move.row), swap_transform)
            vertex = board_to_engine_vertex(col, row, board.game_type)
        engine_moves.append(["B" if side == Side.RED else "W", vertex])

    request = {
        "id": request_id,
        "moves": engine_moves,
        "rules": "tromp-taylor",
        "boardXSize": board.n,
        "boardYSize": board.n,
        "maxVisits": 1 if visits is None else visits,
        "includePolicy": visits is None,
        "analysisPVLen": 1,
    }
    if visits is not None:
        settings: dict[str, int | float] = {
            "maxPlayouts": 1 << 50,
            "maxTime": 1.0e20,
        }
        if analysis_wide_root_noise is not None:
            settings["wideRootNoise"] = analysis_wide_root_noise
        request["overrideSettings"] = settings

    if filter_moves:
        if visits is None:
            raise ValueError("Filtered analyze requires --visits")
        vertices: list[str] = []
        for col, row in filter_moves:
            move = coord_to_human(col, row)
            if not board.in_bounds(col, row):
                raise ValueError(f"Filtered move out of bounds: {move}")
            if not board.is_empty(col, row):
                raise ValueError(f"Filtered move not empty: {move}")
            engine_col, engine_row = map_coords_to_engine(col, row, swap_transform)
            vertices.append(board_to_engine_vertex(engine_col, engine_row, board.game_type))
        side = Side.RED if len(board.history) % 2 == 0 else Side.BLUE
        side = map_side_to_engine(side, swap_transform)
        request["allowMoves"] = [
            {
                "player": "B" if side == Side.RED else "W",
                "moves": vertices,
                "untilDepth": 1,
            }
        ]
    return request, swap_transform


def _raw_payload_from_response(
    response: dict,
    *,
    board: Board,
    swap_transform: bool,
    top_n: Optional[int],
) -> dict:
    policy = response.get("policy")
    root_info = response.get("rootInfo")
    if not isinstance(policy, list) or len(policy) != board.n * board.n + 1:
        raise ValueError("Analysis response missing raw policy")
    if not isinstance(root_info, dict) or not isinstance(root_info.get("winrate"), (int, float)):
        raise ValueError("Analysis response missing root winrate")

    rows: list[tuple[str, float]] = []
    for row in range(1, board.n + 1):
        for col in range(1, board.n + 1):
            if not board.is_empty(col, row):
                continue
            engine_col, engine_row = map_coords_to_engine(col, row, swap_transform)
            prior = policy[(engine_row - 1) * board.n + engine_col - 1]
            rows.append((coord_to_human(col, row), max(0.0, float(prior))))
    rows.append(("pass", max(0.0, float(policy[-1]))))
    rows.sort(key=lambda item: (-item[1], item[0]))
    moves = [
        {"move": move, "rank": rank, "prior": _round6(prior)}
        for rank, (move, prior) in enumerate(rows, start=1)
    ]
    best = {"move": moves[0]["move"], "prior": moves[0]["prior"]} if moves else None
    if top_n is not None:
        moves = moves[:top_n]
    black_winrate = float(root_info["winrate"])
    red_winrate = 1.0 - black_winrate if swap_transform else black_winrate
    return {
        "method": "raw_nn",
        "best": best,
        "root_eval": {"red_winrate": _round6(red_winrate)},
        "moves": moves,
    }


def _search_payload_from_response(
    response: dict,
    *,
    board: Board,
    swap_transform: bool,
    top_n: Optional[int],
) -> dict:
    move_infos = response.get("moveInfos")
    root_info = response.get("rootInfo")
    if not isinstance(move_infos, list):
        raise ValueError("Analysis response missing moves")
    if not isinstance(root_info, dict) or not isinstance(root_info.get("visits"), int):
        raise ValueError("Analysis response missing root visits")

    side_to_play = Side.RED if len(board.history) % 2 == 0 else Side.BLUE
    recs: list[AnalysisMove] = []
    for fallback_order, info in enumerate(move_infos):
        if not isinstance(info, dict) or not isinstance(info.get("move"), str):
            raise ValueError("Analysis response contains an invalid move")
        token = info["move"]
        col = row = None
        if token.lower() != "pass":
            coords = parse_analysis_move_token(token, board.n, board.game_type)
            if coords is None:
                raise ValueError(f"Analysis response contains an invalid move: {token}")
            col, row = coords
            if swap_transform:
                col, row = row, col
        black_winrate = info.get("winrate")
        if not isinstance(black_winrate, (int, float)):
            raise ValueError("Analysis response move missing winrate")
        visits = info.get("visits")
        if not isinstance(visits, int):
            raise ValueError("Analysis response move missing visits")
        order = info.get("order")
        prior = info.get("prior")
        red_winrate = 1.0 - float(black_winrate) if swap_transform else float(black_winrate)
        recs.append(
            AnalysisMove(
                move=token,
                order=order if isinstance(order, int) else fallback_order,
                col=col,
                row=row,
                winrate=red_winrate if side_to_play == Side.RED else 1.0 - red_winrate,
                visits=visits,
                prior=float(prior) if isinstance(prior, (int, float)) else None,
                pv=None,
            )
        )
    recs.sort(key=lambda rec: rec.order)
    payload = _search_payload_from_moves(recs, side_to_play=side_to_play, top_n=top_n)
    payload["total_visits"] = root_info["visits"]
    return payload


def _run_analysis_subcommand(
    requests: list[dict],
    *,
    engine_cmd: list[str],
    raw_nn: bool,
) -> tuple[dict[str, dict], int, bool]:
    started_at = time.monotonic()
    proc = subprocess.run(
        _analysis_engine_command(engine_cmd, raw_nn=raw_nn),
        input="".join(json.dumps(request, separators=(",", ":")) + "\n" for request in requests),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    elapsed_ms = int(round((time.monotonic() - started_at) * 1000))
    responses: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            responses[response["id"]] = response
    return responses, elapsed_ms, proc.returncode != 0


def _analysis_payload_from_response(
    response: dict,
    *,
    board: Board,
    swap_transform: bool,
    visits: Optional[int],
    top_n: Optional[int],
) -> dict:
    if visits is None:
        return _raw_payload_from_response(
            response,
            board=board,
            swap_transform=swap_transform,
            top_n=top_n,
        )
    payload = _search_payload_from_response(
        response,
        board=board,
        swap_transform=swap_transform,
        top_n=top_n,
    )
    if payload["total_visits"] != visits:
        raise ValueError(
            f"Analysis returned {payload['total_visits']} visits; expected {visits}"
        )
    return payload


def _resolve_analysis_jobs(
    requests: list[dict],
    jobs: dict[str, tuple[dict, dict, Board, bool, Optional[int]]],
    *,
    engine_cmd: list[str],
    visits: Optional[int],
) -> tuple[int, bool]:
    responses, elapsed_ms, had_error = _run_analysis_subcommand(
        requests,
        engine_cmd=engine_cmd,
        raw_nn=visits is None,
    )
    for request_id, (record, target, board, swap_transform, top_n) in jobs.items():
        response = responses.get(request_id)
        try:
            if response is None:
                raise ValueError("Engine exited before analysis completed")
            if response.get("error"):
                raise ValueError(str(response["error"]))
            target["analyze"] = _analysis_payload_from_response(
                response,
                board=board,
                swap_transform=swap_transform,
                visits=visits,
                top_n=top_n,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            record["error"] = str(exc)
            had_error = True
    return elapsed_ms, had_error


def _emit_analysis_records(records: list[dict], elapsed_ms: int, had_error: bool) -> int:
    for record in records:
        record.setdefault("meta", {"elapsed_ms": elapsed_ms})
        record["ok"] = record["error"] is None
        had_error = had_error or not record["ok"]
        _emit(record)
    return 1 if had_error else 0


def _run_analyze_positions(
    args: argparse.Namespace,
    *,
    engine_cmd: list[str],
    game_type: GameType,
) -> int:
    records: list[dict] = []
    requests: list[dict] = []
    jobs: dict[str, tuple[dict, dict, Board, bool, Optional[int]]] = {}
    for index, position_spec in enumerate(_iter_cli_positions(args.position)):
        position = position_spec.rsplit("::", 1)[0]
        position_started_at = time.monotonic()
        request_id = str(index)
        try:
            position, filter_moves = _parse_analyze_position_spec(position_spec)
            board, _past_moves, _future_moves, canonical_url = _parse_analysis_position(
                position,
                game_type=game_type,
            )
            request, swap_transform = _analysis_request(
                board,
                request_id=request_id,
                visits=args.visits,
                filter_moves=filter_moves,
                analysis_wide_root_noise=args.analysis_wide_root_noise,
            )
        except ValueError as exc:
            records.append(
                {
                    "hexworld": _input_hexworld_url(position, game_type),
                    "ok": False,
                    "error": str(exc),
                    "meta": {
                        "elapsed_ms": int(
                            round((time.monotonic() - position_started_at) * 1000)
                        )
                    },
                }
            )
            continue
        record = {"hexworld": canonical_url, "ok": False, "error": None}
        records.append(record)
        requests.append(request)
        jobs[request_id] = (record, record, board, swap_transform, args.top_n)

    if not requests:
        return _emit_analysis_records(records, 0, False)

    try:
        elapsed_ms, had_error = _resolve_analysis_jobs(
            requests,
            jobs,
            engine_cmd=engine_cmd,
            visits=args.visits,
        )
    except OSError:
        return _fail("Engine executable not found")
    return _emit_analysis_records(records, elapsed_ms, had_error)


def _run_batch_positions(
    args: argparse.Namespace,
    *,
    engine_cmd: list[str],
    game_type: GameType,
) -> int:
    records: list[dict] = []
    requests: list[dict] = []
    jobs: dict[str, tuple[dict, dict, Board, bool, Optional[int]]] = {}

    for position_input in _iter_cli_positions(args.position):
        position_started_at = time.monotonic()
        record = {
            "hexworld": _input_hexworld_url(position_input, game_type),
            "ok": False,
            "error": None,
        }
        records.append(record)
        try:
            cursor_board, past_moves, future_moves, canonical_url = _parse_analysis_position(
                position_input,
                game_type=game_type,
            )
            all_moves = [*past_moves, *future_moves]
            board = Board(cursor_board.n, game_type)
            plies: list[dict] = []
            batch: dict = {"plies": plies}
            result = hexworld.terminal_result_from_text(position_input)
            if result is not None:
                batch["result"] = result
            record.update(hexworld=canonical_url, batch=batch)

            def add_analysis(target: dict) -> None:
                request_id = str(len(requests))
                snapshot = board.copy()
                request, swap_transform = _analysis_request(
                    snapshot,
                    request_id=request_id,
                    visits=args.visits,
                )
                requests.append(request)
                jobs[request_id] = (record, target, snapshot, swap_transform, None)

            for ply, move in enumerate(all_moves, start=1):
                row = {
                    "ply": ply,
                    "side": _side_to_text(move.side),
                    "played": _move_to_text(move),
                }
                if ply != 1 and move.kind != MoveKind.SWAP:
                    add_analysis(row)
                plies.append(row)
                if not board.apply_move(move):
                    raise AssertionError("Validated batch move became illegal")

            final_side = Side.RED if len(all_moves) % 2 == 0 else Side.BLUE
            final = {"side": _side_to_text(final_side)}
            batch["final"] = final
            add_analysis(final)
        except ValueError as exc:
            record["error"] = str(exc)
            record["meta"] = {
                "elapsed_ms": int(round((time.monotonic() - position_started_at) * 1000))
            }

    if not requests:
        return _emit_analysis_records(records, 0, False)

    try:
        elapsed_ms, had_error = _resolve_analysis_jobs(
            requests,
            jobs,
            engine_cmd=engine_cmd,
            visits=args.visits,
        )
    except OSError:
        return _fail("Engine executable not found")
    return _emit_analysis_records(records, elapsed_ms, had_error)


def _add_cli_engine_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--engine",
        default=None,
        help="Engine profile name from config.ini (defaults to [engine.<game>].default/first profile)",
    )


def add_cli_arguments(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="cli_cmd", required=True)

    analyze_ap = sub.add_parser("analyze", help="Analyze root position(s)")
    _add_cli_engine_argument(analyze_ap)
    analyze_ap.add_argument(
        "position",
        nargs="+",
        help="HexWorld URL(s), hash(es), optional search-only ::moves filter, or '-' for stdin",
    )
    analyze_ap.add_argument("--top-n", type=int, default=None, help="Limit number of returned moves")
    analyze_ap.add_argument(
        "--visits",
        type=int,
        default=None,
        help="Search visits per position (omit for raw-NN)",
    )
    analyze_ap.add_argument(
        "--analysis-wide-root-noise",
        "--awrn",
        dest="analysis_wide_root_noise",
        type=float,
        default=None,
        help="analysisWideRootNoise",
    )

    batch_ap = sub.add_parser("batch", help="Analyze positions along the full main line")
    _add_cli_engine_argument(batch_ap)
    batch_ap.add_argument(
        "position",
        nargs="+",
        help="HexWorld URL(s), hash(es), or '-' for stdin",
    )
    batch_ap.add_argument(
        "--visits",
        type=int,
        default=None,
        help="Search visits per position (omit for raw-NN)",
    )

    match_ap = sub.add_parser("match", help="Run paired engine-vs-engine search games over openings")
    match_ap.add_argument("--engine-a", required=True, help="Engine A profile name from config.ini")
    match_ap.add_argument("--engine-b", required=True, help="Engine B profile name from config.ini")
    match_ap.add_argument(
        "--openings",
        required=True,
        help="Comma-separated opening moves, e.g. c3,d4,e5",
    )
    match_ap.add_argument(
        "--size",
        required=True,
        type=int,
        help="Board size",
    )
    match_ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Paired rounds per opening (A-first and B-first each round)",
    )
    match_ap.add_argument(
        "--search-seconds",
        type=float,
        default=1.0,
        help="Search time per move",
    )
    match_ap.add_argument(
        "--visits-temp",
        type=float,
        default=0.5,
        help="Visit-sampling temperature (0 means argmax with random tie-break)",
    )
    match_ap.add_argument(
        "--visits-temp-decay",
        type=float,
        default=0.5,
        help="Paired power-law decay exponent for visit-sampling temperature",
    )
    match_ap.add_argument(
        "--resign-winrate",
        type=float,
        default=0.01,
        help="Resign when side-to-play estimated winrate drops below this",
    )


def run_cli(
    args: argparse.Namespace,
    *,
    engine_cmd: list[str] | None = None,
    engine_a_cmd: list[str] | None = None,
    engine_b_cmd: list[str] | None = None,
    game_type: GameType | str = GameType.HEX,
) -> int:
    game_type = GameType.parse(game_type)
    if args.cli_cmd == "match":
        if engine_a_cmd is None or engine_b_cmd is None:
            return _fail("Internal error: missing match engine commands")
        ok, payload = _run_cli_match(
            args,
            engine_a_cmd=engine_a_cmd,
            engine_b_cmd=engine_b_cmd,
            game_type=game_type,
        )
        if ok:
            return 0
        if payload.get("already_emitted"):
            return 1
        return _fail(payload["error"])

    if engine_cmd is None:
        return _fail("Internal error: missing CLI engine command")
    if args.cli_cmd == "analyze":
        if args.top_n is not None and args.top_n < 1:
            return _fail("--top-n must be >= 1")
        if args.visits is not None and args.visits < 1:
            return _fail("--visits must be >= 1")
        if args.analysis_wide_root_noise is not None:
            if args.visits is None:
                return _fail("--analysis-wide-root-noise requires --visits")
            if not math.isfinite(args.analysis_wide_root_noise) or not (
                0.0 <= args.analysis_wide_root_noise <= 5.0
            ):
                return _fail("--analysis-wide-root-noise must be between 0 and 5")
    elif args.cli_cmd == "batch":
        if args.visits is not None and args.visits < 1:
            return _fail("--visits must be >= 1")
    else:
        return _fail(f"Unknown cli command: {args.cli_cmd}")

    try:
        if args.cli_cmd == "analyze":
            return _run_analyze_positions(
                args,
                engine_cmd=engine_cmd,
                game_type=game_type,
            )
        return _run_batch_positions(
            args,
            engine_cmd=engine_cmd,
            game_type=game_type,
        )
    except ValueError as exc:
        return _fail(str(exc))
