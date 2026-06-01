#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from typing import Iterator, Optional

from board import DEFAULT_BOARD_SIZE, MAX_BOARD_SIZE, MIN_BOARD_SIZE, HexBoard, MoveKind, Side, coord_to_human
from engine import AnalysisMove, KataHexEngine, RawNNResult
from gui.core import GuiCore
from formats import hexworld
from formats.hexworld import cell_to_col_row

STARTUP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.02


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

def _is_nonnegative_finite(x: Optional[float]) -> bool:
    return x is None or (math.isfinite(x) and x >= 0.0)


def _side_winrate_to_red(winrate: Optional[float], side: Side) -> Optional[float]:
    if winrate is None:
        return None
    return _round6(winrate if side == Side.RED else (1.0 - winrate))


def _parse_candidates(raw: str) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        key = cell_to_col_row(tok)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


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


def _run_for_seconds_from_first_update(core: GuiCore, seconds: float) -> str:
    first_update_at: Optional[float] = None
    wait_started_at = time.monotonic()
    while True:
        now = time.monotonic()
        core.tick(now)

        if first_update_at is None and core.get_engine_analysis():
            first_update_at = now

        if first_update_at is not None and now - first_update_at >= seconds:
            return "completed"
        proc = getattr(core.engine, "proc", None)
        if proc is not None and proc.poll() is not None:
            return "engine_exited"
        if first_update_at is None and now - wait_started_at >= STARTUP_TIMEOUT_SECONDS:
            return "startup_timeout"
        time.sleep(POLL_SECONDS)


def _run_kata_raw_nn_once(engine: KataHexEngine) -> Optional[RawNNResult]:
    if not engine.start_kata_raw_nn(0):
        return None
    started_at = time.monotonic()
    while True:
        done, raw = engine.poll_kata_raw_nn()
        if done:
            return raw
        if engine.proc.poll() is not None:
            return None
        if time.monotonic() - started_at >= STARTUP_TIMEOUT_SECONDS:
            return None
        time.sleep(POLL_SECONDS)


def _raw_policy_grid_value(raw: RawNNResult, col: int, row: int) -> Optional[float]:
    if row <= 0 or col <= 0 or row > len(raw.policy_rows):
        return None
    row_vals = raw.policy_rows[row - 1]
    if col > len(row_vals):
        return None
    return row_vals[col - 1]


def _raw_red_winrate(core: GuiCore, raw: RawNNResult) -> Optional[float]:
    if raw.white_win is None:
        return None
    if core._map_side_to_engine(Side.BLUE) == Side.BLUE:
        blue_win = raw.white_win
    else:
        blue_win = 1.0 - raw.white_win
    return _round6(1.0 - blue_win)


def _raw_policy_rows_cli(core: GuiCore, raw: RawNNResult) -> list[dict]:
    board = core.board
    rows: list[tuple[str, float]] = []

    for row in range(1, board.n + 1):
        for col in range(1, board.n + 1):
            if not board.is_empty(col, row):
                continue
            eng_col, eng_row = core._map_coords_to_engine(col, row)
            raw_v = _raw_policy_grid_value(raw, eng_col, eng_row)
            p = 0.0 if raw_v is None else max(0.0, raw_v)
            rows.append((coord_to_human(col, row), p))

    pass_p = 0.0 if raw.policy_pass is None else max(0.0, raw.policy_pass)
    rows.append(("pass", pass_p))

    rows.sort(key=lambda r: (-r[1], r[0]))
    return [
        {
            "move": move,
            "rank": i + 1,
            "prior": _round6(p),
        }
        for i, (move, p) in enumerate(rows)
    ]


def _move_to_text(mv) -> str:
    if mv.kind == MoveKind.PASS:
        return "pass"
    if mv.kind == MoveKind.SWAP:
        return "swap"
    return coord_to_human(mv.col, mv.row)


def _opening_line_to_text(line: tuple[tuple[int, int], ...]) -> str:
    return ",".join(coord_to_human(col, row) for col, row in line)


def _match_hexworld_url(board: HexBoard) -> str:
    return hexworld.build_hexworld_url(board.n, board.history)


def _input_hexworld_url(text: str) -> str:
    return f"https://hexworld.org/board/#{hexworld.extract_hash(text)}"


def _search_failure_error(status: str, *, move: Optional[str] = None) -> str:
    if status == "engine_exited":
        base = "Engine exited before analysis completed"
    else:
        base = "No analysis update received from engine"
    return f"{base} for {move}" if move is not None else base


def _raw_analyze_payload(core: GuiCore, raw: RawNNResult, *, top_n: Optional[int]) -> dict:
    moves = _raw_policy_rows_cli(core, raw)
    best = None
    if moves:
        m0 = moves[0]
        best = {
            "move": m0["move"],
            "prior": m0["prior"],
        }
    if top_n is not None:
        moves = moves[:top_n]
    return {
        "method": "raw_nn",
        "best": best,
        "root_eval": {"red_winrate": _raw_red_winrate(core, raw)},
        "moves": moves,
    }


def _search_analyze_payload(
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


def _run_search_for_cli(core: GuiCore, seconds: float) -> tuple[bool, list[AnalysisMove] | dict]:
    core.toggle_analysis()
    try:
        status = _run_for_seconds_from_first_update(core, seconds)
        if status != "completed":
            return False, {"error": _search_failure_error(status)}
        return True, core.get_active_analysis()
    finally:
        if core.app.analysis_enabled:
            core.toggle_analysis()


def _run_cli_analyze(core: GuiCore, args: argparse.Namespace) -> tuple[bool, dict]:
    if args.search_seconds is None:
        raw = _run_kata_raw_nn_once(core.engine)
        if raw is None:
            return False, {"error": "No raw-NN reply received from engine"}
        return True, {"analyze": _raw_analyze_payload(core, raw, top_n=args.top_n)}

    side_to_play = core.current_side()
    ok, recs_or_error = _run_search_for_cli(core, args.search_seconds)
    if not ok:
        return False, recs_or_error
    return True, {"analyze": _search_analyze_payload(recs_or_error, side_to_play=side_to_play, top_n=args.top_n)}


def _run_cli_candidate(core: GuiCore, args: argparse.Namespace) -> tuple[bool, dict]:
    core.set_analysis_wide_root_noise(0.0)
    try:
        requested_moves = _parse_candidates(args.moves)
    except Exception as exc:
        return False, {"error": f"Invalid --moves: {exc}"}
    if not requested_moves:
        return False, {"error": "--moves must include at least one coordinate"}

    board = core.board
    for col, row in requested_moves:
        if not board.in_bounds(col, row):
            return False, {"error": f"Move out of bounds: {coord_to_human(col, row)}"}
        if not board.is_empty(col, row):
            return False, {"error": f"Move not empty: {coord_to_human(col, row)}"}

    side_to_move = core.current_side()
    if args.search_seconds is None:
        moves = []
        for col, row in requested_moves:
            core.play_engine_mapped(side_to_move, col, row)
            try:
                raw = _run_kata_raw_nn_once(core.engine)
            finally:
                core.engine.undo()
            if raw is None:
                return False, {
                    "error": f"No raw-NN reply received from engine for {coord_to_human(col, row)}"
                }
            moves.append(
                {
                    "move": coord_to_human(col, row),
                    "red_winrate": _raw_red_winrate(core, raw),
                }
            )
        return True, {
            "candidate": {"method": "raw_nn", "moves": moves},
        }

    moves = []
    for i, (col, row) in enumerate(requested_moves):
        if i > 0:
            core.clear_candidates()
        core.add_candidate(col, row)
        if not core.app.analysis_enabled:
            core.toggle_analysis()
        status = _run_for_seconds_from_first_update(core, args.search_seconds)
        if status != "completed":
            core.clear_candidates()
            return False, {"error": _search_failure_error(status, move=coord_to_human(col, row))}
        winrate, visits = core.candidate_result((col, row))
        moves.append(
            {
                "move": coord_to_human(col, row),
                "red_winrate": _side_winrate_to_red(winrate, side_to_move),
                "visits": visits,
            }
        )
    core.clear_candidates()
    return True, {
        "candidate": {"method": "search", "moves": moves},
    }


def _run_cli_batch(core: GuiCore, args: argparse.Namespace) -> tuple[bool, dict]:
    if core.current_ply() and not core.go_first():
        return False, {"error": "Failed to rewind to start for batch analysis"}

    plies: list[dict] = []
    plies_total = 0
    while True:
        mv = core.next_mainline_move()
        if mv is None:
            break

        plies_total += 1
        row = {
            "ply": plies_total,
            "side": _side_to_text(mv.side),
            "played": _move_to_text(mv),
        }
        if plies_total != 1 and mv.kind != MoveKind.SWAP:
            side_to_play = core.current_side()
            ok, recs_or_error = _run_search_for_cli(core, args.search_seconds)
            if not ok:
                return False, recs_or_error
            row["analyze"] = _search_analyze_payload(
                recs_or_error,
                side_to_play=side_to_play,
                top_n=None,
            )
        plies.append(row)

        if not core.step_forward():
            return False, {"error": "Failed to step forward during batch analysis"}

    batch = {"plies": plies}
    return True, {"batch": batch}


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
    board: HexBoard,
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
    return payload


def _emit_match_record(
    *,
    ok: bool,
    error: Optional[str],
    board: HexBoard,
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


def _run_cli_match(
    args: argparse.Namespace,
    *,
    engine_a_cmd: list[str],
    engine_b_cmd: list[str],
) -> tuple[bool, dict]:
    if args.size < MIN_BOARD_SIZE or args.size > MAX_BOARD_SIZE:
        return False, {"error": f"--size must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}"}
    if args.rounds < 1:
        return False, {"error": "--rounds must be >= 1"}
    if not math.isfinite(args.visits_temp) or args.visits_temp < 0.0:
        return False, {"error": "--visits-temp must be finite and >= 0"}
    if not math.isfinite(args.visits_temp_decay) or args.visits_temp_decay < 0.0:
        return False, {"error": "--visits-temp-decay must be finite and >= 0"}
    if not math.isfinite(args.resign_winrate) or not (0.0 <= args.resign_winrate <= 1.0):
        return False, {"error": "--resign-winrate must be between 0 and 1"}
    if not math.isfinite(args.search_seconds) or args.search_seconds < 0.0:
        return False, {"error": "--search-seconds must be finite and >= 0"}
    try:
        openings = _parse_openings(args.openings)
    except Exception as exc:
        return False, {"error": f"Invalid --openings: {exc}"}
    if not openings:
        return False, {"error": "--openings must include at least one coordinate"}

    board_n = args.size
    for line in openings:
        probe = HexBoard(board_n)
        side = Side.RED
        for col, row in line:
            if not probe.place(side, col, row):
                return False, {"error": f"Illegal opening line: {_opening_line_to_text(line)}"}
            side = Side.BLUE if side == Side.RED else Side.RED

    rng = random.SystemRandom()
    jobs = [(round_idx + 1, line) for round_idx in range(args.rounds) for line in openings]

    def fail_game(
        *,
        game_index: int,
        round_num: int,
        opening: tuple[tuple[int, int], ...],
        red_name: str,
        blue_name: str,
        error: str,
        board: HexBoard,
        game_plies: list[dict],
        started_at: float,
    ) -> tuple[bool, dict]:
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
        return False, {"error": error, "already_emitted": True}

    engine_a: KataHexEngine | None = None
    engine_b: KataHexEngine | None = None
    try:
        engine_a = KataHexEngine(
            board_size=board_n,
            cmd=engine_a_cmd,
            engine_echo=False,
            suppress_stderr=True,
        )
        engine_b = KataHexEngine(
            board_size=board_n,
            cmd=engine_b_cmd,
            engine_echo=False,
            suppress_stderr=True,
        )
    except OSError:
        if engine_a is not None:
            engine_a.close()
        return False, {"error": "Engine executable not found"}
    except RuntimeError as exc:
        if engine_a is not None:
            engine_a.close()
        return False, {"error": str(exc)}

    try:
        engine_a.kata_set_param("analysisWideRootNoise", 0.0)
        engine_b.kata_set_param("analysisWideRootNoise", 0.0)
        game_index = 0
        for round_num, opening in jobs:
            for red_engine_key in ("engine_a", "engine_b"):
                game_index += 1
                started_at = time.monotonic()
                board = HexBoard(board_n)
                engine_a.clear_board()
                engine_b.clear_board()
                engine_a.clear_cache()
                engine_b.clear_cache()
                red_name = args.engine_a if red_engine_key == "engine_a" else args.engine_b
                blue_name = args.engine_b if red_engine_key == "engine_a" else args.engine_a
                side = Side.RED
                game_plies: list[dict] = []
                for col, row in opening:
                    if not board.place(side, col, row):
                        error = f"Failed to apply opening move: {coord_to_human(col, row)}"
                        return fail_game(
                            game_index=game_index,
                            round_num=round_num,
                            opening=opening,
                            red_name=red_name,
                            blue_name=blue_name,
                            error=error,
                            board=board,
                            game_plies=game_plies,
                            started_at=started_at,
                        )
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

                red_actor = red_engine_key
                blue_actor = "engine_b" if red_engine_key == "engine_a" else "engine_a"
                while True:
                    actor_key = red_actor if side == Side.RED else blue_actor
                    actor_engine = engine_a if actor_key == "engine_a" else engine_b
                    actor_name = args.engine_a if actor_key == "engine_a" else args.engine_b
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
                        error = _search_failure_error(status)
                        return fail_game(
                            game_index=game_index,
                            round_num=round_num,
                            opening=opening,
                            red_name=red_name,
                            blue_name=blue_name,
                            error=error,
                            board=board,
                            game_plies=game_plies,
                            started_at=started_at,
                        )
                    top = recs[0] if recs else None
                    side_wr = top.winrate if top is not None else None
                    move = _sample_match_search_move(
                        board,
                        recs,
                        temp=visits_temp,
                        rng=rng,
                    )
                    if side_wr is not None and side_wr < args.resign_winrate:
                        winner_key = "engine_b" if actor_key == "engine_a" else "engine_a"
                        winner_name = args.engine_a if winner_key == "engine_a" else args.engine_b
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
                                winner=winner_name,
                                result="red_resigned" if side == Side.RED else "blue_resigned",
                            ),
                        )
                        break
                    if move is None:
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
                                winner=None,
                                result="no_resign",
                            ),
                        )
                        break
                    col, row = move
                    if not board.place(side, col, row):
                        error = f"Illegal sampled move: {coord_to_human(col, row)}"
                        return fail_game(
                            game_index=game_index,
                            round_num=round_num,
                            opening=opening,
                            red_name=red_name,
                            blue_name=blue_name,
                            error=error,
                            board=board,
                            game_plies=game_plies,
                            started_at=started_at,
                        )
                    game_plies.append(
                        {
                            "ply": len(board.history),
                            "side": _side_to_text(side),
                            "engine": actor_name,
                            "visits_temp": _round6(visits_temp),
                            "played": coord_to_human(col, row),
                            "analyze": _search_analyze_payload(
                                recs,
                                side_to_play=side,
                                top_n=None,
                            ),
                        }
                    )
                    engine_a.play(side, col, row)
                    engine_b.play(side, col, row)
                    side = Side.BLUE if side == Side.RED else Side.RED

        return True, {}
    finally:
        if engine_a is not None:
            engine_a.close()
        if engine_b is not None:
            engine_b.close()


def _iter_analyze_positions(positions: list[str]) -> Iterator[str]:
    for position in positions:
        if position != "-":
            yield position
            continue
        for line in sys.stdin:
            position = line.strip()
            if position:
                yield position


def _add_cli_position_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("position", help="HexWorld URL or hash")


def _add_cli_engine_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--engine",
        default=None,
        help="Engine profile name from config.ini (defaults to default_engine/first profile)",
    )


def add_cli_arguments(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="cli_cmd", required=True)

    analyze_ap = sub.add_parser("analyze", help="Analyze root position(s)")
    _add_cli_engine_argument(analyze_ap)
    analyze_ap.add_argument(
        "position",
        nargs="+",
        help="HexWorld URL(s), hash(es), or '-' to read one per line from stdin",
    )
    analyze_ap.add_argument("--top-n", type=int, default=None, help="Limit number of returned moves")
    analyze_ap.add_argument(
        "--search-seconds",
        type=float,
        default=None,
        help="Search time (omit for raw-NN)",
    )
    analyze_ap.add_argument(
        "--analysis-wide-root-noise",
        "--awrn",
        dest="analysis_wide_root_noise",
        type=float,
        default=None,
        help="analysisWideRootNoise",
    )

    candidate_ap = sub.add_parser("candidate", help="Analyze specified candidate moves")
    _add_cli_engine_argument(candidate_ap)
    _add_cli_position_argument(candidate_ap)
    candidate_ap.add_argument(
        "--moves",
        required=True,
        help="Comma-separated candidate coordinates, e.g. c3,d4,e5",
    )
    candidate_ap.add_argument(
        "--search-seconds",
        type=float,
        default=None,
        help="Search time per candidate (omit for raw-NN)",
    )

    batch_ap = sub.add_parser("batch", help="Analyze the full main line with per-ply search")
    _add_cli_engine_argument(batch_ap)
    _add_cli_position_argument(batch_ap)
    batch_ap.add_argument(
        "--search-seconds",
        type=float,
        default=1.0,
        help="Search time per ply",
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
) -> int:
    if args.cli_cmd == "match":
        if engine_a_cmd is None or engine_b_cmd is None:
            return _fail("Internal error: missing match engine commands")
        ok, payload = _run_cli_match(args, engine_a_cmd=engine_a_cmd, engine_b_cmd=engine_b_cmd)
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
        if not _is_nonnegative_finite(args.search_seconds):
            return _fail("--search-seconds must be finite and >= 0")
    elif args.cli_cmd == "candidate":
        if not _is_nonnegative_finite(args.search_seconds):
            return _fail("--search-seconds must be finite and >= 0")
    elif args.cli_cmd == "batch":
        if not _is_nonnegative_finite(getattr(args, "search_seconds", None)):
            return _fail("--search-seconds must be finite and >= 0")

    board = HexBoard(DEFAULT_BOARD_SIZE)
    try:
        engine = KataHexEngine(
            board_size=board.n,
            cmd=engine_cmd,
            engine_echo=False,
            suppress_stderr=True,
        )
    except OSError:
        return _fail("Engine executable not found")
    except RuntimeError as exc:
        return _fail(str(exc))

    core = GuiCore(board, engine)
    try:
        if args.cli_cmd == "analyze":
            awrn = getattr(args, "analysis_wide_root_noise", None)
            if awrn is not None:
                core.set_analysis_wide_root_noise(awrn)
            for i, position_input in enumerate(_iter_analyze_positions(args.position)):
                if i > 0:
                    engine.clear_cache()

                started_at = time.monotonic()
                position_error = core.load_hexworld_text(position_input)
                record = {
                    "hexworld": _input_hexworld_url(position_input),
                    "ok": False,
                    "error": position_error,
                }
                if position_error is None:
                    position_hexworld = core.build_hexworld_url()
                    ok, payload = _run_cli_analyze(core, args)
                    record.update(
                        {
                            "hexworld": position_hexworld,
                            "ok": ok,
                            "error": None if ok else payload["error"],
                            **({} if not ok else payload),
                        }
                    )
                record["meta"] = {"elapsed_ms": int(round((time.monotonic() - started_at) * 1000))}
                _emit(record)
            return 0
        position_error = core.load_hexworld_text(args.position)
        if position_error is not None:
            return _fail(position_error)

        started_at = time.monotonic()
        position_hexworld = core.build_hexworld_url()
        if args.cli_cmd == "candidate":
            ok, payload = _run_cli_candidate(core, args)
        elif args.cli_cmd == "batch":
            ok, payload = _run_cli_batch(core, args)
        else:
            return _fail(f"Unknown cli command: {args.cli_cmd}")
        if not ok:
            return _fail(payload["error"])
        _emit(
            {
                "hexworld": position_hexworld,
                "ok": True,
                "error": None,
                **payload,
                "meta": {"elapsed_ms": int(round((time.monotonic() - started_at) * 1000))},
            }
        )
        return 0
    finally:
        engine.close()
