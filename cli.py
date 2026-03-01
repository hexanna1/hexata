#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Optional

from board import DEFAULT_BOARD_SIZE, HexBoard, MoveKind, Side, coord_to_human
from engine import AnalysisMove, KataHexEngine, RawNNResult
from gui_core import GuiCore
from hexworld import cell_to_col_row

STARTUP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.02
LOG_MIX_LAMBDA = 1e-6


def _emit(payload: dict) -> None:
    print(json.dumps(payload))


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
        "rank": r.order + 1,
        "red_winrate": _side_winrate_to_red(r.winrate, side_to_play),
        "visits": r.visits,
        "prior": _round6(r.prior),
    }


def _run_for_seconds_from_first_update(core: GuiCore, seconds: float) -> bool:
    first_update_at: Optional[float] = None
    wait_started_at = time.monotonic()
    while True:
        now = time.monotonic()
        core.tick(now)

        if first_update_at is None and core.get_engine_analysis():
            first_update_at = now

        if first_update_at is not None and now - first_update_at >= seconds:
            return True
        if first_update_at is None and now - wait_started_at >= STARTUP_TIMEOUT_SECONDS:
            return False
        time.sleep(POLL_SECONDS)


def _run_kata_raw_nn_once(engine: KataHexEngine) -> Optional[RawNNResult]:
    if not engine.start_kata_raw_nn(0):
        return None
    while True:
        done, raw = engine.poll_kata_raw_nn()
        if done:
            return raw
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
            "red_winrate": None,
            "visits": None,
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


def _policy_tag_to_text(tag: tuple[str, Optional[tuple[int, int]]]) -> str:
    kind, pos = tag
    if kind == "pass":
        return "pass"
    col, row = pos or (None, None)
    return coord_to_human(col, row)


def _empty_batch_acc() -> dict:
    return {
        "moves_total": 0,
        "moves_policy_scored": 0,
        "ranks": [],
        "log_num": 0.0,
        "log_den": 0.0,
    }


def _finalize_batch_acc(acc: dict) -> dict:
    scored = acc["moves_policy_scored"]
    regret_log_bits = None
    geom_mean_rank = None
    if scored > 0:
        regret_log_bits = ((acc["log_den"] - acc["log_num"]) / scored) / math.log(2.0)
        geom_mean_rank = math.exp(sum(math.log(float(r)) for r in acc["ranks"]) / scored)
    return {
        "moves_total": acc["moves_total"],
        "moves_policy_scored": scored,
        "policy_log_score": None if acc["log_den"] <= 0.0 else (acc["log_num"] / acc["log_den"]),
        "regret_log_bits": regret_log_bits,
        "geom_mean_rank": geom_mean_rank,
    }


def _score_policy_move_fast_batch(core: GuiCore, raw: RawNNResult, mv) -> tuple | None:
    if mv.kind == MoveKind.SWAP:
        return None
    if mv.kind == MoveKind.PLACE and (mv.col is None or mv.row is None):
        return None

    board = core.board
    legal_tags: list[tuple[str, Optional[tuple[int, int]]]] = []
    legal_weights: list[float] = []
    played_tag = ("pass", None) if mv.kind == MoveKind.PASS else ("place", (mv.col, mv.row))
    played_raw_is_nan = False

    for row in range(1, board.n + 1):
        for col in range(1, board.n + 1):
            if not board.is_empty(col, row):
                continue
            eng_col, eng_row = core._map_coords_to_engine(col, row)
            raw_v = _raw_policy_grid_value(raw, eng_col, eng_row)
            if played_tag == ("place", (col, row)):
                played_raw_is_nan = raw_v is None
            legal_tags.append(("place", (col, row)))
            legal_weights.append(0.0 if raw_v is None else max(0.0, raw_v))

    if mv.kind == MoveKind.PASS:
        played_raw_is_nan = raw.policy_pass is None
    legal_tags.append(("pass", None))
    legal_weights.append(0.0 if raw.policy_pass is None else max(0.0, raw.policy_pass))

    if played_raw_is_nan:
        return None

    k = len(legal_weights)
    if k <= 0:
        return None
    total = sum(legal_weights)
    if total <= 0.0:
        return None

    probs = [w / total for w in legal_weights]
    try:
        played_idx = legal_tags.index(played_tag)
    except ValueError:
        return None

    played_p = probs[played_idx]
    played_raw_p = legal_weights[played_idx]
    played_rank = 1 + sum(1 for p in probs if p > played_p)
    best_idx = max(range(k), key=lambda i: probs[i])
    best_raw_p = legal_weights[best_idx]
    u = 1.0 / k
    q_probs = [((1.0 - LOG_MIX_LAMBDA) * p) + (LOG_MIX_LAMBDA * u) for p in probs]
    mean_log_q = sum(math.log(q) for q in q_probs) / k
    log_headroom = math.log(max(q_probs)) - mean_log_q
    if log_headroom <= 0.0:
        return None

    return (
        math.log(q_probs[played_idx]) - mean_log_q,
        log_headroom,
        played_rank,
        played_raw_p,
        _policy_tag_to_text(legal_tags[best_idx]),
        best_raw_p,
    )


def _run_batch_fast_cli(core: GuiCore, *, include_plies: bool) -> tuple[bool, dict]:
    if core.board.history and not core.go_first():
        return False, {"error": "Failed to rewind to start for batch analysis"}

    acc_red = _empty_batch_acc()
    acc_blue = _empty_batch_acc()
    plies_total = 0
    plies: list[dict] | None = [] if include_plies else None

    while core.app.future_moves:
        mv = core.app.future_moves[-1]
        ply = plies_total + 1
        acc_side = acc_red if mv.side == Side.RED else acc_blue
        plies_total += 1
        acc_side["moves_total"] += 1

        raw = _run_kata_raw_nn_once(core.engine)
        if raw is None:
            return False, {"error": "No raw-NN reply received from engine"}

        if plies:
            # Raw-NN returns a root value (for the current/pre-move position)
            # plus root policy over moves. For move-labeled batch rows we want
            # `red_winrate` to read as post-move eval, so we shift the current
            # root eval onto the previous row. (Regular search move rows don't
            # have this mismatch because their winrates are per-child/per-reply.)
            pre_move_red_wr = _raw_red_winrate(core, raw)
            plies[-1]["red_winrate"] = pre_move_red_wr

        row = {
            "ply": ply,
            "side": _side_to_text(mv.side),
            "move": _move_to_text(mv),
            "red_winrate": None,
        }
        scored = None
        # KataHex is not swap-rule aware, so move 1 policy scoring is
        # systematically misleading in swap-rule games (the opening is chosen
        # under swap considerations, not no-swap policy optimality).
        if ply != 1:
            scored = _score_policy_move_fast_batch(core, raw, mv)
        if scored is not None:
            log_num, log_den, played_rank, played_raw_p, best_move, best_raw_p = scored
            acc_side["moves_policy_scored"] += 1
            acc_side["log_num"] += log_num
            acc_side["log_den"] += log_den
            acc_side["ranks"].append(played_rank)
            row["policy_rank"] = played_rank
            row["policy_prior"] = _round6(played_raw_p)
            row["policy_best_move"] = best_move
            row["policy_best_prior"] = _round6(best_raw_p)
        if plies is not None:
            plies.append(row)

        if not core.step_forward():
            return False, {"error": "Failed to step forward during batch analysis"}

    if plies:
        raw = _run_kata_raw_nn_once(core.engine)
        if raw is None:
            return False, {"error": "No raw-NN reply received from engine"}
        plies[-1]["red_winrate"] = _raw_red_winrate(core, raw)

    out = {
        "plies_total": plies_total,
        "summary": {
            "red": _finalize_batch_acc(acc_red),
            "blue": _finalize_batch_acc(acc_blue),
        },
    }
    if plies is not None:
        out["plies"] = plies
    return True, out


def _position_payload(core: GuiCore) -> dict:
    return {
        "to_play": _side_to_text(core.current_side()),
        "past_len": len(core.board.history),
        "future_len": len(core.app.future_moves),
    }


def _run_cli_analyze(core: GuiCore, args: argparse.Namespace) -> tuple[bool, dict]:
    if args.search_seconds is None:
        raw = _run_kata_raw_nn_once(core.engine)
        if raw is None:
            return False, {"error": "No raw-NN reply received from engine"}
        moves = _raw_policy_rows_cli(core, raw)
        if args.top_n is not None:
            moves = moves[: args.top_n]
        return True, {
            "mode": "analyze",
            "method": "raw_nn",
            "best_reply": None,
            "root_eval": {"red_winrate": _raw_red_winrate(core, raw)},
            "moves": moves,
        }

    side_to_play = core.current_side()
    core.toggle_analysis()
    if not _run_for_seconds_from_first_update(core, args.search_seconds):
        return False, {"error": "No analysis update received from engine"}

    recs = core.get_active_analysis()
    moves = [_to_output_row(r, side_to_play=side_to_play) for r in recs]
    if args.top_n is not None:
        moves = moves[: args.top_n]

    best = None
    if moves:
        m0 = moves[0]
        best = {
            "move": m0["move"],
            "red_winrate": m0["red_winrate"],
            "visits": m0["visits"],
        }
    return True, {"mode": "analyze", "method": "search", "best_reply": best, "moves": moves}


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
    if args.total_search_seconds is None:
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
                    "visits": None,
                }
            )
        return True, {"mode": "candidate", "method": "raw_nn", "best_reply": None, "moves": moves}

    per_candidate_seconds = args.total_search_seconds / len(requested_moves)
    moves = []
    for i, (col, row) in enumerate(requested_moves):
        if i > 0:
            core.clear_candidates()
        core.add_candidate(col, row)
        if not core.app.analysis_running:
            core.toggle_analysis()
        if not _run_for_seconds_from_first_update(core, per_candidate_seconds):
            core.clear_candidates()
            return False, {
                "error": f"No analysis update received from engine for {coord_to_human(col, row)}"
            }
        winrate, visits = core.app.candidate_state.results.get((col, row), (None, None))
        moves.append(
            {
                "move": coord_to_human(col, row),
                "red_winrate": _side_winrate_to_red(winrate, side_to_move),
                "visits": visits,
            }
        )
    core.clear_candidates()
    return True, {"mode": "candidate", "method": "search", "best_reply": None, "moves": moves}


def _run_cli_batch(core: GuiCore, args: argparse.Namespace) -> tuple[bool, dict]:
    ok, batch = _run_batch_fast_cli(core, include_plies=args.plies)
    if not ok:
        return False, batch
    return True, {"mode": "batch", "batch": batch}


def _add_cli_position_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("position", help="HexWorld URL or hash")


def add_cli_arguments(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="cli_cmd", required=True)

    analyze_ap = sub.add_parser("analyze", help="Analyze root position")
    _add_cli_position_argument(analyze_ap)
    analyze_ap.add_argument("--top-n", type=int, default=None, help="Limit number of returned moves")
    analyze_ap.add_argument(
        "--search-seconds",
        type=float,
        default=None,
        help="Search time starting from first analysis update (omit for raw-NN)",
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
    _add_cli_position_argument(candidate_ap)
    candidate_ap.add_argument(
        "--moves",
        required=True,
        help="Comma-separated candidate coordinates, e.g. c3,d4,e5",
    )
    candidate_ap.add_argument(
        "--total-search-seconds",
        type=float,
        default=None,
        help="Total search time budget split evenly across --moves (omit for raw-NN)",
    )

    batch_ap = sub.add_parser("batch", help="Fast raw-NN batch summary over the full line")
    _add_cli_position_argument(batch_ap)
    batch_ap.add_argument(
        "--plies",
        action="store_true",
        help="Include per-ply rows in batch output",
    )


def run_cli(
    args: argparse.Namespace,
    *,
    engine_cmd: list[str],
) -> int:
    if args.cli_cmd == "analyze":
        if args.top_n is not None and args.top_n < 1:
            return _fail("--top-n must be >= 1")
        if not _is_nonnegative_finite(args.search_seconds):
            return _fail("--search-seconds must be finite and >= 0")
    elif args.cli_cmd == "candidate":
        if not _is_nonnegative_finite(args.total_search_seconds):
            return _fail("--total-search-seconds must be finite and >= 0")

    board = HexBoard(DEFAULT_BOARD_SIZE)
    try:
        engine = KataHexEngine(
            board_size=board.n,
            cmd=engine_cmd,
            engine_echo=False,
            suppress_stderr=True,
        )
    except FileNotFoundError:
        return _fail("Engine executable not found")

    core = GuiCore(board, engine)
    started_at = time.monotonic()
    try:
        if not core.load_hexworld_text(args.position):
            return _fail("Invalid position")

        awrn = getattr(args, "analysis_wide_root_noise", None)
        if awrn is not None:
            core.set_analysis_wide_root_noise(awrn)

        position_payload = _position_payload(core)

        if args.cli_cmd == "analyze":
            ok, payload = _run_cli_analyze(core, args)
        elif args.cli_cmd == "candidate":
            ok, payload = _run_cli_candidate(core, args)
        elif args.cli_cmd == "batch":
            ok, payload = _run_cli_batch(core, args)
        else:
            return _fail(f"Unknown cli command: {args.cli_cmd}")

        if not ok:
            return _fail(payload["error"])

        _emit(
            {
                "ok": True,
                "error": None,
                "position": position_payload,
                **payload,
                "meta": {"elapsed_ms": int(round((time.monotonic() - started_at) * 1000))},
            }
        )
        return 0
    finally:
        engine.close()
