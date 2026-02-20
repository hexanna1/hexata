#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

from board import DEFAULT_BOARD_SIZE, HexBoard, Side, coord_to_human
from engine import AnalysisMove, KataHexEngine
from gui_core import GuiCore
from hexworld import cell_to_col_row

STARTUP_TIMEOUT_SECONDS = 10.0
POLL_SECONDS = 0.02


def _emit(payload: dict) -> None:
    print(json.dumps(payload))


def _fail(msg: str) -> int:
    _emit({"ok": False, "error": msg})
    return 1


def _side_to_text(side: Side) -> str:
    return "red" if side == Side.RED else "blue"


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


def _to_output_row(r: AnalysisMove) -> dict:
    move = r.move.lower()
    if r.col is not None and r.row is not None:
        move = coord_to_human(r.col, r.row)
    return {
        "move": move,
        "order": r.order,
        "winrate": r.winrate,
        "visits": r.visits,
        "prior": r.prior,
    }


def add_cli_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("position", help="HexWorld URL or hash")
    ap.add_argument("--top-n", type=int, default=None, help="Limit number of returned moves")
    ap.add_argument(
        "--search-seconds",
        type=float,
        default=1.0,
        help="Search time starting from first analysis update",
    )
    ap.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated candidate coordinates, e.g. c3,d4,e5",
    )
    ap.add_argument(
        "--analysis-wide-root-noise",
        "--awrn",
        dest="analysis_wide_root_noise",
        type=float,
        default=None,
        help="analysisWideRootNoise",
    )


def run_cli(
    args: argparse.Namespace,
    *,
    engine_cmd: list[str],
) -> int:
    if args.top_n is not None and args.top_n < 1:
        return _fail("--top-n must be >= 1")
    if args.search_seconds < 0:
        return _fail("--search-seconds must be >= 0")

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
    mode = "root"
    requested_candidates: list[tuple[int, int]] = []
    try:
        if not core.load_hexworld_text(args.position):
            return _fail("Invalid position")

        if args.candidates is not None:
            try:
                requested_candidates = _parse_candidates(args.candidates)
            except Exception as exc:
                return _fail(f"Invalid --candidates: {exc}")
            if not requested_candidates:
                return _fail("--candidates must include at least one coordinate")
            for col, row in requested_candidates:
                if not board.in_bounds(col, row):
                    return _fail(f"Candidate out of bounds: {coord_to_human(col, row)}")
                if not board.is_empty(col, row):
                    return _fail(f"Candidate not empty: {coord_to_human(col, row)}")
                core.add_candidate(col, row)
            mode = "candidates"

        if args.analysis_wide_root_noise is not None:
            core.set_analysis_wide_root_noise(args.analysis_wide_root_noise)
        core.toggle_analysis()

        first_update_at: Optional[float] = None
        wait_started_at = time.monotonic()
        while True:
            now = time.monotonic()
            core.tick(now)

            if first_update_at is None and core.get_engine_analysis():
                first_update_at = now

            if first_update_at is not None and now - first_update_at >= args.search_seconds:
                break
            if first_update_at is None and now - wait_started_at >= STARTUP_TIMEOUT_SECONDS:
                return _fail("No analysis update received from engine")
            time.sleep(POLL_SECONDS)

        best = None
        if mode == "candidates":
            moves = []
            for col, row in requested_candidates:
                winrate, visits = core.app.candidate_state.results.get((col, row), (None, None))
                moves.append(
                    {
                        "move": coord_to_human(col, row),
                        "winrate": winrate,
                        "visits": visits,
                    }
                )
        else:
            recs = core.get_active_analysis()
            moves = [_to_output_row(r) for r in recs]
            if moves:
                m0 = moves[0]
                best = {
                    "move": m0["move"],
                    "winrate": m0["winrate"],
                    "visits": m0["visits"],
                }

        if args.top_n is not None:
            moves = moves[: args.top_n]

        _emit(
            {
                "ok": True,
                "error": None,
                "mode": mode,
                "position": {
                    "input": args.position,
                    "to_play": _side_to_text(core.current_side()),
                    "past_len": len(core.board.history),
                    "future_len": len(core.app.future_moves),
                },
                "best_reply": best,
                "moves": moves,
                "meta": {"elapsed_ms": int(round((time.monotonic() - started_at) * 1000))},
            }
        )
        return 0
    finally:
        engine.close()
