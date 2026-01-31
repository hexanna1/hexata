#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from board import HexBoard
from engine import KataHexEngine
from gui import run_gui


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal Hex GUI + KataHex analysis")
    ap.add_argument("--size", type=int, default=14, help="initial board size (default 14)")
    ap.add_argument("--interval", type=int, default=15, help="kata-analyze interval (centiseconds)")
    ap.add_argument("--echo", action="store_true", help="echo engine output to terminal")
    args = ap.parse_args()

    size = max(4, min(42, int(args.size)))

    board = HexBoard(size)
    try:
        engine = KataHexEngine(board_size=size, echo_engine_output=args.echo, suppress_stderr=True)
    except FileNotFoundError:
        print("Engine executable not found. Check BASE/ENGINE_CMD paths in engine.py.")
        return 1

    try:
        run_gui(board, engine, analyze_interval_cs=int(args.interval))
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
