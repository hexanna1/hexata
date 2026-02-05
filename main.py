#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import os
import shlex
import sys

from board import HexBoard
from engine import KataHexEngine
from gui import run_gui


def _expand_cmd(cmd_str: str) -> list[str]:
    parts = shlex.split(cmd_str)
    return [os.path.expanduser(p) for p in parts]


def _require_config_value(
    parser: configparser.ConfigParser, section: str, option: str
) -> str:
    value = parser.get(section, option, fallback="").strip()
    if not value:
        raise ValueError(f"Missing {section}.{option} in config.ini")
    return value


def _load_engine_cmd() -> list[str]:
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    config_local_path = os.path.join(os.path.dirname(__file__), "config.local.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config.ini")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_path, config_local_path])
    cmd = _require_config_value(parser, "engine", "cmd")
    return _expand_cmd(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal Hex GUI + KataHex analysis")
    ap.add_argument("--size", type=int, default=14, help="initial board size (default 14)")
    ap.add_argument("--interval", type=int, default=15, help="kata-analyze interval (centiseconds)")
    ap.add_argument(
        "--engine-echo",
        action="store_true",
        help="echo engine output to stderr",
    )
    args = ap.parse_args()

    size = max(4, min(42, int(args.size)))

    board = HexBoard(size)
    try:
        engine_cmd = _load_engine_cmd()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Engine config error: {exc}")
        print("Edit config.ini and set [engine].cmd to your KataHex command.")
        return 1

    try:
        engine = KataHexEngine(
            board_size=size,
            cmd=engine_cmd,
            engine_echo=args.engine_echo,
            suppress_stderr=True,
        )
    except FileNotFoundError:
        print("Engine executable not found. Check [engine].cmd in config.ini.")
        return 1

    try:
        run_gui(board, engine, analyze_interval_cs=int(args.interval))
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
