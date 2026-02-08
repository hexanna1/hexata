#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import logging
import os
import shlex

from board import HexBoard
from engine import KataHexEngine
from gui import UiPrefs, run_gui


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


def _save_ui_prefs(config_local_path: str, prefs: UiPrefs, *, board_size: int) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_local_path])

    if not parser.has_section("ui"):
        parser.add_section("ui")
    parser.set("ui", "show_move_numbers", "true" if prefs.show_move_numbers else "false")
    parser.set("ui", "show_elo", "true" if prefs.show_elo else "false")
    if not parser.has_section("game"):
        parser.add_section("game")
    parser.set("game", "size", str(max(4, min(42, board_size))))
    with open(config_local_path, "w", encoding="utf-8") as f:
        parser.write(f)


def _load_runtime_config(base_dir: str) -> tuple[list[str], int, UiPrefs, str]:
    config_path = os.path.join(base_dir, "config.ini")
    config_local_path = os.path.join(base_dir, "config.local.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config.ini")

    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_path, config_local_path])
    default_prefs = UiPrefs()
    size = parser.getint("game", "size", fallback=14)
    show_move_numbers = parser.getboolean(
        "ui", "show_move_numbers", fallback=default_prefs.show_move_numbers
    )
    show_elo = parser.getboolean("ui", "show_elo", fallback=default_prefs.show_elo)
    return (
        _expand_cmd(_require_config_value(parser, "engine", "cmd")),
        max(4, min(42, size)),
        UiPrefs(show_move_numbers=show_move_numbers, show_elo=show_elo),
        config_local_path,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser(description="Minimal Hex GUI + KataHex analysis")
    ap.add_argument("--interval", type=int, default=15, help="kata-analyze interval (centiseconds)")
    ap.add_argument(
        "--engine-echo",
        action="store_true",
        help="echo engine output to stderr",
    )
    args = ap.parse_args()

    try:
        engine_cmd, size, ui_prefs, config_local_path = _load_runtime_config(os.path.dirname(__file__))
    except (FileNotFoundError, ValueError, configparser.Error) as exc:
        print(f"Engine config error: {exc}")
        print("Edit config.ini and set [engine].cmd to your KataHex command.")
        return 1

    board = HexBoard(size)
    exit_code = 0

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
        run_gui(
            board,
            engine,
            analyze_interval_cs=int(args.interval),
            ui_prefs=ui_prefs,
        )
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        _save_ui_prefs(config_local_path, ui_prefs, board_size=board.n)
        engine.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
