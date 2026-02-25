#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import shlex

from board import DEFAULT_BOARD_SIZE, MAX_BOARD_SIZE, MIN_BOARD_SIZE, HexBoard
from cli import add_cli_arguments, run_cli
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


def _load_config_parser(base_dir: str) -> tuple[configparser.ConfigParser, str]:
    config_path = os.path.join(base_dir, "config.ini")
    config_local_path = os.path.join(base_dir, "config.local.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config.ini")

    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_path, config_local_path])
    return parser, config_local_path


def _engine_cmd_from_parser(parser: configparser.ConfigParser) -> list[str]:
    return _expand_cmd(_require_config_value(parser, "engine", "cmd"))


def _emit_config_error(exc: Exception, *, json_mode: bool) -> int:
    msg = f"Engine config error: {exc}"
    print(json.dumps({"ok": False, "error": msg}) if json_mode else msg)
    if not json_mode:
        print("Edit config.ini and set [engine].cmd to your KataHex command.")
    return 1


def _save_ui_prefs(config_local_path: str, prefs: UiPrefs, *, board_size: int) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_local_path])

    if not parser.has_section("ui"):
        parser.add_section("ui")
    parser.set("ui", "show_move_numbers", "true" if prefs.show_move_numbers else "false")
    parser.set("ui", "show_elo", "true" if prefs.show_elo else "false")
    if not parser.has_section("game"):
        parser.add_section("game")
    parser.set("game", "size", str(max(MIN_BOARD_SIZE, min(MAX_BOARD_SIZE, board_size))))
    with open(config_local_path, "w", encoding="utf-8") as f:
        parser.write(f)


def _load_gui_runtime_config(parser: configparser.ConfigParser) -> tuple[int, UiPrefs]:
    default_prefs = UiPrefs()
    size = parser.getint("game", "size", fallback=DEFAULT_BOARD_SIZE)
    return (
        max(MIN_BOARD_SIZE, min(MAX_BOARD_SIZE, size)),
        UiPrefs(
            show_move_numbers=parser.getboolean(
                "ui", "show_move_numbers", fallback=default_prefs.show_move_numbers
            ),
            show_elo=parser.getboolean("ui", "show_elo", fallback=default_prefs.show_elo),
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Minimal Hex GUI + KataHex analysis")
    subparsers = ap.add_subparsers(dest="mode", required=True)

    gui_ap = subparsers.add_parser("gui", help="Run GUI")
    gui_ap.add_argument(
        "--engine-echo",
        action="store_true",
        help="echo engine output to stderr",
    )

    cli_ap = subparsers.add_parser("cli", help="CLI analysis tools for HexWorld positions")
    add_cli_arguments(cli_ap)
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    base_dir = os.path.dirname(__file__)
    args = _build_parser().parse_args()

    try:
        parser, config_local_path = _load_config_parser(base_dir)
        engine_cmd = _engine_cmd_from_parser(parser)
    except (FileNotFoundError, ValueError, configparser.Error) as exc:
        return _emit_config_error(exc, json_mode=(args.mode == "cli"))

    if args.mode == "cli":
        logging.getLogger("gui_core").setLevel(logging.WARNING)
        return run_cli(args, engine_cmd=engine_cmd)

    try:
        size, ui_prefs = _load_gui_runtime_config(parser)
    except ValueError as exc:
        return _emit_config_error(exc, json_mode=False)

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
