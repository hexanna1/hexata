#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import shlex

from board import (
    DEFAULT_BOARD_SIZE,
    MAX_BOARD_SIZE,
    MIN_BOARD_SIZE,
    Board,
    GameType,
)
from cli import add_cli_arguments, run_cli
from engine import KataHexEngine
from gui.app import EngineProfile, UiPrefs, run_gui


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


def _engine_profiles_from_parser(parser: configparser.ConfigParser) -> tuple[EngineProfile, ...]:
    game_sections = {f"engine.{game_type.value}" for game_type in GameType}
    default_names = {
        game_type: parser.get(
            f"engine.{game_type.value}",
            "default",
            fallback="",
        ).strip()
        for game_type in GameType
    }
    profiles: list[EngineProfile] = []
    for section in parser.sections():
        if not section.startswith("engine."):
            continue
        parts = section.split(".")
        if section in game_sections:
            continue
        if len(parts) != 3:
            raise ValueError("Engine sections must be [engine.<game>.<name>]")
        game_type = GameType.parse(parts[1])
        name = parts[2].strip()
        if not name:
            raise ValueError("Empty [engine.<game>.<name>] section in config.ini")
        profiles.append(
            EngineProfile(
                name=name,
                cmd=tuple(_expand_cmd(_require_config_value(parser, section, "cmd"))),
                game_type=game_type,
                is_default=(name == default_names[game_type]),
            )
        )
    if not profiles:
        raise ValueError("Missing [engine.<game>.<name>] cmd in config.ini")
    for game_type, default_name in default_names.items():
        if default_name and not any(
            profile.game_type == game_type and profile.name == default_name
            for profile in profiles
        ):
            raise ValueError(f"Unknown engine.{game_type.value}.default in config.ini: {default_name}")
    return tuple(profiles)


def _select_engine_profile(
    profiles: tuple[EngineProfile, ...],
    *,
    game_type: GameType,
    requested_name: str | None = "",
) -> EngineProfile:
    requested = (requested_name or "").strip()
    compatible = [profile for profile in profiles if profile.game_type == game_type]
    if not compatible:
        raise ValueError(f"Missing [engine.{game_type.value}.<name>] cmd in config.ini")
    if requested:
        for profile in compatible:
            if profile.name == requested:
                return profile
        raise ValueError(
            f"Unknown {game_type.value} CLI engine profile in config.ini: {requested}"
        )
    return next((profile for profile in compatible if profile.is_default), compatible[0])


def _emit_engine_config_error(exc: Exception, *, json_mode: bool) -> int:
    msg = f"Engine config error: {exc}"
    print(json.dumps({"ok": False, "error": msg}) if json_mode else msg)
    if not json_mode:
        print("Edit config.ini and set [engine.<game>.<name>].cmd.")
    return 1


def _save_ui_prefs(
    config_local_path: str,
    prefs: UiPrefs,
    *,
    board_size: int,
    game_type: GameType,
) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_local_path])

    if not parser.has_section("ui"):
        parser.add_section("ui")
    parser.set("ui", "show_move_numbers", "true" if prefs.show_move_numbers else "false")
    parser.set("ui", "show_elo", "true" if prefs.show_elo else "false")
    parser.set("ui", "board_orientation", prefs.board_orientation)
    if not parser.has_section("game"):
        parser.add_section("game")
    parser.set("game", "size", str(max(MIN_BOARD_SIZE, min(MAX_BOARD_SIZE, board_size))))
    parser.set("game", "type", game_type.value)
    with open(config_local_path, "w", encoding="utf-8") as f:
        parser.write(f)


def _load_board_orientation(parser: configparser.ConfigParser, default: str) -> str:
    value = parser.get("ui", "board_orientation", fallback=default).strip().lower()
    if value not in ("flat", "diamond"):
        raise ValueError("ui.board_orientation must be flat or diamond")
    return value


def _load_game_type(parser: configparser.ConfigParser) -> GameType:
    return GameType.parse(parser.get("game", "type", fallback=GameType.HEX.value))


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
            board_orientation=_load_board_orientation(
                parser, default_prefs.board_orientation
            ),
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
    cli_ap.add_argument(
        "--game",
        choices=[game_type.value for game_type in GameType],
        default=GameType.HEX.value,
        help="Game to analyze (default: hex)",
    )
    add_cli_arguments(cli_ap)
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    base_dir = os.path.dirname(__file__)
    args = _build_parser().parse_args()

    try:
        parser, config_local_path = _load_config_parser(base_dir)
        engine_profiles = _engine_profiles_from_parser(parser)
        game_type = GameType.parse(args.game) if args.mode == "cli" else _load_game_type(parser)
        if args.mode == "cli" and getattr(args, "cli_cmd", "") == "match":
            selected_engine = None
            selected_engine_a = _select_engine_profile(
                engine_profiles,
                game_type=game_type,
                requested_name=getattr(args, "engine_a", ""),
            )
            selected_engine_b = _select_engine_profile(
                engine_profiles,
                game_type=game_type,
                requested_name=getattr(args, "engine_b", ""),
            )
        else:
            selected_engine = _select_engine_profile(
                engine_profiles,
                game_type=game_type,
                requested_name=getattr(args, "engine", ""),
            )
            selected_engine_a = None
            selected_engine_b = None
    except (FileNotFoundError, ValueError, configparser.Error) as exc:
        return _emit_engine_config_error(exc, json_mode=(args.mode == "cli"))

    if args.mode == "cli":
        logging.getLogger("gui.core").setLevel(logging.WARNING)
        if args.cli_cmd == "match":
            return run_cli(
                args,
                engine_a_cmd=list(selected_engine_a.cmd),
                engine_b_cmd=list(selected_engine_b.cmd),
                game_type=game_type,
            )
        return run_cli(args, engine_cmd=list(selected_engine.cmd), game_type=game_type)

    try:
        size, ui_prefs = _load_gui_runtime_config(parser)
    except ValueError as exc:
        print(f"Config error: {exc}")
        return 1

    board = Board(size, game_type)
    exit_code = 0

    try:
        engine = KataHexEngine(
            board_size=size,
            cmd=list(selected_engine.cmd),
            game_type=game_type,
            engine_echo=args.engine_echo,
            suppress_stderr=True,
        )
    except (OSError, RuntimeError) as exc:
        print(f"Engine startup failed: {exc}")
        return 1

    try:
        run_gui(
            board,
            engine,
            engine_profiles=engine_profiles,
            current_engine_name=selected_engine.name,
            engine_echo=args.engine_echo,
            ui_prefs=ui_prefs,
        )
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        _save_ui_prefs(
            config_local_path,
            ui_prefs,
            board_size=board.n,
            game_type=board.game_type,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
