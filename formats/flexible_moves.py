from __future__ import annotations

import re
from typing import Callable

from board import Board, GameType, Move, MoveKind, Side
from formats.hexworld import cell_to_col_row

_MOVE_TOKEN_RE = re.compile(r"resign(?![0-9])|swap(?![0-9])|pass(?![0-9])|[a-z]+[0-9]+", re.IGNORECASE)
_MOVE_NUMBER_RE = re.compile(r"[0-9]+(?:\.\s*|\s+|$)")


def _tokenize(text: str) -> list[str]:
    # Move numbers are optional labels; only their placement is validated.
    # Their numeric values are intentionally not checked for order or gaps.
    tokens: list[str] = []
    need_move = False
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        if text[i].isdigit():
            match = _MOVE_NUMBER_RE.match(text, i)
            if match is None:
                raise ValueError("Move number must be followed by '.' or whitespace")
            if need_move:
                raise ValueError("Move number must be followed by a move")
            need_move = True
            i = match.end()
            continue

        match = _MOVE_TOKEN_RE.match(text, i)
        if match is None:
            raise ValueError(f"Unexpected text at position {i + 1}")
        tokens.append(match.group(0).lower())
        need_move = False
        i = match.end()

    if need_move:
        raise ValueError("Move number must be followed by a move")
    return tokens


def parse_flexible_move_format(
    text: str,
    *,
    board_size: int,
    game_type: GameType | str = GameType.HEX,
    cell_parser: Callable[[str], tuple[int, int]] = cell_to_col_row,
) -> list[Move]:
    game_type = GameType.parse(game_type)
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("Move text is empty")

    moves: list[Move] = []
    board = Board(board_size, game_type)
    side = Side.RED
    for tok in tokens:
        if tok == "resign":
            continue
        if tok == "pass":
            mv = Move.pass_(side=side)
        elif tok == "swap":
            first = board.history[0] if len(board.history) == 1 else None
            if first is None or first.kind != MoveKind.PLACE:
                raise ValueError("Swap token is only legal on move 2")
            mv = Move.swap(side=side, col=first.col, row=first.row)
        else:
            col, row = cell_parser(tok)
            if not board.in_bounds(col, row):
                raise ValueError(f"Move {tok!r} out of bounds for size {board_size}")
            mv = Move.place(side=side, col=col, row=row)
        if not board.apply_move(mv):
            raise ValueError(f"Illegal move: {tok}")
        moves.append(mv)
        side = Side.BLUE if side == Side.RED else Side.RED
    return moves


def alternate_y_cell_to_col_row(cell: str, *, board_size: int) -> tuple[int, int]:
    band_from_bottom, position = cell_to_col_row(cell)
    if not (1 <= band_from_bottom <= board_size and 1 <= position <= band_from_bottom):
        raise ValueError(f"Move {cell!r} out of bounds for alternate Y size {board_size}")
    return position, board_size + 1 - band_from_bottom


def parse_alternate_y_move_format(text: str, *, board_size: int) -> list[Move]:
    return parse_flexible_move_format(
        text,
        board_size=board_size,
        game_type=GameType.Y,
        cell_parser=lambda cell: alternate_y_cell_to_col_row(cell, board_size=board_size),
    )
