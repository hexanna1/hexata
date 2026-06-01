from __future__ import annotations

import re

from board import HexBoard, Move, MoveKind, Side
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


def parse_flexible_move_format(text: str, *, board_size: int) -> list[Move]:
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("Move text is empty")

    moves: list[Move] = []
    board = HexBoard(board_size)
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
            col, row = cell_to_col_row(tok)
            if not (1 <= col <= board_size and 1 <= row <= board_size):
                raise ValueError(f"Move {tok!r} out of bounds for size {board_size}")
            mv = Move.place(side=side, col=col, row=row)
        if not board.apply_move(mv):
            raise ValueError(f"Illegal move: {tok}")
        moves.append(mv)
        side = Side.BLUE if side == Side.RED else Side.RED
    return moves
