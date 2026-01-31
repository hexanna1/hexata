from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Optional, Tuple


class Side(IntEnum):
    RED = 0
    BLUE = 1


class MoveKind(Enum):
    PLACE = "place"
    PASS = "pass"


@dataclass(frozen=True, slots=True)
class Move:
    kind: MoveKind
    side: Side
    col: Optional[int]
    row: Optional[int]

    @staticmethod
    def place(side: Side, col: int, row: int) -> "Move":
        return Move(kind=MoveKind.PLACE, side=side, col=col, row=row)

    @staticmethod
    def pass_(side: Side) -> "Move":
        return Move(kind=MoveKind.PASS, side=side, col=None, row=None)

    def is_pass(self) -> bool:
        return self.kind == MoveKind.PASS


class HexBoard:
    """
    Swap-less Hex.
    - Side.RED connects TOP<->BOTTOM
    - Side.BLUE connects LEFT<->RIGHT
    """

    def __init__(self, n: int):
        self.rev = 0
        self.set_size(n)

    def _bump_rev(self) -> None:
        self.rev += 1

    def set_size(self, n: int) -> None:
        self._bump_rev()
        self.n = n
        self.edge_a, self.edge_b = n * n, n * n + 1
        self.occ = [-1] * (n * n)  # -1 empty, 0 red, 1 blue
        self.history: List[Move] = []

    def clear(self) -> None:
        self.set_size(self.n)

    def in_bounds(self, col: int, row: int) -> bool:
        return 1 <= col <= self.n and 1 <= row <= self.n

    def _idx(self, col: int, row: int) -> int:
        return (row - 1) * self.n + (col - 1)

    def get(self, col: int, row: int) -> int:
        return self.occ[self._idx(col, row)]

    def is_empty(self, col: int, row: int) -> bool:
        return self.in_bounds(col, row) and self.get(col, row) < 0

    def place(self, side: Side, col: int, row: int) -> bool:
        if not self.in_bounds(col, row):
            return False
        if not self.is_empty(col, row):
            return False

        idx = self._idx(col, row)
        self.occ[idx] = int(side)
        self.history.append(Move.place(side=side, col=col, row=row))
        self._bump_rev()

        return True

    def pass_move(self, side: Side) -> bool:
        self.history.append(Move.pass_(side=side))
        self._bump_rev()
        return True

    def move_in_history(self, index: int, col: int, row: int) -> bool:
        if index < 0 or index >= len(self.history):
            return False
        if not self.in_bounds(col, row):
            return False
        if not self.is_empty(col, row):
            return False

        mv = self.history[index]
        if mv.kind != MoveKind.PLACE:
            return False
        self.occ[self._idx(mv.col, mv.row)] = -1
        self.occ[self._idx(col, row)] = int(mv.side)
        self.history[index] = Move.place(side=mv.side, col=col, row=row)
        self._bump_rev()
        return True

    def undo(self) -> bool:
        if not self.history:
            return False
        mv = self.history.pop()
        if mv.kind == MoveKind.PLACE:
            self.occ[self._idx(mv.col, mv.row)] = -1
        self._bump_rev()
        return True


def col_to_human_letters(col: int) -> str:
    # 1->a, 26->z, 27->aa ...
    out: List[str] = []
    v = col
    while v > 0:
        v -= 1
        out.append(chr(ord("a") + (v % 26)))
        v //= 26
    return "".join(reversed(out))


def coord_to_human(col: int, row: int) -> str:
    return f"{col_to_human_letters(col)}{row}"
