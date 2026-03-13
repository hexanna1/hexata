from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Optional, Tuple

MIN_BOARD_SIZE = 4
MAX_BOARD_SIZE = 42
DEFAULT_BOARD_SIZE = 14


class Side(IntEnum):
    RED = 0
    BLUE = 1


class MoveKind(Enum):
    PLACE = "place"
    PASS = "pass"
    SWAP = "swap"


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

    @staticmethod
    def swap(side: Side, col: int, row: int) -> "Move":
        return Move(kind=MoveKind.SWAP, side=side, col=col, row=row)

    def is_pass(self) -> bool:
        return self.kind == MoveKind.PASS

    def is_swap(self) -> bool:
        return self.kind == MoveKind.SWAP


class HexBoard:
    """
    Hex board state with pass and swap handling.
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

    def copy(self) -> "HexBoard":
        out = HexBoard(self.n)
        out.rev = self.rev
        out.occ = list(self.occ)
        out.history = list(self.history)
        return out

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

    @staticmethod
    def _flip_side(side: Side) -> Side:
        return Side.BLUE if side == Side.RED else Side.RED

    def swap_move(self, side: Side) -> bool:
        if len(self.history) != 1:
            return False
        first = self.history[0]
        if first.kind != MoveKind.PLACE:
            return False
        if first.col is None or first.row is None:
            return False

        swapped_side = self._flip_side(first.side)
        if side != swapped_side:
            return False

        old_col, old_row = first.col, first.row
        new_col, new_row = old_row, old_col
        old_idx = self._idx(old_col, old_row)
        new_idx = self._idx(new_col, new_row)
        if self.occ[old_idx] != int(first.side):
            return False

        self.occ[old_idx] = -1
        self.occ[new_idx] = int(swapped_side)
        self.history[0] = Move.place(side=swapped_side, col=new_col, row=new_row)
        self.history.append(Move.swap(side=side, col=old_col, row=old_row))
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
        # Keep SWAP metadata aligned so move 1 label/export still uses the
        # opening token from pre-transpose perspective.
        if (
            index == 0
            and len(self.history) >= 2
            and self.history[1].kind == MoveKind.SWAP
            and self.history[1].col is not None
            and self.history[1].row is not None
        ):
            swap_mv = self.history[1]
            self.history[1] = Move.swap(side=swap_mv.side, col=row, row=col)
        self._bump_rev()
        return True

    def undo(self) -> bool:
        if not self.history:
            return False
        mv = self.history[-1]
        if mv.kind == MoveKind.PLACE:
            self.history.pop()
            self.occ[self._idx(mv.col, mv.row)] = -1
        elif mv.kind == MoveKind.PASS:
            self.history.pop()
        elif mv.kind == MoveKind.SWAP:
            if len(self.history) < 2:
                return False
            first = self.history[0]
            if first.kind != MoveKind.PLACE:
                return False
            if first.col is None or first.row is None or mv.col is None or mv.row is None:
                return False
            self.history.pop()
            self.occ[self._idx(first.col, first.row)] = -1
            restored_side = self._flip_side(first.side)
            self.occ[self._idx(mv.col, mv.row)] = int(restored_side)
            self.history[0] = Move.place(side=restored_side, col=mv.col, row=mv.row)
        self._bump_rev()
        return True

    def apply_move(self, mv: Move) -> bool:
        if mv.kind == MoveKind.PLACE:
            return self.place(mv.side, mv.col, mv.row)
        if mv.kind == MoveKind.PASS:
            return self.pass_move(mv.side)
        if mv.kind == MoveKind.SWAP:
            return self.swap_move(mv.side)
        raise AssertionError(f"Unhandled move kind: {mv.kind}")


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
