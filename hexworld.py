from __future__ import annotations

import re
from typing import List, Tuple

from board import Move, MoveKind, Side

_MOVE_TOKEN_RE = re.compile(r":p|:s|:rw|:rb|[A-Za-z]+[0-9]+")
_CELL_RE = re.compile(r"^([A-Za-z]+)([0-9]+)$")


def extract_hash(s: str) -> str:
    """Accept bare hashes or full URLs; if '#' exists, return the fragment after it."""
    s = s.strip()
    if "#" in s:
        return s.split("#", 1)[1].strip()
    return s


def _letters_to_idx(letters: str) -> int:
    """
    Excel-like letters -> 0-index:
      a->0, z->25, aa->26, ab->27, ...
    """
    letters = letters.lower()
    n = 0
    for ch in letters:
        if not ("a" <= ch <= "z"):
            raise ValueError(f"Bad column letters: {letters!r}")
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n - 1


def cell_to_col_row(cell: str) -> Tuple[int, int]:
    m = _CELL_RE.match(cell.strip())
    if not m:
        raise ValueError(f"Bad cell token: {cell!r}")
    col_letters = m.group(1)
    row_num = int(m.group(2))
    if row_num <= 0:
        raise ValueError(f"Row numbers must be >= 1: {cell!r}")
    col = _letters_to_idx(col_letters) + 1
    row = row_num
    return col, row


def _tokenize_moves(move_stream: str) -> List[str]:
    move_stream = re.sub(r"\s+", "", move_stream)
    if move_stream == "":
        return []
    toks: List[str] = [m.group(0).lower() for m in _MOVE_TOKEN_RE.finditer(move_stream)]
    if "".join(toks) != move_stream:
        raise ValueError(f"Unparsed moves in: {move_stream!r}")
    return toks


def _other_side(side: Side) -> Side:
    return Side.BLUE if side == Side.RED else Side.RED


def parse_hexworld_position(s: str) -> Tuple[int, List[Move], List[Move], Side]:
    """
    Parse a HexWorld hash/URL into (size, past_moves, future_moves, to_play_at_cursor).

    Supported move tokens:
      - ':p' pass (toggles side, no placement)
      - ':s' swap (move 2 only; consumes turn and records explicit swap move)
      - ':rw' / ':rb' resign markers (ignored)
      - cell coords like 'h9'
    """
    h = extract_hash(s)
    parts = h.split(",")
    if not parts or parts[0].strip() == "":
        raise ValueError("Empty hash/prefix")

    m = re.match(r"^\s*([0-9]+)(.*)$", parts[0].strip())
    if not m:
        raise ValueError(f"Bad prefix (missing size): {parts[0]!r}")
    size = int(m.group(1))

    past_stream = parts[1] if len(parts) >= 2 else ""
    future_stream = parts[2] if len(parts) >= 3 else ""

    def apply_stream(stream: str, *, to_move: Side, all_moves: List[Move]) -> Tuple[List[Move], Side]:
        out: List[Move] = []
        for tok in _tokenize_moves(stream):
            if tok == ":p":
                mv = Move.pass_(side=to_move)
                out.append(mv)
                all_moves.append(mv)
                to_move = _other_side(to_move)
                continue
            if tok == ":s":
                if any(mv.kind == MoveKind.SWAP for mv in all_moves):
                    raise ValueError("Duplicate swap token ':s'")
                if len(all_moves) != 1:
                    raise ValueError("Swap token ':s' is only legal on move 2")
                first = all_moves[0]
                if first.kind != MoveKind.PLACE or first.side != Side.RED:
                    raise ValueError("Swap token ':s' requires a red opening placement")
                if first.col is None or first.row is None:
                    raise ValueError("Bad opening move before swap")
                mv = Move.swap(side=to_move, col=first.col, row=first.row)
                out.append(mv)
                all_moves.append(mv)
                to_move = _other_side(to_move)
                continue
            if tok in (":rw", ":rb"):
                continue

            col, row = cell_to_col_row(tok)
            if not (1 <= col <= size and 1 <= row <= size):
                raise ValueError(f"Move {tok!r} out of bounds for size {size}")

            mv = Move.place(side=to_move, col=col, row=row)
            out.append(mv)
            all_moves.append(mv)
            to_move = _other_side(to_move)

        return out, to_move

    to_move = Side.RED
    all_moves: List[Move] = []
    past_moves, to_move = apply_stream(past_stream, to_move=to_move, all_moves=all_moves)
    to_play = to_move
    future_moves, _to_move_end = apply_stream(
        future_stream, to_move=to_move, all_moves=all_moves
    )

    return size, past_moves, future_moves, to_play
