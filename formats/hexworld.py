from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from board import Move, MoveKind, Side, coord_to_human

_MOVE_TOKEN_RE = re.compile(r":p|:s|:S|:rw|:rb|:fw|:fb|[a-z]+[0-9]+")
_CELL_RE = re.compile(r"^([a-z]+)([0-9]+)$")
_PREFIX_RE = re.compile(r"^([0-9]+)(?:x([0-9]+))?([a-z0-9]*)$")
_RESULT_BY_TOKEN = {
    ":rw": "blue_resigned",
    ":rb": "red_resigned",
}


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
    toks: List[str] = [m.group(0) for m in _MOVE_TOKEN_RE.finditer(move_stream)]
    if "".join(toks) != move_stream:
        raise ValueError(f"Unparsed moves in: {move_stream!r}")
    return toks


def _other_side(side: Side) -> Side:
    return Side.BLUE if side == Side.RED else Side.RED


def opening_token_coords(moves: Sequence[Move]) -> Optional[Tuple[int, int]]:
    if len(moves) < 2:
        return None
    first, second = moves[0], moves[1]
    if (
        first.kind == MoveKind.PLACE
        and second.kind == MoveKind.SWAP
        and second.col is not None
        and second.row is not None
    ):
        return (second.col, second.row)
    return None


def moves_to_hexworld_stream(moves: Sequence[Move]) -> str:
    out: List[str] = []
    opening = opening_token_coords(moves)
    for idx, mv in enumerate(moves):
        if idx == 0 and opening is not None:
            out.append(coord_to_human(*opening))
            continue
        match mv.kind:
            case MoveKind.PLACE:
                out.append(coord_to_human(mv.col, mv.row))
            case MoveKind.PASS:
                out.append(":p")
            case MoveKind.SWAP:
                out.append(":s")
            case _:
                raise AssertionError(f"Unhandled move kind: {mv.kind}")
    return "".join(out)


def build_hexworld_url(
    board_size: int,
    past_moves: Sequence[Move],
    future_moves: Sequence[Move] = (),
) -> str:
    base = f"https://hexworld.org/board/#{board_size}c1"
    past = moves_to_hexworld_stream(past_moves)
    if not future_moves:
        return f"{base},{past}" if past else base
    future = moves_to_hexworld_stream(future_moves)
    return f"{base},{past},{future}"


def terminal_result_from_text(s: str) -> Optional[str]:
    h = extract_hash(s)
    parts = h.split(",")
    result: Optional[str] = None
    for stream in parts[1:]:
        for tok in _tokenize_moves(stream):
            result = _RESULT_BY_TOKEN.get(tok, result)
    return result


def _parse_prefix(prefix: str) -> Tuple[int, int, Tuple[str, ...]]:
    m = _PREFIX_RE.match(prefix.strip())
    if not m:
        raise ValueError(f"Bad prefix: {prefix!r}")
    cols = int(m.group(1))
    rows = int(m.group(2)) if m.group(2) is not None else cols
    if not (1 <= cols <= 53) or not (1 <= rows <= 53):
        raise ValueError(f"Board size out of range: {cols}x{rows}")

    tail = m.group(3)
    configs: List[str] = []
    i = 0
    while i < len(tail):
        if tail.startswith("c1", i):
            configs.append("c1")
            i += 2
            continue
        if tail.startswith("n", i):
            configs.append("n")
            i += 1
            continue
        if tail.startswith("r", i):
            j = i + 1
            while j < len(tail) and tail[j].isdigit():
                j += 1
            if j == i + 1:
                raise ValueError(f"Bad rotation config in prefix: {prefix!r}")
            rot = int(tail[i + 1 : j])
            if not (1 <= rot <= 12):
                raise ValueError(f"Rotation out of range (1-12): r{rot}")
            configs.append(f"r{rot}")
            i = j
            continue
        raise ValueError(f"Unsupported config in prefix: {prefix!r}")
    return cols, rows, tuple(configs)


def _apply_stream(
    stream: str, *, cols: int, rows: int, to_move: Side, all_moves: Sequence[Move]
) -> Tuple[List[Move], Side]:
    all_seen: List[Move] = list(all_moves)
    out: List[Move] = []
    for tok in _tokenize_moves(stream):
        if tok == ":p":
            mv = Move.pass_(side=to_move)
            out.append(mv)
            all_seen.append(mv)
            to_move = _other_side(to_move)
            continue
        if tok == ":s":
            if any(mv.kind == MoveKind.SWAP for mv in all_seen):
                raise ValueError("Duplicate swap token ':s'")
            if len(all_seen) != 1:
                raise ValueError("Swap token ':s' is only legal on move 2")
            first = all_seen[0]
            if first.kind != MoveKind.PLACE or first.side != Side.RED:
                raise ValueError("Swap token ':s' requires a red opening placement")
            if first.col is None or first.row is None:
                raise ValueError("Bad opening move before swap")
            mv = Move.swap(side=to_move, col=first.col, row=first.row)
            out.append(mv)
            all_seen.append(mv)
            to_move = _other_side(to_move)
            continue
        if tok in (":S", ":rw", ":rb", ":fw", ":fb"):
            # These tokens are part of the grammar, but Hexata currently has no
            # player-identity result model; parse and ignore them. Ignored tokens
            # do not consume turn/index in Hexata's move sequence view.
            continue

        col, row = cell_to_col_row(tok)
        if not (1 <= col <= cols and 1 <= row <= rows):
            raise ValueError(f"Move {tok!r} out of bounds for size {cols}x{rows}")
        mv = Move.place(side=to_move, col=col, row=row)
        out.append(mv)
        all_seen.append(mv)
        to_move = _other_side(to_move)
    return out, to_move


def parse_hexworld_state(s: str) -> Tuple[int, int, Tuple[str, ...], List[Move], List[Move], Side]:
    """
    Parse a HexWorld hash/URL into:
      (cols, rows, configs, past_moves, future_moves, to_play_at_cursor)
    """
    h = extract_hash(s)
    parts = h.split(",")
    if not parts or parts[0].strip() == "":
        raise ValueError("Empty hash/prefix")
    if len(parts) > 3:
        raise ValueError(f"Too many comma sections in hash: {h!r}")

    cols, rows, configs = _parse_prefix(parts[0])
    past_stream = parts[1] if len(parts) >= 2 else ""
    future_stream = parts[2] if len(parts) >= 3 else ""

    to_move = Side.RED
    past_moves, to_move = _apply_stream(past_stream, cols=cols, rows=rows, to_move=to_move, all_moves=[])
    to_play = to_move
    future_moves, _to_move_end = _apply_stream(
        future_stream, cols=cols, rows=rows, to_move=to_move, all_moves=past_moves
    )
    return cols, rows, configs, past_moves, future_moves, to_play


def parse_hexworld_position(s: str) -> Tuple[int, List[Move], List[Move], Side]:
    """
    Hexata-specific parser wrapper:
      - accepts full grammar via parse_hexworld_state()
      - rejects non-square boards (Hexata board model is square-only)
    """
    cols, rows, _configs, past, future, to_play = parse_hexworld_state(s)
    if cols != rows:
        raise ValueError(f"Non-square boards are not supported: {cols}x{rows}")
    return cols, past, future, to_play
