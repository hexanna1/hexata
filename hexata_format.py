from __future__ import annotations

from dataclasses import dataclass, field

from board import HexBoard, Move, MoveKind, Side, coord_to_human
from history_tree import HistoryNode, MoveTree


def _flip_side(side: Side) -> Side:
    return Side.BLUE if side == Side.RED else Side.RED


def _move_to_token(mv: Move) -> str:
    match mv.kind:
        case MoveKind.PLACE:
            return coord_to_human(mv.col, mv.row)
        case MoveKind.PASS:
            return ":p"
        case MoveKind.SWAP:
            return ":s"
    raise AssertionError(f"Unhandled move kind: {mv.kind}")


def _mainline_tail_node(tree: MoveTree) -> HistoryNode:
    node = tree.root
    while node.preferred_child is not None:
        node = node.preferred_child
    return node


def _cursor_marker_node(tree: MoveTree) -> HistoryNode | None:
    # Omitting the cursor marker is canonical shorthand for "end of mainline".
    return None if tree.cursor is _mainline_tail_node(tree) else tree.cursor


def _serialize_line(node: HistoryNode, *, include_peer_variations: bool, cursor_id: int | None) -> str:
    parts: list[str] = []
    stack: list[tuple[str, HistoryNode | None, bool]] = [("visit", node, include_peer_variations)]
    while stack:
        action, current, include_peers = stack.pop()
        if action == "open":
            parts.append("(")
            continue
        if action == "close":
            parts.append(")")
            continue
        if current is None or current.move is None:
            raise AssertionError("History tree node missing move")
        parts.append(_move_to_token(current.move))
        if current.id == cursor_id:
            parts.append(",")
        child = current.preferred_child
        if child is not None:
            stack.append(("visit", child, True))
        if include_peers and current.parent is not None:
            for peer in reversed(current.parent.children):
                if peer is current:
                    continue
                stack.append(("close", None, False))
                stack.append(("visit", peer, False))
                stack.append(("open", None, False))
    return "".join(parts)


def build_hexata_format(size: int, tree: MoveTree) -> str:
    first = tree.root.preferred_child
    if first is None:
        return f"{size},"
    cursor = _cursor_marker_node(tree)
    prefix = f"{size},"
    if tree.cursor is tree.root:
        prefix += ","
    return f"{prefix}{_serialize_line(first, include_peer_variations=True, cursor_id=None if cursor is None else cursor.id)}"


@dataclass(slots=True)
class _Parser:
    text: str
    pos: int = field(init=False, default=0)
    tree: MoveTree = field(init=False, default_factory=MoveTree)
    size: int = field(init=False, default=0)
    cursor_node: HistoryNode | None = field(init=False, default=None)

    def parse(self) -> tuple[int, MoveTree]:
        if not self.text:
            raise ValueError("Move text is empty")
        if any(ch.isspace() for ch in self.text):
            raise ValueError("Whitespace is not allowed in move text")
        size_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == size_start:
            raise ValueError("Move text must start with board size")
        self.size = int(self.text[:self.pos])
        self._consume(",")
        if self._peek() == ",":
            self._mark_cursor(self.tree.root)
            self._consume(",")
        if self.pos < len(self.text):
            self._parse_line(self.tree.root, HexBoard(self.size), Side.RED)
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected trailing text at position {self.pos + 1}")
        self.tree.cursor = self.cursor_node or _mainline_tail_node(self.tree)
        return self.size, self.tree

    def _peek(self) -> str | None:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def _consume(self, token: str) -> None:
        if not self.text.startswith(token, self.pos):
            raise ValueError(f"Expected '{token}' at position {self.pos + 1}")
        self.pos += len(token)

    def _mark_cursor(self, node: HistoryNode) -> None:
        if self.cursor_node is not None:
            raise ValueError("Only one cursor marker is allowed")
        self.cursor_node = node

    def _parse_move_token(self, board: HexBoard, side: Side) -> Move:
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of move text")
        if self.text.startswith(":p", self.pos):
            self.pos += 2
            return Move.pass_(side=side)
        if self.text.startswith(":s", self.pos):
            self.pos += 2
            first = board.history[0] if board.history else None
            if first is None or first.kind != MoveKind.PLACE:
                raise ValueError("Swap token can only appear after an opening move")
            return Move.swap(side=side, col=first.col, row=first.row)

        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalpha():
            if not self.text[self.pos].islower():
                raise ValueError(f"Move coordinates must be lowercase at position {self.pos + 1}")
            self.pos += 1
        if self.pos == start:
            raise ValueError(f"Expected move token at position {start + 1}")
        letters = self.text[start:self.pos]
        digits_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == digits_start:
            raise ValueError(f"Expected row number at position {digits_start + 1}")
        row = int(self.text[digits_start:self.pos])
        col = 0
        for ch in letters:
            col = (col * 26) + (ord(ch) - ord("a") + 1)
        return Move.place(side=side, col=col, row=row)

    def _parse_line(self, parent: HistoryNode, board: HexBoard, side: Side) -> None:
        saw_move = False
        current_parent = parent
        current_board = board.copy()
        current_side = side

        while True:
            ch = self._peek()
            if ch is None or ch == ")":
                break
            if ch == ",":
                raise ValueError(f"Invalid cursor marker at position {self.pos + 1}")
            before_move = current_board.copy()
            mv = self._parse_move_token(before_move, current_side)
            if self.tree.find_child(current_parent, mv) is not None:
                raise ValueError(f"Duplicate sibling move in tree text: {_move_to_token(mv)}")
            child = self.tree.append_child(current_parent, mv)
            if not current_board.apply_move(mv):
                raise ValueError(f"Illegal move in tree text: {_move_to_token(mv)}")
            saw_move = True
            if self._peek() == ",":
                self._mark_cursor(child)
                self._consume(",")

            while self._peek() == "(":
                self._consume("(")
                self._parse_line(current_parent, before_move, current_side)
                self._consume(")")

            current_parent = child
            current_side = _flip_side(current_side)

        if not saw_move:
            raise ValueError("Empty variation is not allowed")


def parse_hexata_format(text: str) -> tuple[int, MoveTree]:
    try:
        return _Parser(text=text).parse()
    except RecursionError as exc:
        raise ValueError("Move text nesting too deep") from exc
