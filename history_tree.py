from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from board import Move


@dataclass(slots=True)
class HistoryNode:
    id: int
    move: Optional[Move]
    parent: Optional["HistoryNode"] = None
    children: list["HistoryNode"] = field(default_factory=list)

    @property
    def preferred_child(self) -> Optional["HistoryNode"]:
        return self.children[0] if self.children else None

    def variation_children(self) -> list["HistoryNode"]:
        return self.children[1:]

    def remove_child(self, child: "HistoryNode") -> bool:
        try:
            self.children.remove(child)
        except ValueError:
            return False
        child.parent = None
        return True

    def promote_child(self, child: "HistoryNode") -> bool:
        try:
            idx = self.children.index(child)
        except ValueError:
            return False
        if idx > 0:
            self.children.insert(0, self.children.pop(idx))
        return True


class MoveTree:
    def __init__(self) -> None:
        self.root = HistoryNode(id=0, move=None)
        self.cursor = self.root
        self._next_id = 1

    def clear(self) -> None:
        self.root.children.clear()
        self.cursor = self.root
        self._next_id = 1

    def clone(self) -> "MoveTree":
        out = MoveTree()
        out.root = HistoryNode(id=self.root.id, move=self.root.move, parent=None)
        id_to_new: dict[int, HistoryNode] = {out.root.id: out.root}
        stack: list[tuple[HistoryNode, HistoryNode]] = [(self.root, out.root)]
        while stack:
            src, dst = stack.pop()
            for child in src.children:
                new_child = HistoryNode(id=child.id, move=child.move, parent=dst)
                dst.children.append(new_child)
                id_to_new[new_child.id] = new_child
                stack.append((child, new_child))
        out.cursor = id_to_new[self.cursor.id]
        out._next_id = self._next_id
        return out

    def signature(self) -> tuple:
        tokens: list[tuple[Optional[Move], int]] = []
        stack: list[HistoryNode] = [self.root]
        while stack:
            node = stack.pop()
            tokens.append((node.move, len(node.children)))
            for child in reversed(node.children):
                stack.append(child)
        return (self.cursor.id, tuple(tokens))

    def current_path_nodes(self) -> list[HistoryNode]:
        out: list[HistoryNode] = []
        node = self.cursor
        while node.parent is not None:
            out.append(node)
            node = node.parent
        out.reverse()
        return out

    def current_path_moves(self) -> list[Move]:
        return [node.move for node in self.current_path_nodes()]

    def mainline_tail_moves(self) -> list[Move]:
        out: list[Move] = []
        node = self.cursor.preferred_child
        while node is not None:
            out.append(node.move)
            node = node.preferred_child
        return out

    def visible_line_moves(self) -> list[Move]:
        return self.current_path_moves() + self.mainline_tail_moves()

    def current_ply(self) -> int:
        return len(self.current_path_nodes())

    def next_mainline_move(self) -> Optional[Move]:
        child = self.cursor.preferred_child
        return None if child is None else child.move

    def variation_children(self) -> list[HistoryNode]:
        return self.cursor.variation_children()

    def variation_moves(self) -> list[Move]:
        return [child.move for child in self.variation_children()]

    def _frontier_at_ply(self, target_ply: int) -> list[HistoryNode]:
        frontier: list[HistoryNode] = []
        if target_ply <= 0:
            return frontier

        stack: list[tuple[HistoryNode, int]] = [
            (child, 1) for child in reversed(self.root.children)
        ]
        while stack:
            node, ply = stack.pop()
            if ply == target_ply:
                frontier.append(node)
                continue
            for child in reversed(node.children):
                stack.append((child, ply + 1))
        return frontier

    def _same_ply_neighbor_cursor(self, direction: int) -> Optional[HistoryNode]:
        frontier = self._frontier_at_ply(self.current_ply())
        cursor_idx = next((idx for idx, node in enumerate(frontier) if node.id == self.cursor.id), None)
        if cursor_idx is None:
            return None
        neighbor_idx = cursor_idx + direction
        if 0 <= neighbor_idx < len(frontier):
            return frontier[neighbor_idx]
        return None

    def sibling_cursor(self, direction: int) -> Optional[HistoryNode]:
        if direction not in (-1, 1):
            raise AssertionError(f"Invalid sibling direction: {direction}")
        same_ply = self._same_ply_neighbor_cursor(direction)
        if same_ply is not None or direction == 1:
            return same_ply

        path = self.current_path_nodes()
        for depth in range(len(path) - 1, -1, -1):
            parent = self.root if depth == 0 else path[depth - 1]
            idx = parent.children.index(path[depth])
            if idx > 0:
                node = parent.children[idx - 1]
                while node.preferred_child is not None:
                    node = node.preferred_child
                return node
        return None

    def find_child(self, parent: HistoryNode, move: Move) -> Optional[HistoryNode]:
        for child in parent.children:
            if child.move == move:
                return child
        return None

    def append_child(self, parent: HistoryNode, move: Move) -> HistoryNode:
        child = HistoryNode(id=self._next_id, move=move, parent=parent)
        self._next_id += 1
        parent.children.append(child)
        return child

    def add_or_select_child(self, move: Move, *, promote: bool = True) -> tuple[HistoryNode, bool]:
        child = self.find_child(self.cursor, move)
        created = False
        if child is None:
            child = self.append_child(self.cursor, move)
            if promote:
                self.cursor.promote_child(child)
            created = True
        elif promote:
            if not self.cursor.promote_child(child):
                raise AssertionError("Child missing during promotion")
        self.cursor = child
        return child, created

    def follow_child(self, move: Move, *, promote: bool) -> bool:
        child = self.find_child(self.cursor, move)
        if child is None:
            return False
        if promote:
            if not self.cursor.promote_child(child):
                raise AssertionError("Child missing during promotion")
        self.cursor = child
        return True

    def step_back(self) -> bool:
        if self.cursor.parent is None:
            return False
        self.cursor = self.cursor.parent
        return True

    def step_forward(self) -> bool:
        child = self.cursor.preferred_child
        if child is None:
            return False
        self.cursor = child
        return True

    def delete_selected_tail(self) -> bool:
        child = self.cursor.preferred_child
        if child is None:
            return False
        self.cursor.remove_child(child)
        return True

    def delete_cursor_node(self) -> bool:
        node = self.cursor
        parent = node.parent
        if parent is None:
            return False
        parent.remove_child(node)
        self.cursor = parent
        return True

    def _absorb_matching_subtree(self, preferred: HistoryNode, absorbed: HistoryNode) -> HistoryNode:
        stack: list[tuple[HistoryNode, HistoryNode, object | None]] = [(preferred, absorbed, None)]
        while stack:
            current_preferred, current_absorbed, child_iter = stack.pop()
            if child_iter is None:
                if current_preferred.move != current_absorbed.move:
                    raise AssertionError("Cannot merge subtrees with different moves")
                child_iter = iter(list(current_absorbed.children))
            for child in child_iter:
                if not current_absorbed.remove_child(child):
                    raise AssertionError("Failed to detach absorbed child during merge")
                match = self.find_child(current_preferred, child.move)
                if match is None:
                    child.parent = current_preferred
                    current_preferred.children.append(child)
                    continue
                stack.append((current_preferred, current_absorbed, child_iter))
                stack.append((match, child, None))
                break
            else:
                parent = current_absorbed.parent
                if parent is not None and not parent.remove_child(current_absorbed):
                    raise AssertionError("Failed to remove absorbed node during merge")
        return preferred

    def merge_equivalent_siblings(self, first: HistoryNode, second: HistoryNode) -> HistoryNode:
        if first is second:
            return first
        parent = first.parent
        if parent is None or second.parent is not parent:
            raise AssertionError("Can only merge siblings")
        if first.move != second.move:
            raise AssertionError("Can only merge siblings with identical moves")
        # Child order is the local preference rule, so the earlier sibling keeps
        # precedence when two branches collapse onto the same move.
        first_idx = parent.children.index(first)
        second_idx = parent.children.index(second)
        if first_idx <= second_idx:
            return self._absorb_matching_subtree(first, second)
        return self._absorb_matching_subtree(second, first)

    def rebuild_from_line(self, past_moves: Sequence[Move], future_moves: Sequence[Move]) -> None:
        self.clear()
        node = self.root
        for mv in past_moves:
            child = self.append_child(node, mv)
            node = child
        self.cursor = node
        for mv in future_moves:
            child = self.append_child(node, mv)
            node = child
