import unittest
import sys

from board import Move, Side
from history_tree import MoveTree


class MoveTreeTests(unittest.TestCase):
    @staticmethod
    def _pass_line(depth: int, start_side: Side = Side.RED) -> list[Move]:
        moves = []
        side = start_side
        for _ in range(depth):
            moves.append(Move.pass_(side=side))
            side = Side.BLUE if side == Side.RED else Side.RED
        return moves

    def test_deep_operations_handle_large_trees(self):
        old_limit = sys.getrecursionlimit()
        depth = 50
        recursion_limit = depth - 5
        sys.setrecursionlimit(recursion_limit)
        try:
            moves = self._pass_line(depth)

            tree = MoveTree()
            tree.rebuild_from_line(moves, [])

            sig = tree.signature()
            clone = tree.clone()

            self.assertEqual(sig, clone.signature())
            self.assertEqual(clone.cursor.id, tree.cursor.id)
            self.assertEqual(clone._next_id, tree._next_id)
            self.assertIsNone(tree.sibling_cursor(1))
            merge_tree = MoveTree()
            root = merge_tree.root
            first = merge_tree.append_child(root, Move.place(side=Side.RED, col=1, row=1))
            second = merge_tree.append_child(root, Move.place(side=Side.RED, col=1, row=1))
            node_a, node_b = first, second
            for mv in self._pass_line(depth, start_side=Side.BLUE):
                node_a = merge_tree.append_child(node_a, mv)
                node_b = merge_tree.append_child(node_b, mv)

            merged = merge_tree.merge_equivalent_siblings(first, second)

            self.assertIs(merged, root.children[0])
            self.assertEqual(len(root.children), 1)
        finally:
            sys.setrecursionlimit(old_limit)


if __name__ == "__main__":
    unittest.main()
