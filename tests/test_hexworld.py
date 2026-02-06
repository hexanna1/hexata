import unittest

from board import MoveKind, Side
from hexworld import parse_hexworld_position


class HexWorldSwapTests(unittest.TestCase):
    def test_parse_swap_in_past_stream(self):
        size, past, future, to_play = parse_hexworld_position("https://hexworld.org/board/#5c1,c2:se4")

        self.assertEqual(size, 5)
        self.assertEqual(future, [])
        self.assertEqual([mv.kind for mv in past], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual((past[0].side, past[0].col, past[0].row), (Side.RED, 3, 2))
        self.assertEqual((past[1].side, past[1].col, past[1].row), (Side.BLUE, 3, 2))
        self.assertEqual((past[2].side, past[2].col, past[2].row), (Side.RED, 5, 4))
        self.assertEqual(to_play, Side.BLUE)

    def test_parse_swap_in_future_stream(self):
        size, past, future, to_play = parse_hexworld_position("https://hexworld.org/board/#5c1,c2,:se4")

        self.assertEqual(size, 5)
        self.assertEqual([mv.kind for mv in past], [MoveKind.PLACE])
        self.assertEqual([mv.kind for mv in future], [MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual((future[0].side, future[0].col, future[0].row), (Side.BLUE, 3, 2))
        self.assertEqual((future[1].side, future[1].col, future[1].row), (Side.RED, 5, 4))
        self.assertEqual(to_play, Side.BLUE)

    def test_parse_swap_illegal_position_rejected(self):
        with self.assertRaises(ValueError):
            parse_hexworld_position("https://hexworld.org/board/#5c1,:s")


if __name__ == "__main__":
    unittest.main()
