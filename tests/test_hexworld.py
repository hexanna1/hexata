import unittest

from board import Move, MoveKind, Side
from formats.hexworld import build_hexworld_url, parse_hexworld_position, parse_hexworld_state


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

    def test_parse_swap_sides_token_is_ignored_not_swap_piece(self):
        size, past, future, to_play = parse_hexworld_position("https://hexworld.org/board/#5c1,c2:Sd3")

        self.assertEqual(size, 5)
        self.assertEqual(future, [])
        self.assertEqual([mv.kind for mv in past], [MoveKind.PLACE, MoveKind.PLACE])
        self.assertEqual((past[0].side, past[0].col, past[0].row), (Side.RED, 3, 2))
        self.assertEqual((past[1].side, past[1].col, past[1].row), (Side.BLUE, 4, 3))
        self.assertEqual(to_play, Side.RED)

    def test_parse_forfeit_and_resign_tokens_are_ignored(self):
        size, past, future, to_play = parse_hexworld_position("https://hexworld.org/board/#5c1,c2:rw:fbd3")

        self.assertEqual(size, 5)
        self.assertEqual(future, [])
        self.assertEqual([mv.kind for mv in past], [MoveKind.PLACE, MoveKind.PLACE])
        self.assertEqual((past[0].side, past[0].col, past[0].row), (Side.RED, 3, 2))
        self.assertEqual((past[1].side, past[1].col, past[1].row), (Side.BLUE, 4, 3))
        self.assertEqual(to_play, Side.RED)

    def test_parse_full_grammar_state_keeps_nonsquare_size_and_configs(self):
        cols, rows, configs, past, future, to_play = parse_hexworld_state(
            "https://hexworld.org/board/#5x6r3nc1,a1"
        )

        self.assertEqual((cols, rows), (5, 6))
        self.assertEqual(configs, ("r3", "n", "c1"))
        self.assertEqual([mv.kind for mv in past], [MoveKind.PLACE])
        self.assertEqual((past[0].side, past[0].col, past[0].row), (Side.RED, 1, 1))
        self.assertEqual(future, [])
        self.assertEqual(to_play, Side.BLUE)

    def test_hexata_wrapper_rejects_nonsquare_board(self):
        with self.assertRaises(ValueError):
            parse_hexworld_position("https://hexworld.org/board/#5x6c1,a1")

    def test_parse_rejects_too_many_comma_sections(self):
        with self.assertRaises(ValueError):
            parse_hexworld_position("https://hexworld.org/board/#5c1,a1,b2,c3")

    def test_build_url_with_future_moves(self):
        self.assertEqual(
            build_hexworld_url(5, [Move.place(Side.RED, 1, 1)], [Move.place(Side.BLUE, 2, 2)]),
            "https://hexworld.org/board/#5c1,a1,b2",
        )

if __name__ == "__main__":
    unittest.main()
