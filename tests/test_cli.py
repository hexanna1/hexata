import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
from board import Side
from engine import RawNNResult


class _AlwaysEmptyBoard:
    def __init__(self, n: int):
        self.n = n

    def is_empty(self, col: int, row: int) -> bool:
        return True

    def in_bounds(self, col: int, row: int) -> bool:
        return 1 <= col <= self.n and 1 <= row <= self.n


class CliTests(unittest.TestCase):
    def test_raw_winrate_helpers_handle_swap_perspective(self):
        raw = RawNNResult(white_win=0.7, policy_rows=(), policy_pass=None)

        core_no_swap = SimpleNamespace(_map_side_to_engine=lambda side: side)
        self.assertEqual(cli._raw_red_winrate(core_no_swap, raw), 0.3)
        self.assertEqual(cli._side_winrate_to_red(0.3, Side.RED), 0.3)
        self.assertEqual(cli._side_winrate_to_red(0.3, Side.BLUE), 0.7)

        core_swap = SimpleNamespace(
            _map_side_to_engine=lambda side: (Side.BLUE if side == Side.RED else Side.RED)
        )
        self.assertEqual(cli._raw_red_winrate(core_swap, raw), 0.7)
        self.assertEqual(cli._side_winrate_to_red(0.7, Side.RED), 0.7)
        self.assertEqual(cli._side_winrate_to_red(0.7, Side.BLUE), 0.3)

    def test_run_cli_analyze_omitted_search_seconds_uses_raw_nn(self):
        core = SimpleNamespace(
            engine=object(),
            board=_AlwaysEmptyBoard(2),
            _map_coords_to_engine=lambda col, row: (col, row),
            _map_side_to_engine=lambda side: side,
        )
        args = SimpleNamespace(search_seconds=None, top_n=3)
        raw = RawNNResult(
            white_win=0.6,
            policy_rows=((0.1, 0.2), (0.0, None)),
            policy_pass=0.3,
        )

        with patch("cli._run_kata_raw_nn_once", return_value=raw) as raw_once, patch(
            "cli._run_for_seconds_from_first_update", side_effect=AssertionError("search path used")
        ) as run_search:
            ok, payload = cli._run_cli_analyze(core, args)

        self.assertTrue(ok)
        self.assertEqual(payload["mode"], "analyze")
        self.assertEqual(payload["method"], "raw_nn")
        self.assertEqual(payload["best_reply"], None)
        self.assertEqual(payload["root_eval"], {"red_winrate": 0.4})
        self.assertEqual(len(payload["moves"]), 3)
        raw_once.assert_called_once_with(core.engine)
        run_search.assert_not_called()

    def test_run_cli_candidate_raw_nn_uses_undo_on_success_and_failure(self):
        board = _AlwaysEmptyBoard(3)
        engine = SimpleNamespace(undo=Mock())
        play_engine_mapped = Mock()
        core = SimpleNamespace(
            board=board,
            engine=engine,
            app=SimpleNamespace(analysis_running=False),
            set_analysis_wide_root_noise=Mock(),
            current_side=lambda: Side.RED,
            _map_side_to_engine=lambda side: side,
            play_engine_mapped=play_engine_mapped,
        )

        success_args = SimpleNamespace(total_search_seconds=None, moves="a1,b1")
        raw1 = RawNNResult(white_win=0.2, policy_rows=(), policy_pass=None)
        raw2 = RawNNResult(white_win=0.8, policy_rows=(), policy_pass=None)
        with patch("cli._run_kata_raw_nn_once", side_effect=[raw1, raw2]) as raw_once, patch(
            "cli._run_for_seconds_from_first_update", side_effect=AssertionError("search path used")
        ) as run_search:
            ok, payload = cli._run_cli_candidate(core, success_args)

        self.assertTrue(ok)
        self.assertEqual(payload["mode"], "candidate")
        self.assertEqual(payload["method"], "raw_nn")
        self.assertEqual([row["move"] for row in payload["moves"]], ["a1", "b1"])
        self.assertEqual([row["red_winrate"] for row in payload["moves"]], [0.8, 0.2])
        self.assertEqual(engine.undo.call_count, 2)
        self.assertEqual(play_engine_mapped.call_count, 2)
        self.assertEqual(raw_once.call_count, 2)
        run_search.assert_not_called()

        engine.undo.reset_mock()
        play_engine_mapped.reset_mock()
        fail_args = SimpleNamespace(total_search_seconds=None, moves="a1")
        with patch("cli._run_kata_raw_nn_once", return_value=None):
            ok, payload = cli._run_cli_candidate(core, fail_args)

        self.assertFalse(ok)
        self.assertIn("No raw-NN reply received", payload["error"])
        self.assertEqual(engine.undo.call_count, 1)
        self.assertEqual(play_engine_mapped.call_count, 1)


if __name__ == "__main__":
    unittest.main()
