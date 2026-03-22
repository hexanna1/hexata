import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
from board import Move, Side
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

    def test_run_batch_fast_cli_plies_red_winrate_is_post_move_with_final_extra_call(self):
        class _BatchCore:
            def __init__(self):
                self.engine = object()
                self.board = _AlwaysEmptyBoard(3)
                self._past = []
                self._future = [
                    Move.place(Side.RED, 1, 1),
                    Move.place(Side.BLUE, 2, 1),
                ]

            def go_first(self):
                self._future = self._past + self._future
                self._past = []
                return True

            def step_forward(self):
                if not self._future:
                    return False
                self._past.append(self._future.pop(0))
                return True

            def current_ply(self):
                return len(self._past)

            def next_mainline_move(self):
                if not self._future:
                    return None
                return self._future[0]

            def mainline_tail_moves(self):
                return list(self._future)

            def _map_coords_to_engine(self, col, row):
                return (col, row)

            def _map_side_to_engine(self, side):
                return side

        core = _BatchCore()
        raw_seq = [
            RawNNResult(white_win=0.9, policy_rows=(), policy_pass=0.0),  # pre-ply1
            RawNNResult(white_win=0.8, policy_rows=(), policy_pass=0.0),  # pre-ply2 = post-ply1
            RawNNResult(white_win=0.7, policy_rows=(), policy_pass=0.0),  # final = post-ply2
        ]
        with patch("cli._run_kata_raw_nn_once", side_effect=raw_seq) as raw_once, patch(
            "cli._score_policy_move_fast_batch", return_value=None
        ) as score:
            ok, payload = cli._run_batch_fast_cli(core, include_plies=True)

        self.assertTrue(ok)
        plies = payload["plies"]
        self.assertEqual(len(plies), 2)
        # red_winrate is post-move, filled by the next position's raw-NN eval.
        self.assertEqual(plies[0]["red_winrate"], 0.2)  # from white_win=0.8
        self.assertEqual(plies[1]["red_winrate"], 0.3)  # from final extra call white_win=0.7
        self.assertEqual(raw_once.call_count, 3)  # 2 plies + 1 final fill
        self.assertEqual(score.call_count, 1)  # ply 1 skipped; ply 2 attempted

    def test_run_cli_analyze_streams_one_result_per_position_and_clears_cache_between(self):
        engine = SimpleNamespace(clear_cache=Mock(), close=Mock())
        core = SimpleNamespace(
            load_hexworld_text=Mock(
                side_effect=lambda text: None if text != "bad" else "HexWorld parse failed: bad"
            ),
        )
        args = SimpleNamespace(
            cli_cmd="analyze",
            positions=["good-1", "bad", "good-2"],
            top_n=None,
            search_seconds=None,
            analysis_wide_root_noise=None,
        )
        ok_payload = {
            "mode": "analyze",
            "method": "raw_nn",
            "best_reply": None,
            "root_eval": {"red_winrate": 0.4},
            "moves": [],
        }

        with patch("cli.KataHexEngine", return_value=engine), patch(
            "cli.GuiCore", return_value=core
        ), patch("cli._position_payload", return_value={}), patch(
            "cli._run_cli_analyze", side_effect=[(True, ok_payload), (True, ok_payload)]
        ) as run_analyze, patch("cli._emit") as emit:
            exit_code = cli.run_cli(args, engine_cmd=["katahex"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(engine.clear_cache.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in core.load_hexworld_text.call_args_list],
            ["good-1", "bad", "good-2"],
        )
        self.assertEqual(run_analyze.call_count, 2)
        self.assertEqual(emit.call_count, 3)
        self.assertEqual(
            [(call.args[0]["input"], call.args[0]["ok"]) for call in emit.call_args_list],
            [("good-1", True), ("bad", False), ("good-2", True)],
        )
        engine.close.assert_called_once_with()

    def test_iter_analyze_positions_expands_stdin_sentinel_in_position_order(self):
        with patch("cli.sys.stdin", io.StringIO("\nstdin-1\nstdin-2\n\n")):
            self.assertEqual(
                list(cli._iter_analyze_positions(["good-1", "-", "good-2"])),
                ["good-1", "stdin-1", "stdin-2", "good-2"],
            )


if __name__ == "__main__":
    unittest.main()
