import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
from board import Side
from engine import AnalysisMove, RawNNResult


class _AlwaysEmptyBoard:
    def __init__(self, n: int):
        self.n = n

    def is_empty(self, col: int, row: int) -> bool:
        return True


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
        self.assertEqual(payload["analyze"]["method"], "raw_nn")
        self.assertEqual(payload["analyze"]["best"], {"move": "pass", "prior": 0.3})
        self.assertEqual(payload["analyze"]["root_eval"], {"red_winrate": 0.4})
        self.assertEqual(len(payload["analyze"]["moves"]), 3)
        self.assertNotIn("red_winrate", payload["analyze"]["moves"][0])
        self.assertNotIn("visits", payload["analyze"]["moves"][0])
        raw_once.assert_called_once_with(core.engine)
        run_search.assert_not_called()

    def test_run_cli_analyze_streams_one_result_per_position_and_clears_cache_between(self):
        engine = SimpleNamespace(clear_cache=Mock(), close=Mock())
        core = SimpleNamespace(
            load_hexworld_text=Mock(
                side_effect=lambda text: None if text != "bad" else "HexWorld parse failed: bad"
            ),
            build_hexworld_url=Mock(
                side_effect=[
                    "https://hexworld.org/board/#14c1,a1",
                    "https://hexworld.org/board/#14c1,b1",
                ]
            ),
        )
        args = SimpleNamespace(
            cli_cmd="analyze",
            position=["good-1", "bad", "good-2"],
            top_n=None,
            search_seconds=None,
            analysis_wide_root_noise=None,
        )
        ok_payload = {
            "analyze": {
                "method": "raw_nn",
                "best": {"move": "a1", "prior": 0.2},
                "root_eval": {"red_winrate": 0.4},
                "moves": [],
            },
        }

        with patch("cli.KataHexEngine", return_value=engine), patch(
            "cli.GuiCore", return_value=core
        ), patch("cli._run_cli_analyze", side_effect=[(True, ok_payload), (True, ok_payload)]
        ) as run_analyze, patch("cli._emit") as emit:
            exit_code = cli.run_cli(args, engine_cmd=["katahex"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(emit.call_count, 3)
        self.assertEqual(
            [(call.args[0]["hexworld"], call.args[0]["ok"]) for call in emit.call_args_list],
            [
                ("https://hexworld.org/board/#14c1,a1", True),
                ("https://hexworld.org/board/#bad", False),
                ("https://hexworld.org/board/#14c1,b1", True),
            ],
        )
        engine.close.assert_called_once_with()

    def test_iter_cli_positions_expands_stdin_sentinel_in_position_order(self):
        with patch("cli.sys.stdin", io.StringIO("\nstdin-1\nstdin-2\n\n")):
            self.assertEqual(
                list(cli._iter_cli_positions(["good-1", "-", "good-2"])),
                ["good-1", "stdin-1", "stdin-2", "good-2"],
            )

    def test_run_cli_match_search_mode_uses_top_child_eval_for_resignation(self):
        engine_a = SimpleNamespace(
            clear_board=Mock(), clear_cache=Mock(), kata_set_param=Mock(), play=Mock(), close=Mock()
        )
        engine_b = SimpleNamespace(
            clear_board=Mock(), clear_cache=Mock(), kata_set_param=Mock(), play=Mock(), close=Mock()
        )
        args = SimpleNamespace(
            cli_cmd="match",
            engine_a="main",
            engine_b="alt",
            openings="a1",
            size=14,
            rounds=1,
            search_seconds=1.0,
            visits_temp=0.5,
            visits_temp_decay=1.0,
            resign_winrate=0.01,
        )
        recs = [
            AnalysisMove("b1", order=0, col=2, row=1, winrate=0.0, visits=100, prior=None, pv=None),
        ]

        with patch("cli.KataHexEngine", side_effect=[engine_a, engine_b]), patch(
            "cli._run_match_search_for_seconds", side_effect=[("completed", recs), ("completed", recs)]
        ) as run_search, patch("cli._emit") as emit:
            ok, payload = cli._run_cli_match(args, engine_a_cmd=["a"], engine_b_cmd=["b"])

        self.assertTrue(ok)
        self.assertEqual(payload, {})
        self.assertEqual(run_search.call_count, 2)
        self.assertEqual(run_search.call_args_list[0].args[2], 1.0)
        self.assertEqual(run_search.call_args_list[1].args[2], 1.0)
        self.assertEqual(
            [(call.args[0]["match"]["winner"], call.args[0]["match"]["result"]) for call in emit.call_args_list],
            [("main", "blue_resigned"), ("alt", "blue_resigned")],
        )
        game0 = emit.call_args_list[0].args[0]
        self.assertEqual(game0["error"], None)
        self.assertIn("meta", game0)
        self.assertEqual(game0["match"]["plies"], [{"ply": 1, "side": "red", "played": "a1"}])

if __name__ == "__main__":
    unittest.main()
