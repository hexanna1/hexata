import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
from board import GameType, Side
from engine import AnalysisMove, parse_analysis_move_token, to_analysis_token


class CliTests(unittest.TestCase):
    def test_winrate_helper_handles_side_perspective(self):
        self.assertEqual(cli._side_winrate_to_red(0.3, Side.RED), 0.3)
        self.assertEqual(cli._side_winrate_to_red(0.3, Side.BLUE), 0.7)
        self.assertEqual(cli._side_winrate_to_red(0.7, Side.RED), 0.7)
        self.assertEqual(cli._side_winrate_to_red(0.7, Side.BLUE), 0.3)

    def test_y_analysis_tokens_use_direct_coords(self):
        self.assertEqual(parse_analysis_move_token("b1", 5, GameType.Y), (2, 1))
        self.assertEqual(to_analysis_token(2, 1, 5, GameType.Y), "b1")
        self.assertIsNone(parse_analysis_move_token("d3", 5, GameType.Y))

    def test_run_cli_analyze_uses_analysis_subcommand(self):
        args = SimpleNamespace(
            cli_cmd="analyze",
            position=["good-1", "bad", "good-2"],
            top_n=None,
            visits=None,
            analysis_wide_root_noise=None,
        )

        with patch("cli._run_analyze_positions", return_value=1) as run_analyze:
            exit_code = cli.run_cli(args, engine_cmd=["katahex"])

        self.assertEqual(exit_code, 1)
        run_analyze.assert_called_once_with(
            args,
            engine_cmd=["katahex"],
            game_type=GameType.HEX,
        )

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
