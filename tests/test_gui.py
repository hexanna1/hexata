import unittest
from types import SimpleNamespace
from unittest import mock

import gui


class GuiModuleTests(unittest.TestCase):
    def test_load_position_text_parser_order_and_relayout(self):
        for text, hexworld_error, hexata_error, expected_ok, expected_attempts in (
            ("hexworld", None, "hexata fail", True, [("hexworld", "hexworld")]),
            ("hexata", "hexworld fail", None, True, [("hexworld", "hexata"), ("hexata", "hexata")]),
            ("bad", "hexworld fail", "hexata fail", False, [("hexworld", "bad"), ("hexata", "bad")]),
        ):
            with self.subTest(text=text):
                attempts = []
                calls = []
                core = SimpleNamespace(
                    load_hexworld_text=lambda value, err=hexworld_error: attempts.append(("hexworld", value)) or err,
                    load_hexata_format=lambda value, err=hexata_error: attempts.append(("hexata", value)) or err,
                )

                with mock.patch("gui.logger.info") as info:
                    ok = gui.load_position_text(
                        text,
                        core=core,
                        on_success=lambda: calls.append("relayout"),
                    )

                self.assertEqual(ok, expected_ok)
                self.assertEqual(attempts, expected_attempts)
                self.assertEqual(calls, ["relayout"] if expected_ok else [])
                if expected_ok:
                    info.assert_not_called()
                else:
                    self.assertEqual(info.call_count, 2)

    def test_cycle_engine_profile_success_swaps_engine_and_resumes_analysis(self):
        old_engine = SimpleNamespace(close=mock.Mock())
        new_engine = SimpleNamespace(close=mock.Mock())
        core = SimpleNamespace(
            board=SimpleNamespace(n=11),
            engine=old_engine,
            app=SimpleNamespace(analysis_enabled=True),
            toggle_analysis=mock.Mock(),
            clear_candidates=mock.Mock(),
            clear_all_cached_analysis=mock.Mock(),
            rebuild_engine_from_applied_history=mock.Mock(),
        )
        ui = gui.UiState(
            prefs=gui.UiPrefs(),
            engine_profiles=(
                gui.EngineProfile("main", ("main",)),
                gui.EngineProfile("alt", ("alt",)),
            ),
            current_engine_idx=0,
            speed_last_t=1.0,
            speed_last_total=10,
            speed_vps=20.0,
        )

        with mock.patch("gui.KataHexEngine", return_value=new_engine) as ctor:
            ok = gui.cycle_engine_profile(core, ui, engine_echo=True)

        self.assertTrue(ok)
        ctor.assert_called_once_with(
            board_size=11,
            cmd=["alt"],
            engine_echo=True,
            suppress_stderr=True,
        )
        self.assertIs(core.engine, new_engine)
        old_engine.close.assert_called_once_with()
        core.rebuild_engine_from_applied_history.assert_called_once_with()
        core.clear_candidates.assert_called_once_with()
        core.clear_all_cached_analysis.assert_called_once_with()
        self.assertEqual(core.toggle_analysis.call_count, 2)
        self.assertEqual(ui.current_engine_name, "alt")
        self.assertIsNone(ui.speed_last_t)
        self.assertIsNone(ui.speed_last_total)
        self.assertIsNone(ui.speed_vps)

    def test_cycle_engine_profile_failure_leaves_current_engine_running(self):
        old_engine = SimpleNamespace(close=mock.Mock())
        core = SimpleNamespace(
            board=SimpleNamespace(n=11),
            engine=old_engine,
            app=SimpleNamespace(analysis_enabled=True),
            toggle_analysis=mock.Mock(),
            clear_candidates=mock.Mock(),
            clear_all_cached_analysis=mock.Mock(),
            rebuild_engine_from_applied_history=mock.Mock(),
        )
        ui = gui.UiState(
            prefs=gui.UiPrefs(),
            engine_profiles=(
                gui.EngineProfile("main", ("main",)),
                gui.EngineProfile("alt", ("alt",)),
            ),
            current_engine_idx=0,
        )

        with (
            mock.patch("gui.KataHexEngine", side_effect=RuntimeError("bad model")),
            mock.patch("gui.logger.warning") as warning,
        ):
            ok = gui.cycle_engine_profile(core, ui, engine_echo=False)

        self.assertFalse(ok)
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[:2], ("Engine switch to %s failed: %s", "alt"))
        self.assertEqual(str(warning.call_args.args[2]), "bad model")
        self.assertIs(core.engine, old_engine)
        old_engine.close.assert_not_called()
        core.toggle_analysis.assert_not_called()
        core.clear_candidates.assert_not_called()
        core.clear_all_cached_analysis.assert_not_called()
        core.rebuild_engine_from_applied_history.assert_not_called()
        self.assertEqual(ui.current_engine_name, "main")

    def test_cycle_engine_profile_skips_bad_profile_and_wraps(self):
        old_engine = SimpleNamespace(close=mock.Mock())
        new_engine = SimpleNamespace(close=mock.Mock())
        core = SimpleNamespace(
            board=SimpleNamespace(n=11),
            engine=old_engine,
            app=SimpleNamespace(analysis_enabled=False),
            toggle_analysis=mock.Mock(),
            clear_candidates=mock.Mock(),
            clear_all_cached_analysis=mock.Mock(),
            rebuild_engine_from_applied_history=mock.Mock(),
        )
        ui = gui.UiState(
            prefs=gui.UiPrefs(),
            engine_profiles=(
                gui.EngineProfile("main", ("main",)),
                gui.EngineProfile("bad", ("bad",)),
                gui.EngineProfile("alt", ("alt",)),
            ),
            current_engine_idx=0,
        )

        with (
            mock.patch("gui.KataHexEngine", side_effect=[RuntimeError("bad model"), new_engine]) as ctor,
            mock.patch("gui.logger.warning") as warning,
        ):
            ok = gui.cycle_engine_profile(core, ui, engine_echo=False)

        self.assertTrue(ok)
        self.assertEqual(
            ctor.call_args_list,
            [
                mock.call(
                    board_size=11,
                    cmd=["bad"],
                    engine_echo=False,
                    suppress_stderr=True,
                ),
                mock.call(
                    board_size=11,
                    cmd=["alt"],
                    engine_echo=False,
                    suppress_stderr=True,
                ),
            ],
        )
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[:2], ("Engine switch to %s failed: %s", "bad"))
        self.assertEqual(str(warning.call_args.args[2]), "bad model")
        self.assertIs(core.engine, new_engine)
        old_engine.close.assert_called_once_with()
        core.toggle_analysis.assert_not_called()
        self.assertEqual(ui.current_engine_name, "alt")


if __name__ == "__main__":
    unittest.main()
