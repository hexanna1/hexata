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
