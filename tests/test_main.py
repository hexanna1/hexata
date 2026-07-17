import configparser
import unittest

import main


class MainTests(unittest.TestCase):
    def _parser(self, text: str) -> configparser.ConfigParser:
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(text)
        return parser

    def test_select_engine_profile_defaults_to_first(self):
        parser = self._parser(
            """
[engine.hex.beta]
cmd = beta

[engine.hex.alpha]
cmd = alpha
"""
        )

        profiles = main._engine_profiles_from_parser(parser)
        selected = main._select_engine_profile(profiles, game_type=main.GameType.HEX)

        self.assertEqual([p.name for p in profiles], ["beta", "alpha"])
        self.assertEqual(selected.name, "beta")

    def test_select_engine_profile_uses_game_default(self):
        parser = self._parser(
            """
[engine.hex]
default = alpha

[engine.hex.beta]
cmd = beta

[engine.hex.alpha]
cmd = alpha
"""
        )

        profiles = main._engine_profiles_from_parser(parser)
        selected = main._select_engine_profile(profiles, game_type=main.GameType.HEX)

        self.assertEqual(selected.name, "alpha")

    def test_select_engine_profile_uses_requested_name(self):
        parser = self._parser(
            """
[engine.hex]
default = alpha

[engine.hex.beta]
cmd = beta

[engine.hex.alpha]
cmd = alpha
"""
        )

        profiles = main._engine_profiles_from_parser(parser)
        selected = main._select_engine_profile(
            profiles,
            game_type=main.GameType.HEX,
            requested_name="beta",
        )

        self.assertEqual(selected.name, "beta")

if __name__ == "__main__":
    unittest.main()
