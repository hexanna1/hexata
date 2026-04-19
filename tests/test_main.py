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
[engine.beta]
cmd = beta

[engine.alpha]
cmd = alpha
"""
        )

        profiles = main._engine_profiles_from_parser(parser)
        selected = main._select_engine_profile(parser, profiles)

        self.assertEqual([p.name for p in profiles], ["beta", "alpha"])
        self.assertEqual(selected.name, "beta")

    def test_select_engine_profile_uses_default_engine(self):
        parser = self._parser(
            """
[engine]
default_engine = alpha

[engine.beta]
cmd = beta

[engine.alpha]
cmd = alpha
"""
        )

        profiles = main._engine_profiles_from_parser(parser)
        selected = main._select_engine_profile(parser, profiles)

        self.assertEqual(selected.name, "alpha")

    def test_select_engine_profile_uses_requested_name(self):
        parser = self._parser(
            """
[engine]
default_engine = alpha

[engine.beta]
cmd = beta

[engine.alpha]
cmd = alpha
"""
        )

        profiles = main._engine_profiles_from_parser(parser)
        selected = main._select_engine_profile(parser, profiles, requested_name="beta")

        self.assertEqual(selected.name, "beta")

if __name__ == "__main__":
    unittest.main()
