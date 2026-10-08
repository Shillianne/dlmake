import unittest
from dlmake import Config


class ConfigTest(unittest.TestCase):

    def test_correct_config_mha_simple_json(self):
        config = Config.from_json("tests/configs/mha_simple.json")

    def test_correct_config_mha_simple_yaml(self):
        config = Config.from_yaml("tests/configs/mha_simple.yaml")

    def cross_type_continuity(self):
        config_json = Config.from_json("tests/configs/mha_simple.json")
        config_yaml = Config.from_yaml("tests/configs/mha_simple.yaml")
        self.assertEqual(config_json, config_yaml, "Configurations in different languages should be equal")
