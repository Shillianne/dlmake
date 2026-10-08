import importlib
import unittest


class ImportTest(unittest.TestCase):

    def test_imports(self):
        module = importlib.import_module("dlmake")
        for name in module.__all__:
            with self.subTest(name=name):
                namespace = {}
                exec(f"from dlmake import {name}", namespace)
                self.assertIn(name, namespace)
        utils = importlib.import_module("dlmake.utils")
        for name in utils.__all__:
            with self.subTest(name=name):
                namespace = {}
                exec(f"from dlmake.utils import {name}", namespace)
                self.assertIn(name, namespace)
