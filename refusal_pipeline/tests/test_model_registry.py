import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multilingual_support.model_registry import get_backbone_spec, list_backbones


class ModelRegistryTests(unittest.TestCase):
    def test_expected_backbones_exist(self):
        names = set(list_backbones())
        expected = {
            "roberta-base",
            "xlm-roberta-base",
            "microsoft/mdeberta-v3-base",
            "google/muril-base-cased",
            "ai4bharat/IndicBERTv2-SS",
            "google/mt5-base",
            "google/byt5-small",
            "sentence-transformers/LaBSE",
            "intfloat/multilingual-e5-base",
        }
        self.assertTrue(expected.issubset(names))

    def test_alias_lookup_resolves(self):
        muril = get_backbone_spec("muril")
        self.assertEqual(muril.model_name, "google/muril-base-cased")


if __name__ == "__main__":
    unittest.main()
