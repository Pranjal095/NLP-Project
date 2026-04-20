import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multilingual_support.preprocessing import PreprocessingConfig, preprocess_text


class PreprocessingTests(unittest.TestCase):
    def test_detects_english(self):
        result = preprocess_text("Please help me with this issue.")
        self.assertEqual(result.metadata.detected_language, "en")
        self.assertEqual(result.metadata.dominant_script, "latin")

    def test_detects_native_script_indic(self):
        result = preprocess_text("आप मेरी बात मानिए, अभी कीजिए।")
        self.assertTrue(result.metadata.detected_language.startswith("indic-"))
        self.assertEqual(result.metadata.dominant_script, "devanagari")

    def test_detects_romanized_indic(self):
        result = preprocess_text(
            "bhai please abhi karo na, kya tum meri help karoge",
            PreprocessingConfig(transliteration_normalization="basic"),
        )
        self.assertTrue(result.metadata.is_romanized_indic)
        self.assertTrue("please" in result.processed_text.lower())


if __name__ == "__main__":
    unittest.main()
