import csv
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multilingual_support.zero_shot import load_indic_eval_dataframe


class IndicEvalLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="indic-eval-test-"))
        self.csv_path = self.temp_dir / "mini.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "Dialogue", "Manipulative", "Technique", "Vulnerability"])
            writer.writerow(["1", "Please help me with this.", "0", "", ""])
            writer.writerow(["2", "bhai abhi karo na please", "1", "Pressure", "Dependency"])
            writer.writerow(["3", "आप अभी यह कीजिए", "1", "Pressure", "Dependency"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loader_adds_language_metadata(self):
        frame = load_indic_eval_dataframe(str(self.csv_path))
        self.assertIn("language", frame.columns)
        self.assertIn("script", frame.columns)
        self.assertIn("is_code_mixed", frame.columns)
        self.assertEqual(len(frame), 3)


if __name__ == "__main__":
    unittest.main()
