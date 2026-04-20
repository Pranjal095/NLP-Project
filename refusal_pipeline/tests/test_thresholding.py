import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multilingual_support.thresholding import tune_binary_threshold, tune_multilabel_thresholds


class ThresholdingTests(unittest.TestCase):
    def test_binary_threshold_returns_reasonable_value(self):
        result = tune_binary_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        self.assertGreaterEqual(result.threshold, 0.1)
        self.assertLessEqual(result.threshold, 0.9)
        self.assertGreaterEqual(result.metric_value, 0.0)

    def test_multilabel_thresholds_match_num_labels(self):
        results = tune_multilabel_thresholds(
            [[1, 0], [0, 1], [1, 1]],
            [[0.8, 0.2], [0.4, 0.9], [0.7, 0.8]],
        )
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
