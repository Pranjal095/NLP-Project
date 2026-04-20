import shutil
import tempfile
import unittest
from pathlib import Path
import sys

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multilingual_support.inference import load_text_model, predict_text
from multilingual_support.modeling import MultilingualModelConfig, UnifiedTextClassifier, build_tokenizer, save_multilingual_checkpoint
from smoke_test import build_tiny_checkpoint


class MultilingualInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="refusal-pipeline-test-"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_custom_multilingual_checkpoint_loads_and_predicts(self):
        backbone_dir = build_tiny_checkpoint(self.temp_dir / "tiny_backbone")
        tokenizer = build_tokenizer(str(backbone_dir))
        config = MultilingualModelConfig(
            backbone_name=str(backbone_dir),
            task_name="detection",
            problem_type="single_label_classification",
            num_labels=2,
            max_length=64,
            threshold=0.5,
        )
        model = UnifiedTextClassifier(config)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
            model.classifier.bias[0] = -4.0
            model.classifier.bias[1] = 4.0
        checkpoint_dir = save_multilingual_checkpoint(model, tokenizer, self.temp_dir / "multilingual_checkpoint")

        device = torch.device("cpu")
        loaded_model, loaded_tokenizer, loaded_config, mode = load_text_model(str(checkpoint_dir), device=device)
        prediction = predict_text(
            loaded_model,
            loaded_tokenizer,
            "Person1: If you really loved me, do this right now.",
            device,
            max_length=64,
            threshold=0.5,
            task_config=loaded_config,
            mode=mode,
        )
        self.assertEqual(mode, "multilingual")
        self.assertEqual(prediction["prediction"], 1)
        self.assertIn("language", prediction["metadata"])


if __name__ == "__main__":
    unittest.main()
