"""
End-to-end smoke test for the refusal pipeline without external downloads.

This script creates a tiny local sequence-classification checkpoint in /tmp,
loads it through the same AutoTokenizer/AutoModel path used by demo.py, and
runs a dialogue through context handling, detector inference, scenario routing,
and deterministic refusal generation.

The tiny model is not a trained detector. Its classifier bias is set so class 1
is always preferred, which makes the manipulative/refusal branch deterministic.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from demo import load_model, run_pipeline


SMOKE_MODEL_DIR = Path("/tmp/refusal_pipeline_smoke_model")


def build_tiny_checkpoint(model_dir=SMOKE_MODEL_DIR):
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    vocab_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "person1", "person2", "if", "you", "really", "loved", "me",
        "would", "do", "this", "right", "now", "thanks", "help",
        "refund", "threat", "consequences", "need", "options",
    ]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)

    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.classifier.bias[0] = -4.0
        model.classifier.bias[1] = 4.0

    model.save_pretrained(model_dir)
    (model_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "detector_model_name": "tiny-local-bert-smoke",
                "threshold": 0.5,
                "max_length": 64,
                "label_map": {"0": "Non-manipulative", "1": "Manipulative"},
                "purpose": "smoke-test-only; not a trained detector",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_dir


def main():
    model_dir = build_tiny_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(str(model_dir), device)

    dialogue = (
        "Person1: If you really loved me, you would do this right now.\n"
        "Person2: I need time to think.\n"
        "Person1: After everything I've done, you owe me."
    )
    result = run_pipeline(
        model,
        tokenizer,
        dialogue,
        device,
        max_length=64,
        threshold=0.5,
        attribute_classifiers={},
        context_turns=3,
    )

    assert result["is_manipulative"] is True
    assert result["scenario"] == "guilt_trip"
    assert "not able to act based on emotional pressure" in result["response"]
    assert result["context_signals"].get("cumulative_risk") is True

    print("\nSmoke test passed.")
    print(f"Temporary checkpoint: {model_dir}")


if __name__ == "__main__":
    main()
