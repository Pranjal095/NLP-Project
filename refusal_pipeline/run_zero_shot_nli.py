"""CLI entry point for zero-shot multilingual NLI evaluation."""

from __future__ import annotations

import argparse
import json

from multilingual_support.preprocessing import PreprocessingConfig
from multilingual_support.zero_shot import DEFAULT_NLI_MODEL, evaluate_zero_shot_nli


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot multilingual NLI baseline")
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--model_name", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--hypothesis", default="This dialogue is manipulative.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default="./saved_model/zero_shot_nli")
    parser.add_argument("--transliteration_normalization", choices=["auto", "basic", "external", "off"], default="auto")
    args = parser.parse_args()

    preprocessing = PreprocessingConfig(transliteration_normalization=args.transliteration_normalization)
    metrics = evaluate_zero_shot_nli(
        args.data_path,
        model_name=args.model_name,
        hypothesis=args.hypothesis,
        threshold=args.threshold,
        output_dir=args.output_dir,
        preprocessing_config=preprocessing,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
