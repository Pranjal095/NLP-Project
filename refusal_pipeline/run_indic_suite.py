"""Benchmark runner for Indic-style multilingual evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from multilingual_support.dataset_utils import enrich_language_metadata, load_dataframe
from multilingual_support.inference import load_text_model, predict_text
from multilingual_support.metrics import compute_binary_metrics, save_structured_results, save_summary_plot
from multilingual_support.preprocessing import PreprocessingConfig


def _evaluate_subset(
    frame: pd.DataFrame,
    *,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    threshold: float,
    config: dict,
    mode: str,
    preprocessing: PreprocessingConfig,
) -> dict:
    probabilities = []
    for text in frame["Dialogue"].astype(str):
        prediction = predict_text(
            model,
            tokenizer,
            text,
            device,
            max_length=max_length,
            threshold=threshold,
            task_config=config,
            mode=mode,
            preprocessing_config=preprocessing,
        )
        probabilities.append(prediction["probability"])
    return compute_binary_metrics(
        frame["Manipulative"].astype(int).tolist(),
        probabilities,
        threshold=threshold,
        language_labels=frame["language"].tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Indic-style multilingual benchmark suite")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--output_dir", default="./saved_model/indic_suite")
    parser.add_argument("--train_languages", nargs="*", default=["en"])
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    preprocessing = PreprocessingConfig(transliteration_normalization="auto")
    dataframe = enrich_language_metadata(load_dataframe(args.data_path), config=preprocessing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config, mode = load_text_model(args.model_path, device=device)
    max_length = int(config.get("max_length", 256))
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))

    suites = {
        "english_only": dataframe[dataframe["language"] == "en"].reset_index(drop=True),
        "multilingual_all": dataframe.reset_index(drop=True),
        "zero_shot_unseen_languages": dataframe[~dataframe["language"].isin(args.train_languages)].reset_index(drop=True),
        "code_mixed": dataframe[dataframe["is_code_mixed"]].reset_index(drop=True),
        "romanized_indic": dataframe[dataframe["is_romanized_indic"]].reset_index(drop=True),
    }

    results = {}
    for name, frame in suites.items():
        results[name] = (
            _evaluate_subset(
                frame,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                threshold=threshold,
                config=config,
                mode=mode,
                preprocessing=preprocessing,
            )
            if not frame.empty
            else {"note": "No samples available for this subset"}
        )

    output_dir = Path(args.output_dir)
    save_structured_results(results, output_dir, "indic_suite")
    save_summary_plot({"per_language": {key: value for key, value in results.items() if isinstance(value, dict) and "f1" in value}}, output_dir, "indic_suite")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
