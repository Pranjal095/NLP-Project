"""Evaluate multilingual checkpoints with per-language and robustness reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attribute_classifier import LABELS_BY_TASK, labels_to_multihot
from multilingual_support.dataset_utils import build_multilabel_frame, enrich_language_metadata, load_dataframe
from multilingual_support.inference import load_text_model, predict_text
from multilingual_support.metrics import (
    compute_binary_metrics,
    compute_multilabel_metrics,
    save_structured_results,
    save_summary_plot,
)
from multilingual_support.preprocessing import PreprocessingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multilingual or legacy checkpoints")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--task", choices=["detection", "technique", "vulnerability"], default="detection")
    parser.add_argument("--output_dir", default="./saved_model/multilingual_eval")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--transliteration_normalization", choices=["auto", "basic", "external", "off"], default="auto")
    args = parser.parse_args()

    preprocessing = PreprocessingConfig(
        lowercase=args.lowercase,
        transliteration_normalization=args.transliteration_normalization,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config, mode = load_text_model(args.model_path, device=device)
    max_length = int(config.get("max_length", 256))
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))

    dataframe = enrich_language_metadata(load_dataframe(args.data_path), config=preprocessing)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.task == "detection":
        probabilities = []
        languages = []
        code_mixed_probs = []
        code_mixed_targets = []
        romanized_probs = []
        romanized_targets = []
        for _, row in dataframe.iterrows():
            prediction = predict_text(
                model,
                tokenizer,
                str(row["Dialogue"]),
                device,
                max_length=max_length,
                threshold=threshold,
                task_config=config,
                mode=mode,
                preprocessing_config=preprocessing,
            )
            probabilities.append(prediction["probability"])
            languages.append(row["language"])
            if bool(row["is_code_mixed"]):
                code_mixed_probs.append(prediction["probability"])
                code_mixed_targets.append(int(row["Manipulative"]))
            if bool(row["is_romanized_indic"]):
                romanized_probs.append(prediction["probability"])
                romanized_targets.append(int(row["Manipulative"]))

        metrics = compute_binary_metrics(
            dataframe["Manipulative"].astype(int).tolist(),
            probabilities,
            threshold=threshold,
            language_labels=languages,
        )
        metrics["robustness"] = {
            "code_mixed": compute_binary_metrics(code_mixed_targets, code_mixed_probs, threshold=threshold)
            if code_mixed_targets
            else {},
            "romanized_indic": compute_binary_metrics(romanized_targets, romanized_probs, threshold=threshold)
            if romanized_targets
            else {},
        }
    else:
        task_frame = build_multilabel_frame(dataframe, args.task)
        probabilities = []
        for _, row in task_frame.iterrows():
            prediction = predict_text(
                model,
                tokenizer,
                str(row["Dialogue"]),
                device,
                max_length=max_length,
                threshold=threshold,
                task_config=config,
                mode=mode,
                preprocessing_config=preprocessing,
            )
            probabilities.append(prediction["probabilities"])
        thresholds = config.get("per_label_thresholds") or [threshold] * len(LABELS_BY_TASK[args.task])
        metrics = compute_multilabel_metrics(
            np.asarray([labels_to_multihot(value, args.task) for value in task_frame["Technique" if args.task == "technique" else "Vulnerability"]]),
            np.asarray(probabilities),
            thresholds=thresholds,
            label_names=LABELS_BY_TASK[args.task],
        )

    save_structured_results(metrics, output_path, args.task)
    save_summary_plot(metrics, output_path, args.task)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
