"""Metrics and result export helpers for multilingual experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .thresholding import expected_calibration_error


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def compute_binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.5,
    language_labels: Sequence[str] | None = None,
) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    preds = (y_prob_arr >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, preds)),
        "precision": float(precision_score(y_true_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_true_arr, preds, zero_division=0)),
        "f1": float(f1_score(y_true_arr, preds, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, preds).tolist(),
        "auroc": _safe_auroc(y_true_arr, y_prob_arr),
        "auprc": _safe_auprc(y_true_arr, y_prob_arr),
        "ece": float(expected_calibration_error(y_true_arr, y_prob_arr)),
        "brier": float(np.mean((y_prob_arr - y_true_arr) ** 2)),
    }

    if language_labels is not None:
        per_language = {}
        for language in sorted(set(language_labels)):
            indices = [idx for idx, label in enumerate(language_labels) if label == language]
            if not indices:
                continue
            per_language[language] = compute_binary_metrics(
                y_true_arr[indices],
                y_prob_arr[indices],
                threshold=threshold,
            )
        metrics["per_language"] = per_language
    return metrics


def compute_multilabel_metrics(
    y_true: Sequence[Sequence[int]],
    y_prob: Sequence[Sequence[float]],
    *,
    thresholds: Sequence[float] | float = 0.5,
    label_names: Sequence[str] | None = None,
) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if np.isscalar(thresholds):
        threshold_arr = np.full(y_true_arr.shape[1], float(thresholds))
    else:
        threshold_arr = np.asarray(thresholds, dtype=float)
    preds = (y_prob_arr >= threshold_arr.reshape(1, -1)).astype(int)

    result = {
        "subset_accuracy": float(accuracy_score(y_true_arr, preds)),
        "precision_micro": float(precision_score(y_true_arr, preds, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true_arr, preds, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true_arr, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_arr, preds, average="macro", zero_division=0)),
        "thresholds": threshold_arr.tolist(),
    }
    if label_names is not None:
        result["per_label"] = {
            label_name: {
                "precision": float(precision_score(y_true_arr[:, index], preds[:, index], zero_division=0)),
                "recall": float(recall_score(y_true_arr[:, index], preds[:, index], zero_division=0)),
                "f1": float(f1_score(y_true_arr[:, index], preds[:, index], zero_division=0)),
            }
            for index, label_name in enumerate(label_names)
        }
    return result


def save_structured_results(results: Mapping[str, object], output_dir: str | Path, prefix: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{prefix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat_rows = []
    per_language = results.get("per_language")
    if isinstance(per_language, dict):
        for language, payload in per_language.items():
            row = {"language": language}
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        row[key] = value
            flat_rows.append(row)

    if flat_rows:
        fieldnames = sorted({key for row in flat_rows for key in row.keys()})
        with (output_path / f"{prefix}_per_language.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def save_summary_plot(results: Mapping[str, object], output_dir: str | Path, prefix: str) -> Path | None:
    per_language = results.get("per_language")
    if not isinstance(per_language, dict) or not per_language:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = list(per_language.keys())
    scores = [per_language[label].get("f1", per_language[label].get("macro_f1", 0.0)) for label in labels]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(labels, scores)
    axis.set_ylabel("F1")
    axis.set_title(f"{prefix} per-language F1")
    axis.tick_params(axis="x", rotation=30)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = output_path / f"{prefix}_summary.png"
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    return image_path
