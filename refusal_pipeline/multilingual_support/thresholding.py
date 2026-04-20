"""Threshold tuning and calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score


@dataclass
class ThresholdSelectionResult:
    threshold: float
    metric_name: str
    metric_value: float


@dataclass
class CalibrationSummary:
    temperature: float
    ece: float
    brier: float
    nll: float


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (y_prob_arr >= lower) & (y_prob_arr < upper if upper < 1.0 else y_prob_arr <= upper)
        if not np.any(mask):
            continue
        confidence = y_prob_arr[mask].mean()
        accuracy = y_true_arr[mask].mean()
        ece += np.abs(confidence - accuracy) * mask.mean()
    return float(ece)


def fit_temperature_scaler(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    max_steps: int = 200,
    learning_rate: float = 0.05,
) -> CalibrationSummary:
    logits_tensor = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_tensor = torch.tensor(np.asarray(labels), dtype=torch.long)
    temperature = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.LBFGS([temperature], lr=learning_rate, max_iter=max_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits_tensor / temperature.clamp(min=1e-2), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    scaled_probs = torch.softmax(logits_tensor / temperature.clamp(min=1e-2), dim=-1)[:, 1].detach().cpu().numpy()
    labels_np = labels_tensor.detach().cpu().numpy()
    return CalibrationSummary(
        temperature=float(temperature.detach().cpu().item()),
        ece=expected_calibration_error(labels_np, scaled_probs),
        brier=float(np.mean((scaled_probs - labels_np) ** 2)),
        nll=float(log_loss(labels_np, np.column_stack([1.0 - scaled_probs, scaled_probs]), labels=[0, 1])),
    )


def tune_binary_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    candidate_thresholds: Iterable[float] | None = None,
    metric_name: str = "macro_f1",
) -> ThresholdSelectionResult:
    thresholds = list(candidate_thresholds or np.linspace(0.1, 0.9, 17))
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    best = ThresholdSelectionResult(threshold=0.5, metric_name=metric_name, metric_value=-1.0)

    for threshold in thresholds:
        preds = (y_prob_arr >= threshold).astype(int)
        if metric_name == "macro_f1":
            score = f1_score(y_true_arr, preds, average="macro", zero_division=0)
        elif metric_name == "binary_f1":
            score = f1_score(y_true_arr, preds, average="binary", zero_division=0)
        elif metric_name == "auprc":
            score = average_precision_score(y_true_arr, y_prob_arr)
        elif metric_name == "auroc":
            score = roc_auc_score(y_true_arr, y_prob_arr)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        if score > best.metric_value:
            best = ThresholdSelectionResult(float(threshold), metric_name, float(score))
    return best


def tune_multilabel_thresholds(
    y_true: Sequence[Sequence[int]],
    y_prob: Sequence[Sequence[float]],
    *,
    candidate_thresholds: Iterable[float] | None = None,
) -> list[ThresholdSelectionResult]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    results = []
    for column in range(y_true_arr.shape[1]):
        results.append(
            tune_binary_threshold(
                y_true_arr[:, column],
                y_prob_arr[:, column],
                candidate_thresholds=candidate_thresholds,
                metric_name="binary_f1",
            )
        )
    return results
