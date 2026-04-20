"""Training helpers for multilingual and Indic-ready experiments."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments

from attribute_classifier import LABELS_BY_TASK
from .dataset_utils import build_multilabel_frame, enrich_language_metadata, load_dataframe, split_dataframe
from .metrics import compute_binary_metrics, compute_multilabel_metrics, save_structured_results, save_summary_plot
from .model_registry import get_backbone_spec
from .modeling import MultilingualModelConfig, UnifiedTextClassifier, build_tokenizer, save_multilingual_checkpoint
from .preprocessing import PreprocessingConfig
from .thresholding import fit_temperature_scaler, tune_binary_threshold, tune_multilabel_thresholds


@dataclass
class TrainingRunConfig:
    backbone_name: str = "roberta-base"
    task_name: str = "detection"
    output_dir: str = "./saved_model/multilingual"
    training_mode: str = "full"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-5
    max_length: int = 256
    seed: int = 42
    threshold: float = 0.5
    train_ratio: float = 0.6
    valid_ratio: float = 0.2


def _make_training_args(**kwargs):
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value
    return TrainingArguments(**kwargs)


def _build_dataset(dataframe, tokenizer, max_length: int, *, label_key: str = "label") -> Dataset:
    dataset = Dataset.from_dict(
        {
            "text": dataframe["ProcessedDialogue"].tolist(),
            "labels": dataframe[label_key].tolist(),
        }
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=64)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def _binary_compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def _multilabel_compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)
    return {
        "subset_accuracy": accuracy_score(labels, preds),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def _binary_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2)
    total = counts.sum()
    weights = total / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _multilabel_pos_weights(labels: np.ndarray) -> torch.Tensor:
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weights = negatives / np.maximum(positives, 1)
    return torch.tensor(weights, dtype=torch.float32)


def _train_model(
    *,
    model: UnifiedTextClassifier,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    run_config: TrainingRunConfig,
    compute_metrics: Callable,
    metric_for_best_model: str,
) -> Trainer:
    training_args = _make_training_args(
        output_dir=run_config.output_dir,
        num_train_epochs=run_config.epochs,
        per_device_train_batch_size=run_config.batch_size,
        per_device_eval_batch_size=run_config.batch_size,
        learning_rate=run_config.lr,
        weight_decay=0.01,
        warmup_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=run_config.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def train_binary_detector(
    data_path: str,
    run_config: TrainingRunConfig,
    *,
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> dict:
    dataframe = enrich_language_metadata(load_dataframe(data_path), config=preprocessing_config)
    dataframe["label"] = dataframe["Manipulative"].astype(int)
    splits = split_dataframe(
        dataframe,
        label_column="label",
        train_ratio=run_config.train_ratio,
        valid_ratio=run_config.valid_ratio,
        seed=run_config.seed,
    )

    spec = get_backbone_spec(run_config.backbone_name)
    tokenizer = build_tokenizer(spec.name)
    class_weights = _binary_class_weights(splits.train_df["label"].to_numpy())
    model_config = MultilingualModelConfig(
        backbone_name=spec.name,
        task_name=run_config.task_name,
        problem_type="single_label_classification",
        num_labels=2,
        max_length=run_config.max_length or spec.max_length,
        pooling=spec.pooling,
        classifier_dropout=spec.classifier_dropout,
        training_mode=run_config.training_mode,
        threshold=run_config.threshold,
        preprocessing=(preprocessing_config.__dict__ if preprocessing_config else {}),
        class_weights=class_weights.tolist(),
        architecture=spec.architecture,
    )
    model = UnifiedTextClassifier(model_config, class_weights=class_weights)

    train_dataset = _build_dataset(splits.train_df, tokenizer, model_config.max_length)
    valid_dataset = _build_dataset(splits.valid_df, tokenizer, model_config.max_length)
    test_dataset = _build_dataset(splits.test_df, tokenizer, model_config.max_length)

    trainer = _train_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        run_config=run_config,
        compute_metrics=_binary_compute_metrics,
        metric_for_best_model="f1",
    )

    valid_predictions = trainer.predict(valid_dataset)
    valid_probs = torch.softmax(torch.tensor(valid_predictions.predictions), dim=-1).numpy()[:, 1]
    threshold_result = tune_binary_threshold(splits.valid_df["label"].to_numpy(), valid_probs)
    calibration = fit_temperature_scaler(valid_predictions.predictions, splits.valid_df["label"].to_numpy())
    model.model_config.threshold = threshold_result.threshold
    model.model_config.temperature = calibration.temperature

    test_predictions = trainer.predict(test_dataset)
    test_probs = torch.softmax(torch.tensor(test_predictions.predictions) / calibration.temperature, dim=-1).numpy()[:, 1]
    metrics = compute_binary_metrics(
        splits.test_df["label"].to_numpy(),
        test_probs,
        threshold=threshold_result.threshold,
        language_labels=splits.test_df["language"].tolist(),
    )
    metrics["threshold_tuning"] = threshold_result.__dict__
    metrics["calibration"] = calibration.__dict__

    save_multilingual_checkpoint(model, tokenizer, Path(run_config.output_dir) / "final")
    save_structured_results(metrics, Path(run_config.output_dir) / "reports", "binary_eval")
    save_summary_plot(metrics, Path(run_config.output_dir) / "reports", "binary_eval")
    return metrics


def train_multilabel_classifier(
    data_path: str,
    run_config: TrainingRunConfig,
    *,
    task: str,
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> dict:
    dataframe = enrich_language_metadata(load_dataframe(data_path), config=preprocessing_config)
    multilabel_frame = build_multilabel_frame(dataframe, task)
    multilabel_frame["label"] = multilabel_frame["labels"]
    splits = split_dataframe(
        multilabel_frame,
        label_column="primary_label",
        train_ratio=run_config.train_ratio,
        valid_ratio=run_config.valid_ratio,
        seed=run_config.seed,
    )

    spec = get_backbone_spec(run_config.backbone_name)
    tokenizer = build_tokenizer(spec.name)
    label_names = LABELS_BY_TASK[task]
    pos_weights = _multilabel_pos_weights(np.asarray(splits.train_df["label"].tolist(), dtype=np.float32))
    model_config = MultilingualModelConfig(
        backbone_name=spec.name,
        task_name=task,
        problem_type="multi_label_classification",
        num_labels=len(label_names),
        label_names=list(label_names),
        max_length=run_config.max_length or spec.max_length,
        pooling=spec.pooling,
        classifier_dropout=spec.classifier_dropout,
        training_mode=run_config.training_mode,
        threshold=run_config.threshold,
        preprocessing=(preprocessing_config.__dict__ if preprocessing_config else {}),
        pos_weights=pos_weights.tolist(),
        architecture=spec.architecture,
    )
    model = UnifiedTextClassifier(model_config, pos_weights=pos_weights)

    train_dataset = _build_dataset(splits.train_df, tokenizer, model_config.max_length)
    valid_dataset = _build_dataset(splits.valid_df, tokenizer, model_config.max_length)
    test_dataset = _build_dataset(splits.test_df, tokenizer, model_config.max_length)

    trainer = _train_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        run_config=run_config,
        compute_metrics=_multilabel_compute_metrics,
        metric_for_best_model="f1_micro",
    )
    valid_predictions = trainer.predict(valid_dataset)
    valid_probs = torch.sigmoid(torch.tensor(valid_predictions.predictions)).numpy()
    threshold_results = tune_multilabel_thresholds(np.asarray(splits.valid_df["label"].tolist()), valid_probs)
    per_label_thresholds = [item.threshold for item in threshold_results]
    model.model_config.per_label_thresholds = per_label_thresholds

    test_predictions = trainer.predict(test_dataset)
    test_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).numpy()
    metrics = compute_multilabel_metrics(
        np.asarray(splits.test_df["label"].tolist()),
        test_probs,
        thresholds=per_label_thresholds,
        label_names=label_names,
    )
    metrics["threshold_tuning"] = [result.__dict__ for result in threshold_results]
    save_multilingual_checkpoint(model, tokenizer, Path(run_config.output_dir) / task)
    save_structured_results(metrics, Path(run_config.output_dir) / "reports", f"{task}_eval")
    return metrics


def save_run_manifest(output_dir: str | Path, payload: dict) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
