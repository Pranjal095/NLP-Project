"""Inference helpers shared by demo/evaluation and multilingual experiments."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .modeling import is_multilingual_checkpoint, load_multilingual_checkpoint
from .preprocessing import PreprocessingConfig, preprocess_text


def load_text_model(model_path: str, device: Optional[torch.device] = None):
    if is_multilingual_checkpoint(model_path):
        model, tokenizer, config = load_multilingual_checkpoint(model_path, device=device)
        return model, tokenizer, config.to_dict(), "multilingual"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if device is not None:
        model = model.to(device)
    config_path = os.path.join(model_path, "pipeline_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    model.eval()
    return model, tokenizer, config, "legacy"


def _extract_probabilities(logits: torch.Tensor, mode: str, task_config: dict) -> torch.Tensor:
    temperature = float(task_config.get("temperature", 1.0) or 1.0)
    logits = logits / max(temperature, 1e-3)
    if mode == "multilingual" and task_config.get("problem_type") == "multi_label_classification":
        return torch.sigmoid(logits)
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def predict_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_length: int = 256,
    threshold: float = 0.5,
    task_config: Optional[dict] = None,
    mode: str = "legacy",
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> dict:
    preprocessed = preprocess_text(text, preprocessing_config)
    enc = tokenizer(
        preprocessed.processed_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        output = model(**enc)
        logits = output["logits"] if isinstance(output, dict) else output.logits
        probs = _extract_probabilities(logits, mode, task_config or {})

    if probs.shape[-1] == 1:
        positive_prob = probs[0, 0].item()
        pred_label = int(positive_prob >= threshold)
        probabilities = [1.0 - positive_prob, positive_prob]
    elif probs.shape[-1] == 2:
        positive_prob = probs[0, 1].item()
        pred_label = int(positive_prob >= threshold)
        probabilities = probs[0].detach().cpu().tolist()
    else:
        values = probs[0].detach().cpu().tolist()
        thresholds = task_config.get("per_label_thresholds") or [threshold] * len(values)
        pred_label = [int(value >= thresholds[idx]) for idx, value in enumerate(values)]
        positive_prob = max(values) if values else 0.0
        probabilities = values

    return {
        "processed_text": preprocessed.processed_text,
        "metadata": {
            "language": preprocessed.metadata.detected_language,
            "script": preprocessed.metadata.dominant_script,
            "is_code_mixed": preprocessed.metadata.is_code_mixed,
            "is_romanized_indic": preprocessed.metadata.is_romanized_indic,
        },
        "probability": positive_prob,
        "probabilities": probabilities,
        "prediction": pred_label,
    }
