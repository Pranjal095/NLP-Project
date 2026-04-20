"""Zero-shot multilingual NLI baseline utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dataset_utils import enrich_language_metadata, load_dataframe
from .metrics import compute_binary_metrics, save_structured_results, save_summary_plot
from .preprocessing import PreprocessingConfig, preprocess_text


DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def _infer_nli_label_indices(model) -> tuple[int, int]:
    label2id = {key.lower(): value for key, value in model.config.label2id.items()}
    entailment = label2id.get("entailment")
    contradiction = label2id.get("contradiction")
    if entailment is None or contradiction is None:
        raise ValueError("NLI checkpoint must expose entailment and contradiction labels")
    return entailment, contradiction


def score_zero_shot_nli(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    *,
    device: torch.device,
) -> float:
    encoded = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    entailment_idx, contradiction_idx = _infer_nli_label_indices(model)
    pair_logits = logits[:, [contradiction_idx, entailment_idx]]
    probs = torch.softmax(pair_logits, dim=-1)[0, 1].item()
    return probs


def evaluate_zero_shot_nli(
    data_path: str,
    *,
    model_name: str = DEFAULT_NLI_MODEL,
    hypothesis: str = "This dialogue is manipulative.",
    threshold: float = 0.5,
    output_dir: str = "./saved_model/zero_shot_nli",
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    dataframe = enrich_language_metadata(load_dataframe(data_path), config=preprocessing_config)
    probabilities = []
    for text in dataframe["Dialogue"].astype(str):
        processed = preprocess_text(text, preprocessing_config).processed_text
        probabilities.append(score_zero_shot_nli(model, tokenizer, processed, hypothesis, device=device))

    metrics = compute_binary_metrics(
        dataframe["Manipulative"].astype(int).tolist(),
        probabilities,
        threshold=threshold,
        language_labels=dataframe["language"].tolist(),
    )
    metrics["model_name"] = model_name
    save_structured_results(metrics, Path(output_dir) / "reports", "zero_shot_nli")
    save_summary_plot(metrics, Path(output_dir) / "reports", "zero_shot_nli")
    (Path(output_dir) / "config.json").write_text(
        json.dumps({"model_name": model_name, "hypothesis": hypothesis, "threshold": threshold}, indent=2),
        encoding="utf-8",
    )
    return metrics


def load_indic_eval_dataframe(data_path: str, *, preprocessing_config: Optional[PreprocessingConfig] = None) -> pd.DataFrame:
    return enrich_language_metadata(load_dataframe(data_path), config=preprocessing_config)
