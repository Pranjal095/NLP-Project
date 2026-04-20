"""Experimental multi-task training with a shared multilingual encoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from attribute_classifier import LABELS_BY_TASK, labels_to_multihot
from .dataset_utils import enrich_language_metadata, load_dataframe, split_dataframe
from .metrics import compute_binary_metrics, compute_multilabel_metrics, save_structured_results
from .model_registry import get_backbone_spec
from .modeling import UnifiedTextEncoder, build_tokenizer
from .preprocessing import PreprocessingConfig
from .thresholding import tune_binary_threshold, tune_multilabel_thresholds


@dataclass
class MultitaskRunConfig:
    backbone_name: str = "xlm-roberta-base"
    output_dir: str = "./saved_model/multitask"
    training_mode: str = "full"
    epochs: int = 2
    batch_size: int = 4
    lr: float = 2e-5
    max_length: int = 256
    seed: int = 42


class MultitaskMentalManipDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.frame = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["ProcessedDialogue"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "detection_labels": torch.tensor(int(row["Manipulative"]), dtype=torch.long),
            "technique_labels": torch.tensor(row["technique_labels"], dtype=torch.float32),
            "vulnerability_labels": torch.tensor(row["vulnerability_labels"], dtype=torch.float32),
            "technique_mask": torch.tensor(float(row["Technique"] != ""), dtype=torch.float32),
            "vulnerability_mask": torch.tensor(float(row["Vulnerability"] != ""), dtype=torch.float32),
        }


class MultitaskClassifier(nn.Module):
    def __init__(self, backbone_name: str, *, training_mode: str = "full") -> None:
        super().__init__()
        spec = get_backbone_spec(backbone_name)
        self.encoder = UnifiedTextEncoder(spec, training_mode=training_mode)
        hidden_size = self.encoder.hidden_size
        self.dropout = nn.Dropout(spec.classifier_dropout)
        self.detection_head = nn.Linear(hidden_size, 2)
        self.technique_head = nn.Linear(hidden_size, len(LABELS_BY_TASK["technique"]))
        self.vulnerability_head = nn.Linear(hidden_size, len(LABELS_BY_TASK["vulnerability"]))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(pooled)
        return {
            "detection_logits": self.detection_head(pooled),
            "technique_logits": self.technique_head(pooled),
            "vulnerability_logits": self.vulnerability_head(pooled),
        }


def _prepare_frame(data_path: str, preprocessing_config: Optional[PreprocessingConfig]) -> pd.DataFrame:
    frame = enrich_language_metadata(load_dataframe(data_path), config=preprocessing_config)
    frame["technique_labels"] = frame["Technique"].apply(
        lambda value: labels_to_multihot(str(value), "technique") if str(value) else [0.0] * len(LABELS_BY_TASK["technique"])
    )
    frame["vulnerability_labels"] = frame["Vulnerability"].apply(
        lambda value: labels_to_multihot(str(value), "vulnerability") if str(value) else [0.0] * len(LABELS_BY_TASK["vulnerability"])
    )
    frame["split_label"] = frame["Manipulative"].astype(str)
    return frame


def _evaluate_multitask(model, data_loader, device):
    detection_labels = []
    detection_probs = []
    technique_true = []
    technique_probs = []
    vulnerability_true = []
    vulnerability_probs = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            detection_prob = torch.softmax(outputs["detection_logits"], dim=-1)[:, 1]
            detection_labels.extend(batch["detection_labels"].cpu().numpy().tolist())
            detection_probs.extend(detection_prob.cpu().numpy().tolist())
            technique_true.extend(batch["technique_labels"].cpu().numpy().tolist())
            technique_probs.extend(torch.sigmoid(outputs["technique_logits"]).cpu().numpy().tolist())
            vulnerability_true.extend(batch["vulnerability_labels"].cpu().numpy().tolist())
            vulnerability_probs.extend(torch.sigmoid(outputs["vulnerability_logits"]).cpu().numpy().tolist())
    return {
        "detection_labels": np.asarray(detection_labels),
        "detection_probs": np.asarray(detection_probs),
        "technique_true": np.asarray(technique_true),
        "technique_probs": np.asarray(technique_probs),
        "vulnerability_true": np.asarray(vulnerability_true),
        "vulnerability_probs": np.asarray(vulnerability_probs),
    }


def train_multitask_model(
    data_path: str,
    run_config: MultitaskRunConfig,
    *,
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> dict:
    torch.manual_seed(run_config.seed)
    frame = _prepare_frame(data_path, preprocessing_config)
    splits = split_dataframe(frame, label_column="split_label", train_ratio=0.6, valid_ratio=0.2, seed=run_config.seed)

    tokenizer = build_tokenizer(run_config.backbone_name)
    train_ds = MultitaskMentalManipDataset(splits.train_df, tokenizer, run_config.max_length)
    valid_ds = MultitaskMentalManipDataset(splits.valid_df, tokenizer, run_config.max_length)
    test_ds = MultitaskMentalManipDataset(splits.test_df, tokenizer, run_config.max_length)

    train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=run_config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=run_config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskClassifier(run_config.backbone_name, training_mode=run_config.training_mode).to(device)
    optimizer = AdamW(model.parameters(), lr=run_config.lr)
    detection_loss_fn = nn.CrossEntropyLoss()
    multilabel_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for _ in range(run_config.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            det_loss = detection_loss_fn(outputs["detection_logits"], batch["detection_labels"])
            tech_loss = multilabel_loss_fn(outputs["technique_logits"], batch["technique_labels"]).mean(dim=1)
            vul_loss = multilabel_loss_fn(outputs["vulnerability_logits"], batch["vulnerability_labels"]).mean(dim=1)
            tech_loss = (tech_loss * batch["technique_mask"]).mean()
            vul_loss = (vul_loss * batch["vulnerability_mask"]).mean()
            loss = det_loss + tech_loss + vul_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    valid_outputs = _evaluate_multitask(model, valid_loader, device)
    detection_threshold = tune_binary_threshold(valid_outputs["detection_labels"], valid_outputs["detection_probs"]).threshold
    technique_thresholds = [item.threshold for item in tune_multilabel_thresholds(valid_outputs["technique_true"], valid_outputs["technique_probs"])]
    vulnerability_thresholds = [item.threshold for item in tune_multilabel_thresholds(valid_outputs["vulnerability_true"], valid_outputs["vulnerability_probs"])]

    test_outputs = _evaluate_multitask(model, test_loader, device)
    results = {
        "detection": compute_binary_metrics(
            test_outputs["detection_labels"],
            test_outputs["detection_probs"],
            threshold=detection_threshold,
        ),
        "technique": compute_multilabel_metrics(
            test_outputs["technique_true"],
            test_outputs["technique_probs"],
            thresholds=technique_thresholds,
            label_names=LABELS_BY_TASK["technique"],
        ),
        "vulnerability": compute_multilabel_metrics(
            test_outputs["vulnerability_true"],
            test_outputs["vulnerability_probs"],
            thresholds=vulnerability_thresholds,
            label_names=LABELS_BY_TASK["vulnerability"],
        ),
    }

    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "multitask_model.bin")
    (output_dir / "multitask_config.json").write_text(json.dumps(run_config.__dict__, indent=2), encoding="utf-8")
    save_structured_results(results, output_dir / "reports", "multitask_eval")
    return results
