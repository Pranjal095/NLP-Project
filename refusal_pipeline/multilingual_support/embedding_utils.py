"""Embedding, retrieval, and weak-supervision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

from .model_registry import get_backbone_spec
from .modeling import UnifiedTextEncoder, build_tokenizer
from .preprocessing import PreprocessingConfig, preprocess_text


@dataclass
class EmbeddingBatchResult:
    embeddings: np.ndarray
    metadata: list[dict]


def encode_texts(
    texts: Sequence[str],
    *,
    backbone_name: str = "sentence-transformers/LaBSE",
    batch_size: int = 16,
    device: torch.device | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> EmbeddingBatchResult:
    spec = get_backbone_spec(backbone_name)
    tokenizer = build_tokenizer(spec.name)
    encoder = UnifiedTextEncoder(spec, training_mode="frozen")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()

    normalized = [preprocess_text(text, preprocessing_config) for text in texts]
    processed_texts = [
        (spec.input_prefix + item.processed_text).strip()
        if spec.input_prefix
        else item.processed_text
        for item in normalized
    ]
    batches = []
    for start in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=spec.max_length,
        ).to(device)
        with torch.no_grad():
            pooled = encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        batches.append(pooled.detach().cpu().numpy())
    embeddings = np.concatenate(batches, axis=0) if batches else np.empty((0, encoder.hidden_size))
    metadata = [
        {
            "language": item.metadata.detected_language,
            "script": item.metadata.dominant_script,
            "is_code_mixed": item.metadata.is_code_mixed,
            "is_romanized_indic": item.metadata.is_romanized_indic,
        }
        for item in normalized
    ]
    return EmbeddingBatchResult(embeddings=embeddings, metadata=metadata)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    normalized = embeddings / norms
    return normalized @ normalized.T


def deduplicate_by_similarity(dataframe: pd.DataFrame, embeddings: np.ndarray, *, threshold: float = 0.98) -> pd.DataFrame:
    similarity = cosine_similarity_matrix(embeddings)
    keep_indices = []
    removed = set()
    for index in range(len(dataframe)):
        if index in removed:
            continue
        keep_indices.append(index)
        near_duplicates = np.where(similarity[index] >= threshold)[0]
        removed.update(int(item) for item in near_duplicates if int(item) != index)
    return dataframe.iloc[keep_indices].reset_index(drop=True)


def mine_hard_negatives(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    label_column: str = "Manipulative",
    top_k: int = 3,
) -> pd.DataFrame:
    similarity = cosine_similarity_matrix(embeddings)
    labels = dataframe[label_column].astype(int).to_numpy()
    rows = []
    for anchor_index, anchor_label in enumerate(labels):
        candidate_indices = np.where(labels != anchor_label)[0]
        if candidate_indices.size == 0:
            continue
        ranked = candidate_indices[np.argsort(similarity[anchor_index, candidate_indices])[::-1][:top_k]]
        for candidate in ranked:
            rows.append(
                {
                    "anchor_index": int(anchor_index),
                    "candidate_index": int(candidate),
                    "similarity": float(similarity[anchor_index, candidate]),
                    "anchor_label": int(anchor_label),
                    "candidate_label": int(labels[candidate]),
                }
            )
    return pd.DataFrame(rows)


def cluster_examples(embeddings: np.ndarray, *, num_clusters: int = 8) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0,), dtype=int)
    estimator = MiniBatchKMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42, n_init="auto")
    return estimator.fit_predict(embeddings)


def pseudo_label_by_nearest_neighbors(
    labeled_embeddings: np.ndarray,
    labeled_targets: Sequence[int],
    unlabeled_embeddings: np.ndarray,
    *,
    k: int = 5,
) -> np.ndarray:
    labeled_targets_arr = np.asarray(labeled_targets, dtype=int)
    if unlabeled_embeddings.size == 0:
        return np.empty((0,), dtype=int)
    scores = unlabeled_embeddings @ labeled_embeddings.T
    pseudo = []
    for row in scores:
        nearest = np.argsort(row)[::-1][:k]
        votes = labeled_targets_arr[nearest]
        pseudo.append(int(votes.mean() >= 0.5))
    return np.asarray(pseudo, dtype=int)
