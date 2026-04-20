"""Dataset loading, metadata enrichment, and split helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from attribute_classifier import LABELS_BY_TASK, labels_to_multihot
from .preprocessing import PreprocessingConfig, preprocess_text


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    rows = []
    columns = None
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for index, row in enumerate(reader):
            if index == 0:
                columns = row
            else:
                rows.append(row)
    if columns is None:
        raise ValueError(f"No columns found in {csv_path}")
    return pd.DataFrame(rows, columns=columns)


def enrich_language_metadata(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "Dialogue",
    config: Optional[PreprocessingConfig] = None,
) -> pd.DataFrame:
    enriched = dataframe.copy()
    processed_texts = []
    languages = []
    scripts = []
    code_mixed = []
    romanized = []
    for text in enriched[text_column].fillna("").astype(str):
        result = preprocess_text(text, config)
        processed_texts.append(result.processed_text)
        languages.append(result.metadata.detected_language)
        scripts.append(result.metadata.dominant_script)
        code_mixed.append(result.metadata.is_code_mixed)
        romanized.append(result.metadata.is_romanized_indic)
    enriched["ProcessedDialogue"] = processed_texts
    if "language" not in enriched.columns:
        enriched["language"] = languages
    if "script" not in enriched.columns:
        enriched["script"] = scripts
    enriched["is_code_mixed"] = code_mixed
    enriched["is_romanized_indic"] = romanized
    return enriched


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    label_column: str,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    test_ratio = 1.0 - train_ratio - valid_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + valid_ratio must be less than 1.0")

    stratify = dataframe[label_column] if dataframe[label_column].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        dataframe,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    valid_fraction = valid_ratio / (valid_ratio + test_ratio)
    temp_stratify = temp_df[label_column] if temp_df[label_column].value_counts().min() >= 2 else None
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_fraction,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )
    return DatasetBundle(
        train_df=train_df.reset_index(drop=True),
        valid_df=valid_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def build_multilabel_frame(dataframe: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in LABELS_BY_TASK:
        raise ValueError(f"Unknown task: {task}")
    target_column = "Technique" if task == "technique" else "Vulnerability"
    filtered = dataframe[dataframe[target_column].fillna("") != ""].copy()
    filtered["labels"] = filtered[target_column].apply(lambda value: labels_to_multihot(str(value), task))
    filtered["primary_label"] = filtered[target_column].apply(lambda value: str(value).split(",")[0].strip())
    return filtered.reset_index(drop=True)
