"""Active learning helpers for multilingual annotation loops."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def uncertainty_sample(
    dataframe: pd.DataFrame,
    probabilities: Sequence[float],
    *,
    budget: int = 32,
) -> pd.DataFrame:
    scores = np.abs(np.asarray(probabilities) - 0.5)
    order = np.argsort(scores)
    return dataframe.iloc[order[:budget]].copy().reset_index(drop=True)


def diversity_sample(
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    budget: int = 32,
) -> pd.DataFrame:
    if len(dataframe) <= budget:
        return dataframe.copy().reset_index(drop=True)
    chosen = [0]
    for _ in range(1, budget):
        distances = np.min(
            [np.linalg.norm(embeddings - embeddings[index], axis=1) for index in chosen],
            axis=0,
        )
        next_index = int(np.argmax(distances))
        if next_index in chosen:
            break
        chosen.append(next_index)
    return dataframe.iloc[chosen].copy().reset_index(drop=True)
