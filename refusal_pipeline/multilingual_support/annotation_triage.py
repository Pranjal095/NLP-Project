"""Annotation triage helpers for multilingual data curation."""

from __future__ import annotations

import pandas as pd


def add_triage_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    triaged = dataframe.copy()
    triaged["triage_needs_language_review"] = triaged.get("language", "unknown").isin(["unknown", "indic-romanized"])
    triaged["triage_code_mixed"] = triaged.get("is_code_mixed", False).astype(bool)
    triaged["triage_romanized"] = triaged.get("is_romanized_indic", False).astype(bool)
    return triaged


def prioritize_for_review(dataframe: pd.DataFrame) -> pd.DataFrame:
    triaged = add_triage_flags(dataframe)
    triaged["triage_priority"] = (
        triaged["triage_needs_language_review"].astype(int)
        + triaged["triage_code_mixed"].astype(int)
        + triaged["triage_romanized"].astype(int)
    )
    return triaged.sort_values("triage_priority", ascending=False).reset_index(drop=True)
