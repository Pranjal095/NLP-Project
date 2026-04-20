"""Reusable multilingual/Indic support utilities for the refusal pipeline.

These modules are intentionally isolated from the deterministic refusal path.
They support multilingual model selection, preprocessing, training, evaluation,
retrieval, and annotation workflows while keeping refusal generation rule-based.
"""

from .model_registry import BackboneSpec, get_backbone_spec, list_backbones
from .preprocessing import PreprocessingConfig, PreprocessingResult, preprocess_text
from .thresholding import (
    CalibrationSummary,
    ThresholdSelectionResult,
    fit_temperature_scaler,
    tune_binary_threshold,
    tune_multilabel_thresholds,
)
from .metrics import compute_binary_metrics, compute_multilabel_metrics
from .inference import load_text_model, predict_text

__all__ = [
    "BackboneSpec",
    "CalibrationSummary",
    "PreprocessingConfig",
    "PreprocessingResult",
    "ThresholdSelectionResult",
    "compute_binary_metrics",
    "compute_multilabel_metrics",
    "fit_temperature_scaler",
    "get_backbone_spec",
    "list_backbones",
    "load_text_model",
    "predict_text",
    "preprocess_text",
    "tune_binary_threshold",
    "tune_multilabel_thresholds",
]
