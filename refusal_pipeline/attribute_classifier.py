"""
Technique and vulnerability inference helpers for the refusal pipeline.

These classifiers are optional downstream enrichers: the binary detector decides
whether a dialogue is manipulative, then these models can identify which
manipulation techniques and victim vulnerabilities are present.
"""

import json
import os
from dataclasses import dataclass

import torch

from multilingual_support.inference import load_text_model, predict_text


TECHNIQUE_LABELS = [
    "Denial",
    "Evasion",
    "Feigning Innocence",
    "Rationalization",
    "Playing Victim Role",
    "Playing Servant Role",
    "Shaming or Belittlement",
    "Intimidation",
    "Brandishing Anger",
    "Accusation",
    "Persuasion or Seduction",
]

VULNERABILITY_LABELS = [
    "Naivete",
    "Dependency",
    "Over-responsibility",
    "Over-intellectualization",
    "Low self-esteem",
]

LABELS_BY_TASK = {
    "technique": TECHNIQUE_LABELS,
    "vulnerability": VULNERABILITY_LABELS,
}

DEFAULT_ATTRIBUTE_DIRS = {
    "technique": "./saved_model/technique",
    "vulnerability": "./saved_model/vulnerability",
}


def normalize_label(label):
    return " ".join(label.lower().replace("the ", "").split())


def labels_to_multihot(label_text, task):
    """Convert a comma-separated label string from MentalManip into a vector."""
    labels = LABELS_BY_TASK[task]
    label_to_index = {normalize_label(label): i for i, label in enumerate(labels)}
    vector = [0.0] * len(labels)

    for raw_label in label_text.split(","):
        key = normalize_label(raw_label.strip())
        if not key:
            continue
        if key not in label_to_index:
            raise ValueError(f"Unknown {task} label: {raw_label!r}")
        vector[label_to_index[key]] = 1.0

    return vector


@dataclass
class AttributePrediction:
    task: str
    labels: list
    scores: dict


class AttributeClassifier:
    def __init__(self, model_path, task, device=None, threshold=None, max_length=None):
        if task not in LABELS_BY_TASK:
            raise ValueError(f"task must be one of {sorted(LABELS_BY_TASK)}")

        self.model_path = model_path
        self.task = task
        self.labels = LABELS_BY_TASK[task]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = self._load_config(model_path)
        self.threshold = threshold if threshold is not None else config.get("threshold", 0.5)
        self.max_length = max_length or config.get("max_length", 128)

        self.model, self.tokenizer, self.runtime_config, self.runtime_mode = load_text_model(
            model_path,
            device=self.device,
        )

    @staticmethod
    def _load_config(model_path):
        config_path = os.path.join(model_path, "attribute_config.json")
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict(self, text):
        prediction = predict_text(
            self.model,
            self.tokenizer,
            text,
            self.device,
            max_length=self.max_length,
            threshold=self.threshold,
            task_config=self.runtime_config,
            mode=self.runtime_mode,
        )
        probs = prediction["probabilities"]

        selected = [
            label for label, prob in zip(self.labels, probs)
            if prob >= self.threshold
        ]
        scores = {
            label: round(float(prob), 4)
            for label, prob in zip(self.labels, probs)
        }

        return AttributePrediction(task=self.task, labels=selected, scores=scores)


def load_available_attribute_classifiers(base_dirs=None, device=None):
    """Load any trained attribute classifiers that exist locally."""
    base_dirs = base_dirs or DEFAULT_ATTRIBUTE_DIRS
    classifiers = {}
    for task, model_path in base_dirs.items():
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            classifiers[task] = AttributeClassifier(model_path, task, device=device)
    return classifiers


def predict_available_attributes(text, classifiers):
    return {
        task: classifier.predict(text)
        for task, classifier in classifiers.items()
    }
