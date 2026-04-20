"""Unified backbone/encoder abstraction for multilingual classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .model_registry import BackboneSpec, get_backbone_spec

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


MULTILINGUAL_METADATA_FILE = "multilingual_model_config.json"
MULTILINGUAL_STATE_FILE = "multilingual_model.bin"


@dataclass
class MultilingualModelConfig:
    backbone_name: str
    task_name: str
    problem_type: str
    num_labels: int
    label_names: list[str] = field(default_factory=list)
    max_length: int = 256
    pooling: str = "mean"
    classifier_dropout: float = 0.1
    training_mode: str = "full"
    threshold: float = 0.5
    per_label_thresholds: list[float] = field(default_factory=list)
    preprocessing: dict = field(default_factory=dict)
    class_weights: list[float] = field(default_factory=list)
    pos_weights: list[float] = field(default_factory=list)
    temperature: float = 1.0
    architecture: str = "encoder"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "MultilingualModelConfig":
        return cls(**payload)


class UnifiedTextEncoder(nn.Module):
    """Backbone abstraction shared by classifier and embedding tasks."""

    def __init__(self, spec: BackboneSpec, *, training_mode: str = "full") -> None:
        super().__init__()
        self.spec = spec
        self.training_mode = training_mode
        self.backbone = AutoModel.from_pretrained(
            spec.model_name,
            trust_remote_code=spec.trust_remote_code,
        )
        self.hidden_size = self._infer_hidden_size()
        self._apply_training_mode(training_mode)

    def _infer_hidden_size(self) -> int:
        for attribute in ("hidden_size", "d_model", "dim"):
            value = getattr(self.backbone.config, attribute, None)
            if value:
                return int(value)
        raise ValueError(f"Could not infer hidden size for {self.spec.model_name}")

    def _apply_training_mode(self, training_mode: str) -> None:
        if training_mode == "full":
            return
        if training_mode == "frozen":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            return
        if training_mode == "lora":
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT/LoRA requested but peft is not installed")
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
            self.backbone = get_peft_model(self.backbone, config)
            return
        raise ValueError(f"Unsupported training mode: {training_mode}")

    def _masked_mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        numerator = torch.sum(hidden * mask, dim=1)
        denominator = mask.sum(dim=1).clamp(min=1e-6)
        return numerator / denominator

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if getattr(self.backbone.config, "is_encoder_decoder", False):
            outputs = self.backbone.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            pooled = self._masked_mean_pool(hidden, attention_mask)
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            if self.spec.pooling == "cls" or self.spec.pooling == "cls_pooler":
                pooled = hidden[:, 0]
            else:
                pooled = self._masked_mean_pool(hidden, attention_mask)
        if self.spec.embedding_normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled


class UnifiedTextClassifier(nn.Module):
    """Custom classifier head on top of a shared encoder abstraction."""

    def __init__(
        self,
        config: MultilingualModelConfig,
        *,
        class_weights: Optional[torch.Tensor] = None,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        spec = get_backbone_spec(config.backbone_name)
        self.model_config = config
        self.encoder = UnifiedTextEncoder(spec, training_mode=config.training_mode)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(self.encoder.hidden_size, config.num_labels)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.float())
        else:
            self.pos_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: dict,
    ) -> dict:
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            if self.model_config.problem_type == "single_label_classification":
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fn(logits, labels.long())
            elif self.model_config.problem_type == "multi_label_classification":
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
                loss = loss_fn(logits, labels.float())
            else:
                raise ValueError(f"Unsupported problem type: {self.model_config.problem_type}")
        return {"loss": loss, "logits": logits}


def build_tokenizer(backbone_name: str):
    spec = get_backbone_spec(backbone_name)
    return AutoTokenizer.from_pretrained(spec.model_name, trust_remote_code=spec.trust_remote_code)


def save_multilingual_checkpoint(
    model: UnifiedTextClassifier,
    tokenizer,
    output_dir: str | Path,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target_dir / MULTILINGUAL_STATE_FILE)
    tokenizer.save_pretrained(target_dir)
    (target_dir / MULTILINGUAL_METADATA_FILE).write_text(
        json.dumps(model.model_config.to_dict(), indent=2),
        encoding="utf-8",
    )
    return target_dir


def is_multilingual_checkpoint(model_path: str | Path) -> bool:
    path = Path(model_path)
    return (path / MULTILINGUAL_METADATA_FILE).exists() and (path / MULTILINGUAL_STATE_FILE).exists()


def load_multilingual_checkpoint(
    model_path: str | Path,
    *,
    device: Optional[torch.device] = None,
) -> tuple[UnifiedTextClassifier, object, MultilingualModelConfig]:
    path = Path(model_path)
    config = MultilingualModelConfig.from_dict(
        json.loads((path / MULTILINGUAL_METADATA_FILE).read_text(encoding="utf-8"))
    )
    model = UnifiedTextClassifier(config)
    state = torch.load(path / MULTILINGUAL_STATE_FILE, map_location="cpu")
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer, config
