"""Backbone registry for multilingual and Indic-ready experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "backbones"


@dataclass(frozen=True)
class BackboneSpec:
    """Metadata for a supported backbone."""

    name: str
    model_name: str
    architecture: str
    pooling: str = "mean"
    max_length: int = 256
    embedding_normalize: bool = False
    classifier_dropout: float = 0.1
    trust_remote_code: bool = False
    language_focus: str = "multilingual"
    description: str = ""
    input_prefix: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict) -> "BackboneSpec":
        return cls(
            name=payload["name"],
            model_name=payload.get("model_name", payload["name"]),
            architecture=payload.get("architecture", "encoder"),
            pooling=payload.get("pooling", "mean"),
            max_length=int(payload.get("max_length", 256)),
            embedding_normalize=bool(payload.get("embedding_normalize", False)),
            classifier_dropout=float(payload.get("classifier_dropout", 0.1)),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            language_focus=payload.get("language_focus", "multilingual"),
            description=payload.get("description", ""),
            input_prefix=payload.get("input_prefix", ""),
            aliases=tuple(payload.get("aliases", [])),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "embedding_normalize": self.embedding_normalize,
            "classifier_dropout": self.classifier_dropout,
            "trust_remote_code": self.trust_remote_code,
            "language_focus": self.language_focus,
            "description": self.description,
            "input_prefix": self.input_prefix,
            "aliases": list(self.aliases),
        }


def _load_registry() -> Dict[str, BackboneSpec]:
    registry: Dict[str, BackboneSpec] = {}
    if not CONFIG_DIR.exists():
        return registry

    for path in sorted(CONFIG_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = BackboneSpec.from_dict(payload)
        registry[spec.name] = spec
        for alias in spec.aliases:
            registry[alias] = spec
    return registry


_REGISTRY = _load_registry()


def list_backbones() -> list[str]:
    """Return canonical registered backbone names."""
    return sorted({spec.name for spec in _REGISTRY.values()})


def iter_backbone_specs() -> Iterable[BackboneSpec]:
    seen = set()
    for spec in _REGISTRY.values():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        yield spec


def get_backbone_spec(name: Optional[str]) -> BackboneSpec:
    """Look up a backbone spec by canonical name, alias, or local checkpoint path."""
    if not name:
        raise ValueError("A backbone name or local model path is required")

    if name in _REGISTRY:
        return _REGISTRY[name]

    path = Path(name)
    if path.exists():
        return BackboneSpec(
            name=path.name,
            model_name=str(path),
            architecture="encoder",
            pooling="mean",
            max_length=256,
            language_focus="local",
            description="Local backbone checkpoint",
        )

    available = ", ".join(list_backbones())
    raise KeyError(f"Unknown backbone {name!r}. Available: {available}")


def save_backbone_config(spec: BackboneSpec, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
