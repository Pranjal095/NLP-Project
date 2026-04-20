"""Synthetic text generation and augmentation utilities.

These helpers are intentionally isolated from the refusal decision path.
They can be used to build translated or paraphrased training data, but they
must never be used to generate actual refusal responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class GenerationConfig:
    model_name: str
    max_new_tokens: int = 96
    num_beams: int = 4
    temperature: float = 1.0
    device: str = "cpu"


def _load_generator(config: GenerationConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(config.device)
    model.eval()
    return model, tokenizer


def generate_paraphrases(texts: Iterable[str], config: GenerationConfig, *, prompt_prefix: str = "paraphrase: ") -> list[str]:
    model, tokenizer = _load_generator(config)
    outputs = []
    for text in texts:
        encoded = tokenizer(
            prompt_prefix + text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(config.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                temperature=config.temperature,
            )
        outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return outputs


def build_adversarial_augmentations(text: str) -> list[str]:
    return [
        text,
        text.replace("you", "u").replace("please", "plz"),
        text.replace("do not", "don't").replace("I am", "I'm"),
        text + " !!!",
    ]


def translate_texts(texts: Iterable[str], config: GenerationConfig, *, task_prefix: str) -> list[str]:
    return generate_paraphrases(texts, config, prompt_prefix=task_prefix)
