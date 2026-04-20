#!/usr/bin/env python3
"""
Translate the MentalManip dialogue column into an Indic language while
preserving every original annotation column exactly as-is.

This script uses Meta's local NLLB model through a Hugging Face translation
pipeline. It never calls an external API.

Default behavior:
  - Reads:  mentalmanip_dataset/mentalmanip_con.csv
  - Uses:   facebook/nllb-200-distilled-600M
  - Source: eng_Latn
  - Target: hin_Deva
  - Writes: mentalmanip_dataset/mentalmanip_con_hindi.csv

Examples:
  python3 translate_dataset.py
  python3 translate_dataset.py --target-language ben_Beng
  python3 translate_dataset.py --target-language tel_Telu
  python3 translate_dataset.py --batch-size 8 --model facebook/nllb-200-distilled-600M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


DEFAULT_INPUT = Path("mentalmanip_dataset/mentalmanip_con.csv")
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_SOURCE_LANGUAGE = "eng_Latn"
DEFAULT_TARGET_LANGUAGE = "hin_Deva"

# The user specifically asked for Hindi, Bengali, and Telugu support, and also
# mentioned Tamil as another easy swap option.
LANGUAGE_FILE_SUFFIXES: Dict[str, str] = {
    "hin_Deva": "hindi",
    "ben_Beng": "bengali",
    "tel_Telu": "telugu",
    "tam_Taml": "tamil",
}

# Common dataset variants in this repo or future derivatives.
TEXT_COLUMN_CANDIDATES = ("text", "Dialogue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate MentalManip dialogues into an Indic language using a local NLLB model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the source CSV file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output CSV path. If omitted, a language-specific filename is generated.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name or local path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--source-language",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"NLLB source language code. Default: {DEFAULT_SOURCE_LANGUAGE}",
    )
    parser.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANGUAGE,
        choices=sorted(LANGUAGE_FILE_SUFFIXES),
        help="NLLB target language code. Default: hin_Deva",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Optional explicit text column. If omitted, the script auto-detects from: text, Dialogue.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for translation. Lower this if you run out of GPU memory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens per translated dialogue.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam width for generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )
    return parser.parse_args()


def detect_device() -> tuple[int, str]:
    """
    Return both the transformers pipeline device id and a human-readable label.

    For Hugging Face pipelines:
      - GPU 0 is passed as device=0
      - CPU is passed as device=-1
    """
    if torch.cuda.is_available():
        return 0, "cuda:0"
    return -1, "cpu"


def resolve_text_column(df: pd.DataFrame, explicit_column: str | None) -> str:
    if explicit_column:
        if explicit_column not in df.columns:
            raise ValueError(
                f"Requested text column {explicit_column!r} was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit_column

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not auto-detect the dialogue column. "
        f"Tried {TEXT_COLUMN_CANDIDATES}, available columns are: {list(df.columns)}"
    )


def build_output_path(input_path: Path, target_language: str) -> Path:
    suffix = LANGUAGE_FILE_SUFFIXES.get(target_language, target_language.replace("_", "-").lower())
    return input_path.with_name(f"{input_path.stem}_{suffix}.csv")


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_translator(model_name: str, device_id: int):
    """
    Build a local translation pipeline.

    We load model + tokenizer explicitly so the behavior is clear and entirely local.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if device_id >= 0:
        model = model.to("cuda:0")

    return pipeline(
        task="translation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
    )


def translate_texts(
    translator,
    texts: List[str],
    *,
    source_language: str,
    target_language: str,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    translated: List[str] = []

    for batch in tqdm(
        list(batched(texts, batch_size)),
        desc=f"Translating to {target_language}",
        unit="batch",
    ):
        outputs = translator(
            batch,
            src_lang=source_language,
            tgt_lang=target_language,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            truncation=True,
        )
        translated.extend(item["translation_text"] for item in outputs)

    return translated


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    output_path = args.output or build_output_path(args.input, args.target_language)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it or pass --output with a different path."
        )

    print(f"Loading dataset from: {args.input}")
    df = pd.read_csv(args.input)
    text_column = resolve_text_column(df, args.text_column)
    print(f"Using dialogue column: {text_column}")

    # Preserve all original columns exactly, but operate on a clean string view
    # of the dialogue field for translation.
    source_texts = df[text_column].fillna("").astype(str).tolist()

    device_id, device_label = detect_device()
    print(f"Loading translation model: {args.model}")
    print(f"Running on device: {device_label}")
    translator = build_translator(args.model, device_id)

    translated_texts = translate_texts(
        translator,
        source_texts,
        source_language=args.source_language,
        target_language=args.target_language,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    if len(translated_texts) != len(df):
        raise RuntimeError(
            "Row count mismatch after translation: "
            f"{len(translated_texts)} translated rows for {len(df)} input rows."
        )

    output_df = df.copy()
    output_df[text_column] = translated_texts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved translated dataset to: {output_path}")
    print(f"Rows translated: {len(output_df)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTranslation interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
