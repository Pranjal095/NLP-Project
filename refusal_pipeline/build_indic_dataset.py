"""Optional translation/augmentation pipeline for creating Indic variants.

This script is intentionally kept separate from classifier training so that
translation augmentation does not become part of the refusal decision path.
It can be used later when a local translation model is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from multilingual_support.dataset_utils import load_dataframe
from multilingual_support.generation_utils import GenerationConfig, translate_texts


TARGET_LANGUAGE_PREFIXES = {
    "hi": "translate English to Hindi: ",
    "bn": "translate English to Bengali: ",
    "ta": "translate English to Tamil: ",
    "te": "translate English to Telugu: ",
    "mr": "translate English to Marathi: ",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build translated Indic dataset variants from MentalManip")
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--output_path", default="../mentalmanip_dataset/mentalmanip_indic_augmented.csv")
    parser.add_argument("--model_name", required=True, help="Local translation or seq2seq model checkpoint")
    parser.add_argument("--languages", nargs="+", default=["hi", "bn", "ta", "te", "mr"])
    args = parser.parse_args()

    dataframe = load_dataframe(args.data_path)
    config = GenerationConfig(model_name=args.model_name)
    rows = []
    for language in args.languages:
        prefix = TARGET_LANGUAGE_PREFIXES.get(language)
        if not prefix:
            raise ValueError(f"No translation prompt configured for language {language}")
        translated = translate_texts(dataframe["Dialogue"].astype(str).tolist(), config, task_prefix=prefix)
        augmented = dataframe.copy()
        augmented["Dialogue"] = translated
        augmented["language"] = language
        augmented["source_language"] = "en"
        rows.append(augmented)

    combined = pd.concat([dataframe.assign(language="en", source_language="en")] + rows, ignore_index=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote translated dataset to {output_path}")


if __name__ == "__main__":
    main()
