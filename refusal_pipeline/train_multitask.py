"""Experimental multi-task training entry point."""

from __future__ import annotations

import argparse
import json

from multilingual_support.multitask import MultitaskRunConfig, train_multitask_model
from multilingual_support.preprocessing import PreprocessingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shared-encoder multitask detector/attribute models")
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--backbone", default="xlm-roberta-base")
    parser.add_argument("--output_dir", default="./saved_model/multitask")
    parser.add_argument("--training_mode", choices=["full", "frozen", "lora"], default="full")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transliteration_normalization", choices=["auto", "basic", "external", "off"], default="auto")
    args = parser.parse_args()

    run_config = MultitaskRunConfig(
        backbone_name=args.backbone,
        output_dir=args.output_dir,
        training_mode=args.training_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )
    preprocessing = PreprocessingConfig(transliteration_normalization=args.transliteration_normalization)
    results = train_multitask_model(args.data_path, run_config, preprocessing_config=preprocessing)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
