"""Train multilingual manipulation detectors and attribute classifiers.

Examples:
  python3 train_multilingual.py --task detection --backbone roberta-base \
    --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
    --output_dir ./saved_model/en_roberta

  python3 train_multilingual.py --task detection --backbone xlm-roberta-base \
    --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
    --output_dir ./saved_model/xlmr_multilingual

  python3 train_multilingual.py --task technique --backbone google/muril-base-cased \
    --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
    --output_dir ./saved_model/muril_technique
"""

from __future__ import annotations

import argparse
import json

from multilingual_support.model_registry import list_backbones
from multilingual_support.preprocessing import PreprocessingConfig
from multilingual_support.training import (
    TrainingRunConfig,
    save_run_manifest,
    train_binary_detector,
    train_multilabel_classifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multilingual/Indic-ready classifiers")
    parser.add_argument("--task", choices=["detection", "technique", "vulnerability"], required=True)
    parser.add_argument("--backbone", default="roberta-base", help=f"Registered backbone. Choices: {', '.join(list_backbones())}")
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--output_dir", default="./saved_model/multilingual")
    parser.add_argument("--training_mode", choices=["full", "frozen", "lora"], default="full")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_urls", action="store_true")
    parser.add_argument("--collapse_elongations", action="store_true")
    parser.add_argument(
        "--transliteration_normalization",
        choices=["auto", "basic", "external", "off"],
        default="auto",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preprocessing = PreprocessingConfig(
        lowercase=args.lowercase,
        strip_urls=args.strip_urls,
        collapse_elongations=args.collapse_elongations,
        transliteration_normalization=args.transliteration_normalization,
    )
    run_config = TrainingRunConfig(
        backbone_name=args.backbone,
        task_name=args.task,
        output_dir=args.output_dir,
        training_mode=args.training_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        threshold=args.threshold,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    if args.task == "detection":
        metrics = train_binary_detector(args.data_path, run_config, preprocessing_config=preprocessing)
    else:
        metrics = train_multilabel_classifier(
            args.data_path,
            run_config,
            task=args.task,
            preprocessing_config=preprocessing,
        )

    save_run_manifest(
        args.output_dir,
        {
            "task": args.task,
            "backbone": args.backbone,
            "training_mode": args.training_mode,
            "metrics": metrics,
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
