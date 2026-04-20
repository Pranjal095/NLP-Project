"""
Train optional multi-label classifiers for manipulation technique or vulnerability.

Examples:
  python train_attributes.py --task technique --data_path ../mentalmanip_dataset/mentalmanip_con.csv
  python train_attributes.py --task vulnerability --data_path ../mentalmanip_dataset/mentalmanip_con.csv
"""

import argparse
import csv
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from attribute_classifier import LABELS_BY_TASK, labels_to_multihot


TARGET_COLUMN_BY_TASK = {
    "technique": "Technique",
    "vulnerability": "Vulnerability",
}


def make_training_args(**kwargs):
    """Handle the evaluation_strategy -> eval_strategy rename across Transformers releases."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value
    return TrainingArguments(**kwargs)


def load_attribute_data(file_path, task, train_ratio=0.6, valid_ratio=0.2, random_state=42):
    rows = []
    columns = None
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for idx, row in enumerate(reader):
            if idx == 0:
                columns = row
            else:
                rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    target_column = TARGET_COLUMN_BY_TASK[task]
    df = df[df[target_column].fillna("") != ""].copy()
    df["labels"] = df[target_column].apply(lambda value: labels_to_multihot(value, task))
    df["primary_label"] = df[target_column].apply(lambda value: value.split(",")[0].strip())

    test_ratio = 1.0 - train_ratio - valid_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + valid_ratio must be less than 1.0")

    stratify = df["primary_label"] if df["primary_label"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    valid_fraction_of_temp = valid_ratio / (valid_ratio + test_ratio)
    temp_stratify = (
        temp_df["primary_label"]
        if temp_df["primary_label"].value_counts().min() >= 2
        else None
    )
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_fraction_of_temp,
        random_state=random_state,
        shuffle=True,
        stratify=temp_stratify,
    )

    for name, split in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        print(f"  {name}: {len(split)} samples")

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_hf_dataset(df, tokenizer, max_length):
    ds = Dataset.from_dict({
        "text": df["Dialogue"].tolist(),
        "labels": df["labels"].tolist(),
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True, batch_size=64)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def make_compute_metrics(threshold=0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= threshold).astype(int)
        labels = np.asarray(labels).astype(int)

        return {
            "subset_accuracy": accuracy_score(labels, preds),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train technique/vulnerability classifier")
    parser.add_argument("--task", choices=sorted(LABELS_BY_TASK), required=True)
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default="./saved_model",
                        help="Base output directory. The task name is appended.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_output_dir = os.path.join(args.output_dir, args.task)
    labels = LABELS_BY_TASK[args.task]

    print(f"\nLoading {args.task} data from {args.data_path}")
    train_df, valid_df, test_df = load_attribute_data(
        args.data_path,
        args.task,
        random_state=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        problem_type="multi_label_classification",
    )

    train_ds = make_hf_dataset(train_df, tokenizer, args.max_length)
    valid_ds = make_hf_dataset(valid_df, tokenizer, args.max_length)
    test_ds = make_hf_dataset(test_df, tokenizer, args.max_length)

    training_args = make_training_args(
        output_dir=task_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=make_compute_metrics(args.threshold),
    )

    print("\nTraining attribute classifier...")
    trainer.train()

    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_ds)
    for key, value in sorted(results.items()):
        print(f"  {key:24s} = {value}")

    trainer.save_model(task_output_dir)
    tokenizer.save_pretrained(task_output_dir)
    with open(os.path.join(task_output_dir, "attribute_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": args.task,
                "labels": labels,
                "threshold": args.threshold,
                "max_length": args.max_length,
                "model_name": args.model_name,
                "split": {
                    "train_ratio": 0.6,
                    "valid_ratio": 0.2,
                    "test_ratio": 0.2,
                    "seed": args.seed,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved {args.task} classifier to {task_output_dir}")


if __name__ == "__main__":
    main()
