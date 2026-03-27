"""
Train a RoBERTa-based binary classifier for manipulation detection.

Fine-tunes roberta-base on the MentalManip consensus (or majority) dataset.
Saves the best model to `saved_model/` for use in evaluate.py and demo.py.

Adapted from experiments/manipulation_detection/model_roberta.py
"""

import argparse
import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data_loader import load_mentalmanip


# ---- metrics callback used by Trainer ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, average="binary", zero_division=0),
        "macro_f1":  f1_score(labels, preds, average="macro", zero_division=0),
    }


def make_hf_dataset(df, tokenizer, max_length=512):
    """Convert a pandas DataFrame into a tokenized HuggingFace Dataset."""
    ds = Dataset.from_dict({
        "text":  df["Dialogue"].tolist(),
        "label": df["Manipulative"].tolist(),
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True, batch_size=64)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def main():
    parser = argparse.ArgumentParser(description="Train manipulation detector")
    parser.add_argument("--data_path",
                        default="../mentalmanip_dataset/mentalmanip_con.csv",
                        help="Path to MentalManip CSV file")
    parser.add_argument("--model_name", default="roberta-base",
                        help="HuggingFace model id (default: roberta-base)")
    parser.add_argument("--output_dir", default="./saved_model",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length (128 is fast on CPU, 512 for GPU)")
    args = parser.parse_args()

    # ---- load data ----
    print(f"\n📂 Loading data from {args.data_path}")
    train_df, valid_df, test_df = load_mentalmanip(args.data_path)

    # ---- model & tokenizer ----
    print(f"\n🤖 Loading model: {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
    )

    # ---- tokenize datasets ----
    print("🔢 Tokenizing datasets …")
    train_ds = make_hf_dataset(train_df, tokenizer, args.max_length)
    valid_ds = make_hf_dataset(valid_df, tokenizer, args.max_length)
    test_ds  = make_hf_dataset(test_df,  tokenizer, args.max_length)

    # ---- training args ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",          # no wandb / tensorboard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    # ---- train ----
    print("\n🏋️  Training …")
    trainer.train()

    # ---- evaluate on test set ----
    print("\n📊 Evaluating on test set …")
    results = trainer.evaluate(test_ds)
    for k, v in sorted(results.items()):
        print(f"  {k:20s} = {v}")

    # ---- save model + tokenizer ----
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✅ Model saved to {final_dir}")

    # ---- confusion matrix ----
    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n  Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(labels, preds, average='binary', zero_division=0):.4f}")
    print(f"  Macro-F1:  {f1_score(labels, preds, average='macro', zero_division=0):.4f}")


if __name__ == "__main__":
    main()
