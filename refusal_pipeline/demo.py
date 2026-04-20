"""
Demo script — end-to-end manipulation detection and response generation.

Takes a dialogue (from CLI or interactive stdin) and outputs:
  1. Manipulation probability and label
  2. Scenario classification
  3. Final assistant response (refusal or normal)
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from attribute_classifier import load_available_attribute_classifiers, predict_available_attributes
from context_window import build_context_window
from refusal_policy import generate_response


def load_pipeline_config(model_path):
    config_path = os.path.join(model_path, "pipeline_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path, device):
    """Load the fine-tuned sequence-classification model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text, device, max_length=512, threshold=0.5):
    """Run the detector on a single text input."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        prob_manip = probs[0, 1].item()
        pred_label = int(prob_manip >= threshold)

    return pred_label, prob_manip


def run_pipeline(
    model,
    tokenizer,
    dialogue,
    device,
    max_length=512,
    threshold=0.5,
    attribute_classifiers=None,
    context_turns=None,
    context_chars=None,
):
    """Full pipeline: detect → classify scenario → generate response."""
    context = build_context_window(dialogue, max_turns=context_turns, max_chars=context_chars)
    detector_input = context.text
    pred_label, prob_manip = predict(model, tokenizer, detector_input, device, max_length, threshold)
    is_manip = pred_label == 1
    attributes = (
        predict_available_attributes(detector_input, attribute_classifiers)
        if is_manip and attribute_classifiers
        else {}
    )
    result = generate_response(dialogue, is_manip, prob_manip, attributes, context.signals)

    print("\n" + "=" * 60)
    print("  PIPELINE OUTPUT")
    print("=" * 60)
    print(f"\n📝 Input Dialogue:\n   {dialogue.replace(chr(10), chr(10) + '   ')}\n")
    print(f"🪟 Context Window:            {context.used_turns}/{context.total_turns} turns")
    if context.signals.get("active"):
        print(f"🧩 Context Signals:           {', '.join(context.signals['active'])}")
    print(f"🔍 Manipulation Probability: {prob_manip:.4f}")
    print(f"🎚️  Threshold:                {threshold:.2f}")
    print(f"🏷️  Predicted Label:          {'Manipulative' if is_manip else 'Non-manipulative'}")
    print(f"📂 Scenario:                 {result['scenario']}")
    if result["attributes"]:
        for task, prediction in result["attributes"].items():
            labels = ", ".join(prediction["labels"]) or "none above threshold"
            print(f"🧭 {task.title()}:                {labels}")
    print(f"\n💬 Assistant Response:")
    print(f"   {result['response'].replace(chr(10), chr(10) + '   ')}")
    print("=" * 60)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Demo: detect manipulation and generate a response"
    )
    parser.add_argument("--model_path", default="./saved_model/final",
                        help="Path to saved fine-tuned model")
    parser.add_argument("--input", type=str, default=None,
                        help="Dialogue text (if omitted, enters interactive mode)")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the saved manipulation probability threshold")
    parser.add_argument("--context_turns", type=int, default=None,
                        help="Use only the most recent N turns for detection")
    parser.add_argument("--context_chars", type=int, default=None,
                        help="Use only the most recent N characters after turn windowing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path} …")
    config = load_pipeline_config(args.model_path)
    max_length = args.max_length or config.get("max_length", 512)
    threshold = args.threshold if args.threshold is not None else config.get("threshold", 0.5)
    print(f"Using max_length={max_length}, threshold={threshold:.2f}")
    model, tokenizer = load_model(args.model_path, device)
    attribute_base_dir = os.path.dirname(os.path.normpath(args.model_path)) or "./saved_model"
    attribute_classifiers = load_available_attribute_classifiers(
        {
            "technique": os.path.join(attribute_base_dir, "technique"),
            "vulnerability": os.path.join(attribute_base_dir, "vulnerability"),
        },
        device=device,
    )
    if attribute_classifiers:
        print(f"Loaded attribute classifiers: {', '.join(sorted(attribute_classifiers))}")
    print("Model loaded ✅\n")

    if args.input:
        run_pipeline(
            model, tokenizer, args.input, device, max_length, threshold,
            attribute_classifiers, args.context_turns, args.context_chars,
        )
    else:
        # Interactive mode
        print("═" * 60)
        print("  Interactive Mode — type a dialogue and press Enter.")
        print("  Use \\n for newlines. Type 'quit' to exit.")
        print("═" * 60)
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue
            # Allow literal \n in input for multi-turn dialogues
            dialogue = user_input.replace("\\n", "\n")
            run_pipeline(
                model, tokenizer, dialogue, device, max_length, threshold,
                attribute_classifiers, args.context_turns, args.context_chars,
            )


if __name__ == "__main__":
    main()
