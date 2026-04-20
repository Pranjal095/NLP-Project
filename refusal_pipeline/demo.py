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

from attribute_classifier import load_available_attribute_classifiers, predict_available_attributes
from context_window import build_context_window
from multilingual_support.inference import load_text_model, predict_text
from multilingual_support.preprocessing import PreprocessingConfig
from refusal_policy import generate_response


def load_pipeline_config(model_path):
    config_path = os.path.join(model_path, "pipeline_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path, device):
    """Load the fine-tuned sequence-classification model and tokenizer."""
    model, tokenizer, _, _ = load_text_model(model_path, device=device)
    return model, tokenizer


def load_model_bundle(model_path, device):
    return load_text_model(model_path, device=device)


def predict(
    model,
    tokenizer,
    text,
    device,
    max_length=512,
    threshold=0.5,
    *,
    task_config=None,
    model_mode="legacy",
    preprocessing_config=None,
):
    """Run the detector on a single text input."""
    prediction = predict_text(
        model,
        tokenizer,
        text,
        device,
        max_length=max_length,
        threshold=threshold,
        task_config=task_config,
        mode=model_mode,
        preprocessing_config=preprocessing_config,
    )
    return prediction["prediction"], prediction["probability"], prediction["metadata"], prediction["processed_text"]


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
    task_config=None,
    model_mode="legacy",
    preprocessing_config=None,
):
    """Full pipeline: detect → classify scenario → generate response."""
    context = build_context_window(dialogue, max_turns=context_turns, max_chars=context_chars)
    detector_input = context.text
    pred_label, prob_manip, language_metadata, processed_text = predict(
        model,
        tokenizer,
        detector_input,
        device,
        max_length,
        threshold,
        task_config=task_config,
        model_mode=model_mode,
        preprocessing_config=preprocessing_config,
    )
    is_manip = pred_label == 1
    attributes = (
        predict_available_attributes(processed_text, attribute_classifiers)
        if is_manip and attribute_classifiers
        else {}
    )
    result = generate_response(dialogue, is_manip, prob_manip, attributes, context.signals)
    result["language_metadata"] = language_metadata

    print("\n" + "=" * 60)
    print("  PIPELINE OUTPUT")
    print("=" * 60)
    print(f"\n📝 Input Dialogue:\n   {dialogue.replace(chr(10), chr(10) + '   ')}\n")
    print(f"🪟 Context Window:            {context.used_turns}/{context.total_turns} turns")
    print(f"🌐 Detected Language:         {language_metadata['language']} ({language_metadata['script']})")
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
    parser.add_argument("--transliteration_normalization", choices=["auto", "basic", "external", "off"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path} …")
    model, tokenizer, config, model_mode = load_model_bundle(args.model_path, device)
    max_length = args.max_length or config.get("max_length", 512)
    threshold = args.threshold if args.threshold is not None else config.get("threshold", 0.5)
    print(f"Using max_length={max_length}, threshold={threshold:.2f}")
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
    preprocessing = PreprocessingConfig(transliteration_normalization=args.transliteration_normalization)

    if args.input:
        run_pipeline(
            model, tokenizer, args.input, device, max_length, threshold,
            attribute_classifiers, args.context_turns, args.context_chars,
            task_config=config, model_mode=model_mode, preprocessing_config=preprocessing,
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
                task_config=config, model_mode=model_mode, preprocessing_config=preprocessing,
            )


if __name__ == "__main__":
    main()
