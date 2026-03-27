"""
Demo script — end-to-end manipulation detection and response generation.

Takes a dialogue (from CLI or interactive stdin) and outputs:
  1. Manipulation probability and label
  2. Scenario classification
  3. Final assistant response (refusal or normal)
"""

import argparse
import sys
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from refusal_policy import generate_response


def load_model(model_path, device):
    """Load the fine-tuned RoBERTa model and tokenizer."""
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text, device, max_length=512):
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
        pred_label = int(torch.argmax(logits, dim=-1).item())

    return pred_label, prob_manip


def run_pipeline(model, tokenizer, dialogue, device):
    """Full pipeline: detect → classify scenario → generate response."""
    pred_label, prob_manip = predict(model, tokenizer, dialogue, device)
    is_manip = pred_label == 1
    result = generate_response(dialogue, is_manip, prob_manip)

    print("\n" + "=" * 60)
    print("  PIPELINE OUTPUT")
    print("=" * 60)
    print(f"\n📝 Input Dialogue:\n   {dialogue.replace(chr(10), chr(10) + '   ')}\n")
    print(f"🔍 Manipulation Probability: {prob_manip:.4f}")
    print(f"🏷️  Predicted Label:          {'Manipulative' if is_manip else 'Non-manipulative'}")
    print(f"📂 Scenario:                 {result['scenario']}")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path} …")
    model, tokenizer = load_model(args.model_path, device)
    print("Model loaded ✅\n")

    if args.input:
        run_pipeline(model, tokenizer, args.input, device)
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
            run_pipeline(model, tokenizer, dialogue, device)


if __name__ == "__main__":
    main()
