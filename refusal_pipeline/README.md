# P2: Manipulation-Aware Refusal Pipeline

A fully **local**, **free**, open-source pipeline that:
1. **Detects** manipulative dialogue using a fine-tuned sequence classifier (`roberta-base` by default)
2. **Classifies** optional manipulation techniques and victim vulnerabilities
3. **Models multi-turn context** with an explicit context window and cumulative signal checks
4. **Refuses** politely and sets boundaries using rule-based templates
5. **Evaluates** detection quality, refusal appropriateness, and adversarial stress cases

Built on the [MentalManip](https://aclanthology.org/2024.acl-long.206/) dataset (ACL 2024).

> **No paid APIs, no cloud dependencies.** Runs entirely on CPU or a single GPU.

For a detailed implementation walkthrough, see [`IMPLEMENTATION.md`](./IMPLEMENTATION.md).

---

## Quick Start

```bash
# 1. Install dependencies
cd refusal_pipeline
pip install -r requirements.txt

# 2. Train the manipulation detector
python3 train_detector.py --data_path ../mentalmanip_dataset/mentalmanip_con.csv --epochs 3 --threshold 0.50

# 3. Optional: train fine-grained enrichers from the project plan
python3 train_attributes.py --task technique --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 train_attributes.py --task vulnerability --data_path ../mentalmanip_dataset/mentalmanip_con.csv

# 4. Evaluate (quantitative metrics + qualitative examples)
python3 evaluate.py --model_path ./saved_model/final --data_path ../mentalmanip_dataset/mentalmanip_con.csv

# 5. Run adversarial/hard-negative checks
python3 stress_tests.py --model_path ./saved_model/final

# If you do not have a trained detector yet, validate the policy layer only
python3 stress_tests.py --policy_only

# 6. Run an end-to-end smoke test without downloads or a trained checkpoint
python3 smoke_test.py

# 7. Run the demo
python3 demo.py --model_path ./saved_model/final --input "Person1: If you really loved me you'd do this. Person2: I don't know..."

# 8. Interactive mode with a recent-turn context window
python3 demo.py --model_path ./saved_model/final --context_turns 8

# 9. Interactive mode with full dialogue context
python3 demo.py --model_path ./saved_model/final
```

## Multilingual / Indic Extension

The multilingual stack is additive: it does **not** replace the deterministic
refusal path. It adds configurable training/evaluation utilities around the
existing detector so we can benchmark English, multilingual, Indic, code-mixed,
and Romanized inputs while keeping refusal generation rule-based.

### Recommended Model Choices

| Model | Best Use |
|------|---------|
| `xlm-roberta-base` | Best general multilingual baseline across English + Indic + code-mixed text |
| `google/muril-base-cased` | Best practical choice for Indian languages and transliterated/Romanized inputs |
| `ai4bharat/IndicBERTv2-SS` | Best Indic-first encoder when you want a lighter regional-language focus |
| `google/byt5-small` | Best fallback for noisy Romanized text, OCR noise, and spelling corruption |
| `microsoft/mdeberta-v3-base` | Strong multilingual/NLI backbone and recommended zero-shot transfer base |
| `sentence-transformers/LaBSE` | Retrieval, clustering, hard-negative mining, weak supervision |
| `intfloat/multilingual-e5-base` | Retrieval, pseudo-labeling, deduplication, and semantic search |

### Reproducible Commands

English baseline:

```bash
python3 train_multilingual.py \
  --task detection \
  --backbone roberta-base \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --output_dir ./saved_model/en_roberta
```

Multilingual baseline:

```bash
python3 train_multilingual.py \
  --task detection \
  --backbone xlm-roberta-base \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --output_dir ./saved_model/xlmr_multilingual
```

Indic-focused experiment:

```bash
python3 train_multilingual.py \
  --task detection \
  --backbone google/muril-base-cased \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --output_dir ./saved_model/muril_detection \
  --transliteration_normalization auto

python3 run_indic_suite.py \
  --model_path ./saved_model/muril_detection/final \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --output_dir ./saved_model/muril_detection/indic_suite
```

Zero-shot multilingual NLI baseline:

```bash
python3 run_zero_shot_nli.py \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --output_dir ./saved_model/zero_shot_nli
```

Optional shared-encoder multitask prototype:

```bash
python3 train_multitask.py \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --backbone xlm-roberta-base \
  --output_dir ./saved_model/multitask_xlmr
```

### Indic Data Note

The repository still ships only the original English MentalManip data. No
translated Indic benchmark set is committed here. To avoid changing the task
distribution with a different external dataset, the multilingual utilities are
designed to work on:

- the original English data
- future translated variants of `mentalmanip_con.csv`
- code-mixed/Romanized augmentations derived from the same source data

If you have a local translation model, you can generate a same-label Indic
variant with:

```bash
python3 build_indic_dataset.py \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --model_name <local-translation-model> \
  --languages hi bn ta te mr \
  --output_path ../mentalmanip_dataset/mentalmanip_indic_augmented.csv
```

This translation/augmentation step is intentionally isolated from the actual
runtime refusal path.

---

## Project Structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Load & split MentalManip CSV data |
| `train_detector.py` | Fine-tune the default RoBERTa-compatible detector |
| `train_attributes.py` | Fine-tune multi-label technique or vulnerability classifiers |
| `attribute_classifier.py` | Load and run optional attribute classifiers |
| `context_window.py` | Build recent-turn context windows and cumulative signal features |
| `refusal_policy.py` | Rule-based refusal templates by scenario |
| `evaluate.py` | Quantitative + qualitative evaluation |
| `stress_tests.py` | Adversarial and hard-negative regression checks |
| `smoke_test.py` | End-to-end runtime smoke test using a tiny local checkpoint |
| `demo.py` | End-to-end demo (CLI or interactive) |
| `requirements.txt` | Python dependencies |
| `train_multilingual.py` | Unified multilingual/Indic detector + attribute training |
| `train_multitask.py` | Experimental shared-encoder multi-task training |
| `evaluate_multilingual.py` | Per-language, robustness, and structured-result evaluation |
| `run_zero_shot_nli.py` | Zero-shot multilingual NLI benchmark |
| `run_indic_suite.py` | IndicXTREME-style multilingual benchmark runner |
| `build_indic_dataset.py` | Optional translation/augmentation utility for same-label Indic variants |
| `multilingual_support/` | Backbone registry, preprocessing, metrics, retrieval, weak supervision, utilities |

---

## Components

### A. Manipulation Detector
- Fine-tunes `roberta-base` (125M params) on `mentalmanip_con.csv`
- Uses HuggingFace `AutoTokenizer` and `AutoModelForSequenceClassification`, so compatible local checkpoints can be loaded for smoke tests or model swaps
- Stratified 60/20/20 train/valid/test split, 3 epochs by default, batch size 8
- Uses an explicit manipulation probability threshold for metrics and refusal routing
- Outputs: accuracy, precision, recall, F1, confusion matrix

### B. Fine-Grained Attribute Classifiers
- Optional multi-label classifiers for:
  - 11 manipulation techniques
  - 5 victim vulnerability types
- Saved under `saved_model/technique` and `saved_model/vulnerability`
- Loaded automatically by `demo.py` and qualitative evaluation if the models exist

### C. Multi-Turn Context
- `context_window.py` parses dialogue turns and can keep either:
  - the full dialogue
  - the most recent `--context_turns N` turns
  - the most recent `--context_chars N` characters after turn selection
- It also surfaces cumulative signals such as repeated guilt, pressure, intimidation, or flattery-plus-pressure.
- These signals are deterministic features used by the refusal policy; they are not a replacement for the detector.

### D. Refusal Policy
- Keyword-based scenario detection: **guilt-trip**, **pressure**, **intimidation**, **refund demand**, **escalation**
- Each scenario maps to a polite, boundary-setting response template that:
  - Acknowledges the speaker's feelings
  - Sets a clear boundary
  - Offers a constructive alternative
- If attribute classifiers are present, deterministic add-ons adapt the tone for detected techniques and vulnerabilities.
- If context signals indicate repeated pressure/escalation, deterministic add-ons make the refusal more bounded and procedural.

### E. Evaluation
- **Quantitative**: Standard classification metrics on the held-out test set
- **Qualitative**: 6 curated examples checking:
  - ✅ Correct detection
  - ✅ Polite refusal
  - ✅ Offers alternative help
- **Stress tests**: adversarial manipulative prompts and hard non-manipulative negatives. Run with a trained detector or with `--policy_only`.
- **Smoke test**: creates a tiny local checkpoint in `/tmp`, runs the real model/tokenizer loading path, then verifies context signals, detector routing, scenario selection, and refusal generation.

### F. Multilingual / Indic Tooling
- `multilingual_support/model_registry.py` exposes a unified backbone registry so experiments can swap between RoBERTa, XLM-R, MuRIL, IndicBERTv2, mDeBERTa-v3, mT5, ByT5, LaBSE, and multilingual-e5.
- `multilingual_support/preprocessing.py` adds language/script detection, robust text cleaning, code-mixed detection, and optional transliteration normalization.
- `multilingual_support/training.py` adds configurable multilingual training with full fine-tuning, frozen-encoder mode, or optional LoRA/PEFT.
- `multilingual_support/metrics.py` and `evaluate_multilingual.py` add per-language metrics, AUROC/AUPRC, calibration metrics, and structured JSON/CSV outputs plus summary plots.
- `multilingual_support/embedding_utils.py` adds LaBSE / multilingual-e5 retrieval for hard-negative mining, clustering, deduplication, and pseudo-labeling.
- `multilingual_support/generation_utils.py`, `active_learning.py`, and `annotation_triage.py` provide augmentation and annotation helpers that remain outside the refusal path.

---

## Current Plan Status

| Planned Phase | Current Status |
|---------------|----------------|
| Fine-grained classification | Implemented as optional default-RoBERTa multi-label classifiers |
| Vulnerability modeling | Implemented as optional default-RoBERTa multi-label classifier |
| LLM adaptation via LoRA | Not active in this local runtime; original experiment code contains Llama/LoRA research scaffolding |
| Multi-turn context | Implemented via explicit context windows and cumulative signal features |
| Robust evaluation | Implemented as qualitative checks plus `stress_tests.py` |
| Generative refusals | Intentionally not enabled in the refusal path; templates preserve deterministic safety behavior |

---

## Latest Full Pipeline Run

The full local command sequence was run from `refusal_pipeline/` with
`mentalmanip_dataset/mentalmanip_con.csv`. On this machine the `python` command
was not available, so the verified commands use `python3`.

The first detector training run required network access to download
`roberta-base` from HuggingFace. After that, later commands reused the local
cache. Some commands still printed HuggingFace `HEAD` retry warnings while
offline, but they continued successfully from cached model files.

Commands run successfully:

```bash
python3 train_detector.py --data_path ../mentalmanip_dataset/mentalmanip_con.csv --epochs 3
python3 train_attributes.py --task technique --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 train_attributes.py --task vulnerability --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 evaluate.py --model_path ./saved_model/final --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 stress_tests.py --model_path ./saved_model/final
```

### Binary Detector Results

The detector was trained for 3 epochs on CPU and saved to
`./saved_model/final`. Observed training runtime was about 2,755 seconds
(roughly 46 minutes).

```text
Stratified split:
  Train: 1749 samples  (manip=1210, non-manip=539)
  Valid: 583 samples   (manip=403, non-manip=180)
  Test:  583 samples   (manip=403, non-manip=180)

Validation at epoch 3:
  Accuracy:  0.7341
  Precision: 0.7672
  Recall:    0.8834
  F1:        0.8212
  Macro-F1:  0.6514

Held-out test set:
  Accuracy:  0.7033
  Precision: 0.7010
  Recall:    0.9950
  F1:        0.8226
  Macro-F1:  0.4584

Confusion matrix:
  [[  9 171]
   [  2 401]]
```

Conclusion: the detector is extremely recall-oriented at the default threshold
of `0.50`. It catches almost all manipulative test examples (401/403) but
misclassifies most non-manipulative test examples as manipulative (171/180).
This is appropriate for a conservative safety screen, but too high-friction for
production use without threshold tuning, calibration, or additional hard-negative
training.

### Attribute Classifier Results

Both optional attribute models trained successfully and were saved beside the
detector:

```text
Technique classifier:
  Train/valid/test: 1048 / 350 / 350 annotated samples
  Runtime:          about 2,077 seconds
  Test micro-F1:    0.3585
  Test macro-F1:    0.1473
  Test precision:   0.6205
  Test recall:      0.2521
  Subset accuracy:  0.2343
  Saved to:         ./saved_model/technique

Vulnerability classifier:
  Train/valid/test: 363 / 121 / 121 annotated samples
  Runtime:          about 781 seconds
  Test micro-F1:    0.4089
  Test macro-F1:    0.1227
  Test precision:   0.4946
  Test recall:      0.3485
  Subset accuracy:  0.3306
  Saved to:         ./saved_model/vulnerability
```

Insight: both multi-label classifiers are conservative at the default sigmoid
threshold of `0.50`. Early validation epochs predicted no labels above
threshold; later epochs produced usable labels with moderate micro-F1 but low
macro-F1. The low macro-F1 suggests rare technique/vulnerability classes remain
weak and should not be treated as definitive explanations.

### Integrated Evaluation And Stress Tests

`evaluate.py` loaded the detector and both attribute classifiers successfully.
The quantitative detector metrics matched the held-out test metrics above.
Qualitative examples showed four manipulative examples detected correctly, one
normal password-help exchange handled correctly, and one polite order-status
inquiry misclassified as manipulative. That qualitative false positive aligns
with the detector confusion matrix.

`stress_tests.py --model_path ./saved_model/final` produced:

```text
Detection accuracy: 0.6667
Precision:          0.6667
Recall:             1.0000
F1:                 0.8000
Scenario match:     4/6
Response checks:    5/6
```

The detector caught all four manipulative stress cases: layered guilt trip,
false urgency, veiled threat, and flattery plus pressure. It also flagged both
benign stress cases as manipulative, again confirming the high-recall /
low-specificity behavior. The refusal policy generally stayed polite and
constructive, but one refund-like benign case did not set a boundary because the
scenario layer routed it as `refund_demand` rather than a stronger pressure
scenario.

### Overall Conclusion

The implementation is now complete enough to run end to end with trained local
artifacts: detector, optional attribute enrichers, context windowing,
deterministic refusal templates, integrated evaluation, and stress tests. The
main correctness gap is no longer missing pipeline pieces; it is calibration.
The binary detector and attribute classifiers need threshold tuning and stronger
hard-negative coverage before the pipeline should be used where false positives
carry a meaningful user-experience cost.

---

## Requirements

- Python 3.8+
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- scikit-learn ≥ 1.2
- NumPy ≥ 1.23
- No internet needed after initial model download
