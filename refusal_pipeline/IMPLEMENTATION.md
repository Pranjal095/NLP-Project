# Manipulation-Aware Refusal Pipeline Implementation

This document describes the implementation in `refusal_pipeline/`: what each
module does, how data moves through the pipeline, what has been implemented from
the project plan, and which limitations remain.

## 1. Pipeline Overview

The runtime pipeline is:

```text
raw dialogue
  -> context_window.build_context_window
  -> binary manipulation detector
  -> optional technique/vulnerability classifiers
  -> refusal_policy.generate_response
  -> deterministic assistant response
```

The key safety design choice is unchanged from the progress report: the refusal
path does not use free-form generation. All refusal responses are deterministic
templates, optionally expanded with deterministic attribute/context add-ons.

## 2. Data Loading And Splitting

`data_loader.py` reads MentalManip CSV files with `csv.reader`, following the
dataset README's warning that plain `pandas.read_csv` can parse these files
incorrectly.

The binary detector uses a stratified 60/20/20 split:

```text
train: 60%
valid: 20%
test:  20%
```

For `mentalmanip_con.csv`, this currently produces:

```text
Train: 1749 samples  (manip=1210, non-manip=539)
Valid: 583 samples   (manip=403, non-manip=180)
Test: 583 samples    (manip=403, non-manip=180)
```

This fixes the earlier mismatch where the report described a stratified split
but the loader only shuffled and sliced.

## 3. Binary Detector

`train_detector.py` fine-tunes a HuggingFace sequence-classification model with
two labels. The production default is `roberta-base`, loaded through
`AutoTokenizer` and `AutoModelForSequenceClassification`:

```text
0 = Non-manipulative
1 = Manipulative
```

Important implementation details:

- Default model: `roberta-base`
- Loader: HuggingFace `AutoTokenizer` and `AutoModelForSequenceClassification`
- Default epochs: `3`
- Default batch size: `8`
- Default max token length: `128`
- Default manipulation threshold: `0.50`
- Metric for best checkpoint: binary F1
- Output directory: `saved_model/final`

The detector now uses thresholded manipulation probability for metrics and
downstream routing. This fixes the earlier ambiguity where the report mentioned
a confidence threshold but the code used only `argmax`.

Training writes `pipeline_config.json` beside the saved detector so evaluation
and demo runs can reuse:

- `threshold`
- `max_length`
- split ratios
- random seed
- label map

## 4. Fine-Grained Attribute Classifiers

`train_attributes.py` implements optional multi-label classifiers for the first
two planned extensions. It also defaults to `roberta-base` and uses HuggingFace
`Auto*` loading:

- `--task technique`: 11 manipulation techniques
- `--task vulnerability`: 5 victim vulnerability types

Technique labels:

```text
Denial
Evasion
Feigning Innocence
Rationalization
Playing Victim Role
Playing Servant Role
Shaming or Belittlement
Intimidation
Brandishing Anger
Accusation
Persuasion or Seduction
```

Vulnerability labels:

```text
Naivete
Dependency
Over-responsibility
Over-intellectualization
Low self-esteem
```

The script filters out rows without the target annotation and converts
comma-separated labels into multi-hot vectors. Models are saved under:

```text
saved_model/technique
saved_model/vulnerability
```

`attribute_classifier.py` loads any trained attribute models that exist beside
the detector and returns labels whose sigmoid probability is above the saved
threshold.

## 5. Multi-Turn Context

`context_window.py` makes context handling explicit.

It supports:

- full-dialogue detection by default
- recent-turn detection with `--context_turns N`
- recent-character detection with `--context_chars N`

The module parses common turn prefixes:

```text
Person1:
Person2:
Speaker1:
User:
Assistant:
Turn 3:
```

It also computes deterministic cumulative signals:

- `guilt`
- `pressure`
- `intimidation`
- `flattery`

These signals are not classifier labels. They are lightweight context features
used to make the refusal response more bounded when pressure or escalation
appears across multiple turns.

## 6. Refusal Policy

`refusal_policy.py` has three layers:

1. Scenario detection by keyword matching.
2. Base deterministic refusal template.
3. Optional deterministic add-ons from attribute and context signals.

Scenario tags:

```text
guilt_trip
pressure
intimidation
refund_demand
escalation
general
none
```

Every manipulative response keeps the same structure:

```text
acknowledge concern
set a boundary
offer a constructive alternative
```

For non-manipulative input, the policy returns a normal helpful response and
does not issue a refusal.

## 7. Demo And Evaluation

`demo.py` runs the complete pipeline on one dialogue or in interactive mode.

Useful commands:

```bash
python3 demo.py --model_path ./saved_model/final \
  --input "Person1: If you really loved me you would do this."

python3 demo.py --model_path ./saved_model/final --context_turns 8

python3 demo.py --model_path ./saved_model/final --threshold 0.65
```

`evaluate.py` runs:

- quantitative test-set metrics
- qualitative curated examples
- optional attribute-enriched refusal responses if attribute models exist

Useful commands:

```bash
python3 evaluate.py --model_path ./saved_model/final \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv

python3 evaluate.py --model_path ./saved_model/final \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --context_turns 8
```

## 8. Stress Tests

`stress_tests.py` adds adversarial and hard-negative regression checks.

It includes:

- layered guilt-tripping
- false urgency
- veiled threat
- flattery plus pressure
- polite non-manipulative request
- urgent but legitimate request

With a trained detector:

```bash
python3 stress_tests.py --model_path ./saved_model/final
```

Before a detector is trained:

```bash
python3 stress_tests.py --policy_only
```

Policy-only mode uses expected labels to exercise the scenario/refusal layer.
It does not measure detector quality.

## 9. End-To-End Smoke Test

`smoke_test.py` tests the runtime path without requiring internet access or a
trained checkpoint. It creates a tiny BERT sequence-classification checkpoint in
`/tmp/refusal_pipeline_smoke_model`, then loads it through the same
`demo.py` path used by a real trained detector.

The smoke model is deliberately biased toward class `1` so the manipulative
branch is deterministic. It is not a meaningful detector and should never be
used for quality claims.

Command:

```bash
python3 smoke_test.py
```

Verified behavior:

```text
context window: 3/3 turns
context signals: guilt, pressure
manipulation probability: 0.9997
predicted label: Manipulative
scenario: guilt_trip
response: deterministic refusal with context-aware pressure add-on
```

This proves that the pipeline wiring works end to end:

```text
local checkpoint -> tokenizer/model load -> context window -> detector probability
-> threshold routing -> scenario selection -> deterministic refusal
```

It does not prove detector accuracy.

## 10. Verification Performed

The following lightweight checks were run first to verify local wiring before
the expensive training commands:

```bash
python3 -m compileall refusal_pipeline
python3 refusal_pipeline/data_loader.py --data_path mentalmanip_dataset/mentalmanip_con.csv
python3 refusal_pipeline/refusal_policy.py
python3 refusal_pipeline/stress_tests.py --policy_only
python3 refusal_pipeline/stress_tests.py --policy_only --context_turns 2
python3 refusal_pipeline/smoke_test.py
```

The policy-only stress suite currently passes:

```text
Scenario match:  6/6
Response checks: 6/6
```

After that, the full command sequence was run from `refusal_pipeline/`:

```bash
python3 train_detector.py --data_path ../mentalmanip_dataset/mentalmanip_con.csv --epochs 3
python3 train_attributes.py --task technique --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 train_attributes.py --task vulnerability --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 evaluate.py --model_path ./saved_model/final --data_path ../mentalmanip_dataset/mentalmanip_con.csv
python3 stress_tests.py --model_path ./saved_model/final
```

Operational notes:

- `python` was not available on this machine; `python3` was used for verified
  commands.
- The first detector training run needed network access to download
  `roberta-base` from HuggingFace.
- After the initial download, the attribute and evaluation commands reused the
  local HuggingFace cache. Some commands still printed offline `HEAD` retry
  warnings, but they continued successfully from cached files.
- Matplotlib used temporary cache directories under `/tmp` because
  `/home/spidey/.config/matplotlib` was not writable. This warning did not
  affect the pipeline results.

## 11. Results And Conclusions

Current local results:

```text
Stratified data split:
  Train: 1749 samples  (manip=1210, non-manip=539)
  Valid: 583 samples   (manip=403, non-manip=180)
  Test:  583 samples   (manip=403, non-manip=180)

Policy-only adversarial stress tests:
  Scenario match:  6/6
  Response checks: 6/6

Policy-only stress tests with --context_turns 2:
  Scenario match:  6/6
  Response checks: 6/6

End-to-end smoke test:
  Checkpoint loading: passed
  Context signal extraction: passed
  Threshold routing: passed
  Scenario routing: passed
  Refusal generation: passed
```

Full detector training completed and saved the binary model to
`saved_model/final`:

```text
Training runtime: 2754.9493 seconds
Train loss:       0.5290

Validation at epoch 3:
  Loss:      0.5988
  Accuracy:  0.7341
  Precision: 0.7672
  Recall:    0.8834
  F1:        0.8212
  Macro-F1:  0.6514

Held-out test set:
  Loss:      0.5646
  Accuracy:  0.7033
  Precision: 0.7010
  Recall:    0.9950
  F1:        0.8226
  Macro-F1:  0.4584

Confusion matrix:
  [[  9 171]
   [  2 401]]
```

The confusion matrix is the most important detector result. The model catches
401 of 403 manipulative test examples, but only 9 of 180 non-manipulative test
examples. At threshold `0.50`, this is a high-recall detector with very low
specificity.

The optional attribute classifiers also trained and saved successfully:

```text
Technique classifier:
  Annotated split:    1048 train / 350 valid / 350 test
  Training runtime:   2076.8605 seconds
  Test loss:          0.2888
  Test subset acc:    0.2343
  Test precision:     0.6205
  Test recall:        0.2521
  Test micro-F1:      0.3585
  Test macro-F1:      0.1473
  Saved model:        saved_model/technique

Vulnerability classifier:
  Annotated split:    363 train / 121 valid / 121 test
  Training runtime:   781.2466 seconds
  Test loss:          0.4713
  Test subset acc:    0.3306
  Test precision:     0.4946
  Test recall:        0.3485
  Test micro-F1:      0.4089
  Test macro-F1:      0.1227
  Saved model:        saved_model/vulnerability
```

Attribute-classifier insight:

- Both multi-label heads were conservative at the default sigmoid threshold of
  `0.50`.
- Early validation passes predicted no labels above threshold.
- Later epochs produced usable labels, but recall stayed low and macro-F1
  remained weak.
- The labels are useful as refusal enrichers and diagnostic hints, but they
  should not be presented as authoritative explanations without calibration.

Integrated `evaluate.py` results:

```text
Loaded model from:       ./saved_model/final
Loaded attributes:       technique, vulnerability
Detector threshold:      0.50
Detector max length:     128
Quantitative metrics:    matched the held-out test metrics above
Qualitative examples:    5/6 detection correctness
```

Qualitative conclusions:

- Guilt-tripping, false urgency, veiled threat, and aggressive refund-pressure
  examples were detected as manipulative.
- A normal password-reset exchange was correctly treated as non-manipulative.
- A polite order-status inquiry was incorrectly flagged as manipulative with
  probability `0.5672`.
- Attribute enrichment worked technically, but often returned `Dependency` for
  vulnerability and sometimes no technique above threshold. This reinforces the
  need for per-label threshold tuning.

Stress-test results with the trained detector:

```text
Detection accuracy: 0.6667
Precision:          0.6667
Recall:             1.0000
F1:                 0.8000
Scenario match:     4/6
Response checks:    5/6
```

Stress-test conclusions:

- The detector caught all manipulative stress cases: layered guilt-tripping,
  false urgency, veiled threat, and flattery plus pressure.
- Both benign hard negatives were flagged as manipulative:
  `polite boundary request` and `urgent but legitimate`.
- Scenario routing worked for all manipulative stress cases, but benign false
  positives naturally caused scenario mismatches.
- The refusal templates stayed polite and constructive on 5/6 cases. The one
  response-check miss came from a refund-like benign case that did not include a
  strong enough boundary after being routed through the refund template.

Conclusions:

- The implementation now matches the report more closely: stratified split,
  explicit thresholding, deterministic refusal path, and modular evaluation.
- The pipeline has progressed beyond the report by adding optional
  technique/vulnerability classifiers, multi-turn context features, and stress
  tests.
- The refusal path remains deterministic, which preserves the original safety
  rationale.
- The full command sequence now runs end to end with trained local artifacts.
- The main empirical limitation is calibration: the detector and attribute heads
  are too conservative at threshold `0.50`, producing excellent manipulative
  recall but many benign false positives.
- The next highest-value implementation step is validation-set threshold tuning,
  especially per-label thresholds for technique and vulnerability outputs and a
  detector threshold chosen for the intended safety/false-positive tradeoff.

## 12. Plan Status And Remaining Work

Implemented:

- Binary manipulation detector pipeline.
- Stratified split.
- Explicit thresholding.
- Technique classifier training/inference.
- Vulnerability classifier training/inference.
- Multi-turn context windowing.
- Cumulative deterministic context signals.
- Attribute/context-aware deterministic refusal add-ons.
- Qualitative and stress-test evaluation harnesses.

Not implemented as active runtime behavior:

- LoRA/prefix-tuned LLM adaptation. The original `experiments/` directory has
  Llama/LoRA research code, but this local refusal pipeline does not train or
  serve an LLM adapter.
- Generative refusals. This is intentionally deferred because deterministic
  templates are safer for the refusal path and match the report's key design
  choice.
- Human evaluation. The current checks are automated and qualitative; human
  ratings for politeness/helpfulness/resistance remain future work.

## 13. Recommended Training Order

From `refusal_pipeline/`:

```bash
python3 train_detector.py \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --epochs 3 \
  --threshold 0.50

python3 train_attributes.py \
  --task technique \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --epochs 5

python3 train_attributes.py \
  --task vulnerability \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv \
  --epochs 5

python3 evaluate.py \
  --model_path ./saved_model/final \
  --data_path ../mentalmanip_dataset/mentalmanip_con.csv

python3 stress_tests.py --model_path ./saved_model/final
```
