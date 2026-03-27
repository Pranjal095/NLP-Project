# P2: Manipulation-Aware Refusal Pipeline

A fully **local**, **free**, open-source pipeline that:
1. **Detects** manipulative dialogue using a fine-tuned RoBERTa classifier
2. **Refuses** politely and sets boundaries using rule-based templates
3. **Evaluates** detection quality and refusal appropriateness

Built on the [MentalManip](https://aclanthology.org/2024.acl-long.206/) dataset (ACL 2024).

> **No paid APIs, no cloud dependencies.** Runs entirely on CPU or a single GPU.

---

## Quick Start

```bash
# 1. Install dependencies
cd p2_refusal_pipeline
pip install -r requirements.txt

# 2. Train the manipulation detector (~10 min on GPU, ~1 hr on CPU)
python train_detector.py --data_path ../mentalmanip_dataset/mentalmanip_con.csv --epochs 3

# 3. Evaluate (quantitative metrics + qualitative examples)
python evaluate.py --model_path ./saved_model/final --data_path ../mentalmanip_dataset/mentalmanip_con.csv

# 4. Run the demo
python demo.py --model_path ./saved_model/final --input "Person1: If you really loved me you'd do this. Person2: I don't know..."

# 5. Interactive mode
python demo.py --model_path ./saved_model/final
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Load & split MentalManip CSV data |
| `train_detector.py` | Fine-tune RoBERTa for manipulation detection |
| `refusal_policy.py` | Rule-based refusal templates by scenario |
| `evaluate.py` | Quantitative + qualitative evaluation |
| `demo.py` | End-to-end demo (CLI or interactive) |
| `requirements.txt` | Python dependencies |

---

## Components

### A. Manipulation Detector
- Fine-tunes `roberta-base` (125M params) on `mentalmanip_con.csv`
- 60/20/20 train/valid/test split, 3 epochs, batch size 8
- Outputs: accuracy, precision, recall, F1, confusion matrix

### B. Refusal Policy
- Keyword-based scenario detection: **guilt-trip**, **pressure**, **intimidation**, **refund demand**, **escalation**
- Each scenario maps to a polite, boundary-setting response template that:
  - Acknowledges the speaker's feelings
  - Sets a clear boundary
  - Offers a constructive alternative

### C. Evaluation
- **Quantitative**: Standard classification metrics on the held-out test set
- **Qualitative**: 6 curated examples checking:
  - ✅ Correct detection
  - ✅ Polite refusal
  - ✅ Offers alternative help

---

## Requirements

- Python 3.8+
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- scikit-learn ≥ 1.2
- No internet needed after initial model download
