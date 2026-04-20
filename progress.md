# MentalManip Refusal Pipeline: Progress Report

Based on the codebase (`refusal_pipeline/IMPLEMENTATION.md` and related files) and the Mid-Project Progress Report (`NLP_Project_Group_39.pdf`), here is a comprehensive breakdown of what has been implemented so far and how it contrasts with the originally planned phases.

## 1. What Has Been Implemented So Far?

The `refusal_pipeline` has matured significantly since the mid-progress report, shifting from a rudimentary binary classifier to a modular, multi-turn, and attribute-aware system:

- **Enhanced Binary Detector Pipeline:** The core `roberta-base` pipeline remains, but critical bugs from the mid-point have been fixed. It now correctly leverages **stratified splits** (60/20/20) and uses explicit confidence thresholding instead of native `argmax` for routing logic.
- **Fine-Grained Attribute Classifiers:** The system now ships with multi-label classifiers targeting specific traits. It extracts bounding boxes around **11 manipulation techniques** (e.g., "Shaming," "Evasion") and **5 victim vulnerabilities** (e.g., "Dependency," "Low self-esteem").
- **Multi-Turn Context Handling:** A newly added `context_window.py` handles full dialogue contexts natively instead of just single-turn prompts. It calculates deterministic cumulative signals (like escalating guilt or intimidation) across multiple turns.
- **Adversarial Stress Testing:** A standalone `stress_tests.py` module evaluates the pipeline specifically against layered guilt boundaries, hard-negative requests (e.g., urgent but legitimate requests), and veiled threats.
- **Attribute-Enriched deterministic Refusals:** The system dynamically adjusts its deterministic refusal templates by injecting the fine-grained multi-turn traits (e.g., adjusting the boundary's firmness if long-term manipulation is detected).

## 2. Comparison to the Planned Phases

Here is how the current implementation compares to the "Next Steps" listed in the mid-project report:

| Original Planned Phase | Status | Contrast & Observations |
| :--- | :--- | :--- |
| **Phase 1: Fine-Grained Classification** | **Completed** | Implemented through `train_attributes.py`. It actively parses the 11-class multi-label taxonomy. However, currently, the recall is relatively low at the default `0.50` threshold and macro-F1 is weak; it requires per-label calibration. |
| **Phase 2: Vulnerability Modeling** | **Completed** | Implemented alongside Phase 1. It actively predicts the 5 victim vulnerability types to adapt refusal tones, giving multi-hot vectors as diagnostic hints. |
| **Phase 3: LLM Adaptation via LoRA** | **Not Implemented** | This was skipped for the local runtime refusal policy. While original experiments have LLaMA/LoRA research code, it wasn't integrated into the final local pipeline, preserving the system's ability to run completely locally on CPU without extreme memory footprints. |
| **Phase 4: Multi-Turn Context** | **Completed** | Full multi-turn context extraction and continuous token signals were built via `context_window.py`, resolving the mid-point's "single-turn only" limitation. |
| **Phase 5: Robust Evaluation** | **Partially Completed** | The adversarial prompt stress tests and expanded evaluation suites (`stress_tests.py`) are fully complete. **Missing:** Human evaluation framework design (e.g., rating helpfulness/politeness locally) and annotator uncertainty analysis remain unaddressed. |
| **Phase 6: Generative Refusals** | **Intentionally Skipped** | The plan to replace deterministic templates with a generative (LLM) model was intentionally deferred and practically scrapped. The team opted to preserve the deterministic template approach because allowing a generative path for refusals risks being manipulated or bypassed recursively. Instead, the team upgraded the deterministic templates to be "attribute-aware." |

## Summary
The project succeeded heavily in expanding its detection capabilities contextually (Phases 1, 2, and 4) and building out programmatic stress-testing frameworks (Phase 5). The primary deviation from the plan was abandoning LLM adaptation and generative refusals (Phases 3 and 6)—which turned out to be an excellent architectural decision, retaining predictability and eliminating the risk of recursive jailbreaks inside the refusal path itself.
