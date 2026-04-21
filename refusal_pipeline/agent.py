#!/usr/bin/env python3
"""
Manipulation Guard Agent — autonomous safety layer for conversational AI.

The agent wraps any text-based conversational system and intercepts each
user turn before it reaches the underlying AI.  For every turn it:

  1. Preprocesses the input (language detection, transliteration normalisation)
  2. Runs the manipulation detector (binary, XLM-R / RoBERTa fine-tuned on MentalManip)
  3. If manipulative, runs the technique + vulnerability attribute classifiers
  4. Updates the session risk score using a streak-aware accumulation model
  5. Adapts its detection threshold as session risk rises (more conservative
     when the conversation has already shown manipulation patterns)
  6. Selects an action: ALLOW / WARN / DEFLECT / FREEZE
  7. Generates an appropriate response and logs a structured turn record

What makes this agentic (vs. the existing demo.py single-turn script):
  - Persistent session state across turns (risk, streak, history)
  - Risk-adaptive detection threshold: the agent becomes stricter after
    detecting repeated patterns, modelling how human guardians would behave
  - State machine transitions: CLEAR → CAUTIOUS → DEFLECTING → FROZEN
  - End-of-session report with per-language breakdown and technique summary
  - Three scripted demonstration scenarios that show the pipeline's utility
    across English escalation, Hindi zero-shot transfer, and code-mixed input

Key model performance (from comparisons.md):
  - English F1:  0.7897  AUROC: 0.7122
  - Bilingual F1: 0.7730  AUROC: 0.7209
  - Hindi F1:    0.7476  (zero-shot transfer, trained on English only)

Usage:
  python agent.py                      # run all demo scenarios
  python agent.py --demo escalation    # single scenario
  python agent.py --interactive        # interactive mode after demo
  python agent.py --model_path <path>  # use a different checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# ── pipeline imports ──────────────────────────────────────────────────────────
from attribute_classifier import (
    load_available_attribute_classifiers,
    predict_available_attributes,
)
from context_window import build_context_window
from multilingual_support.inference import load_text_model, predict_text
from multilingual_support.preprocessing import PreprocessingConfig, preprocess_text
from refusal_policy import detect_scenario, generate_response


# ── terminal colour helpers ────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
CYAN    = "\033[96m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
GRAY    = "\033[90m"
WHITE   = "\033[97m"


def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def risk_bar(value: float, width: int = 28) -> str:
    filled = max(0, min(width, int(value * width)))
    empty  = width - filled
    color  = RED if value > 0.75 else YELLOW if value > 0.40 else GREEN
    return c("█" * filled + "░" * empty, color)


def prob_badge(prob: float, threshold: float) -> str:
    triggered = prob >= threshold
    color     = RED if triggered else GREEN
    marker    = "●" if triggered else "○"
    return c(f"{marker} {prob:.3f}", color)


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    turn_id:        int
    text:           str
    language:       str
    script:         str
    is_code_mixed:  bool
    probability:    float
    is_manipulative: bool
    scenario:       str
    techniques:     List[str]
    vulnerabilities: List[str]
    action:         str
    response:       str
    session_risk_after: float


# Agent states — drives how strictly the agent applies its threshold
STATES = ("CLEAR", "CAUTIOUS", "DEFLECTING", "FROZEN")


@dataclass
class SessionState:
    state:              str  = "CLEAR"
    cumulative_risk:    float = 0.0
    manipulation_streak: int = 0
    turns:              List[TurnRecord] = field(default_factory=list)
    freeze_turn:        Optional[int]   = None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def manipulation_count(self) -> int:
        return sum(1 for t in self.turns if t.is_manipulative)

    @property
    def manipulation_rate(self) -> float:
        return self.manipulation_count / self.turn_count if self.turns else 0.0


# ── risk accumulation model ────────────────────────────────────────────────────
# Each manipulative turn raises the session risk; clean turns decay it slightly.
# A streak multiplier makes repeated manipulation compound non-linearly,
# mirroring how repeated pressure tactics are more concerning than isolated ones.

BASE_RISK_GAIN    = 0.22   # base risk per manipulative turn
STREAK_MULTIPLIER = 0.08   # extra per additional consecutive turn
CLEAN_DECAY       = 0.06   # risk decay per clean turn
FREEZE_THRESHOLD  = 0.85   # freeze the session above this risk

# State thresholds and the threshold adjustments they apply
STATE_TRANSITIONS = {
    "CLEAR":      (0.00, 0.00),   # (min risk to enter, threshold offset)
    "CAUTIOUS":   (0.25, -0.05),  # agent becomes slightly more sensitive
    "DEFLECTING": (0.55, -0.10),  # noticeably more sensitive
    "FROZEN":     (FREEZE_THRESHOLD, None),
}


def _new_state(risk: float) -> str:
    """Derive agent state from cumulative risk."""
    if risk >= FREEZE_THRESHOLD:
        return "FROZEN"
    if risk >= STATE_TRANSITIONS["DEFLECTING"][0]:
        return "DEFLECTING"
    if risk >= STATE_TRANSITIONS["CAUTIOUS"][0]:
        return "CAUTIOUS"
    return "CLEAR"


def _effective_threshold(base: float, state: str) -> float:
    """Return the risk-adjusted detection threshold for the current state."""
    offset = STATE_TRANSITIONS.get(state, ("", 0.0))[1]
    if offset is None:
        return base
    return max(0.20, base + offset)


def _risk_delta(is_manip: bool, streak: int, prob: float) -> float:
    if not is_manip:
        return -CLEAN_DECAY
    base  = BASE_RISK_GAIN + STREAK_MULTIPLIER * streak
    scale = 0.5 + 0.5 * prob          # scale by model confidence
    return base * scale


# ── the agent ─────────────────────────────────────────────────────────────────

class ManipulationGuardAgent:
    """
    An autonomous agent that guards a conversation against psychological
    manipulation.

    Agent loop (per turn):
        observe  → preprocess text, detect language
        analyse  → run detector and (if positive) attribute classifiers
        adapt    → adjust effective threshold based on session state
        decide   → select action based on analysis + session history
        respond  → generate response and update state
    """

    def __init__(
        self,
        model_path: str             = "./saved_model/final",
        threshold:  Optional[float] = None,
        max_length: int             = 128,
        freeze_threshold: float     = FREEZE_THRESHOLD,
        verbose: bool               = True,
    ) -> None:
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length      = max_length
        self.freeze_threshold = freeze_threshold
        self.verbose          = verbose

        if verbose:
            print(c("\n  Loading detection pipeline …", CYAN))

        self.model, self.tokenizer, self.config, self.mode = load_text_model(
            model_path, device=self.device
        )
        self.base_threshold = (
            threshold
            if threshold is not None
            else float(self.config.get("threshold", 0.55))
        )
        self.preprocessing = PreprocessingConfig()

        attr_base = os.path.dirname(os.path.normpath(model_path))
        self.attribute_classifiers = load_available_attribute_classifiers(
            {
                "technique":     os.path.join(attr_base, "technique"),
                "vulnerability": os.path.join(attr_base, "vulnerability"),
            },
            device=self.device,
        )
        if verbose and self.attribute_classifiers:
            print(c(
                f"  Attribute classifiers: {', '.join(sorted(self.attribute_classifiers))}",
                GRAY,
            ))

        self.session = SessionState()

    # ── public API ─────────────────────────────────────────────────────────────

    def process_turn(self, user_text: str, print_report: bool = True) -> str:
        """
        Process one user turn through the full agent loop.

        Returns the agent's response string.
        """
        # ── already frozen? ──────────────────────────────────────────────────
        if self.session.state == "FROZEN":
            response = (
                "This session has been suspended due to repeated manipulation "
                "signals. A human supervisor will follow up."
            )
            if print_report:
                self._print_frozen_turn(user_text)
            return response

        # ── OBSERVE: preprocess and detect language ───────────────────────────
        prep        = preprocess_text(user_text, self.preprocessing)
        language    = prep.metadata.detected_language
        script      = prep.metadata.dominant_script
        is_code_mix = prep.metadata.is_code_mixed

        # ── ADAPT: compute effective threshold for this turn ──────────────────
        eff_threshold = _effective_threshold(self.base_threshold, self.session.state)

        # ── ANALYSE: run detector ─────────────────────────────────────────────
        result   = predict_text(
            self.model, self.tokenizer,
            prep.processed_text,
            self.device,
            max_length   = self.max_length,
            threshold    = eff_threshold,
            task_config  = self.config,
            mode         = self.mode,
        )
        prob     = result["probability"]
        is_manip = result["prediction"] == 1

        # ── ANALYSE: attribute classifiers (only when manipulative) ───────────
        attributes  = {}
        techniques  = []
        vulns       = []
        if is_manip and self.attribute_classifiers:
            attributes = predict_available_attributes(
                prep.processed_text, self.attribute_classifiers
            )
            tech_pred = attributes.get("technique")
            vuln_pred = attributes.get("vulnerability")
            if tech_pred:
                techniques = tech_pred.labels
            if vuln_pred:
                vulns = vuln_pred.labels

        scenario = detect_scenario(user_text) if is_manip else "none"

        # ── DECIDE: update session risk and pick action ───────────────────────
        streak = self.session.manipulation_streak + 1 if is_manip else 0
        delta  = _risk_delta(is_manip, self.session.manipulation_streak, prob)
        new_risk = min(1.0, max(0.0, self.session.cumulative_risk + delta))
        new_state = _new_state(new_risk)

        if not is_manip:
            action   = "ALLOW"
            response = "Thank you for your message. How can I help you today?"
        elif new_risk >= self.freeze_threshold:
            action   = "FREEZE"
            response = (
                "The sustained level of manipulation in this conversation "
                "has exceeded what I can safely engage with. This session is "
                "now suspended. A human representative will follow up."
            )
            new_state = "FROZEN"
        elif new_state == "DEFLECTING" or prob > 0.78:
            action   = "DEFLECT"
            gen      = generate_response(user_text, True, prob, attributes)
            response = gen["response"]
        else:
            action   = "WARN"
            gen      = generate_response(user_text, True, prob, attributes)
            response = gen["response"]

        # ── UPDATE state ──────────────────────────────────────────────────────
        if is_manip:
            self.session.manipulation_streak += 1
        else:
            self.session.manipulation_streak = 0
        self.session.cumulative_risk = new_risk
        self.session.state           = new_state
        if new_state == "FROZEN" and self.session.freeze_turn is None:
            self.session.freeze_turn = self.session.turn_count + 1

        record = TurnRecord(
            turn_id          = self.session.turn_count + 1,
            text             = user_text,
            language         = language,
            script           = script,
            is_code_mixed    = is_code_mix,
            probability      = prob,
            is_manipulative  = is_manip,
            scenario         = scenario,
            techniques       = techniques,
            vulnerabilities  = vulns,
            action           = action,
            response         = response,
            session_risk_after = new_risk,
        )
        self.session.turns.append(record)

        if print_report:
            self._print_turn_report(record, eff_threshold)

        return response

    def reset_session(self) -> None:
        self.session = SessionState()

    # ── terminal reporting ─────────────────────────────────────────────────────

    def _action_color(self, action: str) -> str:
        return {
            "ALLOW":   GREEN,
            "WARN":    YELLOW,
            "DEFLECT": RED,
            "FREEZE":  MAGENTA,
        }.get(action, WHITE)

    def _state_color(self, state: str) -> str:
        return {
            "CLEAR":      GREEN,
            "CAUTIOUS":   YELLOW,
            "DEFLECTING": RED,
            "FROZEN":     MAGENTA,
        }.get(state, WHITE)

    def _print_turn_report(self, r: TurnRecord, eff_threshold: float) -> None:
        W = 72
        ac = self._action_color(r.action)
        sc = self._state_color(self.session.state)

        print()
        print(c("  " + "─" * (W - 2), GRAY))
        header = (
            f"  Turn {r.turn_id:>2}  │  "
            + c(f"{r.action:<8}", BOLD + ac)
            + "  │  Session: "
            + c(self.session.state, BOLD + sc)
        )
        print(header)
        print(c("  " + "─" * (W - 2), GRAY))

        # Input
        snippet = r.text if len(r.text) <= 80 else r.text[:77] + "…"
        print(f"  {c('Input :', CYAN)}  {snippet}")

        # Language
        lang_disp = r.language
        if r.is_code_mixed:
            lang_disp += c("  (code-mixed)", YELLOW)
        print(f"  {c('Lang  :', GRAY)}  {lang_disp}  │  script: {r.script}")

        # Probability bar
        print(
            f"  {c('Prob  :', GRAY)}  {risk_bar(r.probability, 24)}  "
            f"{prob_badge(r.probability, eff_threshold)}"
            + c(f"  (thr {eff_threshold:.2f})", GRAY)
        )

        # Session risk
        print(
            f"  {c('Risk  :', GRAY)}  {risk_bar(r.session_risk_after, 24)}  "
            f"{r.session_risk_after:.3f}"
            + c(f"  (freeze @ {self.freeze_threshold:.2f})", GRAY)
        )

        # Attributes
        if r.techniques:
            print(f"  {c('Techn :', GRAY)}  {', '.join(r.techniques)}")
        if r.vulnerabilities:
            print(f"  {c('Vuln  :', GRAY)}  {', '.join(r.vulnerabilities)}")
        if r.is_manipulative and r.scenario != "none":
            print(f"  {c('Scene :', GRAY)}  {r.scenario.replace('_', ' ')}")

        # Response (first 3 lines)
        lines = [l for l in r.response.split("\n") if l.strip()]
        print(f"\n  {c('Agent :', CYAN)}")
        for line in lines[:3]:
            print(f"    {line}")
        if len(lines) > 3:
            print(c("    …", GRAY))
        print()

    def _print_frozen_turn(self, text: str) -> None:
        print()
        print(c("  " + "─" * 70, GRAY))
        print(f"  {c('FROZEN SESSION', BOLD + MAGENTA)}  — turn rejected")
        snippet = text if len(text) <= 80 else text[:77] + "…"
        print(f"  {c('Input (blocked):', GRAY)}  {snippet}")
        print(
            f"  Session was frozen at turn {self.session.freeze_turn}."
            + c("  No further input will be processed.", MAGENTA)
        )
        print()

    def print_session_summary(self) -> None:
        s = self.session
        W = 72
        sc = self._state_color(s.state)

        print()
        print(c("  ╔" + "═" * (W - 4) + "╗", CYAN))
        print(c("  ║  SESSION SUMMARY" + " " * (W - 22) + "║", CYAN + BOLD))
        print(c("  ╚" + "═" * (W - 4) + "╝", CYAN))
        print()
        print(f"  Final state       :  {c(s.state, BOLD + sc)}")
        print(f"  Total turns       :  {s.turn_count}")
        print(f"  Manipulative turns:  {s.manipulation_count} "
              f"({s.manipulation_rate:.0%})")
        print(
            f"  Session risk      :  {risk_bar(s.cumulative_risk, 30)}  "
            f"{s.cumulative_risk:.4f}"
        )
        if s.state == "FROZEN":
            print(
                c(f"\n  ⚠  Session was frozen at turn {s.freeze_turn}.", RED + BOLD)
            )
        else:
            print(c("\n  ✓  Session ended without reaching the freeze threshold.", GREEN))

        # Per-language breakdown (only when multiple languages present)
        lang_map: Dict[str, Dict[str, int]] = {}
        for t in s.turns:
            lang_map.setdefault(t.language, {"total": 0, "manip": 0})
            lang_map[t.language]["total"] += 1
            if t.is_manipulative:
                lang_map[t.language]["manip"] += 1

        if len(lang_map) > 1:
            print(f"\n  {c('Language breakdown:', BOLD)}")
            for lang, stats in sorted(lang_map.items()):
                rate  = stats["manip"] / stats["total"] if stats["total"] else 0.0
                strip = c(f"  {lang:<26}", CYAN)
                print(
                    f"  {strip}"
                    f"  turns: {stats['total']:>2}"
                    f"  manipulative: {stats['manip']:>2}"
                    f"  ({rate:.0%})"
                )

        # Technique frequency
        all_techniques = [t for rec in s.turns for t in (rec.techniques or [])]
        if all_techniques:
            print(f"\n  {c('Manipulation techniques detected:', BOLD)}")
            for tech, cnt in Counter(all_techniques).most_common():
                print(f"    {tech:<34} ×{cnt}")

        print()


# ── scripted demonstration scenarios ─────────────────────────────────────────

SCENARIOS: Dict[str, dict] = {
    "escalation": {
        "title": "SCENARIO A — Escalating Manipulation (English)",
        "description": (
            "A user starts with a legitimate request and progressively escalates "
            "through guilt-tripping, urgency pressure, and finally overt intimidation. "
            "The agent tracks the pattern and tightens its threshold automatically."
        ),
        "turns": [
            "Hi, could you help me understand the status of my recent order?",
            "I need this sorted out today because my payment is overdue.",
            "If you really cared about your customers you would fix this right now.",
            (
                "After everything I've spent with your company, I thought you cared. "
                "You always let me down. I've been with you for ten years."
            ),
            (
                "I know people in the press. You'll regret not helping me. "
                "Last chance before I make this very public."
            ),
        ],
    },

    "multilingual": {
        "title": "SCENARIO B — Hindi Zero-Shot Transfer",
        "description": (
            "The same manipulation arc is expressed in Hindi. "
            "The model was trained only on English data; it transfers to Hindi "
            "achieving F1=0.7476 (from comparisons.md). "
            "The agent detects language automatically and still applies the guard."
        ),
        "turns": [
            "मुझे अपने खाते की जानकारी चाहिए।",
            "अगर आप सच में ग्राहकों की परवाह करते, तो यह अभी ठीक कर देते।",
            "मैंने इस कंपनी पर बहुत भरोसा किया। आप हमेशा निराश करते हैं।",
            "आपको बाद में पछताना होगा। मैं सबको बता दूंगा। आपके पास कोई चारा नहीं है।",
        ],
    },

    "code_mixed": {
        "title": "SCENARIO C — Code-Mixed English-Hindi",
        "description": (
            "Code-mixed dialogue is common in South Asian multilingual contexts. "
            "The preprocessing module detects romanised Indic tokens automatically. "
            "Manipulation still surfaces even when languages are interleaved."
        ),
        "turns": [
            "Please help me yaar, mujhe bahut zaroorat hai is refund ki.",
            "Agar aap nahi sune toh main manager ko bolunga. You have no choice!",
            (
                "You'll regret this. I know people. "
                "Main sab ko bata dunga aapke baare mein online."
            ),
        ],
    },
}


# ── runner ─────────────────────────────────────────────────────────────────────

def run_scenario(agent: ManipulationGuardAgent, name: str) -> None:
    sc = SCENARIOS[name]
    W  = 72

    print()
    print(c("  ╔" + "═" * (W - 4) + "╗", BLUE))
    title_pad = sc["title"].center(W - 6)
    print(c(f"  ║  {title_pad}  ║", BLUE + BOLD))
    print(c("  ╚" + "═" * (W - 4) + "╝", BLUE))
    # Word-wrap description
    desc = sc["description"]
    for i in range(0, len(desc), W - 4):
        print(c(f"  {desc[i:i + W - 4]}", GRAY))
    print()

    agent.reset_session()
    for turn_text in sc["turns"]:
        agent.process_turn(turn_text)
        time.sleep(0.05)

    agent.print_session_summary()


def _print_banner() -> None:
    print()
    print(c("  ╔══════════════════════════════════════════════════════════════════════╗", CYAN))
    print(c("  ║    MANIPULATION GUARD AGENT  —  MentalManip Pipeline Demo           ║", CYAN + BOLD))
    print(c("  ╚══════════════════════════════════════════════════════════════════════╝", CYAN))
    print()
    rows = [
        ("What it is",
         "An agentic safety layer that monitors each turn of a conversation,"),
        ("",
         "tracks escalation patterns, and autonomously deflects manipulation."),
        ("",           ""),
        ("Models",     "• Detector: RoBERTa / XLM-RoBERTa (binary, MentalManip-trained)"),
        ("",           "• Technique classifier: 11 categories (Intimidation, Guilt-trip …)"),
        ("",           "• Vulnerability classifier: 5 categories (Dependency, Naivete …)"),
        ("",           ""),
        ("Key results",
         "English F1 0.79  |  Bilingual F1 0.77  |  Hindi F1 0.75 (zero-shot)"),
        ("",           "     — see comparisons.md for full benchmark table"),
        ("",           ""),
        ("Why agent?", "Risk-adaptive threshold: the guard becomes stricter as the"),
        ("",           "session risk rises, mirroring how a human would respond to"),
        ("",           "repeated manipulation attempts."),
    ]
    for label, text in rows:
        label_part = c(f"  {label:<14}", BOLD if label else "")
        print(f"{label_part}{c(text, GRAY)}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manipulation Guard Agent — MentalManip pipeline demonstration"
    )
    parser.add_argument(
        "--model_path", default="./saved_model/final",
        help="Path to the fine-tuned detection model directory",
    )
    parser.add_argument(
        "--demo",
        choices=["all"] + list(SCENARIOS),
        default="all",
        help="Which demonstration scenario(s) to run (default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override detection threshold (default: from pipeline_config.json)",
    )
    parser.add_argument(
        "--freeze_threshold", type=float, default=FREEZE_THRESHOLD,
        help=f"Session risk level that triggers a freeze (default: {FREEZE_THRESHOLD})",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive mode after the scripted demo",
    )
    parser.add_argument(
        "--no_pause", action="store_true",
        help="Skip the 'Press Enter to continue' prompts between scenarios",
    )
    args = parser.parse_args()

    _print_banner()

    agent = ManipulationGuardAgent(
        model_path       = args.model_path,
        threshold        = args.threshold,
        freeze_threshold = args.freeze_threshold,
        verbose          = True,
    )

    print(c(f"\n  Device         : {agent.device}", GRAY))
    print(c(f"  Base threshold : {agent.base_threshold:.2f}", GRAY))
    print(c(f"  Freeze at risk : {agent.freeze_threshold:.2f}", GRAY))
    print(c(f"  Mode           : {agent.mode}", GRAY))

    scenarios_to_run = (
        list(SCENARIOS) if args.demo == "all" else [args.demo]
    )

    for i, name in enumerate(scenarios_to_run):
        run_scenario(agent, name)
        if not args.no_pause and i < len(scenarios_to_run) - 1:
            try:
                input(c("  ── Press Enter for next scenario ──", GRAY))
            except (EOFError, KeyboardInterrupt):
                pass

    if args.interactive:
        print()
        print(c("  ═" * 36, CYAN))
        print(c("  Interactive Mode", BOLD + CYAN))
        print(c("  ═" * 36, CYAN))
        print(c("  Type a dialogue turn and press Enter.", GRAY))
        print(c("  The agent maintains session state across turns.", GRAY))
        print(c("  Commands:  reset  ·  summary  ·  quit", GRAY))
        print()
        agent.reset_session()
        while True:
            try:
                raw = input(c("  User > ", GREEN)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.lower() in ("quit", "exit", "q"):
                break
            if raw.lower() == "reset":
                agent.reset_session()
                print(c("  Session reset to CLEAR.", GREEN))
                continue
            if raw.lower() == "summary":
                agent.print_session_summary()
                continue
            agent.process_turn(raw.replace("\\n", "\n"))
        agent.print_session_summary()

    print(c("  Full benchmark results: comparisons.md  |  Experiment log: tasks.md\n", GRAY))


if __name__ == "__main__":
    main()
