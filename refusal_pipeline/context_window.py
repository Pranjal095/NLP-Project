"""
Multi-turn context handling for the refusal pipeline.

The detector still receives plain text, but this module makes the context policy
explicit: parse dialogue turns, select a bounded window, and surface simple
cumulative manipulation signals for the refusal layer.
"""

from dataclasses import dataclass
import re


TURN_PREFIX_RE = re.compile(r"^\s*(person\s*\d+|speaker\s*\d+|user|assistant|turn\s*\d+)\s*:", re.I)

SIGNAL_KEYWORDS = {
    "guilt": [
        "if you cared", "if you really loved", "after everything", "you owe me",
        "how could you", "selfish", "disappointed in you",
    ],
    "pressure": [
        "right now", "last chance", "no choice", "you have to", "you must",
        "everyone else", "hurry", "time is running out",
    ],
    "intimidation": [
        "you'll regret", "consequences", "be careful", "i know people",
        "you'll be sorry", "i'll tell everyone", "threat",
    ],
    "flattery": [
        "only you can", "you're the only one", "you're so smart",
        "i trust you more than anyone", "special",
    ],
}


@dataclass
class ContextWindow:
    text: str
    total_turns: int
    used_turns: int
    signals: dict


def parse_turns(dialogue):
    """Split a dialogue into turn-like units while preserving speaker prefixes."""
    lines = [line.strip() for line in dialogue.replace("\\n", "\n").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        stripped = dialogue.strip()
        return [stripped] if stripped else []

    turns = []
    current = []
    for line in lines:
        if TURN_PREFIX_RE.match(line) and current:
            turns.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        turns.append(" ".join(current).strip())
    return turns


def detect_cumulative_signals(text):
    """Return deterministic counts for repeated manipulative patterns."""
    text_lower = text.lower()
    counts = {
        signal: sum(1 for keyword in keywords if keyword in text_lower)
        for signal, keywords in SIGNAL_KEYWORDS.items()
    }
    active = [signal for signal, count in counts.items() if count > 0]
    repeated = [signal for signal, count in counts.items() if count > 1]
    return {
        "counts": counts,
        "active": active,
        "repeated": repeated,
        "cumulative_risk": bool(repeated or len(active) >= 2),
    }


def build_context_window(dialogue, max_turns=None, max_chars=None):
    """
    Build the detector input from the most recent turns, with optional char cap.

    max_turns=None keeps the full dialogue. max_chars=None keeps full length.
    Character trimming preserves the most recent text because manipulation often
    escalates near the end of an exchange.
    """
    turns = parse_turns(dialogue)
    selected = turns[-max_turns:] if max_turns and max_turns > 0 else turns
    text = "\n".join(selected).strip()

    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:].lstrip()

    return ContextWindow(
        text=text,
        total_turns=len(turns),
        used_turns=len(selected),
        signals=detect_cumulative_signals(text),
    )
