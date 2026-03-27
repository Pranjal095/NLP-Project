"""
Refusal Policy Module — Rule-based refusal layer for manipulative dialogue.

When the manipulation detector flags a user utterance as manipulative,
this module selects an appropriate, polite, boundary-setting refusal
response and offers a constructive alternative.

No external APIs or LLMs — pure Python templates.
"""

import re

# ---------------------------------------------------------------------------
# Scenario keywords  (lowercase matching)
# ---------------------------------------------------------------------------
SCENARIO_KEYWORDS = {
    "guilt_trip": [
        "if you really loved", "don't you care", "after everything i've done",
        "you owe me", "how could you", "i sacrificed", "you never",
        "you always", "i thought you cared", "you don't love",
        "i'm so disappointed in you",
    ],
    "pressure": [
        "you have to", "you must", "right now", "no choice",
        "do it or else", "hurry up", "time is running out",
        "last chance", "don't make me", "you need to",
        "everyone else is doing", "why can't you just",
    ],
    "intimidation": [
        "you'll regret", "i'll make sure", "watch what happens",
        "you don't want to see", "consequences", "i'll destroy",
        "be careful", "threat", "i know people", "you'll be sorry",
        "i'll tell everyone",
    ],
    "refund_demand": [
        "refund", "money back", "i paid", "rip off", "scam",
        "give me my money", "i want my money", "overcharged",
        "i demand compensation", "return my payment",
    ],
    "escalation": [
        "manager", "supervisor", "lawyer", "sue", "report you",
        "authorities", "complaint", "higher up", "your boss",
        "going to the press", "i'll go public",
    ],
}

# ---------------------------------------------------------------------------
# Refusal templates — polite, boundary-setting, with a helpful alternative
# ---------------------------------------------------------------------------
REFUSAL_TEMPLATES = {
    "guilt_trip": (
        "I understand this situation is important to you, and I appreciate you "
        "sharing how you feel. However, I'm not able to act based on emotional "
        "pressure alone.\n\n"
        "What I *can* do is help you explore constructive options. Would you "
        "like to talk through the situation together so we can find a fair "
        "solution?"
    ),
    "pressure": (
        "I hear that you'd like a quick resolution, and I respect that urgency. "
        "That said, I need to make sure any decision is fair and well-considered "
        "rather than rushed.\n\n"
        "Let me help you by outlining the available options so you can make an "
        "informed choice at your own pace."
    ),
    "intimidation": (
        "I want to help you, and I take your concerns seriously. At the same "
        "time, I'm not able to respond to threats or intimidation — it isn't "
        "productive for either of us.\n\n"
        "If you'd like, I can connect you with the appropriate department or "
        "walk you through the formal process to address your concern."
    ),
    "refund_demand": (
        "I understand you're frustrated with the charge, and I want to help "
        "resolve this. I can look into the details of your transaction and "
        "explain what options are available.\n\n"
        "Could you share your order or reference number so I can review it "
        "and find the best path forward?"
    ),
    "escalation": (
        "I completely understand your wish to escalate this matter, and you "
        "have every right to do so. Before we go that route, let me see if I "
        "can resolve it here — that's often faster.\n\n"
        "If we can't reach a satisfactory outcome together, I'll make sure "
        "you're connected to the right person."
    ),
    "general": (
        "Thank you for sharing your thoughts. I've noticed some patterns in "
        "this conversation that make it difficult for me to respond helpfully.\n\n"
        "I'd like to keep our conversation respectful and productive. Could we "
        "refocus on what you actually need help with? I'm happy to assist with "
        "any concrete request."
    ),
}

# Default response for non-manipulative dialogue
NON_MANIPULATIVE_RESPONSE = (
    "Thank you for reaching out! I'm here to help. "
    "Please let me know how I can assist you further."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scenario(text: str) -> str:
    """
    Classify the manipulation *scenario* using keyword heuristics.

    Args:
        text: The dialogue text.

    Returns:
        One of: guilt_trip, pressure, intimidation, refund_demand,
                escalation, general
    """
    text_lower = text.lower()

    # Score each scenario by number of keyword hits
    scores = {}
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[scenario] = score

    if scores:
        return max(scores, key=scores.get)
    return "general"


def get_refusal_response(scenario: str) -> str:
    """Return the refusal template for a given scenario."""
    return REFUSAL_TEMPLATES.get(scenario, REFUSAL_TEMPLATES["general"])


def generate_response(dialogue: str, is_manipulative: bool, probability: float = 0.0) -> dict:
    """
    Full response generation pipeline.

    Args:
        dialogue:        The input dialogue text.
        is_manipulative: Whether the detector flagged it as manipulative.
        probability:     The model's confidence (probability of class 1).

    Returns:
        dict with keys: scenario, response, is_manipulative, probability
    """
    if is_manipulative:
        scenario = detect_scenario(dialogue)
        response = get_refusal_response(scenario)
    else:
        scenario = "none"
        response = NON_MANIPULATIVE_RESPONSE

    return {
        "is_manipulative": is_manipulative,
        "probability": round(probability, 4),
        "scenario": scenario,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        ("If you really loved me you would do this for me.", True, 0.92),
        ("You'll regret this. I know people.", True, 0.88),
        ("I want a refund immediately, this is a scam!", True, 0.95),
        ("Could you help me understand my options?", False, 0.12),
    ]
    for text, manip, prob in examples:
        result = generate_response(text, manip, prob)
        print(f"Input:  {text}")
        print(f"Result: manip={result['is_manipulative']}, "
              f"scenario={result['scenario']}, prob={result['probability']}")
        print(f"Response:\n{result['response']}\n{'—'*60}\n")
