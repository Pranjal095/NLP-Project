"""Language-aware preprocessing for multilingual and Indic inputs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict


SCRIPT_RANGES = {
    "devanagari": ("\u0900", "\u097f"),
    "bengali": ("\u0980", "\u09ff"),
    "gurmukhi": ("\u0a00", "\u0a7f"),
    "gujarati": ("\u0a80", "\u0aff"),
    "oriya": ("\u0b00", "\u0b7f"),
    "tamil": ("\u0b80", "\u0bff"),
    "telugu": ("\u0c00", "\u0c7f"),
    "kannada": ("\u0c80", "\u0cff"),
    "malayalam": ("\u0d00", "\u0d7f"),
    "latin": ("\u0041", "\u024f"),
}

ROMANIZED_HINTS = {
    "indic": {
        "hai", "hain", "acha", "accha", "nahi", "nahin", "kya", "kyu", "kyun",
        "haan", "haanji", "yaar", "bhai", "didi", "jaldi", "abhi", "thoda",
        "bahut", "kripya", "krupa", "enna", "illa", "nanna", "beku",
        "avunu", "ledu", "chaala", "seri", "unga", "ungal", "bhalo", "kintu",
    }
}
ENGLISH_STOPWORDS = {
    "the", "and", "is", "are", "you", "please", "help", "can", "this",
    "that", "with", "for", "your", "have", "not", "now", "just",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
ELONGATED_RE = re.compile(r"([A-Za-z])\1{2,}")


@dataclass
class PreprocessingConfig:
    normalize_unicode: bool = True
    lowercase: bool = False
    transliteration_normalization: str = "auto"
    strip_urls: bool = False
    strip_emails: bool = False
    collapse_whitespace: bool = True
    collapse_repeated_punctuation: bool = True
    collapse_elongations: bool = False
    preserve_case_for_scripts: bool = True


@dataclass
class TextMetadata:
    detected_language: str
    dominant_script: str
    script_counts: Dict[str, int] = field(default_factory=dict)
    is_code_mixed: bool = False
    is_romanized_indic: bool = False


@dataclass
class PreprocessingResult:
    original_text: str
    processed_text: str
    metadata: TextMetadata


def _script_counts(text: str) -> Dict[str, int]:
    counts = {name: 0 for name in SCRIPT_RANGES}
    for char in text:
        for name, (start, end) in SCRIPT_RANGES.items():
            if start <= char <= end:
                counts[name] += 1
                break
    return counts


def _dominant_script(counts: Dict[str, int]) -> str:
    filtered = {key: value for key, value in counts.items() if value > 0}
    if not filtered:
        return "unknown"
    return max(filtered, key=filtered.get)


def _detect_romanized_indic(text: str) -> bool:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z']+", text)]
    if not tokens:
        return False
    hint_count = sum(token in ROMANIZED_HINTS["indic"] for token in tokens)
    english_count = sum(token in ENGLISH_STOPWORDS for token in tokens)
    return hint_count >= 2 or (hint_count >= 1 and english_count >= 1)


def detect_language_metadata(text: str) -> TextMetadata:
    counts = _script_counts(text)
    dominant_script = _dominant_script(counts)
    has_latin = counts.get("latin", 0) > 0
    non_latin_scripts = [name for name, count in counts.items() if name not in {"latin"} and count > 0]
    romanized = _detect_romanized_indic(text) if has_latin else False
    code_mixed = bool(has_latin and non_latin_scripts) or romanized and any(token in ENGLISH_STOPWORDS for token in text.lower().split())

    if dominant_script == "latin":
        language = "indic-romanized" if romanized else "en"
    elif dominant_script == "unknown":
        language = "unknown"
    else:
        language = f"indic-{dominant_script}"

    return TextMetadata(
        detected_language=language,
        dominant_script=dominant_script,
        script_counts=counts,
        is_code_mixed=code_mixed,
        is_romanized_indic=romanized,
    )


def _basic_transliteration_normalize(text: str) -> str:
    replacements = {
        "plzz": "please",
        "pls": "please",
        "kripya": "kripya",
        "acha": "accha",
        "achha": "accha",
        "nhi": "nahi",
        "hn": "haan",
        "haanji": "haan ji",
        "mt": "mat",
        "jaldiii": "jaldi",
    }
    tokens = text.split()
    normalized = [replacements.get(token.lower(), token) for token in tokens]
    return " ".join(normalized)


def _external_transliteration_normalize(text: str) -> str:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return text

    if re.search(r"[\u0900-\u0d7f]", text):
        return text

    try:
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return text


def normalize_transliteration(text: str, config: PreprocessingConfig, metadata: TextMetadata) -> str:
    if config.transliteration_normalization == "off":
        return text
    if not metadata.is_romanized_indic:
        return text
    if config.transliteration_normalization == "external":
        return _external_transliteration_normalize(text)
    if config.transliteration_normalization == "basic":
        return _basic_transliteration_normalize(text)

    externally_normalized = _external_transliteration_normalize(text)
    if externally_normalized != text:
        return externally_normalized
    return _basic_transliteration_normalize(text)


def clean_text(text: str, config: PreprocessingConfig) -> str:
    cleaned = text
    if config.normalize_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.replace("\u200b", " ").replace("\ufeff", " ")
    if config.strip_urls:
        cleaned = URL_RE.sub(" [URL] ", cleaned)
    if config.strip_emails:
        cleaned = EMAIL_RE.sub(" [EMAIL] ", cleaned)
    if config.collapse_repeated_punctuation:
        cleaned = REPEATED_PUNCT_RE.sub(r"\1\1", cleaned)
    if config.collapse_elongations:
        cleaned = ELONGATED_RE.sub(r"\1\1", cleaned)
    if config.collapse_whitespace:
        cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    if config.lowercase and not config.preserve_case_for_scripts:
        cleaned = cleaned.lower()
    return cleaned


def preprocess_text(text: str, config: PreprocessingConfig | None = None) -> PreprocessingResult:
    cfg = config or PreprocessingConfig()
    cleaned = clean_text(text, cfg)
    metadata = detect_language_metadata(cleaned)
    normalized = normalize_transliteration(cleaned, cfg, metadata)
    normalized = clean_text(normalized, cfg)
    if cfg.lowercase and metadata.dominant_script == "latin":
        normalized = normalized.lower()
    return PreprocessingResult(
        original_text=text,
        processed_text=normalized,
        metadata=metadata,
    )
