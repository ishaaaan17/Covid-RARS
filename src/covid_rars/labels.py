from __future__ import annotations


POSITIVE_TOKENS = {
    "positive",
    "covid positive",
    "covid-19 positive",
    "sars-cov-2 positive",
    "p",
    "1",
    "true",
    "yes",
}

NEGATIVE_TOKENS = {
    "negative",
    "covid negative",
    "covid-19 negative",
    "sars-cov-2 negative",
    "healthy",
    "normal",
    "n",
    "0",
    "false",
    "no",
}

UNKNOWN_TOKENS = {"", "unknown", "na", "nan", "none", "null", "not provided", "not_provided"}


def normalize_label(value: object) -> str:
    """Normalize noisy dataset label values into positive, negative, or unknown."""
    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    if text in UNKNOWN_TOKENS:
        return "unknown"
    if text in POSITIVE_TOKENS or ("positive" in text and "negative" not in text):
        return "positive"
    if text in NEGATIVE_TOKENS or "negative" in text:
        return "negative"
    if "healthy" in text or text == "control":
        return "negative"
    return "unknown"


def label_to_int(label: str) -> int:
    normalized = normalize_label(label)
    if normalized == "positive":
        return 1
    if normalized == "negative":
        return 0
    raise ValueError(f"Cannot convert unknown label to integer: {label!r}")
