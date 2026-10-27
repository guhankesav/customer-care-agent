"""Redact PII from customer text before it reaches a model or a log."""
import re

PATTERNS = [
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("PHONE", re.compile(r"(?:\+?1[ -.]?)?\(?\b\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b")),
]


def redact(text):
    """Return (redacted_text, vault) where vault maps tokens back to values."""
    vault = {}
    for label, pat in PATTERNS:
        def sub(m, label=label):
            token = f"[{label}_{len(vault) + 1}]"
            vault[token] = m.group(0)
            return token
        text = pat.sub(sub, text)
    return text, vault


def digits(value):
    return re.sub(r"\D", "", value)[-10:]
