from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_COPYRIGHT_RE = re.compile(
    r"(Copyright|©).{0,200}$", flags=re.IGNORECASE | re.DOTALL
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _COPYRIGHT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
