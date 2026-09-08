from __future__ import annotations

import json
import re

from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"^```(json)?|```$", re.MULTILINE)

_LINE_RE = re.compile(
    r'^(\s*"(?:[^"\\]|\\.)*"\s*:)'                       
    r'(?!\s*(?:["\[{]|-?\d|true\b|false\b|null\b))'
    r'\s*(.*?)'   
    r'(,?\s*)$'   
)


def _quote_bare_value_line(match: re.Match) -> str:
    prefix, value, suffix = match.group(1), match.group(2), match.group(3)
    value = value.strip()
    if not value:
        return match.group(0)
    escaped = value.replace('"', '\\"')
    return f'{prefix}"{escaped}"{suffix}'


def _repair_unquoted_values(raw: str) -> str:
    lines = raw.split("\n")
    repaired_lines = [_LINE_RE.sub(_quote_bare_value_line, line) for line in lines]
    return "\n".join(repaired_lines)


def parse_json_loose(raw: str) -> dict | list | None:
    """Strip code fences, try strict json.loads, then try a repair pass."""
    text = _FENCE_RE.sub("", raw).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    repaired = _repair_unquoted_values(text)
    try:
        parsed = json.loads(repaired)
        logger.warning("Recovered malformed JSON via repair pass")
        return parsed
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM output as JSON even after repair: %.300s", text)
        return None
