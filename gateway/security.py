"""Prompt-injection detection and sanitization.

Two goals:
  1. Detect injection attempts in user prompts (so we can log + alert).
  2. NEVER trust the injection — never alter routing based on it.

Patterns are configurable via gateway-policy.json -> security.injection_regex.
Default list catches common prompt-injection patterns.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class InjectionCheckResult:
    has_injection_signal: bool
    matched_patterns: list[str]
    sanitized_text: str


def compile_patterns(regexes: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(r) for r in regexes]


def check_injection(
    text: str,
    patterns: list[re.Pattern],
    strip_control_tokens: bool = True,
) -> InjectionCheckResult:
    """Check text against injection patterns. Returns matched + sanitized.

    Sanitization rules:
      - Strip null bytes and other control chars (except \\n\\r\\t).
      - DO NOT strip matched-injection phrases (we want to flag them, not
        erase them — but routing is independent of whether injection matched).
    """
    matched = []
    for p in patterns:
        m = p.search(text)
        if m:
            matched.append(p.pattern)

    sanitized = text
    if strip_control_tokens:
        sanitized = _strip_control_tokens(sanitized)

    return InjectionCheckResult(
        has_injection_signal=bool(matched),
        matched_patterns=matched,
        sanitized_text=sanitized,
    )


def _strip_control_tokens(text: str) -> str:
    """Strip control chars but keep common whitespace."""
    return "".join(
        c for c in text
        if c in ("\n", "\r", "\t") or (ord(c) >= 0x20 and ord(c) != 0x7F)
    )
