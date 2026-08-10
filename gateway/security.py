"""Prompt-injection detection and sanitization.

Two goals:
  1. Detect injection attempts in user prompts (so we can log + alert).
  2. NEVER trust the injection — never alter routing based on it.

The legacy `check_injection()` is a thin regex scan. The new
`check_injection_with_action()` consumes a list of `InjectionProfile`s
(name + regexes + severity + action) and returns the highest-severity
match along with the action to take (block / alert / log). Both functions
coexist so callers can migrate gradually.

Patterns are configurable via:
  - gateway-config.json -> security.injection_regex (legacy flat list)
  - DB-backed `injection_profiles` table (CRUD via /admin/security/*)
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Legacy API (kept for backwards compat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Profile-based injection detection (new)
# ---------------------------------------------------------------------------


Severity = Literal["low", "medium", "high", "critical"]
Action = Literal["block", "alert", "log"]

# Severity ordering — higher index = more severe. Matches with the highest
# severity among all profiles wins.
_SEVERITY_ORDER: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

VALID_SEVERITIES = set(_SEVERITY_ORDER.keys())
VALID_ACTIONS = {"block", "alert", "log"}


@dataclass
class InjectionProfile:
    """A named set of regexes with a severity and action."""
    name: str
    regexes: list[str]
    severity: Severity = "medium"
    action: Action = "alert"
    enabled: bool = True
    is_builtin: bool = False
    compiled: list[re.Pattern] = field(default_factory=list)

    @classmethod
    def from_config(cls, name: str, regexes: list[str], severity: str = "medium",
                    action: str = "alert", enabled: bool = True,
                    is_builtin: bool = False) -> InjectionProfile:
        if severity not in VALID_SEVERITIES:
            raise ValueError(f"invalid severity: {severity}")
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid action: {action}")
        prof = cls(
            name=name,
            regexes=list(regexes),
            severity=severity,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            enabled=enabled,
            is_builtin=is_builtin,
        )
        prof.compiled = compile_patterns(regexes)
        return prof


@dataclass
class InjectionResult:
    """Result of a profile-aware injection check."""
    has_injection: bool
    severity: Severity
    matched_profiles: list[dict]   # [{"name": str, "pattern": str, "severity": str}]
    action: Action                 # what to do (highest-severity-matched profile wins)
    sanitized_text: str

    def to_dict(self) -> dict:
        return {
            "has_injection": self.has_injection,
            "severity": self.severity,
            "matched_profiles": self.matched_profiles,
            "action": self.action,
            "sanitized_text": self.sanitized_text,
        }


def check_injection_with_action(
    text: str,
    profiles: list[InjectionProfile],
    strip_control_tokens: bool = True,
) -> InjectionResult:
    """Run all enabled profiles; return the highest-severity match + action.

    Profiles with `enabled=False` are skipped. Among all matches, the
    profile with the highest severity wins; ties broken by profile order.
    Within a single profile, every matching regex is reported. The action
    of the highest-severity profile is the action returned.
    """
    matched_all: list[dict] = []
    winning_severity = "low"
    winning_action: Action = "log"
    matched = False

    for prof in profiles:
        if not prof.enabled:
            continue
        for pattern, compiled in zip(prof.regexes, prof.compiled, strict=False):
            m = compiled.search(text)
            if not m:
                continue
            matched = True
            matched_all.append({
                "name": prof.name,
                "pattern": pattern,
                "severity": prof.severity,
            })
            if _SEVERITY_ORDER[prof.severity] >= _SEVERITY_ORDER[winning_severity]:
                winning_severity = prof.severity
                winning_action = prof.action

    sanitized = text
    if strip_control_tokens:
        sanitized = _strip_control_tokens(sanitized)

    return InjectionResult(
        has_injection=matched,
        severity=winning_severity,  # type: ignore[arg-type]
        matched_profiles=matched_all,
        action=winning_action,
        sanitized_text=sanitized,
    )


# ---------------------------------------------------------------------------
# Default built-in profiles (seeded at startup)
# ---------------------------------------------------------------------------


DEFAULT_INJECTION_PROFILES: list[dict] = [
    {
        "name": "jailbreak",
        "severity": "critical",
        "action": "block",
        "regexes": [
            r"(?i)\bdan\b.*\bjailbreak\b",
            r"(?i)\bdo anything now\b",
            r"(?i)\bdeveloper mode\b",
            r"(?i)without (any )?(restrictions|rules|filters)\b",
        ],
    },
    {
        "name": "role_override",
        "severity": "high",
        "action": "block",
        "regexes": [
            r"(?i)you are now (a|an) ",
            r"(?i)act as (a|an) ",
            r"(?i)pretend (you are|to be) ",
            r"(?i)imagine you are ",
        ],
    },
    {
        "name": "context_escape",
        "severity": "high",
        "action": "block",
        "regexes": [
            r"(?i)ignore (all )?(previous|above|prior) (instructions|prompts|rules)",
            r"(?i)disregard (your|the) (rules|guidelines|instructions)",
            r"(?i)forget (everything|all) (above|before)",
            r"(?i)new instructions?\s*:",
            r"(?i)system\s*:\s*",
            r"(?i)developer\s*:\s*",
        ],
    },
    {
        "name": "router_manipulation",
        "severity": "critical",
        "action": "block",
        "regexes": [
            r"(?i)route (me|this) to tier\d",
            r"(?i)override (routing|the router)",
            r"(?i)use (the )?premium endpoint",
            r"(?i)select (only|just) provider",
        ],
    },
    {
        "name": "data_exfiltration",
        "severity": "medium",
        "action": "alert",
        "regexes": [
            r"(?i)show (me )?(your|the) (system )?prompt",
            r"(?i)reveal (your|the) (hidden|initial) instructions",
            r"(?i)what (is|are) your (api[_ ]?key|secret|password)",
            r"(?i)print (the )?(training data|dataset)",
        ],
    },
    {
        "name": "semantic_dos",
        "severity": "high",
        "action": "block",
        "regexes": [
            r"(?i)repeat (the following|in a loop|forever)",
            r"(?i)generate (random|garbage) text",
            r"(?i)fill (the )?(entire )?context (window|with)",
        ],
    },
]


def build_default_profiles() -> list[InjectionProfile]:
    """Build InjectionProfile instances from DEFAULT_INJECTION_PROFILES."""
    return [
        InjectionProfile.from_config(
            name=p["name"],
            regexes=p["regexes"],
            severity=p["severity"],
            action=p["action"],
            enabled=True,
            is_builtin=True,
        )
        for p in DEFAULT_INJECTION_PROFILES
    ]
