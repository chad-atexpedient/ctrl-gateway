"""Translation transcoder.

Pattern adapted from universal_translation_filter: when a request asks
"translate X to language Y", handle the translation specially. We can
detect via regex in the router and route to a translation-tuned endpoint
or just pass through with a translation system prompt.

Three modes (configurable via gateway-policy.json -> translation):
  - off: no special handling
  - detect_only: regex-detect and tag the decision, but don't change routing
  - rewrite: rewrite the system prompt to be a translator; tag the decision
  - dedicated_endpoint: route to a translation-specific tier if defined

Why this is useful:
  - Translation is a constrained task; a small translation-tuned model
    can outperform a general LLM at much lower cost.
  - The Universal Translation Filter pattern translates all messages,
    but that's wasteful for non-translation queries. We detect on intent
    and only invoke translation mode when appropriate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Regex patterns for detecting "translate X to Y"
TRANSLATE_PATTERNS = [
    re.compile(r"\btranslate\s+(?:this|the following|it|that|.+?)?\s*(?:to|into)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE),
    re.compile(r"\bin\s+(english|spanish|french|german|italian|portuguese|chinese|japanese|korean|russian|arabic|hindi|dutch|polish|swedish|norwegian|finnish)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:do\s+(?:you|i|we)\s+)?say\s+[\"']?(.+?)[\"']?\s+in\s+(\w+)", re.IGNORECASE),
    re.compile(r"\b翻译\s*[:：]\s*(.+)", re.IGNORECASE),
    re.compile(r"\btradu[ct]e[zr]?\s+(?:a\s+)?(.+?)\s+(?:al|en|au|auf|in|into|to)\s+(\w+)", re.IGNORECASE),
]

LANGUAGE_NAMES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "chinese": "zh", "japanese": "ja",
    "korean": "ko", "russian": "ru", "arabic": "ar", "hindi": "hi",
    "dutch": "nl", "polish": "pl", "swedish": "sv", "norwegian": "no",
    "finnish": "fi", "turkish": "tr", "vietnamese": "vi", "thai": "th",
}


@dataclass
class TranslationIntent:
    is_translation: bool
    target_language: str | None = None
    source_language: str | None = None
    source_text: str | None = None
    pattern_matched: str | None = None


def detect_intent(text: str) -> TranslationIntent:
    """Detect if the user is asking for a translation."""
    for p in TRANSLATE_PATTERNS:
        m = p.search(text)
        if m:
            # Try to extract target language
            for grp in m.groups():
                if grp:
                    grp_l = grp.lower().strip()
                    if grp_l in LANGUAGE_NAMES:
                        return TranslationIntent(
                            is_translation=True,
                            target_language=LANGUAGE_NAMES[grp_l],
                            pattern_matched=p.pattern,
                            source_text=text,
                        )
            return TranslationIntent(
                is_translation=True,
                pattern_matched=p.pattern,
                source_text=text,
            )
    return TranslationIntent(is_translation=False, source_text=text)


TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {target_language}. "
    "Output ONLY the translation — no explanations, no extra text, no quotation marks "
    "unless they appear in the original."
)


def build_translation_rewrite(
    intent: TranslationIntent,
    target_language: str | None = None,
) -> dict | None:
    """If intent is translation, return a payload override that adds a translation system prompt."""
    if not intent.is_translation:
        return None
    target = target_language or intent.target_language or "en"
    sys_prompt = TRANSLATION_SYSTEM_PROMPT.format(target_language=_lang_name(target))
    return {
        "system_prompt_override": sys_prompt,
        "translation_mode": True,
        "target_language": target,
    }


def _lang_name(code: str) -> str:
    inv = {v: k for k, v in LANGUAGE_NAMES.items()}
    return inv.get(code, code)


def apply_to_payload(payload: dict, override: dict) -> dict:
    """Insert translation system prompt into messages."""
    if not override:
        return payload
    new_payload = dict(payload)
    messages = list(new_payload.get("messages", []))
    sys_prompt = override["system_prompt_override"]
    # Replace or insert system message
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": sys_prompt + "\n\n" + messages[0].get("content", "")}
    else:
        messages.insert(0, {"role": "system", "content": sys_prompt})
    new_payload["messages"] = messages
    return new_payload
