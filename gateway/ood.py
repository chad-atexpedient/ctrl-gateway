"""Out-of-distribution detector.

OOD is detected when the router's max vertical probability is below a
configured threshold. OOD queries are escalated to a safe tier (tier3 by
default) instead of being routed through cost arithmetic — we don't trust
the embedding's classification on something it's never seen.

The OOD score is also logged so the flywheel can flag samples that may
represent a new vertical worth adding to taxonomy.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OODResult:
    is_ood: bool
    score: float  # 0-1; higher = more OOD
    max_prob: float
    top_vertical: str | None
    threshold: float


def detect(
    vertical_top2: list[tuple[str, float]],
    threshold: float,
) -> OODResult:
    """Decide if a router prediction is out-of-distribution.

    Args:
        vertical_top2: list of (vertical_name, probability) sorted desc, len >= 1
        threshold: max prob below this -> OOD. Configurable via gateway-policy.json.

    Returns:
        OODResult
    """
    if not vertical_top2:
        return OODResult(is_ood=True, score=1.0, max_prob=0.0, top_vertical=None, threshold=threshold)
    top_vertical, top_prob = vertical_top2[0]
    is_ood = top_prob < threshold
    score = max(0.0, min(1.0, 1.0 - top_prob))
    return OODResult(
        is_ood=is_ood,
        score=score,
        max_prob=top_prob,
        top_vertical=top_vertical,
        threshold=threshold,
    )
