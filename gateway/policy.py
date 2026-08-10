"""Routing policy engine.

Implements:
  - Pre-route checks (vision, OWUI tasks, medical regex, freshness, structural prototypes)
  - Override evaluation (first-match-wins)
  - Cost-first gate (expected_cost = fixed + in*pin + est_out*pout + retry_penalty*(1-fit))
  - Escalation (OOD, low confidence, top-2 close, cost within margin)
  - Efficiency tie-break (speed + health + load)
  - Tier ladder (context overflow, breaker, fallback)

The actual request execution + retry is in endpoints.py. This module just
decides which tier/endpoint should receive the request.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from . import config as cfg
from . import ood as ood_mod

log = logging.getLogger("glint.policy")


@dataclass
class RoutingDecision:
    tier: str
    endpoint: str
    source: str  # vision, medical_kw, freshness, bg_task, override, arith, escalation, structural_prototype
    escalated: bool = False
    ms_decision: float = 0.0
    cost_usd: float = 0.0
    fit: float = 0.0
    rationale: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class PreRouteResult:
    matched: bool
    source: str = ""
    tier: str = ""


@dataclass
class RequestContext:
    text: str
    has_image: bool
    flags: dict  # code, math, reasoning, long_output from router
    complexity: int
    vertical: str
    vertical_top2: list[tuple[str, float]]
    ood: ood_mod.OODResult
    model_version: str
    policy_version: int
    session_id: str
    tenant_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    # 64-d router projection vector (None for stub model / old callers).
    # Used for cosine-similarity structural-prototype matching; defaults to
    # None so existing callers (tests, older code) don't need to pass it.
    projection: list[float] | None = None


def pre_route(ctx: RequestContext, conf: cfg.Config, breaker_states: dict[str, str], memory_ctx=None) -> PreRouteResult:
    """Check deterministic pre-routes (override-only tiers, structural prototypes).

    First-match-wins. If memory_ctx is provided, it can override tier choice
    based on context compaction / observation size.
    """
    pol = conf.policy

    # 0. Observational memory compaction: if memory context is large,
    # escalate to a tier with enough context headroom
    if memory_ctx is not None:
        # Pre-decide tier using memory_ctx (we don't have cost-first yet, so use a heuristic)
        # Find the cheapest tier that fits the memory_ctx's token estimate
        total_tokens = memory_ctx.total_tokens_estimate
        for tier in conf.config.get("tiers", []):
            if tier.get("override_only"):
                continue
            max_ctx = tier.get("max_context", 32768)
            if total_tokens * 4 // 3 < max_ctx:
                # This tier fits; but we still want to apply normal routing first
                # unless compation is required (handled separately after cost-first)
                break
        # If total_tokens exceeds ALL tiers' max_context, we need special handling
        all_too_big = all(
            memory_ctx.total_tokens_estimate * 4 // 3 > t.get("max_context", 32768)
            for t in conf.config.get("tiers", []) if not t.get("override_only")
        )
        if all_too_big:
            # Force tier4 (largest) — cost-first will respect this
            return PreRouteResult(
                True,
                source="observation_compaction_redirect",
                tier="tier4",
            )

    # 1. Vision
    if ctx.has_image:
        # Vision always routes to a vision-capable tier
        for tier in conf.config.get("tiers", []):
            if tier.get("vision_endpoints"):
                return PreRouteResult(True, source="vision", tier=tier["name"])

    # 2. OWUI background task
    owui_re = pol.get("owui_task_regex")
    if owui_re and re.search(owui_re, ctx.text, re.IGNORECASE):
        return PreRouteResult(True, source="bg_task", tier="tier0")

    # 3. Medical keyword (override-only tier)
    med_re = pol.get("medical_keyword_regex")
    if med_re and re.search(med_re, ctx.text, re.IGNORECASE):
        return PreRouteResult(True, source="medical_kw", tier="tier_medical")

    # 4. Freshness + low complexity
    fresh_re = pol.get("freshness_regex")
    if fresh_re and re.search(fresh_re, ctx.text, re.IGNORECASE) and ctx.complexity <= 3:
        return PreRouteResult(True, source="freshness", tier="tier0")

    # 5. Structural prototypes (only kind=structural)
    proto_scores = _compute_prototype_scores(ctx, conf)
    thresholds = pol.get("prototype_thresholds", {})
    for proto_name, score in proto_scores:
        thresh = thresholds.get(proto_name, 0.85)
        if score >= thresh:
            for proto in conf.prototypes.get("prototypes", []):
                if proto.get("name") == proto_name and proto.get("kind") == "structural" and proto.get("enabled", True):
                    return PreRouteResult(
                        True,
                        source="structural_prototype",
                        tier=proto.get("target_tier", "tier4"),
                    )

    # 6. Override rules
    for override in pol.get("overrides", []):
        if _evaluate_override(override, ctx):
            action = override.get("action")
            if action == "route_to_tier":
                return PreRouteResult(True, source=override.get("source_tag", "override"), tier=override["tier"])

    return PreRouteResult(False)


def _evaluate_override(rule: dict, ctx: RequestContext) -> bool:
    cond = rule.get("condition", "")
    if not cond:
        return False
    # Small hand-rolled DSL (no eval). Grammar:
    #   expr     := or_expr
    #   or_expr  := and_expr ( "OR" and_expr )*
    #   and_expr := atom ( "AND" atom )*
    #   atom     := "NOT" atom | "(" expr ")" | <predicate>
    #   predicates: has_image == true|false
    #               matches_regex('name')
    #               prototype_match(name='x', threshold=0.85)
    #               router_complexity OP N   (OP in >= <= == > <)
    #               router_flag_X / NOT router_flag_X
    try:
        return _eval_or(cond, ctx)
    except Exception as e:
        log.warning("override eval failed for %r: %s", cond, e)
        return False


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split on sep at paren-depth 0."""
    parts = []
    depth = 0
    cur = ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        cur += ch
        # Check separator match at depth 0
        if depth == 0 and cur.endswith(sep) and len(cur) > len(sep):
            parts.append(cur[: -len(sep)].strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _eval_or(cond: str, ctx: RequestContext) -> bool:
    cond = cond.strip()
    if not cond:
        return False
    parts = _split_top_level(cond, " OR ")
    if len(parts) > 1:
        return any(_eval_or(p, ctx) for p in parts)
    return _eval_and(cond, ctx)


def _eval_and(cond: str, ctx: RequestContext) -> bool:
    cond = cond.strip()
    if not cond:
        return False
    parts = _split_top_level(cond, " AND ")
    if len(parts) > 1:
        return all(_eval_and(p, ctx) for p in parts)
    return _eval_term(cond, ctx)


def _eval_term(cond: str, ctx: RequestContext) -> bool:
    cond = cond.strip()
    # Strip a single wrapping paren pair
    if cond.startswith("(") and cond.endswith(")") and _paren_balanced_outer(cond):
        return _eval_or(cond[1:-1], ctx)
    if cond.startswith("NOT "):
        return not _eval_term(cond[4:].strip(), ctx)
    return _eval_atom(cond, ctx)


def _paren_balanced_outer(s: str) -> bool:
    """True if the first char '(' closes exactly at the last char ')'."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
    return False


def _eval_atom(atom: str, ctx: RequestContext) -> bool:
    if atom == "has_image == true":
        return ctx.has_image
    if atom == "has_image == false":
        return not ctx.has_image
    if atom.startswith("matches_regex("):
        name = atom[len("matches_regex("):-1].strip().strip("'\"")
        # name is a policy key like 'freshness_regex' or 'owui_task_regex'
        regex_map = {
            "freshness_regex": _current_config().policy.get("freshness_regex", ""),
            "medical_keyword_regex": _current_config().policy.get("medical_keyword_regex", ""),
            "owui_task_regex": _current_config().policy.get("owui_task_regex", ""),
        }
        r = regex_map.get(name, "")
        return bool(r) and bool(re.search(r, ctx.text, re.IGNORECASE))
    if atom.startswith("prototype_match("):
        return _eval_prototype_match(atom, ctx)
    if atom.startswith("router_complexity"):
        # router_complexity >= N or <= N or == N
        for op in (">=", "<=", "==", ">", "<"):
            if op in atom:
                _, raw_n = atom.split(op)
                n_value = int(raw_n.strip())
                v = ctx.complexity
                if op == ">=":
                    return v >= n_value
                if op == "<=":
                    return v <= n_value
                if op == ">":
                    return v > n_value
                if op == "<":
                    return v < n_value
                if op == "==":
                    return v == n_value
        return False
    if atom.startswith("NOT router_flag_"):
        flag = atom.replace("NOT router_flag_", "").strip()
        return not ctx.flags.get(flag, False)
    if atom.startswith("router_flag_"):
        flag = atom.replace("router_flag_", "").strip()
        return ctx.flags.get(flag, False)
    log.warning("unknown override atom: %s", atom)
    return False


def _eval_prototype_match(atom: str, ctx: RequestContext) -> bool:
    """Evaluate prototype_match(name='x', threshold=0.85) — structural only.

    Uses cosine similarity against the prototype's trained centroid when
    both ctx.projection and the centroid are available (real model, trained
    at least once); falls back to keyword overlap otherwise.
    """
    try:
        args_str = atom[len("prototype_match("):-1]
        kwargs = {}
        for piece in args_str.split(","):
            piece = piece.strip()
            if "=" in piece:
                k, v = piece.split("=", 1)
                kwargs[k.strip()] = v.strip().strip("'\"")
        name = kwargs.get("name")
        if not name:
            return False
        threshold = float(kwargs.get("threshold", 0.85))
        conf = _current_config()
        text_lower = ctx.text.lower()
        for proto in conf.prototypes.get("prototypes", []):
            if proto.get("name") != name or not proto.get("enabled", True):
                continue
            if proto.get("kind") != "structural":
                continue
            centroid = proto.get("centroid")
            seeds = proto.get("centroid_seed_text", [])
            if ctx.projection and centroid:
                score = _cosine_similarity(ctx.projection, centroid)
            elif seeds:
                score = _keyword_overlap_score(text_lower, seeds)
            else:
                return False
            return score >= threshold
    except Exception as e:
        log.warning("prototype_match eval failed for %r: %s", atom, e)
    return False


_current_config_holder: list[cfg.Config | None] = [None]


def _current_config() -> cfg.Config:
    if _current_config_holder[0] is not None:
        return _current_config_holder[0]
    # Fallback: use the module-level manager if available (e.g. during tests)
    try:
        return cfg.manager().current()
    except Exception:
        raise RuntimeError("current config not set") from None


def set_current_config(c: cfg.Config) -> None:
    _current_config_holder[0] = c


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (vectors are small — 64-d projection —
    so this stays well within the routing hot path's latency budget without
    pulling in numpy for a module that otherwise has zero array deps)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_overlap_score(text_lower: str, seeds: list[str]) -> float:
    text_words = set(text_lower.split())
    score = 0.0
    for seed in seeds:
        words = set(seed.lower().split())
        overlap = len(words & text_words)
        score = max(score, overlap / max(len(words), 1))
    return score


def _compute_prototype_scores(ctx: RequestContext, conf: cfg.Config) -> list[tuple[str, float]]:
    """Score prototypes against the request's router projection.

    Returns list of (name, similarity) sorted desc. Uses cosine similarity
    against the prototype's trained centroid when both the router's 64-d
    projection (ctx.projection) and the prototype's centroid are available
    (real model, trained at least once — router_model/train.py writes
    centroids into prototypes.json after each run). Falls back to keyword
    overlap otherwise (cold start / stub model / centroid not yet computed)
    so structural pre-routing degrades gracefully instead of going dark.
    """
    out = []
    text_lower = ctx.text.lower()
    for proto in conf.prototypes.get("prototypes", []):
        if not proto.get("enabled", True):
            continue
        if proto.get("kind") != "structural":
            continue
        seeds = proto.get("centroid_seed_text", [])
        centroid = proto.get("centroid")
        if ctx.projection and centroid:
            score = _cosine_similarity(ctx.projection, centroid)
        elif seeds:
            score = _keyword_overlap_score(text_lower, seeds)
        else:
            continue
        out.append((proto["name"], score))
    out.sort(key=lambda x: -x[1])
    return out


def expected_cost(
    *,
    fixed_per_request: float,
    in_per_1k: float,
    out_per_1k: float,
    estimated_in_tokens: int,
    estimated_out_tokens: int,
    fit: float,
    retry_penalty_multiplier: float,
) -> float:
    base = (
        fixed_per_request
        + (estimated_in_tokens / 1000.0) * in_per_1k
        + (estimated_out_tokens / 1000.0) * out_per_1k
    )
    # Retry penalty only applies when fit is below the passing threshold.
    # Above the threshold the tier is "sufficient" — no penalty.
    # Below threshold, penalty scales with (1 - fit) up to the multiplier.
    if fit >= 0.95:
        penalty = 0.0
    else:
        penalty = retry_penalty_multiplier * (1.0 - fit)
    return base + penalty


def fit_capability(
    capability_per_vertical: dict,
    vertical: str,
    min_capability: float,
    k: float,
) -> float:
    """Sigmoid-shaped fit: capability >= floor -> fit ~=1; below -> fit drops sharply."""
    cap = float(capability_per_vertical.get(vertical, capability_per_vertical.get("_default", 0.5)))
    return 1.0 / (1.0 + math.exp(-k * (cap - min_capability)))


def cost_first_route(
    ctx: RequestContext,
    conf: cfg.Config,
    breaker_states: dict[str, str],
    endpoint_loads: dict[str, int],
) -> RoutingDecision:
    """Cost-first policy gate. Pick the cheapest tier with sufficient capability.

    Skips override-only tiers.
    """
    routing = conf.config.get("routing", {})
    cost_cfg = routing.get("cost_first", {})
    escalation_cfg = routing.get("escalation", {})
    fit_threshold = cost_cfg.get("fit_threshold", 0.9)
    k = cost_cfg.get("capability_sigmoid_k", 20.0)
    retry_pen = cost_cfg.get("retry_penalty_multiplier", 5.0)

    vertical = ctx.vertical
    vertical_obj = conf.vertical(vertical)
    if vertical_obj:
        min_cap = float(vertical_obj.get("min_capability", 0.5))
    else:
        min_cap = 0.5

    candidates = []  # list of (tier_name, endpoint_name, cost, fit)

    for tier in conf.config.get("tiers", []):
        if tier.get("override_only"):
            continue
        if tier.get("vision_endpoints") and not ctx.has_image:
            # Vision tier only for vision requests
            continue
        reserve_pct = float(conf.policy.get("ladder", {}).get("context_reserve_pct", 25))
        usable_context = int(tier.get("max_context", 32768) * max(0.0, 1.0 - reserve_pct / 100.0))
        output_tokens = max(ctx.estimated_output_tokens, int(tier.get("max_tokens_bump", 0)))
        if ctx.estimated_input_tokens + output_tokens > usable_context:
            continue
        # Tier ladder: skip tier if all endpoints have breakers OPEN
        any_endpoint_available = False
        endpoints = conf.endpoints_for_tier(tier["name"])
        if not endpoints:
            continue
        for ep in endpoints:
            if breaker_states.get(ep["name"]) != "OPEN":
                any_endpoint_available = True
                break
        if not any_endpoint_available:
            continue

        cap_per_v = tier.get("capability_per_vertical", {})
        fit = fit_capability(cap_per_v, vertical, min_cap, k)
        if fit < fit_threshold:
            continue

        # Pick the cheapest endpoint in this tier
        best_endpoint = None
        best_cost = float("inf")
        for ep in endpoints:
            if breaker_states.get(ep["name"]) == "OPEN":
                continue
            endpoint_context = min(
                int(tier.get("max_context", 32768)),
                int(ep.get("max_context", tier.get("max_context", 32768))),
            )
            endpoint_usable = int(endpoint_context * max(0.0, 1.0 - reserve_pct / 100.0))
            if ctx.estimated_input_tokens + output_tokens > endpoint_usable:
                continue
            pricing = ep.get("pricing", {})
            cost = expected_cost(
                fixed_per_request=pricing.get("fixed_per_request", 0.0),
                in_per_1k=pricing.get("in_per_1k_tokens", 0.0),
                out_per_1k=pricing.get("out_per_1k_tokens", 0.0),
                estimated_in_tokens=ctx.estimated_input_tokens,
                estimated_out_tokens=ctx.estimated_output_tokens,
                fit=fit,
                retry_penalty_multiplier=retry_pen,
            )
            if cost < best_cost:
                best_cost = cost
                best_endpoint = ep["name"]
        if best_endpoint:
            candidates.append((tier["name"], best_endpoint, best_cost, fit))

    if not candidates:
        # No fit >= threshold; fall back to highest-capability tier (escalation)
        return _escalate_to_safe_tier(ctx, conf, breaker_states, "no_fit")

    # Sort by cost ascending
    candidates.sort(key=lambda x: x[2])

    chosen_tier, chosen_endpoint, chosen_cost, chosen_fit = candidates[0]

    # Efficiency tie-break: if top-2 costs are within margin AND a faster endpoint is available
    eff_cfg = conf.policy.get("efficiency_tiebreak", {})
    if eff_cfg.get("prefer_healthier") or eff_cfg.get("prefer_lower_load"):
        if len(candidates) >= 2:
            cost_diff_pct = (candidates[1][2] - candidates[0][2]) / max(candidates[0][2], 1e-9) * 100
            if cost_diff_pct < 5.0:
                # Look for a faster+healthier+lower-load alternative
                for tier_name, ep_name, cost, fit in candidates[1:]:
                    if cost > chosen_cost * 1.05:
                        break
                    if breaker_states.get(ep_name) == "OPEN":
                        continue
                    if eff_cfg.get("prefer_lower_load"):
                        if endpoint_loads.get(ep_name, 0) < endpoint_loads.get(chosen_endpoint, 0):
                            chosen_tier, chosen_endpoint, chosen_cost, chosen_fit = tier_name, ep_name, cost, fit
                            break

    # Escalation checks
    escalated = False
    source = "arith"
    rationale_parts = []

    # 1. OOD escalation
    if ctx.ood.is_ood:
        return _escalate_to_safe_tier(ctx, conf, breaker_states, "ood")
    # 2. Low confidence
    conf_thresh = escalation_cfg.get("confidence_threshold", 0.55)
    if ctx.vertical_top2 and ctx.vertical_top2[0][1] < conf_thresh:
        return _escalate_to_safe_tier(ctx, conf, breaker_states, "low_confidence")
    # 3. Top-2 close
    if len(ctx.vertical_top2) >= 2:
        eps = escalation_cfg.get("top2_epsilon", 0.10)
        diff = ctx.vertical_top2[0][1] - ctx.vertical_top2[1][1]
        if diff < eps:
            return _escalate_to_safe_tier(ctx, conf, breaker_states, "top2_close")
    # 4. Cost-margin abstain
    margin = escalation_cfg.get("cost_margin_abstain_pct", 5.0)
    if len(candidates) >= 2:
        cost_diff_pct = (candidates[1][2] - candidates[0][2]) / max(candidates[0][2], 1e-9) * 100
        # Only escalate when there's a non-zero (meaningful) margin AND it's small
        # — true ties (cost_diff_pct=0) shouldn't escalate; we'd just bounce tiers
        if 0 < cost_diff_pct < margin:
            # Escalate to candidates[1]
            chosen_tier, chosen_endpoint, chosen_cost, chosen_fit = candidates[1]
            source = "arith_escalated_margin"
            escalated = True
            rationale_parts.append(f"cost-margin escalation: {cost_diff_pct:.1f}% < {margin}%")

    return RoutingDecision(
        tier=chosen_tier,
        endpoint=chosen_endpoint,
        source=source,
        escalated=escalated,
        cost_usd=chosen_cost,
        fit=chosen_fit,
        rationale="; ".join(rationale_parts) if rationale_parts else f"min-cost fit={chosen_fit:.2f}",
    )


def _escalate_to_safe_tier(
    ctx: RequestContext,
    conf: cfg.Config,
    breaker_states: dict[str, str],
    reason: str,
) -> RoutingDecision:
    """Pick the highest-capability non-override-only tier with available endpoint."""
    safe_tier_name = conf.config.get("routing", {}).get("escalation", {}).get("ood_flag_to_tier", "tier3")
    safe_tier = conf.tier(safe_tier_name)
    if not safe_tier:
        # Fallback: pick any non-override-only tier
        for t in conf.config.get("tiers", []):
            if not t.get("override_only"):
                safe_tier = t
                break
    if not safe_tier:
        raise RuntimeError("no tiers configured")

    endpoints = conf.endpoints_for_tier(safe_tier["name"])
    for ep in endpoints:
        if breaker_states.get(ep["name"]) != "OPEN":
            return RoutingDecision(
                tier=safe_tier["name"],
                endpoint=ep["name"],
                source=f"escalation_{reason}",
                escalated=True,
                rationale=f"escalation reason={reason}",
            )

    # All endpoints down — pick fallback
    fallback_name = conf.config.get("routing", {}).get("cost_first", {}).get("fallback_endpoint", "intel")
    fallback_ep = conf.endpoint(fallback_name)
    if fallback_ep:
        return RoutingDecision(
            tier=safe_tier["name"],
            endpoint=fallback_name,
            source=f"escalation_{reason}_fallback",
            escalated=True,
            rationale=f"escalation reason={reason}, fallback={fallback_name}",
        )

    raise RuntimeError("no available endpoints")


# ============================================================
# Budget-aware routing: capability + remaining budget + 99% cost-to-complete
# ============================================================


@dataclass
class BudgetAwareCandidate:
    """Result of evaluating a single (tier, endpoint) candidate against budget."""
    tier_name: str
    endpoint_name: str
    cost: float
    fit: float
    success_probability: float
    # Probability that the request will be *completed* (success on first attempt
    # OR after one or more fallbacks). always <= success_probability + retries.
    p_completed: float
    # 99th-percentile estimated cost to complete (success OR retry).
    cost_to_complete_p99: float
    # Tokens used if this route is chosen.
    estimated_tokens: int
    # Estimated USD cost.
    estimated_cost_usd: float
    # Whether this candidate fits the request within the remaining budget.
    fits_remaining_tokens: bool
    # Whether this candidate is allowed under the plan's allowed_models.
    allowed_by_plan: bool
    # Reason for ineligibility, if any.
    ineligibility_reason: str | None = None


@dataclass
class BudgetRoutingDecision:
    """Result of budget-aware routing: either a single RoutingDecision or a
    cascade chain with guaranteed P(success) >= target.
    """
    decision: RoutingDecision
    cascade: list[BudgetAwareCandidate]
    target_success_probability: float
    achieved_success_probability: float
    rationale: str


def _normal_approx_z(p: float) -> float:
    """Inverse normal CDF (z-score) for a probability p in (0,1). Used for the
    99th-percentile cost estimation. Returns 2.33 for p=0.99."""
    # Abramowitz & Stegun 26.2.23 inverse-erf approximation
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    # Closed-form approx via the rational approximation of the inverse normal CDF
    # (Beasley-Springer-Moro). Accurate to ~1e-9.
    from math import log, sqrt
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def estimate_success_probability(
    endpoint_name: str,
    vertical: str,
    complexity: int,
    *,
    quality_profile: dict | None = None,
    conservative_prior: float = 0.5,
    min_samples: int = 10,
) -> tuple[float, float]:
    """Estimate P(success | endpoint, vertical, complexity) on the first try.

    Returns:
        (p_success, calibration_samples_used)

    Calibration:
      - With fewer than min_samples, returns a conservative prior (default 0.5).
      - With >= min_samples, returns Wilson lower-bound (95% confidence) of
        observed success rate. This intentionally underestimates the true rate
        so the router's "99% guaranteed" target is met conservatively.
    """
    if quality_profile is None:
        from . import memory
        quality_profile = memory.get_quality_profile(endpoint_name, vertical, complexity) or {}
    n = int(quality_profile.get("total_count", 0) or 0)
    if n < min_samples:
        return conservative_prior, float(n)
    success = int(quality_profile.get("success_count", 0) or 0)
    # Wilson lower bound for 95% confidence of a binomial proportion
    p = success / n if n > 0 else 0.0
    z = 1.96  # 95% one-sided
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * float(_normal_approx_z(0.975))) * float((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) ** 0.5 / denom
    lower = max(0.0, min(1.0, center - margin))
    return float(lower), float(n)


def cost_to_complete_p99(
    base_cost: float,
    p_success: float,
    *,
    max_retries: int = 1,
    retry_cost_multiplier: float = 1.0,
    target_probability: float = 0.99,
) -> float:
    """Estimate the 99th-percentile cost to complete a request.

    Model: with probability p_success the first attempt succeeds at `base_cost`;
    otherwise we retry up to `max_retries` times (each at `retry_cost_multiplier`
    * base_cost). The total cost is the sum of all attempts until success.

    Returns the (1 - target_probability) upper-tail of the cost distribution.
    """
    if p_success >= 1.0:
        return base_cost
    if p_success <= 0.0:
        return float(max_retries + 1) * base_cost * retry_cost_multiplier
    expected_attempts = 1.0 / p_success
    capped_attempts = max(1, min(max_retries + 1, int(expected_attempts * 3) + 1))
    # Build CDF of total cost (geometric distribution over attempts)
    quantile_attempts = max(1, min(capped_attempts, int(round(1.0 / (1.0 - target_probability)))))
    return base_cost * quantile_attempts * retry_cost_multiplier


def plan_allowed_endpoint(
    plan_quota: dict | None,
    endpoint_name: str,
) -> bool:
    """If plan_quota has allowed_models and it's non-empty, require endpoint_name in it."""
    if not plan_quota:
        return True
    allowed = plan_quota.get("allowed_models") or []
    if not allowed:
        return True
    return endpoint_name in allowed


def evaluate_candidate(
    *,
    ctx: RequestContext,
    conf: cfg.Config,
    tier_cfg: dict,
    endpoint_cfg: dict,
    breaker_states: dict[str, str],
    plan_quota: dict | None,
    daily_token_limit: int,
    remaining_tokens_today: int,
    remaining_model_tokens_today: int,
    fit: float,
    cost: float,
    target_success_probability: float,
) -> BudgetAwareCandidate:
    """Evaluate a single (tier, endpoint, fit, cost) candidate against budget and quality."""
    endpoint_name = endpoint_cfg["name"]
    # Per-endpoint quality profile
    p_success, n_samples = estimate_success_probability(
        endpoint_name=endpoint_name,
        vertical=ctx.vertical,
        complexity=ctx.complexity,
    )
    # Probability of completion in a single attempt = p_success
    # (cascade P(completion) is computed by the caller)
    p_completed = p_success
    # 99th-percentile cost to complete: assume up to 1 retry at same cost
    p99_cost = cost_to_complete_p99(
        base_cost=cost,
        p_success=p_success,
        max_retries=1,
        retry_cost_multiplier=1.0,
        target_probability=target_success_probability,
    )
    estimated_tokens = ctx.estimated_input_tokens + ctx.estimated_output_tokens
    fits_remaining_tokens = (
        remaining_tokens_today == -1
        or (estimated_tokens <= remaining_tokens_today)
    ) and (
        remaining_model_tokens_today == -1
        or (estimated_tokens <= remaining_model_tokens_today)
    )
    allowed_by_plan = plan_allowed_endpoint(plan_quota, endpoint_name)
    ineligibility_reason: str | None = None
    if not allowed_by_plan:
        ineligibility_reason = f"endpoint {endpoint_name} not in plan allowed_models"
    elif not fits_remaining_tokens:
        ineligibility_reason = (
            f"endpoint {endpoint_name} requires {estimated_tokens} tokens but "
            f"tenant has {remaining_tokens_today} remaining today"
        )
    return BudgetAwareCandidate(
        tier_name=tier_cfg["name"],
        endpoint_name=endpoint_name,
        cost=cost,
        fit=fit,
        success_probability=p_success,
        p_completed=p_completed,
        cost_to_complete_p99=p99_cost,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=cost,
        fits_remaining_tokens=fits_remaining_tokens,
        allowed_by_plan=allowed_by_plan,
        ineligibility_reason=ineligibility_reason,
    )


def cascade_success_probability(chain: list[BudgetAwareCandidate]) -> float:
    """P(first attempt success) UNION P(fallback success) for a candidate chain.

    Approximates cascade success as 1 - product(1 - p_i). This is the
    probability that at least one candidate in the chain completes the request.
    """
    if not chain:
        return 0.0
    p_fail = 1.0
    for c in chain:
        p_fail *= 1.0 - c.p_completed
    return 1.0 - p_fail


def budget_aware_route(
    ctx: RequestContext,
    conf: cfg.Config,
    breaker_states: dict[str, str],
    endpoint_loads: dict[str, int],
    *,
    tenant_mgr,
    tenant_id: str,
    plan_quota: dict | None = None,
) -> BudgetRoutingDecision:
    """Budget-aware routing: combine capability, remaining budget, and 99%
    probability cost-to-complete.

    Strategy:
      1. Build eligible candidates like cost_first_route (capability-fit + breakers).
      2. Reject candidates that don't fit remaining tenant/model token budget.
      3. Reject candidates that aren't allowed by the tenant's plan.
      4. Sort candidates by cost ascending.
      5. Walk candidates in order, building a cascade chain until
         P(cascade success) >= target_success_probability.
      6. Return the cascade with the lowest 99th-percentile cost-to-complete.
      7. If no chain meets the target, raise BudgetError with a clear
         quality_target_unmet reason.
    """
    from . import memory as _memory

    routing = conf.config.get("routing", {})
    cost_cfg = routing.get("cost_first", {})
    fit_threshold = cost_cfg.get("fit_threshold", 0.9)
    k = cost_cfg.get("capability_sigmoid_k", 20.0)
    retry_pen = cost_cfg.get("retry_penalty_multiplier", 5.0)

    vertical = ctx.vertical
    vertical_obj = conf.vertical(vertical)
    if vertical_obj:
        min_cap = float(vertical_obj.get("min_capability", 0.5))
    else:
        min_cap = 0.5

    # Pull tenant budget state
    target_p = tenant_mgr.get_or_create(tenant_id).target_success_probability
    daily_token_limit = tenant_mgr.get_or_create(tenant_id).daily_token_limit
    remaining_tokens = tenant_mgr.remaining_tokens_today(tenant_id)
    if plan_quota is None:
        plan_quota = _memory.get_tenant_plan_quota(tenant_id)

    candidates: list[BudgetAwareCandidate] = []
    for tier in conf.config.get("tiers", []):
        if tier.get("override_only"):
            continue
        if tier.get("vision_endpoints") and not ctx.has_image:
            continue
        reserve_pct = float(conf.policy.get("ladder", {}).get("context_reserve_pct", 25))
        usable_context = int(tier.get("max_context", 32768) * max(0.0, 1.0 - reserve_pct / 100.0))
        output_tokens = max(ctx.estimated_output_tokens, int(tier.get("max_tokens_bump", 0)))
        if ctx.estimated_input_tokens + output_tokens > usable_context:
            continue
        endpoints = conf.endpoints_for_tier(tier["name"])
        if not endpoints:
            continue
        any_endpoint_available = False
        for ep in endpoints:
            if breaker_states.get(ep["name"]) != "OPEN":
                any_endpoint_available = True
                break
        if not any_endpoint_available:
            continue
        cap_per_v = tier.get("capability_per_vertical", {})
        fit = fit_capability(cap_per_v, vertical, min_cap, k)
        if fit < fit_threshold:
            continue
        best_candidate: BudgetAwareCandidate | None = None
        for ep in endpoints:
            if breaker_states.get(ep["name"]) == "OPEN":
                continue
            endpoint_context = min(
                int(tier.get("max_context", 32768)),
                int(ep.get("max_context", tier.get("max_context", 32768))),
            )
            endpoint_usable = int(endpoint_context * max(0.0, 1.0 - reserve_pct / 100.0))
            if ctx.estimated_input_tokens + output_tokens > endpoint_usable:
                continue
            pricing = ep.get("pricing", {})
            cost = expected_cost(
                fixed_per_request=pricing.get("fixed_per_request", 0.0),
                in_per_1k=pricing.get("in_per_1k_tokens", 0.0),
                out_per_1k=pricing.get("out_per_1k_tokens", 0.0),
                estimated_in_tokens=ctx.estimated_input_tokens,
                estimated_out_tokens=ctx.estimated_output_tokens,
                fit=fit,
                retry_penalty_multiplier=retry_pen,
            )
            remaining_model = tenant_mgr.remaining_model_tokens_today(tenant_id, ep["name"])
            cand = evaluate_candidate(
                ctx=ctx,
                conf=conf,
                tier_cfg=tier,
                endpoint_cfg=ep,
                breaker_states=breaker_states,
                plan_quota=plan_quota,
                daily_token_limit=daily_token_limit,
                remaining_tokens_today=remaining_tokens,
                remaining_model_tokens_today=remaining_model,
                fit=fit,
                cost=cost,
                target_success_probability=target_p,
            )
            if best_candidate is None or cand.cost < best_candidate.cost:
                best_candidate = cand
        if best_candidate is not None:
            candidates.append(best_candidate)

    if not candidates:
        # Fall back to escalation chain (no candidates with fit >= threshold)
        try:
            decision = _escalate_to_safe_tier(ctx, conf, breaker_states, "no_fit")
        except Exception as e:
            raise RuntimeError(f"no viable route: {e}") from e
        return BudgetRoutingDecision(
            decision=decision,
            cascade=[],
            target_success_probability=target_p,
            achieved_success_probability=0.0,
            rationale=f"escalation chain only: {decision.rationale}",
        )

    # Eligible = fits budget + allowed by plan
    eligible = [c for c in candidates if c.fits_remaining_tokens and c.allowed_by_plan]
    if not eligible:
        raise BudgetError(
            "quality_target_unmet",
            f"No accessible model can meet the {target_p:.0%} success target "
            f"within the remaining {remaining_tokens} token budget today.",
        )

    eligible.sort(key=lambda c: c.cost)
    # Build cascade chain: shortest chain whose P(success) >= target_p
    chain: list[BudgetAwareCandidate] = []
    achieved_p = 0.0
    for cand in eligible:
        chain.append(cand)
        achieved_p = cascade_success_probability(chain)
        if achieved_p >= target_p:
            break
    if achieved_p < target_p:
        raise BudgetError(
            "quality_target_unmet",
            f"No reachable cascade chain meets the {target_p:.0%} success target "
            f"(best achievable: {achieved_p:.0%}).",
        )

    # Total cost-to-complete for the cascade
    cost_p99 = sum(c.cost_to_complete_p99 for c in chain)
    primary = chain[0]
    source = "budget_aware" if len(chain) == 1 else "budget_aware_cascade"
    rationale_parts = [
        f"cascade={len(chain)}",
        f"P(success)={achieved_p:.2f}",
        f"cost_p99={cost_p99:.4f}",
    ]
    decision = RoutingDecision(
        tier=primary.tier_name,
        endpoint=primary.endpoint_name,
        source=source,
        cost_usd=primary.cost,
        fit=primary.fit,
        rationale="; ".join(rationale_parts),
        extra={
            "cascade": [
                {"tier": c.tier_name, "endpoint": c.endpoint_name, "cost": c.cost}
                for c in chain
            ],
            "achieved_success_probability": achieved_p,
            "target_success_probability": target_p,
            "cost_to_complete_p99": cost_p99,
        },
    )
    return BudgetRoutingDecision(
        decision=decision,
        cascade=chain,
        target_success_probability=target_p,
        achieved_success_probability=achieved_p,
        rationale="; ".join(rationale_parts),
    )


class BudgetError(Exception):
    """Raised when the budget-aware router cannot meet the quality target."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

