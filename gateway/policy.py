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
    """Evaluate prototype_match(name='x', threshold=0.85) — structural only."""
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
        text_words = set(ctx.text.lower().split())
        for proto in conf.prototypes.get("prototypes", []):
            if proto.get("name") != name or not proto.get("enabled", True):
                continue
            if proto.get("kind") != "structural":
                continue
            seeds = proto.get("centroid_seed_text", [])
            if not seeds:
                continue
            score = 0.0
            for seed in seeds:
                words = set(seed.lower().split())
                overlap = len(words & text_words)
                score = max(score, overlap / max(len(words), 1))
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


def _compute_prototype_scores(ctx: RequestContext, conf: cfg.Config) -> list[tuple[str, float]]:
    """Score prototypes against the request's router projection.

    Returns list of (name, similarity) sorted desc.
    Currently uses keyword overlap as a fallback when projection isn't available
    (stub model case). When real model has projection, use cosine similarity.
    """
    out = []
    text_lower = ctx.text.lower()
    for proto in conf.prototypes.get("prototypes", []):
        if not proto.get("enabled", True):
            continue
        if proto.get("kind") != "structural":
            continue
        # Keyword overlap score (fallback / stub)
        seeds = proto.get("centroid_seed_text", [])
        if not seeds:
            continue
        score = 0.0
        for seed in seeds:
            words = set(seed.lower().split())
            text_words = set(text_lower.split())
            overlap = len(words & text_words)
            score = max(score, overlap / max(len(words), 1))
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
