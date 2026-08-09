"""Swarm / synthesis routing mode.

Inspired by the orchestrator's SwarmRouter pattern: when a request is complex
enough that a single-model response is the bottleneck, decompose into parallel
sub-tasks across multiple tiers, then synthesize.

Cost-aware: swarm mode only triggers if its expected total cost is less than
the monolithic tier4 path. If not, fall back to single-tier cost-first routing.

Modes supported (configurable via gateway-policy.json -> swarm):
  - off: never use swarm
  - auto: trigger when cost-first would pick tier4 + complexity >= 4
  - always: always try swarm for complex requests

Triggered by:
  - Tier chosen by cost-first is tier4 (or high-capability tier)
  - Complexity >= 4 AND (code OR reasoning OR long_output)
  - Estimated cost of swarm < estimated cost of monolithic tier4
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from . import config as cfg
from . import endpoints, transcoder

log = logging.getLogger("glint.swarm")


@dataclass
class SubTask:
    id: str
    prompt: str
    target_tier: str  # which tier to use for this subtask
    rationale: str = ""


@dataclass
class SwarmResult:
    synthesis: str
    subtask_count: int
    total_cost_usd: float
    per_subtask_cost: dict[str, float]
    model_versions: dict[str, str]
    synthesis_tier: str
    duration_ms: float
    aborted: bool = False
    abort_reason: str = ""


def should_swarm(
    ctx,
    conf: cfg.Config,
    cost_first_decision_tier: str,
    cost_first_estimated_cost: float,
) -> tuple[bool, str]:
    """Decide whether to enter swarm mode."""
    swarm_cfg = conf.policy.get("swarm", {})
    if not swarm_cfg:
        return False, "swarm not configured"
    mode = swarm_cfg.get("mode", "off")
    if mode == "off":
        return False, "swarm off"
    if mode == "always":
        if ctx.complexity < 4:
            return False, "always but cx<4"
    elif mode == "auto":
        if cost_first_decision_tier not in swarm_cfg.get("trigger_tiers", ["tier4"]):
            return False, f"tier {cost_first_decision_tier} not in trigger_tiers"
        if ctx.complexity < swarm_cfg.get("min_complexity", 4):
            return False, f"cx {ctx.complexity} < min"
        if not (ctx.flags.get("code") or ctx.flags.get("reasoning") or ctx.flags.get("long_output")):
            return False, "no swarm-eligible flag"

    # Cost check: swarm must be cheaper than monolithic
    configured_max = swarm_cfg.get("max_swarm_cost_usd")
    max_swarm_cost = cost_first_estimated_cost * 0.8 if configured_max is None else float(configured_max)
    estimated_swarm_cost = swarm_cfg.get("estimated_cost_usd")
    if estimated_swarm_cost is None:
        estimated_swarm_cost = cost_first_estimated_cost * float(swarm_cfg.get("estimated_cost_multiplier", 0.75))
    estimated_swarm_cost = float(estimated_swarm_cost)
    if estimated_swarm_cost > max_swarm_cost:
        return False, f"estimated swarm cost {estimated_swarm_cost:.4f} exceeds cap {max_swarm_cost:.4f}"
    if cost_first_estimated_cost > 0 and estimated_swarm_cost >= cost_first_estimated_cost:
        return False, "swarm is not cheaper than monolithic route"
    return True, "triggered"


def decompose(
    ctx,
    conf: cfg.Config,
    swarm_cfg: dict,
) -> list[SubTask]:
    """Decompose the request into parallel subtasks across tiers.

    For now, uses a simple strategy: split the prompt by paragraph/section
    boundaries and dispatch each chunk to a different tier based on sub-routing.
    Heavy/abstract parts go to higher tiers; concrete/extractive parts to lower.

    A more advanced strategy could use an LLM to plan subtasks (like the
    orchestrator's `EvaluatorEngine`), but that's opt-in via swarm_cfg.llm_plan=true.
    """
    chunks = _split_into_chunks(ctx.text, max_chars=swarm_cfg.get("chunk_max_chars", 2000))
    if not chunks:
        chunks = [ctx.text]

    tier_pyramid = swarm_cfg.get("tier_pyramid", ["tier0", "tier2", "tier3"])
    subtasks = []
    for i, chunk in enumerate(chunks):
        tier = tier_pyramid[i % len(tier_pyramid)]
        # Heavier chunks go higher
        if len(chunk) > 4000 and "tier4" not in tier_pyramid:
            tier = "tier3"
        subtasks.append(SubTask(
            id=f"sub-{i}",
            prompt=chunk,
            target_tier=tier,
            rationale=f"chunk {i+1}/{len(chunks)}; tier={tier}; len={len(chunk)}",
        ))
    return subtasks


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    # Split by double newlines first
    paragraphs = text.split("\n\n")
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 > max_chars:
            if cur:
                chunks.append(cur.strip())
            cur = p
        else:
            cur += ("\n\n" if cur else "") + p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


async def execute_swarm(
    ctx,
    conf: cfg.Config,
    subtasks: list[SubTask],
    pool: endpoints.EndpointPool,
    synthesis_tier: str = "tier3",
    request_id: str = "",
) -> SwarmResult:
    """Execute subtasks in parallel, then synthesize. Cost-aware."""
    t0 = time.time()
    per_cost: dict[str, float] = {}
    model_versions: dict[str, str] = {}

    # Build all requests
    async def run_subtask(st: SubTask):
        tier_cfg = conf.tier(st.target_tier)
        if not tier_cfg:
            return st.id, None, 0.0, ""
        endpoints_list = conf.endpoints_for_tier(st.target_tier)
        if not endpoints_list:
            return st.id, None, 0.0, ""
        ep_cfg = endpoints_list[0]
        ep_client = pool.get(ep_cfg["name"])

        payload = {
            "model": ep_cfg.get("model_alias", "default"),
            "messages": [
                {"role": "system", "content": "You are a careful analyst. Answer only the chunk you are given."},
                {"role": "user", "content": st.prompt},
            ],
            "stream": False,
        }
        transcoded = transcoder.transcode(ep_cfg, tier_cfg, payload)
        try:
            result = await ep_client.send(transcoded, stream=False)
            if not isinstance(result, dict):
                raise RuntimeError(f"subtask {st.id}: unexpected non-dict response")
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            cost = _subtask_cost(result, ep_cfg, tier_cfg)
            return st.id, content, cost, ep_cfg.get("model_alias", "")
        except Exception as e:
            log.warning("subtask %s failed: %s", st.id, e)
            return st.id, None, 0.0, ep_cfg.get("model_alias", "")

    results = await asyncio.gather(*[run_subtask(st) for st in subtasks])

    # Collect
    pieces = []
    total_cost = 0.0
    for st_id, content, cost, model in results:
        per_cost[st_id] = cost
        total_cost += cost
        if model:
            model_versions[st_id] = model
        if content:
            pieces.append(f"[{st_id}]\n{content}")

    if not pieces:
        return SwarmResult(
            synthesis="[swarm: all subtasks failed]",
            subtask_count=len(subtasks),
            total_cost_usd=total_cost,
            per_subtask_cost=per_cost,
            model_versions=model_versions,
            synthesis_tier=synthesis_tier,
            duration_ms=(time.time() - t0) * 1000,
            aborted=True,
            abort_reason="all subtasks failed",
        )

    # Synthesis
    syn_tier_cfg = conf.tier(synthesis_tier)
    syn_endpoints = conf.endpoints_for_tier(synthesis_tier)
    syn_text, synthesis_cost, synthesis_model = await _synthesize(
        original_request=ctx.text,
        subtask_pieces=pieces,
        endpoint_cfg=syn_endpoints[0] if syn_endpoints else None,
        tier_cfg=syn_tier_cfg,
        pool=pool,
    )
    total_cost += synthesis_cost
    per_cost["synthesis"] = synthesis_cost
    if synthesis_model:
        model_versions["synthesis"] = synthesis_model

    return SwarmResult(
        synthesis=syn_text,
        subtask_count=len(subtasks),
        total_cost_usd=total_cost,
        per_subtask_cost=per_cost,
        model_versions=model_versions,
        synthesis_tier=synthesis_tier,
        duration_ms=(time.time() - t0) * 1000,
    )


def _subtask_cost(result: dict, ep_cfg: dict, tier_cfg: dict) -> float:
    usage = result.get("usage", {})
    in_t = usage.get("prompt_tokens", 0)
    out_t = usage.get("completion_tokens", 0)
    pricing = ep_cfg.get("pricing", {})
    return (
        pricing.get("fixed_per_request", 0.0)
        + (in_t / 1000.0) * pricing.get("in_per_1k_tokens", 0.0)
        + (out_t / 1000.0) * pricing.get("out_per_1k_tokens", 0.0)
    )


async def _synthesize(
    original_request: str,
    subtask_pieces: list[str],
    endpoint_cfg: dict | None,
    tier_cfg: dict | None,
    pool: endpoints.EndpointPool,
) -> tuple[str, float, str]:
    if not endpoint_cfg or not tier_cfg:
        return "\n\n".join(subtask_pieces), 0.0, ""
    combined = "\n\n---\n\n".join(subtask_pieces)
    payload = {
        "model": endpoint_cfg.get("model_alias", "default"),
        "messages": [
            {"role": "system", "content": "You are a synthesis assistant. Combine the following subtask outputs into a coherent, complete answer to the user's original request. Remove duplication. Preserve all key facts."},
            {"role": "user", "content": f"Original request: {original_request}\n\nSubtask outputs:\n{combined}"},
        ],
        "stream": False,
    }
    transcoded = transcoder.transcode(endpoint_cfg, tier_cfg, payload)
    try:
        client = pool.get(endpoint_cfg["name"])
        result = await client.send(transcoded, stream=False)
        if not isinstance(result, dict):
            raise RuntimeError("synthesis: unexpected non-dict response")
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", combined),
            _subtask_cost(result, endpoint_cfg, tier_cfg),
            endpoint_cfg.get("model_alias", ""),
        )
    except Exception as e:
        log.warning("synthesis failed: %s", e)
        return combined, 0.0, endpoint_cfg.get("model_alias", "")
