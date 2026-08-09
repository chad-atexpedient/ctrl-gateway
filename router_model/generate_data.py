"""Synthetic training data generator.

Calls the configured teacher/reviewer model to generate labeled examples per
vertical in taxonomy.yaml. Generates ~120 examples per vertical × 5 difficulty
levels, plus adversarial confusable pairs (Glint Roman-Empire lesson).

Output:
  data/base/train.jsonl    - main training set (~6,000+ examples)
  data/base/eval.jsonl     - held-out eval set (~600 examples)
  Each line: {text, vertical, complexity, code, math, reasoning, long_output, query_hash}

Run:
  python router_model/generate_data.py --out router_model/data/base
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import aiohttp
import yaml

GEN_SYSTEM_PROMPT = (
    "You are a synthetic data generator for a routing classifier. "
    "You will receive a list of verticals, each with description and example queries. "
    "For each vertical, generate {n_per_vertical} example user prompts covering a range of "
    "difficulty levels (1=trivial, 5=expert). Mark code/math/reasoning/long_output flags. "
    "Make prompts realistic and varied — short chat, long technical questions, multi-step, etc. "
    "Also include 5 adversarial 'confusable' prompts per vertical that look like they could "
    "belong to a different vertical (to harden the classifier against the Roman-Empire-medical "
    "lesson). "
    "\n\nReturn ONLY a JSON object with this structure:\n"
    "{{verticals: [...]}} where each vertical is "
    "{{name: str, examples: [{{text, complexity, code, math, reasoning, long_output, "
    "confusable_with: null | str}}]}}"
)


async def generate_for_all_verticals(
    taxonomy_path: str,
    out_dir: str,
    n_per_vertical: int = 25,
    eval_split_pct: float = 0.1,
    api_endpoint: str | None = None,
    api_key_env: str = "TEACHER_API_KEY",
    model: str | None = None,
    batch_size: int = 5,
):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    api_endpoint = api_endpoint or os.environ.get("TEACHER_BASE_URL", "")
    if not api_endpoint:
        print("ERROR: api_endpoint not set. Set TEACHER_BASE_URL env or pass --endpoint", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get(api_key_env, "")

    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    verticals = taxonomy.get("verticals", [])
    print(f"Loaded {len(verticals)} verticals")

    # Batches
    all_train = []
    all_eval = []
    for i in range(0, len(verticals), batch_size):
        batch = verticals[i:i + batch_size]
        try:
            data = await generate_batch(batch, n_per_vertical, api_endpoint, api_key, model)
        except Exception as e:
            print(f"Batch {i} failed: {e}", file=sys.stderr)
            continue
        for _, vert in enumerate(data.get("verticals", [])):
            examples = vert.get("examples", [])
            for ex in examples:
                ex["vertical"] = vert["name"]
                ex["query_hash"] = hashlib.sha256(ex["text"].encode("utf-8")).hexdigest()[:16]
                if random.random() < eval_split_pct:
                    all_eval.append(ex)
                else:
                    all_train.append(ex)
        print(f"Batch {i}: total now {len(all_train)} train + {len(all_eval)} eval")
        # Rate limit politeness
        await asyncio.sleep(0.5)

    # Deduplicate by hash
    seen = set()
    dedup_train = []
    for ex in all_train:
        if ex["query_hash"] not in seen:
            seen.add(ex["query_hash"])
            dedup_train.append(ex)
    dedup_eval = []
    for ex in all_eval:
        if ex["query_hash"] not in seen:
            seen.add(ex["query_hash"])
            dedup_eval.append(ex)

    train_path = out_dir_path / "train.jsonl"
    eval_path = out_dir_path / "eval.jsonl"
    write_jsonl(train_path, dedup_train)
    write_jsonl(eval_path, dedup_eval)
    print(f"Wrote {len(dedup_train)} train → {train_path}")
    print(f"Wrote {len(dedup_eval)} eval → {eval_path}")


async def generate_batch(
    verticals_batch: list[dict],
    n_per_vertical: int,
    api_endpoint: str,
    api_key: str,
    model: str | None,
) -> dict:
    verticals_text = []
    for v in verticals_batch:
        ex_lines = "\n".join(f"  - {e}" for e in v.get("examples", [])[:3])
        verticals_text.append(
            f"- name: {v['name']}\n"
            f"  description: {v.get('description', '')}\n"
            f"  examples: {ex_lines}"
        )
    user_payload = (
        f"Generate {n_per_vertical} examples per vertical below.\n\n"
        + "\n".join(verticals_text)
    )
    sys_prompt = GEN_SYSTEM_PROMPT.format(n_per_vertical=n_per_vertical)
    body = {
        "model": model or os.environ.get("TEACHER_MODEL", "GPT-5.6 Luna"),
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.7,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.post(
            f"{api_endpoint.rstrip('/')}/chat/completions",
            headers=headers, json=body,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                import re
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if m:
                    return json.loads(m.group(1))
                raise


def write_jsonl(path: Path, items: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default="./router_model/taxonomy.yaml")
    parser.add_argument("--out", default="./router_model/data/base")
    parser.add_argument("--n-per-vertical", type=int, default=25)
    parser.add_argument("--eval-split-pct", type=float, default=0.1)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(generate_for_all_verticals(
        taxonomy_path=args.taxonomy,
        out_dir=args.out,
        n_per_vertical=args.n_per_vertical,
        eval_split_pct=args.eval_split_pct,
        api_endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        model=args.model,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
