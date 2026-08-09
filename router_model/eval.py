"""Evaluate trained router on base-eval and live-eval sets.

Usage:
  python router_model/eval.py \
    --heads ./router_model/checkpoints/v1_heads.npz \
    --onnx ./router_model/checkpoints/v1_model.onnx \
    --base-eval ./router_model/data/base/eval.jsonl \
    --live-eval ./router_model/data/live-eval/live_eval.jsonl \
    --output-json ./router_model/checkpoints/v1_eval.json

Computes:
  - per_vertical_accuracy
  - top-20 confusion matrix
  - base_accuracy, live_accuracy
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from train import _sigmoid, _softmax, encode_with_embedding


def predict_with_heads(embeddings: np.ndarray, weights: dict, vertical_names: list[str]) -> list[dict]:
    """Run heads on embeddings. Returns list of {vertical, complexity, code, math, reasoning, long_output}."""
    logits_v = embeddings @ weights["W_vertical"].T + weights["b_vertical"]
    probs_v = _softmax(logits_v)
    vertical_idx = probs_v.argmax(axis=1)

    logits_c = embeddings @ weights["W_complexity"].T + weights["b_complexity"]
    probs_c = _sigmoid(logits_c)
    complexity = (probs_c > 0.5).sum(axis=1).clip(min=1, max=5)

    code = _sigmoid(embeddings @ weights["W_code"].T + weights["b_code"]).flatten() > 0.5
    math = _sigmoid(embeddings @ weights["W_math"].T + weights["b_math"]).flatten() > 0.5
    reasoning = _sigmoid(embeddings @ weights["W_reasoning"].T + weights["b_reasoning"]).flatten() > 0.5
    long_output = _sigmoid(embeddings @ weights["W_long_output"].T + weights["b_long_output"]).flatten() > 0.5

    return [
        {
            "vertical": vertical_names[i],
            "vertical_idx": int(j),
            "complexity": int(complexity[i]),
            "code": bool(code[i]),
            "math": bool(math[i]),
            "reasoning": bool(reasoning[i]),
            "long_output": bool(long_output[i]),
        }
        for i, j in enumerate(vertical_idx)
    ]


def evaluate(
    eval_path: Path,
    weights: dict,
    embedding_id: str,
    vertical_names: list[str],
    onnx_path: str,
) -> dict:
    data = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if not data:
        return {
            "accuracy": 0.0,
            "complexity_accuracy": 0.0,
            "flag_accuracy": {},
            "per_vertical_accuracy": {},
            "confusion_top20": [],
            "n": 0,
        }

    texts = [d["text"] for d in data]
    print(f"  encoding {len(texts)} eval samples...")
    embeddings = encode_with_embedding(texts, model_id=embedding_id, onnx_path=onnx_path)
    preds = predict_with_heads(embeddings, weights, vertical_names)

    correct = 0
    complexity_correct = 0
    flag_correct = defaultdict(int)
    flag_total = defaultdict(int)
    per_v = defaultdict(lambda: {"correct": 0, "wrong": 0})
    confusions = defaultdict(int)
    for d, p in zip(data, preds, strict=False):
        true_v = d.get("vertical", "")
        if true_v == p["vertical"]:
            correct += 1
            per_v[true_v]["correct"] += 1
        else:
            per_v[true_v]["wrong"] += 1
            confusions[(true_v, p["vertical"])] += 1
        complexity_correct += int(int(d.get("complexity", 2)) == p["complexity"])
        for flag in ("code", "math", "reasoning", "long_output"):
            expected = bool(d.get(flag) or d.get(f"flag_{flag}"))
            flag_correct[flag] += int(expected == p[flag])
            flag_total[flag] += 1

    n = len(data)
    accuracy = correct / n if n else 0.0
    per_v_acc = {v: s["correct"] / max(s["correct"] + s["wrong"], 1) for v, s in per_v.items()}
    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:20]
    confusion_top20 = [
        {"true": t[0], "pred": t[1], "count": c}
        for t, c in top_conf
    ]
    return {
        "accuracy": accuracy,
        "complexity_accuracy": complexity_correct / n if n else 0.0,
        "flag_accuracy": {
            flag: flag_correct[flag] / max(flag_total[flag], 1)
            for flag in flag_total
        },
        "per_vertical_accuracy": per_v_acc,
        "confusion_top20": confusion_top20,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", required=True)
    parser.add_argument("--onnx", required=True, help="path to embedding ONNX (for encoding eval)")
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--live-eval", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--verticals", default="./router_model/taxonomy.yaml")
    parser.add_argument("--embedding-id", default="BAAI/bge-small-en-v1.5",
                        help="embedding model id (must match training + gateway config)")
    args = parser.parse_args()

    import yaml

    with open(args.verticals, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    verticals = taxonomy.get("verticals", [])
    vertical_names = [v["name"] for v in verticals]

    weights_npz = np.load(args.heads)
    weights = {k: weights_npz[k] for k in weights_npz.files}

    results = {"would_have_picked": {}}
    print("evaluating base...")
    base_res = evaluate(
        Path(args.base_eval), weights, embedding_id=args.embedding_id,
        vertical_names=vertical_names, onnx_path=args.onnx,
    )
    results["base_accuracy"] = base_res["accuracy"]
    results["per_vertical_accuracy"] = base_res["per_vertical_accuracy"]
    results["confusion_top20"] = base_res["confusion_top20"]
    results["n_base"] = base_res["n"]
    results["base_complexity_accuracy"] = base_res["complexity_accuracy"]
    results["base_flag_accuracy"] = base_res["flag_accuracy"]

    if args.live_eval and Path(args.live_eval).exists():
        print("evaluating live...")
        live_res = evaluate(
            Path(args.live_eval), weights, embedding_id=args.embedding_id,
            vertical_names=vertical_names, onnx_path=args.onnx,
        )
        results["live_accuracy"] = live_res["accuracy"]
        results["n_live"] = live_res["n"]
        results["live_complexity_accuracy"] = live_res["complexity_accuracy"]
        results["live_flag_accuracy"] = live_res["flag_accuracy"]
    else:
        results["live_accuracy"] = None
        results["n_live"] = 0

    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"saved → {args.output_json}")
    print(f"base accuracy: {base_res['accuracy']:.3f}")
    if "live_accuracy" in results and results["live_accuracy"] is not None:
        print(f"live accuracy: {results['live_accuracy']:.3f}")


if __name__ == "__main__":
    main()
