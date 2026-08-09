"""Contrastive embedding fine-tuning — NOT IMPLEMENTED (stub).

Manually gated via POST /retrain --allow-embedding-finetune.
This script currently only validates the disagreement pool and reports stats;
real fine-tuning requires PyTorch + a transformers training loop.

The trainer_worker logs a warning when this is requested and proceeds with
heads-only training. Do not rely on this script producing a model yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def contrastive_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 0.5) -> float:
    """Triplet loss: minimize d(anchor, positive), maximize d(anchor, negative)."""
    pos_dist = np.linalg.norm(anchor - positive, axis=-1)
    neg_dist = np.linalg.norm(anchor - negative, axis=-1)
    loss = np.maximum(0.0, pos_dist - neg_dist + margin).mean()
    return float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disagreement-pool", required=True, help="path to disagreement samples jsonl")
    parser.add_argument("--base-data-dir", required=True)
    parser.add_argument("--output-onnx", required=True, help="path to save fine-tuned embedding")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()

    print("WARNING: This is a stub. Real implementation requires PyTorch + fine-tune the embedding model.")
    print("For now, this script just verifies the disagreement pool and reports stats.")

    if not Path(args.disagreement_pool).exists():
        print(f"No disagreement pool at {args.disagreement_pool}; nothing to fine-tune against.")
        return

    samples = []
    with open(args.disagreement_pool, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} disagreement samples")

    if not samples:
        return

    # Stats only
    by_field = {}
    for s in samples:
        for f in ("vertical", "complexity", "code", "math", "reasoning", "long_output"):
            if s.get(f"agreement_{f}") is False:
                by_field[f] = by_field.get(f, 0) + 1
    print("Disagreements by field:")
    for f, c in sorted(by_field.items(), key=lambda x: -x[1]):
        print(f"  {f}: {c}")

    print("\nFine-tuning not implemented in stub. Install pytorch and adapt this script.")
    print("See: https://huggingface.co/docs/transformers/training")


if __name__ == "__main__":
    main()
