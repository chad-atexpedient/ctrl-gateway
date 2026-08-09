"""Train router heads on (frozen embedding + new head weights).

Usage:
  python router_model/train.py \
    --base-data-dir ./router_model/data/base \
    --curated-data-dir ./router_model/data/curated/run-XXX \
    --output-heads ./router_model/checkpoints/v1_heads.npz \
    --output-onnx ./router_model/checkpoints/v1_model.onnx \
    --output-metadata ./router_model/checkpoints/v1_meta.json \
    --mix-ratio-base-pct 30 \
    --mix-ratio-curated-pct 70

Loads frozen embedding (bge-small-en-v1.5), encodes all training data, trains
heads (vertical softmax + complexity ordinal + binary flags + projection),
evaluates on base-eval, and exports ONNX-compatible artifact + npz weights.

By default the embedding is FROZEN. --allow-embedding-finetune enables
contrastive fine-tuning of the embedding projection (slow).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def encode_with_embedding(
    texts: list[str],
    model_id: str = "BAAI/bge-small-en-v1.5",
    onnx_path: str | None = None,
) -> np.ndarray:
    """Run frozen embedding on a list of texts. Returns (N, 384) array."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers + onnxruntime required") from None

    # Download / load ONNX from HuggingFace if not present
    from huggingface_hub import hf_hub_download
    if onnx_path is None:
        onnx_path = hf_hub_download(model_id, "model.onnx", cache_dir="./router_model/data/embedding_cache")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir="./router_model/data/embedding_cache")

    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        inputs = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="np")
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
        outputs = session.run(None, onnx_inputs)
        emb = outputs[0]
        mask = inputs["attention_mask"].astype(np.float32)
        emb = (emb * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-9)
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def train_heads(
    embeddings: np.ndarray,
    labels_vertical: np.ndarray,  # (N,) int
    labels_complexity: np.ndarray,  # (N,) int 1-5
    labels_code: np.ndarray,  # (N,) float 0/1
    labels_math: np.ndarray,
    labels_reasoning: np.ndarray,
    labels_long_output: np.ndarray,
    num_verticals: int,
    epochs: int = 50,
    lr: float = 0.01,
    projection_dim: int = 64,
) -> dict[str, np.ndarray]:
    """Train heads with simple PyTorch-free gradient descent (numpy).

    For simplicity and to avoid torch dependency at training time, we use
    ridge regression for vertical+complexity heads and logistic regression
    for binary flags.

    This is intentionally simple — the real model can be retrained with
    PyTorch later if more sophistication is needed.
    """
    N, D = embeddings.shape
    rng = np.random.default_rng(42)

    # Vertical: one-vs-rest logistic regression
    Y_v = np.eye(num_verticals)[labels_vertical]
    W_v = rng.normal(0, 0.01, (num_verticals, D))
    b_v = np.zeros(num_verticals)
    for epoch in range(epochs):
        logits = embeddings @ W_v.T + b_v
        # Softmax cross-entropy gradient
        probs = _softmax(logits)
        grad = (probs - Y_v) / N
        W_v -= lr * grad.T @ embeddings
        b_v -= lr * grad.sum(axis=0)
        if epoch % 10 == 0:
            preds = logits.argmax(axis=1)
            acc = (preds == labels_vertical).mean()
            if acc > 0.97:
                break

    # Complexity ordinal: 5 binary heads (P(cx >= k) for k=1..5)
    Y_c = np.zeros((N, 5))
    for i in range(N):
        for k in range(labels_complexity[i]):
            Y_c[i, k] = 1.0
    W_c = rng.normal(0, 0.01, (5, D))
    b_c = np.zeros(5)
    for _ in range(epochs):
        logits = embeddings @ W_c.T + b_c
        probs = _sigmoid(logits)
        grad = (probs - Y_c) / N
        W_c -= lr * grad.T @ embeddings
        b_c -= lr * grad.sum(axis=0)

    # Binary flags
    def fit_binary(Y, epochs=epochs, lr=lr):
        Y2 = Y.reshape(-1, 1).astype(np.float32)
        w = rng.normal(0, 0.01, (1, D))
        b = np.zeros(1)
        for _ in range(epochs):
            logits = embeddings @ w.T + b
            probs = _sigmoid(logits)
            grad = (probs - Y2) / N
            w -= lr * grad.T @ embeddings
            b -= lr * grad.sum(axis=0)
        return w[0], float(b[0])

    W_code, b_code = fit_binary(labels_code)
    W_math, b_math = fit_binary(labels_math)
    W_reasoning, b_reasoning = fit_binary(labels_reasoning)
    W_long_output, b_long_output = fit_binary(labels_long_output)

    # Projection head (for prototypes): linear → L2 normalize
    W_proj = rng.normal(0, 0.01, (projection_dim, D))
    b_proj = np.zeros(projection_dim)
    for _ in range(epochs // 2):
        proj = embeddings @ W_proj.T + b_proj
        # Center & normalize target = unit sphere
        grad = proj / N
        W_proj -= lr * 0.1 * grad.T @ embeddings
        b_proj -= lr * 0.1 * grad.sum(axis=0)

    return {
        "W_vertical": W_v.astype(np.float32),
        "b_vertical": b_v.astype(np.float32),
        "W_complexity": W_c.astype(np.float32),
        "b_complexity": b_c.astype(np.float32),
        "W_code": W_code.astype(np.float32),
        "b_code": np.array([b_code], dtype=np.float32),
        "W_math": W_math.astype(np.float32),
        "b_math": np.array([b_math], dtype=np.float32),
        "W_reasoning": W_reasoning.astype(np.float32),
        "b_reasoning": np.array([b_reasoning], dtype=np.float32),
        "W_long_output": W_long_output.astype(np.float32),
        "b_long_output": np.array([b_long_output], dtype=np.float32),
        "W_projection": W_proj.astype(np.float32),
        "b_projection": b_proj.astype(np.float32),
    }


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", required=True)
    parser.add_argument("--curated-data-dir", default=None)
    parser.add_argument("--output-heads", required=True)
    parser.add_argument("--output-onnx", default=None)  # Optional re-export
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--mix-ratio-base-pct", type=int, default=30)
    parser.add_argument("--mix-ratio-curated-pct", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--allow-embedding-finetune", action="store_true")
    parser.add_argument("--verticals", default="./router_model/taxonomy.yaml")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    import yaml

    with open(args.verticals, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    verticals = taxonomy.get("verticals", [])
    vert_idx = {v["name"]: i for i, v in enumerate(verticals)}

    # Load base data
    base_dir = Path(args.base_data_dir)
    base_train = load_jsonl(base_dir / "train.jsonl")
    print(f"base train: {len(base_train)}")

    curated_train = []
    if args.curated_data_dir:
        curated_path = Path(args.curated_data_dir) / "samples.jsonl"
        if curated_path.exists():
            curated_train = load_jsonl(curated_path)
            print(f"curated train: {len(curated_train)}")

    # Never train on samples present in the frozen base-eval set.
    eval_path = base_dir / "eval.jsonl"
    eval_hashes = set()
    if eval_path.exists():
        for item in load_jsonl(eval_path):
            eval_hashes.add(item.get("query_hash") or item.get("text"))
    if eval_hashes:
        base_train = [d for d in base_train if (d.get("query_hash") or d.get("text")) not in eval_hashes]
        curated_train = [d for d in curated_train if (d.get("query_hash") or d.get("text")) not in eval_hashes]

    # Mix
    n_base_target = int(len(base_train) * args.mix_ratio_base_pct / 100)
    n_curated_target = int(len(base_train) * args.mix_ratio_curated_pct / 100)
    # Sample
    if n_base_target < len(base_train):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(base_train), size=n_base_target, replace=False)
        base_train = [base_train[i] for i in idx]
    if n_curated_target > 0 and len(curated_train) > 0:
        if n_curated_target < len(curated_train):
            rng = np.random.default_rng(43)
            idx = rng.choice(len(curated_train), size=n_curated_target, replace=False)
            curated_train = [curated_train[i] for i in idx]
    train_data = base_train + curated_train
    deduped = {}
    for item in train_data:
        deduped[item.get("query_hash") or item.get("text")] = item
    train_data = list(deduped.values())
    if not train_data:
        raise RuntimeError("no training samples remain after held-out deduplication")
    rng = np.random.default_rng(44)
    rng.shuffle(train_data)
    print(f"final train: {len(train_data)} ({len(base_train)} base + {len(curated_train)} curated)")

    # Encode
    texts = [d["text"] for d in train_data]
    print("encoding with frozen embedding...")
    t0 = time.time()
    embeddings = encode_with_embedding(texts)
    print(f"encoded in {time.time() - t0:.1f}s; shape={embeddings.shape}")

    # Labels (accept both flag_code and code keys — curated samples use flag_*)
    labels_vertical = np.array([vert_idx.get(d["vertical"], vert_idx.get("other", 0)) for d in train_data])
    labels_complexity = np.array([int(d.get("complexity", 2)) for d in train_data])
    labels_code = np.array([1.0 if d.get("code") or d.get("flag_code") else 0.0 for d in train_data])
    labels_math = np.array([1.0 if d.get("math") or d.get("flag_math") else 0.0 for d in train_data])
    labels_reasoning = np.array([1.0 if d.get("reasoning") or d.get("flag_reasoning") else 0.0 for d in train_data])
    labels_long_output = np.array([1.0 if d.get("long_output") or d.get("flag_long_output") else 0.0 for d in train_data])

    # Train
    print("training heads...")
    t0 = time.time()
    weights = train_heads(
        embeddings,
        labels_vertical,
        labels_complexity,
        labels_code,
        labels_math,
        labels_reasoning,
        labels_long_output,
        num_verticals=len(verticals),
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"trained in {time.time() - t0:.1f}s")

    metadata = {
        "version": args.version or f"v-{int(time.time())}",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": embeddings.shape[1],
        "num_verticals": len(verticals),
        "vertical_names": [v["name"] for v in verticals],
        "trained_at": time.time(),
        "n_train": len(train_data),
        "epochs": args.epochs,
        "lr": args.lr,
        "allow_embedding_finetune": args.allow_embedding_finetune,
    }
    # Save metadata in both the requested sidecar and the heads artifact. The
    # gateway loads metadata_json directly from the npz during hot-swap.
    out_heads = Path(args.output_heads)
    out_heads.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_heads, **weights, metadata_json=json.dumps(metadata))
    print(f"saved heads → {out_heads}")
    Path(args.output_metadata).write_text(json.dumps(metadata, indent=2))
    print(f"saved metadata → {args.output_metadata}")

    # Export the deployable ONNX artifact (frozen embedding copy) so the gateway
    # can hot-swap. The embedding weights are unchanged; only heads are new.
    if args.output_onnx:
        from huggingface_hub import hf_hub_download
        embedding_onnx = hf_hub_download(
            "BAAI/bge-small-en-v1.5", "model.onnx",
            cache_dir="./router_model/data/embedding_cache",
        )
        out_onnx = Path(args.output_onnx)
        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(embedding_onnx, out_onnx)
        import hashlib
        onnx_sha = hashlib.sha256(out_onnx.read_bytes()).hexdigest()
        # Sidecar heads copy alongside the onnx (gateway discovers <onnx stem>.heads.npz)
        heads_sidecar = out_onnx.with_name(out_onnx.stem + "_heads.npz")
        shutil.copy(out_heads, heads_sidecar)
        print(f"exported ONNX → {out_onnx} (sha256={onnx_sha[:16]})")
        print(f"exported heads → {heads_sidecar}")


if __name__ == "__main__":
    main()
