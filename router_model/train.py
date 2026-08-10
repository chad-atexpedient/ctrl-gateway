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

Architecture: a shared trunk (Linear 384->hidden, ReLU) feeds per-task linear
heads (vertical/complexity/flags/projection). Trained with PyTorch (AdamW,
mini-batch, class-weighted CE for vertical imbalance, BCE for the ordinal
complexity + binary flag heads, a supervised-contrastive auxiliary loss for
the projection head, early stopping on a held-out validation split carved out
of the training mix). The embedding itself is frozen (numpy/ONNX at
inference) unless --embedding-onnx points at a fine-tuned copy produced by
embed_finetune.py.

The numpy forward pass in gateway/router.py._RealModel._run_heads and this
script's PyTorch model MUST stay architecturally identical (same trunk shape,
ReLU activation) — see AGENTS.md invariants. router_model/eval.py mirrors the
same forward pass in numpy for evaluation without a torch dependency.

After training, structural-prototype centroids (router_model/prototypes.json)
are recomputed from each prototype's centroid_seed_text using the trained
trunk + projection head, so gateway/policy.py can do real cosine-similarity
prototype matching instead of the keyword-overlap fallback. Pass
--skip-centroids to disable.
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
    """Run frozen embedding on a list of texts. Returns (N, 384) array.

    onnx_path, when given, overrides the stock HuggingFace download — this is
    how a fine-tuned embedding (router_model/embed_finetune.py output) gets
    used for training/centroid encoding instead of the stock weights. The
    tokenizer is unaffected by fine-tuning (vocab doesn't change), so it's
    always loaded from model_id.
    """
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


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ordinal_targets(labels_complexity: np.ndarray, num_levels: int = 5) -> np.ndarray:
    """P(cx >= k) targets for k=1..num_levels, one-hot-cumulative encoding."""
    n = len(labels_complexity)
    y = np.zeros((n, num_levels), dtype=np.float32)
    for i in range(n):
        for k in range(min(int(labels_complexity[i]), num_levels)):
            y[i, k] = 1.0
    return y


def _supervised_contrastive_loss(projections, labels, temperature: float = 0.1):
    """SupCon-style loss (Khosla et al.) over one mini-batch.

    projections: (N, D) L2-normalized torch tensor. labels: (N,) int tensor
    (vertical id). Anchors with no other same-label example in the batch
    contribute zero loss (skipped) rather than raising — small/rare verticals
    just get a weaker contrastive signal until batches happen to include a
    same-label pair, which is fine since this is an auxiliary loss term.
    """
    import torch

    n = projections.shape[0]
    if n < 2:
        return torch.zeros((), device=projections.device)
    sim = projections @ projections.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim)
    self_mask = torch.eye(n, dtype=torch.bool, device=projections.device)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & (~self_mask)
    denom = exp_sim.sum(dim=1)
    valid = positive_mask.any(dim=1) & (denom > 0)
    if not bool(valid.any()):
        return torch.zeros((), device=projections.device)
    log_prob = sim - torch.log(denom.unsqueeze(1) + 1e-12)
    pos_count = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos[valid].mean()


def train_heads(
    embeddings: np.ndarray,
    labels_vertical: np.ndarray,  # (N,) int
    labels_complexity: np.ndarray,  # (N,) int 1-5
    labels_code: np.ndarray,  # (N,) float 0/1
    labels_math: np.ndarray,
    labels_reasoning: np.ndarray,
    labels_long_output: np.ndarray,
    num_verticals: int,
    hidden_dim: int = 256,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    val_pct: float = 0.1,
    patience: int = 8,
    projection_dim: int = 64,
    complexity_weight: float = 1.0,
    flags_weight: float = 1.0,
    projection_weight: float = 0.3,
    seed: int = 42,
):
    """Train a small MLP (shared trunk + per-task heads) with PyTorch.

    Returns (weights: dict[str, np.ndarray], train_info: dict, model: the
    trained torch.nn.Module in eval mode — kept around so the caller can
    reuse it to encode structural-prototype centroids without re-loading).
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required to train heads. Install with: pip install torch "
            "(see requirements.txt)."
        ) from e

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n, d = embeddings.shape
    idx = rng.permutation(n)
    n_val = int(n * val_pct) if n >= 20 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        train_idx, val_idx = idx, np.array([], dtype=int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_t(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    x_all = to_t(embeddings)
    y_v_all = to_t(labels_vertical, torch.long)
    y_c_all = to_t(_ordinal_targets(labels_complexity))
    y_code_all = to_t(labels_code)
    y_math_all = to_t(labels_math)
    y_reason_all = to_t(labels_reasoning)
    y_long_all = to_t(labels_long_output)

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Linear(d, hidden_dim)
            self.dropout = nn.Dropout(0.1)
            self.vertical = nn.Linear(hidden_dim, num_verticals)
            self.complexity = nn.Linear(hidden_dim, 5)
            self.code = nn.Linear(hidden_dim, 1)
            self.math = nn.Linear(hidden_dim, 1)
            self.reasoning = nn.Linear(hidden_dim, 1)
            self.long_output = nn.Linear(hidden_dim, 1)
            self.projection = nn.Linear(hidden_dim, projection_dim)

        def forward(self, x):
            h = functional.relu(self.trunk(x))
            hd = self.dropout(h)
            return {
                "vertical": self.vertical(hd),
                "complexity": self.complexity(hd),
                "code": self.code(hd).squeeze(-1),
                "math": self.math(hd).squeeze(-1),
                "reasoning": self.reasoning(hd).squeeze(-1),
                "long_output": self.long_output(hd).squeeze(-1),
                # No dropout on the projection path — keep it a stable,
                # deterministic representation for prototype centroids.
                "projection": functional.normalize(self.projection(h), dim=-1),
            }

    model = Heads().to(device)

    # Inverse-frequency class weights for the vertical CE loss (57-way,
    # naturally imbalanced). Unseen-in-train classes get weight 1 (never
    # contribute gradient anyway — no samples to compute loss on).
    counts = np.bincount(labels_vertical[train_idx], minlength=num_verticals).astype(np.float64)
    counts_safe = np.where(counts == 0, 1.0, counts)
    class_weights = counts_safe.sum() / (num_verticals * counts_safe)
    class_weights_t = to_t(class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights_t)
    bce_loss = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        x_all[train_idx], y_v_all[train_idx], y_c_all[train_idx],
        y_code_all[train_idx], y_math_all[train_idx], y_reason_all[train_idx], y_long_all[train_idx],
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_acc = -1.0
    best_state = None
    stall = 0
    val_acc = None

    for epoch in range(epochs):
        model.train()
        for xb, yvb, ycb, ycob, ymab, yreb, ylob in loader:
            opt.zero_grad()
            out = model(xb)
            loss = ce_loss(out["vertical"], yvb)
            loss = loss + complexity_weight * bce_loss(out["complexity"], ycb)
            flag_loss = (
                bce_loss(out["code"], ycob) + bce_loss(out["math"], ymab)
                + bce_loss(out["reasoning"], yreb) + bce_loss(out["long_output"], ylob)
            ) / 4.0
            loss = loss + flags_weight * flag_loss
            if projection_weight > 0:
                loss = loss + projection_weight * _supervised_contrastive_loss(out["projection"], yvb)
            loss.backward()
            opt.step()

        if len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                val_out = model(x_all[val_idx])
                val_acc = (val_out["vertical"].argmax(dim=1) == y_v_all[val_idx]).float().mean().item()
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1}/{epochs} val_acc={val_acc:.3f} (best={best_val_acc:.3f})")
            if stall >= patience:
                print(f"  early stop at epoch {epoch + 1} (best val_acc={best_val_acc:.3f})")
                break
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs} (no validation split — dataset too small)")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Suggested calibration temperature: grid search minimizing NLL on the
    # validation split. This is a printed/metadata SUGGESTION only — the
    # live knob stays gateway-policy.json -> calibration.temperature (manual,
    # by design; see README "Reviewer model selection" / policy doc).
    suggested_temp = 1.0
    if len(val_idx) > 0:
        with torch.no_grad():
            val_logits = model(x_all[val_idx])["vertical"]
        best_nll = float("inf")
        for t in np.arange(0.5, 2.55, 0.05):
            nll = functional.cross_entropy(val_logits / float(t), y_v_all[val_idx]).item()
            if nll < best_nll:
                best_nll = nll
                suggested_temp = round(float(t), 2)

    sd = model.state_dict()

    def w(name):
        return sd[name].detach().cpu().numpy().astype(np.float32)

    weights = {
        "W_trunk1": w("trunk.weight"), "b_trunk1": w("trunk.bias"),
        "W_vertical": w("vertical.weight"), "b_vertical": w("vertical.bias"),
        "W_complexity": w("complexity.weight"), "b_complexity": w("complexity.bias"),
        "W_code": w("code.weight"), "b_code": w("code.bias"),
        "W_math": w("math.weight"), "b_math": w("math.bias"),
        "W_reasoning": w("reasoning.weight"), "b_reasoning": w("reasoning.bias"),
        "W_long_output": w("long_output.weight"), "b_long_output": w("long_output.bias"),
        "W_projection": w("projection.weight"), "b_projection": w("projection.bias"),
    }
    train_info = {
        "val_accuracy": best_val_acc if best_val_acc >= 0 else None,
        "val_n": int(len(val_idx)),
        "train_n": int(len(train_idx)),
        "suggested_calibration_temperature": suggested_temp,
        "hidden_dim": hidden_dim,
    }
    return weights, train_info, model


def compute_prototype_centroids(model, embed_fn, prototypes: dict) -> tuple[dict, int]:
    """Fill prototypes[*].centroid from centroid_seed_text using the trained
    trunk + projection head. Mutates and returns `prototypes`; second return
    value is the count of prototypes updated. STRUCTURAL kind only — never
    touches (and never adds) topic prototypes (Glint Roman Empire lesson)."""
    import torch

    updated = 0
    for proto in prototypes.get("prototypes", []):
        if proto.get("kind") != "structural":
            continue
        seeds = proto.get("centroid_seed_text") or []
        if not seeds:
            continue
        seed_embeddings = embed_fn(seeds)
        with torch.no_grad():
            x = torch.as_tensor(seed_embeddings, dtype=torch.float32)
            proj = model(x)["projection"]  # (K, dim), already L2-normalized per-row
            centroid = proj.mean(dim=0)
            centroid = centroid / centroid.norm().clamp_min(1e-9)
        proto["centroid"] = [float(v) for v in centroid.cpu().numpy()]
        updated += 1
    return prototypes, updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", required=True)
    parser.add_argument("--curated-data-dir", default=None)
    parser.add_argument("--output-heads", required=True)
    parser.add_argument("--output-onnx", default=None)  # Optional re-export
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--mix-ratio-base-pct", type=int, default=30)
    parser.add_argument("--mix-ratio-curated-pct", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-pct", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--complexity-weight", type=float, default=1.0)
    parser.add_argument("--flags-weight", type=float, default=1.0)
    parser.add_argument("--projection-weight", type=float, default=0.3,
                        help="Weight of the supervised-contrastive projection loss; 0 disables it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-embedding-finetune", action="store_true",
                        help="Marker flag for the caller's intent; actual fine-tuning happens in "
                             "embed_finetune.py, whose output ONNX should be passed via --embedding-onnx.")
    parser.add_argument("--embedding-onnx", default=None,
                        help="Path to a (possibly fine-tuned) embedding ONNX to encode with, "
                             "instead of downloading the stock model_id weights.")
    parser.add_argument("--verticals", default="./router_model/taxonomy.yaml")
    parser.add_argument("--prototypes", default="./router_model/prototypes.json")
    parser.add_argument("--skip-centroids", action="store_true",
                        help="Skip recomputing structural-prototype centroids after training.")
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
    embeddings = encode_with_embedding(texts, onnx_path=args.embedding_onnx)
    print(f"encoded in {time.time() - t0:.1f}s; shape={embeddings.shape}")

    # Labels (accept both flag_code and code keys — curated samples use flag_*)
    labels_vertical = np.array([vert_idx.get(d["vertical"], vert_idx.get("other", 0)) for d in train_data])
    labels_complexity = np.array([int(d.get("complexity", 2)) for d in train_data])
    labels_code = np.array([1.0 if d.get("code") or d.get("flag_code") else 0.0 for d in train_data])
    labels_math = np.array([1.0 if d.get("math") or d.get("flag_math") else 0.0 for d in train_data])
    labels_reasoning = np.array([1.0 if d.get("reasoning") or d.get("flag_reasoning") else 0.0 for d in train_data])
    labels_long_output = np.array([1.0 if d.get("long_output") or d.get("flag_long_output") else 0.0 for d in train_data])

    # Train
    print("training heads (PyTorch: trunk + vertical/complexity/flags/projection heads)...")
    t0 = time.time()
    weights, train_info, model = train_heads(
        embeddings,
        labels_vertical,
        labels_complexity,
        labels_code,
        labels_math,
        labels_reasoning,
        labels_long_output,
        num_verticals=len(verticals),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        val_pct=args.val_pct,
        patience=args.patience,
        complexity_weight=args.complexity_weight,
        flags_weight=args.flags_weight,
        projection_weight=args.projection_weight,
        seed=args.seed,
    )
    print(f"trained in {time.time() - t0:.1f}s; {train_info}")

    metadata = {
        "version": args.version or f"v-{int(time.time())}",
        "architecture": "mlp1_relu",
        "hidden_dim": args.hidden_dim,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_onnx_override": args.embedding_onnx,
        "embedding_dim": embeddings.shape[1],
        "num_verticals": len(verticals),
        "vertical_names": [v["name"] for v in verticals],
        "trained_at": time.time(),
        "n_train": len(train_data),
        "epochs_requested": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "allow_embedding_finetune": args.allow_embedding_finetune,
        **train_info,
    }
    # Save metadata in both the requested sidecar and the heads artifact. The
    # gateway loads metadata_json directly from the npz during hot-swap.
    out_heads = Path(args.output_heads)
    out_heads.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_heads, **weights, metadata_json=json.dumps(metadata))
    print(f"saved heads → {out_heads}")
    Path(args.output_metadata).write_text(json.dumps(metadata, indent=2))
    print(f"saved metadata → {args.output_metadata}")

    # Recompute structural-prototype centroids from the trained model so
    # gateway/policy.py can do real cosine-similarity matching.
    if not args.skip_centroids:
        proto_path = Path(args.prototypes)
        if proto_path.exists():
            prototypes = json.loads(proto_path.read_text(encoding="utf-8"))

            def embed_fn(seeds: list[str]) -> np.ndarray:
                return encode_with_embedding(seeds, onnx_path=args.embedding_onnx)

            prototypes, n_updated = compute_prototype_centroids(model, embed_fn, prototypes)
            proto_path.write_text(json.dumps(prototypes, indent=2), encoding="utf-8")
            print(f"updated {n_updated} structural-prototype centroids → {proto_path}")
        else:
            print(f"WARNING: --prototypes path {proto_path} not found; skipping centroid update")

    # Export the deployable ONNX artifact (frozen — or fine-tuned — embedding
    # copy) so the gateway can hot-swap. The embedding weights are unchanged
    # from whatever was used to encode training data; only heads are new.
    if args.output_onnx:
        if args.embedding_onnx:
            embedding_onnx = args.embedding_onnx
        else:
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
