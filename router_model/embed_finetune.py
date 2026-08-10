"""Contrastive embedding fine-tuning.

Manually gated via POST /retrain --allow-embedding-finetune. trainer_worker.py
runs this as its own subprocess BEFORE train.py, then (if it produced an
artifact) passes the resulting ONNX to train.py via --embedding-onnx so heads
get trained on top of the fine-tuned embedding instead of the stock one.

Builds anchor/positive/negative triplets from two sources:
  - The vertical-disagreement pool (router vs. reviewer mismatches, exported
    by trainer_worker.py._export_disagreement_pool): anchor = the confused
    prompt, positive = another example of its TRUE vertical, negative =
    another example of the vertical the router WRONGLY predicted. This
    directly targets the actual confusion pairs seen in production traffic
    instead of guessing at what's confusable.
  - The base training set alone: anchor/positive = same vertical, negative =
    a random different vertical. Keeps the embedding well-behaved generally
    instead of overfitting to whatever disagreements happened to occur so far
    (the disagreement pool can be small and noisy early on).

Fine-tunes only the top --unfrozen-layers transformer layers (default 2);
everything else (embeddings, lower layers) stays frozen. This limits
catastrophic forgetting / drift on everything else that implicitly relies on
this embedding staying well-behaved on inputs it hasn't specifically been
tuned on: the OOD max-probability threshold, cost-first's min_capability fit
curves, and (after this run's train.py pass) the structural-prototype
centroids. Small LR, few epochs — this is a light nudge, not retraining from
scratch.

SAFETY: if fine-tuning does not improve held-out triplet accuracy over the
pre-fine-tune baseline, this script deliberately does NOT export an ONNX
artifact. trainer_worker.py detects the missing file and falls back to
heads-only training on the existing embedding. Fine-tuning a sentence
embedder on a few thousand triplets can regress if the disagreement pool is
noisy or too small — silently shipping a worse embedding would be worse than
skipping the step.

Exports an ONNX artifact with the SAME interface as the stock
BAAI/bge-small-en-v1.5 ONNX (input_ids/attention_mask[/token_type_ids] ->
last_hidden_state; pooling + L2-norm stay in the numpy consumer code, exactly
like router_model/train.py.encode_with_embedding and
gateway/router.py._RealModel._encode) so it's a drop-in replacement — no
changes needed anywhere that loads/uses the embedding ONNX.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_triplets(
    disagreements: list[dict],
    base_by_vertical: dict[str, list[str]],
    max_general_triplets: int,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Returns a list of (anchor, positive, negative) text triplets."""
    rng = random.Random(seed)
    triplets: list[tuple[str, str, str]] = []

    # 1. Disagreement-derived triplets — the highest-value signal: these are
    # the EXACT confusable pairs the router got wrong in production.
    for d in disagreements:
        anchor = d.get("text")
        true_v = d.get("true_vertical")
        wrong_v = d.get("router_vertical")
        if not anchor or not true_v or not wrong_v or true_v == wrong_v:
            continue
        pos_pool = base_by_vertical.get(true_v, [])
        neg_pool = base_by_vertical.get(wrong_v, [])
        if not pos_pool or not neg_pool:
            continue
        positive = rng.choice(pos_pool)
        if positive == anchor:
            others = [p for p in pos_pool if p != anchor]
            if not others:
                continue
            positive = rng.choice(others)
        negative = rng.choice(neg_pool)
        triplets.append((anchor, positive, negative))

    # 2. General triplets sampled from base data — keeps the embedding from
    # overfitting to whatever disagreements happened to occur so far.
    verticals = [v for v, texts in base_by_vertical.items() if len(texts) >= 2]
    for _ in range(max_general_triplets):
        if len(verticals) < 2:
            break
        v_pos = rng.choice(verticals)
        v_neg = rng.choice([v for v in verticals if v != v_pos])
        pos_pair = rng.sample(base_by_vertical[v_pos], 2)
        negative = rng.choice(base_by_vertical[v_neg])
        triplets.append((pos_pair[0], pos_pair[1], negative))

    rng.shuffle(triplets)
    return triplets


def _mean_pool_normalize(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, dim=-1)


def export_finetuned_onnx(model, tokenizer, output_path: Path):
    """Export the fine-tuned encoder with the SAME I/O contract as the stock
    bge-small ONNX: raw last_hidden_state out, pooling stays external (done
    by the numpy consumer, matching train.py / gateway/router.py exactly)."""
    import torch

    class _Wrapped(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            return self.base(**kwargs).last_hidden_state

    model.eval()
    dummy = tokenizer(
        ["example query for onnx export"], return_tensors="pt",
        padding=True, truncation=True, max_length=32,
    )
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    }
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}
        inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Wrapped(model),
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disagreement-pool", required=True, help="path to disagreement samples jsonl")
    parser.add_argument("--base-data-dir", required=True)
    parser.add_argument("--output-onnx", required=True, help="path to save the fine-tuned embedding ONNX")
    parser.add_argument("--model-id", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument(
        "--unfrozen-layers", type=int, default=2,
        help="Only fine-tune the top N transformer layers; embeddings + lower layers stay frozen "
             "to limit drift on everything that relies on this embedding staying well-behaved "
             "(OOD threshold, min_capability fit curves, structural-prototype centroids).",
    )
    parser.add_argument("--max-general-triplets", type=int, default=2000)
    parser.add_argument(
        "--min-triplets", type=int, default=20,
        help="Below this many usable triplets, skip fine-tuning entirely (not enough signal).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.disagreement_pool).exists():
        print(f"No disagreement pool at {args.disagreement_pool}; nothing to fine-tune against.")
        return

    disagreements = load_jsonl(Path(args.disagreement_pool))
    print(f"Loaded {len(disagreements)} disagreement samples")

    base_train_path = Path(args.base_data_dir) / "train.jsonl"
    if not base_train_path.exists():
        print(f"No base train data at {base_train_path}; cannot build positive/negative pools.")
        return
    base_data = load_jsonl(base_train_path)
    base_by_vertical: dict[str, list[str]] = defaultdict(list)
    for d in base_data:
        if d.get("text") and d.get("vertical"):
            base_by_vertical[d["vertical"]].append(d["text"])

    triplets = build_triplets(disagreements, base_by_vertical, args.max_general_triplets, seed=args.seed)
    print(f"Built {len(triplets)} triplets (disagreement-derived + general)")
    if len(triplets) < args.min_triplets:
        print(f"Only {len(triplets)} triplets (< --min-triplets {args.min_triplets}); skipping fine-tune.")
        return

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print(f"PyTorch + transformers required for embedding fine-tune: {e}")
        print("Install with: pip install torch transformers")
        return

    random.Random(args.seed).shuffle(triplets)
    n_val = max(1, len(triplets) // 10)
    val_triplets, train_triplets = triplets[:n_val], triplets[n_val:]
    if not train_triplets:
        print("Not enough triplets left for a train split after carving out validation; skipping.")
        return

    print(f"Loading {args.model_id} for fine-tuning (CPU; this is a light top-layer nudge, not full retraining)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir="./router_model/data/embedding_cache")
    model = AutoModel.from_pretrained(args.model_id, cache_dir="./router_model/data/embedding_cache")

    # Freeze everything except the top N encoder layers (limits drift).
    for param in model.parameters():
        param.requires_grad = False
    encoder_layers = getattr(getattr(model, "encoder", None), "layer", None)
    if encoder_layers is not None and args.unfrozen_layers > 0:
        for layer in list(encoder_layers)[-args.unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        # Unknown architecture (custom --model-id) — fine-tune everything
        # rather than silently training on zero trainable params.
        for param in model.parameters():
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Fine-tuning {sum(p.numel() for p in trainable):,} params (top {args.unfrozen_layers} layers)")

    def encode_batch(texts: list[str]):
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
        out = model(**enc)
        return _mean_pool_normalize(out.last_hidden_state, enc["attention_mask"])

    def batch_triplet_loss(anchors, positives, negatives):
        a, p, n = encode_batch(anchors), encode_batch(positives), encode_batch(negatives)
        pos_dist = 1.0 - (a * p).sum(dim=-1)  # cosine distance; vectors are L2-normalized
        neg_dist = 1.0 - (a * n).sum(dim=-1)
        return functional.relu(pos_dist - neg_dist + args.margin), pos_dist, neg_dist

    @torch.no_grad()
    def val_metrics(trips: list[tuple[str, str, str]]) -> tuple[float, float]:
        model.eval()
        losses, correct, total = [], 0, 0
        for i in range(0, len(trips), args.batch_size):
            batch = trips[i:i + args.batch_size]
            anchors, positives, negatives = zip(*batch, strict=False)
            per_ex_loss, pos_dist, neg_dist = batch_triplet_loss(list(anchors), list(positives), list(negatives))
            losses.append(per_ex_loss.mean().item())
            correct += int((pos_dist < neg_dist).sum().item())
            total += len(batch)
        return (sum(losses) / max(len(losses), 1)), (correct / max(total, 1))

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    pre_loss, pre_acc = val_metrics(val_triplets)
    print(f"Before fine-tune: val_loss={pre_loss:.4f} val_triplet_acc={pre_acc:.3f} (n_val={len(val_triplets)})")

    best_acc = pre_acc
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train_triplets)
        for i in range(0, len(train_triplets), args.batch_size):
            batch = train_triplets[i:i + args.batch_size]
            anchors, positives, negatives = zip(*batch, strict=False)
            opt.zero_grad()
            per_ex_loss, _, _ = batch_triplet_loss(list(anchors), list(positives), list(negatives))
            per_ex_loss.mean().backward()
            opt.step()
        val_loss, val_acc = val_metrics(val_triplets)
        print(f"epoch {epoch + 1}/{args.epochs}: val_loss={val_loss:.4f} val_triplet_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None or best_acc <= pre_acc:
        print(
            f"Fine-tuning did not improve triplet accuracy ({pre_acc:.3f} -> {best_acc:.3f}); "
            "NOT exporting. The gateway keeps its current (stock or previously fine-tuned) embedding.",
        )
        return

    model.load_state_dict(best_state)
    print(f"Best val_triplet_acc={best_acc:.3f} (started at {pre_acc:.3f}); exporting.")

    export_finetuned_onnx(model, tokenizer, Path(args.output_onnx))
    print(f"Exported fine-tuned embedding -> {args.output_onnx}")


if __name__ == "__main__":
    main()
