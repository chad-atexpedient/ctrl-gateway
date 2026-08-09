"""Export trained router to a deployable ONNX + npz artifact set.

Reads:
  - frozen embedding ONNX (from HuggingFace, downloaded by train.py)
  - trained heads npz (from train.py)
  - metadata json (from train.py)

Writes:
  - data/embedding/model.onnx  (copy of frozen embedding)
  - data/embedding/heads.npz   (copy of trained heads)
  - data/embedding/metadata.json

The gateway loads these atomically on startup or on /reload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-heads", required=True, help="path to trained heads .npz")
    parser.add_argument("--source-onnx", required=True, help="path to frozen embedding ONNX")
    parser.add_argument("--source-metadata", required=True, help="path to metadata json")
    parser.add_argument("--target-dir", default="./router_model/data/embedding")
    parser.add_argument(
        "--update-config", default=None,
        help="path to gateway-config.json — fills embedding.checksum_sha256 with the exported ONNX hash",
    )
    args = parser.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Copy embedding ONNX
    onnx_dst = target / "model.onnx"
    shutil.copy(args.source_onnx, onnx_dst)
    onnx_sha = hashlib.sha256(onnx_dst.read_bytes()).hexdigest()
    print(f"copied ONNX → {onnx_dst} (sha256={onnx_sha[:16]})")

    # Copy heads npz
    heads_dst = target / "heads.npz"
    shutil.copy(args.source_heads, heads_dst)
    heads_sha = hashlib.sha256(heads_dst.read_bytes()).hexdigest()
    print(f"copied heads → {heads_dst} (sha256={heads_sha[:16]})")

    # Merge metadata + checksums
    with open(args.source_metadata, encoding="utf-8") as f:
        meta = json.load(f)
    meta["onnx_sha256"] = onnx_sha
    meta["heads_sha256"] = heads_sha
    meta_dst = target / "metadata.json"
    meta_dst.write_text(json.dumps(meta, indent=2))
    print(f"wrote metadata → {meta_dst}")

    # Fill the config checksum so the gateway verifies on load
    if args.update_config:
        cfg_path = Path(args.update_config)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        embedding = cfg.setdefault("embedding", {})
        embedding["checksum_sha256"] = onnx_sha
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"updated {cfg_path}: embedding.checksum_sha256 = {onnx_sha[:16]}…")

    print("\nDone. Update gateway-config.json -> embedding.onnx_path to point at this dir.")
    print(f"  embedding.onnx_path = {target / 'model.onnx'}")
    print(f"  heads will be auto-discovered at {target / 'heads.npz'}")


if __name__ == "__main__":
    main()
