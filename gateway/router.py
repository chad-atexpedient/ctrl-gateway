"""Glint-V2 router inference engine.

Hybrid architecture:
  - Frozen ONNX sentence embedding (bge-small-en-v1.5 default, 384-d)
  - Small classifier heads (vertical softmax, complexity ordinal, binary flags, 64-d projection)

Inference: encode -> heads -> softmax + temperature scale -> result.

Atomic hot-swap: a new model object is loaded in the background, then the
reference is swapped under the GIL. Old model stays alive for in-flight
inference; GC'd when no references remain.

Cold start: if no ONNX/heads model is available yet, falls back to a
keyword-based stub classifier so the gateway is operational. The stub is
low-quality but lets the rest of the system function. Stub is replaced
automatically once router_model/export_onnx.py has produced artifacts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("glint.router")

EMBED_DIM_DEFAULT = 384
PROJECTION_DIM = 64

# Synonym table for the stub classifier. Maps a vertical-name token (e.g.
# "programming", "math") to keywords that indicate that vertical. Keeps the
# stub functional before the real model is trained.
STUB_TOKEN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chat": ("hi", "hello", "hey", "how are you", "what's up", "chat", "conversation"),
    "programming": ("code", "function", "python", "rust", "class", "def ", "json", "debug", "implement", "program", "script", "api"),
    "math": ("equation", "calculate", "solve", "derivative", "integral", "theorem", "matrix", "math", "algebra", "calculus"),
    "reasoning": ("why", "how does", "explain", "compare", "analyze", "logical", "argument"),
    "trivia": ("trivia", "quiz", "fact", "history", "capital of"),
    "translation": ("translate", "translation", "in french", "in spanish", "翻译"),
    "summarization": ("summarize", "summary", "condense", "tl;dr"),
    "writing": ("write a", "essay", "email draft", "rewrite", "compose"),
    "medical": ("metformin", "symptom", "diagnosis", "dosage", "prescription", "side effect"),
    "codebase": ("codebase", "repository", "where is", "find all", "locate"),
    "debugging": ("debug", "race condition", "deadlock", "memory leak", "heisenbug"),
    "system_design": ("design a", "architecture", "distributed", "scalable", "load balancer"),
    "planning": ("plan", "roadmap", "milestone", "strategy", "break down"),
    "finance": ("stock", "portfolio", "budget", "revenue", "investment"),
    "health": ("workout", "diet", "calories", "exercise", "sleep"),
    "education": ("explain to me", "teach me", "tutor", "lesson"),
}  # fmt: off


@dataclass
class RouterOutput:
    text_hash: str
    text_preview: str
    text_chars: int
    vertical: str
    vertical_top2: list[tuple[str, float]]
    complexity: int
    flags: dict
    projection: list[float] | None = None
    model_version: str = "stub"
    ms_classify: float = 0.0
    ood_flag: bool = False
    ood_score: float = 0.0
    is_stub: bool = False
    extra: dict = field(default_factory=dict)


class _StubModel:
    """Keyword-based stub classifier used until ONNX + heads are trained.

    Quality is intentionally low. It exists so the gateway runs end-to-end
    while training data is being generated and the real model is being
    trained. Once router_model/export_onnx.py produces artifacts, the real
    model is loaded and the stub is replaced.
    """

    def __init__(self, vertical_names: list[str]):
        self.vertical_names = list(vertical_names)
        self.version = "stub-v0"
        self.created_at = time.time()
        # Keyword map: token -> vertical. Built from the vertical names so the
        # stub can make *some* guess even with zero external config.
        self._keyword_map: dict[str, tuple[str, ...]] = {}
        for v in self.vertical_names:
            tokens = re.findall(r"[a-z]{3,}", v.lower())
            keywords: set[str] = set()
            for token in tokens:
                if token not in ("other", "unknown", "general"):
                    keywords.update(STUB_TOKEN_KEYWORDS.get(token, ()))
                    keywords.add(token)
            if keywords:
                self._keyword_map[v] = tuple(keywords)

    def predict(self, text: str, projection_target: int = PROJECTION_DIM) -> RouterOutput:
        text_lower = text.lower()
        text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

        # Keyword-based vertical guess
        scores = {v: 0.0 for v in self.vertical_names}
        for vert, keywords in self._keyword_map.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits:
                scores[vert] = scores.get(vert, 0.0) + hits
        if not any(scores.values()):
            scores["other"] = 1.0
        total = sum(scores.values()) or 1.0
        norm = {k: v / total for k, v in scores.items()}
        sorted_v = sorted(norm.items(), key=lambda x: -x[1])
        top_vertical = sorted_v[0][0]
        top1_prob = sorted_v[0][1]

        # Complexity heuristics
        words = text.split()
        complexity = 1
        if len(words) > 50:
            complexity = 3
        elif len(words) > 20:
            complexity = 2
        if any(w in text_lower for w in ["prove", "design", "implement", "complex", "optimize", "distributed"]):
            complexity = min(5, complexity + 2)
        elif any(w in text_lower for w in ["explain", "compare", "analyze"]):
            complexity = min(5, complexity + 1)

        flags = {
            "code": any(w in text_lower for w in ["code", "function", "python", "rust", "class", "def ", "algorithm"]),
            "math": any(w in text_lower for w in ["equation", "calculate", "solve", "derivative", "integral", "theorem"]),
            "reasoning": any(w in text_lower for w in ["why", "how does", "explain", "analyze", "compare"]),
            "long_output": len(words) > 100 or any(w in text_lower for w in ["write a", "generate", "comprehensive"]),
        }

        # OOD: confidence low -> flag
        ood_flag = top1_prob < 0.25
        ood_score = 1.0 - top1_prob

        return RouterOutput(
            text_hash=text_hash,
            text_preview=text[:200],
            text_chars=len(text),
            vertical=top_vertical,
            vertical_top2=sorted_v[:2],
            complexity=complexity,
            flags=flags,
            projection=None,
            model_version=self.version,
            ms_classify=0.5,
            ood_flag=ood_flag,
            ood_score=ood_score,
            is_stub=True,
            extra={"stub_reason": "real model not trained yet"},
        )


class _RealModel:
    """Wraps ONNX embedding + heads numpy inference."""

    def __init__(
        self,
        embedding_session,
        heads_weights: dict[str, np.ndarray],
        heads_metadata: dict,
        vertical_names: list[str],
        calibration_temperature: float = 1.0,
    ):
        self.embedding_session = embedding_session
        self.heads_weights = heads_weights
        self.heads_metadata = heads_metadata
        self.vertical_names = vertical_names
        self.calibration_temperature = calibration_temperature
        self.version = heads_metadata.get("version", "real-v?")
        self.created_at = time.time()
        self._lock = threading.Lock()
        # Inference cache: text_hash -> (ts, RouterOutput) (capped + TTL)
        self._cache: dict[str, tuple[float, RouterOutput]] = {}
        self._cache_max = heads_metadata.get("cache_size", 10000)
        self._cache_ttl = float(heads_metadata.get("cache_ttl_seconds", 3600))

    def predict(self, text: str, projection_target: int = PROJECTION_DIM) -> RouterOutput:
        text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        now = time.time()
        with self._lock:
            cached = self._cache.get(text_hash)
            if cached is not None and (now - cached[0]) < self._cache_ttl:
                return cached[1]

        emb = self._encode(text)
        heads_out = self._run_heads(emb)
        out = self._build_output(text, text_hash, heads_out)

        with self._lock:
            if len(self._cache) >= self._cache_max:
                # FIFO eviction: drop oldest 10%
                for k in list(self._cache.keys())[: self._cache_max // 10]:
                    del self._cache[k]
            self._cache[text_hash] = (now, out)
        return out

    def _encode(self, text: str) -> np.ndarray:
        """Run embedding model on the input text. Returns 1D vector."""
        try:
            tok = self.embedding_session["tokenizer"]
            inputs = tok(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            outputs = self.embedding_session["session"].run(None, onnx_inputs)
            emb = outputs[0]
            # Mean pool over non-padding tokens
            mask = inputs["attention_mask"].astype(np.float32)
            emb = (emb * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)
            norm = np.linalg.norm(emb, axis=-1, keepdims=True)
            return (emb / np.maximum(norm, 1e-9))[0]
        except Exception as e:
            raise RuntimeError(f"embedding encode failed: {e}") from e

    def _run_heads(self, emb: np.ndarray) -> dict[str, np.ndarray]:
        """Run classifier heads. Returns dict of {head_name: np.ndarray}.

        Architecture: shared trunk (Linear 384->hidden, ReLU) -> per-task
        linear heads. ReLU (not GELU/tanh) is used deliberately so this numpy
        forward pass is an EXACT match for the PyTorch training graph in
        router_model/train.py — no activation-approximation drift between
        train and inference. MUST stay in sync with train.py's model and
        router_model/eval.py's predict_with_heads (AGENTS.md invariant).
        """
        out = {}
        # Shared trunk
        W_t = self.heads_weights["W_trunk1"]
        b_t = self.heads_weights["b_trunk1"]
        trunk = emb @ W_t.T + b_t
        trunk = np.maximum(trunk, 0.0)  # ReLU

        # Vertical softmax: W_v @ trunk + b_v
        W_v = self.heads_weights["W_vertical"]
        b_v = self.heads_weights["b_vertical"]
        logits_v = trunk @ W_v.T + b_v
        # Apply temperature
        logits_v = logits_v / max(self.calibration_temperature, 1e-6)
        # Stable softmax
        logits_v = logits_v - logits_v.max()
        exp_v = np.exp(logits_v)
        out["vertical_probs"] = exp_v / exp_v.sum()

        # Complexity ordinal: cumulative sigmoid heads
        W_c = self.heads_weights["W_complexity"]
        b_c = self.heads_weights["b_complexity"]
        logits_c = trunk @ W_c.T + b_c  # shape (5,)
        # Convert to ordinal cumulative: sigmoid of each threshold
        cum = 1.0 / (1.0 + np.exp(-logits_c))
        out["complexity_probs"] = cum

        # Binary flags
        for fname in ["code", "math", "reasoning", "long_output"]:
            W = self.heads_weights[f"W_{fname}"]
            b = self.heads_weights[f"b_{fname}"]
            logit = float(trunk @ W.T + b)
            out[f"{fname}_prob"] = 1.0 / (1.0 + math.exp(-logit))

        # Projection head (64-d for prototypes)
        W_p = self.heads_weights.get("W_projection")
        b_p = self.heads_weights.get("b_projection")
        if W_p is not None and b_p is not None:
            proj = trunk @ W_p.T + b_p
            n = np.linalg.norm(proj)
            out["projection"] = proj / max(n, 1e-9)
        else:
            out["projection"] = None

        return out

    def _build_output(self, text: str, text_hash: str, heads_out: dict) -> RouterOutput:
        v_probs = heads_out["vertical_probs"]
        if not self.vertical_names:
            # No verticals configured — safe fallback
            return RouterOutput(
                text_hash=text_hash,
                text_preview=text[:200],
                text_chars=len(text),
                vertical="other",
                vertical_top2=[("other", 1.0)],
                complexity=int(np.clip(int((heads_out["complexity_probs"] > 0.5).sum()), 1, 5)),
                flags={
                    "code": heads_out["code_prob"] > 0.5,
                    "math": heads_out["math_prob"] > 0.5,
                    "reasoning": heads_out["reasoning_prob"] > 0.5,
                    "long_output": heads_out["long_output_prob"] > 0.5,
                },
                projection=None,
                model_version=self.version,
                ms_classify=0.0,
                ood_flag=True,
                ood_score=1.0,
                is_stub=False,
            )
        sorted_idx = np.argsort(-v_probs)
        vertical = self.vertical_names[sorted_idx[0]]
        top2 = [(self.vertical_names[i], float(v_probs[i])) for i in sorted_idx[:2]]
        top1_prob = float(v_probs[sorted_idx[0]])

        # Complexity from cumulative ordinal heads (P(cx >= k), k=1..5).
        # MUST match router_model/eval.py: complexity = count of heads > 0.5,
        # clamped to [1, 5]. argmax is WRONG here (head 0 is almost always
        # highest, so argmax would collapse everything to cx=1).
        c_probs = heads_out["complexity_probs"]
        complexity = int(np.clip(int((c_probs > 0.5).sum()), 1, 5))

        flags = {
            "code": heads_out["code_prob"] > 0.5,
            "math": heads_out["math_prob"] > 0.5,
            "reasoning": heads_out["reasoning_prob"] > 0.5,
            "long_output": heads_out["long_output_prob"] > 0.5,
        }

        # OOD detection: max prob below threshold
        ood_threshold = 0.25
        ood_flag = top1_prob < ood_threshold
        ood_score = 1.0 - top1_prob

        return RouterOutput(
            text_hash=text_hash,
            text_preview=text[:200],
            text_chars=len(text),
            vertical=vertical,
            vertical_top2=top2,
            complexity=complexity,
            flags=flags,
            projection=heads_out["projection"].tolist() if heads_out["projection"] is not None else None,
            model_version=self.version,
            ms_classify=0.0,
            ood_flag=ood_flag,
            ood_score=ood_score,
            is_stub=False,
        )


class Router:
    """Top-level router manager. Atomic hot-swap of underlying model.

    The active model reference is held in self._model. On swap, a new model
    is loaded in background, then self._model is reassigned atomically.
    """

    def __init__(self):
        self._model: _StubModel | _RealModel | None = None
        self._lock = threading.RLock()
        self._model_path = None

    def init_stub(self, vertical_names: list[str]):
        """Initialize with stub model. Used at startup before any real model exists."""
        with self._lock:
            self._model = _StubModel(vertical_names)
            log.info("router initialized with stub model (verticals=%d)", len(vertical_names))

    def try_load_real(
        self,
        onnx_path: str,
        heads_path: str,
        vertical_names: list[str],
        calibration_temperature: float = 1.0,
        checksum_sha256: str | None = None,
        tokenizer_source: str | None = None,
    ):
        """Try to load a real ONNX + heads model from disk. If anything fails, keep current.

        When checksum_sha256 is provided (config embedding.checksum_sha256),
        the ONNX file is verified before load — mismatch refuses to load
        (documented contract: "mismatch = refuse to start").
        """
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            log.warning("real model deps not installed (%s); staying on stub", e)
            return False

        try:
            onnx_file = Path(onnx_path)
            heads_file = Path(heads_path)
            if not onnx_file.exists():
                log.info("onnx_path %s does not exist; staying on stub", onnx_file)
                return False
            if not heads_file.exists():
                log.info("heads_path %s does not exist; staying on stub", heads_file)
                return False

            # Checksum verification (documented contract: refuse on mismatch)
            if checksum_sha256 and not str(checksum_sha256).startswith("PLACEHOLDER"):
                import hashlib
                actual = hashlib.sha256(onnx_file.read_bytes()).hexdigest()
                if actual != checksum_sha256:
                    log.error(
                        "embedding ONNX checksum mismatch (expected %s, got %s); refusing to load",
                        checksum_sha256, actual,
                    )
                    return False

            log.info("loading real router model: onnx=%s heads=%s", onnx_file, heads_file)
            t0 = time.time()

            session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
            tok = AutoTokenizer.from_pretrained(tokenizer_source or onnx_file.parent)
            embedding_session = {"session": session, "tokenizer": tok}

            heads_npz = np.load(heads_file, allow_pickle=False)
            heads_weights = {k: heads_npz[k] for k in heads_npz.files if k.startswith(("W_", "b_"))}
            heads_metadata = {}
            if "metadata_json" in heads_npz.files:
                heads_metadata = json.loads(str(heads_npz["metadata_json"]))
            else:
                # Look for sidecar json
                sidecar = heads_file.with_suffix(".json")
                if sidecar.exists():
                    heads_metadata = json.loads(sidecar.read_text())

            metadata_verticals = heads_metadata.get("vertical_names")
            if metadata_verticals and list(metadata_verticals) != list(vertical_names):
                raise ValueError("heads taxonomy does not match configured vertical order")
            required_weights = {
                "W_trunk1", "b_trunk1",
                "W_vertical", "b_vertical", "W_complexity", "b_complexity",
                "W_code", "b_code", "W_math", "b_math", "W_reasoning", "b_reasoning",
                "W_long_output", "b_long_output",
            }
            missing = required_weights - heads_weights.keys()
            if missing:
                raise ValueError(f"heads artifact missing weights: {sorted(missing)}")
            if heads_weights["W_vertical"].shape[0] != len(vertical_names):
                raise ValueError("vertical head output size does not match taxonomy")
            if heads_weights["W_complexity"].shape[0] != 5:
                raise ValueError("complexity head must contain five ordinal thresholds")
            if heads_weights["W_trunk1"].shape[1] != EMBED_DIM_DEFAULT:
                raise ValueError(
                    f"trunk input dim {heads_weights['W_trunk1'].shape[1]} != "
                    f"embedding dim {EMBED_DIM_DEFAULT}"
                )
            hidden_dim = heads_weights["W_trunk1"].shape[0]
            for head_name in (
                "W_vertical", "W_complexity", "W_code", "W_math", "W_reasoning", "W_long_output",
            ):
                if heads_weights[head_name].shape[1] != hidden_dim:
                    raise ValueError(
                        f"{head_name} input dim {heads_weights[head_name].shape[1]} != "
                        f"trunk hidden dim {hidden_dim}"
                    )

            new_model = _RealModel(
                embedding_session=embedding_session,
                heads_weights=heads_weights,
                heads_metadata=heads_metadata,
                vertical_names=vertical_names,
                calibration_temperature=calibration_temperature,
            )

            # Atomic swap
            with self._lock:
                self._model = new_model
                self._model_path = (str(onnx_file), str(heads_file))
            log.info(
                "real model loaded in %.2fs; swapped atomically (version=%s)",
                time.time() - t0, new_model.version,
            )
            return True
        except Exception as e:
            log.error("failed to load real model: %s", e)
            return False

    def predict(self, text: str) -> RouterOutput:
        if not text or not text.strip():
            return _StubModel([]).predict(text)  # safe empty
        with self._lock:
            model = self._model
        if model is None:
            raise RuntimeError("router not initialized — call init_stub() first")
        t0 = time.time()
        out = model.predict(text)
        out.ms_classify = (time.time() - t0) * 1000.0
        return out

    def embed(self, text: str) -> np.ndarray | None:
        """Embed a text with the real model. Returns None while on the stub."""
        if not text or not text.strip():
            return None
        with self._lock:
            model = self._model
        if not isinstance(model, _RealModel):
            return None
        try:
            return model._encode(text)
        except Exception as e:
            log.warning("embed failed: %s", e)
            return None

    def is_stub(self) -> bool:
        with self._lock:
            return isinstance(self._model, _StubModel)

    def model_version(self) -> str:
        with self._lock:
            return self._model.version if self._model else "none"

    def artifact_paths(self) -> tuple[str, str] | None:
        with self._lock:
            return self._model_path


# Module-level singleton
_router: Router | None = None


def init_router(vertical_names: list[str]) -> Router:
    global _router
    _router = Router()
    _router.init_stub(vertical_names)
    return _router


def router() -> Router:
    if _router is None:
        raise RuntimeError("router not initialized — call init_router() first")
    return _router
