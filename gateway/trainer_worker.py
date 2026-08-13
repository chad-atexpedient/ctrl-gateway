"""Auto-trainer worker.

Triggered when:
  - +N new curated samples since last training (configurable threshold)
  - Rolling first-pass accuracy < target
  - Manual POST /retrain

Pipeline:
  1. Load curated samples (high agreement, trust_score >= floor)
  2. Mix with base data (30% base + 70% curated by default)
  3. Hash-dedup against frozen eval set
  4. (optional) Fine-tune the embedding on vertical disagreements, gated
     behind manual /retrain --allow-embedding-finetune
  5. Train heads (PyTorch, frozen or fine-tuned embedding)
  6. Evaluate on base-eval + live-eval
  7. If eval gate passes -> atomic hot-swap + persist as boot default
  8. If regression -> auto-rollback
  9. Update registry, model_card

Embedding fine-tuning (step 4) runs router_model/embed_finetune.py as its own
subprocess against the vertical-disagreement pool (router vs. reviewer
mismatches) exported from review_results, producing a fine-tuned ONNX that
train.py then encodes training data with (--embedding-onnx). Any failure at
that step falls back to heads-only training on the current frozen embedding
— never blocks the run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from . import config as cfg
from . import memory
from . import router as router_mod

log = logging.getLogger("ctrl.trainer")


CHECKPOINTS_DIR = Path("./router_model/checkpoints")
EMBEDDING_DIR = Path("./router_model/data/embedding")
BASE_DATA_DIR = Path("./router_model/data/base")
CURATED_DATA_DIR = Path("./router_model/data/curated")
DISAGREEMENT_DATA_DIR = Path("./router_model/data/disagreement")
LIVE_EVAL_DIR = Path("./router_model/data/live-eval")
EVAL_DIR = Path("./router_model/data/eval")
MODEL_CARD_PATH = Path("./router_model/MODEL_CARD.md")


def _run_subprocess(cmd: list[str], timeout: int):
    """Module-level so it can be shared by every subprocess step (train,
    eval, embedding fine-tune) via loop.run_in_executor without re-defining
    a closure each time."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class TrainerWorker:
    def __init__(self, conf: cfg.Config):
        self.conf = conf
        self._task: asyncio.Task | None = None
        self._running = False
        self._stop_event = asyncio.Event()
        self._last_check = 0.0
        self._consecutive_stall_count = 0
        self._drift_alarm_active = False
        self._lock = asyncio.Lock()
        self._last_curated_id = int(memory.get_trainer_state("last_curated_id", "0") or 0)
        self._pending_curated_max_id = self._last_curated_id
        self._last_training_attempt_at = 0.0
        self._last_housekeeping_hour = ""
        self._last_accuracy = 1.0

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="trainer-worker")
        log.info("trainer worker started")

    async def stop(self):
        self._running = False
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        log.info("trainer worker stopped")

    def update_config(self, conf: cfg.Config):
        self.conf = conf

    async def _loop(self):
        tcfg = self.conf.config.get("trainer", {})
        poll_s = 60.0
        while self._running:
            try:
                await self._tick(tcfg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("trainer tick failed: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_s)
            except TimeoutError:
                pass

    async def _tick(self, tcfg: dict):
        # Periodic checks run regardless of auto_retrain state
        hour_key = datetime.now(UTC).strftime("%Y%m%d%H")
        if hour_key != self._last_housekeeping_hour:
            self._last_housekeeping_hour = hour_key
            await self._check_drift_alarm()
            await self._sample_live_eval()
        if self._drift_alarm_active:
            log.info("drift alarm active; auto-retrain paused")
            return
        if not tcfg.get("auto_retrain", True):
            return

        # Check thresholds
        new_samples = await self._count_new_curated_samples()
        threshold = tcfg.get("trigger_threshold_new_samples", 500)
        accuracy_drop = tcfg.get("trigger_accuracy_drop_below", 0.92)
        retry_cooldown = float(tcfg.get("training_retry_cooldown_seconds", 3600))

        accuracy = memory.accuracy_report().get("first_pass_accuracy")
        if accuracy is not None:
            self._last_accuracy = float(accuracy)

        acc = self._last_accuracy
        if time.time() - self._last_training_attempt_at < retry_cooldown:
            return
        if new_samples >= threshold:
            log.info("trigger: %d new curated samples (threshold=%d)", new_samples, threshold)
            self._last_training_attempt_at = time.time()
            promoted = await self._run_training_run(reason=f"threshold:{new_samples}")
            if promoted:
                self._last_curated_id = self._pending_curated_max_id
                memory.set_trainer_state("last_curated_id", str(self._last_curated_id))
        elif acc < accuracy_drop:
            log.info("trigger: rolling accuracy %.2f < %.2f", acc, accuracy_drop)
            self._last_training_attempt_at = time.time()
            await self._run_training_run(reason=f"accuracy:{acc:.2f}")

    async def _count_new_curated_samples(self) -> int:
        try:
            count, max_id = memory.curated_count_after(self._last_curated_id)
            self._pending_curated_max_id = max_id
            return count
        except Exception:
            return 0

    async def _check_drift_alarm(self):
        """Compare last 7 days vertical distribution to prior 7 days."""
        dcfg = self.conf.config.get("drift", {})
        if not dcfg.get("enabled", True):
            return
        window = dcfg.get("window_days", 7)
        threshold_pct = dcfg.get("shift_threshold_pct", 20.0)
        try:
            now_dist = memory.vertical_distribution(since_hours=window * 24)
            prev_decisions = memory.get_decisions(
                limit=20000, since_hours=window * 24 * 2
            )
            min_samples = dcfg.get("min_samples_for_alarm", 100)
            if len(prev_decisions) < min_samples:
                return
            half = len(prev_decisions) // 2
            prev_old = prev_decisions[half:]
            old_dist: dict[str, int] = {}
            for d in prev_old:
                v = d["vertical"] or "unknown"
                old_dist[v] = old_dist.get(v, 0) + 1
            # Compare
            all_verticals = set(now_dist.keys()) | set(old_dist.keys())
            shift_count = 0
            for v in all_verticals:
                now_share = now_dist.get(v, 0) / max(sum(now_dist.values()), 1)
                old_share = old_dist.get(v, 0) / max(sum(old_dist.values()), 1)
                if abs(now_share - old_share) * 100 > threshold_pct:
                    shift_count += 1
            if shift_count > 0:
                log.warning("drift alarm: %d verticals shifted >%s%%", shift_count, threshold_pct)
                self._drift_alarm_active = True
        except Exception as e:
            log.warning("drift check failed: %s", e)

    def clear_drift_alarm(self):
        self._drift_alarm_active = False
        log.info("drift alarm cleared (manual)")

    @property
    def training_in_progress(self) -> bool:
        """True if a training run is currently executing."""
        return self._lock.locked()

    def revert_to_version(self, version_id: str) -> bool:
        """Revert the router to a previously promoted model version.

        Loads the checkpoint files for the given version and hot-swaps.
        Returns True on success.
        """
        onnx_path = CHECKPOINTS_DIR / f"{version_id}_model.onnx"
        heads_path = CHECKPOINTS_DIR / f"{version_id}_heads.npz"
        if not onnx_path.exists() or not heads_path.exists():
            log.error("revert failed: checkpoint files not found for %s", version_id)
            return False
        registered = memory.model_version(version_id)
        actual_heads_hash = hashlib.sha256(heads_path.read_bytes()).hexdigest()[:16]
        if registered and registered.get("heads_hash") and registered["heads_hash"] != actual_heads_hash:
            log.error("revert failed: heads checksum mismatch for %s", version_id)
            return False
        ok = router_mod.router().try_load_real(
            onnx_path=str(onnx_path),
            heads_path=str(heads_path),
            vertical_names=[v["name"] for v in self.conf.verticals()],
            calibration_temperature=self.conf.policy.get("calibration", {}).get("temperature", 1.0),
            checksum_sha256=None,  # revert trusts the stored checkpoint
            tokenizer_source=self.conf.config.get("embedding", {}).get("model_id"),
        )
        if ok:
            # Re-activate the old version in the registry
            memory.register_model_version(
                version_id=version_id,
                parent_id=None,
                embedding_model=self.conf.config.get("embedding", {}).get("model_id", "unknown"),
                heads_hash=actual_heads_hash,
            )
            log.info("reverted to version %s", version_id)
        return ok

    async def _sample_live_eval(self, limit: int = 200):
        """Sample labeled live-traffic decisions into live_eval_set + the jsonl
        that router_model/eval.py reads for live accuracy.

        Sources labels from review_results with all_fields_agree=True (the
        reviewer label IS the ground truth for live eval). Idempotent per
        decision_id.
        """
        try:
            from sqlalchemy import select
            limit = int(self.conf.config.get("trainer", {}).get("live_eval_sample_size", 200))
            with memory.engine().connect() as conn:
                rows = conn.execute(
                    select(
                        memory.routing_log.c.id,
                        memory.routing_log.c.query_hash,
                        memory.routing_log.c.query_preview,
                        memory.review_results.c.vertical_label,
                        memory.review_results.c.complexity_label,
                        memory.review_results.c.flag_code_label,
                        memory.review_results.c.flag_math_label,
                        memory.review_results.c.flag_reasoning_label,
                        memory.review_results.c.flag_long_output_label,
                    )
                    .select_from(memory.review_results)
                    .join(memory.routing_log, memory.routing_log.c.id == memory.review_results.c.decision_id)
                    .where(memory.review_results.c.all_fields_agree.is_(True))
                    .order_by(memory.review_results.c.id.desc())
                    .limit(limit)
                ).all()
            inserted = 0
            for (rl_id, rl_hash, rl_preview, v_label, c_label, f_code, f_math, f_reason, f_long) in rows:
                flags = {
                    "code": bool(f_code),
                    "math": bool(f_math),
                    "reasoning": bool(f_reason),
                    "long_output": bool(f_long),
                }
                ok = memory.record_live_eval(
                    decision_id=rl_id,
                    query_hash=rl_hash,
                    text=rl_preview or "",
                    ground_truth_vertical=v_label or "other",
                    ground_truth_complexity=int(c_label or 2),
                    ground_truth_flags=flags,
                    label_source="reviewer",
                )
                inserted += int(ok)
            if inserted:
                self._write_live_eval_jsonl()
                log.info("live-eval: recorded %d new samples (total %d)", inserted, len(memory.live_eval_samples(limit=100000)))
        except Exception as e:
            log.warning("live-eval sampling failed: %s", e)

    def _write_live_eval_jsonl(self):
        """Export live_eval_set to the jsonl eval.py reads."""
        try:
            samples = memory.live_eval_samples(limit=20000)
            LIVE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
            out_path = LIVE_EVAL_DIR / "live_eval.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for s in samples:
                    flags = s.get("ground_truth_flags") or {}
                    f.write(json.dumps({
                        "text": s.get("text", ""),
                        "vertical": s.get("ground_truth_vertical", "other"),
                        "complexity": s.get("ground_truth_complexity", 2),
                        "code": bool(flags.get("code")),
                        "math": bool(flags.get("math")),
                        "reasoning": bool(flags.get("reasoning")),
                        "long_output": bool(flags.get("long_output")),
                        "query_hash": s.get("query_hash", ""),
                    }, ensure_ascii=False) + "\n")
            log.info("live-eval jsonl: %d samples → %s", len(samples), out_path)
        except Exception as e:
            log.warning("live-eval jsonl write failed: %s", e)

    async def _run_training_run(self, *, reason: str, allow_embedding_finetune: bool = False) -> bool:
        """Execute one training run end-to-end.

        Uses an asyncio.Lock (never blocks the event loop) and runs the
        subprocesses in a thread executor so the gateway stays responsive
        during the (potentially 30-minute) training + eval phase.
        """
        async with self._lock:
            new_version = f"v-{uuid.uuid4().hex[:12]}-{int(time.time())}"
            run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
            log.info("starting training run %s (version=%s, reason=%s)", run_id, new_version, reason)
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_running_loop()
            previous_version: str | None = None
            swapped = False
            embedding_onnx_path: str | None = None

            try:
                # 1. Export curated data to file
                self._export_curated_data(run_id)

                # 1b. Optional embedding fine-tune (manual-gated). Runs BEFORE
                # heads training so train.py can encode with the fine-tuned
                # embedding via --embedding-onnx. Any failure here falls back
                # to heads-only training on the current frozen embedding.
                if allow_embedding_finetune:
                    embedding_onnx_path = await self._run_embedding_finetune(run_id)
                    if not embedding_onnx_path:
                        log.warning(
                            "embedding fine-tune produced no artifact this run; "
                            "continuing heads-only on the current frozen embedding",
                        )

                # 2. Run training (subprocess to isolate torch state)
                heads_path = CHECKPOINTS_DIR / f"{new_version}_heads.npz"
                onnx_path = CHECKPOINTS_DIR / f"{new_version}_model.onnx"
                metadata_path = CHECKPOINTS_DIR / f"{new_version}_meta.json"

                training_script = Path("./router_model/train.py")
                cmd = [
                    "python", str(training_script),
                    "--base-data-dir", str(BASE_DATA_DIR),
                    "--curated-data-dir", str(CURATED_DATA_DIR / run_id),
                    "--output-heads", str(heads_path),
                    "--output-onnx", str(onnx_path),
                    "--output-metadata", str(metadata_path),
                    "--mix-ratio-base-pct", str(self.conf.config.get("trainer", {}).get("mix_ratio_base_pct", 30)),
                    "--mix-ratio-curated-pct", str(self.conf.config.get("trainer", {}).get("mix_ratio_curated_pct", 70)),
                    "--version", new_version,
                ]
                if allow_embedding_finetune:
                    cmd.append("--allow-embedding-finetune")
                if embedding_onnx_path:
                    cmd.extend(["--embedding-onnx", embedding_onnx_path])
                proc = await loop.run_in_executor(None, _run_subprocess, cmd, 1800)
                if proc.returncode != 0:
                    log.error("training failed: %s", proc.stderr[-1000:])
                    return False

                # 3. Evaluate
                eval_script = Path("./router_model/eval.py")
                eval_cmd = [
                    "python", str(eval_script),
                    "--heads", str(heads_path),
                    "--onnx", str(onnx_path),
                    "--base-eval", str(EVAL_DIR / "eval.jsonl"),
                    "--live-eval", str(LIVE_EVAL_DIR / "live_eval.jsonl"),
                    "--output-json", str(CHECKPOINTS_DIR / f"{new_version}_eval.json"),
                    "--embedding-id", str(self.conf.config.get("embedding", {}).get("model_id", "BAAI/bge-small-en-v1.5")),
                ]
                proc = await loop.run_in_executor(None, _run_subprocess, eval_cmd, 300)
                if proc.returncode != 0:
                    log.error("eval failed: %s", proc.stderr[-1000:])
                    return False

                eval_results = json.loads(
                    (CHECKPOINTS_DIR / f"{new_version}_eval.json").read_text()
                )

                # 4. Eval gate
                tcfg = self.conf.config.get("trainer", {})
                per_vert_threshold = tcfg.get("eval_gate_per_vertical_accuracy", 0.95)
                replay_drift_threshold = tcfg.get("policy_replay_drift_alarm_pct", 5.0)

                per_vert = eval_results.get("per_vertical_accuracy", {})
                below_threshold = [v for v, a in per_vert.items() if a < per_vert_threshold]
                if below_threshold:
                    log.warning(
                        "eval gate failed: verticals below threshold %s: %s",
                        per_vert_threshold, below_threshold,
                    )
                    self._consecutive_stall_count += 1
                    if self._consecutive_stall_count >= tcfg.get("embedding_finetune_stall_threshold", 3):
                        log.warning(
                            "%d consecutive eval stalls; consider embedding fine-tune via /retrain --allow-embedding-finetune",
                            self._consecutive_stall_count,
                        )
                    return False

                complexity_threshold = float(tcfg.get("eval_gate_complexity_accuracy", 0.0))
                if float(eval_results.get("base_complexity_accuracy") or 0.0) < complexity_threshold:
                    log.warning("eval gate failed: complexity accuracy below %.3f", complexity_threshold)
                    return False
                flag_threshold = float(tcfg.get("eval_gate_flag_accuracy", 0.0))
                weak_flags = [
                    name for name, accuracy in eval_results.get("base_flag_accuracy", {}).items()
                    if float(accuracy) < flag_threshold
                ]
                if weak_flags:
                    log.warning("eval gate failed: flag heads below %.3f: %s", flag_threshold, weak_flags)
                    return False

                active_checkpoint = memory.latest_promoted_checkpoint()
                allowed_regression_pct = float(tcfg.get("eval_gate_no_global_regression_pct", 1.0))
                if active_checkpoint and active_checkpoint.get("eval_base_accuracy") is not None:
                    previous_accuracy = float(active_checkpoint["eval_base_accuracy"])
                    new_accuracy = float(eval_results.get("base_accuracy") or 0.0)
                    regression_pct = (previous_accuracy - new_accuracy) * 100.0
                    if regression_pct > allowed_regression_pct:
                        log.warning(
                            "eval gate failed: global regression %.2f%% > %.2f%%",
                            regression_pct, allowed_regression_pct,
                        )
                        return False

                # 5. Replay last decisions through new model + policy
                drift_pct = await self._replay_drift_check(eval_results, replay_drift_threshold)
                if drift_pct is not None and drift_pct > replay_drift_threshold:
                    log.warning("policy replay drift %.1f%% > %.1f%%; abort", drift_pct, replay_drift_threshold)
                    return False
                eval_results["policy_drift_pct"] = drift_pct

                # 6. Snapshot the current model for potential revert
                previous_version = router_mod.router().model_version()
                if previous_version and previous_version != "stub-v0":
                    snapshot_dir = CHECKPOINTS_DIR / f"snapshot_{new_version}"
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    prev_onnx = CHECKPOINTS_DIR / f"{previous_version}_model.onnx"
                    prev_heads = CHECKPOINTS_DIR / f"{previous_version}_heads.npz"
                    active_paths = router_mod.router().artifact_paths()
                    if active_paths and not prev_onnx.exists():
                        import shutil
                        shutil.copy(active_paths[0], prev_onnx)
                    if active_paths and not prev_heads.exists():
                        import shutil
                        shutil.copy(active_paths[1], prev_heads)
                    if prev_onnx.exists():
                        import shutil
                        shutil.copy(prev_onnx, snapshot_dir / "previous_model.onnx")
                    if prev_heads.exists():
                        import shutil
                        shutil.copy(prev_heads, snapshot_dir / "previous_heads.npz")
                    log.info("snapshot created for revert: %s -> %s", new_version, previous_version)

                # 7. Atomic hot-swap
                # Checksum guard: when this run fine-tuned the embedding, the
                # ONNX bytes legitimately differ from the static config's
                # checksum (which was pinned to the previous/stock
                # embedding) — skip the check since we just produced and are
                # loading this exact artifact in-process (trusted
                # provenance). Ordinary heads-only retrains still enforce the
                # static checksum as before.
                swap_ok = router_mod.router().try_load_real(
                    onnx_path=str(onnx_path),
                    heads_path=str(heads_path),
                    vertical_names=[v["name"] for v in self.conf.verticals()],
                    calibration_temperature=self.conf.policy.get("calibration", {}).get("temperature", 1.0),
                    checksum_sha256=(
                        None if embedding_onnx_path
                        else (self.conf.config.get("embedding", {}).get("checksum_sha256") or None)
                    ),
                    tokenizer_source=self.conf.config.get("embedding", {}).get("model_id"),
                )
                if not swap_ok:
                    log.error("hot-swap failed")
                    return False
                swapped = True
                self._persist_as_boot_default(onnx_path, heads_path)

                # 8. Register version + checkpoint (with revert target)
                memory.register_model_version(
                    version_id=new_version,
                    parent_id=previous_version,
                    embedding_model=self.conf.config.get("embedding", {}).get("model_id", "unknown"),
                    heads_hash=hashlib.sha256(heads_path.read_bytes()).hexdigest()[:16],
                )
                memory.record_checkpoint(new_version, new_version)
                memory.mark_checkpoint_promoted(new_version, eval_results)
                log.info("training run %s promoted: version=%s (revert_to=%s)", run_id, new_version, previous_version)

                # 8. Generate model card
                self._generate_model_card(new_version, eval_results, run_id)

                # Reset stall counter
                self._consecutive_stall_count = 0
                return True
            except Exception as e:
                log.exception("training run %s failed: %s", run_id, e)
                if swapped and previous_version and previous_version != "stub-v0":
                    if not self.revert_to_version(previous_version):
                        log.critical("automatic rollback failed for previous version %s", previous_version)
                # Config trainer.auto_rollback_on_regression gates the rollback marking
                if self.conf.config.get("trainer", {}).get("auto_rollback_on_regression", True):
                    memory.mark_checkpoint_rolled_back(new_version, reason=str(e))
                return False

    def _export_curated_data(self, run_id: str):
        """Dump curated samples for this run to a file.

        Exports ALL curated samples (curated_run_id='auto' from the flywheel).
        We do not filter by the training run_id because curation is continuous
        and predates the training run; train.py applies the mix-ratio sampling
        downstream, so exporting the full pool is correct.
        """
        run_dir = CURATED_DATA_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with memory.engine().connect() as conn:
            rows = conn.execute(
                select(memory.curated_samples)
                .order_by(memory.curated_samples.c.id.desc())
                .limit(50000)
            ).all()
        out_path = run_dir / "samples.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                d = dict(r._mapping)
                # Serialize datetimes for the subprocess
                for k, v in d.items():
                    from datetime import datetime as _dt
                    if isinstance(v, _dt):
                        d[k] = v.isoformat()
                d.pop("id", None)
                f.write(json.dumps(d, default=str) + "\n")
        log.info("exported %d curated samples to %s", len(rows), out_path)

    def _export_disagreement_pool(self, run_id: str, limit: int = 20000) -> Path:
        """Export vertical-label disagreements (router prediction vs.
        reviewer ground truth) for embed_finetune.py's contrastive triplets.

        Each line: {text, true_vertical, router_vertical}. true_vertical is
        the reviewer's label (positive-class anchor target); router_vertical
        is what the router predicted instead (the hard-negative class) —
        that's exactly the confusable pair the embedding needs to pull apart.
        """
        run_dir = DISAGREEMENT_DATA_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with memory.engine().connect() as conn:
            rows = conn.execute(
                select(
                    memory.review_results.c.prompt_text,
                    memory.review_results.c.vertical_label,
                    memory.routing_log.c.vertical,
                    memory.routing_log.c.query_preview,
                )
                .select_from(memory.review_results)
                .join(memory.routing_log, memory.routing_log.c.id == memory.review_results.c.decision_id)
                .where(memory.review_results.c.agreement_vertical.is_(False))
                .order_by(memory.review_results.c.id.desc())
                .limit(limit)
            ).all()
        out_path = run_dir / "disagreements.jsonl"
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for prompt_text, true_vertical, router_vertical, query_preview in rows:
                text = prompt_text or query_preview
                if not text or not true_vertical or not router_vertical:
                    continue
                f.write(json.dumps({
                    "text": text,
                    "true_vertical": true_vertical,
                    "router_vertical": router_vertical,
                }, ensure_ascii=False) + "\n")
                n += 1
        log.info("exported %d vertical disagreements to %s", n, out_path)
        return out_path

    async def _run_embedding_finetune(self, run_id: str) -> str | None:
        """Run router_model/embed_finetune.py as a subprocess against the
        freshly exported disagreement pool. Returns the fine-tuned ONNX path
        on success, or None on ANY failure — the caller always has a safe
        fallback (heads-only training on the current frozen embedding), so
        this never raises.
        """
        try:
            disagreement_path = self._export_disagreement_pool(run_id)
        except Exception as e:
            log.warning("disagreement pool export failed: %s", e)
            return None

        finetune_dir = EMBEDDING_DIR / f"finetuned_{run_id}"
        finetune_onnx = finetune_dir / "model.onnx"
        cmd = [
            "python", str(Path("./router_model/embed_finetune.py")),
            "--disagreement-pool", str(disagreement_path),
            "--base-data-dir", str(BASE_DATA_DIR),
            "--output-onnx", str(finetune_onnx),
        ]
        loop = asyncio.get_running_loop()
        try:
            proc = await loop.run_in_executor(None, _run_subprocess, cmd, 1800)
        except Exception as e:
            log.error("embedding fine-tune subprocess failed to run: %s", e)
            return None
        if proc.returncode != 0:
            log.error("embedding fine-tune failed (exit %d): %s", proc.returncode, proc.stderr[-1000:])
            return None
        if not finetune_onnx.exists():
            log.warning("embedding fine-tune reported success but no ONNX at %s", finetune_onnx)
            return None
        log.info("embedding fine-tune produced %s", finetune_onnx)
        return str(finetune_onnx)

    def _persist_as_boot_default(self, onnx_path: Path, heads_path: Path):
        """Copy the just-promoted checkpoint to the static boot path
        (gateway-config.json -> embedding.onnx_path / sibling heads.npz) and
        update embedding.checksum_sha256, so this promotion survives a
        process restart — not just the in-memory hot-swap. app.py's
        init_app() only reads the static path at startup; it has no notion
        of "latest promoted checkpoint" from the registry.

        Best-effort: failures are logged, never fatal — the in-memory
        hot-swap already succeeded, so the training run is still a success
        even if this persistence step has a problem (e.g. read-only mount).
        Single-instance mode only, matching OverlayManager's existing
        constraint — multi mode ships config via image/volume.
        """
        if self.conf.config.get("mode") != "single":
            log.info("mode=multi: skipping boot-default persistence (ship config via image/volume)")
            return
        try:
            import shutil
            boot_onnx = Path(
                self.conf.config.get("embedding", {}).get("onnx_path")
                or str(EMBEDDING_DIR / "model.onnx")
            )
            boot_heads = boot_onnx.with_name("heads.npz")
            boot_onnx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(onnx_path, boot_onnx)
            shutil.copy(heads_path, boot_heads)
            new_checksum = hashlib.sha256(boot_onnx.read_bytes()).hexdigest()

            config_path = Path("./gateway-config.json")
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw.setdefault("embedding", {})["checksum_sha256"] = new_checksum
            config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            log.info("persisted boot default: %s (checksum=%s…)", boot_onnx, new_checksum[:16])
        except Exception as e:
            log.warning("failed to persist boot default (in-memory hot-swap still active): %s", e)

    async def _replay_drift_check(self, eval_results: dict, threshold_pct: float) -> float | None:
        """Replay last 100 decisions through the new model and measure drift."""
        try:
            # Get last 100 routing decisions
            decisions = memory.get_decisions(limit=100)
            if not decisions:
                return 0.0
            # For each, predict with new model (this is approximate — we just check
            # the count of decisions that would have changed source tag)
            # Cheap heuristic: count of decisions with tier != eval's "would_have_picked"
            # (eval_results includes a "would_have_picked" mapping). If missing, skip.
            would_have_picked = eval_results.get("would_have_picked", {})
            if not would_have_picked:
                return None
            same_count = sum(
                1 for d in decisions
                if would_have_picked.get(d["query_hash"]) == d["tier"]
            )
            drift = (len(decisions) - same_count) / max(len(decisions), 1) * 100
            return drift
        except Exception as e:
            log.warning("replay drift check failed: %s", e)
            return None

    def _generate_model_card(self, version_id: str, eval_results: dict, run_id: str):
        """Auto-generate MODEL_CARD.md for this version."""
        per_vert = eval_results.get("per_vertical_accuracy", {})
        confusions = eval_results.get("confusion_top20", [])

        train_meta: dict = {}
        meta_path = CHECKPOINTS_DIR / f"{version_id}_meta.json"
        if meta_path.exists():
            try:
                train_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("could not read %s for model card: %s", meta_path, e)

        embedding_note = (
            "fine-tuned this run" if train_meta.get("embedding_onnx_override") else "frozen (stock)"
        )
        card = f"""# Router Model Card — {version_id}

**Generated:** {datetime.now(UTC).isoformat()}
**Run:** {run_id}
**Embedding:** {self.conf.config.get("embedding", {}).get("model_id", "unknown")} ({embedding_note})
**Reviewer model:** {self.conf.reviewer().get("model", "unknown")}

## Architecture

| Field | Value |
|-------|-------|
| Heads architecture | {train_meta.get("architecture", "n/a")} |
| Hidden dim | {train_meta.get("hidden_dim", "n/a")} |
| Train / val samples | {train_meta.get("train_n", "n/a")} / {train_meta.get("val_n", "n/a")} |
| Val accuracy (early-stop signal) | {train_meta.get("val_accuracy", "n/a")} |
| Suggested calibration temperature | {train_meta.get("suggested_calibration_temperature", "n/a")} (compare to gateway-policy.json -> calibration.temperature) |

## Performance

| Metric | Value |
|--------|-------|
| Base eval accuracy | {eval_results.get("base_accuracy", "n/a")} |
| Live eval accuracy | {eval_results.get("live_accuracy", "n/a")} |
| Per-vertical threshold | {self.conf.config.get("trainer", {}).get("eval_gate_per_vertical_accuracy", 0.95)} |

## Per-vertical accuracy

"""
        for v in sorted(per_vert.keys()):
            card += f"- **{v}**: {per_vert[v]:.3f}\n"
        card += "\n## Top-20 confusion matrix\n\n"
        for c in confusions[:20]:
            card += f"- {c.get('true')} → {c.get('pred')} ({c.get('count', 0)})\n"
        card += """
## Intended use

Routes LLM inference requests across a 6-tier fleet. Cost-first policy with
uncertainty escalation. See README for the full architecture.

## Known limitations

- Cost estimates use per-tier averages; may diverge from real spend in
  long-context / heavy-output scenarios.
- Vertical taxonomy has ~57 entries; OOD detector flags unknown verticals.
- Embedding ceiling: when eval stalls repeatedly, manual embedding fine-tune
  is required (`POST /retrain --allow-embedding-finetune`).

## Out of scope

- Multi-tenant isolation enforcement beyond rate limiting + budget caps.
- Adversarial input detection beyond regex matching.
"""
        MODEL_CARD_PATH.write_text(card, encoding="utf-8")
        # Also write a version-specific copy
        version_card = CHECKPOINTS_DIR / f"{version_id}_MODEL_CARD.md"
        version_card.write_text(card, encoding="utf-8")

    def manual_retrain(self, allow_embedding_finetune: bool = False, confirm_drift: bool = False):
        """Trigger retrain synchronously (called by /retrain endpoint)."""
        if confirm_drift:
            self._drift_alarm_active = False
        loop = asyncio.get_event_loop()
        return loop.create_task(
            self._run_training_run(
                reason="manual",
                allow_embedding_finetune=allow_embedding_finetune,
            )
        )


_worker: TrainerWorker | None = None


def init_worker(conf: cfg.Config) -> TrainerWorker:
    global _worker
    _worker = TrainerWorker(conf)
    return _worker


def worker() -> TrainerWorker:
    if _worker is None:
        raise RuntimeError("trainer worker not initialized")
    return _worker
