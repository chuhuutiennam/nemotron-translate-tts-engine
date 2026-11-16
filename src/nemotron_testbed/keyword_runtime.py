from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

from .domain import KeywordJob, KeywordResult, QueueFullError, SubmitStatus, TopicLabel
from .logging_config import quiet_transformers_logging

logger = logging.getLogger(__name__)


class KeywordRuntime:
    """Single-owner semantic topic detector for ASR-final text (task=keyword).

    Same submit/shutdown contract as the translation/TTS runtimes: one worker
    thread owns the model, ``try_submit_nowait`` never blocks, every accepted
    job produces exactly one KeywordResult through ``result_callback`` (with
    ``error`` set when detection failed).

    The multilingual-e5-small model is loaded once at startup. Topic example
    embeddings are precomputed once and averaged into one normalized centroid
    per topic. Runtime jobs only embed the ASR-final text and compare it against
    the cached centroids with cosine similarity.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        threshold: float,
        topics: dict[str, dict[str, list[str]]],
        queue_size: int,
        result_callback: Callable[[KeywordResult], None],
        third_party_log_level: str = "ERROR",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.topics = topics
        self._result_callback = result_callback
        self._third_party_log_level = third_party_log_level

        self._jobs: queue.Queue[KeywordJob | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run, name="keyword-topic-runtime", daemon=True
        )
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._shutdown_requested = threading.Event()
        self._startup_error: BaseException | None = None

        self.tokenizer = None
        self.model = None
        self._torch = None
        self._topic_keys: list[tuple[TopicLabel, str]] = []
        self._topic_matrix = None

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def start(self, timeout: float = 180.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out while loading keyword/topic runtime")
        if self._startup_error is not None:
            raise RuntimeError("Keyword/topic runtime failed to start") from self._startup_error

    def try_submit_nowait(self, job: KeywordJob) -> SubmitStatus:
        if self._stopped.is_set():
            return SubmitStatus.UNAVAILABLE
        try:
            self._jobs.put_nowait(job)
            return SubmitStatus.ACCEPTED
        except queue.Full:
            return SubmitStatus.OVERLOADED

    def submit_nowait(self, job: KeywordJob) -> None:
        status = self.try_submit_nowait(job)
        if status is SubmitStatus.UNAVAILABLE:
            raise QueueFullError("keyword runtime is stopped")
        if status is SubmitStatus.OVERLOADED:
            raise QueueFullError("keyword runtime queue is full")

    def shutdown(self, timeout: float = 10.0) -> None:
        if self._stopped.is_set():
            return
        self._shutdown_requested.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                self._jobs.put(None, timeout=1.0)
            except queue.Full:
                logger.warning("Could not enqueue keyword runtime shutdown sentinel")
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error("KeywordRuntime did not stop in %.1fs", timeout)
        self._stopped.set()

    def _emit(self, result: KeywordResult) -> None:
        try:
            self._result_callback(result)
        except Exception:
            logger.exception("Keyword result callback failed")

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        quiet_transformers_logging(self._third_party_log_level)

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"KEYWORD_MODEL_PATH is not a directory: {model_dir}"
            )
        if not self.topics:
            raise ValueError("keyword topics config must contain at least one topic")

        cleaned_topics: dict[TopicLabel, dict[str, list[str]]] = {}
        for raw_label, raw_infos in self.topics.items():
            label_name = str(raw_label).strip().upper()
            if not label_name:
                continue
            try:
                label = TopicLabel[label_name]
            except KeyError as exc:
                allowed = ", ".join(x.name for x in TopicLabel if x is not TopicLabel.UNKNOWN)
                raise ValueError(
                    f"keyword topic label {label_name!r} is not defined in TopicLabel enum; "
                    f"allowed: {allowed}"
                ) from exc
            if label is TopicLabel.UNKNOWN:
                raise ValueError("UNKNOWN cannot be configured as a keyword topic")

            cleaned_infos: dict[str, list[str]] = {}
            for raw_info, raw_examples in raw_infos.items():
                info = str(raw_info).strip()
                examples = [str(x).strip() for x in raw_examples if str(x).strip()]
                if examples:
                    cleaned_infos[info] = examples
            if cleaned_infos:
                cleaned_topics[label] = cleaned_infos

        if not cleaned_topics:
            raise ValueError("keyword topics config has no non-empty topic examples")

        topic_count = sum(len(infos) for infos in cleaned_topics.values())
        logger.info(
            "Loading multilingual E5 keyword runtime model=%s device=%s topics=%s threshold=%.4f",
            model_dir, self.device, topic_count, self.threshold,
        )

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        topic_keys: list[tuple[TopicLabel, str]] = []
        centroids = []
        with torch.inference_mode():
            for label, infos in cleaned_topics.items():
                for info, examples in infos.items():
                    passage_texts = [f"passage: {text}" for text in examples]
                    embeddings = self._encode(passage_texts)
                    centroid = embeddings.mean(dim=0, keepdim=True)
                    centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
                    topic_keys.append((label, info))
                    centroids.append(centroid)

        self._topic_keys = topic_keys
        self._topic_matrix = torch.cat(centroids, dim=0)
        logger.info(
            "Keyword/topic runtime ready: topics=%s",
            [f"{label.name}/{info}" if info else label.name for label, info in self._topic_keys],
        )

    def _encode(self, texts: list[str]):
        if self.tokenizer is None or self.model is None or self._torch is None:
            raise RuntimeError("BUG: keyword model is not loaded")
        torch = self._torch
        batch = self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        outputs = self.model(**batch)
        token_embeddings = outputs.last_hidden_state
        attention_mask = batch["attention_mask"].unsqueeze(-1)
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _detect(self, text: str) -> tuple[TopicLabel, str, float]:
        if self._topic_matrix is None or self._torch is None:
            raise RuntimeError("BUG: keyword topic embeddings are not initialized")
        stripped = text.strip()
        if not stripped:
            return TopicLabel.UNKNOWN, "", 0.0

        with self._torch.inference_mode():
            query = self._encode([f"query: {stripped}"])
            scores = query @ self._topic_matrix.T
            best_score, best_index = scores[0].max(dim=0)
            score = float(best_score.item())
            label, info = self._topic_keys[int(best_index.item())]
            if score < self.threshold:
                return TopicLabel.UNKNOWN, "", score
            return label, info, score

    def _run(self) -> None:
        try:
            self._load()
        except BaseException as exc:
            self._startup_error = exc
            logger.exception("Keyword/topic runtime startup failed")
            self._ready.set()
            self._stopped.set()
            return

        self._ready.set()
        while True:
            try:
                job = self._jobs.get(timeout=5.0)
            except queue.Empty:
                if self._shutdown_requested.is_set():
                    break
                continue
            if job is None:
                self._jobs.task_done()
                break

            started = time.perf_counter()
            queue_ms = (started - job.enqueue_time) * 1000.0
            try:
                label, info, score = self._detect(job.source_text)
                self._emit(
                    KeywordResult(
                        key=job.key,
                        utterance_id=job.utterance_id,
                        source_text=job.source_text,
                        source_language=job.source_language,
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        asr_confidence=job.asr_confidence,
                        topic_label=label,
                        topic_info=info,
                        topic_score=score,
                        wall_start=job.wall_start,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        queue_ms=queue_ms,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "Keyword/topic detection failed",
                    extra={
                        "rpc_id": job.key.rpc_id,
                        "session_id": job.key.session_id,
                        "stream_id": job.key.stream_id,
                        "utterance_id": job.utterance_id,
                    },
                )
                self._emit(
                    KeywordResult(
                        key=job.key,
                        utterance_id=job.utterance_id,
                        source_text=job.source_text,
                        source_language=job.source_language,
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        asr_confidence=job.asr_confidence,
                        wall_start=job.wall_start,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        queue_ms=queue_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                self._jobs.task_done()

        self._stopped.set()
        logger.info("Keyword/topic runtime stopped")
