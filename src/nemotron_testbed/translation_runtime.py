from __future__ import annotations

import logging
import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable

from .domain import QueueFullError, SubmitStatus, TranslationJob, TranslationResult
from .languages import envit5_output_prefix, envit5_prefix
from .metrics import TRANSLATION_LATENCY, TRANSLATION_QUEUE, TRANSLATION_QUEUE_WAIT_MS
from .logging_config import quiet_transformers_logging

logger = logging.getLogger(__name__)


class TranslationRuntime:
    """Single-owner CTranslate2 EnViT5 runtime.

    Translation runs outside the ASR owner thread so slow NMT never blocks
    Nemotron streaming inference. Jobs are final-utterance only.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str,
        compute_type: str,
        inter_threads: int,
        intra_threads: int,
        queue_size: int,
        beam_size: int,
        max_decoding_length: int,
        vi_prefix: str,
        en_prefix: str,
        result_callback: Callable[[TranslationResult], None],
        worker_id: int = 0,
        batch_size: int = 1,
        batch_wait_ms: int = 0,
        third_party_log_level: str = "ERROR",
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.compute_type = compute_type
        self.inter_threads = inter_threads
        self.intra_threads = intra_threads
        self.beam_size = beam_size
        self.max_decoding_length = max_decoding_length
        self.vi_prefix = vi_prefix
        self.en_prefix = en_prefix
        self._result_callback = result_callback
        self.worker_id = worker_id
        self._max_batch_size = max(1, batch_size)
        self._batch_wait_s = max(0, batch_wait_ms) / 1000.0
        self._third_party_log_level = third_party_log_level

        self._jobs: queue.Queue[TranslationJob | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run, name=f"envit5-translation-runtime-{worker_id}", daemon=True
        )
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._shutdown_requested = threading.Event()
        self._startup_error: BaseException | None = None
        self.translator = None
        self.tokenizer = None

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def start(self, timeout: float = 180.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out while loading EnViT5 CTranslate2 runtime")
        if self._startup_error is not None:
            raise RuntimeError("Translation runtime failed to start") from self._startup_error

    def submit_nowait(self, job: TranslationJob) -> None:
        if self._stopped.is_set():
            raise QueueFullError("translation runtime is stopped")
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise QueueFullError("translation runtime queue is full") from exc
        TRANSLATION_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())

    def try_submit_nowait(self, job: TranslationJob) -> SubmitStatus:
        if self._stopped.is_set():
            return SubmitStatus.UNAVAILABLE
        try:
            self._jobs.put_nowait(job)
            TRANSLATION_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())
            return SubmitStatus.ACCEPTED
        except queue.Full:
            return SubmitStatus.OVERLOADED

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
                logger.warning("Could not enqueue translation shutdown sentinel")
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(
                "TranslationRuntime worker=%s did not stop in %.1fs",
                self.worker_id,
                timeout,
            )
        self._stopped.set()

    def _emit(self, result: TranslationResult) -> None:
        try:
            self._result_callback(result)
        except Exception:
            logger.exception("Translation result callback failed")

    def _load(self) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        # Transformers can install its own stderr handler at import time. Apply
        # the configured quiet level again after import so tokenizer warnings do
        # not leak to console.
        quiet_transformers_logging(self._third_party_log_level)

        model_dir = Path(self.model_path)
        tokenizer_dir = Path(self.tokenizer_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"TRANSLATION_MODEL_PATH is not a directory: {model_dir}")
        if not tokenizer_dir.exists():
            raise FileNotFoundError(
                f"TRANSLATION_TOKENIZER_PATH does not exist: {tokenizer_dir}"
            )

        logger.info(
            "Loading EnViT5 CTranslate2 worker=%s model=%s device=%s compute_type=%s",
            self.worker_id,
            model_dir,
            self.device,
            self.compute_type,
        )
        # CTranslate2 uses device="cuda" + device_index=N separately.
        # PyTorch-style "cuda:0" is not valid for CTranslate2.
        if ":" in self.device:
            ct2_device, _idx = self.device.split(":", 1)
            ct2_device_index = int(_idx) if _idx.isdigit() else 0
        else:
            ct2_device = self.device
            ct2_device_index = 0

        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=ct2_device,
            device_index=ct2_device_index,
            compute_type=self.compute_type,
            inter_threads=max(1, self.inter_threads),
            intra_threads=max(1, self.intra_threads),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
            # EnViT5 uses the T5 tokenizer. Passing the legacy mode explicitly
            # avoids Transformers warning that `legacy=None` defaults to True.
            legacy=True,
        )
        logger.info("EnViT5 translation runtime ready: worker=%s", self.worker_id)

    def _run(self) -> None:
        try:
            self._load()
        except BaseException as exc:
            self._startup_error = exc
            logger.exception("Translation startup failed")
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
            TRANSLATION_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())
            if job is None:
                self._jobs.task_done()
                break

            TRANSLATION_QUEUE_WAIT_MS.observe(
                (time.perf_counter() - job.enqueue_time) * 1000.0
            )
            batch: list[TranslationJob] = [job]
            shutdown_seen = False

            if self._max_batch_size > 1 and self._batch_wait_s > 0:
                deadline = time.perf_counter() + self._batch_wait_s
                while len(batch) < self._max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        extra = self._jobs.get(timeout=remaining)
                        TRANSLATION_QUEUE.labels(worker=str(self.worker_id)).set(
                            self._jobs.qsize()
                        )
                        if extra is None:
                            self._jobs.task_done()
                            shutdown_seen = True
                            break
                        TRANSLATION_QUEUE_WAIT_MS.observe(
                            (time.perf_counter() - extra.enqueue_time) * 1000.0
                        )
                        batch.append(extra)
                    except queue.Empty:
                        break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "POOL_TRACE TRANSLATE worker=%s batch=%s streams=%s",
                    self.worker_id,
                    len(batch),
                    [j.key.stream_id for j in batch],
                )

            started = time.perf_counter()
            self._process_batch(batch, started)

            if shutdown_seen:
                break

        self._stopped.set()
        logger.info("Translation runtime stopped: worker=%s", self.worker_id)

    def _process_batch(self, batch: list[TranslationJob], started: float) -> None:
        try:
            if len(batch) == 1:
                translated_list = [self._translate(batch[0])]
            else:
                translated_list = self._translate_many(batch)
            elapsed = time.perf_counter() - started
            elapsed_per_job = elapsed / max(1, len(batch))
            for job, translated in zip(batch, translated_list):
                TRANSLATION_LATENCY.observe(elapsed_per_job)
                self._emit(
                    TranslationResult(
                        key=job.key,
                        utterance_id=job.utterance_id,
                        source_text=job.source_text,
                        translated_text=translated,
                        source_language=job.source_language,
                        target_language=job.target_language,
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        segment_idx=job.segment_idx,
                        latency_ms=elapsed * 1000.0,
                        queue_ms=(started - job.enqueue_time) * 1000.0,
                    )
                )
        except Exception:
            # Batch call failed — fall back to per-job translation so one bad
            # input does not suppress results for the rest of the batch.
            logger.exception(
                "Translation batch failed (size=%s worker=%s), retrying individually",
                len(batch),
                self.worker_id,
            )
            for job in batch:
                job_started = time.perf_counter()
                try:
                    translated = self._translate(job)
                    elapsed = time.perf_counter() - job_started
                    TRANSLATION_LATENCY.observe(elapsed)
                    self._emit(
                        TranslationResult(
                            key=job.key,
                            utterance_id=job.utterance_id,
                            source_text=job.source_text,
                            translated_text=translated,
                            source_language=job.source_language,
                            target_language=job.target_language,
                            ts_start=job.ts_start,
                            ts_end=job.ts_end,
                            segment_idx=job.segment_idx,
                            latency_ms=elapsed * 1000.0,
                            queue_ms=(job_started - job.enqueue_time) * 1000.0,
                        )
                    )
                except Exception as exc:
                    logger.exception(
                        "Translation job failed",
                        extra={
                            "rpc_id": job.key.rpc_id,
                            "session_id": job.key.session_id,
                            "stream_id": job.key.stream_id,
                            "utterance_id": job.utterance_id,
                            "worker_id": self.worker_id,
                        },
                    )
                    self._emit(
                        TranslationResult(
                            key=job.key,
                            utterance_id=job.utterance_id,
                            source_text=job.source_text,
                            source_language=job.source_language,
                            target_language=job.target_language,
                            ts_start=job.ts_start,
                            ts_end=job.ts_end,
                            segment_idx=job.segment_idx,
                            latency_ms=(time.perf_counter() - job_started) * 1000.0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        finally:
            for _ in batch:
                self._jobs.task_done()

    def _tokenize(self, job: TranslationJob) -> list[str]:
        if self.tokenizer is None:
            raise RuntimeError("BUG: tokenizer is None in _tokenize")
        prefix = envit5_prefix(job.source_language, self.vi_prefix, self.en_prefix)
        model_input = f"{prefix}{job.source_text.strip()}"
        source_ids = self.tokenizer.encode(
            model_input,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
        )
        return self.tokenizer.convert_ids_to_tokens(source_ids)

    def _decode(self, hypothesis: list[str], target_language: str) -> str:
        if self.tokenizer is None:
            raise RuntimeError("BUG: tokenizer is None in _decode")
        output_ids = self.tokenizer.convert_tokens_to_ids(hypothesis)
        translated = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        lang_code = re.escape(envit5_output_prefix(target_language)[:-1])  # "en" or "vi"
        translated = re.sub(rf'(?i)^{lang_code}\s*:\s*', '', translated)
        return translated

    def _translate(self, job: TranslationJob) -> str:
        if self.translator is None:
            raise RuntimeError("BUG: translator is None in _translate")

        source_text = job.source_text.strip()
        if not source_text:
            return ""
        source_tokens = self._tokenize(job)
        results = self.translator.translate_batch(
            [source_tokens],
            beam_size=max(1, self.beam_size),
            max_decoding_length=max(8, self.max_decoding_length),
        )
        if not results or not results[0].hypotheses:
            logger.warning(
                "CTranslate2 returned no hypothesis: worker=%s text=%r",
                self.worker_id,
                source_text[:80],
            )
            return ""
        return self._decode(results[0].hypotheses[0], job.target_language)

    def _translate_many(self, batch: list[TranslationJob]) -> list[str]:
        if self.translator is None:
            raise RuntimeError("BUG: translator is None in _translate_many")

        # Empty-text jobs bypass the model; track their positions so results
        # align with the input batch after the model call.
        all_tokens: list[list[str]] = []
        model_indices: list[int] = []
        outputs: list[str] = [""] * len(batch)

        for i, job in enumerate(batch):
            if job.source_text.strip():
                model_indices.append(i)
                all_tokens.append(self._tokenize(job))

        if all_tokens:
            results = self.translator.translate_batch(
                all_tokens,
                beam_size=max(1, self.beam_size),
                max_decoding_length=max(8, self.max_decoding_length),
            )
            for result, i in zip(results, model_indices):
                if not result.hypotheses:
                    logger.warning(
                        "CTranslate2 returned no hypothesis: worker=%s text=%r",
                        self.worker_id,
                        batch[i].source_text[:80],
                    )
                    continue
                outputs[i] = self._decode(result.hypotheses[0], batch[i].target_language)

        return outputs
