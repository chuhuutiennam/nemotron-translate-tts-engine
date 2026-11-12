from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Callable

from .domain import (
    QueueFullError,
    RuntimeCommand,
    RuntimeCommandType,
    RuntimeResult,
    RuntimeResultType,
    StreamKey,
    SubmitStatus,
    TranslationJob,
    TranslationResult,
    TtsJob,
    TtsResult,
)
from .runtime import NemotronRuntime
from .translation_runtime import TranslationRuntime
from .tts_runtime import TtsRuntime
from .magpie_tts_runtime import MagpieTtsRuntime
from .metrics import (
    ASR_POOL_WORKERS,
    ASR_STREAM_AFFINITIES,
    TRANSLATION_POOL_WORKERS,
    TTS_POOL_WORKERS,
)

logger = logging.getLogger(__name__)


class NemotronRuntimePool:
    """Pool of stateful Nemotron runtimes with strict stream affinity.

    A START command assigns one logical stream/utterance to exactly one runtime.
    Every STEP/FINALIZE/CANCEL for that stream is routed to the same owner until
    FINAL or CANCELLED is emitted. Each worker owns its own model replica and
    all encoder/RNNT caches for the streams assigned to it. A worker emits
    CANCELLED only after it has dropped the utterance's queued commands and
    freed its state, so the affinity is released last.
    """

    def __init__(
        self,
        *,
        workers: int,
        model_path: str,
        devices: list[str],
        language: str,
        left_context: int,
        lookahead_tokens: int,
        queue_size: int,
        warmup_seconds: float,
        result_callback: Callable[[RuntimeResult], None],
        suppress_model_init_output: bool = True,
        beam_size: int = 1,
        runtime_factory: Callable[..., NemotronRuntime] = NemotronRuntime,
    ) -> None:
        if workers < 1:
            raise ValueError("ASR_WORKERS must be >= 1")
        self.workers = workers
        self._result_callback = result_callback
        self._lock = threading.Lock()
        # Affinity key includes utterance_id so two concurrent utterances on the
        # same logical stream can be assigned to different workers independently.
        self._owners: dict[tuple[StreamKey, int], int] = {}
        self._rr = 0
        ASR_POOL_WORKERS.set(workers)
        ASR_STREAM_AFFINITIES.set(0)
        self._runtimes = [
            runtime_factory(
                model_path=model_path,
                device=devices[index % len(devices)],
                language=language,
                left_context=left_context,
                lookahead_tokens=lookahead_tokens,
                queue_size=queue_size,
                warmup_seconds=warmup_seconds,
                result_callback=self._on_worker_result,
                suppress_model_init_output=suppress_model_init_output,
                beam_size=beam_size,
                worker_id=index,
            )
            for index in range(workers)
        ]

    @property
    def queue_depth(self) -> int:
        return sum(runtime.queue_depth for runtime in self._runtimes)

    @property
    def active_owners(self) -> int:
        with self._lock:
            return len(self._owners)

    @property
    def pending_cancellations(self) -> int:
        """CANCELs recorded by the workers and not yet cleaned up."""
        return sum(runtime.pending_cancellations for runtime in self._runtimes)

    def _cancel_without_owner(self, command: RuntimeCommand) -> bool:
        """A CANCEL with no owner (no START was assigned, or FINAL/CANCELLED already released it) has nothing to
        clean up: confirm it here. False for any other command."""
        if command.type is not RuntimeCommandType.CANCEL or command.key is None:
            return False
        self._result_callback(
            RuntimeResult(
                type=RuntimeResultType.CANCELLED,
                key=command.key,
                utterance_id=command.utterance_id,
            )
        )
        return True

    def start(self, timeout: float = 300.0) -> None:
        started: list[NemotronRuntime] = []
        try:
            for runtime in self._runtimes:
                runtime.start(timeout=timeout)
                started.append(runtime)
        except Exception:
            for runtime in reversed(started):
                try:
                    runtime.shutdown()
                except Exception:
                    logger.exception("Failed to stop ASR worker after pool startup failure")
            raise
        logger.info("Nemotron runtime pool ready: workers=%s", self.workers)

    def shutdown(self, timeout: float = 10.0) -> None:
        per_worker_timeout = max(2.0, timeout / max(1, self.workers))
        for runtime in self._runtimes:
            try:
                runtime.shutdown(timeout=per_worker_timeout)
            except Exception:
                logger.exception("Failed to stop ASR worker")
        with self._lock:
            self._owners.clear()
            ASR_STREAM_AFFINITIES.set(0)
        logger.info("Nemotron runtime pool stopped")

    def _select_worker_locked(self) -> int:
        owner_counts = Counter(self._owners.values())
        start = self._rr
        candidates = []
        for offset in range(self.workers):
            index = (start + offset) % self.workers
            runtime = self._runtimes[index]
            candidates.append((runtime.queue_depth, owner_counts[index], offset, index))
        _, _, _, selected = min(candidates)
        self._rr = (selected + 1) % self.workers
        return selected

    def _owner_for(self, command: RuntimeCommand) -> int | None:
        if command.key is None:
            return None
        affinity_key = (command.key, command.utterance_id)
        with self._lock:
            if command.type is RuntimeCommandType.START:
                existing = self._owners.get(affinity_key)
                if existing is not None:
                    return existing
                owner = self._select_worker_locked()
                self._owners[affinity_key] = owner
                ASR_STREAM_AFFINITIES.set(len(self._owners))
                return owner
            return self._owners.get(affinity_key)

    def submit_nowait(self, command: RuntimeCommand) -> None:
        if command.type is RuntimeCommandType.SHUTDOWN:
            raise RuntimeError("SHUTDOWN is managed by NemotronRuntimePool.shutdown()")

        owner = self._owner_for(command)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "POOL_TRACE ASR route worker=%s command=%s rpc=%s session=%s stream=%s utterance=%s",
                owner,
                command.type.name,
                command.key.rpc_id if command.key else "",
                command.key.session_id if command.key else "",
                command.key.stream_id if command.key else "",
                command.utterance_id,
            )

        if owner is None:
            if self._cancel_without_owner(command):
                return
            raise RuntimeError(
                f"No ASR runtime owner for command={command.type.name} key={command.key}"
            )

        try:
            self._runtimes[owner].submit_nowait(command)
        except Exception:
            if command.type is RuntimeCommandType.START and command.key is not None:
                affinity_key = (command.key, command.utterance_id)
                with self._lock:
                    if self._owners.get(affinity_key) == owner:
                        self._owners.pop(affinity_key, None)
                        ASR_STREAM_AFFINITIES.set(len(self._owners))
            raise

    def try_submit_nowait(self, command: RuntimeCommand) -> SubmitStatus:
        if command.type is RuntimeCommandType.SHUTDOWN:
            raise RuntimeError("SHUTDOWN is managed by NemotronRuntimePool.shutdown()")

        owner = self._owner_for(command)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "POOL_TRACE ASR route worker=%s command=%s rpc=%s session=%s stream=%s utterance=%s",
                owner,
                command.type.name,
                command.key.rpc_id if command.key else "",
                command.key.session_id if command.key else "",
                command.key.stream_id if command.key else "",
                command.utterance_id,
            )

        if owner is None:
            if self._cancel_without_owner(command):
                return SubmitStatus.ACCEPTED
            return SubmitStatus.UNAVAILABLE

        status = self._runtimes[owner].try_submit_nowait(command)
        if status is not SubmitStatus.ACCEPTED and command.type is RuntimeCommandType.START and command.key is not None:
            affinity_key = (command.key, command.utterance_id)
            with self._lock:
                if self._owners.get(affinity_key) == owner:
                    self._owners.pop(affinity_key, None)
                    ASR_STREAM_AFFINITIES.set(len(self._owners))
        return status

    def submit(self, command: RuntimeCommand, timeout: float = 2.0) -> None:
        # Coordinator uses submit_nowait; keep this compatibility method for tools/tests.
        if command.type is RuntimeCommandType.SHUTDOWN:
            raise RuntimeError("SHUTDOWN is managed by NemotronRuntimePool.shutdown()")
        owner = self._owner_for(command)
        if owner is None:
            if self._cancel_without_owner(command):
                return
            raise RuntimeError(
                f"No ASR runtime owner for command={command.type.name} key={command.key}"
            )
        try:
            self._runtimes[owner].submit(command, timeout=timeout)
        except Exception:
            if command.type is RuntimeCommandType.START and command.key is not None:
                affinity_key = (command.key, command.utterance_id)
                with self._lock:
                    if self._owners.get(affinity_key) == owner:
                        self._owners.pop(affinity_key, None)
                        ASR_STREAM_AFFINITIES.set(len(self._owners))
            raise

    def _on_worker_result(self, result: RuntimeResult) -> None:
        if result.key is not None and result.type in {
            RuntimeResultType.FINAL,
            RuntimeResultType.CANCELLED,
        }:
            affinity_key = (result.key, result.utterance_id)
            with self._lock:
                self._owners.pop(affinity_key, None)
                ASR_STREAM_AFFINITIES.set(len(self._owners))
        self._result_callback(result)


class TranslationRuntimePool:
    """Stateless utterance-level translation pool.

    Jobs may be processed by any worker. Each worker owns a separate
    CTranslate2 Translator/tokenizer instance, so no model instance is called
    concurrently by multiple Python worker threads.
    """

    def __init__(
        self,
        *,
        workers: int,
        model_path: str,
        tokenizer_path: str,
        devices: list[str],
        compute_type: str,
        inter_threads: int,
        intra_threads: int,
        queue_size: int,
        beam_size: int,
        max_decoding_length: int,
        vi_prefix: str,
        en_prefix: str,
        result_callback: Callable[[TranslationResult], None],
        batch_size: int = 1,
        batch_wait_ms: int = 0,
        third_party_log_level: str = "ERROR",
        runtime_factory: Callable[..., TranslationRuntime] = TranslationRuntime,
    ) -> None:
        if workers < 1:
            raise ValueError("TRANSLATION_WORKERS must be >= 1")
        self.workers = workers
        self._rr = 0
        self._rr_lock = threading.Lock()
        TRANSLATION_POOL_WORKERS.set(workers)
        self._runtimes = [
            runtime_factory(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device=devices[index % len(devices)],
                compute_type=compute_type,
                inter_threads=inter_threads,
                intra_threads=intra_threads,
                queue_size=queue_size,
                beam_size=beam_size,
                max_decoding_length=max_decoding_length,
                vi_prefix=vi_prefix,
                en_prefix=en_prefix,
                result_callback=result_callback,
                worker_id=index,
                batch_size=batch_size,
                batch_wait_ms=batch_wait_ms,
                third_party_log_level=third_party_log_level,
            )
            for index in range(workers)
        ]

    @property
    def queue_depth(self) -> int:
        return sum(runtime.queue_depth for runtime in self._runtimes)

    def start(self, timeout: float = 180.0) -> None:
        started: list[TranslationRuntime] = []
        try:
            for runtime in self._runtimes:
                runtime.start(timeout=timeout)
                started.append(runtime)
        except Exception:
            for runtime in reversed(started):
                try:
                    runtime.shutdown()
                except Exception:
                    logger.exception("Failed to stop translation worker after startup failure")
            raise
        logger.info("Translation runtime pool ready: workers=%s", self.workers)

    def shutdown(self, timeout: float = 10.0) -> None:
        per_worker_timeout = max(2.0, timeout / max(1, self.workers))
        for runtime in self._runtimes:
            try:
                runtime.shutdown(timeout=per_worker_timeout)
            except Exception:
                logger.exception("Failed to stop translation worker")
        logger.info("Translation runtime pool stopped")

    def submit_nowait(self, job: TranslationJob) -> None:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (
                self._runtimes[index].queue_depth,
                (index - rr) % self.workers,
            ),
        )
        last_error: Exception | None = None
        for index in order:
            try:
                self._runtimes[index].submit_nowait(job)
                with self._rr_lock:
                    self._rr = (index + 1) % self.workers
                return
            except RuntimeError as exc:
                last_error = exc
        raise QueueFullError("all translation worker queues are full or stopped") from last_error

    def try_submit_nowait(self, job: TranslationJob) -> SubmitStatus:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (
                self._runtimes[index].queue_depth,
                (index - rr) % self.workers,
            ),
        )
        worst = SubmitStatus.UNAVAILABLE
        for index in order:
            status = self._runtimes[index].try_submit_nowait(job)
            if status is SubmitStatus.ACCEPTED:
                with self._rr_lock:
                    self._rr = (index + 1) % self.workers
                return SubmitStatus.ACCEPTED
            if status is SubmitStatus.OVERLOADED:
                worst = SubmitStatus.OVERLOADED
        return worst


class TtsRuntimePool:
    """Piper pool where every worker owns independent EN/VI voice instances."""

    def __init__(
        self,
        *,
        workers: int,
        en_model_path: str,
        en_config_path: str | None,
        vi_model_path: str,
        vi_config_path: str | None,
        target_sample_rate: int,
        queue_size: int,
        max_text_chars: int,
        length_scale: float,
        noise_scale: float,
        noise_w_scale: float,
        volume: float,
        intra_threads: int = 2,
        devices: list[str],
        result_callback: Callable[[TtsResult], None],
        runtime_factory: Callable[..., TtsRuntime] = TtsRuntime,
    ) -> None:
        if workers < 1:
            raise ValueError("TTS_WORKERS must be >= 1")
        self.workers = workers
        self._rr = 0
        self._rr_lock = threading.Lock()
        TTS_POOL_WORKERS.set(workers)
        self._runtimes = [
            runtime_factory(
                en_model_path=en_model_path,
                en_config_path=en_config_path,
                vi_model_path=vi_model_path,
                vi_config_path=vi_config_path,
                target_sample_rate=target_sample_rate,
                queue_size=queue_size,
                max_text_chars=max_text_chars,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale,
                volume=volume,
                intra_threads=intra_threads,
                device=devices[index % len(devices)],
                result_callback=result_callback,
                worker_id=index,
            )
            for index in range(workers)
        ]

    @property
    def queue_depth(self) -> int:
        return sum(runtime.queue_depth for runtime in self._runtimes)

    def start(self, timeout: float = 180.0) -> None:
        started: list[TtsRuntime] = []
        try:
            for runtime in self._runtimes:
                runtime.start(timeout=timeout)
                started.append(runtime)
        except Exception:
            for runtime in reversed(started):
                try:
                    runtime.shutdown()
                except Exception:
                    logger.exception("Failed to stop TTS worker after startup failure")
            raise
        logger.info("TTS runtime pool ready: workers=%s", self.workers)

    def shutdown(self, timeout: float = 10.0) -> None:
        per_worker_timeout = max(2.0, timeout / max(1, self.workers))
        for runtime in self._runtimes:
            try:
                runtime.shutdown(timeout=per_worker_timeout)
            except Exception:
                logger.exception("Failed to stop TTS worker")
        logger.info("TTS runtime pool stopped")

    def submit_nowait(self, job: TtsJob) -> None:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (
                self._runtimes[index].queue_depth,
                (index - rr) % self.workers,
            ),
        )
        last_error: Exception | None = None
        for index in order:
            try:
                self._runtimes[index].submit_nowait(job)
                with self._rr_lock:
                    self._rr = (index + 1) % self.workers
                return
            except RuntimeError as exc:
                last_error = exc
        raise QueueFullError("all TTS worker queues are full or stopped") from last_error

    def try_submit_nowait(self, job: TtsJob) -> SubmitStatus:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (
                self._runtimes[index].queue_depth,
                (index - rr) % self.workers,
            ),
        )
        worst = SubmitStatus.UNAVAILABLE
        for index in order:
            status = self._runtimes[index].try_submit_nowait(job)
            if status is SubmitStatus.ACCEPTED:
                with self._rr_lock:
                    self._rr = (index + 1) % self.workers
                return SubmitStatus.ACCEPTED
            if status is SubmitStatus.OVERLOADED:
                worst = SubmitStatus.OVERLOADED
        return worst


class MagpieTtsRuntimePool:
    """Magpie multilingual TTS pool — one NeMo model replica per worker.

    Interface is identical to TtsRuntimePool so the coordinator uses both
    interchangeably. Each worker loads the same checkpoint independently on
    its assigned device (one entry per worker from TTS_DEVICES round-robin).
    """

    def __init__(
        self,
        *,
        workers: int,
        model_path: str,
        speaker_index: int,
        cfg_scale: float,
        temperature: float,
        top_k: int,
        volume: float,
        target_sample_rate: int,
        queue_size: int,
        max_text_chars: int,
        devices: list[str],
        result_callback: Callable[[TtsResult], None],
        suppress_model_init_output: bool = True,
        apply_tn: bool = False,
        codec_model_path: str | None = None,
        runtime_factory: Callable[..., MagpieTtsRuntime] = MagpieTtsRuntime,
    ) -> None:
        if workers < 1:
            raise ValueError("TTS_WORKERS must be >= 1")
        self.workers = workers
        self._rr = 0
        self._rr_lock = threading.Lock()
        TTS_POOL_WORKERS.set(workers)
        self._runtimes = [
            runtime_factory(
                model_path=model_path,
                speaker_index=speaker_index,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k,
                volume=volume,
                target_sample_rate=target_sample_rate,
                queue_size=queue_size,
                max_text_chars=max_text_chars,
                device=devices[index % len(devices)],
                result_callback=result_callback,
                worker_id=index,
                suppress_model_init_output=suppress_model_init_output,
                apply_tn=apply_tn,
                codec_model_path=codec_model_path,
            )
            for index in range(workers)
        ]

    @property
    def queue_depth(self) -> int:
        return sum(runtime.queue_depth for runtime in self._runtimes)

    def start(self, timeout: float = 300.0) -> None:
        started: list[MagpieTtsRuntime] = []
        try:
            for runtime in self._runtimes:
                runtime.start(timeout=timeout)
                started.append(runtime)
        except Exception:
            for runtime in reversed(started):
                try:
                    runtime.shutdown()
                except Exception:
                    logger.exception("Failed to stop Magpie TTS worker after startup failure")
            raise
        logger.info("Magpie TTS runtime pool ready: workers=%s", self.workers)

    def shutdown(self, timeout: float = 10.0) -> None:
        per_worker_timeout = max(2.0, timeout / max(1, self.workers))
        for runtime in self._runtimes:
            try:
                runtime.shutdown(timeout=per_worker_timeout)
            except Exception:
                logger.exception("Failed to stop Magpie TTS worker")
        logger.info("Magpie TTS runtime pool stopped")

    def submit_nowait(self, job: TtsJob) -> None:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (
                self._runtimes[index].queue_depth,
                (index - rr) % self.workers,
            ),
        )
        last_error: Exception | None = None
        for index in order:
            try:
                self._runtimes[index].submit_nowait(job)
                with self._rr_lock:
                    self._rr = (index + 1) % self.workers
                return
            except RuntimeError as exc:
                last_error = exc
        raise QueueFullError("all Magpie TTS worker queues are full or stopped") from last_error

    def try_submit_nowait(self, job: TtsJob) -> SubmitStatus:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (
                self._runtimes[index].queue_depth,
                (index - rr) % self.workers,
            ),
        )
        worst = SubmitStatus.UNAVAILABLE
        for index in order:
            status = self._runtimes[index].try_submit_nowait(job)
            if status is SubmitStatus.ACCEPTED:
                with self._rr_lock:
                    self._rr = (index + 1) % self.workers
                return SubmitStatus.ACCEPTED
            if status is SubmitStatus.OVERLOADED:
                worst = SubmitStatus.OVERLOADED
        return worst
