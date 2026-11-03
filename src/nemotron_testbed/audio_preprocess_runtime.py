from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from .config import AudioPreprocessSettings
from .domain import AudioPreprocessJob, AudioPreprocessResult, SubmitStatus
from .metrics import (
    AUDIO_PREPROC_BACKEND,
    AUDIO_PREPROC_POOL_WORKERS,
    AUDIO_PREPROC_QUEUE,
    AUDIO_PREPROC_QUEUE_WAIT_SECONDS,
    AUDIO_PREPROC_WINDOW_SECONDS,
)
from .resemble_models import WindowModel, load_window_model

logger = logging.getLogger(__name__)

ModelLoader = Callable[[str], WindowModel]


def make_model_loader(settings: AudioPreprocessSettings) -> ModelLoader:
    def load(device: str) -> WindowModel:
        if settings.mode == "audx":
            from pathlib import Path

            from .audx_model import AudxModel

            return AudxModel(
                Path(settings.audx_checkpoint),
                device,
                agc_target_dbfs=settings.audx_agc_target_dbfs,
                agc_max_gain_db=settings.audx_agc_max_gain_db,
            )
        return load_window_model(
            settings.mode,
            settings.assets_dir,
            device,
            enhancer_nfe=settings.enhancer_nfe,
            enhancer_lambd=settings.enhancer_lambd,
            enhancer_tau=settings.enhancer_tau,
            enhancer_align=settings.enhancer_align_regions,
            enhancer_align_max_ms=settings.enhancer_align_max_ms,
        )

    return load


class AudioPreprocessRuntime:
    """One model instance owned by one worker thread. Windows are stateless and stream jobs carry their stream's
    model state (one job per stream in flight), so any worker can serve any stream."""

    def __init__(
        self,
        *,
        model_loader: ModelLoader,
        device: str,
        queue_size: int,
        warmup_samples: int,
        result_callback: Callable[[AudioPreprocessResult], None],
        worker_id: int = 0,
        backend: str = AUDIO_PREPROC_BACKEND,
    ) -> None:
        self.device = device
        self.worker_id = worker_id
        self.backend = backend
        self._model_loader = model_loader
        self._warmup_samples = warmup_samples
        self._result_callback = result_callback
        self._jobs: queue.Queue[AudioPreprocessJob | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run, name=f"audio-preprocess-runtime-{worker_id}", daemon=True
        )
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._shutdown_requested = threading.Event()
        self._startup_error: BaseException | None = None
        self._model: WindowModel | None = None

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def start(self, timeout: float = 300.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out while loading the audio preprocessing model")
        if self._startup_error is not None:
            raise RuntimeError("Audio preprocessing runtime failed to start") from self._startup_error

    def try_submit_nowait(self, job: AudioPreprocessJob) -> SubmitStatus:
        if self._stopped.is_set():
            return SubmitStatus.UNAVAILABLE
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            return SubmitStatus.OVERLOADED
        AUDIO_PREPROC_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())
        return SubmitStatus.ACCEPTED

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
                logger.warning("Could not enqueue audio preprocessing shutdown sentinel")
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(
                "AudioPreprocessRuntime worker=%s did not stop in %.1fs", self.worker_id, timeout
            )
        self._stopped.set()

    def _emit(self, result: AudioPreprocessResult) -> None:
        try:
            self._result_callback(result)
        except Exception:
            logger.exception("Audio preprocessing result callback failed")

    def _load(self) -> None:
        logger.info(
            "Loading audio preprocessing model: worker=%s device=%s", self.worker_id, self.device
        )
        self._model = self._model_loader(self.device)
        if self._warmup_samples > 0:
            rng = np.random.default_rng(0)
            probe = (rng.standard_normal(self._warmup_samples) * 0.01).astype(np.float32)
            started = time.perf_counter()
            self._model(probe)
            logger.info(
                "Audio preprocessing warm-up: worker=%s window=%d samples took %.3fs",
                self.worker_id,
                self._warmup_samples,
                time.perf_counter() - started,
            )

    def _run(self) -> None:
        try:
            self._load()
        except BaseException as exc:
            self._startup_error = exc
            logger.exception("Audio preprocessing startup failed")
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
            AUDIO_PREPROC_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())
            try:
                if job is None:
                    break
                self._process(job)
            finally:
                self._jobs.task_done()

        self._stopped.set()
        logger.info("Audio preprocessing runtime stopped: worker=%s", self.worker_id)

    def _process(self, job: AudioPreprocessJob) -> None:
        started = time.perf_counter()
        queue_s = started - job.enqueue_time
        AUDIO_PREPROC_QUEUE_WAIT_SECONDS.labels(backend=self.backend).observe(queue_s)
        try:
            assert self._model is not None
            if job.state is not None:
                # stream job: any length up to what the stream has sent, the state advances in place
                audio = np.asarray(self._model.step(job.state, job.window, job.final))
                bad = audio.ndim != 1
            else:
                audio = self._model(job.window)
                bad = audio.shape != job.window.shape
            if bad or not np.isfinite(audio).all():
                raise ValueError(
                    f"bad model output: shape {audio.shape} for input {job.window.shape}, "
                    f"finite={bool(np.isfinite(audio).all())}"
                )
            audio = audio.astype(np.float32, copy=False)
            error = ""
        except Exception as exc:
            logger.exception(
                "Audio preprocessing window failed",
                extra={
                    "rpc_id": job.key.rpc_id,
                    "session_id": job.key.session_id,
                    "stream_id": job.key.stream_id,
                    "worker_id": self.worker_id,
                },
            )
            audio = None
            error = f"{type(exc).__name__}: {exc}"
        compute_s = time.perf_counter() - started
        AUDIO_PREPROC_WINDOW_SECONDS.labels(backend=self.backend).observe(compute_s)
        self._emit(
            AudioPreprocessResult(
                key=job.key,
                region_id=job.region_id,
                audio=audio,
                error=error,
                compute_s=compute_s,
                queue_s=queue_s,
            )
        )


class AudioPreprocessRuntimePool:
    def __init__(
        self,
        *,
        workers: int,
        devices: list[str],
        queue_size: int,
        warmup_samples: int,
        model_loader: ModelLoader,
        result_callback: Callable[[AudioPreprocessResult], None],
        runtime_factory: Callable[..., AudioPreprocessRuntime] = AudioPreprocessRuntime,
        backend: str = AUDIO_PREPROC_BACKEND,
    ) -> None:
        if workers < 1:
            raise ValueError("audio_preprocessing.workers must be >= 1")
        if not devices:
            raise ValueError("audio_preprocessing.devices must not be empty")
        self.workers = workers
        self._rr = 0
        self._rr_lock = threading.Lock()
        AUDIO_PREPROC_POOL_WORKERS.labels(backend=backend).set(workers)
        self._runtimes = [
            runtime_factory(
                model_loader=model_loader,
                device=devices[index % len(devices)],
                queue_size=queue_size,
                warmup_samples=warmup_samples,
                result_callback=result_callback,
                worker_id=index,
                backend=backend,
            )
            for index in range(workers)
        ]

    @property
    def queue_depth(self) -> int:
        return sum(runtime.queue_depth for runtime in self._runtimes)

    def start(self, timeout: float = 300.0) -> None:
        started: list[AudioPreprocessRuntime] = []
        try:
            for runtime in self._runtimes:
                runtime.start(timeout=timeout)
                started.append(runtime)
        except Exception:
            for runtime in reversed(started):
                try:
                    runtime.shutdown()
                except Exception:
                    logger.exception("Failed to stop audio preprocessing worker after startup failure")
            raise
        logger.info("Audio preprocessing runtime pool ready: workers=%s", self.workers)

    def shutdown(self, timeout: float = 10.0) -> None:
        per_worker_timeout = max(2.0, timeout / max(1, self.workers))
        for runtime in self._runtimes:
            try:
                runtime.shutdown(timeout=per_worker_timeout)
            except Exception:
                logger.exception("Failed to stop audio preprocessing worker")
        logger.info("Audio preprocessing runtime pool stopped")

    def try_submit_nowait(self, job: AudioPreprocessJob) -> SubmitStatus:
        with self._rr_lock:
            rr = self._rr
        order = sorted(
            range(self.workers),
            key=lambda index: (self._runtimes[index].queue_depth, (index - rr) % self.workers),
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
