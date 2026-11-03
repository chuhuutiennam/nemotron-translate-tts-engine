from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import resample
from .domain import QueueFullError, SubmitStatus, TtsJob, TtsResult
from .g711 import encode_g711
from .languages import short_language
from .logging_config import suppress_external_stdio
from .metrics import (
    TTS_AUDIO_SECONDS,
    TTS_LATENCY,
    TTS_NO_AUDIO_TOTAL,
    TTS_QUEUE,
    TTS_QUEUE_WAIT_MS,
    TTS_SYNTHESIS_RTF,
    TTS_WORKER_ERRORS,
)

logger = logging.getLogger(__name__)

# Language codes accepted by MagpieTTSModel.do_tts()
_MAGPIE_SUPPORTED_LANGS = {"vi", "en"}


class MagpieTtsRuntime:
    """Single-owner NeMo MagpieTTS runtime (multilingual, one model per worker).

    Each runtime owns an independent NeMo MagpieTTSModel instance loaded on
    a specific device. The single multilingual checkpoint handles both
    Vietnamese and English synthesis.

    Interface is identical to TtsRuntime so TtsRuntimePool and MagpieTtsRuntimePool
    use the same coordinator-facing contract.
    """

    def __init__(
        self,
        model_path: str,
        speaker_index: int,
        cfg_scale: float,
        temperature: float,
        top_k: int,
        volume: float,
        target_sample_rate: int,
        queue_size: int,
        max_text_chars: int,
        result_callback: Callable[[TtsResult], None],
        worker_id: int = 0,
        suppress_model_init_output: bool = True,
        apply_tn: bool = False,
        device: str = "cpu",
        codec_model_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.speaker_index = speaker_index
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.top_k = top_k
        self.volume = volume
        self.target_sample_rate = target_sample_rate
        self.max_text_chars = max_text_chars
        self._result_callback = result_callback
        self.worker_id = worker_id
        self.suppress_model_init_output = suppress_model_init_output
        self.apply_tn = apply_tn
        self.device = device
        self.codec_model_path = codec_model_path

        self._jobs: queue.Queue[TtsJob | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run,
            name=f"magpie-tts-runtime-{worker_id}",
            daemon=True,
        )
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: BaseException | None = None
        self._model = None
        self._sample_rate: int = 0

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def start(self, timeout: float = 300.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out while loading Magpie TTS model")
        if self._startup_error is not None:
            raise RuntimeError("Magpie TTS runtime failed to start") from self._startup_error

    def submit_nowait(self, job: TtsJob) -> None:
        if self._stopped.is_set():
            raise QueueFullError("Magpie TTS runtime is stopped")
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise QueueFullError("Magpie TTS runtime queue is full") from exc
        TTS_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())

    def try_submit_nowait(self, job: TtsJob) -> SubmitStatus:
        if self._stopped.is_set():
            return SubmitStatus.UNAVAILABLE
        try:
            self._jobs.put_nowait(job)
            TTS_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())
            return SubmitStatus.ACCEPTED
        except queue.Full:
            return SubmitStatus.OVERLOADED

    def shutdown(self, timeout: float = 10.0) -> None:
        if self._stopped.is_set():
            return
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                self._jobs.put(None, timeout=1.0)
            except queue.Full:
                logger.warning("Could not enqueue Magpie TTS shutdown sentinel")
        self._thread.join(timeout)
        self._stopped.set()

    def _emit(self, result: TtsResult) -> None:
        try:
            self._result_callback(result)
        except Exception:
            logger.exception("Magpie TTS result callback failed")

    def _patch_codec_for_airgap(self, codec_local_path: str) -> None:
        """Redirect AudioCodecModel.from_pretrained to a local .nemo file.

        MagpieTTSModel.__init__ calls AudioCodecModel.from_pretrained(codec_model_path)
        which tries to connect to HuggingFace. In air-gapped Kubernetes environments
        this fails with a DNS error. Patching the class before restore_from is called
        redirects to restore_from(local_path) instead.

        Set TTS_MAGPIE_CODEC_MODEL_PATH to the local codec .nemo file path to enable.
        HF repo: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps
        """
        try:
            import nemo.collections.tts.models.magpietts as _magpietts_mod

            _ACM = getattr(_magpietts_mod, "AudioCodecModel", None)
            if _ACM is None:
                logger.warning(
                    "AudioCodecModel not found in magpietts module namespace; "
                    "air-gap patch skipped: worker=%s",
                    self.worker_id,
                )
                return

            _orig = _ACM.from_pretrained
            codec_path = codec_local_path
            worker_id = self.worker_id

            @classmethod  # type: ignore[misc]
            def _airgap_from_pretrained(cls, name_or_path, **kwargs):
                from pathlib import Path as _Path
                if _Path(codec_path).is_file():
                    logger.info(
                        "AudioCodecModel: air-gap → restore_from(%s) worker=%s",
                        codec_path,
                        worker_id,
                    )
                    return cls.restore_from(restore_path=codec_path)
                return _orig(name_or_path, **kwargs)

            _ACM.from_pretrained = _airgap_from_pretrained
            logger.info(
                "Patched AudioCodecModel.from_pretrained for air-gap: worker=%s codec=%s",
                self.worker_id,
                codec_local_path,
            )
        except Exception:
            logger.warning(
                "Could not patch AudioCodecModel.from_pretrained: worker=%s",
                self.worker_id,
                exc_info=True,
            )

    def _load(self) -> None:
        model_path = Path(self.model_path)
        use_local = model_path.exists()
        if not use_local and model_path.suffix:
            # Has a file extension but file not found — almost certainly a typo in a
            # local path.  Fail before importing NeMo so the error is immediate and
            # clear; fall-through would produce a confusing HuggingFace error.
            raise FileNotFoundError(
                f"Magpie TTS model file not found: {model_path!r}. "
                f"Supply a valid local .nemo path or a HuggingFace repo ID (no extension)."
            )

        from nemo.collections.tts.models import MagpieTTSModel

        if self.codec_model_path:
            self._patch_codec_for_airgap(self.codec_model_path)

        if use_local:
            logger.info(
                "Loading Magpie TTS from local path: worker=%s device=%s path=%s",
                self.worker_id,
                self.device,
                model_path,
            )
            self._model = MagpieTTSModel.restore_from(str(model_path), map_location=self.device)
        else:
            logger.info(
                "Loading Magpie TTS from HuggingFace: worker=%s device=%s repo=%s",
                self.worker_id,
                self.device,
                self.model_path,
            )
            self._model = MagpieTTSModel.from_pretrained(self.model_path, map_location=self.device)

        self._model.eval()
        # map_location already placed the model on self.device; a second .to()
        # would briefly double GPU memory allocation for no benefit.

        # Dump speaker info so TTS_SPEAKER_INDEX can be verified against the actual checkpoint.
        try:
            cfg_keys = list(self._model.cfg.keys())
            num_speakers = self._model.cfg.get("num_speakers", None)
            speakers = self._model.cfg.get("speakers", None)
            logger.info(
                "Magpie model cfg keys: worker=%s keys=%s",
                self.worker_id,
                cfg_keys,
            )
            logger.info(
                "Magpie speaker info: worker=%s num_speakers=%s speakers=%s",
                self.worker_id,
                num_speakers,
                list(speakers) if speakers is not None else None,
            )
        except Exception:
            logger.warning("Could not dump Magpie speaker cfg: worker=%s", self.worker_id)

        # Apply inference hyper-parameters via model cfg where the key exists.
        try:
            from omegaconf import open_dict
            with open_dict(self._model.cfg):
                infer = self._model.cfg.get("inference", None)
                if infer is not None:
                    infer.cfg_scale = self.cfg_scale
                    infer.temperature = self.temperature
                    infer.top_k = self.top_k
                    logger.debug(
                        "Magpie inference cfg applied: worker=%s cfg_scale=%s "
                        "temperature=%s top_k=%s",
                        self.worker_id,
                        self.cfg_scale,
                        self.temperature,
                        self.top_k,
                    )
                else:
                    logger.warning(
                        "Magpie model cfg has no 'inference' section: worker=%s; "
                        "TTS_CFG_SCALE/TTS_TEMPERATURE/TTS_TOP_K will be ignored",
                        self.worker_id,
                    )
        except Exception:
            logger.warning(
                "Could not set Magpie inference hyper-params via model cfg "
                "(worker=%s); TTS_CFG_SCALE/TTS_TEMPERATURE/TTS_TOP_K will be ignored",
                self.worker_id,
            )

        sr = getattr(self._model, "sample_rate", None)
        if not sr or int(sr) <= 0:
            raise RuntimeError(
                f"Magpie TTS model does not expose a valid sample_rate "
                f"(got {sr!r}): worker={self.worker_id}"
            )
        self._sample_rate = int(sr)

        # Confirm actual PyTorch device — map_location may differ from self.device
        # if the checkpoint was loaded with a device override.
        try:
            import torch
            actual_device = next(self._model.parameters()).device
            on_gpu = actual_device.type == "cuda"
            logger.info(
                "Magpie TTS model device confirmed: worker=%s requested=%s actual=%s gpu=%s",
                self.worker_id,
                self.device,
                actual_device,
                on_gpu,
            )
            if self.device.startswith("cuda") and not on_gpu:
                logger.warning(
                    "Magpie TTS: requested cuda but model is on CPU — "
                    "check map_location and CUDA availability: worker=%s",
                    self.worker_id,
                )
        except Exception:
            logger.warning("Could not confirm Magpie TTS model device: worker=%s", self.worker_id)

        logger.info(
            "Magpie TTS runtime ready: worker=%s device=%s speaker=%s sample_rate=%s",
            self.worker_id,
            self.device,
            self.speaker_index,
            self._sample_rate,
        )

    def _run(self) -> None:
        try:
            with suppress_external_stdio(self.suppress_model_init_output, stderr=False):
                self._load()
        except BaseException as exc:
            self._startup_error = exc
            logger.exception("Magpie TTS startup failed")
            self._ready.set()
            self._stopped.set()
            return

        self._ready.set()

        while True:
            job = self._jobs.get()
            TTS_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())

            if job is None:
                self._jobs.task_done()
                break

            TTS_QUEUE_WAIT_MS.observe(
                (time.perf_counter() - job.enqueue_time) * 1000.0
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "POOL_TRACE TTS(magpie) worker=%s stream=%s utterance=%s language=%s",
                    self.worker_id,
                    job.key.stream_id,
                    job.utterance_id,
                    job.target_language,
                )

            started = time.perf_counter()

            try:
                payload, source_rate, duration_s = self._synthesize(job)
                elapsed = time.perf_counter() - started
                TTS_LATENCY.observe(elapsed)
                TTS_AUDIO_SECONDS.inc(duration_s)
                self._emit(
                    TtsResult(
                        key=job.key,
                        utterance_id=job.utterance_id,
                        text=job.text,
                        target_language=job.target_language,
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        audio_payload=payload,
                        audio_pt=job.audio_pt,
                        latency_ms=elapsed * 1000.0,
                        source_sample_rate=source_rate,
                        phrase_id=job.phrase_id,
                        is_last_phrase=job.is_last_phrase,
                        translated_text=job.translated_text,
                        segment_idx=job.segment_idx,
                        queue_ms=(started - job.enqueue_time) * 1000.0,
                    )
                )
            except Exception as exc:
                TTS_WORKER_ERRORS.labels(
                    worker=str(self.worker_id),
                    device=self.device,
                ).inc()
                logger.exception(
                    "Magpie TTS job failed",
                    extra={
                        "rpc_id": job.key.rpc_id,
                        "session_id": job.key.session_id,
                        "stream_id": job.key.stream_id,
                        "utterance_id": job.utterance_id,
                        "worker_id": self.worker_id,
                    },
                )
                self._emit(
                    TtsResult(
                        key=job.key,
                        utterance_id=job.utterance_id,
                        text=job.text,
                        target_language=job.target_language,
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        audio_pt=job.audio_pt,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        error=f"{type(exc).__name__}: {exc}",
                        phrase_id=job.phrase_id,
                        is_last_phrase=job.is_last_phrase,
                        translated_text=job.translated_text,
                        segment_idx=job.segment_idx,
                    )
                )
            finally:
                self._jobs.task_done()

        self._stopped.set()
        logger.info("Magpie TTS runtime stopped: worker=%s", self.worker_id)

    def _synthesize(self, job: TtsJob) -> tuple[bytes, int, float]:
        import torch

        text = job.text.strip()
        if not text:
            return b"", 0, 0.0

        if len(text) > self.max_text_chars:
            raise ValueError(
                f"TTS text length {len(text)} exceeds TTS_MAX_TEXT_CHARS={self.max_text_chars}"
            )

        lang = short_language(job.target_language)
        if lang not in _MAGPIE_SUPPORTED_LANGS:
            raise ValueError(
                f"Magpie TTS: unsupported target language {job.target_language!r} "
                f"(resolved={lang!r}, supported={sorted(_MAGPIE_SUPPORTED_LANGS)})"
            )

        _t0 = time.perf_counter()

        with torch.inference_mode():
            audio_t, audio_len_t = self._model.do_tts(
                text,
                language=lang,
                apply_TN=self.apply_tn,
                speaker_index=self.speaker_index,
            )

        if audio_t.ndim != 2 or audio_len_t.ndim != 1:
            raise RuntimeError(
                f"Magpie do_tts returned unexpected tensor shapes: "
                f"audio={tuple(audio_t.shape)} len={tuple(audio_len_t.shape)}"
            )

        source_rate: int = self._sample_rate
        reported_len = int(audio_len_t[0].item())
        available_len = audio_t.shape[1]
        if reported_len > available_len:
            logger.warning(
                "Magpie TTS audio_len (%d) > tensor length (%d): worker=%s text=%r; "
                "audio will be truncated",
                reported_len,
                available_len,
                self.worker_id,
                text[:80],
            )
            reported_len = available_len
        # .float() ensures float32 before .numpy() — soxr does not accept float16
        # which NeMo may produce under mixed-precision inference.
        audio_np: np.ndarray = audio_t[0, :reported_len].cpu().float().numpy()

        if audio_np.size == 0:
            TTS_NO_AUDIO_TOTAL.labels(worker=str(self.worker_id)).inc()
            logger.warning(
                "Magpie TTS produced no audio: worker=%s text=%r",
                self.worker_id,
                text[:120],
            )
            return b"", 0, 0.0

        # Always clip: neural codec reconstruction can produce values outside [-1, 1].
        audio_np = np.clip(audio_np * self.volume, -1.0, 1.0)

        audio_target = resample(audio_np, source_rate, self.target_sample_rate)
        payload = encode_g711(audio_target, job.audio_pt)
        duration_s = float(audio_target.size) / float(self.target_sample_rate)

        _total_ms = (time.perf_counter() - _t0) * 1000.0
        _rtf = _total_ms / (duration_s * 1000.0) if duration_s > 0 else 0.0
        logger.info(
            "Magpie TTS synthesize: worker=%s device=%s lang=%s chars=%d "
            "audio_s=%.2f payload_bytes=%d total_ms=%.1f RTF=%.3f",
            self.worker_id,
            self.device,
            lang,
            len(text),
            duration_s,
            len(payload),
            _total_ms,
            _rtf,
        )
        TTS_SYNTHESIS_RTF.labels(
            worker=str(self.worker_id),
            device=self.device,
        ).observe(_rtf)

        return payload, source_rate, duration_s
