from __future__ import annotations

import copy
import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

import numpy as np

from .metrics import (
    VAD_MAX_SCORE,
    VAD_SCORE,
    VAD_SPEECH_END_DELAY_MS,
    VAD_SPEECH_START_DELAY_MS,
    VAD_STARTS_TOTAL,
)

logger = logging.getLogger(__name__)

# Cache the loaded Silero JIT model template so each stream gets a deep copy
# (independent LSTM states) without re-reading from disk on every new call.
_silero_model_cache: dict[str, object] = {}
_silero_model_cache_lock = threading.Lock()


class VadEvent(Enum):
    NONE = auto()
    START = auto()
    SPEECH = auto()
    END = auto()


class SpeechProbability(Protocol):
    def score(self, audio: np.ndarray) -> float:
        ...

    def reset(self) -> None:
        ...


class EnergySpeechProbability:
    needs_executor: bool = False

    def __init__(self, db_threshold: float = -42.0):
        self.db_threshold = db_threshold

    def score(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))
        db = 20.0 * math.log10(max(rms, 1e-8))
        # A smooth score helps the same endpoint controller work with energy or Silero.
        return 1.0 / (1.0 + math.exp(-(db - self.db_threshold) / 3.0))

    def reset(self) -> None:
        return None


class SileroSpeechProbability:
    """Streaming Silero probability adapter for 16 kHz mono audio.

    needs_executor = True signals to callers that inference should be
    dispatched to a thread-pool executor to avoid blocking the asyncio
    event loop during PyTorch forward passes.

    The adapter accepts arbitrary transport chunk sizes and internally runs the
    VAD on 512-sample windows. The package path works fully offline once the
    silero-vad package is installed. SILERO_MODEL_PATH may point to a local JIT
    model when package loading is not desired.

    Thread safety: score() runs on an executor thread; reset() runs on the
    asyncio event loop thread. _lock serializes all accesses to _pending,
    _last_score, and the model's LSTM state (model() forward pass and
    reset_states()). Holding the lock during inference is acceptable because
    each stream owns its own instance and score() is called sequentially.

    Model sharing: the JIT model is loaded once per (path, device) pair and
    deep-copied per stream so each stream has isolated LSTM states without
    re-reading from disk.
    """

    needs_executor: bool = True

    def __init__(
        self,
        model_path: str | None = None,
        sample_rate: int = 16_000,
        device: str = "cpu",
    ):
        import torch

        self._torch = torch
        self._device = device
        self.sample_rate = sample_rate
        self.window_samples = 512 if sample_rate == 16_000 else 256
        self._pending = np.empty(0, dtype=np.float32)
        self._last_score: float = 0.0
        self._lock = threading.Lock()

        cache_key = f"{model_path or 'package'}:{device}"
        with _silero_model_cache_lock:
            if cache_key not in _silero_model_cache:
                if model_path:
                    path = Path(model_path)
                    if not path.is_file():
                        raise FileNotFoundError(f"Silero JIT model not found: {path}")
                    template = torch.jit.load(str(path), map_location=device)
                else:
                    from silero_vad import load_silero_vad  # type: ignore
                    template = load_silero_vad()
                    template = template.to(device)
                template.eval()
                _silero_model_cache[cache_key] = template
            self.model = copy.deepcopy(_silero_model_cache[cache_key])

    def score(self, audio: np.ndarray) -> float:
        scores = self._window_scores(audio)
        with self._lock:
            if scores:
                self._last_score = max(scores)
            return self._last_score

    def frame_scores(self, audio: np.ndarray, duration_ms: float) -> list[tuple[float, float]]:
        """(score, ms) per 512-sample window; time is counted by windows, not by chunk."""
        scores = self._window_scores(audio)
        with self._lock:
            if not scores:
                return [(self._last_score, 0.0)]
            self._last_score = max(scores)
        window_ms = self.window_samples * 1000.0 / self.sample_rate
        return [(value, window_ms) for value in scores]

    @property
    def pending_ms(self) -> float:
        with self._lock:
            return self._pending.size * 1000.0 / self.sample_rate

    def _window_scores(self, audio: np.ndarray) -> list[float]:
        if audio.size:
            with self._lock:
                self._pending = np.concatenate(
                    [self._pending, np.asarray(audio, dtype=np.float32).reshape(-1)]
                )
        scores: list[float] = []
        while True:
            with self._lock:
                if self._pending.size < self.window_samples:
                    break
                window = self._pending[: self.window_samples]
                self._pending = self._pending[self.window_samples :]
                # Inference runs inside the lock so reset_states() cannot race
                # with an in-progress forward pass on the LSTM state.
                tensor = self._torch.from_numpy(window).to(self._device)
                with self._torch.inference_mode():
                    scores.append(float(self.model(tensor, self.sample_rate).item()))
        return scores

    def reset(self) -> None:
        with self._lock:
            self._pending = np.empty(0, dtype=np.float32)
            self._last_score = 0.0
            if hasattr(self.model, "reset_states"):
                self.model.reset_states()


@dataclass(slots=True)
class VadController:
    probability: SpeechProbability
    threshold: float
    min_speech_ms: int
    end_silence_ms: int
    enabled: bool = True
    # Hysteresis like Silero's VADIterator: scores in [neg_threshold, threshold) are a
    # dead band that is neither speech nor silence. None = threshold (no dead band).
    neg_threshold: float | None = None
    active: bool = False
    speech_candidate_ms: float = 0.0
    silence_ms: float = 0.0
    max_score: float = 0.0
    end_tail_ms: float = 0.0
    end_silence_observed_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.neg_threshold is None:
            self.neg_threshold = self.threshold

    @property
    def needs_executor(self) -> bool:
        return getattr(self.probability, 'needs_executor', False)

    def process(self, audio: np.ndarray, duration_ms: float) -> VadEvent:
        self.end_tail_ms = 0.0
        self.end_silence_observed_ms = 0.0
        if not self.enabled:
            if not self.active:
                self.active = True
                VAD_STARTS_TOTAL.inc()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "VAD step",
                        extra={
                            "vad_score": 1.0,
                            "vad_speech": True,
                            "vad_active": True,
                            "vad_candidate_ms": 0.0,
                            "vad_silence_ms": 0.0,
                            "vad_event": "START",
                            "duration_ms": round(duration_ms, 1),
                        },
                    )
                return VadEvent.START
            return VadEvent.SPEECH

        frame_scores = getattr(self.probability, "frame_scores", None)
        if frame_scores is not None:
            frames = frame_scores(audio, duration_ms)
        else:
            frames = [(self.probability.score(audio), duration_ms)]
        probability = max(score for score, _ in frames)
        speech = probability >= self.threshold
        VAD_SCORE.observe(probability)
        VAD_MAX_SCORE.observe(probability)
        if probability > self.max_score:
            self.max_score = probability

        log_candidate_ms = self.speech_candidate_ms
        log_silence_ms = self.silence_ms

        if not self.active:
            # START is counted per scored window, like END: min_speech_ms must be
            # consecutive speech windows, independent of the transport chunk size.
            # A window below neg_threshold hard-resets the count; a dead-band window
            # (neg_threshold <= score < threshold) neither adds nor resets. Pre-roll
            # compensates for the speech missed before START fires. Zero-length
            # frames carry no new information (stale score) and are skipped.
            event = VadEvent.NONE
            for score, frame_ms in frames:
                if frame_ms <= 0.0:
                    continue
                if score >= self.threshold:
                    self.speech_candidate_ms += frame_ms
                elif score < self.neg_threshold:
                    self.speech_candidate_ms = 0.0
                if self.speech_candidate_ms >= self.min_speech_ms:
                    VAD_SPEECH_START_DELAY_MS.observe(self.speech_candidate_ms)
                    self.active = True
                    self.silence_ms = 0.0
                    VAD_STARTS_TOTAL.inc()
                    event = VadEvent.START
                    break
            log_candidate_ms = self.speech_candidate_ms
        else:
            # END is counted per scored window so end_silence_ms means "silence of at
            # least this long", independent of how the transport chunks the audio.
            # Only windows below neg_threshold count as silence; dead-band windows
            # neither add to nor reset the silence run.
            event = VadEvent.SPEECH
            for index, (score, frame_ms) in enumerate(frames):
                if score >= self.threshold:
                    self.silence_ms = 0.0
                    continue
                if score >= self.neg_threshold:
                    continue
                self.silence_ms += frame_ms
                if self.silence_ms >= self.end_silence_ms:
                    VAD_SPEECH_END_DELAY_MS.observe(self.silence_ms)
                    log_silence_ms = self.silence_ms
                    self.end_silence_observed_ms = self.silence_ms
                    # Audio of this chunk after the END window belongs to the next utterance.
                    self.end_tail_ms = sum(ms for _, ms in frames[index + 1 :]) + float(
                        getattr(self.probability, "pending_ms", 0.0)
                    )
                    self.active = False
                    self.speech_candidate_ms = 0.0
                    self.silence_ms = 0.0
                    event = VadEvent.END
                    break
            if event is not VadEvent.END:
                log_silence_ms = self.silence_ms

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VAD step",
                extra={
                    "vad_score": round(probability, 4),
                    "vad_speech": speech,
                    "vad_active": self.active,
                    "vad_candidate_ms": round(log_candidate_ms, 1),
                    "vad_silence_ms": round(log_silence_ms, 1),
                    "vad_event": event.name,
                    "vad_end_tail_ms": round(self.end_tail_ms, 1),
                    "duration_ms": round(duration_ms, 1),
                },
            )

        return event

    def reset(self) -> None:
        self.active = False
        self.speech_candidate_ms = 0.0
        self.silence_ms = 0.0
        self.probability.reset()


def build_vad_controller(
    backend: str,
    enabled: bool,
    threshold: float,
    energy_db_threshold: float,
    min_speech_ms: int,
    end_silence_ms: int,
    silero_model_path: str | None,
    device: str = "cpu",
    neg_threshold: float | None = None,
) -> VadController:
    if backend == "energy":
        probability: SpeechProbability = EnergySpeechProbability(energy_db_threshold)
    elif backend == "silero":
        probability = SileroSpeechProbability(silero_model_path, device=device)
    else:
        raise ValueError(f"Unsupported VAD_BACKEND: {backend}")
    logger.debug("VAD initialized with backend=%s enabled=%s", backend, enabled)
    return VadController(
        probability=probability,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        end_silence_ms=end_silence_ms,
        enabled=enabled,
        neg_threshold=neg_threshold,
    )


def warmup_silero_cache(model_path: str | None, device: str) -> None:
    """Pre-populate the Silero JIT model cache without creating a stream instance.

    Call once at startup in a thread (not the event loop) so that the first
    stream only needs copy.deepcopy instead of loading the model from disk.
    """
    cache_key = f"{model_path or 'package'}:{device}"
    with _silero_model_cache_lock:
        if cache_key in _silero_model_cache:
            return
    SileroSpeechProbability(model_path=model_path, device=device)
