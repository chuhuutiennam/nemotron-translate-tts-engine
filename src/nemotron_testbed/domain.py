from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any

import numpy as np


class QueueFullError(RuntimeError):
    """Raised when a worker queue is full or the runtime has been stopped.

    Maps to gRPC RESOURCE_EXHAUSTED — the condition is transient and the
    caller should retry when capacity becomes available.
    """


class SubmitStatus(Enum):
    ACCEPTED = auto()
    OVERLOADED = auto()
    UNAVAILABLE = auto()


class TopicLabel(IntEnum):
    """Internal enum kept numerically aligned with speech.TopicLabel (proto/speech.proto)."""
    UNKNOWN = 0
    WEATHER = 1
    FOOD = 2


@dataclass(frozen=True, slots=True)
class StreamKey:
    rpc_id: str
    session_id: str
    stream_id: str


class StreamPhase(Enum):
    OPEN = auto()    # accepting audio; new utterances may start
    CLOSING = auto() # input ended; finishing utterances draining
    CLOSED = auto()  # terminal


class RuntimeCommandType(Enum):
    START = auto()
    STEP = auto()
    FINALIZE = auto()
    CANCEL = auto()
    SHUTDOWN = auto()


class RuntimeResultType(Enum):
    STARTED = auto()
    PARTIAL = auto()
    FINAL = auto()
    CANCELLED = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass(slots=True)
class RuntimeCommand:
    type: RuntimeCommandType
    key: StreamKey | None = None
    utterance_id: int = 0
    language: str = "vi-VN"
    audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass(slots=True)
class RuntimeResult:
    type: RuntimeResultType
    key: StreamKey | None = None
    utterance_id: int = 0
    text: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    detected_language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True, slots=True)
class KeywordJob:
    """ASR-final text of one task=keyword utterance, sent to KeywordRuntime for topic detection."""
    key: StreamKey
    utterance_id: int
    source_text: str
    source_language: str
    ts_start: int
    ts_end: int
    asr_confidence: float = 0.0
    wall_start: float = 0.0
    enqueue_time: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True, slots=True)
class KeywordResult:
    key: StreamKey
    utterance_id: int
    source_text: str
    source_language: str
    ts_start: int
    ts_end: int
    asr_confidence: float = 0.0
    topic_label: TopicLabel = TopicLabel.UNKNOWN
    topic_info: str = ""
    topic_score: float = 0.0
    wall_start: float = 0.0
    latency_ms: float = 0.0
    queue_ms: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True)
class TranslationJob:
    key: StreamKey
    utterance_id: int
    source_text: str
    source_language: str
    target_language: str
    ts_start: int
    ts_end: int
    enqueue_time: float = field(default_factory=time.perf_counter)
    segment_idx: int = 0


@dataclass(frozen=True, slots=True)
class TranslationResult:
    key: StreamKey
    utterance_id: int
    source_text: str
    translated_text: str = ""
    source_language: str = ""
    target_language: str = ""
    ts_start: int = 0
    ts_end: int = 0
    latency_ms: float = 0.0
    error: str = ""
    segment_idx: int = 0
    queue_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AudioPreprocessJob:
    key: StreamKey
    region_id: int  # window jobs: region index; stream jobs: job id within the stream
    window: np.ndarray  # window jobs: the whole window; stream jobs: the next input samples
    # Stream jobs only (stateful model): the stream's model state, advanced in place by model.step();
    # final=True ends the stream and returns the samples the model still holds.
    state: Any = None
    final: bool = False
    enqueue_time: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True, slots=True)
class AudioPreprocessResult:
    key: StreamKey
    region_id: int
    audio: np.ndarray | None = None
    error: str = ""
    compute_s: float = 0.0
    queue_s: float = 0.0


@dataclass(frozen=True, slots=True)
class TtsJob:
    key: StreamKey
    utterance_id: int
    text: str
    target_language: str
    ts_start: int
    ts_end: int
    audio_pt: int
    phrase_id: int = 0
    is_last_phrase: bool = True
    translated_text: str = ""
    enqueue_time: float = field(default_factory=time.perf_counter)
    segment_idx: int = 0


@dataclass(frozen=True, slots=True)
class TtsResult:
    key: StreamKey
    utterance_id: int
    text: str
    target_language: str
    ts_start: int
    ts_end: int
    audio_payload: bytes = b""
    audio_pt: int = 0
    latency_ms: float = 0.0
    source_sample_rate: int = 0
    error: str = ""
    phrase_id: int = 0
    is_last_phrase: bool = True
    translated_text: str = ""
    segment_idx: int = 0
    queue_ms: float = 0.0
