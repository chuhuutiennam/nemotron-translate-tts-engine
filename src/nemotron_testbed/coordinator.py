from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
import unicodedata
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from .audio import AudioFormatError, decode_and_resample, resample
from .audio_preprocess import StreamAudioPreprocessor
from .audio_preprocess_runtime import AudioPreprocessRuntimePool, make_model_loader
from .audio_preprocess_stateful import StatefulStreamPreprocessor
from .audx_model import AudxStreamState
from .buffers import SampleBuffer
from .config import Settings
from .domain import (
    AudioPreprocessResult,
    KeywordJob,
    KeywordResult,
    TopicLabel,
    RuntimeCommand,
    RuntimeCommandType,
    RuntimeResult,
    RuntimeResultType,
    StreamKey,
    StreamPhase,
    SubmitStatus,
    TranslationJob,
    TranslationResult,
    TtsJob,
    TtsResult,
)
from .g711 import encode_g711
from .languages import (
    canonical_translatable_language,
    is_translatable_language,
    normalize_language,
    short_language,
)
from .metrics import (
    ACCEPT_CHUNK_MS,
    ACTIVE_RPCS,
    ACTIVE_STREAMS,
    ACTIVE_UTTERANCES,
    ASR_BACKLOG_HARD,
    ASR_BACKLOG_SOFT,
    ASR_CONFIDENCE,
    ASR_CONFIDENCE_SOURCE,
    ASR_FINALIZATION_LATENCY_MS,
    ASR_FIRST_PARTIAL_LATENCY_MS,
    ASR_UTTERANCE_RTF,
    AUDIO_PREPROC_POOL_WORKERS,
    AUDIO_PREPROC_SHED_STREAMS,
    AUDIO_SECONDS,
    DEFERRED_BACKLOG_MS,
    DROPPED_PARTIALS,
    DURATION_MISMATCHES,
    EMPTY_RESULTS_SKIPPED,
    FINAL_RESULTS,
    FIRST_AUDIO_FROM_SPEECH_MS,
    FIRST_AUDIO_MS,
    INGRESS_BACKLOG_MS,
    INGRESS_CHUNKS,
    INGRESS_PEAK,
    KEYWORD_ERRORS,
    KEYWORD_JOBS,
    KEYWORD_LATENCY_MS,
    KEYWORD_QUEUE_WAIT_MS,
    KEYWORD_TOPICS,
    PARTIAL_RESULTS,
    PASSTHROUGH_TOTAL,
    PIPELINE_DEGRADED,
    PIPELINE_PENDING,
    PIPELINE_RESULTS,
    POST_SPEECH_LATENCY_MS,
    RTP_GAPS,
    RTP_GAP_SILENCE_INSERTED,
    RTP_JITTER_CORRECTED,
    RTP_OUT_OF_ORDER,
    SEGMENT_COMMITS,
    SEGMENT_EARLY_COMMIT_GAIN_MS,
    SEGMENT_EMITTED,
    SEGMENT_PHASE_MS,
    SEGMENT_ERRORS,
    SEGMENT_ALIGN,
    SEGMENT_PREMATURE,
    SEGMENT_REVISED,
    STALE_RESULTS,
    STREAM_ERRORS,
    TRANSLATION_ERRORS,
    TRANSLATION_JOBS,
    TTS_ERRORS,
    TTS_JOBS,
    UTTERANCE_E2E_MS,
    VAD_EMPTY_ASR_TOTAL,
    VAD_SHORT_UTTERANCES_TOTAL,
    VAD_SPEECH_AUDIO_SECONDS,
    VAD_UTTERANCES_TOTAL,
    audio_preproc_backend,
    audio_preproc_reason,
)
from .segmenter import (
    ALIGN_EXACT,
    Alignment,
    SegmentCommit,
    SegmenterParams,
    StableSentenceSegmenter,
    normalize_words,
    remainder_after,
)
from .tasks import (
    TASK_KEYWORD,
    TASK_REALTIME_TRANSLATION,
    TASK_SPEECH_TO_TEXT,
    normalize_task,
)
from .trace_pcap import PcapTracer, StreamTrace
from .keyword_runtime import KeywordRuntime
from .pools import NemotronRuntimePool, TranslationRuntimePool, TtsRuntimePool, MagpieTtsRuntimePool
from .vad import VadEvent, build_vad_controller, warmup_silero_cache

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger(f"{__name__}.pipeline_trace")

# Backlog and short-utterance thresholds are now configurable via Settings
# (ASR_BACKLOG_SOFT_MS, ASR_BACKLOG_HARD_MS, SHORT_UTTERANCE_MS).


def _clamp_confidence(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


_RPC_END = object()

_SENTENCE_END = re.compile(r"[.?!](?=\s|$)")

# Phrase split: boundaries after [!?] (unconditional) or after [.] (only when
# followed by an uppercase letter — checked via unicodedata in _split_phrases).
_PHRASE_BOUNDARY_RE = re.compile(r'(?<=[!?])\s+|(?<=\.)\s+')


class FeatureUnavailableError(RuntimeError):
    pass


class DataLossError(RuntimeError):
    pass


class CapacityError(RuntimeError):
    pass


@dataclass(slots=True)
class RecognitionEnvelope:
    session_id: str
    stream_id: str
    text: str
    is_final: bool
    ts_start: int
    ts_end: int
    confidence: float
    language: str
    seq: int
    utterance_id: int
    audio_payload: bytes = b""
    audio_pt: int = 0
    topic_label: int = 0  # task=keyword FINAL: domain.TopicLabel value (proto speech.TopicLabel)
    topic_info: str = ""
    topic_score: float = 0.0
    created: float = field(default_factory=time.monotonic)
    speech_end_wall: float = 0.0  # estimated end of speech (monotonic); 0 = unknown
    preproc_delay_s: float = 0.0  # audio_preprocessing delay D that speech_end_wall does not include; 0 = no preprocessor


@dataclass(slots=True)
class RpcFailure:
    code_name: str
    details: str


ResponseItem = RecognitionEnvelope | RpcFailure | object


class AsyncResponseBuffer:
    """Soft-bounded response buffer.

    Text-only partial snapshots may be coalesced when a client is slow.
    Finals, failures, and audio-carrying partials (phrase audio) are never
    dropped or replaced by the coalescing logic.
    """

    def __init__(self, soft_limit: int) -> None:
        self._soft_limit = max(1, soft_limit)
        self._items: Deque[ResponseItem] = deque()
        self._event = asyncio.Event()
        self._closed = False

    def put(self, item: ResponseItem) -> bool:
        if self._closed:
            return False

        if isinstance(item, RecognitionEnvelope) and not item.is_final:
            # Audio-carrying partials (incremental phrase audio) must never be
            # coalesced — losing a synthesized audio chunk produces a gap in
            # playback that cannot be recovered.  Only text-only partials enter
            # the eviction path below.
            if not item.audio_payload and len(self._items) >= self._soft_limit:
                remove_index: int | None = None
                # Pass 1: prefer evicting a stale partial from the same stream.
                for index, existing in enumerate(self._items):
                    if (
                        isinstance(existing, RecognitionEnvelope)
                        and not existing.is_final
                        and not existing.audio_payload
                        and existing.stream_id == item.stream_id
                    ):
                        remove_index = index
                        break
                # Pass 2: fall back to the oldest text-only partial from any
                # stream so a slow stream cannot starve other streams.
                if remove_index is None:
                    for index, existing in enumerate(self._items):
                        if (
                            isinstance(existing, RecognitionEnvelope)
                            and not existing.is_final
                            and not existing.audio_payload
                        ):
                            remove_index = index
                            break
                if remove_index is None:
                    DROPPED_PARTIALS.inc()
                    return False
                del self._items[remove_index]
                DROPPED_PARTIALS.inc()

        self._items.append(item)
        self._event.set()
        return True

    async def get(self) -> ResponseItem:
        while True:
            if self._items:
                item = self._items.popleft()
                if not self._items:
                    self._event.clear()
                return item
            if self._closed:
                return _RPC_END
            self._event.clear()
            if self._items:  # re-check: put() may have fired between the empty-check and clear()
                continue
            await self._event.wait()

    def close(self) -> None:
        self._closed = True
        self._event.set()


@dataclass(slots=True)
class RpcContext:
    rpc_id: str
    response_buffer: AsyncResponseBuffer
    session_id: str | None = None
    stream_keys: set[StreamKey] = field(default_factory=set)
    input_closed: bool = False
    cancelled: bool = False
    finished: bool = False
    pending_postprocess: int = 0


@dataclass(slots=True)
class UtteranceContext:
    """Per-utterance state.

    Decoupled from StreamContext so a stream can hold one active utterance
    and zero or more finishing utterances concurrently.  When FINALIZE is
    submitted the utterance moves from stream.active_utterance to
    EngineCoordinator._finishing, freeing the active slot for the next
    VAD-triggered utterance without waiting for the FINAL result.
    """

    utterance_id: int
    source_language: str
    task: str
    target_language: str
    inference_pending: bool = False
    active_gauge_held: bool = False
    final_requested: bool = False
    first_partial_emitted: bool = False
    vad_end_wall: float = 0.0
    speech_end_wall: float = 0.0  # vad_end_wall minus the silence window and chunk tail
    latest_text: str = ""
    partial_steps: int = 0
    ts_start: int = 0
    ts_end: int = 0
    samples: int = 0
    wall_start: float = field(default_factory=time.monotonic)
    speech_lead_ms: float = 0.0  # wall_start minus arrival of the first speech chunk
    segmenter: StableSentenceSegmenter | None = None
    segmenter_failed: bool = False
    segments_submitted: int = 0
    committed_tokens: int = 0  # tokens of the segments sent so far (trace only)
    committed_text: str = ""  # the segments sent so far, joined; the remainder is cut after it
    align_traced: bool = False  # a partial already reported SEGMENT_ALIGN for this utterance
    segment_ts_cursor: int | None = None
    language_detected: bool = False  # source_language came from a vi/en detection, not a fallback
    passthrough_reason: str = ""  # why a translation utterance is replayed instead of translated
    source_audio: list[np.ndarray] = field(default_factory=list)  # 16 kHz audio kept for passthrough
    model_audio: SampleBuffer = field(init=False)

    def __post_init__(self) -> None:
        self.model_audio = SampleBuffer()


@dataclass(slots=True)
class _SttHeld:
    """Transcribe results of one utterance held until every earlier utterance has emitted its FINAL."""

    items: list[dict] = field(default_factory=list)  # _emit_recognition kwargs, in order; a partial only last
    done: bool = False  # its FINAL was handled (emitted, or nothing to emit)


@dataclass(slots=True)
class StreamContext:
    key: StreamKey
    settings: Settings
    phase: StreamPhase = StreamPhase.OPEN
    utterance_id: int = 0
    source_language: str = "vi-VN"
    task: str = "speech_to_text"
    target_language: str = ""
    declared_language: str = "auto"  # transcribe: normalized `language` as sent ("auto" = not sent)
    declared_language_source: str = ""  # normalized `language_source` as sent ("" = not sent / switch off)
    source_rate: int = 16_000
    channels: int = 1
    rtp_clock_rate: int = 16_000
    expected_chunk_seq: int | None = None
    close_after_stream: bool = False
    last_rtp_timestamp: int | None = None
    expected_rtp_timestamp: int | None = None
    last_source_rate: int = 16_000
    last_activity: float = field(default_factory=time.monotonic)
    last_language: str = ""  # last vi/en language detected at FINAL; auto-detect fallback
    active_utterance: UtteranceContext | None = None
    peak_amplitude: float = 0.0
    vad_started: bool = False
    speech_onset_wall: float = 0.0  # arrival of the first chunk of the current VAD speech run
    no_start_audio_ms: float = 0.0
    no_start_warned: bool = False
    preprocessor: StreamAudioPreprocessor | StatefulStreamPreprocessor | None = None
    raw_pre_roll: SampleBuffer | None = None  # raw_aligned twin of pre_roll, only with a preprocessor
    pcap_trace: StreamTrace | None = None  # trace_pcap.enabled: pcap of the audio received and after AudX
    # Transcribe results go out in utterance order (_emit_stt): with several ASR workers a later utterance can
    # reach FINAL first, and the proto result has no utterance_id for the client to sort on.
    stt_emit_next: int = 1  # lowest utterance whose FINAL is not handled yet
    stt_held: dict[int, _SttHeld] = field(default_factory=dict)  # later utterances' results, held
    vad: object = field(init=False)
    pre_roll: SampleBuffer = field(init=False)

    def __post_init__(self) -> None:
        self.vad = build_vad_controller(
            backend=self.settings.vad_backend,
            enabled=self.settings.vad_enabled,
            threshold=self.settings.vad_threshold,
            neg_threshold=self.settings.vad_neg_threshold,
            energy_db_threshold=self.settings.vad_energy_db_threshold,
            min_speech_ms=self.settings.vad_min_speech_ms,
            end_silence_ms=self.settings.vad_end_silence_ms,
            silero_model_path=self.settings.silero_model_path,
            device=self.settings.vad_device,
        )
        pre_roll_ms = self.settings.vad_pre_roll_ms + self.settings.vad_min_speech_ms
        pre_roll_limit = (
            None
            if not self.settings.vad_enabled
            else max(1, int(16_000 * pre_roll_ms / 1000))
        )
        self.pre_roll = SampleBuffer(max_samples=pre_roll_limit)


Slot = tuple[int, int]  # (utterance_id, segment_idx): one playback slot per segment
JobKey = tuple[StreamKey, int, int]  # (stream key, utterance_id, segment_idx)


@dataclass(slots=True)
class PipelineJobState:
    key: StreamKey
    utterance_id: int
    source_text: str
    source_language: str
    target_language: str
    ts_start: int
    ts_end: int
    asr_confidence: float = 0.0
    translated_text: str = ""
    wall_start: float = 0.0
    commit_wall: float = 0.0  # segment commit time; 0 = measure E2E from wall_start
    speech_lead_ms: float = 0.0
    vad_end_wall: float = 0.0  # 0 = VAD END had not fired when the job was created
    speech_end_wall: float = 0.0
    segment_idx: int = 0
    phrases: list[str] = field(default_factory=list)
    next_phrase_id: int = 0
    # Per-segment phase timing (monotonic walls / ms) feeding SEGMENT_LATENCY.
    submit_wall: float = 0.0
    asr_ms: float | None = None
    translate_queue_ms: float = 0.0
    translate_ms: float = 0.0
    translate_done_wall: float = 0.0
    tts_submit_wall: float = 0.0
    tts_queue_ms: float = 0.0
    tts_ms: float = 0.0
    tts_done_wall: float = 0.0
    latency_traced: bool = False


@dataclass(slots=True)
class _FallbackEmit:
    """Result without TTS audio buffered for in-order emission (audio_payload only for passthrough)."""
    utterance_id: int
    text: str
    language: str
    ts_start: int
    ts_end: int
    confidence: float
    wall_start: float
    commit_wall: float = 0.0
    segment_idx: int = 0
    audio_payload: bytes = b""
    audio_pt: int = 0
    speech_end_wall: float = 0.0


@dataclass(slots=True)
class PipelineOrderState:
    """Playback ordering state; every dict/set below is keyed by playback slot.

    A slot is (utterance_id, segment_idx).  Without segmentation every utterance
    has exactly one slot (utterance_id, 0).  With segmentation the number of
    segments of an utterance is only known once its ASR FINAL is processed
    (utt_total); until then a cursor that reaches the end of the committed
    segments waits for the next one instead of moving to the next utterance.
    """

    next_emit_slot: Slot           # next slot whose TTS result should be EMITTED
    next_tts_slot: Slot            # next slot whose TTS job should be SUBMITTED
    segmented: bool = False
    ready: dict[Slot, TtsResult] = field(default_factory=dict)
    fallback_ready: dict[Slot, _FallbackEmit] = field(default_factory=dict)
    translation_pending: dict[Slot, TranslationResult] = field(default_factory=dict)
    tts_skip: set[Slot] = field(default_factory=set)  # slots whose TTS will never come
    utt_total: dict[int, int] = field(default_factory=dict)  # segments per utterance, once known
    open_utts: set[int] = field(default_factory=set)  # utterances with segments but no total yet
    first_audio_seen: set[int] = field(default_factory=set)
    post_speech_seen: set[int] = field(default_factory=set)

    def _total_of(self, utterance_id: int) -> int | None:
        total = self.utt_total.get(utterance_id)
        if total is None and not self.segmented:
            return 1
        return total

    def step(self, slot: Slot) -> Slot:
        utterance_id, segment_idx = slot
        total = self._total_of(utterance_id)
        if total is not None and segment_idx + 1 >= total:
            return (utterance_id + 1, 0)
        return (utterance_id, segment_idx + 1)

    def advance_tts(self) -> None:
        self.next_tts_slot = self.step(self.next_tts_slot)
        self._prune()

    def advance_emit(self) -> None:
        self.next_emit_slot = self.step(self.next_emit_slot)
        self._prune()

    def set_total(self, utterance_id: int, total: int) -> None:
        self.utt_total[utterance_id] = total
        self.open_utts.discard(utterance_id)
        self.next_tts_slot = self._settle(self.next_tts_slot)
        self.next_emit_slot = self._settle(self.next_emit_slot)
        self._prune()

    def _settle(self, slot: Slot) -> Slot:
        while True:
            total = self.utt_total.get(slot[0])
            if total is None or slot[1] < total:
                return slot
            slot = (slot[0] + 1, 0)

    def _prune(self) -> None:
        low = min(self.next_emit_slot[0], self.next_tts_slot[0])
        for utterance_id in [uid for uid in self.utt_total if uid < low]:
            del self.utt_total[utterance_id]
        self.first_audio_seen = {uid for uid in self.first_audio_seen if uid >= low}
        self.post_speech_seen = {uid for uid in self.post_speech_seen if uid >= low}


class EngineCoordinator:
    """Single asyncio owner of RPC/stream lifecycle state.

    Model stages run independently:
      * NemotronRuntimePool: stateful ASR workers with stream affinity
      * KeywordRuntime: semantic topic detection on ASR-final text (task=keyword)
      * TranslationRuntimePool: stateless EnViT5 workers
      * TtsRuntimePool: Piper workers with private voice instances

    A stream holds at most one active utterance plus zero or more finishing
    utterances.  When VAD triggers FINALIZE the active slot is freed
    immediately; new audio flows into pre_roll/VAD while the old utterance
    drains through Nemotron, translation, and TTS in the background.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runtime_results: asyncio.Queue[RuntimeResult] = asyncio.Queue(maxsize=settings.result_queue_maxsize)
        self._translation_results: asyncio.Queue[TranslationResult] = asyncio.Queue(maxsize=settings.result_queue_maxsize)
        self._tts_results: asyncio.Queue[TtsResult] = asyncio.Queue(maxsize=settings.result_queue_maxsize)
        self._keyword_results: asyncio.Queue[KeywordResult] = asyncio.Queue(maxsize=settings.result_queue_maxsize)
        self._dispatcher_tasks: list[asyncio.Task[None]] = []
        self._rpcs: dict[str, RpcContext] = {}
        self._streams: dict[StreamKey, StreamContext] = {}
        self._pipeline_jobs: dict[JobKey, PipelineJobState] = {}
        self._pipeline_jobs_count: dict[StreamKey, int] = {}
        self._pipeline_order: dict[StreamKey, PipelineOrderState] = {}
        self._tts_retry_tasks: set[asyncio.Task] = set()
        self._finishing: dict[StreamKey, dict[int, UtteranceContext]] = {}
        self._stream_seq: dict[StreamKey, int] = {}
        self._active_utterances_count: int = 0
        # task=keyword utterances in topic detection -> their stream. The stream object is kept so the FINAL goes
        # out through _emit_stt (utterance order) even when the stream closed while detection was running.
        self._keyword_jobs: dict[tuple[StreamKey, int], StreamContext] = {}
        self._segmenter_params = SegmenterParams(
            stable_n=settings.segmentation_stable_n,
            stable_ms=settings.segmentation_stable_ms,
            min_tokens=settings.segmentation_min_tokens,
            max_segment_ms=settings.segmentation_max_segment_ms,
            comma_after_ms=settings.segmentation_comma_after_ms,
            punct_stable_n=settings.segmentation_punct_stable_n,
            punct_stable_ms=settings.segmentation_punct_stable_ms,
        )
        self._stt_segmenter_params = SegmenterParams(
            stable_n=settings.segmentation_stable_n,
            stable_ms=settings.segmentation_stable_ms,
            min_tokens=settings.segmentation_min_tokens,
            max_segment_ms=settings.segmentation_max_segment_ms,
            comma_after_ms=settings.segmentation_comma_after_ms,
            punct_stable_n=settings.segmentation_stt_punct_stable_n,
            punct_stable_ms=settings.segmentation_stt_punct_stable_ms,
        )

        self.runtime = NemotronRuntimePool(
            workers=settings.asr_workers,
            model_path=settings.model_path,
            devices=settings.asr_devices,
            language=settings.language,
            left_context=settings.left_context,
            lookahead_tokens=settings.lookahead_tokens,
            queue_size=settings.runtime_queue_size,
            warmup_seconds=settings.warmup_seconds,
            result_callback=self._on_runtime_result_from_thread,
            suppress_model_init_output=settings.suppress_model_init_output,
            beam_size=settings.asr_beam_size,
        )

        self.keyword_runtime: KeywordRuntime | None = None
        if settings.keyword_enabled:
            if not settings.keyword_model_path:
                raise RuntimeError("KEYWORD_ENABLED=true requires KEYWORD_MODEL_PATH to be set")
            self.keyword_runtime = KeywordRuntime(
                model_path=settings.keyword_model_path,
                device=settings.keyword_device,
                threshold=settings.keyword_threshold,
                topics=settings.keyword_topics,
                queue_size=settings.keyword_queue_size,
                result_callback=self._on_keyword_result_from_thread,
                third_party_log_level=settings.third_party_log_level,
            )

        self.translation_runtime: TranslationRuntimePool | None = None
        if settings.translation_enabled:
            if not settings.translation_model_path:
                raise RuntimeError("TRANSLATION_ENABLED=true requires TRANSLATION_MODEL_PATH to be set")
            if not settings.translation_tokenizer_path:
                raise RuntimeError("TRANSLATION_ENABLED=true requires TRANSLATION_TOKENIZER_PATH to be set")
            self.translation_runtime = TranslationRuntimePool(
                workers=settings.translation_workers,
                model_path=settings.translation_model_path,
                tokenizer_path=settings.translation_tokenizer_path,
                devices=settings.translation_devices,
                compute_type=settings.translation_compute_type,
                inter_threads=settings.translation_inter_threads,
                intra_threads=settings.translation_intra_threads,
                queue_size=settings.translation_queue_size,
                beam_size=settings.translation_beam_size,
                max_decoding_length=settings.translation_max_decoding_length,
                vi_prefix=settings.translation_vi_prefix,
                en_prefix=settings.translation_en_prefix,
                result_callback=self._on_translation_result_from_thread,
                batch_size=settings.translation_batch_size,
                batch_wait_ms=settings.translation_batch_wait_ms,
                third_party_log_level=settings.third_party_log_level,
            )

        self.tts_runtime: TtsRuntimePool | MagpieTtsRuntimePool | None = None
        if settings.tts_enabled:
            if settings.tts_backend == "magpie":
                self.tts_runtime = MagpieTtsRuntimePool(
                    workers=settings.tts_workers,
                    model_path=settings.tts_magpie_model_path,
                    speaker_index=settings.tts_speaker_index,
                    cfg_scale=settings.tts_cfg_scale,
                    temperature=settings.tts_temperature,
                    top_k=settings.tts_top_k,
                    volume=settings.tts_volume,
                    target_sample_rate=settings.tts_target_sample_rate,
                    queue_size=settings.tts_queue_size,
                    max_text_chars=settings.tts_max_text_chars,
                    devices=settings.tts_devices,
                    result_callback=self._on_tts_result_from_thread,
                    suppress_model_init_output=settings.suppress_model_init_output,
                    apply_tn=settings.tts_magpie_apply_tn,
                    codec_model_path=settings.tts_magpie_codec_model_path,
                )
            else:  # piper (default)
                if not settings.tts_en_model_path:
                    raise RuntimeError("TTS_ENABLED=true with TTS_BACKEND=piper requires TTS_EN_MODEL_PATH to be set")
                if not settings.tts_vi_model_path:
                    raise RuntimeError("TTS_ENABLED=true with TTS_BACKEND=piper requires TTS_VI_MODEL_PATH to be set")
                self.tts_runtime = TtsRuntimePool(
                    workers=settings.tts_workers,
                    en_model_path=settings.tts_en_model_path,
                    en_config_path=settings.tts_en_config_path,
                    vi_model_path=settings.tts_vi_model_path,
                    vi_config_path=settings.tts_vi_config_path,
                    target_sample_rate=settings.tts_target_sample_rate,
                    queue_size=settings.tts_queue_size,
                    max_text_chars=settings.tts_max_text_chars,
                    length_scale=settings.tts_length_scale,
                    noise_scale=settings.tts_noise_scale,
                    noise_w_scale=settings.tts_noise_w_scale,
                    volume=settings.tts_volume,
                    intra_threads=settings.tts_intra_threads,
                    devices=settings.tts_devices,
                    result_callback=self._on_tts_result_from_thread,
                )

        self._preprocessors: dict[StreamKey, StreamAudioPreprocessor | StatefulStreamPreprocessor] = {}
        self._vad_on_raw = settings.audio_preprocessing.vad_input == "raw"
        self._preproc_backend = audio_preproc_backend(settings.audio_preprocessing.mode)
        self._preproc_mode = settings.audio_preprocessing.mode  # reasons: DENOISER_* (resemble) or AUDX_*
        self.audio_preproc_pool: AudioPreprocessRuntimePool | None = None
        preproc = settings.audio_preprocessing
        if preproc.enabled:
            self.audio_preproc_pool = AudioPreprocessRuntimePool(
                workers=preproc.workers,
                devices=list(preproc.devices),
                queue_size=preproc.queue_max,
                warmup_samples=(preproc.left_context_ms + preproc.hop_ms + preproc.right_context_ms) * 16,
                model_loader=make_model_loader(preproc),
                result_callback=self._on_audio_preprocess_result_from_thread,
                backend=self._preproc_backend,
            )
        # docs/trace-pcap-design.md: off -> no tracer, no thread, nothing written
        self.pcap_tracer: PcapTracer | None = (
            PcapTracer(settings.trace_pcap) if settings.trace_pcap.enabled else None
        )

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        timeout = self.settings.model_loading_timeout_s
        await asyncio.to_thread(self.runtime.start, timeout)
        if self.keyword_runtime is not None:
            await asyncio.to_thread(self.keyword_runtime.start, timeout)
        if self.translation_runtime is not None:
            await asyncio.to_thread(self.translation_runtime.start, timeout)
        if self.tts_runtime is not None:
            await asyncio.to_thread(self.tts_runtime.start, timeout)
        if self.audio_preproc_pool is not None:
            try:
                await asyncio.to_thread(self.audio_preproc_pool.start, timeout)
            except Exception:
                # Optional stage: calls keep working on raw audio rather than the engine refusing to start.
                logger.error(
                    "AUDIO_PREPROCESSING_DISABLED: pool failed to start, engine runs without audio preprocessing",
                    exc_info=True,
                    extra={
                        "reason": audio_preproc_reason(self._preproc_mode, "START_FAILED"),
                        "stage": "AUDIO_PREPROCESSING_DISABLED",
                    },
                )
                AUDIO_PREPROC_POOL_WORKERS.labels(backend=self._preproc_backend).set(0)
                self.audio_preproc_pool = None
        if self.settings.vad_backend == "silero":
            await asyncio.to_thread(
                warmup_silero_cache,
                self.settings.silero_model_path,
                self.settings.vad_device,
            )

        self._dispatcher_tasks = [
            asyncio.create_task(self._runtime_dispatch_loop(), name="asr-result-dispatcher"),
            asyncio.create_task(self._keyword_dispatch_loop(), name="keyword-result-dispatcher"),
            asyncio.create_task(
                self._translation_dispatch_loop(), name="translation-result-dispatcher"
            ),
            asyncio.create_task(self._tts_dispatch_loop(), name="tts-result-dispatcher"),
        ]

    async def shutdown(self) -> None:
        for rpc_id in list(self._rpcs):
            await self.cancel_rpc(rpc_id, "engine shutdown")

        pools: list[tuple[str, object]] = [("ASR", self.runtime)]
        if self.keyword_runtime is not None:
            pools.append(("Keyword", self.keyword_runtime))
        if self.translation_runtime is not None:
            pools.append(("Translation", self.translation_runtime))
        if self.tts_runtime is not None:
            pools.append(("TTS", self.tts_runtime))
        if self.audio_preproc_pool is not None:
            pools.append(("Audio preprocessing", self.audio_preproc_pool))
        for name, pool in pools:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(pool.shutdown),  # type: ignore[attr-defined]
                    timeout=self.settings.shutdown_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "%s runtime shutdown timed out after %.0fs; forcing continue",
                    name,
                    self.settings.shutdown_timeout_s,
                )

        if self.pcap_tracer is not None:
            await asyncio.to_thread(self.pcap_tracer.close)

        for task in list(self._tts_retry_tasks):
            task.cancel()
        for task in self._dispatcher_tasks:
            task.cancel()
        for task in self._dispatcher_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _on_runtime_result_from_thread(self, result: RuntimeResult) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        def _put() -> None:
            try:
                self._runtime_results.put_nowait(result)
            except asyncio.QueueFull:
                logger.error("ASR result queue full; dropping %s", result.type.name)
                if result.key is not None and result.type is not RuntimeResultType.CANCELLED:
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        self.fail_rpc(result.key.rpc_id, "INTERNAL", "ASR result queue full"),
                    )
        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass  # loop closed in narrow race during shutdown

    def _on_keyword_result_from_thread(self, result: KeywordResult) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        def _put() -> None:
            try:
                self._keyword_results.put_nowait(result)
            except asyncio.QueueFull:
                logger.error("Keyword result queue full; dropping utterance=%s", result.utterance_id)
                loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self.fail_rpc(result.key.rpc_id, "INTERNAL", "keyword result queue full"),
                )
        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass

    def _on_translation_result_from_thread(self, result: TranslationResult) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        def _put() -> None:
            try:
                self._translation_results.put_nowait(result)
            except asyncio.QueueFull:
                logger.error("Translation result queue full; dropping utterance=%s", result.utterance_id)
                if result.key is not None:
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        self.fail_rpc(result.key.rpc_id, "INTERNAL", "translation result queue full"),
                    )
        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass

    def _on_tts_result_from_thread(self, result: TtsResult) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        def _put() -> None:
            try:
                self._tts_results.put_nowait(result)
            except asyncio.QueueFull:
                logger.error("TTS result queue full; dropping utterance=%s", result.utterance_id)
                if result.key is not None:
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        self.fail_rpc(result.key.rpc_id, "INTERNAL", "TTS result queue full"),
                    )
        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass

    def _on_audio_preprocess_result_from_thread(self, result: AudioPreprocessResult) -> None:
        pre = self._preprocessors.get(result.key)
        if pre is not None:
            pre.deliver(result)

    def _attach_preprocessor(self, stream: StreamContext) -> None:
        pool = self.audio_preproc_pool
        if pool is None:
            return
        cfg = self.settings.audio_preprocessing
        if random.random() * 100.0 >= cfg.sample_percent:
            return
        if cfg.max_streams > 0 and len(self._preprocessors) >= cfg.max_streams:
            AUDIO_PREPROC_SHED_STREAMS.labels(backend=self._preproc_backend).inc()
            logger.warning(
                "AUDIO_PREPROCESSING_FALLBACK",
                extra={
                    "rpc_id": stream.key.rpc_id,
                    "session_id": stream.key.session_id,
                    "stream_id": stream.key.stream_id,
                    "reason": audio_preproc_reason(self._preproc_mode, "MAX_STREAMS"),
                    "stage": "AUDIO_PREPROCESSING_FALLBACK",
                },
            )
            return
        pre: StreamAudioPreprocessor | StatefulStreamPreprocessor
        if cfg.mode == "audx":
            # the model state lives in the stream (one job in flight), see docs/audx-stateful-design.md
            pre = StatefulStreamPreprocessor(stream.key, cfg, pool.try_submit_nowait, new_state=AudxStreamState)
        else:
            pre = StreamAudioPreprocessor(stream.key, cfg, pool.try_submit_nowait)
        stream.preprocessor = pre
        stream.raw_pre_roll = SampleBuffer(max_samples=stream.pre_roll.max_samples)
        # VAD sees D ms of padding zeros first; they must not count as audio without speech.
        stream.no_start_audio_ms = -pre.delay_samples * 1000.0 / 16_000
        self._preprocessors[stream.key] = pre

    def _open_pcap_trace(self, stream: StreamContext, first_rtp_timestamp: int) -> StreamTrace | None:
        """trace_pcap: one pcap per stream; the "audx" flow only when the stream runs a preprocessor."""
        pre = stream.preprocessor
        return self.pcap_tracer.open_stream(  # type: ignore[union-attr]
            rpc_id=stream.key.rpc_id,
            session_id=stream.key.session_id,
            stream_id=stream.key.stream_id,
            task=stream.task,
            rtp_clock_rate=stream.rtp_clock_rate,
            first_rtp_timestamp=first_rtp_timestamp,
            audx_delay_samples=pre.delay_samples if pre is not None else None,
            preproc_stats=pre.stats if pre is not None else None,
        )

    @staticmethod
    def _close_pcap_trace(stream: StreamContext) -> None:
        trace, stream.pcap_trace = stream.pcap_trace, None
        if trace is not None:
            trace.close()

    def _detach_preprocessor(self, stream: StreamContext) -> None:
        pre = stream.preprocessor
        if pre is None:
            return
        stream.preprocessor = None
        if self._preprocessors.get(stream.key) is pre:
            del self._preprocessors[stream.key]
        pre.close()

    def _preprocess_chunk(
        self, stream: StreamContext, audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """(audio for VAD/ASR, raw_aligned or None without a preprocessor, held-back raw to play first after a failure)."""
        pre = stream.preprocessor
        if pre is None:
            return audio, None, None
        received = pre.samples_received
        try:
            out = pre.process(audio)
        except Exception:
            # A preprocessor bug must not cost the call: drop it, the stream continues on raw audio.
            held = pre.unreleased_raw(received)
            logger.exception(
                "AUDIO_PREPROCESSING_FALLBACK",
                extra={
                    "rpc_id": stream.key.rpc_id,
                    "session_id": stream.key.session_id,
                    "stream_id": stream.key.stream_id,
                    "reason": audio_preproc_reason(self._preproc_mode, "ERROR"),
                    "stage": "AUDIO_PREPROCESSING_FALLBACK",
                    "salvaged_ms": round(held.size * 1000.0 / 16_000, 1),
                },
            )
            self._detach_preprocessor(stream)
            return audio, None, held
        return out.enhanced, out.raw_aligned, None

    async def _flush_preprocessor(self, stream: StreamContext) -> None:
        """Release the audio still held back by the preprocessor delay through VAD/ASR as if the stream continued."""
        pre = stream.preprocessor
        if pre is None:
            return
        try:
            flushed = await pre.flush()
            enhanced, raw = flushed.enhanced, flushed.raw_aligned
        except Exception:
            enhanced = raw = pre.unreleased_raw()
            logger.exception(
                "AUDIO_PREPROCESSING_FALLBACK",
                extra={
                    "rpc_id": stream.key.rpc_id,
                    "session_id": stream.key.session_id,
                    "stream_id": stream.key.stream_id,
                    "reason": audio_preproc_reason(self._preproc_mode, "ERROR"),
                    "stage": "AUDIO_PREPROCESSING_FALLBACK",
                    "salvaged_ms": round(enhanced.size * 1000.0 / 16_000, 1),
                },
            )
        finally:
            self._detach_preprocessor(stream)
        if stream.pcap_trace is not None:  # the last D: content up to the end of what entered the preprocessor
            stream.pcap_trace.write_audx(enhanced, lag=0)
        await self._feed_released(stream, enhanced, raw, stream.expected_rtp_timestamp or 0)

    async def _feed_released(
        self, stream: StreamContext, enhanced: np.ndarray, raw: np.ndarray, rtp_ts: int
    ) -> None:
        """Push audio the preprocessor still held through VAD/ASR in 80 ms pieces, as if the stream continued."""
        piece = 16_000 * 80 // 1000
        for start in range(0, enhanced.size, piece):
            if stream.phase is StreamPhase.CLOSED:
                return
            part = enhanced[start : start + piece]
            samples = int(part.size)
            source_samples = int(round(samples * stream.source_rate / 16_000))
            await self._process_audio(
                stream=stream,
                audio=part,
                raw_audio=raw[start : start + piece],
                rtp_timestamp=rtp_ts,
                source_samples=source_samples,
                source_rate=stream.source_rate,
                duration_ms=samples * 1000.0 / 16_000,
                end_of_stream=False,
            )
            rtp_ts = self._rtp_add(
                rtp_ts, int(round(source_samples * stream.rtp_clock_rate / stream.source_rate))
            )

    @staticmethod
    def _slot_for(key: StreamKey, utterance_id: int, segment_idx: int = 0) -> Slot:
        """Playback slot of one segment of an utterance's output."""
        return (utterance_id, segment_idx)

    def _pipeline_job_add(self, job_key: JobKey, state: PipelineJobState) -> None:
        self._pipeline_jobs[job_key] = state
        key = job_key[0]
        self._pipeline_jobs_count[key] = self._pipeline_jobs_count.get(key, 0) + 1

    def _pipeline_job_remove(self, job_key: JobKey) -> PipelineJobState | None:
        state = self._pipeline_jobs.pop(job_key, None)
        if state is not None:
            key = job_key[0]
            count = self._pipeline_jobs_count.get(key, 0) - 1
            if count <= 0:
                self._pipeline_jobs_count.pop(key, None)
            else:
                self._pipeline_jobs_count[key] = count
        return state

    def _validate_chunk_seq(self, stream: StreamContext, chunk_seq: int) -> None:
        """Check transport-layer ordering via chunk_seq (separate from RTP media continuity)."""
        if chunk_seq < 0:
            return  # field not provided by this gateway version
        expected = stream.expected_chunk_seq
        if expected is None:
            stream.expected_chunk_seq = chunk_seq + 1
            return
        if chunk_seq == expected:
            stream.expected_chunk_seq = chunk_seq + 1
            return
        if chunk_seq > expected:
            logger.warning(
                "Transport chunk_seq gap detected",
                extra={
                    "rpc_id": stream.key.rpc_id,
                    "session_id": stream.key.session_id,
                    "stream_id": stream.key.stream_id,
                    "expected_seq": expected,
                    "received_seq": chunk_seq,
                    "missing_count": chunk_seq - expected,
                },
            )
            stream.expected_chunk_seq = chunk_seq + 1
        else:
            RTP_OUT_OF_ORDER.inc()
            logger.warning(
                "Out-of-order chunk_seq detected",
                extra={
                    "rpc_id": stream.key.rpc_id,
                    "session_id": stream.key.session_id,
                    "stream_id": stream.key.stream_id,
                    "expected_seq": expected,
                    "received_seq": chunk_seq,
                },
            )
            # Do not advance expected_chunk_seq; RTP timeline check handles behavior.

    def open_rpc(self) -> RpcContext:
        limit = self.settings.max_concurrent_rpcs
        if limit > 0 and len(self._rpcs) >= limit:
            raise CapacityError(
                f"server at capacity: {len(self._rpcs)}/{limit} concurrent RPCs"
            )
        rpc_id = uuid.uuid4().hex
        rpc = RpcContext(
            rpc_id=rpc_id,
            response_buffer=AsyncResponseBuffer(self.settings.response_queue_size),
        )
        self._rpcs[rpc_id] = rpc
        ACTIVE_RPCS.inc()
        logger.info("RPC opened", extra={"rpc_id": rpc_id, **self._stage_extra("GRPC_OPEN")})
        return rpc

    async def accept_chunk(self, rpc_id: str, request: object) -> None:
        _t0 = time.perf_counter()
        try:
            await self._accept_chunk_inner(rpc_id, request)
        finally:
            ACCEPT_CHUNK_MS.observe((time.perf_counter() - _t0) * 1000.0)

    async def _accept_chunk_inner(self, rpc_id: str, request: object) -> None:
        rpc = self._require_rpc(rpc_id)
        if rpc.cancelled or rpc.input_closed:
            return

        session_id = str(getattr(request, "session_id", "")).strip()
        stream_id = str(getattr(request, "stream_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required")
        if not stream_id:
            raise ValueError("stream_id is required")
        if rpc.session_id is None:
            rpc.session_id = session_id
        elif rpc.session_id != session_id:
            raise ValueError("session_id must not change within one Recognize RPC")

        sample_rate = int(getattr(request, "sample_rate", 0))
        channels = int(getattr(request, "channels", 0))
        payload = bytes(getattr(request, "pcm", b""))
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        end_of_stream = bool(getattr(request, "end_of_stream", False))
        if not payload and not end_of_stream:
            return

        source_samples = len(payload) // (2 * channels)
        actual_duration_ms = source_samples * 1000.0 / sample_rate
        provided_duration_ms = float(getattr(request, "duration_ms", 0) or 0)
        if provided_duration_ms > 0 and abs(provided_duration_ms - actual_duration_ms) > max(
            5.0, actual_duration_ms * 0.10
        ):
            DURATION_MISMATCHES.inc()
            logger.warning(
                "AudioChunk duration_ms does not match PCM payload; using payload duration",
                extra={
                    "rpc_id": rpc_id,
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "provided_duration_ms": provided_duration_ms,
                    "actual_duration_ms": actual_duration_ms,
                },
            )

        audio = decode_and_resample(payload, sample_rate, channels, 16_000)

        key = StreamKey(rpc_id, session_id, stream_id)
        stream = self._streams.get(key)

        raw_task = str(getattr(request, "task", "") or "").strip().lower()
        raw_language = str(getattr(request, "language", "") or "").strip()
        raw_language_source = str(getattr(request, "language_source", "") or "").strip()
        declared_language_source = (
            normalize_language(raw_language_source, self.settings.language)
            if raw_language_source and self.settings.use_language_source
            else ""
        )
        rtp_clock_rate_from_chunk = int(getattr(request, "rtp_clock_rate", 0))
        chunk_seq = int(getattr(request, "chunk_seq", -1))
        new_stream = stream is None

        if stream is None:
            task = normalize_task(raw_task or None)
            if task == TASK_KEYWORD and self.keyword_runtime is None:
                raise FeatureUnavailableError(
                    "task='keyword' requires KEYWORD_ENABLED=true and a configured local keyword model"
                )
            if task == TASK_REALTIME_TRANSLATION:
                if self.translation_runtime is None or self.tts_runtime is None:
                    raise FeatureUnavailableError(
                        "task='realtime_translation' requires TRANSLATION_ENABLED=true and TTS_ENABLED=true"
                    )
            target_language = ""
            declared_language = "auto"
            source_mode = "disabled" if raw_language_source and not declared_language_source else "auto"
            if task == TASK_REALTIME_TRANSLATION:
                # `language` is the language to translate INTO; the source is language_source
                # (when enabled and valid), otherwise detected by the ASR.
                source_language = "auto"
                if raw_language:
                    target_language = normalize_language(raw_language, self.settings.language)
                    if not is_translatable_language(target_language):
                        raise ValueError(
                            f"realtime_translation target language must be vi or en, got {raw_language!r}"
                        )
                if declared_language_source:
                    canonical = canonical_translatable_language(declared_language_source)
                    if canonical:
                        source_language, source_mode = canonical, "explicit"
                    else:
                        source_mode = "ignored"
                        logger.warning(
                            "language_source is not vi or en; ignored, source language stays auto",
                            extra={
                                "rpc_id": rpc_id,
                                "session_id": session_id,
                                "stream_id": stream_id,
                                "language_source": raw_language_source,
                            },
                        )
            else:
                declared_language = (
                    "auto" if not raw_language else normalize_language(raw_language, self.settings.language)
                )
                source_language = declared_language
                if declared_language_source:
                    source_language, source_mode = declared_language_source, "explicit"
            _rtp_clock = rtp_clock_rate_from_chunk or 8_000
            stream = await asyncio.to_thread(
                StreamContext,
                key=key,
                settings=self.settings,
                source_language=source_language,
                task=task,
                target_language=target_language,
                declared_language=declared_language,
                declared_language_source=declared_language_source,
                source_rate=sample_rate,
                channels=channels,
                rtp_clock_rate=_rtp_clock,
            )
            self._streams[key] = stream
            rpc.stream_keys.add(key)
            ACTIVE_STREAMS.inc()
            self._attach_preprocessor(stream)
            logger.info(
                "Logical stream opened",
                extra={
                    "rpc_id": rpc_id,
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "task": task,
                    "source_language": source_language,
                    "target_language": target_language,
                    "language_source": raw_language_source,
                    "source_mode": source_mode,
                    **self._stage_extra("STREAM_OPEN"),
                },
            )
        else:
            if raw_task:
                normalized_task = normalize_task(raw_task)
                if normalized_task != stream.task:
                    raise ValueError("task must not change within one logical stream")
            if sample_rate != stream.source_rate:
                raise ValueError("sample_rate must not change within one logical stream")
            if channels != stream.channels:
                raise ValueError("channels must not change within one logical stream")
            if raw_language:
                normalized = normalize_language(raw_language, self.settings.language)
                fixed_language = (
                    stream.target_language
                    if stream.task == TASK_REALTIME_TRANSLATION
                    else stream.declared_language
                )
                if normalized != fixed_language:
                    raise ValueError("language must not change within one logical stream")
            if (
                declared_language_source
                and declared_language_source != stream.declared_language_source
            ):
                raise ValueError("language_source must not change within one logical stream")
            if rtp_clock_rate_from_chunk and rtp_clock_rate_from_chunk != stream.rtp_clock_rate:
                raise ValueError("rtp_clock_rate must not change within one logical stream")

        rtp_timestamp = int(getattr(request, "rtp_timestamp", 0))

        # trace_pcap: the "in" flow is the chunk as the gateway sent it, before chunk_seq / RTP policy
        trace = stream.pcap_trace
        arrival = 0.0
        if new_stream and self.pcap_tracer is not None:
            trace = stream.pcap_trace = self._open_pcap_trace(stream, rtp_timestamp)
        if trace is not None:
            arrival = time.time()
            trace.write_in(audio, rtp_timestamp, arrival)

        AUDIO_SECONDS.inc(max(0.0, actual_duration_ms / 1000.0))
        INGRESS_CHUNKS.inc()
        if audio.size:
            chunk_peak = float(np.max(np.abs(audio)))
            INGRESS_PEAK.observe(chunk_peak)
            if chunk_peak > stream.peak_amplitude:
                stream.peak_amplitude = chunk_peak

        self._validate_chunk_seq(stream, chunk_seq)

        # _apply_rtp_policy may prepend silence for gap repair; it returns
        # None when the chunk should be dropped (out-of-order policy).
        result = self._apply_rtp_policy(
            stream=stream,
            audio=audio,
            rtp_timestamp=rtp_timestamp,
            source_samples=source_samples,
            source_rate=sample_rate,
            duration_ms=actual_duration_ms,
        )
        if result is not None:
            proc_audio, proc_rtp_ts, proc_source_samples, proc_duration_ms = result
            entered = proc_audio.size
            had_preprocessor = stream.preprocessor is not None
            if trace is not None:
                trace.note_processed(proc_rtp_ts, entered, arrival)
            proc_audio, raw_audio, held = self._preprocess_chunk(stream, proc_audio)
            if trace is not None:
                if held is not None:  # the preprocessor failed: held raw ends where this chunk starts
                    trace.write_audx(held, lag=entered)
                    trace.write_audx(proc_audio, lag=0)
                else:
                    trace.write_audx(proc_audio, lag=trace.delay if had_preprocessor and trace.delay else 0)
            if held is not None and held.size:
                held_ticks = int(round(held.size * stream.rtp_clock_rate / 16_000))
                await self._feed_released(stream, held, held, self._rtp_add(proc_rtp_ts, -held_ticks))
            await self._process_audio(
                stream=stream,
                audio=proc_audio,
                raw_audio=raw_audio,
                rtp_timestamp=proc_rtp_ts,
                source_samples=proc_source_samples,
                source_rate=sample_rate,
                duration_ms=proc_duration_ms,
                end_of_stream=end_of_stream,
            )

    async def close_input(self, rpc_id: str) -> None:
        rpc = self._rpcs.get(rpc_id)
        if rpc is None or rpc.cancelled:
            return
        rpc.input_closed = True
        for key in list(rpc.stream_keys):
            stream = self._streams.get(key)
            if stream is None:
                continue
            await self._flush_preprocessor(stream)
            if stream.phase is StreamPhase.CLOSED:
                continue
            stream.close_after_stream = True
            if stream.active_utterance is not None:
                utt = stream.active_utterance
                utt.final_requested = True
                stream.phase = StreamPhase.CLOSING
                await self._maybe_schedule_utterance(stream, utt)
            elif self._finishing.get(stream.key):
                stream.phase = StreamPhase.CLOSING
            else:
                await self._close_stream(stream)
        self._maybe_finish_rpc(rpc)

    async def cancel_rpc(self, rpc_id: str, reason: str) -> None:
        rpc = self._rpcs.get(rpc_id)
        if rpc is None or rpc.cancelled:
            return
        rpc.cancelled = True
        logger.info(
            "RPC cancelled: %s",
            reason,
            extra={"rpc_id": rpc_id, "reason": reason, **self._stage_extra("GRPC_CANCEL")},
        )

        for key in list(rpc.stream_keys):
            stream = self._streams.pop(key, None)
            if stream is None:
                continue
            stream.phase = StreamPhase.CLOSED
            self._detach_preprocessor(stream)
            self._close_pcap_trace(stream)

            utterances_to_cancel: list[UtteranceContext] = []
            if stream.active_utterance is not None:
                utterances_to_cancel.append(stream.active_utterance)
            utterances_to_cancel.extend(self._finishing.pop(key, {}).values())

            for utt in utterances_to_cancel:
                try:
                    self.runtime.submit_nowait(
                        RuntimeCommand(
                            type=RuntimeCommandType.CANCEL,
                            key=key,
                            utterance_id=utt.utterance_id,
                        )
                    )
                except Exception:
                    logger.exception("Failed to cancel runtime utterance")
                if utt.active_gauge_held:
                    ACTIVE_UTTERANCES.dec()
                    utt.active_gauge_held = False
                self._active_utterances_count = max(0, self._active_utterances_count - 1)

            ACTIVE_STREAMS.dec()
        rpc.stream_keys.clear()

        doomed_jobs = [job_key for job_key in self._pipeline_jobs if job_key[0].rpc_id == rpc_id]
        for job_key in doomed_jobs:
            self._pipeline_job_remove(job_key)
            PIPELINE_PENDING.dec()
        for key in [key for key in self._pipeline_order if key.rpc_id == rpc_id]:
            self._pipeline_order.pop(key, None)
            self._stream_seq.pop(key, None)
        for job_key in [job_key for job_key in self._keyword_jobs if job_key[0].rpc_id == rpc_id]:
            self._keyword_jobs.pop(job_key, None)
            self._stream_seq.pop(job_key[0], None)
        rpc.pending_postprocess = 0

        rpc.response_buffer.close()
        self._remove_rpc(rpc)

    async def fail_rpc(self, rpc_id: str, code_name: str, details: str) -> None:
        rpc = self._rpcs.get(rpc_id)
        if rpc is None or rpc.cancelled:
            return
        rpc.response_buffer.put(RpcFailure(code_name=code_name, details=details))
        await self.cancel_rpc(rpc_id, details)

    def _stream_live(self, stream: StreamContext) -> bool:
        """The stream is still registered and not closed (cancel_rpc/fail_rpc and _close_stream close and drop it).
        Checked again after an await, during which another coroutine may have done that."""
        return stream.phase is not StreamPhase.CLOSED and self._streams.get(stream.key) is stream

    def _stream_open(self, stream: StreamContext) -> bool:
        """Live and still taking new utterances (not CLOSING after close_input / end of stream)."""
        return stream.phase is StreamPhase.OPEN and self._streams.get(stream.key) is stream

    async def _vad_process(
        self, stream: StreamContext, audio: np.ndarray, duration_ms: float
    ) -> VadEvent:
        vad = stream.vad
        if vad.needs_executor:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, vad.process, audio, duration_ms)
        return vad.process(audio, duration_ms)

    def _warn_if_vad_silent(self, stream: StreamContext, duration_ms: float) -> None:
        limit = self.settings.vad_no_start_warn_ms
        if limit <= 0 or stream.no_start_warned:
            return
        stream.no_start_audio_ms += duration_ms
        if stream.no_start_audio_ms < limit:
            return
        stream.no_start_warned = True
        logger.warning(
            "VAD has not detected speech on this stream; check audio format, sample_rate and level",
            extra={
                "rpc_id": stream.key.rpc_id,
                "session_id": stream.key.session_id,
                "stream_id": stream.key.stream_id,
                "audio_ms": round(stream.no_start_audio_ms, 1),
                "vad_max_score": round(getattr(stream.vad, "max_score", 0.0), 3),
                "peak_dbfs": round(20.0 * math.log10(max(stream.peak_amplitude, 1e-8)), 1),
                "sample_rate": stream.source_rate,
                "channels": stream.channels,
            },
        )

    async def _process_audio(
        self,
        *,
        stream: StreamContext,
        audio: np.ndarray,
        rtp_timestamp: int,
        source_samples: int,
        source_rate: int,
        duration_ms: float,
        end_of_stream: bool,
        raw_audio: np.ndarray | None = None,
    ) -> None:
        """`audio` feeds ASR; `raw_audio` (same length) is the untouched, equally delayed audio kept for
        passthrough and, with vad_input=raw, fed to VAD. None means no preprocessing: raw is audio."""
        stream.last_activity = time.monotonic()

        if stream.phase is StreamPhase.CLOSED:
            return

        raw = audio if raw_audio is None else raw_audio
        vad_audio = raw if self._vad_on_raw else audio

        if stream.phase is StreamPhase.OPEN:
            if stream.active_utterance is None:
                stream.pre_roll.append(audio)
                if stream.raw_pre_roll is not None:
                    stream.raw_pre_roll.append(raw)
                if self._finishing.get(stream.key):
                    DEFERRED_BACKLOG_MS.observe(stream.pre_roll.samples * 1000.0 / 16_000)
                arrival = stream.last_activity
                vad_event = await self._vad_process(stream, vad_audio, duration_ms)
                if not self._stream_open(stream):
                    # The RPC failed or the input closed while VAD ran (executor): open no utterance.
                    return
                if getattr(stream.vad, "speech_candidate_ms", 0.0) > 0.0:
                    if stream.speech_onset_wall <= 0.0:
                        stream.speech_onset_wall = arrival
                elif vad_event is not VadEvent.START:
                    stream.speech_onset_wall = 0.0
                if vad_event is VadEvent.START:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "VAD START: utterance triggered",
                            extra={
                                "rpc_id": stream.key.rpc_id,
                                "session_id": stream.key.session_id,
                                "stream_id": stream.key.stream_id,
                                "pre_roll_ms": round(stream.pre_roll.samples * 1000.0 / 16_000, 1),
                                "task": stream.task,
                                "source_language": stream.source_language,
                            },
                        )
                    stream.vad_started = True
                    await self._start_utterance(stream, audio, rtp_timestamp, source_samples, source_rate)
                elif not stream.vad_started:
                    self._warn_if_vad_silent(stream, duration_ms)
            else:
                utt = stream.active_utterance
                if not utt.final_requested:
                    vad_event = await self._vad_process(stream, vad_audio, duration_ms)
                    if not self._stream_live(stream):
                        # Cancelled while VAD ran: the utterance is gone. A CLOSING stream goes on: a FINALIZE
                        # sent after a STEP still in flight takes this audio too.
                        return
                    tail: tuple[np.ndarray, int, float, int] | None = None
                    tail_raw: np.ndarray | None = None
                    if vad_event is VadEvent.END and not end_of_stream:
                        tail_samples = int(round(getattr(stream.vad, "end_tail_ms", 0.0) * 16_000 / 1000.0))
                        head_samples = audio.size - tail_samples
                        if tail_samples > 0 and head_samples > 0:
                            ratio = head_samples / audio.size
                            head_source = int(round(source_samples * ratio))
                            tail = (audio[head_samples:], source_samples - head_source, duration_ms * (1.0 - ratio), head_source)
                            if raw_audio is not None:
                                tail_raw = raw[head_samples:]
                                raw = raw[:head_samples]
                            audio = audio[:head_samples]
                            source_samples = head_source
                            duration_ms = duration_ms * ratio
                    backlog_ms = utt.model_audio.samples * 1000.0 / 16_000
                    INGRESS_BACKLOG_MS.observe(backlog_ms)
                    if backlog_ms > self.settings.asr_backlog_hard_ms:
                        ASR_BACKLOG_HARD.inc()
                    elif backlog_ms > self.settings.asr_backlog_soft_ms:
                        ASR_BACKLOG_SOFT.inc()

                    max_ingress_samples = int(16_000 * self.settings.max_ingress_audio_ms / 1000)
                    if utt.model_audio.samples + audio.size > max_ingress_samples:
                        await self.fail_rpc(
                            stream.key.rpc_id,
                            "RESOURCE_EXHAUSTED",
                            f"ingress audio backlog exceeded {self.settings.max_ingress_audio_ms} ms",
                        )
                        return
                    utt.model_audio.append(audio)
                    if utt.task == TASK_REALTIME_TRANSLATION:
                        utt.source_audio.append(raw)
                    utt.samples += audio.size
                    if vad_event is not VadEvent.END:
                        VAD_SPEECH_AUDIO_SECONDS.inc(duration_ms / 1000.0)
                    rtp_ticks = int(round(source_samples * stream.rtp_clock_rate / source_rate))
                    utt.ts_end = self._rtp_add(rtp_timestamp, rtp_ticks)

                    if vad_event is VadEvent.END:
                        utt.final_requested = True
                        utt.vad_end_wall = time.monotonic()
                        utt.speech_end_wall = utt.vad_end_wall - (
                            getattr(stream.vad, "end_silence_observed_ms", 0.0)
                            + getattr(stream.vad, "end_tail_ms", 0.0)
                        ) / 1000.0
                        self._trace(
                            "VAD_END",
                            stream.key,
                            utt.utterance_id,
                            utterance_audio_ms=round(utt.samples * 1000.0 / 16_000, 1),
                            vad_silence_ms=round((utt.vad_end_wall - utt.speech_end_wall) * 1000.0, 1),
                        )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "VAD END: utterance finalize requested",
                                extra={
                                    "rpc_id": stream.key.rpc_id,
                                    "session_id": stream.key.session_id,
                                    "stream_id": stream.key.stream_id,
                                    "utterance_id": utt.utterance_id,
                                    "utterance_audio_ms": round(utt.samples * 1000.0 / 16_000, 1),
                                    "model_backlog_ms": round(backlog_ms, 1),
                                    "vad_end_tail_ms": round(tail[0].size * 1000.0 / 16_000, 1) if tail else 0.0,
                                },
                            )
                    if utt.samples >= int(16_000 * self.settings.max_utterance_ms / 1000):
                        if not utt.final_requested:
                            utt.final_requested = True
                            self._trace(
                                "VAD_END",
                                stream.key,
                                utt.utterance_id,
                                utterance_audio_ms=round(utt.samples * 1000.0 / 16_000, 1),
                                reason="max_utterance",
                            )
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "VAD: max utterance duration reached, forcing finalize",
                                    extra={
                                        "rpc_id": stream.key.rpc_id,
                                        "session_id": stream.key.session_id,
                                        "stream_id": stream.key.stream_id,
                                        "utterance_id": utt.utterance_id,
                                        "utterance_audio_ms": round(utt.samples * 1000.0 / 16_000, 1),
                                    },
                                )

                    await self._maybe_schedule_utterance(stream, utt)

                    if tail is not None and stream.phase is StreamPhase.OPEN and stream.active_utterance is None:
                        tail_audio, tail_source, tail_ms, head_source = tail
                        head_ticks = int(round(head_source * stream.rtp_clock_rate / source_rate))
                        await self._process_audio(
                            stream=stream,
                            audio=tail_audio,
                            raw_audio=tail_raw,
                            rtp_timestamp=self._rtp_add(rtp_timestamp, head_ticks),
                            source_samples=tail_source,
                            source_rate=source_rate,
                            duration_ms=tail_ms,
                            end_of_stream=False,
                        )

        if end_of_stream:
            await self._flush_preprocessor(stream)
            if stream.phase is StreamPhase.CLOSED:
                return
            stream.close_after_stream = True
            if stream.active_utterance is not None:
                utt = stream.active_utterance
                utt.final_requested = True
                stream.phase = StreamPhase.CLOSING
                await self._maybe_schedule_utterance(stream, utt)
            elif self._finishing.get(stream.key):
                stream.phase = StreamPhase.CLOSING
            else:
                await self._close_stream(stream)

    async def _start_utterance(
        self,
        stream: StreamContext,
        audio: np.ndarray,
        rtp_timestamp: int,
        source_samples: int,
        source_rate: int,
    ) -> None:
        if not self._stream_open(stream):
            # Callers check after their awaits; this keeps a closed or closing stream from ever sending START.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "VAD START dropped: stream no longer open",
                    extra={
                        "rpc_id": stream.key.rpc_id,
                        "session_id": stream.key.session_id,
                        "stream_id": stream.key.stream_id,
                        "phase": stream.phase.name,
                    },
                )
            return
        utt_limit = self.settings.max_concurrent_utterances
        if utt_limit > 0 and self._active_utterances_count >= utt_limit:
            await self.fail_rpc(
                stream.key.rpc_id,
                "RESOURCE_EXHAUSTED",
                f"too many concurrent utterances (limit={utt_limit})",
            )
            return

        stream.utterance_id += 1
        utt = UtteranceContext(
            utterance_id=stream.utterance_id,
            source_language=stream.source_language,
            task=stream.task,
            target_language=stream.target_language,
        )
        if utt.task == TASK_REALTIME_TRANSLATION and utt.source_language != "auto":
            self._route_translation(stream, utt)
        if stream.speech_onset_wall > 0.0:
            utt.speech_lead_ms = max(0.0, (utt.wall_start - stream.speech_onset_wall) * 1000.0)
        stream.speech_onset_wall = 0.0
        initial_audio = stream.pre_roll.pop_all()
        raw_initial = initial_audio if stream.raw_pre_roll is None else stream.raw_pre_roll.pop_all()
        utt.model_audio.append(initial_audio)
        if utt.task == TASK_REALTIME_TRANSLATION:
            utt.source_audio.append(raw_initial)
        utt.samples += initial_audio.size

        rtp_ticks = int(round(source_samples * stream.rtp_clock_rate / source_rate))
        utt.ts_end = self._rtp_add(rtp_timestamp, rtp_ticks)
        # The pre-roll already holds this chunk, so it ends where this chunk ends (ts_end), not where it starts.
        # pre_roll is at 16 kHz internal rate; convert to RTP clock ticks.
        pre_roll_rtp_ticks = int(round(initial_audio.size * stream.rtp_clock_rate / 16_000))
        utt.ts_start = self._rtp_add(utt.ts_end, -pre_roll_rtp_ticks)

        if utt.task == TASK_REALTIME_TRANSLATION:
            # The playback order exists from the start of the utterance, so an end that never reaches
            # translation (empty ASR, passthrough) is recorded in it even if it is the first FINAL.
            self._ensure_pipeline_order(stream, utt.utterance_id)
        stream.active_utterance = utt
        ACTIVE_UTTERANCES.inc()
        utt.active_gauge_held = True
        self._active_utterances_count += 1

        status = self.runtime.try_submit_nowait(
            RuntimeCommand(
                type=RuntimeCommandType.START,
                key=stream.key,
                utterance_id=stream.utterance_id,
                language=stream.source_language,
            )
        )
        if status is not SubmitStatus.ACCEPTED:
            stream.active_utterance = None
            if utt.active_gauge_held:
                ACTIVE_UTTERANCES.dec()
                utt.active_gauge_held = False
            self._active_utterances_count -= 1
            await self.fail_rpc(
                stream.key.rpc_id,
                "RESOURCE_EXHAUSTED",
                "ASR queue full at utterance start",
            )
            return

        utt.inference_pending = True
        self._trace(
            "VAD_START",
            stream.key,
            utt.utterance_id,
            pre_roll_ms=round(initial_audio.size * 1000.0 / 16_000, 1),
            source_language=utt.source_language,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Utterance started",
                extra={
                    "rpc_id": stream.key.rpc_id,
                    "session_id": stream.key.session_id,
                    "stream_id": stream.key.stream_id,
                    "utterance_id": stream.utterance_id,
                    "task": stream.task,
                },
            )

    async def _maybe_schedule_utterance(self, stream: StreamContext, utt: UtteranceContext) -> None:
        if stream.phase is StreamPhase.CLOSED:
            return

        # Free the active slot as soon as finalize is requested, even while a STEP
        # is still in flight: audio arriving in that window must flow into the
        # next utterance's pre_roll/VAD instead of being dropped.
        if utt.final_requested and stream.active_utterance is utt:
            stream.active_utterance = None
            self._finishing.setdefault(stream.key, {})[utt.utterance_id] = utt
            stream.vad.reset()
            stream.pre_roll.clear()
            if stream.raw_pre_roll is not None:
                stream.raw_pre_roll.clear()

        if utt.inference_pending:
            return

        model_chunk_samples = int(16_000 * self.settings.model_chunk_ms / 1000)

        if utt.model_audio.samples >= model_chunk_samples and not utt.final_requested:
            audio = utt.model_audio.pop_exact(model_chunk_samples)
            status = self.runtime.try_submit_nowait(
                RuntimeCommand(
                    type=RuntimeCommandType.STEP,
                    key=stream.key,
                    utterance_id=utt.utterance_id,
                    language=utt.source_language,
                    audio=audio,
                )
            )
            if status is SubmitStatus.ACCEPTED:
                utt.inference_pending = True
            elif status is SubmitStatus.UNAVAILABLE:
                utt.model_audio.prepend(audio)
                await self.fail_rpc(
                    stream.key.rpc_id,
                    "RESOURCE_EXHAUSTED",
                    "ASR runtime stopped during utterance STEP",
                )
            else:  # OVERLOADED — retry on next chunk
                utt.model_audio.prepend(audio)
            return

        if utt.final_requested:
            remainder = utt.model_audio.pop_all()
            status = self.runtime.try_submit_nowait(
                RuntimeCommand(
                    type=RuntimeCommandType.FINALIZE,
                    key=stream.key,
                    utterance_id=utt.utterance_id,
                    language=utt.source_language,
                    audio=remainder,
                )
            )
            if status is SubmitStatus.ACCEPTED:
                utt.inference_pending = True
            else:
                await self.fail_rpc(
                    stream.key.rpc_id,
                    "RESOURCE_EXHAUSTED",
                    "ASR queue full at utterance finalize",
                )

    def _find_utterance(self, stream: StreamContext, utterance_id: int) -> UtteranceContext | None:
        if stream.active_utterance is not None and stream.active_utterance.utterance_id == utterance_id:
            return stream.active_utterance
        return self._finishing.get(stream.key, {}).get(utterance_id)

    def _pop_finishing(self, key: StreamKey, utterance_id: int) -> None:
        finishing = self._finishing.get(key)
        if finishing is None:
            return
        finishing.pop(utterance_id, None)
        if not finishing:
            self._finishing.pop(key, None)

    async def _runtime_dispatch_loop(self) -> None:
        while True:
            result = await self._runtime_results.get()
            try:
                if result.type is RuntimeResultType.SHUTDOWN:
                    return
                await self._handle_runtime_result(result)
            except Exception:
                STREAM_ERRORS.inc()
                logger.exception("Failed to dispatch ASR runtime result")
            finally:
                self._runtime_results.task_done()

    async def _keyword_dispatch_loop(self) -> None:
        while True:
            result = await self._keyword_results.get()
            try:
                await self._handle_keyword_result(result)
            except Exception:
                STREAM_ERRORS.inc()
                logger.exception("Failed to dispatch keyword/topic result")
            finally:
                self._keyword_results.task_done()

    async def _translation_dispatch_loop(self) -> None:
        while True:
            result = await self._translation_results.get()
            try:
                await self._handle_translation_result(result)
            except Exception:
                STREAM_ERRORS.inc()
                logger.exception("Failed to dispatch translation result")
            finally:
                self._translation_results.task_done()

    async def _tts_dispatch_loop(self) -> None:
        while True:
            result = await self._tts_results.get()
            try:
                await self._handle_tts_result(result)
            except Exception:
                STREAM_ERRORS.inc()
                logger.exception("Failed to dispatch TTS result")
            finally:
                self._tts_results.task_done()

    async def _handle_runtime_result(self, result: RuntimeResult) -> None:
        if result.key is None:
            return
        stream = self._streams.get(result.key)
        if stream is None:
            STALE_RESULTS.inc()
            if result.type is RuntimeResultType.FINAL:
                self._trace_stale("asr", result.key, result.utterance_id, "stream_gone")
            return

        utt = self._find_utterance(stream, result.utterance_id)
        if utt is None:
            STALE_RESULTS.inc()
            if result.type is RuntimeResultType.FINAL:
                self._trace_stale("asr", result.key, result.utterance_id, "utterance_gone")
            return

        if result.type is RuntimeResultType.STARTED:
            utt.inference_pending = False
            await self._maybe_schedule_utterance(stream, utt)
            return

        if result.type is RuntimeResultType.PARTIAL:
            utt.inference_pending = False
            if utt.source_language == "auto":
                self._resolve_auto_language(stream, utt, result.detected_language, final=False)
            if self.settings.pipeline_trace_partial:
                self._trace_partial(stream.key, utt, result)
            text_changed = bool(result.text) and result.text != utt.latest_text
            if text_changed:
                utt.latest_text = result.text
                if not utt.first_partial_emitted:
                    ASR_FIRST_PARTIAL_LATENCY_MS.observe(
                        (time.monotonic() - utt.wall_start) * 1000.0
                    )
                    utt.first_partial_emitted = True
            stt_commits: list[SegmentCommit] = []
            stt_segmented = self._stt_segmentation_live(utt)
            if stt_segmented:
                stt_commits = self._segment_update(stream.key, utt, result)
                self._emit_stt_segments(stream, utt, stt_commits, result.confidence)
            if (text_changed or stt_commits) and (
                utt.task in (TASK_SPEECH_TO_TEXT, TASK_KEYWORD)
                or (
                    self.settings.translate_emit_source_partials
                    and utt.source_language != "auto"
                )
            ):
                partial_text = self._segment_remainder(utt) if stt_segmented else result.text
                if partial_text:
                    partial = dict(
                        key=stream.key,
                        utterance_id=utt.utterance_id,
                        text=partial_text,
                        is_final=False,
                        ts_start=self._segment_ts(utt)[0] if stt_segmented else utt.ts_start,
                        ts_end=utt.ts_end,
                        language=utt.source_language,
                        confidence=result.confidence,
                    )
                    if utt.task in (TASK_SPEECH_TO_TEXT, TASK_KEYWORD):
                        self._emit_stt(stream, **partial)
                    else:  # translate source partials (translate_emit_source_partials): not ordered
                        self._emit_recognition(**partial)
            if self.settings.segmentation_mode == "shadow":
                self._segment_update(stream.key, utt, result)
            elif self.settings.segmentation_mode == "on" and self._segmentation_live(utt):
                commits = self._segment_update(stream.key, utt, result)
                if commits:
                    await self._submit_segments(
                        stream, utt, commits, result.confidence, asr_created=result.created
                    )
            await self._maybe_schedule_utterance(stream, utt)
            return

        if result.type is RuntimeResultType.FINAL:
            # Utterance must be in finishing (FINALIZE already submitted).
            self._pop_finishing(stream.key, utt.utterance_id)
            utt.inference_pending = False
            utt.latest_text = result.text or utt.latest_text
            if utt.active_gauge_held:
                ACTIVE_UTTERANCES.dec()
                utt.active_gauge_held = False
            self._active_utterances_count = max(0, self._active_utterances_count - 1)
            finalization_ms: float | None = None
            if utt.vad_end_wall > 0:
                finalization_ms = (time.monotonic() - utt.vad_end_wall) * 1000.0
                ASR_FINALIZATION_LATENCY_MS.observe(finalization_ms)
                audio_duration_s = utt.samples / 16_000
                if audio_duration_s > 0:
                    ASR_UTTERANCE_RTF.observe(finalization_ms / 1000.0 / audio_duration_s)
            VAD_UTTERANCES_TOTAL.inc()
            ASR_CONFIDENCE.observe(_clamp_confidence(result.confidence))
            ASR_CONFIDENCE_SOURCE.labels(
                source=result.metadata.get("confidence_source", "unavailable")
            ).inc()
            if utt.samples * 1000.0 / 16_000 < self.settings.short_utterance_ms:
                VAD_SHORT_UTTERANCES_TOTAL.inc()
            if not utt.latest_text.strip():
                VAD_EMPTY_ASR_TOTAL.inc()
            if utt.source_language == "auto":
                self._resolve_auto_language(stream, utt, result.detected_language, final=True)
            if utt.language_detected:
                stream.last_language = utt.source_language

            segment_trace = self.settings.segmentation_mode == "on" and (
                utt.task == TASK_REALTIME_TRANSLATION
                or (utt.task == TASK_SPEECH_TO_TEXT and self.settings.segmentation_speech_to_text)
            )
            self._trace(
                "ASR_FINAL",
                stream.key,
                utt.utterance_id,
                committed_tokens=utt.committed_tokens if segment_trace else None,
                remainder=self._segment_remainder(utt) if segment_trace else None,
                text=utt.latest_text,
                source_language=utt.source_language,
                target_language=utt.target_language or None,
                detected_language=result.detected_language or None,
                confidence=result.confidence,
                utterance_audio_ms=round(utt.samples * 1000.0 / 16_000, 1),
                latency_ms=None if finalization_ms is None else round(finalization_ms, 1),
                t_ms=round((time.monotonic() - utt.wall_start) * 1000.0, 1),
            )
            if self.settings.segmentation_mode in ("shadow", "on"):
                self._segment_finish(stream.key, utt)
            segmented = (
                self.settings.segmentation_mode == "on" and utt.task == TASK_REALTIME_TRANSLATION
            )

            if utt.task == TASK_SPEECH_TO_TEXT:
                final_text = self._segment_remainder(utt)
                # After mid-utterance commits, a punctuation-only tail is not worth a result.
                if utt.segments_submitted == 0 or normalize_words(final_text):
                    self._emit_stt(
                        stream,
                        last=True,
                        key=stream.key,
                        utterance_id=utt.utterance_id,
                        text=final_text,
                        is_final=True,
                        ts_start=self._segment_ts(utt)[0],
                        ts_end=utt.ts_end,
                        language=utt.source_language,
                        confidence=result.confidence,
                        wall_start=utt.wall_start,
                        segment_idx=utt.segments_submitted if utt.segments_submitted else None,
                    )
                else:
                    self._stt_utterance_done(stream, utt.utterance_id)
            elif utt.task == TASK_KEYWORD:
                if utt.latest_text.strip():
                    # The FINAL (text + topic) goes out when the topic result comes back (_handle_keyword_result).
                    await self._submit_keyword_detection(stream, utt, utt.latest_text, result.confidence)
                else:
                    # Empty ASR: nothing to classify. _emit_recognition skips the empty result
                    # (EMPTY_RESULTS_SKIPPED) and the transcribe order moves on.
                    self._emit_stt(
                        stream,
                        last=True,
                        key=stream.key,
                        utterance_id=utt.utterance_id,
                        text=utt.latest_text,
                        is_final=True,
                        ts_start=utt.ts_start,
                        ts_end=utt.ts_end,
                        language=utt.source_language,
                        confidence=result.confidence,
                        wall_start=utt.wall_start,
                    )
            elif not utt.target_language:
                await self._passthrough_utterance(
                    stream,
                    utt,
                    result.confidence,
                    utt.passthrough_reason or "no_target_language",
                    segmented,
                )
            else:
                remainder = self._segment_remainder(utt) if segmented else utt.latest_text
                segment_idx = utt.segments_submitted if segmented else 0
                # A punctuation-only tail (e.g. a trailing ".") is not worth a segment of its own.
                has_text = bool(normalize_words(remainder)) if segmented else bool(remainder.strip())
                if has_text:
                    ts_start, ts_end = self._segment_ts(utt)
                    await self._submit_translation_pipeline(
                        stream,
                        utt,
                        remainder,
                        result.confidence,
                        segment_idx=segment_idx,
                        ts_start=ts_start,
                        ts_end=ts_end,
                        utt_total=segment_idx + 1 if segmented else None,
                        asr_created=result.created,
                    )
                elif segment_idx > 0:
                    # Every segment was already committed mid-utterance; nothing is left to
                    # translate, so only the end of the utterance has to be recorded.
                    await self._close_utterance(stream.key, utt.utterance_id, segment_idx)
                else:
                    # Empty ASR text: advance pipeline order so subsequent utterances
                    # with non-empty text are not blocked indefinitely.
                    self._trace(
                        "DEGRADE",
                        stream.key,
                        utt.utterance_id,
                        reason="asr_empty",
                        text=utt.latest_text,
                    )
                    order = self._ensure_pipeline_order(stream, utt.utterance_id)
                    slot = self._slot_for(stream.key, utt.utterance_id)
                    if slot >= order.next_emit_slot:
                        if segmented:
                            order.set_total(utt.utterance_id, 1)
                        order.fallback_ready[slot] = _FallbackEmit(
                            utterance_id=utt.utterance_id,
                            text="",
                            language=utt.source_language,
                            ts_start=utt.ts_start,
                            ts_end=utt.ts_end,
                            confidence=0.0,
                            wall_start=utt.wall_start,
                        )
                        await self._skip_tts_slot(stream.key, utt.utterance_id)
                        await self._drain_pipeline_ready(stream.key)

            # Close stream when all audio has been finalized.
            if (
                stream.close_after_stream
                and stream.active_utterance is None
                and not self._finishing.get(stream.key)
            ):
                await self._close_stream(stream)
            return

        if result.type is RuntimeResultType.CANCELLED:
            return

        if result.type is RuntimeResultType.ERROR:
            STREAM_ERRORS.inc()
            await self.fail_rpc(
                stream.key.rpc_id, "INTERNAL", result.error or "Nemotron runtime error"
            )

    def _resolve_auto_language(
        self,
        stream: StreamContext,
        utt: UtteranceContext,
        detected: str,
        *,
        final: bool,
    ) -> None:
        """Set the source language (and, for translation, the target) of an utterance opened as auto.

        Transcription uses the ASR detection, else settings.language. Translation needs a
        vi/en detection; a missing or unsupported one (the ASR mislabels short English as
        pt/es) falls back to the stream's last vi/en language, and without one the utterance
        is replayed untranslated. Partials only lock on a real detection; fallbacks are
        decided at FINAL. The target is the stream's requested language; when it is missing
        or equals the source, the utterance is replayed untranslated (passthrough_reason).
        """
        translating = utt.task == TASK_REALTIME_TRANSLATION
        fallback = ""
        undetected = False
        if detected and (is_translatable_language(detected) or not translating):
            chosen = detected
            utt.language_detected = is_translatable_language(detected)
        elif not final:
            return
        elif translating and stream.last_language:
            chosen, fallback = stream.last_language, "last_language"
        else:
            chosen, fallback = self.settings.language, "default"
            undetected = translating
        utt.source_language = chosen
        if fallback:
            self._trace(
                "LANG_FALLBACK",
                stream.key,
                utt.utterance_id,
                detected_language=detected or None,
                source_language=chosen,
                fallback=fallback,
            )
        if translating:
            self._route_translation(stream, utt, undetected=undetected)

    def _route_translation(
        self, stream: StreamContext, utt: UtteranceContext, *, undetected: bool = False
    ) -> None:
        """Translate the utterance, or set passthrough_reason and clear its target to replay it."""
        if undetected:
            utt.passthrough_reason = "language_undetected"
        elif not stream.target_language:
            utt.passthrough_reason = "no_target_language"
        elif short_language(utt.source_language) == short_language(stream.target_language):
            utt.passthrough_reason = "source_equals_target"
        else:
            utt.passthrough_reason = ""
            utt.target_language = stream.target_language
            return
        utt.target_language = ""

    async def _passthrough_utterance(
        self,
        stream: StreamContext,
        utt: UtteranceContext,
        confidence: float,
        reason: str,
        segmented: bool,
    ) -> None:
        """Return the ASR text and the original audio (G.711) of an utterance that is not translated."""
        chunks, utt.source_audio = utt.source_audio, []
        has_text = bool(utt.latest_text.strip())
        audio_payload = b""
        if has_text and chunks:
            audio = resample(np.concatenate(chunks), 16_000, self.settings.tts_target_sample_rate)
            audio_payload = encode_g711(audio, self.settings.tts_output_pt)
        audio_pt = self.settings.tts_output_pt if audio_payload else 0
        if has_text:
            PASSTHROUGH_TOTAL.labels(reason=reason).inc()
            self._trace(
                "PASSTHROUGH",
                stream.key,
                utt.utterance_id,
                reason=reason,
                text=utt.latest_text,
                source_language=utt.source_language,
                target_language=stream.target_language or None,
                audio_bytes=len(audio_payload),
                audio_ms=round(utt.samples * 1000.0 / 16_000, 1) if audio_payload else 0.0,
            )
        else:
            self._trace(
                "DEGRADE", stream.key, utt.utterance_id, reason="asr_empty", text=utt.latest_text
            )
        order = self._ensure_pipeline_order(stream, utt.utterance_id)
        slot = self._slot_for(stream.key, utt.utterance_id)
        fallback = _FallbackEmit(
            utterance_id=utt.utterance_id,
            text=utt.latest_text if has_text else "",
            language=utt.source_language,
            ts_start=utt.ts_start,
            ts_end=utt.ts_end,
            confidence=confidence if has_text else 0.0,
            wall_start=utt.wall_start,
            audio_payload=audio_payload,
            audio_pt=audio_pt,
            speech_end_wall=utt.speech_end_wall,
        )
        if slot < order.next_emit_slot:  # already passed (not expected): nothing to wait for
            if has_text:
                self._emit_recognition(
                    key=stream.key,
                    utterance_id=fallback.utterance_id,
                    text=fallback.text,
                    is_final=True,
                    ts_start=fallback.ts_start,
                    ts_end=fallback.ts_end,
                    language=fallback.language,
                    confidence=fallback.confidence,
                    audio_payload=audio_payload,
                    audio_pt=audio_pt,
                    wall_start=fallback.wall_start,
                    speech_end_wall=fallback.speech_end_wall,
                )
            return
        # Queue behind earlier utterances still in translation/TTS so the audio keeps its order.
        if segmented:
            order.set_total(utt.utterance_id, 1)
        order.fallback_ready[slot] = fallback
        await self._skip_tts_slot(stream.key, utt.utterance_id)
        await self._drain_pipeline_ready(stream.key)

    def _stage_extra(self, stage: str) -> dict[str, str]:
        """Tag a lifecycle INFO line as a trace stage (only when pipeline_trace is on)."""
        return {"stage": stage} if self.settings.pipeline_trace else {}

    def _trace(
        self,
        stage: str,
        key: StreamKey,
        utterance_id: int,
        *,
        level: int | None = None,
        **fields: object,
    ) -> None:
        """Emit one structured log line for a pipeline stage of a single utterance."""
        if not self.settings.pipeline_trace:
            return
        if level is None:
            level = logging.getLevelName(self.settings.pipeline_trace_level)
        extra: dict[str, object] = {
            "stage": stage,
            "rpc_id": key.rpc_id,
            "session_id": key.session_id,
            "stream_id": key.stream_id,
            "utterance_id": utterance_id,
        }
        for name, value in fields.items():
            if value is None:
                continue
            if name in ("text", "source_text", "translated_text", "remainder", "committed_text"):
                extra[f"{name}_len"] = len(value)  # type: ignore[arg-type]
                if not self.settings.pipeline_trace_text:
                    continue
            elif name == "phrase_texts" and not self.settings.pipeline_trace_text:
                continue
            extra[name] = value
        trace_logger.log(level, "PIPELINE %s", stage, extra=extra)

    def _segmentation_live(self, utt: UtteranceContext) -> bool:
        """True when segments of this utterance can be translated before its FINAL (mode on)."""
        return (
            utt.task == TASK_REALTIME_TRANSLATION
            and not utt.segmenter_failed
            and utt.source_language != "auto"
            and bool(utt.target_language)
            and self.translation_runtime is not None
            and self.tts_runtime is not None
        )

    def _segmentation_applies(self, utt: UtteranceContext) -> bool:
        """True when the segmenter observes this utterance's task (translation, or STT behind its flag)."""
        return utt.task == TASK_REALTIME_TRANSLATION or (
            utt.task == TASK_SPEECH_TO_TEXT and self.settings.segmentation_speech_to_text
        )

    def _stt_segmentation_live(self, utt: UtteranceContext) -> bool:
        """True when committed sentences of a speech_to_text utterance are emitted before its FINAL."""
        return (
            self.settings.segmentation_mode == "on"
            and utt.task == TASK_SPEECH_TO_TEXT
            and self.settings.segmentation_speech_to_text
            and not utt.segmenter_failed
        )

    def _emit_stt(self, stream: StreamContext, *, last: bool = False, **emit) -> None:
        """Emit a transcribe result (kwargs of _emit_recognition) in utterance order; last: the utterance's FINAL.

        Results of an utterance after stream.stt_emit_next are held until the earlier ones have emitted their
        FINAL. Of the held partials only the latest is kept: any newer result of the utterance replaces it.
        """
        utterance_id = emit["utterance_id"]
        if utterance_id <= stream.stt_emit_next:
            self._emit_recognition(**emit)
            if last:
                self._stt_utterance_done(stream, utterance_id)
            return
        held = stream.stt_held.setdefault(utterance_id, _SttHeld())
        if held.items and not held.items[-1]["is_final"]:
            held.items.pop()
        held.items.append(emit)
        held.done = held.done or last

    def _stt_utterance_done(self, stream: StreamContext, utterance_id: int) -> None:
        """The FINAL of a transcribe utterance is handled (also when it had nothing to emit): once the awaited
        utterance is done, emit what the following ones hold, up to the first one still unfinished."""
        if utterance_id > stream.stt_emit_next:
            stream.stt_held.setdefault(utterance_id, _SttHeld()).done = True
            return
        if utterance_id < stream.stt_emit_next:
            return
        stream.stt_emit_next += 1
        while (held := stream.stt_held.pop(stream.stt_emit_next, None)) is not None:
            for emit in held.items:
                self._emit_recognition(**emit)
            if not held.done:
                return
            stream.stt_emit_next += 1

    def _emit_stt_segments(
        self, stream: StreamContext, utt: UtteranceContext, commits: list[SegmentCommit], confidence: float
    ) -> None:
        for commit in commits:
            ts_start, ts_end = self._segment_ts(utt)
            utt.segments_submitted += 1
            self._record_committed(utt, commit.text)
            utt.segment_ts_cursor = ts_end
            self._emit_stt(
                stream,
                key=stream.key,
                utterance_id=utt.utterance_id,
                text=commit.text,
                is_final=True,
                ts_start=ts_start,
                ts_end=ts_end,
                language=utt.source_language,
                confidence=confidence,
                wall_start=utt.wall_start,
                segment_idx=commit.segment_idx,
            )

    @staticmethod
    def _record_committed(utt: UtteranceContext, text: str) -> None:
        utt.committed_tokens += len(text.split())
        utt.committed_text = f"{utt.committed_text} {text}" if utt.committed_text else text

    @staticmethod
    def _segment_remainder(utt: UtteranceContext) -> str:
        """The latest ASR text after the segments already sent, found by word alignment (see committed_end)."""
        if not utt.committed_text:
            return utt.latest_text
        return remainder_after(utt.committed_text, utt.latest_text)

    @staticmethod
    def _segment_ts(utt: UtteranceContext) -> tuple[int, int]:
        start = utt.segment_ts_cursor if utt.segment_ts_cursor is not None else utt.ts_start
        return start, utt.ts_end

    async def _submit_segments(
        self,
        stream: StreamContext,
        utt: UtteranceContext,
        commits: list[SegmentCommit],
        confidence: float,
        asr_created: float | None = None,
    ) -> None:
        for commit in commits:
            ts_start, ts_end = self._segment_ts(utt)
            segment_idx = utt.segments_submitted
            utt.segments_submitted += 1
            self._record_committed(utt, commit.text)
            utt.segment_ts_cursor = ts_end
            await self._submit_translation_pipeline(
                stream,
                utt,
                commit.text,
                confidence,
                segment_idx=segment_idx,
                ts_start=ts_start,
                ts_end=ts_end,
                asr_created=asr_created,
                first_seen_ms=commit.first_seen_ms,
            )

    async def _close_utterance(self, key: StreamKey, utterance_id: int, total: int) -> None:
        """Record how many segments an utterance has, releasing the cursors waiting on it."""
        order = self._pipeline_order.get(key)
        rpc = self._rpcs.get(key.rpc_id)
        if order is None or rpc is None or rpc.cancelled:
            return
        order.set_total(utterance_id, total)
        await self._drain_tts_pending(order, key, rpc)
        await self._drain_pipeline_ready(key)

    def _segment_update(
        self, key: StreamKey, utt: UtteranceContext, result: RuntimeResult
    ) -> list[SegmentCommit]:
        """Observe every PARTIAL, log/measure commits; returns them for mode 'on' to submit."""
        if not self._segmentation_applies(utt) or utt.segmenter_failed or not result.text:
            return []
        try:
            if utt.segmenter is None:
                params = (
                    self._stt_segmenter_params
                    if utt.task == TASK_SPEECH_TO_TEXT
                    else self._segmenter_params
                )
                utt.segmenter = StableSentenceSegmenter(params, utt.source_language)
            else:
                utt.segmenter.set_language(utt.source_language)
            committed_before = utt.segmenter.committed_text
            update = utt.segmenter.update(
                result.text,
                (time.monotonic() - utt.wall_start) * 1000.0,
                model_step=bool(result.metadata.get("model_step_executed", True)),
            )
            self._segment_trace_alignment(
                key, utt, update.alignment, committed_before, result.text, "partial"
            )
            for index in update.revised:
                SEGMENT_REVISED.inc()
                self._trace("SEGMENT_REVISED", key, utt.utterance_id, segment_idx=index, phase="partial")
            for commit in update.commits:
                self._segment_log_commit(key, utt, commit)
            return update.commits
        except Exception:
            self._segment_disable(key, utt)
            return []

    def _segment_finish(self, key: StreamKey, utt: UtteranceContext) -> None:
        segmenter = utt.segmenter
        if segmenter is None or utt.segmenter_failed:
            return
        try:
            committed_before = segmenter.committed_text
            result = segmenter.finish(
                utt.latest_text, (time.monotonic() - utt.wall_start) * 1000.0
            )
            self._segment_trace_alignment(
                key, utt, result.alignment, committed_before, utt.latest_text, "final"
            )
            for index in result.revised:
                SEGMENT_REVISED.inc()
                self._trace("SEGMENT_REVISED", key, utt.utterance_id, segment_idx=index, phase="final")
            for commit in result.commits:
                self._segment_log_commit(key, utt, commit)
            for judgement in result.judgements:
                SEGMENT_EARLY_COMMIT_GAIN_MS.observe(max(0.0, judgement.gain_ms))
                if judgement.premature:
                    SEGMENT_PREMATURE.labels(kind=judgement.kind).inc()
                self._trace(
                    "SEGMENT_SUMMARY",
                    key,
                    utt.utterance_id,
                    segment_idx=judgement.segment_idx,
                    kind=judgement.kind,
                    text=judgement.text,
                    t_ms=round(judgement.t_ms, 1),
                    gain_ms=round(judgement.gain_ms, 1),
                    premature=judgement.premature,
                    revised=judgement.revised,
                )
        except Exception:
            self._segment_disable(key, utt)
        finally:
            utt.segmenter = None

    def _segment_trace_alignment(
        self,
        key: StreamKey,
        utt: UtteranceContext,
        alignment: Alignment,
        committed_text: str,
        text: str,
        phase: str,
    ) -> None:
        """Count and trace a cut that had to re-find committed text revised by ASR (partials: once per utterance)."""
        if alignment.how == ALIGN_EXACT:
            return
        if phase == "partial":
            if utt.align_traced:
                return
            utt.align_traced = True
        SEGMENT_ALIGN.labels(how=alignment.how, phase=phase).inc()
        self._trace(
            "SEGMENT_ALIGN",
            key,
            utt.utterance_id,
            how=alignment.how,
            phase=phase,
            committed_text=committed_text,
            text=text,
            remainder=" ".join(text.split()[alignment.index :]),
        )

    def _segment_log_commit(self, key: StreamKey, utt: UtteranceContext, commit: SegmentCommit) -> None:
        SEGMENT_COMMITS.labels(kind=commit.kind).inc()
        self._trace(
            "SEGMENT_COMMIT",
            key,
            utt.utterance_id,
            segment_idx=commit.segment_idx,
            kind=commit.kind,
            text=commit.text,
            t_ms=round(commit.t_ms, 1),
            step=commit.step,
            forced_cut=commit.forced_cut,
            language_known=utt.source_language != "auto",
        )

    def _segment_disable(self, key: StreamKey, utt: UtteranceContext) -> None:
        utt.segmenter_failed = True
        utt.segmenter = None
        SEGMENT_ERRORS.inc()
        logger.exception(
            "Segmenter failed; disabled for this utterance",
            extra={
                "rpc_id": key.rpc_id,
                "session_id": key.session_id,
                "stream_id": key.stream_id,
                "utterance_id": utt.utterance_id,
            },
        )

    def _trace_partial(self, key: StreamKey, utt: UtteranceContext, result: RuntimeResult) -> None:
        """One ASR_PARTIAL line per ASR step, used to measure how stable partial text is."""
        utt.partial_steps += 1
        prev = utt.latest_text
        text = result.text or ""
        changed = bool(text) and text != prev
        fields: dict[str, object] = {
            "step": utt.partial_steps,
            "model_step": bool(result.metadata.get("model_step_executed")),
            "changed": changed,
            "audio_ms": round(utt.samples * 1000.0 / 16_000, 1),
            "t_ms": round((time.monotonic() - utt.wall_start) * 1000.0, 1),
            "infer_ms": round(result.latency_ms, 1),
            "queue_ms": round(float(result.metadata.get("queue_ms", 0.0)), 1),
        }
        if changed:
            prev_words = prev.split()
            new_words = text.split()
            common = 0
            for old_word, new_word in zip(prev_words, new_words):
                if old_word != new_word:
                    break
                common += 1
            fields.update(
                text=text,
                confidence=_clamp_confidence(result.confidence),
                common_prefix_words=common,
                revised_words=len(prev_words) - common,
                sentence_ends=len(_SENTENCE_END.findall(text)),
            )
        self._trace("ASR_PARTIAL", key, utt.utterance_id, **fields)

    @staticmethod
    def _since(wall_start: float) -> float | None:
        """Milliseconds since the utterance started, or None when the start time is unknown."""
        if wall_start <= 0:
            return None
        return round((time.monotonic() - wall_start) * 1000.0, 1)

    def _trace_tts_continuation_submit(self, tts_job: TtsJob, job: PipelineJobState | None) -> None:
        self._trace(
            "TTS_SUBMIT",
            tts_job.key,
            tts_job.utterance_id,
            segment_idx=tts_job.segment_idx,
            phrase_id=tts_job.phrase_id,
            phrases=len(job.phrases) if job is not None else None,
            text=tts_job.text,
            target_language=tts_job.target_language,
            queue_depth=self._queue_depth(self.tts_runtime),
            t_ms=self._since(job.wall_start) if job is not None else None,
        )

    def _trace_stale(
        self,
        origin: str,
        key: StreamKey,
        utterance_id: int,
        reason: str,
        segment_idx: int | None = None,
    ) -> None:
        self._trace(
            "STALE", key, utterance_id, origin=origin, reason=reason, segment_idx=segment_idx
        )

    def _observe_first_audio(
        self, job: PipelineJobState, order: PipelineOrderState | None = None
    ) -> None:
        """Record the latency to the first audio of an utterance (once per utterance)."""
        if order is None:
            order = self._pipeline_order.get(job.key)
        if order is None:
            return
        now = time.monotonic()
        if job.wall_start > 0 and job.utterance_id not in order.first_audio_seen:
            order.first_audio_seen.add(job.utterance_id)
            elapsed_ms = (now - job.wall_start) * 1000.0
            FIRST_AUDIO_MS.observe(elapsed_ms)
            FIRST_AUDIO_FROM_SPEECH_MS.observe(elapsed_ms + job.speech_lead_ms)
        if job.vad_end_wall > 0 and job.utterance_id not in order.post_speech_seen:
            order.post_speech_seen.add(job.utterance_id)
            POST_SPEECH_LATENCY_MS.observe((now - job.vad_end_wall) * 1000.0)

    def _observe_segment_latency(self, job: PipelineJobState) -> None:
        """Record per-phase latency of a segment at its first audio emit (once per segment)."""
        if job.latency_traced or job.submit_wall <= 0:
            return
        job.latency_traced = True
        now = time.monotonic()
        order_hold_ms = 0.0
        if job.translate_done_wall > 0 and job.tts_submit_wall > 0:
            order_hold_ms += (job.tts_submit_wall - job.translate_done_wall) * 1000.0
        if job.tts_done_wall > 0:
            order_hold_ms += (now - job.tts_done_wall) * 1000.0
        phases: dict[str, float | None] = {
            "asr": job.asr_ms,
            "translate_queue": job.translate_queue_ms,
            "translate": job.translate_ms,
            "tts_queue": job.tts_queue_ms,
            "tts": job.tts_ms,
            "order_hold": order_hold_ms,
            "total": (now - job.submit_wall) * 1000.0,
        }
        for phase, value in phases.items():
            if value is not None:
                SEGMENT_PHASE_MS.labels(phase=phase).observe(max(0.0, value))
        self._trace(
            "SEGMENT_LATENCY",
            job.key,
            job.utterance_id,
            segment_idx=job.segment_idx,
            phrases=len(job.phrases),
            **{
                f"{phase}_ms": None if value is None else round(max(0.0, value), 1)
                for phase, value in phases.items()
            },
        )

    @staticmethod
    def _stale_reason(job: object | None, rpc: RpcContext | None) -> str:
        if rpc is None:
            return "rpc_gone"
        if rpc.cancelled:
            return "rpc_cancelled"
        return "job_missing" if job is None else "unknown"

    @staticmethod
    def _queue_depth(runtime: object | None) -> int | None:
        depth = getattr(runtime, "queue_depth", None)
        return depth if isinstance(depth, int) else None

    async def _skip_tts_slot(self, key: StreamKey, utterance_id: int) -> None:
        """Mark an utterance that never reaches translation/TTS so later ones are not held behind it."""
        order = self._pipeline_order.get(key)
        rpc = self._rpcs.get(key.rpc_id)
        slot = self._slot_for(key, utterance_id)
        if order is None or rpc is None or slot < order.next_tts_slot:
            return
        if slot == order.next_tts_slot:
            order.advance_tts()
            await self._drain_tts_pending(order, key, rpc)
        else:
            order.tts_skip.add(slot)

    async def _submit_keyword_detection(
        self,
        stream: StreamContext,
        utt: UtteranceContext,
        source_text: str,
        asr_confidence: float,
    ) -> None:
        """Send the ASR-final text of a task=keyword utterance to KeywordRuntime.

        The utterance's FINAL (text + topic_label/topic_score/topic_info) is emitted by _handle_keyword_result,
        in utterance order through _emit_stt. When the runtime is unavailable or its queue is full the text is
        emitted at once with TOPIC_UNKNOWN (same graceful degradation as translation).
        """
        if self.keyword_runtime is None:
            raise FeatureUnavailableError("keyword runtime is unavailable")
        rpc = self._rpcs.get(stream.key.rpc_id)
        if rpc is None or rpc.cancelled:
            return
        if rpc.pending_postprocess >= self.settings.max_postprocess_pending_per_rpc:
            await self.fail_rpc(
                rpc.rpc_id,
                "RESOURCE_EXHAUSTED",
                "too many pending keyword/topic utterances",
            )
            return

        job_key = (stream.key, utt.utterance_id)
        job = KeywordJob(
            key=stream.key,
            utterance_id=utt.utterance_id,
            source_text=source_text,
            source_language=utt.source_language,
            ts_start=utt.ts_start,
            ts_end=utt.ts_end,
            asr_confidence=_clamp_confidence(asr_confidence),
            wall_start=utt.wall_start,
        )
        try:
            status = self.keyword_runtime.try_submit_nowait(job)
        except Exception:
            logger.exception("Keyword try_submit_nowait raised unexpectedly; degrading")
            status = SubmitStatus.OVERLOADED

        if status is SubmitStatus.ACCEPTED:
            self._keyword_jobs[job_key] = stream
            rpc.pending_postprocess += 1
            KEYWORD_JOBS.inc()
            self._trace(
                "KEYWORD_SUBMIT",
                stream.key,
                utt.utterance_id,
                source_text=source_text,
                source_language=utt.source_language,
                pending_postprocess=rpc.pending_postprocess,
                queue_depth=self._queue_depth(self.keyword_runtime),
                t_ms=self._since(utt.wall_start),
            )
            return

        KEYWORD_ERRORS.inc()
        self._trace(
            "DEGRADE",
            stream.key,
            utt.utterance_id,
            level=logging.WARNING,
            reason="keyword_queue_full",
            text=source_text,
            status=status.name,
            queue_depth=self._queue_depth(self.keyword_runtime),
        )
        self._emit_keyword_final(stream, job, TopicLabel.UNKNOWN, "", 0.0)

    def _emit_keyword_final(
        self,
        stream: StreamContext,
        job: KeywordJob | KeywordResult,
        topic_label: TopicLabel,
        topic_info: str,
        topic_score: float,
    ) -> None:
        KEYWORD_TOPICS.labels(topic=topic_label.name).inc()
        self._emit_stt(
            stream,
            last=True,
            key=job.key,
            utterance_id=job.utterance_id,
            text=job.source_text,
            is_final=True,
            ts_start=job.ts_start,
            ts_end=job.ts_end,
            language=job.source_language,
            confidence=job.asr_confidence,
            topic_label=int(topic_label),
            topic_info=topic_info,
            topic_score=topic_score,
            wall_start=job.wall_start,
        )

    async def _handle_keyword_result(self, result: KeywordResult) -> None:
        job_key = (result.key, result.utterance_id)
        stream = self._keyword_jobs.pop(job_key, None)
        if stream is None:
            STALE_RESULTS.inc()
            self._trace_stale("keyword", result.key, result.utterance_id, "job_gone")
            return
        rpc = self._rpcs.get(result.key.rpc_id)
        if rpc is None or rpc.cancelled:
            return
        rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
        KEYWORD_LATENCY_MS.observe(result.latency_ms)
        KEYWORD_QUEUE_WAIT_MS.observe(result.queue_ms)

        if result.error:
            KEYWORD_ERRORS.inc()
            self._trace(
                "DEGRADE",
                result.key,
                result.utterance_id,
                level=logging.WARNING,
                reason="keyword_error",
                text=result.source_text,
                error=result.error,
            )
            label, info, score = TopicLabel.UNKNOWN, "", 0.0
        else:
            label, info, score = result.topic_label, result.topic_info, result.topic_score
            self._trace(
                "KEYWORD",
                result.key,
                result.utterance_id,
                text=result.source_text,
                topic_label=label.name,
                topic_info=info or None,
                topic_score=round(score, 4),
                latency_ms=round(result.latency_ms, 1),
                queue_ms=round(result.queue_ms, 1),
                t_ms=self._since(result.wall_start),
            )
        self._emit_keyword_final(stream, result, label, info, score)

        if (
            result.key not in self._streams
            and result.key not in self._pipeline_jobs_count
            and not self._keyword_pending(result.key)
        ):
            self._stream_seq.pop(result.key, None)
        self._maybe_finish_rpc(rpc)

    def _keyword_pending(self, key: StreamKey) -> bool:
        return any(job_key[0] == key for job_key in self._keyword_jobs)

    async def _submit_translation_pipeline(
        self,
        stream: StreamContext,
        utt: UtteranceContext,
        source_text: str,
        asr_confidence: float,
        *,
        segment_idx: int = 0,
        ts_start: int | None = None,
        ts_end: int | None = None,
        utt_total: int | None = None,
        asr_created: float | None = None,
        first_seen_ms: float | None = None,
    ) -> None:
        """Translate + synthesize one segment (the whole utterance when segmentation is off).

        utt_total is the number of segments of the utterance when this is its last one.
        asr_created is RuntimeResult.created of the ASR result that triggered this submit.
        first_seen_ms is the utterance-relative time the committed text first appeared in a partial.
        """
        if self.translation_runtime is None or self.tts_runtime is None:
            raise FeatureUnavailableError("translation/TTS runtimes are unavailable")
        rpc = self._rpcs.get(stream.key.rpc_id)
        if rpc is None or rpc.cancelled:
            return
        if rpc.pending_postprocess >= self.settings.max_postprocess_pending_per_rpc:
            await self.fail_rpc(
                rpc.rpc_id,
                "RESOURCE_EXHAUSTED",
                "too many pending translation/TTS utterances",
            )
            return

        segmented = self.settings.segmentation_mode == "on"
        ts_start = utt.ts_start if ts_start is None else ts_start
        ts_end = utt.ts_end if ts_end is None else ts_end
        job_key: JobKey = (stream.key, utt.utterance_id, segment_idx)
        state = PipelineJobState(
            key=stream.key,
            utterance_id=utt.utterance_id,
            source_text=source_text,
            source_language=utt.source_language,
            target_language=utt.target_language,
            ts_start=ts_start,
            ts_end=ts_end,
            asr_confidence=_clamp_confidence(asr_confidence),
            wall_start=utt.wall_start,
            commit_wall=time.monotonic() if segmented else 0.0,
            speech_lead_ms=utt.speech_lead_ms,
            vad_end_wall=utt.vad_end_wall,
            speech_end_wall=utt.speech_end_wall,
            segment_idx=segment_idx,
        )
        self._pipeline_job_add(job_key, state)
        order = self._ensure_pipeline_order(stream, utt.utterance_id)
        if segmented:
            if utt_total is None:
                order.open_utts.add(utt.utterance_id)
            else:
                order.set_total(utt.utterance_id, utt_total)

        logger.debug(
            "PIPELINE asr→translation: utterance=%s src_lang=%s tgt_lang=%s text=%r",
            utt.utterance_id,
            utt.source_language,
            utt.target_language,
            source_text,
            extra={"session_id": stream.key.session_id, "stream_id": stream.key.stream_id},
        )

        try:
            status = self.translation_runtime.try_submit_nowait(
                TranslationJob(
                    key=stream.key,
                    utterance_id=utt.utterance_id,
                    source_text=source_text,
                    source_language=utt.source_language,
                    target_language=utt.target_language,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    segment_idx=segment_idx,
                )
            )
        except Exception:
            logger.exception("Translation try_submit_nowait raised unexpectedly; degrading")
            status = SubmitStatus.OVERLOADED

        if status is SubmitStatus.ACCEPTED:
            rpc.pending_postprocess += 1
            PIPELINE_PENDING.inc()
            TRANSLATION_JOBS.inc()
            state.submit_wall = time.monotonic()
            if first_seen_ms is not None and utt.wall_start > 0:
                state.asr_ms = round(
                    (state.submit_wall - utt.wall_start) * 1000.0 - first_seen_ms, 1
                )
            self._trace(
                "TRANSLATE_SUBMIT",
                stream.key,
                utt.utterance_id,
                segment_idx=segment_idx,
                source_text=source_text,
                source_language=utt.source_language,
                target_language=utt.target_language,
                pending_postprocess=rpc.pending_postprocess,
                queue_depth=self._queue_depth(self.translation_runtime),
                t_ms=self._since(utt.wall_start),
                asr_to_submit_ms=(
                    None
                    if asr_created is None
                    else round((time.perf_counter() - asr_created) * 1000.0, 1)
                ),
                seen_to_submit_ms=(
                    None
                    if first_seen_ms is None or utt.wall_start <= 0
                    else round((time.monotonic() - utt.wall_start) * 1000.0 - first_seen_ms, 1)
                ),
            )
            return

        # Graceful degradation: queue text-only result for in-order emission.
        PIPELINE_DEGRADED.inc()
        self._trace(
            "DEGRADE",
            stream.key,
            utt.utterance_id,
            level=logging.WARNING,
            reason="translate_queue_full",
            segment_idx=segment_idx,
            text=source_text,
            status=status.name,
            queue_depth=self._queue_depth(self.translation_runtime),
        )
        self._pipeline_job_remove(job_key)
        order = self._pipeline_order.get(stream.key)
        if order is not None:
            slot = self._slot_for(stream.key, utt.utterance_id, segment_idx)
            order.fallback_ready[slot] = _FallbackEmit(
                utterance_id=utt.utterance_id,
                text=source_text,
                language=utt.source_language,
                ts_start=ts_start,
                ts_end=ts_end,
                confidence=asr_confidence,
                wall_start=utt.wall_start,
                commit_wall=state.commit_wall,
                segment_idx=segment_idx,
            )
            if slot == order.next_tts_slot:
                order.advance_tts()
                await self._drain_tts_pending(order, stream.key, rpc)
            else:
                # Previous TTS in-flight; mark this slot to be skipped when drain reaches it.
                order.tts_skip.add(slot)
            await self._drain_pipeline_ready(stream.key)
        self._pipeline_order_cleanup(stream.key)
        self._maybe_finish_rpc(rpc)

    async def _handle_translation_result(self, result: TranslationResult) -> None:
        job_key: JobKey = (result.key, result.utterance_id, result.segment_idx)
        job = self._pipeline_jobs.get(job_key)
        rpc = self._rpcs.get(result.key.rpc_id)
        if job is None or rpc is None or rpc.cancelled:
            STALE_RESULTS.inc()
            self._trace_stale(
                "translate",
                result.key,
                result.utterance_id,
                self._stale_reason(job, rpc),
                result.segment_idx,
            )
            return

        self._trace(
            "TRANSLATE",
            result.key,
            result.utterance_id,
            segment_idx=result.segment_idx,
            source_text=result.source_text,
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            latency_ms=round(result.latency_ms, 1),
            error=result.error or None,
            t_ms=self._since(job.wall_start),
        )

        if result.error:
            TRANSLATION_ERRORS.inc()
            self._trace(
                "DEGRADE",
                result.key,
                result.utterance_id,
                level=logging.WARNING,
                reason="translate_error",
                segment_idx=result.segment_idx,
                text=job.source_text,
                error=result.error,
            )
            # Graceful degradation: queue source text for in-order emission rather than failing the RPC.
            self._pipeline_job_remove(job_key)
            rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
            PIPELINE_PENDING.dec()
            PIPELINE_DEGRADED.inc()
            order = self._pipeline_order.get(result.key)
            if order is not None:
                slot = self._slot_for(result.key, result.utterance_id, result.segment_idx)
                order.fallback_ready[slot] = _FallbackEmit(
                    utterance_id=result.utterance_id,
                    text=job.source_text,
                    language=job.source_language,
                    ts_start=job.ts_start,
                    ts_end=job.ts_end,
                    confidence=job.asr_confidence,
                    wall_start=job.wall_start,
                    commit_wall=job.commit_wall,
                    segment_idx=result.segment_idx,
                )
                if slot == order.next_tts_slot:
                    order.advance_tts()
                    await self._drain_tts_pending(order, result.key, rpc)
                else:
                    order.tts_skip.add(slot)
                await self._drain_pipeline_ready(result.key)
            self._pipeline_order_cleanup(result.key)
            self._maybe_finish_rpc(rpc)
            return

        job.translated_text = result.translated_text
        job.translate_queue_ms = result.queue_ms
        job.translate_ms = result.latency_ms
        job.translate_done_wall = time.monotonic()

        logger.debug(
            "PIPELINE translation→tts: utterance=%s src=%r translated=%r",
            result.utterance_id,
            result.source_text,
            result.translated_text,
            extra={"session_id": result.key.session_id, "stream_id": result.key.stream_id},
        )

        # TTS serialization: only submit TTS when it's this utterance's turn.
        order = self._pipeline_order.get(result.key)
        slot = self._slot_for(result.key, result.utterance_id, result.segment_idx)
        if order is None or slot == order.next_tts_slot:
            submitted = await self._try_submit_tts_for_order(result, job, rpc)
            if not submitted and order is not None:
                # Degraded: advance slot and submit next pending.
                order.advance_tts()
                await self._drain_tts_pending(order, result.key, rpc)
                await self._drain_pipeline_ready(result.key)
                self._pipeline_order_cleanup(result.key)
        else:
            # Previous TTS still in flight; hold until it completes.
            order.translation_pending[slot] = result
            self._trace(
                "ORDER_HOLD",
                result.key,
                result.utterance_id,
                segment_idx=result.segment_idx,
                waiting_for=order.next_tts_slot[0],
                waiting_segment=order.next_tts_slot[1],
                held_count=len(order.translation_pending),
            )

    async def _try_submit_tts_for_order(
        self,
        result: TranslationResult,
        job: PipelineJobState,
        rpc: RpcContext,
    ) -> bool:
        """Submit the first TTS phrase for result. Returns True if accepted, False if degraded.

        Splits the translated text into phrases and submits phrase 0.  Subsequent
        phrases are chained in _handle_tts_result as each phrase completes.

        Degraded path queues a text-only result into order.fallback_ready and removes
        the pipeline job.  The caller is responsible for advancing next_tts_slot
        and calling _drain_pipeline_ready when False is returned.
        """
        if self.tts_runtime is None:
            raise RuntimeError("BUG: tts_runtime is None in _try_submit_tts_for_order")

        translated_text = result.translated_text
        phrases = self._split_phrases(translated_text)
        logger.debug(
            "PIPELINE tts submit: utterance=%s phrases=%d text=%r",
            result.utterance_id,
            len(phrases),
            phrases,
            extra={"session_id": result.key.session_id, "stream_id": result.key.stream_id},
        )
        if not phrases:
            # Model returned an empty translation (e.g. only prefix, no content).
            # Degrade: emit source text as text-only rather than submitting a
            # silent TTS job.
            PIPELINE_DEGRADED.inc()
            self._trace(
                "DEGRADE",
                result.key,
                result.utterance_id,
                level=logging.WARNING,
                reason="translate_empty",
                segment_idx=result.segment_idx,
                text=result.source_text,
                translated_text=translated_text,
            )
            job_key: JobKey = (result.key, result.utterance_id, result.segment_idx)
            self._pipeline_job_remove(job_key)
            rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
            PIPELINE_PENDING.dec()
            order = self._pipeline_order.get(result.key)
            if order is not None:
                order.fallback_ready[
                    self._slot_for(result.key, result.utterance_id, result.segment_idx)
                ] = _FallbackEmit(
                    utterance_id=result.utterance_id,
                    text=result.source_text,
                    language=result.source_language,
                    ts_start=result.ts_start,
                    ts_end=result.ts_end,
                    confidence=job.asr_confidence,
                    wall_start=job.wall_start,
                    commit_wall=job.commit_wall,
                    segment_idx=result.segment_idx,
                )
            return False

        is_last = len(phrases) == 1

        status = self.tts_runtime.try_submit_nowait(
            TtsJob(
                key=result.key,
                utterance_id=result.utterance_id,
                text=phrases[0],
                target_language=result.target_language,
                ts_start=result.ts_start,
                ts_end=result.ts_end,
                audio_pt=self.settings.tts_output_pt,
                phrase_id=0,
                is_last_phrase=is_last,
                translated_text=translated_text,
                segment_idx=result.segment_idx,
            )
        )
        if status is SubmitStatus.ACCEPTED:
            job.phrases = phrases
            job.next_phrase_id = 1
            job.tts_submit_wall = time.monotonic()
            TTS_JOBS.inc()
            self._trace(
                "TTS_SUBMIT",
                result.key,
                result.utterance_id,
                segment_idx=result.segment_idx,
                phrases=len(phrases),
                phrase_texts=phrases,
                text=phrases[0],
                target_language=result.target_language,
                queue_depth=self._queue_depth(self.tts_runtime),
                phrase_id=0,
                t_ms=self._since(job.wall_start),
            )
            return True

        # TTS queue full/stopped: queue text-only translated result for in-order emission.
        PIPELINE_DEGRADED.inc()
        TTS_ERRORS.inc()
        self._trace(
            "DEGRADE",
            result.key,
            result.utterance_id,
            level=logging.WARNING,
            reason="tts_queue_full",
            segment_idx=result.segment_idx,
            text=translated_text,
            status=status.name,
            queue_depth=self._queue_depth(self.tts_runtime),
        )
        full_job_key: JobKey = (result.key, result.utterance_id, result.segment_idx)
        self._pipeline_job_remove(full_job_key)
        rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
        PIPELINE_PENDING.dec()
        order = self._pipeline_order.get(result.key)
        if order is not None:
            order.fallback_ready[
                self._slot_for(result.key, result.utterance_id, result.segment_idx)
            ] = _FallbackEmit(
                utterance_id=result.utterance_id,
                text=translated_text,
                language=result.target_language,
                ts_start=result.ts_start,
                ts_end=result.ts_end,
                confidence=job.asr_confidence,
                wall_start=job.wall_start,
                commit_wall=job.commit_wall,
                segment_idx=result.segment_idx,
            )
        elif translated_text:
            # Defensive: order should always exist; emit directly only if no ordering context.
            self._emit_recognition(
                key=result.key,
                utterance_id=result.utterance_id,
                text=translated_text,
                is_final=True,
                ts_start=result.ts_start,
                ts_end=result.ts_end,
                language=result.target_language,
                confidence=job.asr_confidence,
                wall_start=job.wall_start,
                commit_wall=job.commit_wall,
                segment_idx=result.segment_idx,
            )
            self._maybe_finish_rpc(rpc)
        return False

    async def _drain_tts_pending(
        self, order: PipelineOrderState, key: StreamKey, rpc: RpcContext
    ) -> None:
        """After a TTS slot opens, submit the next held translation result if available."""
        while True:
            slot = order.next_tts_slot
            if slot in order.tts_skip:
                order.tts_skip.discard(slot)
                order.advance_tts()
                continue
            if slot not in order.translation_pending:
                break
            result = order.translation_pending.pop(slot)
            job = self._pipeline_jobs.get((key, result.utterance_id, result.segment_idx))
            if job is None:
                order.advance_tts()
                continue
            submitted = await self._try_submit_tts_for_order(result, job, rpc)
            if submitted:
                break
            # Degraded: try the next utterance.
            order.advance_tts()

    def _ensure_pipeline_order(self, stream: StreamContext, utterance_id: int) -> PipelineOrderState:
        """The stream's playback order, created at the start of a translate utterance (or here if missing).

        It starts at the lowest unresolved utterance: with several ASR workers a later utterance can reach FINAL
        before an earlier one that is still finishing, and that earlier one must not be stranded behind it.
        Every utterance records its end in it (translation, empty ASR, passthrough), so a cursor never waits on
        an utterance whose end was not recorded.
        """
        order = self._pipeline_order.get(stream.key)
        if order is None:
            first_uid = min([utterance_id, *self._finishing.get(stream.key, ())])
            order = self._pipeline_order[stream.key] = PipelineOrderState(
                next_emit_slot=self._slot_for(stream.key, first_uid),
                next_tts_slot=self._slot_for(stream.key, first_uid),
                segmented=self.settings.segmentation_mode == "on",
            )
        return order

    def _pipeline_order_cleanup(self, key: StreamKey) -> None:
        """Remove PipelineOrderState when all pipeline work for the stream is done and no utterance of the
        stream is still in ASR (active or finishing): those will record their end in it."""
        order = self._pipeline_order.get(key)
        if order is None:
            return
        stream = self._streams.get(key)
        if stream is not None and (stream.active_utterance is not None or self._finishing.get(key)):
            return
        if (
            not order.ready
            and not order.fallback_ready
            and not order.translation_pending
            and not order.tts_skip
            and not order.open_utts
            and key not in self._pipeline_jobs_count
        ):
            self._pipeline_order.pop(key, None)
            # _stream_seq persists while the stream is alive so future utterances
            # continue from the current seq value.  If the stream already closed
            # while these pipeline jobs were in flight, _close_stream could not
            # clean it up (pipeline_jobs_count was non-zero then) — do it now.
            if key not in self._streams:
                self._stream_seq.pop(key, None)

    def _try_submit_tts_continuation(self, tts_job: TtsJob) -> SubmitStatus:
        try:
            if self.tts_runtime is None:
                raise RuntimeError("BUG: tts_runtime is None in TTS continuation")
            return self.tts_runtime.try_submit_nowait(tts_job)
        except Exception:
            logger.exception(
                "TTS next phrase submit raised unexpectedly",
                extra={
                    "rpc_id": tts_job.key.rpc_id,
                    "utterance_id": tts_job.utterance_id,
                    "phrase_id": tts_job.phrase_id,
                },
            )
            return SubmitStatus.UNAVAILABLE

    async def _retry_tts_continuation(self, tts_job: TtsJob) -> None:
        """Retry a continuation phrase rejected by a full TTS queue, then degrade on timeout."""
        job_key: JobKey = (tts_job.key, tts_job.utterance_id, tts_job.segment_idx)
        deadline = time.monotonic() + self.settings.tts_continuation_retry_ms / 1000.0
        delay = 0.05
        try:
            while True:
                await asyncio.sleep(delay)
                rpc = self._rpcs.get(tts_job.key.rpc_id)
                if job_key not in self._pipeline_jobs or rpc is None or rpc.cancelled:
                    return
                status = self._try_submit_tts_continuation(tts_job)
                if status is SubmitStatus.ACCEPTED:
                    TTS_JOBS.inc()
                    self._trace_tts_continuation_submit(tts_job, self._pipeline_jobs.get(job_key))
                    return
                if status is not SubmitStatus.OVERLOADED or time.monotonic() >= deadline:
                    await self._degrade_tts_continuation(tts_job)
                    return
                delay = min(delay * 2, 0.25)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("TTS continuation retry failed")

    async def _degrade_tts_continuation(self, tts_job: TtsJob) -> None:
        """Give up on the remaining phrases: emit the full translated text without further audio."""
        job_key: JobKey = (tts_job.key, tts_job.utterance_id, tts_job.segment_idx)
        job = self._pipeline_jobs.get(job_key)
        rpc = self._rpcs.get(tts_job.key.rpc_id)
        if job is None or rpc is None or rpc.cancelled:
            return
        logger.warning(
            "TTS next phrase could not be submitted; degrading utterance",
            extra={
                "rpc_id": tts_job.key.rpc_id,
                "session_id": tts_job.key.session_id,
                "stream_id": tts_job.key.stream_id,
                "utterance_id": tts_job.utterance_id,
                "phrase_id": tts_job.phrase_id,
            },
        )
        self._trace(
            "DEGRADE",
            tts_job.key,
            tts_job.utterance_id,
            level=logging.WARNING,
            reason="tts_continuation_queue_full",
            segment_idx=tts_job.segment_idx,
            phrase_id=tts_job.phrase_id,
            text=tts_job.translated_text,
        )
        TTS_ERRORS.inc()
        PIPELINE_DEGRADED.inc()
        self._pipeline_job_remove(job_key)
        rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
        PIPELINE_PENDING.dec()
        order = self._pipeline_order.get(tts_job.key)
        if order is not None:
            order.fallback_ready[
                self._slot_for(tts_job.key, tts_job.utterance_id, tts_job.segment_idx)
            ] = _FallbackEmit(
                utterance_id=tts_job.utterance_id,
                text=tts_job.translated_text,
                language=tts_job.target_language,
                ts_start=tts_job.ts_start,
                ts_end=tts_job.ts_end,
                confidence=job.asr_confidence,
                wall_start=job.wall_start,
                commit_wall=job.commit_wall,
                segment_idx=tts_job.segment_idx,
            )
            order.advance_tts()
            await self._drain_tts_pending(order, tts_job.key, rpc)
            await self._drain_pipeline_ready(tts_job.key)
            self._pipeline_order_cleanup(tts_job.key)
        self._maybe_finish_rpc(rpc)

    async def _handle_tts_result(self, result: TtsResult) -> None:
        job_key: JobKey = (result.key, result.utterance_id, result.segment_idx)
        job = self._pipeline_jobs.get(job_key)
        rpc = self._rpcs.get(result.key.rpc_id)
        if job is None or rpc is None or rpc.cancelled:
            STALE_RESULTS.inc()
            self._trace_stale(
                "tts",
                result.key,
                result.utterance_id,
                self._stale_reason(job, rpc),
                result.segment_idx,
            )
            return
        self._trace(
            "TTS_PHRASE",
            result.key,
            result.utterance_id,
            segment_idx=result.segment_idx,
            phrase_id=result.phrase_id,
            is_last_phrase=result.is_last_phrase,
            text=result.text,
            target_language=result.target_language,
            audio_bytes=len(result.audio_payload),
            latency_ms=round(result.latency_ms, 1),
            error=result.error or None,
            t_ms=self._since(job.wall_start),
        )

        if result.error:
            TTS_ERRORS.inc()
            PIPELINE_DEGRADED.inc()
            self._trace(
                "DEGRADE",
                result.key,
                result.utterance_id,
                level=logging.WARNING,
                reason="tts_error",
                segment_idx=result.segment_idx,
                phrase_id=result.phrase_id,
                text=job.translated_text,
                error=result.error,
            )
            # Graceful degradation: queue text-only translated result for in-order emission.
            self._pipeline_job_remove(job_key)
            rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
            PIPELINE_PENDING.dec()
            order = self._pipeline_order.get(result.key)
            if order is not None:
                order.fallback_ready[
                    self._slot_for(result.key, result.utterance_id, result.segment_idx)
                ] = _FallbackEmit(
                    utterance_id=result.utterance_id,
                    text=job.translated_text,
                    language=result.target_language,
                    ts_start=result.ts_start,
                    ts_end=result.ts_end,
                    confidence=job.asr_confidence,
                    wall_start=job.wall_start,
                    commit_wall=job.commit_wall,
                    segment_idx=result.segment_idx,
                )
                order.advance_tts()
                await self._drain_tts_pending(order, result.key, rpc)
                await self._drain_pipeline_ready(result.key)
                self._pipeline_order_cleanup(result.key)
            self._maybe_finish_rpc(rpc)
            return

        if result.phrase_id == 0:
            job.tts_queue_ms = result.queue_ms
            job.tts_ms = result.latency_ms
            job.tts_done_wall = time.monotonic()

        if not result.audio_payload:
            self._trace(
                "DEGRADE",
                result.key,
                result.utterance_id,
                level=logging.WARNING,
                reason="tts_empty_audio",
                segment_idx=result.segment_idx,
                phrase_id=result.phrase_id,
                text=result.text,
            )

        # Intermediate phrase: emit audio immediately, then submit the next phrase.
        if not result.is_last_phrase:
            if result.audio_payload:
                self._observe_first_audio(job)
                self._observe_segment_latency(job)
                self._emit_recognition(
                    key=result.key,
                    utterance_id=result.utterance_id,
                    text="",
                    is_final=False,
                    ts_start=result.ts_start,
                    ts_end=result.ts_end,
                    language=result.target_language,
                    confidence=job.asr_confidence,
                    audio_payload=result.audio_payload,
                    audio_pt=result.audio_pt,
                    wall_start=job.wall_start,
                    commit_wall=job.commit_wall,
                    segment_idx=result.segment_idx,
                    speech_end_wall=job.speech_end_wall,
                )
            next_phrase_id = result.phrase_id + 1
            phrases = job.phrases
            if next_phrase_id < len(phrases):
                next_job = TtsJob(
                    key=result.key,
                    utterance_id=result.utterance_id,
                    text=phrases[next_phrase_id],
                    target_language=result.target_language,
                    ts_start=result.ts_start,
                    ts_end=result.ts_end,
                    audio_pt=result.audio_pt,
                    phrase_id=next_phrase_id,
                    is_last_phrase=next_phrase_id == len(phrases) - 1,
                    translated_text=result.translated_text,
                    segment_idx=result.segment_idx,
                )
                status = self._try_submit_tts_continuation(next_job)
                if status is SubmitStatus.ACCEPTED:
                    TTS_JOBS.inc()
                    self._trace_tts_continuation_submit(next_job, job)
                elif (
                    status is SubmitStatus.OVERLOADED
                    and self.settings.tts_continuation_retry_ms > 0
                ):
                    task = asyncio.create_task(
                        self._retry_tts_continuation(next_job),
                        name=f"tts-retry-{result.utterance_id}-{next_phrase_id}",
                    )
                    self._tts_retry_tasks.add(task)
                    task.add_done_callback(self._tts_retry_tasks.discard)
                else:
                    await self._degrade_tts_continuation(next_job)
            return

        # Last phrase: use in-order emission via order.ready.
        order = self._pipeline_order.get(result.key)
        if order is None:
            logger.error(
                "BUG: no pipeline order state for last TTS phrase; dropping utterance=%s",
                result.utterance_id,
                extra={"rpc_id": result.key.rpc_id, "utterance_id": result.utterance_id},
            )
            return
        order.ready[self._slot_for(result.key, result.utterance_id, result.segment_idx)] = result
        await self._drain_pipeline_ready(result.key)

        # TTS-N complete: open the slot for TTS-N+1.
        # Re-fetch both order and rpc: the first _drain_pipeline_ready above may have
        # called _pipeline_order_cleanup (order removed) or _maybe_finish_rpc (rpc removed).
        # Only proceed if both still exist and are not cancelled.
        order = self._pipeline_order.get(result.key)
        rpc = self._rpcs.get(result.key.rpc_id)
        if order is not None and rpc is not None and not rpc.cancelled:
            order.advance_tts()
            # Submit the next held translation result, if any.  If it cannot be submitted
            # (TTS queue full) it lands in order.fallback_ready instead of order.ready.
            await self._drain_tts_pending(order, result.key, rpc)
            # Second drain: picks up any fallback_ready entries that _drain_tts_pending
            # just added when a degraded submission bypassed the normal TTS path.
            await self._drain_pipeline_ready(result.key)
            self._pipeline_order_cleanup(result.key)

    async def _drain_pipeline_ready(self, key: StreamKey) -> None:
        order = self._pipeline_order.get(key)
        if order is None:
            return
        rpc = self._rpcs.get(key.rpc_id)
        if rpc is None or rpc.cancelled:
            return

        while True:
            slot = order.next_emit_slot
            if slot in order.ready:
                result = order.ready.pop(slot)
                job = self._pipeline_job_remove((key, result.utterance_id, result.segment_idx))
                if job is None:
                    order.advance_emit()
                    continue
                if result.audio_payload:
                    self._observe_first_audio(job, order)
                    self._observe_segment_latency(job)
                self._emit_recognition(
                    key=key,
                    utterance_id=result.utterance_id,
                    text=result.translated_text or result.text,
                    is_final=True,
                    ts_start=result.ts_start,
                    ts_end=result.ts_end,
                    language=result.target_language,
                    confidence=job.asr_confidence,
                    audio_payload=result.audio_payload,
                    audio_pt=result.audio_pt,
                    wall_start=job.wall_start,
                    commit_wall=job.commit_wall,
                    segment_idx=result.segment_idx,
                    speech_end_wall=job.speech_end_wall,
                )
                PIPELINE_RESULTS.inc()
                if order.segmented:
                    SEGMENT_EMITTED.inc()
                PIPELINE_PENDING.dec()
                rpc.pending_postprocess = max(0, rpc.pending_postprocess - 1)
                order.advance_emit()
            elif slot in order.fallback_ready:
                fb = order.fallback_ready.pop(slot)
                if fb.text or fb.audio_payload:
                    self._emit_recognition(
                        key=key,
                        utterance_id=fb.utterance_id,
                        text=fb.text,
                        is_final=True,
                        ts_start=fb.ts_start,
                        ts_end=fb.ts_end,
                        language=fb.language,
                        confidence=fb.confidence,
                        wall_start=fb.wall_start,
                        commit_wall=fb.commit_wall,
                        segment_idx=fb.segment_idx,
                        audio_payload=fb.audio_payload,
                        audio_pt=fb.audio_pt,
                        speech_end_wall=fb.speech_end_wall,
                    )
                    if order.segmented:
                        SEGMENT_EMITTED.inc()
                order.advance_emit()
            else:
                break

        self._pipeline_order_cleanup(key)
        self._maybe_finish_rpc(rpc)

    def _emit_recognition(
        self,
        *,
        key: StreamKey,
        utterance_id: int,
        text: str,
        is_final: bool,
        ts_start: int,
        ts_end: int,
        language: str,
        confidence: float = 0.0,
        audio_payload: bytes = b"",
        audio_pt: int = 0,
        topic_label: int = 0,
        topic_info: str = "",
        topic_score: float = 0.0,
        wall_start: float = 0.0,
        commit_wall: float = 0.0,
        segment_idx: int | None = None,
        speech_end_wall: float = 0.0,
    ) -> None:
        rpc = self._rpcs.get(key.rpc_id)
        if rpc is None or rpc.cancelled:
            return
        chunk_bytes = self.settings.tts_max_audio_chunk_bytes
        if chunk_bytes > 0 and len(audio_payload) > chunk_bytes:
            # The gateway drops a single audio_payload above a size limit (observed between
            # 56935 and 66873 bytes) without an error, so large audio is sent as several results.
            offsets = range(0, len(audio_payload), chunk_bytes)
            last_offset = offsets[-1]
            for offset in offsets:
                is_last = offset == last_offset
                self._emit_recognition(
                    key=key,
                    utterance_id=utterance_id,
                    text=text if is_last else "",
                    is_final=is_final if is_last else False,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    language=language,
                    confidence=confidence,
                    audio_payload=audio_payload[offset : offset + chunk_bytes],
                    audio_pt=audio_pt,
                    topic_label=topic_label if is_last else 0,
                    topic_info=topic_info if is_last else "",
                    topic_score=topic_score if is_last else 0.0,
                    wall_start=wall_start,
                    commit_wall=commit_wall,
                    segment_idx=segment_idx,
                    speech_end_wall=speech_end_wall,
                )
            return
        e2e_start = commit_wall if commit_wall > 0 else wall_start
        if is_final and e2e_start > 0:
            UTTERANCE_E2E_MS.observe((time.monotonic() - e2e_start) * 1000.0)
        if not text and not audio_payload:
            EMPTY_RESULTS_SKIPPED.inc()
            self._trace(
                "EMIT",
                key,
                utterance_id,
                segment_idx=segment_idx,
                is_final=is_final,
                language=language,
                audio_bytes=0,
                skipped="empty",
                t_ms=self._since(wall_start),
            )
            return
        self._stream_seq[key] = self._stream_seq.get(key, 0) + 1
        preproc_delay_s = 0.0
        if audio_payload and speech_end_wall > 0:
            stream = self._streams.get(key)
            if stream is not None and stream.preprocessor is not None:
                preproc_delay_s = stream.preprocessor.delay_samples / 16_000
        envelope = RecognitionEnvelope(
            session_id=key.session_id,
            stream_id=key.stream_id,
            text=text,
            is_final=is_final,
            ts_start=ts_start,
            ts_end=ts_end,
            confidence=_clamp_confidence(confidence),
            language=language,
            seq=self._stream_seq[key],
            utterance_id=utterance_id,
            audio_payload=audio_payload,
            audio_pt=audio_pt,
            topic_label=int(topic_label),
            topic_info=topic_info,
            topic_score=float(topic_score),
            speech_end_wall=speech_end_wall if audio_payload else 0.0,
            preproc_delay_s=preproc_delay_s,
        )
        queued = rpc.response_buffer.put(envelope)
        if is_final or audio_payload:
            self._trace(
                "EMIT",
                key,
                utterance_id,
                segment_idx=segment_idx,
                seq=envelope.seq,
                is_final=is_final,
                text=text,
                language=language,
                audio_bytes=len(audio_payload),
                confidence=envelope.confidence,
                queued=queued,
                t_ms=self._since(wall_start),
            )
        if queued:
            if is_final:
                FINAL_RESULTS.inc()
            else:
                PARTIAL_RESULTS.inc()

    async def _close_stream(self, stream: StreamContext) -> None:
        if stream.phase is StreamPhase.CLOSED:
            return
        stream.phase = StreamPhase.CLOSED
        self._detach_preprocessor(stream)
        self._close_pcap_trace(stream)
        self._streams.pop(stream.key, None)
        self._finishing.pop(stream.key, None)
        rpc = self._rpcs.get(stream.key.rpc_id)
        if rpc:
            rpc.stream_keys.discard(stream.key)
        ACTIVE_STREAMS.dec()
        logger.info(
            "Logical stream closed",
            extra={
                "rpc_id": stream.key.rpc_id,
                "session_id": stream.key.session_id,
                "stream_id": stream.key.stream_id,
                **self._stage_extra("STREAM_CLOSE"),
            },
        )
        self._pipeline_order_cleanup(stream.key)
        if stream.key not in self._pipeline_jobs_count and not self._keyword_pending(stream.key):
            self._stream_seq.pop(stream.key, None)
        if rpc:
            self._maybe_finish_rpc(rpc)

    def _maybe_finish_rpc(self, rpc: RpcContext) -> None:
        if rpc.finished:
            return
        if rpc.input_closed and not rpc.stream_keys and rpc.pending_postprocess == 0:
            rpc.finished = True
            rpc.response_buffer.close()
            self._remove_rpc(rpc)

    def _remove_rpc(self, rpc: RpcContext) -> None:
        if self._rpcs.pop(rpc.rpc_id, None) is not None:
            ACTIVE_RPCS.dec()
            logger.info("RPC closed", extra={"rpc_id": rpc.rpc_id, **self._stage_extra("GRPC_CLOSE")})

    def _require_rpc(self, rpc_id: str) -> RpcContext:
        rpc = self._rpcs.get(rpc_id)
        if rpc is None:
            raise RuntimeError(f"unknown RPC: {rpc_id}")
        return rpc

    @staticmethod
    def _split_phrases(text: str) -> list[str]:
        """Split translated text into phrases at punctuation boundaries for incremental TTS.

        Splits unconditionally after ! and ?, and after . only when the first
        character of the next word is a Unicode uppercase letter (category Lu).
        This avoids splitting decimal numbers ("1.5 GHz") while correctly
        handling all languages without maintaining a hardcoded character list.
        """
        stripped = text.strip()
        if not stripped:
            return []
        parts: list[str] = []
        last = 0
        for m in _PHRASE_BOUNDARY_RE.finditer(stripped):
            end = m.end()
            if stripped[m.start() - 1] == '.' and (
                end >= len(stripped) or unicodedata.category(stripped[end]) != 'Lu'
            ):
                continue
            parts.append(stripped[last:m.start()])
            last = end
        parts.append(stripped[last:])
        parts = [p for p in parts if p.strip()]
        return parts if parts else [stripped]

    @staticmethod
    def _rtp_add(timestamp: int, delta: int) -> int:
        return (int(timestamp) + int(delta)) & 0xFFFFFFFF

    @staticmethod
    def _rtp_diff(a: int, b: int) -> int:
        """Signed RTP timestamp difference a - b, wrap-around aware.

        Positive → a is ahead of b (forward gap).
        Negative → a is behind b (out-of-order / duplicate).
        """
        raw = (int(a) - int(b)) & 0xFFFFFFFF
        return raw if raw < 0x80000000 else raw - 0x100000000

    def _apply_rtp_policy(
        self,
        *,
        stream: StreamContext,
        audio: np.ndarray,
        rtp_timestamp: int,
        source_samples: int,
        source_rate: int,
        duration_ms: float,
    ) -> tuple[np.ndarray, int, int, float] | None:
        """Apply RTP continuity policy.

        Returns (audio, rtp_timestamp, source_samples, duration_ms) — possibly
        with silence prepended for gap repair — or None if the chunk should be
        dropped.

        RTP timestamp advances by rtp_clock_rate ticks per second, which may
        differ from source_rate (e.g. G.711 uses 8000 Hz RTP clock with 16000
        Hz PCM). All expected_next calculations use duration_ms × rtp_clock_rate
        to avoid misclassifying valid chunks as out-of-order.
        """
        expected = stream.expected_rtp_timestamp
        rtp_clock_rate = stream.rtp_clock_rate
        # Advance RTP timestamp by clock-rate ticks, not by PCM sample count.
        rtp_duration_ticks = int(round(duration_ms * rtp_clock_rate / 1000))
        actual_end = self._rtp_add(rtp_timestamp, rtp_duration_ticks)

        if expected is None:
            stream.last_rtp_timestamp = rtp_timestamp
            stream.expected_rtp_timestamp = actual_end
            stream.last_source_rate = source_rate
            return audio, rtp_timestamp, source_samples, duration_ms

        diff = self._rtp_diff(rtp_timestamp, expected)
        if diff == 0:
            stream.last_rtp_timestamp = rtp_timestamp
            stream.expected_rtp_timestamp = actual_end
            stream.last_source_rate = source_rate
            return audio, rtp_timestamp, source_samples, duration_ms

        if diff > 0:
            # Forward gap: diff RTP ticks of audio were not delivered.
            gap_rtp_ticks = diff
            gap_ms = gap_rtp_ticks * 1000.0 / rtp_clock_rate
            RTP_GAPS.inc()
            if self.settings.rtp_gap_policy == "error":
                logger.warning(
                    "RTP timestamp gap detected",
                    extra={
                        "rpc_id": stream.key.rpc_id,
                        "session_id": stream.key.session_id,
                        "stream_id": stream.key.stream_id,
                        "expected_ts": expected,
                        "received_ts": rtp_timestamp,
                        "gap_rtp_ticks": gap_rtp_ticks,
                        "gap_ms": gap_ms,
                    },
                )
                raise DataLossError(f"RTP timestamp gap {gap_ms:.1f} ms")
            will_insert = (
                self.settings.rtp_gap_policy == "insert_silence"
                and gap_ms <= self.settings.rtp_max_gap_ms
            )
            if will_insert:
                # Gap within tolerance → silence inserted (DTX/VAD suppression on sender).
                # Log at DEBUG: this is normal phone silence, not a delivery failure.
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RTP timestamp gap detected (silence inserted)",
                        extra={
                            "rpc_id": stream.key.rpc_id,
                            "session_id": stream.key.session_id,
                            "stream_id": stream.key.stream_id,
                            "expected_ts": expected,
                            "received_ts": rtp_timestamp,
                            "gap_rtp_ticks": gap_rtp_ticks,
                            "gap_ms": gap_ms,
                        },
                    )
                # RTP ticks → ms → PCM samples at internal 16 kHz rate.
                pcm_gap_samples = int(round(gap_ms * 16_000 / 1000))
                if pcm_gap_samples > 0:
                    audio = np.concatenate(
                        [np.zeros(pcm_gap_samples, dtype=np.float32), audio]
                    )
                    duration_ms += gap_ms
                    rtp_timestamp = expected
                    # source_samples counts samples at the input rate, not the 16 kHz of `audio`
                    source_samples += int(round(gap_ms * source_rate / 1000))
                    rtp_duration_ticks += gap_rtp_ticks
                    actual_end = self._rtp_add(rtp_timestamp, rtp_duration_ticks)
                    RTP_GAP_SILENCE_INSERTED.inc()
            else:
                # Gap exceeds max threshold → audio loss, silence not inserted.
                logger.warning(
                    "RTP timestamp gap detected (audio loss, silence not inserted)",
                    extra={
                        "rpc_id": stream.key.rpc_id,
                        "session_id": stream.key.session_id,
                        "stream_id": stream.key.stream_id,
                        "expected_ts": expected,
                        "received_ts": rtp_timestamp,
                        "gap_rtp_ticks": gap_rtp_ticks,
                        "gap_ms": gap_ms,
                        "rtp_max_gap_ms": self.settings.rtp_max_gap_ms,
                    },
                )
            stream.last_rtp_timestamp = rtp_timestamp
            stream.expected_rtp_timestamp = actual_end
            stream.last_source_rate = source_rate
            return audio, rtp_timestamp, source_samples, duration_ms

        # diff < 0: timestamp is behind expected — duplicate or out-of-order.
        jitter_ticks = int(round(self.settings.rtp_jitter_tolerance_ms * rtp_clock_rate / 1000))
        if jitter_ticks > 0 and abs(diff) <= jitter_ticks:
            # Small backwards deviation within jitter tolerance — accept without dropping.
            RTP_JITTER_CORRECTED.inc()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "RTP jitter corrected",
                    extra={
                        "rpc_id": stream.key.rpc_id,
                        "session_id": stream.key.session_id,
                        "stream_id": stream.key.stream_id,
                        "expected_ts": expected,
                        "received_ts": rtp_timestamp,
                        "diff_ticks": diff,
                    },
                )
            stream.last_rtp_timestamp = rtp_timestamp
            self._advance_expected_rtp(stream, actual_end)
            return audio, rtp_timestamp, source_samples, duration_ms

        RTP_OUT_OF_ORDER.inc()
        logger.warning(
            "Out-of-order RTP timestamp detected",
            extra={
                "rpc_id": stream.key.rpc_id,
                "session_id": stream.key.session_id,
                "stream_id": stream.key.stream_id,
                "expected_ts": expected,
                "received_ts": rtp_timestamp,
            },
        )
        if self.settings.rtp_out_of_order_policy == "error":
            raise DataLossError("out-of-order RTP timestamp")
        if self.settings.rtp_out_of_order_policy == "drop":
            return None

        stream.last_rtp_timestamp = rtp_timestamp
        self._advance_expected_rtp(stream, actual_end)
        stream.last_source_rate = source_rate
        return audio, rtp_timestamp, source_samples, duration_ms

    def _advance_expected_rtp(self, stream: StreamContext, chunk_end: int) -> None:
        """A chunk that started behind expected_rtp_timestamp (jitter, or OOO accepted): move expected to the end
        of the audio received so far, i.e. to chunk_end when it is past expected, never backward.

        Kept where it was, the next chunk (contiguous with this one) would look like a forward gap and get
        silence that is not there; moved back, audio already received would be counted as a gap again.
        """
        if self._rtp_diff(chunk_end, stream.expected_rtp_timestamp) > 0:
            stream.expected_rtp_timestamp = chunk_end
