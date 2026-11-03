from __future__ import annotations

import threading
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, start_http_server


class IntervalMax:
    """Maximum of observed values since the last take(); consumed by the stats logger."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: float | None = None

    def observe(self, value: float) -> None:
        with self._lock:
            if self._value is None or value > self._value:
                self._value = value

    def take(self) -> float | None:
        with self._lock:
            value, self._value = self._value, None
            return value


ACTIVE_RPCS = Gauge("asr_active_rpcs", "Active gRPC Recognize calls")
ACTIVE_STREAMS = Gauge("asr_active_streams", "Active logical audio streams")
ACTIVE_UTTERANCES = Gauge("asr_active_utterances", "Active ASR utterances")
RUNTIME_QUEUE = Gauge(
    "asr_runtime_queue_depth",
    "Nemotron runtime command queue depth per worker",
    ["worker"],
)
ASR_POOL_WORKERS = Gauge("asr_pool_workers", "Configured Nemotron runtime workers")
ASR_STREAM_AFFINITIES = Gauge(
    "asr_stream_affinities", "Active stream-to-Nemotron-worker affinity mappings"
)
MODEL_STEP_SECONDS = Histogram(
    "asr_model_step_seconds",
    "Time spent in one serialized Nemotron model command",
    ["command"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
ASR_QUEUE_WAIT_MS = Histogram(
    "asr_queue_wait_ms",
    "Time a command waits in the Nemotron worker queue before execution",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500),
)
ASR_FINALIZE_STEPS = Histogram(
    "asr_finalize_steps",
    "Number of model steps executed during one FINALIZE command",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
AUDIO_SECONDS = Counter("asr_ingress_audio_seconds_total", "Accepted source audio seconds")
INGRESS_CHUNKS = Counter("asr_ingress_chunks_total", "Accepted AudioChunk messages")
INGRESS_PEAK = IntervalMax()
PARTIAL_RESULTS = Counter("asr_partial_results_total", "Partial results emitted")
FINAL_RESULTS = Counter("asr_final_results_total", "Final results emitted")
EMPTY_RESULTS_SKIPPED = Counter(
    "asr_empty_results_skipped_total", "Results with neither text nor audio that were not sent to the client"
)
STALE_RESULTS = Counter("asr_stale_results_total", "Discarded stale runtime results")
STREAM_ERRORS = Counter("asr_stream_errors_total", "Stream processing errors")
DROPPED_PARTIALS = Counter(
    "asr_dropped_partial_results_total", "Partial responses dropped due to a full client queue"
)
RTP_GAPS = Counter("asr_rtp_timestamp_gaps_total", "Detected forward RTP timestamp gaps")
RTP_GAP_SILENCE_INSERTED = Counter(
    "asr_rtp_gap_silence_inserted_total", "Forward RTP gaps repaired by inserting silence"
)
RTP_OUT_OF_ORDER = Counter(
    "asr_rtp_out_of_order_total", "Out-of-order or duplicate RTP timestamp chunks"
)
RTP_JITTER_CORRECTED = Counter(
    "asr_rtp_jitter_corrected_total", "Small OOO chunks accepted within jitter tolerance"
)
DURATION_MISMATCHES = Counter(
    "asr_duration_mismatch_total", "AudioChunk duration_ms mismatches against payload duration"
)
INGRESS_BACKLOG_MS = Histogram(
    "asr_ingress_backlog_ms",
    "Per-stream ASR PCM backlog before Nemotron",
    buckets=(0, 100, 250, 500, 1000, 2000, 5000, 10000, 20000, 60000),
)
DEFERRED_BACKLOG_MS = Histogram(
    "asr_deferred_backlog_ms",
    "Per-stream audio deferred while ASR utterance is finalizing",
    buckets=(0, 100, 250, 500, 1000, 2000, 5000, 10000, 20000, 60000),
)

TRANSLATION_QUEUE = Gauge(
    "translation_queue_depth", "EnViT5 translation queue depth per worker", ["worker"]
)
TRANSLATION_POOL_WORKERS = Gauge(
    "translation_pool_workers", "Configured EnViT5 translation workers"
)
TRANSLATION_LATENCY = Histogram(
    "translation_seconds",
    "EnViT5 final-utterance translation latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
TRANSLATION_JOBS = Counter("translation_jobs_total", "Translation jobs accepted")
TRANSLATION_ERRORS = Counter("translation_errors_total", "Translation job failures")
TRANSLATION_QUEUE_WAIT_MS = Histogram(
    "translation_queue_wait_ms",
    "Time a translation job waits in the worker queue before inference begins",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500),
)

AUDIO_PREPROC_BACKEND = "resemble"  # modes denoiser / enhancer
AUDIO_PREPROC_BACKEND_AUDX = "audx"  # mode audx


def audio_preproc_backend(mode: str) -> str:
    """Value of the `backend` label for an audio_preprocessing mode."""
    return AUDIO_PREPROC_BACKEND_AUDX if mode == "audx" else AUDIO_PREPROC_BACKEND


def audio_preproc_reason(mode: str, what: str) -> str:
    """Fallback reason code: DENOISER_<what> for the resemble-enhance modes (denoiser, enhancer), AUDX_<what> for audx."""
    return f"{'AUDX' if mode == 'audx' else 'DENOISER'}_{what}"


AUDIO_PREPROC_STREAMS_ACTIVE = Gauge(
    "audio_preproc_streams_active", "Streams with an audio preprocessor attached", ["backend"]
)
AUDIO_PREPROC_QUEUE = Gauge(
    "audio_preproc_queue_depth", "Audio preprocessing window queue depth per worker", ["worker"]
)
AUDIO_PREPROC_POOL_WORKERS = Gauge(
    "audio_preproc_pool_workers", "Configured audio preprocessing workers", ["backend"]
)
AUDIO_PREPROC_WINDOW_SECONDS = Histogram(
    "audio_preproc_window_seconds",
    "Worker compute time of one preprocessing window (tune compute_budget_ms to ~1.5x its p99)",
    ["backend"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)
AUDIO_PREPROC_QUEUE_WAIT_SECONDS = Histogram(
    "audio_preproc_queue_wait_seconds",
    "Time a preprocessing window waits in the worker queue before compute begins",
    ["backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
AUDIO_PREPROC_LATE_SECONDS = Histogram(
    "audio_preproc_late_seconds",
    "For a region already played raw (outcome late) whose result still arrived: arrival time minus the "
    "moment the region was needed (how far compute_budget_ms fell short; regions whose result never "
    "arrives are only in audio_preproc_regions_total)",
    ["backend"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
AUDIO_PREPROC_SLACK_SECONDS = Histogram(
    "audio_preproc_slack_seconds",
    "For an enhanced region: the moment it was needed minus its result arrival (compute budget left "
    "unused; the low tail p1 shows how far compute_budget_ms can be cut)",
    ["backend"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1),
)
AUDIO_PREPROC_REGIONS = Counter(
    "audio_preproc_regions_total",
    "Preprocessed regions by outcome: enhanced, silent (all digital zero; resemble modes skip the model), "
    "late, shed, error, bypass (the last four play raw audio)",
    ["backend", "outcome"],
)
AUDIO_PREPROC_FALLBACK = Counter(
    "audio_preproc_fallback_total",
    "Region fallbacks to raw audio by reason (resemble-enhance modes: DENOISER_LATE, DENOISER_QUEUE_FULL, "
    "DENOISER_ERROR; mode audx: AUDX_LATE, AUDX_QUEUE_FULL, AUDX_ERROR)",
    ["backend", "reason"],
)
AUDIO_PREPROC_BYPASS = Counter(
    "audio_preproc_stream_bypass_total",
    "Streams switched to raw audio after consecutive bad regions",
    ["backend"],
)
AUDIO_PREPROC_SHED_STREAMS = Counter(
    "audio_preproc_shed_streams_total",
    "Streams that ran without a preprocessor because audio_preprocessing.max_streams was reached",
    ["backend"],
)
AUDIO_PREPROC_FLUSH_SECONDS = Histogram(
    "audio_preproc_flush_seconds",
    "Time end-of-stream flush waited for outstanding windows",
    ["backend"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
AUDIO_PREPROC_STATE_RESETS = Counter(
    "audio_preproc_state_resets_total",
    "Stateful preprocessor (mode audx): model state of a stream restarted from scratch, by reason "
    "(error: model failed; shed: queue full for longer than the backlog limit; bypass: after a bypass)",
    ["backend", "reason"],
)

# trace_pcap (docs/trace-pcap-design.md)
TRACE_PCAP_STREAMS_ACTIVE = Gauge("trace_pcap_streams_active", "Streams whose audio is being written to a trace pcap")
TRACE_PCAP_STREAMS = Counter(
    "trace_pcap_streams_total",
    "trace_pcap stream outcomes: traced, skipped_max_streams, skipped_disk_low, stopped_queue_full, "
    "stopped_io_error, stopped_disk_low, stopped_max_seconds",
    ["outcome"],
)
TRACE_PCAP_BYTES = Counter("trace_pcap_bytes_total", "Bytes written to trace pcap files")

TTS_QUEUE = Gauge(
    "tts_queue_depth", "TTS queue depth per worker", ["worker"]
)
TTS_POOL_WORKERS = Gauge("tts_pool_workers", "Configured Piper TTS workers")
TTS_LATENCY = Histogram(
    "tts_seconds",
    "Piper synthesis plus G.711 encoding latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
TTS_JOBS = Counter("tts_jobs_total", "TTS jobs accepted")
TTS_ERRORS = Counter("tts_errors_total", "TTS job failures")
TTS_QUEUE_WAIT_MS = Histogram(
    "tts_queue_wait_ms",
    "Time a TTS job waits in the worker queue before synthesis begins",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500),
)
TTS_AUDIO_SECONDS = Counter("tts_output_audio_seconds_total", "Synthesized 8 kHz output seconds")
TTS_WORKER_ERRORS = Counter(
    "tts_worker_errors_total",
    "TTS synthesis failures per worker and device",
    ["worker", "device"],
)
TTS_SYNTHESIS_RTF = Histogram(
    "tts_synthesis_rtf",
    "TTS Real Time Factor per worker: synthesis_ms/audio_ms — GPU<0.05 CPU~0.3+",
    ["worker", "device"],
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
TTS_NO_AUDIO_TOTAL = Counter(
    "tts_no_audio_total",
    "Piper synthesize_wav produced no audio (empty phoneme sequence) per worker",
    ["worker"],
)
PIPELINE_RESULTS = Counter(
    "translate_pipeline_results_total", "Completed translate+TTS RecognitionResults"
)
PIPELINE_PENDING = Gauge(
    "translate_pipeline_pending", "Translate/TTS jobs pending across active RPCs"
)
PIPELINE_DEGRADED = Counter(
    "translate_pipeline_degraded_total",
    "Utterances degraded to text-only because TTS/translation queue was full",
)
PASSTHROUGH_TOTAL = Counter(
    "translate_passthrough_total",
    "Translation utterances replayed untranslated (original audio + ASR text), by reason",
    ["reason"],
)
ASR_BACKLOG_SOFT = Counter(
    "asr_backlog_soft_overload_total",
    "Chunks where per-utterance ASR backlog exceeded soft threshold (320 ms)",
)
ASR_BACKLOG_HARD = Counter(
    "asr_backlog_hard_violation_total",
    "Chunks where per-utterance ASR backlog exceeded hard threshold (800 ms)",
)

UTTERANCE_E2E_MS = Histogram(
    "utterance_e2e_ms",
    "Latency to the final RecognitionResult emit, from utterance START "
    "(segmentation off/shadow) or from segment COMMIT (segmentation on)",
    buckets=(100, 250, 500, 1000, 2000, 3000, 5000, 10000, 20000, 60000),
)
FIRST_AUDIO_MS = Histogram(
    "first_audio_ms",
    "Latency from utterance START to the first translated audio emitted for that utterance",
    buckets=(250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 60000),
)
FIRST_AUDIO_FROM_SPEECH_MS = Histogram(
    "first_audio_from_speech_ms",
    "Latency from the arrival of the first speech chunk of the utterance (start of the run "
    "of speech chunks that made VAD fire START) to the first translated audio emitted",
    buckets=(250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 60000),
)
POST_SPEECH_LATENCY_MS = Histogram(
    "post_speech_latency_ms",
    "Latency from VAD END (engine detected the speaker stopped) to the first translated audio "
    "emitted for the utterance; excludes the speech itself. Only audio from jobs created after "
    "VAD END is counted (segments committed mid-utterance are not post-speech)",
    buckets=(50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000),
)
SPEECH_END_TO_AUDIO_SENT_MS = Histogram(
    "speech_end_to_audio_sent_ms",
    "Latency from the estimated end of the speaker's speech (VAD END minus the silence window "
    "and the chunk tail) to the first translated audio of the utterance accepted by gRPC for "
    "sending. Assumes audio arrives at realtime pace; excludes delay before the engine "
    "(network, gateway). Segments committed before VAD END are not observed",
    buckets=(100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000),
)
SPEECH_END_TO_AUDIO_SENT_PREPROC_MS = Histogram(
    "speech_end_to_audio_sent_preproc_ms",
    "Same as speech_end_to_audio_sent_ms for streams running audio_preprocessing, plus the "
    "preprocessing delay D (hop + right context + compute budget). VAD sees the audio D after "
    "the speaker said it, so the estimated speech end in speech_end_to_audio_sent_ms is D late; "
    "this histogram adds D back. Streams without a preprocessor are not observed",
    buckets=(500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000),
)
GRPC_BUFFER_WAIT_MS = Histogram(
    "grpc_buffer_wait_ms",
    "Time a RecognitionResult waited in the response buffer between engine emit and the gRPC "
    "handler picking it up",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
)
GRPC_SEND_MS = Histogram(
    "grpc_send_ms",
    "Time gRPC took to accept one RecognitionResult for sending (grows with a slow reader / "
    "HTTP/2 flow control)",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
)
SEGMENT_PHASE_MS = Histogram(
    "segment_phase_ms",
    "Per-segment latency by phase up to the segment's first audio emit: asr (text first seen to "
    "translation submit, includes segmenter stability wait), translate_queue, translate, "
    "tts_queue, tts (first TTS phrase), order_hold (waiting for the previous segment's TTS/emit), "
    "total (translation submit to first audio emit)",
    ["phase"],
    buckets=(5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000),
)
SEGMENT_EMITTED = Counter(
    "segment_emitted_total",
    "Segment results emitted in order (segmentation on)",
)
ASR_FIRST_PARTIAL_LATENCY_MS = Histogram(
    "asr_first_partial_latency_ms",
    "Time from utterance START to first non-empty partial transcript",
    buckets=(50, 100, 150, 200, 300, 500, 800, 1000, 2000, 5000),
)
ASR_FINALIZATION_LATENCY_MS = Histogram(
    "asr_finalization_latency_ms",
    "Time from VAD END to ASR FINAL result (excludes VAD delay, isolates ASR finalization cost)",
    buckets=(50, 100, 150, 200, 300, 500, 800, 1000, 2000, 5000),
)
ASR_UTTERANCE_RTF = Histogram(
    "asr_utterance_rtf",
    "Per-utterance Real Time Factor: finalization latency / utterance audio duration",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0),
)
ACCEPT_CHUNK_MS = Histogram(
    "asr_accept_chunk_ms",
    "Time spent processing one AudioChunk on the asyncio event loop",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250),
)

VAD_SPEECH_START_DELAY_MS = Histogram(
    "vad_speech_start_delay_ms",
    "Speech duration accumulated before VAD fires START (approx. VAD_MIN_SPEECH_MS + chunk remainder); "
    "large values indicate slow speech onset detection → clipped utterance head",
    buckets=(50, 80, 100, 120, 150, 160, 200, 250, 320, 400, 500, 640),
)
VAD_SPEECH_END_DELAY_MS = Histogram(
    "vad_speech_end_delay_ms",
    "Silence duration accumulated before VAD fires END (approx. VAD_END_SILENCE_MS + chunk remainder); "
    "large values add latency between speaker stopping and ASR result",
    buckets=(100, 200, 300, 400, 480, 560, 640, 800, 1000, 1500, 2000),
)
VAD_STARTS_TOTAL = Counter(
    "vad_starts_total",
    "Times VAD fired START (utterance trigger); zero while audio flows means VAD never hears speech",
)
VAD_SCORE = Histogram(
    "vad_score",
    "Per-chunk speech probability from the VAD backend",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
VAD_MAX_SCORE = IntervalMax()
VAD_UTTERANCES_TOTAL = Counter(
    "vad_utterances_total",
    "Total utterances completed through ASR (FINAL received)",
)
VAD_SHORT_UTTERANCES_TOTAL = Counter(
    "vad_short_utterances_total",
    "Utterances shorter than 300 ms of audio — likely noise trigger or VAD fragmentation",
)
VAD_EMPTY_ASR_TOTAL = Counter(
    "vad_empty_asr_total",
    "Utterances where ASR returned empty text — VAD sent insufficient or non-speech audio",
)
VAD_SPEECH_AUDIO_SECONDS = Counter(
    "vad_speech_audio_seconds_total",
    "Audio seconds received during active utterances (speech only, excludes inter-utterance silence)",
)

ASR_CONFIDENCE = Histogram(
    "asr_confidence",
    "ASR final result confidence score [0, 1]",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
ASR_CONFIDENCE_SOURCE = Counter(
    "asr_confidence_source_total",
    "ASR final results counted by confidence derivation source",
    ["source"],
)

SEGMENT_COMMITS = Counter(
    "segment_commits_total",
    "Sentence segments committed by the segmenter (shadow: not used for output)",
    ["kind"],
)
SEGMENT_EARLY_COMMIT_GAIN_MS = Histogram(
    "segment_early_commit_gain_ms",
    "Time between a mid-utterance segment commit and ASR FINAL (ms the segment could start earlier)",
    buckets=(100, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000),
)
SEGMENT_PREMATURE = Counter(
    "segment_premature_total",
    "Mid-utterance segments whose words are missing from the ASR final text",
    ["kind"],
)
SEGMENT_REVISED = Counter(
    "segment_revised_total",
    "Committed segments whose words changed in a later ASR partial or final",
)
SEGMENT_ALIGN = Counter(
    "segment_align_total",
    "ASR texts in which the committed segments had to be re-found because ASR revised them "
    "(how: words, aligned, snap, fallback; phase partial: first time per utterance, final: at ASR FINAL)",
    ["how", "phase"],
)
SEGMENT_ERRORS = Counter(
    "segment_errors_total",
    "Segmenter exceptions (segmenter is disabled for the affected utterance)",
)


GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU compute utilization per device, sampled by GpuMonitor (%)",
    ["device_id"],
)
GPU_MEMORY_USED_BYTES = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used by all processes per device (bytes) — total minus free",
    ["device_id"],
)
GPU_MEMORY_TOTAL_BYTES = Gauge(
    "gpu_memory_total_bytes",
    "GPU total memory per device (bytes)",
    ["device_id"],
)


@dataclass(slots=True)
class MetricsServer:
    port: int

    def start(self) -> None:
        if self.port > 0:
            start_http_server(self.port)
