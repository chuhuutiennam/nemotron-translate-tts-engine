from __future__ import annotations

import json
import logging
import os
import sys

from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Iterator


# Libraries in the ASR/translation/TTS stack that are commonly noisy at INFO/WARNING.
_NOISY_LOGGERS = (
    "nemo",
    "nemo_logger",
    "lightning",
    "lightning_fabric",
    "pytorch_lightning",
    "transformers",
    "huggingface_hub",
    "onnxruntime",
    "urllib3",
)


PIPELINE_TRACE_LOGGERS = (
    "nemotron_testbed.coordinator.pipeline_trace",
    "nemotron_testbed.service.pipeline_trace",
)


class _TraceOrLevelFilter(logging.Filter):
    """Pass records at/above the file level, plus anything from trace loggers."""

    def __init__(self, file_level: int) -> None:
        super().__init__()
        self.file_level = file_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.file_level or record.name in PIPELINE_TRACE_LOGGERS


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "source": f"{record.filename}:{record.lineno}",
            "message": record.getMessage(),
        }
        for key in (
            # RPC / stream identity
            "rpc_id",
            "session_id",
            "stream_id",
            "utterance_id",
            # Stream lifecycle
            "task",
            "source_language",
            "target_language",
            "language_source",
            "source_mode",
            # Runtime
            "command",
            "latency_ms",
            "worker_id",
            # RTP continuity — gap
            "expected_ts",
            "received_ts",
            "gap_rtp_ticks",
            "gap_ms",
            # RTP continuity — jitter
            "diff_ticks",
            # RTP continuity — chunk_seq
            "expected_seq",
            "received_seq",
            "missing_count",
            # Audio validation
            "provided_duration_ms",
            "actual_duration_ms",
            "rtp_max_gap_ms",
            # ASR debug
            "raw_asr",
            "asr_text",
            "detected_language",
            "input_language",
            "flush",
            "step_num",
            "confidence",
            "confidence_source",
            # VAD debug
            "vad_score",
            "vad_speech",
            "vad_active",
            "vad_candidate_ms",
            "vad_silence_ms",
            "vad_event",
            "vad_end_tail_ms",
            "duration_ms",
            "pre_roll_ms",
            "utterance_audio_ms",
            "model_backlog_ms",
            "audio_ms",
            "vad_max_score",
            "peak_dbfs",
            "sample_rate",
            "channels",
            # Per-stage pipeline trace (ASR_FINAL / TRANSLATE_SUBMIT / TRANSLATE / ORDER_HOLD /
            # TTS_SUBMIT / TTS_PHRASE / EMIT / DEGRADE / STALE, GRPC_OPEN / STREAM_OPEN /
            # VAD_START / VAD_END / GRPC_SENT / STREAM_CLOSE / GRPC_CALL_ENDED / GRPC_CANCEL / GRPC_CLOSE)
            "stage",
            "text",
            "source_text",
            "translated_text",
            "remainder",
            "committed_tokens",
            "text_len",
            "source_text_len",
            "translated_text_len",
            "remainder_len",
            "language",
            "phrase_id",
            "is_last_phrase",
            "audio_bytes",
            "seq",
            "is_final",
            "queued",
            "skipped",
            "reason",
            "error",
            "origin",
            "status",
            "queue_depth",
            "pending_postprocess",
            "phrases",
            "phrase_texts",
            "waiting_for",
            "waiting_segment",
            "held_count",
            # GRPC_SENT (audio handed to the gRPC transport)
            "send_ms",
            "buffer_wait_ms",
            "first_audio",
            "speech_end_to_sent_ms",
            "speech_end_to_sent_preproc_ms",
            # GRPC_CALL_ENDED (call ended before the engine finished it)
            "peer",
            "grpc_code",
            "time_remaining_s",
            # SEGMENT_LATENCY per-phase timing
            "asr_ms",
            "translate_queue_ms",
            "translate_ms",
            "tts_queue_ms",
            "tts_ms",
            "order_hold_ms",
            "total_ms",
            # ASR_PARTIAL stability trace
            "step",
            "model_step",
            "changed",
            "t_ms",
            "infer_ms",
            "queue_ms",
            "asr_to_submit_ms",
            "seen_to_submit_ms",
            "common_prefix_words",
            "revised_words",
            "sentence_ends",
            # Sentence segmentation (SEGMENT_*)
            "segment_idx",
            "kind",
            "gain_ms",
            "premature",
            "revised",
            "forced_cut",
            "language_known",
            "phase",
            "how",
            "committed_text",
            "committed_text_len",
            # Periodic stats snapshot
            "stats",
            # trace_pcap (TRACE_PCAP_ENABLED / OPEN / SKIPPED / STOPPED / CLOSE)
            "path",
            "flows",
            "max_streams",
            "max_seconds",
            "min_free_mb",
            "packets_in",
            "packets_audx",
            "bytes",
            "stopped_reason",
            "seconds_in",
            "seconds_audx",
            "audx_stats",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    resolved = logging.getLevelName(value.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Invalid log level: {value}")
    return resolved


def _set_logger_level(name: str, level: int) -> None:
    target = logging.getLogger(name)
    target.setLevel(level)
    for handler in target.handlers:
        handler.setLevel(level)


def configure_logging(
    level: str,
    log_file: str = "",
    backup_count: int = 10,
    console_level: str = "ERROR",
    third_party_level: str = "ERROR",
    pipeline_trace_level: str | None = None,
) -> None:
    """Configure application logging.

    The log file rotates daily at local midnight and `backup_count` is the
    number of days kept. There is no size cap.

    `pipeline_trace_level` (None = trace disabled) gives the pipeline trace
    loggers their own threshold, independent of `level`: trace lines reach the
    file even when `level` is INFO, while other lines still obey `level`.

    `level` controls the application/file log level. `console_level` is kept
    independent so production can retain INFO diagnostics in the rotating file
    while showing only ERROR (or WARNING) on stdout.
    """

    file_level = _level(level)
    stdout_level = _level(console_level)
    external_level = _level(third_party_level)
    trace_level = _level(pipeline_trace_level) if pipeline_trace_level else None
    for trace_name in PIPELINE_TRACE_LOGGERS:
        logging.getLogger(trace_name).setLevel(
            trace_level if trace_level is not None else logging.NOTSET
        )
    formatter = JsonFormatter()
    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(stdout_level)
    console.setFormatter(formatter)
    handlers.append(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=path,
            when="midnight",
            backupCount=max(0, backup_count),
            encoding="utf-8",
        )
        if trace_level is not None and trace_level < file_level:
            file_handler.setLevel(trace_level)
            file_handler.addFilter(_TraceOrLevelFilter(file_level))
        else:
            file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    for handler in handlers:
        root.addHandler(handler)

    # The root level must not discard messages needed by either handler.
    root.setLevel(min(file_level if log_file else stdout_level, stdout_level))

    # Route Python warnings through logging. With CONSOLE_LOG_LEVEL=ERROR these
    # warnings stay out of the console but can still be retained in the log file.
    logging.captureWarnings(True)

    # Tell libraries that support environment-driven verbosity to stay quiet
    # before they are imported by NeMo/Transformers.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", third_party_level.lower())
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for name in _NOISY_LOGGERS:
        _set_logger_level(name, external_level)


def quiet_nemo_logging(level: str = "ERROR") -> None:
    """Best-effort suppression for NeMo's own logger wrapper."""

    numeric_level = _level(level)
    _set_logger_level("nemo", numeric_level)
    _set_logger_level("nemo_logger", numeric_level)
    try:
        from nemo.utils import logging as nemo_logging  # type: ignore

        set_level = getattr(nemo_logging, "setLevel", None)
        if callable(set_level):
            set_level(numeric_level)
    except Exception:
        # Logging suppression must never make model startup fail.
        pass


def quiet_transformers_logging(level: str = "ERROR") -> None:
    """Apply Transformers/HuggingFace verbosity after those modules exist.

    Transformers may install its own handler during import, so configuring the
    Python logger before import is not always sufficient. This helper updates
    both the library API and the Python logger/handlers.
    """

    numeric_level = _level(level)
    _set_logger_level("transformers", numeric_level)
    _set_logger_level("huggingface_hub", numeric_level)
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity(numeric_level)
        transformers_logging.disable_progress_bar()
    except Exception:
        pass


@contextmanager
def suppress_external_stdio(enabled: bool = True, stderr: bool = True) -> Iterator[None]:
    """Temporarily discard writes to process stdout/stderr file descriptors.

    Important: this deliberately does *not* use contextlib.redirect_stdout or
    redirect_stderr. Those functions replace the process-global `sys.stdout`
    and `sys.stderr` objects. If a library such as Transformers creates a
    logging.StreamHandler while they point at a temporary devnull file, that
    handler keeps the temporary stream after the context exits; once devnull is
    closed, later logging fails with `ValueError: I/O operation on closed file`.

    Redirecting OS file descriptors 1 and 2 keeps the Python stream objects
    stable, so third-party handlers remain valid after model initialization.
    The redirection is process-wide and is therefore intended only for short
    model initialization/warm-up sections.

    ``stderr=False`` suppresses only stdout (fd 1) and leaves stderr (fd 2)
    open.  Use this when a library's CUDA initialisation writes diagnostic
    output to stderr that must reach the kernel driver to succeed — redirecting
    fd 2 to /dev/null in a process with active CUDA contexts from other
    libraries (PyTorch, CTranslate2) can cause cuDNN JIT compilation to fail
    silently, which makes ONNX Runtime fall back to CPUExecutionProvider.

    Note: do NOT wrap the fd manipulation in a threading.Lock. Releasing the
    lock between setup (dup2 fd→/dev/null) and restore exposes a window where
    ONNX Runtime's CUDA/TRT kernel-compilation threads may write to fd 1 and
    fail to communicate back, causing silent CPU fallback. Workers serialise
    naturally via pool.start() / _ready.wait(), so no lock is needed.
    """

    if not enabled:
        yield
        return

    saved_stdout_fd: int | None = None
    saved_stderr_fd: int | None = None
    devnull_fd: int | None = None

    try:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        saved_stdout_fd = os.dup(1)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        if stderr:
            saved_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        if saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
        if saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
