from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

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
    ASR_FINALIZE_STEPS,
    ASR_FIRST_PARTIAL_LATENCY_MS,
    ASR_POOL_WORKERS,
    ASR_QUEUE_WAIT_MS,
    ASR_STREAM_AFFINITIES,
    ASR_UTTERANCE_RTF,
    AUDIO_PREPROC_BYPASS,
    AUDIO_PREPROC_FALLBACK,
    AUDIO_PREPROC_FLUSH_SECONDS,
    AUDIO_PREPROC_LATE_SECONDS,
    AUDIO_PREPROC_POOL_WORKERS,
    AUDIO_PREPROC_QUEUE,
    AUDIO_PREPROC_QUEUE_WAIT_SECONDS,
    AUDIO_PREPROC_REGIONS,
    AUDIO_PREPROC_SHED_STREAMS,
    AUDIO_PREPROC_SLACK_SECONDS,
    AUDIO_PREPROC_STATE_RESETS,
    AUDIO_PREPROC_STREAMS_ACTIVE,
    AUDIO_PREPROC_WINDOW_SECONDS,
    AUDIO_SECONDS,
    DEFERRED_BACKLOG_MS,
    DROPPED_PARTIALS,
    DURATION_MISMATCHES,
    FINAL_RESULTS,
    GPU_MEMORY_TOTAL_BYTES,
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION,
    INGRESS_BACKLOG_MS,
    INGRESS_CHUNKS,
    INGRESS_PEAK,
    MODEL_STEP_SECONDS,
    PARTIAL_RESULTS,
    PIPELINE_DEGRADED,
    PIPELINE_PENDING,
    PIPELINE_RESULTS,
    RTP_GAP_SILENCE_INSERTED,
    RTP_GAPS,
    RTP_JITTER_CORRECTED,
    RTP_OUT_OF_ORDER,
    RUNTIME_QUEUE,
    STALE_RESULTS,
    STREAM_ERRORS,
    TRANSLATION_ERRORS,
    TRANSLATION_JOBS,
    TRANSLATION_LATENCY,
    TRANSLATION_POOL_WORKERS,
    TRANSLATION_QUEUE,
    TRANSLATION_QUEUE_WAIT_MS,
    TTS_AUDIO_SECONDS,
    TTS_ERRORS,
    TTS_JOBS,
    TTS_LATENCY,
    TTS_NO_AUDIO_TOTAL,
    TTS_POOL_WORKERS,
    TTS_QUEUE,
    TTS_QUEUE_WAIT_MS,
    TTS_SYNTHESIS_RTF,
    TTS_WORKER_ERRORS,
    UTTERANCE_E2E_MS,
    VAD_EMPTY_ASR_TOTAL,
    VAD_MAX_SCORE,
    VAD_SCORE,
    VAD_SHORT_UTTERANCES_TOTAL,
    VAD_SPEECH_AUDIO_SECONDS,
    VAD_SPEECH_END_DELAY_MS,
    VAD_SPEECH_START_DELAY_MS,
    VAD_STARTS_TOTAL,
    VAD_UTTERANCES_TOTAL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus read helpers
# ---------------------------------------------------------------------------

def _gauge(metric) -> float:
    for mf in metric.collect():
        for s in mf.samples:
            if not s.labels:
                return s.value
    return 0.0


def _labeled_gauge(metric) -> dict[str, float]:
    result: dict[str, float] = {}
    for mf in metric.collect():
        for s in mf.samples:
            if s.labels:
                key = "/".join(str(v) for v in s.labels.values())
                result[key] = s.value
    return result


def _counter(metric) -> float:
    for mf in metric.collect():
        for s in mf.samples:
            if s.name.endswith("_total") and not s.labels:
                return s.value
    return 0.0


def _labeled_counter(metric, *label_names: str) -> dict[str, float]:
    """Return {composite_label_key: value} for a labeled counter.

    composite_label_key is label_names values joined by '/'.
    """
    result: dict[str, float] = {}
    for mf in metric.collect():
        for s in mf.samples:
            if not s.name.endswith("_total"):
                continue
            key = "/".join(str(s.labels.get(ln, "")) for ln in label_names)
            result[key] = s.value
    return result


def _labeled_histogram_mean(
    metric, *label_names: str
) -> dict[str, tuple[float, float]]:
    """Return {composite_label_key: (sum, count)} for a labeled histogram."""
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for mf in metric.collect():
        for s in mf.samples:
            key = "/".join(str(s.labels.get(ln, "")) for ln in label_names)
            if s.name.endswith("_sum"):
                sums[key] = s.value
            elif s.name.endswith("_count"):
                counts[key] = s.value
    return {k: (sums.get(k, 0.0), counts.get(k, 0.0)) for k in set(sums) | set(counts)}


def _histogram(metric) -> tuple[float, float]:
    """Return (sum, count) for an unlabeled histogram."""
    h_sum = 0.0
    h_count = 0.0
    for mf in metric.collect():
        for s in mf.samples:
            if s.labels:
                continue
            if s.name.endswith("_sum"):
                h_sum = s.value
            elif s.name.endswith("_count"):
                h_count = s.value
    return h_sum, h_count


def _labeled_histogram(metric, label_name: str) -> dict[str, tuple[float, float]]:
    """Return {label_value: (sum, count)} for a labeled histogram."""
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for mf in metric.collect():
        for s in mf.samples:
            if label_name not in s.labels:
                continue
            key = s.labels[label_name]
            if s.name.endswith("_sum"):
                sums[key] = s.value
            elif s.name.endswith("_count"):
                counts[key] = s.value
    return {k: (sums.get(k, 0.0), counts.get(k, 0.0)) for k in set(sums) | set(counts)}



def _counter_by(metric, label_name: str) -> dict[str, float]:
    """{label value: total over every other label (e.g. backend)} for a labeled counter."""
    result: dict[str, float] = {}
    for mf in metric.collect():
        for s in mf.samples:
            if s.name.endswith("_total"):
                key = str(s.labels.get(label_name, ""))
                result[key] = result.get(key, 0.0) + s.value
    return result


def _all_backends(per_backend: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """(sum, count) over every backend label (a process runs one audio_preprocessing backend)."""
    return (sum(v[0] for v in per_backend.values()), sum(v[1] for v in per_backend.values()))

# ---------------------------------------------------------------------------
# StatsLogger
# ---------------------------------------------------------------------------

class StatsLogger:
    """Asyncio task that logs a metric snapshot at a fixed interval."""

    def __init__(self, interval_s: int) -> None:
        self._interval = interval_s
        self._task: asyncio.Task[None] | None = None
        self._prev: dict[str, Any] | None = None

    async def start(self) -> None:
        if self._interval <= 0:
            return
        self._task = asyncio.create_task(self._loop(), name="stats-logger")
        logger.info("Stats logger started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                self._tick()
            except Exception:
                logger.exception("stats-logger tick failed")

    def _tick(self) -> None:
        snap = self._snapshot()
        prev = self._prev
        self._prev = snap
        self._log(snap, prev)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        return {
            "ts": time.monotonic(),
            # gauges (current)
            "g_rpcs": _gauge(ACTIVE_RPCS),
            "g_streams": _gauge(ACTIVE_STREAMS),
            "g_utterances": _gauge(ACTIVE_UTTERANCES),
            "g_pipeline_pending": _gauge(PIPELINE_PENDING),
            "g_asr_pool_workers": _gauge(ASR_POOL_WORKERS),
            "g_translation_pool_workers": _gauge(TRANSLATION_POOL_WORKERS),
            "g_tts_pool_workers": _gauge(TTS_POOL_WORKERS),
            "g_asr_stream_affinities": _gauge(ASR_STREAM_AFFINITIES),
            "g_asr_queue": _labeled_gauge(RUNTIME_QUEUE),
            "g_translation_queue": _labeled_gauge(TRANSLATION_QUEUE),
            "g_tts_queue": _labeled_gauge(TTS_QUEUE),
            # counters (cumulative)
            "c_audio_s": _counter(AUDIO_SECONDS),
            "c_chunks": _counter(INGRESS_CHUNKS),
            "m_peak": INGRESS_PEAK.take(),
            "c_partials": _counter(PARTIAL_RESULTS),
            "c_finals": _counter(FINAL_RESULTS),
            "c_stale": _counter(STALE_RESULTS),
            "c_stream_errors": _counter(STREAM_ERRORS),
            "c_dropped_partials": _counter(DROPPED_PARTIALS),
            "c_duration_mismatches": _counter(DURATION_MISMATCHES),
            "c_rtp_gaps": _counter(RTP_GAPS),
            "c_rtp_silence": _counter(RTP_GAP_SILENCE_INSERTED),
            "c_rtp_ooo": _counter(RTP_OUT_OF_ORDER),
            "c_rtp_jitter": _counter(RTP_JITTER_CORRECTED),
            "c_backlog_soft": _counter(ASR_BACKLOG_SOFT),
            "c_backlog_hard": _counter(ASR_BACKLOG_HARD),
            "c_translation_jobs": _counter(TRANSLATION_JOBS),
            "c_translation_errors": _counter(TRANSLATION_ERRORS),
            "c_tts_jobs": _counter(TTS_JOBS),
            "c_tts_errors": _counter(TTS_ERRORS),
            "c_tts_audio_s": _counter(TTS_AUDIO_SECONDS),
            "c_tts_no_audio": _labeled_counter(TTS_NO_AUDIO_TOTAL, "worker"),
            "c_tts_worker_errors": _labeled_counter(TTS_WORKER_ERRORS, "worker", "device"),
            "h_tts_rtf": _labeled_histogram_mean(TTS_SYNTHESIS_RTF, "worker", "device"),
            "c_pipeline_results": _counter(PIPELINE_RESULTS),
            "c_pipeline_degraded": _counter(PIPELINE_DEGRADED),
            # histograms (sum, count)
            "h_accept_chunk_ms": _histogram(ACCEPT_CHUNK_MS),
            "h_queue_wait_ms": _histogram(ASR_QUEUE_WAIT_MS),
            "h_finalize_steps": _histogram(ASR_FINALIZE_STEPS),
            "h_model_step": _labeled_histogram(MODEL_STEP_SECONDS, "command"),
            "h_translation_s": _histogram(TRANSLATION_LATENCY),
            "h_translation_queue_wait_ms": _histogram(TRANSLATION_QUEUE_WAIT_MS),
            "h_tts_s": _histogram(TTS_LATENCY),
            "h_tts_queue_wait_ms": _histogram(TTS_QUEUE_WAIT_MS),
            "h_e2e_ms": _histogram(UTTERANCE_E2E_MS),
            "h_first_partial_ms": _histogram(ASR_FIRST_PARTIAL_LATENCY_MS),
            "h_finalization_ms": _histogram(ASR_FINALIZATION_LATENCY_MS),
            "h_utterance_rtf": _histogram(ASR_UTTERANCE_RTF),
            "h_ingress_backlog_ms": _histogram(INGRESS_BACKLOG_MS),
            "h_deferred_backlog_ms": _histogram(DEFERRED_BACKLOG_MS),
            "h_vad_start_delay_ms": _histogram(VAD_SPEECH_START_DELAY_MS),
            "h_vad_end_delay_ms": _histogram(VAD_SPEECH_END_DELAY_MS),
            "c_vad_starts": _counter(VAD_STARTS_TOTAL),
            "h_vad_score": _histogram(VAD_SCORE),
            "m_vad_max_score": VAD_MAX_SCORE.take(),
            "c_vad_utterances": _counter(VAD_UTTERANCES_TOTAL),
            "c_vad_short": _counter(VAD_SHORT_UTTERANCES_TOTAL),
            "c_vad_empty_asr": _counter(VAD_EMPTY_ASR_TOTAL),
            "c_vad_speech_s": _counter(VAD_SPEECH_AUDIO_SECONDS),
            "h_asr_confidence": _histogram(ASR_CONFIDENCE),
            "c_asr_confidence_source": _labeled_counter(ASR_CONFIDENCE_SOURCE, "source"),
            # audio preprocessing (section only reported while the pool is running)
            "g_preproc_workers": sum(_labeled_gauge(AUDIO_PREPROC_POOL_WORKERS).values()),
            "g_preproc_streams": sum(_labeled_gauge(AUDIO_PREPROC_STREAMS_ACTIVE).values()),
            "g_preproc_queue": _labeled_gauge(AUDIO_PREPROC_QUEUE),
            "c_preproc_regions": _counter_by(AUDIO_PREPROC_REGIONS, "outcome"),
            "c_preproc_fallback": _counter_by(AUDIO_PREPROC_FALLBACK, "reason"),
            "c_preproc_bypass": sum(_labeled_counter(AUDIO_PREPROC_BYPASS, "backend").values()),
            "c_preproc_shed": sum(_labeled_counter(AUDIO_PREPROC_SHED_STREAMS, "backend").values()),
            "c_preproc_resets": _counter_by(AUDIO_PREPROC_STATE_RESETS, "reason"),
            "h_preproc_window_s": _all_backends(_labeled_histogram(AUDIO_PREPROC_WINDOW_SECONDS, "backend")),
            "h_preproc_queue_wait_s": _all_backends(_labeled_histogram(AUDIO_PREPROC_QUEUE_WAIT_SECONDS, "backend")),
            "h_preproc_flush_s": _all_backends(_labeled_histogram(AUDIO_PREPROC_FLUSH_SECONDS, "backend")),
            "h_preproc_late_s": _all_backends(_labeled_histogram(AUDIO_PREPROC_LATE_SECONDS, "backend")),
            "h_preproc_slack_s": _all_backends(_labeled_histogram(AUDIO_PREPROC_SLACK_SECONDS, "backend")),
            # gpu (updated by GpuMonitor every 5s; empty dict if CPU-only)
            "g_gpu_util": _labeled_gauge(GPU_UTILIZATION),
            "g_gpu_mem_used": _labeled_gauge(GPU_MEMORY_USED_BYTES),
            "g_gpu_mem_total": _labeled_gauge(GPU_MEMORY_TOTAL_BYTES),
        }

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log(self, snap: dict[str, Any], prev: dict[str, Any] | None) -> None:
        interval = snap["ts"] - prev["ts"] if prev else self._interval

        def _delta_int(key: str) -> int:
            if prev is None:
                return int(snap[key])
            return int(snap[key] - prev[key])

        def _delta_f(key: str, dp: int = 3) -> float:
            raw = snap[key] - (prev[key] if prev else 0.0)
            return round(raw, dp)

        def _mean(h_key: str, dp: int = 3) -> float | None:
            s, c = snap[h_key]
            ps, pc = prev[h_key] if prev else (0.0, 0.0)
            dc = c - pc
            if dc <= 0:
                return None
            return round((s - ps) / dc, dp)

        def _mean_ms(h_key: str, dp: int = 3) -> float | None:
            v = _mean(h_key, dp=dp + 3)
            return None if v is None else round(v * 1000.0, dp)

        # model_step per-command breakdown (stored in seconds → emit ms)
        prev_model = prev["h_model_step"] if prev else {}
        model_detail: dict[str, Any] = {}
        for cmd, (s, c) in snap["h_model_step"].items():
            ps, pc = prev_model.get(cmd, (0.0, 0.0))
            dc = c - pc
            model_detail[cmd] = {
                "mean_ms": round((s - ps) / dc * 1000.0, 3) if dc > 0 else None,
                "n": int(dc),
            }

        report: dict[str, Any] = {
            "interval_s": round(interval, 1),
            "active": {
                "rpcs": int(snap["g_rpcs"]),
                "streams": int(snap["g_streams"]),
                "utterances": int(snap["g_utterances"]),
                "pipeline_pending": int(snap["g_pipeline_pending"]),
                "asr_pool_workers": int(snap["g_asr_pool_workers"]),
                "translation_pool_workers": int(snap["g_translation_pool_workers"]),
                "tts_pool_workers": int(snap["g_tts_pool_workers"]),
                "asr_stream_affinities": int(snap["g_asr_stream_affinities"]),
            },
            "queues": {
                "asr": snap["g_asr_queue"],
                "translation": snap["g_translation_queue"],
                "tts": snap["g_tts_queue"],
            },
            "ingress": {
                "audio_s": _delta_f("c_audio_s"),
                "chunks": _delta_int("c_chunks"),
                "peak_dbfs": (
                    None
                    if snap["m_peak"] is None
                    else round(20.0 * math.log10(max(snap["m_peak"], 1e-8)), 1)
                ),
                "partials": _delta_int("c_partials"),
                "finals": _delta_int("c_finals"),
                "stale": _delta_int("c_stale"),
                "dropped_partials": _delta_int("c_dropped_partials"),
                "accept_chunk_mean_ms": _mean("h_accept_chunk_ms"),
                "ingress_backlog_mean_ms": _mean("h_ingress_backlog_ms"),
                "deferred_backlog_mean_ms": _mean("h_deferred_backlog_ms"),
            },
            "asr": {
                "backlog_soft": _delta_int("c_backlog_soft"),
                "backlog_hard": _delta_int("c_backlog_hard"),
                "queue_wait_mean_ms": _mean("h_queue_wait_ms"),
                "finalize_steps_mean": _mean("h_finalize_steps"),
                "first_partial_mean_ms": _mean("h_first_partial_ms"),
                "finalization_mean_ms": _mean("h_finalization_ms"),
                "utterance_rtf_mean": _mean("h_utterance_rtf"),
                "confidence_mean": _mean("h_asr_confidence"),
                "confidence_source": {
                    k: int(v - (prev["c_asr_confidence_source"].get(k, 0.0) if prev else 0.0))
                    for k, v in snap["c_asr_confidence_source"].items()
                },
                "model_step": model_detail,
            },
            "pipeline": {
                "results": _delta_int("c_pipeline_results"),
                "degraded": _delta_int("c_pipeline_degraded"),
                "translation_mean_ms": _mean_ms("h_translation_s"),
                "translation_queue_wait_mean_ms": _mean("h_translation_queue_wait_ms"),
                "tts_mean_ms": _mean_ms("h_tts_s"),
                "tts_queue_wait_mean_ms": _mean("h_tts_queue_wait_ms"),
                "e2e_mean_ms": _mean("h_e2e_ms"),
                "tts_audio_s": _delta_f("c_tts_audio_s"),
                "tts_no_audio": {
                    k: int(v - (prev["c_tts_no_audio"].get(k, 0.0) if prev else 0.0))
                    for k, v in snap["c_tts_no_audio"].items()
                },
                "tts_worker_errors": {
                    k: int(v - (prev["c_tts_worker_errors"].get(k, 0.0) if prev else 0.0))
                    for k, v in snap["c_tts_worker_errors"].items()
                },
                "tts_rtf_mean": {
                    k: (
                        round((s - (prev["h_tts_rtf"].get(k, (0.0, 0.0))[0] if prev else 0.0))
                             / (c - (prev["h_tts_rtf"].get(k, (0.0, 0.0))[1] if prev else 0.0)), 3)
                        if (c - (prev["h_tts_rtf"].get(k, (0.0, 0.0))[1] if prev else 0.0)) > 0
                        else None
                    )
                    for k, (s, c) in snap["h_tts_rtf"].items()
                    if isinstance(s, float) and isinstance(c, float)
                },
            },
            "errors": {
                "stream": _delta_int("c_stream_errors"),
                "translation": _delta_int("c_translation_errors"),
                "tts": _delta_int("c_tts_errors"),
                "duration_mismatches": _delta_int("c_duration_mismatches"),
            },
            "rtp": {
                "gaps": _delta_int("c_rtp_gaps"),
                "silence_inserted": _delta_int("c_rtp_silence"),
                "ooo": _delta_int("c_rtp_ooo"),
                "jitter_corrected": _delta_int("c_rtp_jitter"),
            },
            "vad": {
                "starts": _delta_int("c_vad_starts"),
                "score_mean": _mean("h_vad_score"),
                "max_score": (
                    None if snap["m_vad_max_score"] is None else round(snap["m_vad_max_score"], 3)
                ),
                "utterances": _delta_int("c_vad_utterances"),
                "short": _delta_int("c_vad_short"),
                "empty_asr": _delta_int("c_vad_empty_asr"),
                "speech_s": _delta_f("c_vad_speech_s"),
                "start_delay_mean_ms": _mean("h_vad_start_delay_ms"),
                "end_delay_mean_ms": _mean("h_vad_end_delay_ms"),
            },
            "gpu": {
                dev_id: {
                    "util_pct": round(snap["g_gpu_util"].get(dev_id, -1.0), 1),
                    "mem_used_mb": round(snap["g_gpu_mem_used"].get(dev_id, 0.0) / 1024 / 1024, 1),
                    "mem_total_mb": round(snap["g_gpu_mem_total"].get(dev_id, 0.0) / 1024 / 1024, 1),
                }
                for dev_id in sorted(
                    set(snap["g_gpu_util"]) | set(snap["g_gpu_mem_used"]) | set(snap["g_gpu_mem_total"])
                )
            },
        }

        if snap["g_preproc_workers"] > 0:

            def _delta_map(key: str) -> dict[str, int]:
                before = prev[key] if prev else {}
                return {k: int(v - before.get(k, 0.0)) for k, v in snap[key].items()}

            report["audio_preprocessing"] = {
                "streams": int(snap["g_preproc_streams"]),
                "queue": snap["g_preproc_queue"],
                "regions": _delta_map("c_preproc_regions"),
                "fallback": _delta_map("c_preproc_fallback"),
                "bypass": _delta_int("c_preproc_bypass"),
                "shed_streams": _delta_int("c_preproc_shed"),
                "state_resets": _delta_map("c_preproc_resets"),
                "window_mean_ms": _mean_ms("h_preproc_window_s"),
                "queue_wait_mean_ms": _mean_ms("h_preproc_queue_wait_s"),
                "flush_mean_ms": _mean_ms("h_preproc_flush_s"),
                "late_mean_ms": _mean_ms("h_preproc_late_s"),
                "slack_mean_ms": _mean_ms("h_preproc_slack_s"),
            }

        logger.info("STATS", extra={"stats": report})
