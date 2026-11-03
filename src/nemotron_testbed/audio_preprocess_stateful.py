"""Stateful audio preprocessing for one stream (mode audx): the model keeps its recurrent state for the whole call.

Same contract as StreamAudioPreprocessor (process / flush / deliver / close, constant delay D, raw fallback) so the
coordinator treats both alike; design: docs/audx-stateful-design.md.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

from .audio_preprocess import SAMPLE_RATE, PreprocessResult, SubmitFn, _Track
from .config import AudioPreprocessSettings
from .domain import AudioPreprocessJob, AudioPreprocessResult, StreamKey, SubmitStatus
from .metrics import (
    AUDIO_PREPROC_BYPASS,
    AUDIO_PREPROC_FALLBACK,
    AUDIO_PREPROC_FLUSH_SECONDS,
    AUDIO_PREPROC_LATE_SECONDS,
    AUDIO_PREPROC_REGIONS,
    AUDIO_PREPROC_SLACK_SECONDS,
    AUDIO_PREPROC_STATE_RESETS,
    AUDIO_PREPROC_STREAMS_ACTIVE,
    audio_preproc_backend,
    audio_preproc_reason,
)

logger = logging.getLogger(__name__)

MAX_BACKLOG_S = 1.0  # audio waiting on a full queue longer than this is given up (played raw), the state restarts

StateFactory = Callable[[], Any]


class StatefulStreamPreprocessor:
    """Stateful AudX preprocessing (mode audx) for one continuous audio stream (one StreamKey).

    Content position c is played at output position c + D, D = hop + look-ahead + budget, constant. Audio goes to
    the pool in jobs of at least hop samples; at most one job of the stream is in flight and it carries the
    stream's model state, so the jobs of a stream run in order on any worker. A job returns the enhanced samples
    the model could complete (it lags its input by the look-ahead), placed at the stream's output position.
    Content without enhanced samples when it is played plays raw (late, shed, error, bypass); a late job still
    completes so the model state stays continuous. Switches between raw and enhanced are crossfaded inside the
    enhanced run. Invariants as StreamAudioPreprocessor: len(out) == len(in), D constant, first D samples zero.
    """

    def __init__(
        self,
        key: StreamKey,
        settings: AudioPreprocessSettings,
        submit: SubmitFn,
        new_state: StateFactory,
    ) -> None:
        ms = SAMPLE_RATE // 1000
        self.key = key
        self._submit_fn = submit
        self._new_state = new_state
        self._backend = audio_preproc_backend(settings.mode)
        # fallback reasons: AUDX_LATE / AUDX_QUEUE_FULL / AUDX_ERROR (resemble-enhance modes use DENOISER_*)
        self._bad_outcomes = {
            outcome: audio_preproc_reason(settings.mode, what)
            for outcome, what in (("late", "LATE"), ("shed", "QUEUE_FULL"), ("error", "ERROR"))
        }
        self._h = settings.hop_ms * ms
        self._x = settings.crossfade_ms * ms
        self._delay = settings.delay_ms * ms
        self._min_run = settings.dither_min_zero_run
        self._bad_limit = settings.fallback_bad_regions
        self._bypass_samples = (
            math.inf if settings.fallback_bypass_s <= 0 else int(settings.fallback_bypass_s * SAMPLE_RATE)
        )
        self._max_backlog = int(MAX_BACKLOG_S * SAMPLE_RATE)
        self.flush_timeout_s = max(1.0, 2.0 * settings.compute_budget_ms / 1000.0)
        if self._h <= 0 or not (0 <= self._x <= self._h):
            raise ValueError("invalid stateful audio preprocessing hop/crossfade")

        self._track = _Track()  # raw content + digital-zero-run mask
        self._enh = _Track()  # enhanced content; its mask = enhanced sample present
        self._open_run = 0
        self._total_in = 0
        self._emitted = 0
        self._sent = 0  # content handed to the model (or given up on)
        self._out_pos = 0  # where the current state's next output sample goes
        self._state = new_state()
        self._epoch = 0
        self._job_seq = 0
        self._inflight: int | None = None
        self._jobs: dict[int, int] = {}  # job id -> epoch, for jobs of this stream not answered yet
        self._inbox: list[AudioPreprocessResult] = []
        self._lock = threading.Lock()
        self._bad: deque[list] = deque()  # [start, end, cause]: content the model will not enhance, and why
        # enhanced samples in a row right before _emitted (capped at the crossfade). Nothing precedes the first
        # sample of the stream, so it starts as a full run: enhanced audio from sample 0, no fade in from raw.
        self._run = self._x
        self._region_start = 0
        self._region_missing = False
        self._region_silent = True
        self._consec_bad = 0
        self._bypass_until: float = 0
        self._needs_reset = False
        self._logged_reasons: set[str] = set()
        self._flushed = False
        self._closed = False
        self.stats: dict[str, int] = {
            "samples_in": 0,
            "samples_out": 0,
            "enhanced": 0,
            "silent": 0,
            "late": 0,
            "shed": 0,
            "error": 0,
            "bypass": 0,
            "bypass_events": 0,
            "resets": 0,
        }
        AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=self._backend).inc()

    # ── contract shared with StreamAudioPreprocessor ──────────────────────────

    @property
    def delay_samples(self) -> int:
        return self._delay

    @property
    def bypassed(self) -> bool:
        return self._total_in < self._bypass_until

    @property
    def samples_received(self) -> int:
        return self._total_in

    def unreleased_raw(self, upto: int | None = None) -> np.ndarray:
        """Raw content in [released, upto) still held back; for the caller to play when process()/flush() failed. Never raises."""
        try:
            start = max(self._emitted, self._track.start)
            end = self._track.end if upto is None else min(upto, self._track.end)
            if end <= start:
                return np.zeros(0, dtype=np.float32)
            return self._track.view(start, end)[0].copy()
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def deliver(self, result: AudioPreprocessResult) -> None:
        """Called from a worker thread; the event loop picks the result up on its next process()/flush() step."""
        with self._lock:
            if not self._closed:
                self._inbox.append(result)

    def process(self, audio: np.ndarray) -> PreprocessResult:
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if self._closed or self._flushed:
            return PreprocessResult(audio.copy(), audio.copy())
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return PreprocessResult(empty, empty.copy())
        self._drain()
        self._ingest(audio)
        self._pump()
        count = max(0, self._total_in - self._delay) - self._emitted
        enhanced, raw = self._play(count)
        self._emitted += count
        self.stats["samples_in"] += n
        self.stats["samples_out"] += n
        pad = np.zeros(n - count, dtype=np.float32)
        result = PreprocessResult(np.concatenate((pad, enhanced)), np.concatenate((pad, raw)))
        self._trim()
        return result

    async def flush(self, timeout_s: float | None = None) -> PreprocessResult:
        """End of stream: send the rest with final=True so the model releases its look-ahead, then play it all."""
        empty = np.zeros(0, dtype=np.float32)
        if self._closed or self._flushed:
            return PreprocessResult(empty, empty.copy())
        self._flushed = True
        remaining = self._total_in - self._emitted
        if remaining <= 0:
            return PreprocessResult(empty, empty.copy())
        started = time.monotonic()
        deadline = started + (self.flush_timeout_s if timeout_s is None else timeout_s)
        self._drain()
        if self.bypassed:
            self._mark(self._sent, self._total_in, "bypass")
            self._sent = self._total_in
        else:
            if self._needs_reset:
                self._reset("bypass")
            final_sent = False
            while time.monotonic() < deadline:
                self._drain()
                if self._inflight is None:
                    if final_sent:
                        break
                    status = self._send(final=True)
                    if status is SubmitStatus.ACCEPTED:
                        final_sent = True
                        continue
                    if status is not SubmitStatus.OVERLOADED:
                        break
                await asyncio.sleep(0.005)
            self._drain()
        enhanced, raw = self._play(remaining, stream_end=True)
        self._emitted += remaining
        if self._emitted > self._region_start:
            self._close_region(self._emitted)
        self.stats["samples_out"] += remaining
        AUDIO_PREPROC_FLUSH_SECONDS.labels(backend=self._backend).observe(time.monotonic() - started)
        return PreprocessResult(enhanced, raw)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._inbox.clear()
        self._jobs.clear()
        self._bad.clear()
        self._state = None
        AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=self._backend).dec()
        logger.info("AUDIO_PREPROCESSING_CLOSE", extra=self._log_extra(stats=dict(self.stats)))

    # ── model jobs ────────────────────────────────────────────────────────────

    def _pump(self) -> None:
        if self.bypassed:
            self._mark(self._sent, self._total_in, "bypass")
            self._sent = self._total_in
            self._needs_reset = True
            return
        if self._needs_reset:
            self._reset("bypass")
        if self._inflight is None and self._total_in - self._sent >= self._h:
            self._send(final=False)

    def _send(self, *, final: bool) -> SubmitStatus:
        samples = self._track.view(self._sent, self._total_in)[0].copy()
        job_id = self._job_seq
        self._job_seq += 1
        job = AudioPreprocessJob(key=self.key, region_id=job_id, window=samples, state=self._state, final=final)
        try:
            status = self._submit_fn(job)
        except Exception:
            logger.exception("Audio preprocessing stream job submit failed", extra=self._log_extra())
            status = SubmitStatus.UNAVAILABLE
        if status is SubmitStatus.ACCEPTED:
            self._jobs[job_id] = self._epoch
            self._inflight = job_id
            self._sent = self._total_in
        elif status is SubmitStatus.OVERLOADED:
            # keep the audio and retry on the next chunk; give it up once it waited too long
            self._mark(self._sent, self._total_in, "shed")
            if self._total_in - self._sent > self._max_backlog:
                self._sent = self._total_in
                self._reset("shed")
        else:
            self._mark(self._sent, self._total_in, "error")
            self._sent = self._total_in
            self._reset("error")
        return status

    def _drain(self) -> None:
        with self._lock:
            results, self._inbox = self._inbox, []
        for result in results:
            epoch = self._jobs.pop(result.region_id, None)
            if epoch is None or epoch != self._epoch:
                continue  # a job of a state that was restarted since
            if result.region_id == self._inflight:
                self._inflight = None
            audio = result.audio
            start = self._out_pos
            if audio is None or start + len(audio) > self._sent:
                if audio is not None:
                    logger.error(
                        "Audio preprocessing stream job returned %d samples, at most %d expected",
                        len(audio),
                        self._sent - start,
                        extra=self._log_extra(),
                    )
                self._mark(self._out_pos, self._sent, "error")
                self._reset("error")
                continue
            self._write_enhanced(start, audio)
            self._out_pos = start + len(audio)
            if audio.size and not self._flushed:
                # start is played once content start + D has arrived
                left = start + self._delay - self._total_in
                if left >= 0:
                    AUDIO_PREPROC_SLACK_SECONDS.labels(backend=self._backend).observe(left / SAMPLE_RATE)
                else:
                    AUDIO_PREPROC_LATE_SECONDS.labels(backend=self._backend).observe(-left / SAMPLE_RATE)

    def _write_enhanced(self, start: int, audio: np.ndarray) -> None:
        lo = max(start, self._emitted, self._enh.start)
        hi = min(start + len(audio), self._enh.end)
        if hi > lo:
            data, have = self._enh.view(lo, hi)
            data[:] = audio[lo - start : hi - start]
            have[:] = True

    def _reset(self, reason: str) -> None:
        """Restart the model state at the content not sent yet; jobs of the old state are ignored when they return."""
        self._epoch += 1
        self._state = self._new_state()
        self._inflight = None
        self._jobs.clear()
        self._out_pos = self._sent
        self._needs_reset = False
        self.stats["resets"] += 1
        AUDIO_PREPROC_STATE_RESETS.labels(backend=self._backend, reason=reason).inc()
        key = f"reset:{reason}"
        level = logging.DEBUG if key in self._logged_reasons else logging.WARNING
        self._logged_reasons.add(key)
        logger.log(
            level,
            "AUDIO_PREPROCESSING_STATE_RESET",
            extra=self._log_extra(reason=reason, stage="AUDIO_PREPROCESSING_STATE_RESET"),
        )

    def _mark(self, start: int, end: int, cause: str) -> None:
        if end <= start:
            return
        if self._bad and self._bad[-1][2] == cause and self._bad[-1][1] >= start:
            self._bad[-1][1] = max(self._bad[-1][1], end)
        else:
            self._bad.append([start, end, cause])

    # ── input and output ──────────────────────────────────────────────────────

    def _ingest(self, audio: np.ndarray) -> None:
        base = self._track.end
        self._track.append(audio)
        self._enh.append(np.zeros(len(audio), dtype=np.float32))
        self._total_in += len(audio)
        zero = audio == 0.0
        if not zero.any():
            self._open_run = 0
            return
        edges = np.diff(np.concatenate(([0], zero.view(np.int8), [0])))
        n = len(zero)
        open_run = 0
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            carried = self._open_run if s == 0 else 0
            length = int(e - s) + carried
            if length >= self._min_run:
                # a carried run that was already long enough is masked up to base
                first = base + int(s) - (carried if carried < self._min_run else 0)
                self._track.set_mask(first, base + int(e))
            if e == n:
                open_run = length
        self._open_run = open_run

    def _play(self, count: int, *, stream_end: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Content [emitted, emitted + count): enhanced where present (crossfaded), raw elsewhere, digital zero kept.

        stream_end: this call plays up to the last sample of the stream (flush). Nothing follows it, so an
        enhanced run that reaches the end is not faded out to raw; content still missing its enhanced samples
        (final job timed out) plays raw as always.
        """
        a, b = self._emitted, self._emitted + count
        raw, zero = self._track.view(a, b)
        raw, zero = raw.copy(), zero.copy()
        enh, have = self._enh.view(a, b)
        enh, have = enh.copy(), have.copy()
        if count == 0:
            return raw, raw.copy()
        x = self._x
        if x == 0:
            weight = have.astype(np.float32)
        else:
            idx = np.arange(count)
            # samples of the enhanced run up to and including each position (carried from the previous call)
            last_gap = np.maximum.accumulate(np.where(have, -1 - self._run, idx))
            back = idx - last_gap
            # samples of the run from each position on, looking x past the end (unknown counts as no run)
            ahead_end = min(b + x, self._enh.end)
            ahead = np.zeros(x, dtype=bool)
            if ahead_end > b:
                ahead[: ahead_end - b] = self._enh.view(b, ahead_end)[1]
            if stream_end:
                ahead[max(0, self._total_in - b) :] = True  # past the last sample: no switch to raw to fade into
            ext = np.concatenate((have, ahead))
            ext_idx = np.arange(ext.size)
            next_gap = np.minimum.accumulate(np.where(ext, ext.size, ext_idx)[::-1])[::-1]
            forward = next_gap[:count] - idx
            weight = (np.minimum(np.minimum(back, forward), x) / x).astype(np.float32)
            self._run = int(min(back[-1], x))
        out = raw + weight * (enh - raw)
        out[zero] = 0.0
        self._account(a, zero, have)
        return out.astype(np.float32, copy=False), raw

    def _account(self, a: int, zero: np.ndarray, have: np.ndarray) -> None:
        """Region outcomes (regions of hop samples) for the stats/metrics shared with the window preprocessor."""
        pos, b = a, a + len(zero)
        while pos < b:
            region_end = (pos // self._h + 1) * self._h
            seg_end = min(b, region_end)
            seg_zero, seg_have = zero[pos - a : seg_end - a], have[pos - a : seg_end - a]
            if not seg_zero.all():
                self._region_silent = False
            if (~seg_have & ~seg_zero).any():
                self._region_missing = True
            pos = seg_end
            if seg_end == region_end:
                self._close_region(region_end)

    def _close_region(self, end: int) -> None:
        start = self._region_start
        if self._region_silent:
            outcome = "silent"
        elif not self._region_missing:
            outcome = "enhanced"
        else:
            outcome = self._cause(start, end)
        self._record(outcome)
        self._region_start = end
        self._region_missing = False
        self._region_silent = True

    def _cause(self, start: int, end: int) -> str:
        while self._bad and self._bad[0][1] <= start:
            self._bad.popleft()
        for s, e, cause in self._bad:
            if s >= end:
                break
            if e > start:
                return cause
        return "late"

    def _record(self, outcome: str) -> None:
        self.stats[outcome] += 1
        AUDIO_PREPROC_REGIONS.labels(backend=self._backend, outcome=outcome).inc()
        reason = self._bad_outcomes.get(outcome)
        if reason is None:
            if outcome in ("enhanced", "silent"):
                self._consec_bad = 0
            return
        AUDIO_PREPROC_FALLBACK.labels(backend=self._backend, reason=reason).inc()
        level = logging.DEBUG if reason in self._logged_reasons else logging.WARNING
        self._logged_reasons.add(reason)
        logger.log(
            level,
            "AUDIO_PREPROCESSING_FALLBACK",
            extra=self._log_extra(reason=reason, stage="AUDIO_PREPROCESSING_FALLBACK"),
        )
        self._consec_bad += 1
        if self._consec_bad >= self._bad_limit and not self.bypassed:
            self._consec_bad = 0
            self._bypass_until = self._total_in + self._bypass_samples
            self.stats["bypass_events"] += 1
            AUDIO_PREPROC_BYPASS.labels(backend=self._backend).inc()
            logger.warning(
                "AUDIO_PREPROCESSING_BYPASS",
                extra=self._log_extra(reason=reason, stage="AUDIO_PREPROCESSING_BYPASS"),
            )

    def _trim(self) -> None:
        self._track.drop_before(min(self._sent, self._emitted))
        self._enh.drop_before(self._emitted)
        while self._bad and self._bad[0][1] <= self._region_start:
            self._bad.popleft()

    def _log_extra(self, **extra: object) -> dict[str, object]:
        return {
            "rpc_id": self.key.rpc_id,
            "session_id": self.key.session_id,
            "stream_id": self.key.stream_id,
            **extra,
        }
