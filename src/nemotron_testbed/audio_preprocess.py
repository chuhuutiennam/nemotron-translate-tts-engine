from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AudioPreprocessSettings
from .domain import AudioPreprocessJob, AudioPreprocessResult, StreamKey, SubmitStatus
from .metrics import (
    AUDIO_PREPROC_BACKEND,
    AUDIO_PREPROC_BYPASS,
    AUDIO_PREPROC_FALLBACK,
    AUDIO_PREPROC_FLUSH_SECONDS,
    AUDIO_PREPROC_LATE_SECONDS,
    AUDIO_PREPROC_REGIONS,
    AUDIO_PREPROC_SLACK_SECONDS,
    AUDIO_PREPROC_STREAMS_ACTIVE,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_LSB = 1.0 / 32768.0
_DITHER_BLOCK = 4096
_DITHER_BLOCK_OFFSET = 1 << 12
_BAD_OUTCOMES = {"late": "DENOISER_LATE", "shed": "DENOISER_QUEUE_FULL", "error": "DENOISER_ERROR"}
_LATE_TRACK_REGIONS = 64

SubmitFn = Callable[[AudioPreprocessJob], SubmitStatus]


@dataclass(slots=True)
class PreprocessResult:
    enhanced: np.ndarray
    raw_aligned: np.ndarray


class _Track:
    """Append-only audio + zero-run mask addressed by absolute sample position; the consumed prefix is dropped lazily."""

    def __init__(self, capacity: int = 1 << 16) -> None:
        self._data = np.zeros(capacity, dtype=np.float32)
        self._mask = np.zeros(capacity, dtype=bool)
        self._lo = 0
        self._hi = 0
        self.start = 0

    @property
    def end(self) -> int:
        return self.start + (self._hi - self._lo)

    def append(self, audio: np.ndarray) -> None:
        n = len(audio)
        if self._hi + n > len(self._data):
            self._compact(n)
        self._data[self._hi : self._hi + n] = audio
        self._mask[self._hi : self._hi + n] = False
        self._hi += n

    def _compact(self, extra: int) -> None:
        size = self._hi - self._lo
        if size + extra > len(self._data):
            cap = max(len(self._data) * 2, size + extra)
            data = np.zeros(cap, dtype=np.float32)
            mask = np.zeros(cap, dtype=bool)
            data[:size] = self._data[self._lo : self._hi]
            mask[:size] = self._mask[self._lo : self._hi]
            self._data, self._mask = data, mask
        else:
            self._data[:size] = self._data[self._lo : self._hi]
            self._mask[:size] = self._mask[self._lo : self._hi]
        self._lo, self._hi = 0, size

    def view(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        i = self._lo + (a - self.start)
        j = self._lo + (b - self.start)
        return self._data[i:j], self._mask[i:j]

    def set_mask(self, a: int, b: int) -> None:
        a = max(a, self.start)
        if b > a:
            self._mask[self._lo + (a - self.start) : self._lo + (b - self.start)] = True

    def drop_before(self, pos: int) -> None:
        pos = min(pos, self.end)
        if pos > self.start:
            self._lo += pos - self.start
            self.start = pos


class StreamAudioPreprocessor:
    """Sliding-window denoiser/enhancer for one continuous audio stream (one StreamKey).

    Content position c is played at output position c + D, D = hop + right_context + budget, constant.
    Region k = [k*H, (k+1)*H) comes from window [k*H - Cl, (k+1)*H + L), submitted as soon as it is
    complete and used if its result is there when region k is first needed; otherwise the region plays
    raw audio. process() never waits for the model. Invariants: len(out) == len(in) for both outputs,
    D is constant, the first D output samples are zeros.
    """

    def __init__(
        self,
        key: StreamKey,
        settings: AudioPreprocessSettings,
        submit: SubmitFn,
    ) -> None:
        ms = SAMPLE_RATE // 1000
        self.key = key
        self._submit_fn = submit
        self._cl = settings.left_context_ms * ms
        self._h = settings.hop_ms * ms
        self._l = settings.right_context_ms * ms
        self._x = settings.crossfade_ms * ms
        self._delay = self._h + self._l + settings.compute_budget_ms * ms
        self._window_len = self._cl + self._h + self._l
        self._min_run = settings.dither_min_zero_run
        self._sigma = settings.dither_sigma_lsb
        self._seed = settings.dither_seed & 0xFFFFFFFF
        self._bad_limit = settings.fallback_bad_regions
        self._bypass_samples = (
            math.inf if settings.fallback_bypass_s <= 0 else int(settings.fallback_bypass_s * SAMPLE_RATE)
        )
        self.flush_timeout_s = max(1.0, 2.0 * settings.compute_budget_ms / 1000.0)
        if self._h <= 0 or self._cl < 0 or self._x < 0 or self._x > self._h:
            raise ValueError("invalid audio preprocessing window")
        if self._l < self._x + self._min_run:
            raise ValueError("right context must exceed crossfade by at least dither_min_zero_run samples")
        self._xfade = ((np.arange(self._x, dtype=np.float32) + 0.5) / self._x) if self._x else None

        self._track = _Track()
        self._open_run = 0
        self._total_in = 0
        self._emitted = 0
        self._next_submit = 0
        self._next_materialize = 0
        self._cur: tuple[int, np.ndarray] | None = None
        self._prev_tail: np.ndarray | None = None
        self._plan: dict[int, str] = {}
        self._inbox: dict[int, AudioPreprocessResult] = {}
        self._arrived: dict[int, float] = {}
        self._late_since: dict[int, float] = {}
        self._clock: Callable[[], float] = time.monotonic
        self._lock = threading.Lock()
        self._noise_cache: dict[int, np.ndarray] = {}
        self._consec_bad = 0
        self._bypass_until: float = 0
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
        }
        AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=AUDIO_PREPROC_BACKEND).inc()

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
        """Called from a worker thread; results for regions already played are dropped (their lateness is recorded)."""
        late_s = None
        with self._lock:
            if self._closed:
                return
            now = self._clock()
            if result.region_id < self._next_materialize:
                since = self._late_since.pop(result.region_id, None)
                if since is not None and result.audio is not None:
                    late_s = now - since
            else:
                self._inbox[result.region_id] = result
                self._arrived[result.region_id] = now
        if late_s is not None:
            AUDIO_PREPROC_LATE_SECONDS.labels(backend=AUDIO_PREPROC_BACKEND).observe(late_s)

    def process(self, audio: np.ndarray) -> PreprocessResult:
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if self._closed or self._flushed:
            return PreprocessResult(audio.copy(), audio.copy())
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return PreprocessResult(empty, empty.copy())
        self._ingest(audio)
        self._submit_ready()
        count = max(0, self._total_in - self._delay) - self._emitted
        enhanced, raw = self._take_content(count)
        self._emitted += count
        self.stats["samples_in"] += n
        self.stats["samples_out"] += n
        pad = np.zeros(n - count, dtype=np.float32)
        result = PreprocessResult(np.concatenate((pad, enhanced)), np.concatenate((pad, raw)))
        self._track.drop_before(min(max(self._next_submit * self._h - self._cl, 0), self._emitted))
        return result

    async def flush(self, timeout_s: float | None = None) -> PreprocessResult:
        """End of stream: pad the last windows on the right and release the content still held back."""
        empty = np.zeros(0, dtype=np.float32)
        if self._closed or self._flushed:
            return PreprocessResult(empty, empty.copy())
        self._flushed = True
        remaining = self._total_in - self._emitted
        if remaining <= 0:
            return PreprocessResult(empty, empty.copy())
        started = time.monotonic()
        last_region = (self._total_in - 1) // self._h
        while self._next_submit <= last_region:
            self._submit(self._next_submit)
            self._next_submit += 1
        deadline = started + (self.flush_timeout_s if timeout_s is None else timeout_s)
        while self._awaiting(last_region) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        enhanced, raw = self._take_content(remaining)
        self._emitted += remaining
        self.stats["samples_out"] += remaining
        AUDIO_PREPROC_FLUSH_SECONDS.labels(backend=AUDIO_PREPROC_BACKEND).observe(time.monotonic() - started)
        return PreprocessResult(enhanced, raw)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._inbox.clear()
            self._arrived.clear()
            self._late_since.clear()
        self._plan.clear()
        AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=AUDIO_PREPROC_BACKEND).dec()
        logger.info("AUDIO_PREPROCESSING_CLOSE", extra=self._log_extra(stats=dict(self.stats)))

    def _log_extra(self, **extra: object) -> dict[str, object]:
        return {
            "rpc_id": self.key.rpc_id,
            "session_id": self.key.session_id,
            "stream_id": self.key.stream_id,
            **extra,
        }

    def _awaiting(self, last_region: int) -> bool:
        with self._lock:
            return any(
                self._plan.get(k) == "job" and k not in self._inbox
                for k in range(self._next_materialize, last_region + 1)
            )

    def _ingest(self, audio: np.ndarray) -> None:
        base = self._track.end
        self._track.append(audio)
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

    def _cut(self, a: int, b: int, dither: bool) -> tuple[np.ndarray, np.ndarray]:
        """Samples [a, b); positions outside the stream read as masked digital zero. Always a copy."""
        audio = np.zeros(b - a, dtype=np.float32)
        mask = np.ones(b - a, dtype=bool)
        s = max(a, 0, self._track.start)
        e = min(b, self._track.end)
        if e > s:
            data, flags = self._track.view(s, e)
            audio[s - a : e - a] = data
            mask[s - a : e - a] = flags
        if dither and self._sigma > 0.0:
            self._apply_dither(a, audio, mask)
        return audio, mask

    def _apply_dither(self, a: int, audio: np.ndarray, mask: np.ndarray) -> None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        pos = idx + a
        block = pos // _DITHER_BLOCK
        offset = pos - block * _DITHER_BLOCK
        for b in np.unique(block):
            sel = block == b
            audio[idx[sel]] = self._block_noise(int(b))[offset[sel]]

    def _block_noise(self, block: int) -> np.ndarray:
        noise = self._noise_cache.get(block)
        if noise is None:
            if len(self._noise_cache) > 64:
                self._noise_cache.clear()
            rng = np.random.default_rng([self._seed, block + _DITHER_BLOCK_OFFSET])
            noise = (np.rint(rng.normal(0.0, self._sigma, _DITHER_BLOCK)) * _LSB).astype(np.float32)
            self._noise_cache[block] = noise
        return noise

    def _submit_ready(self) -> None:
        while (self._next_submit + 1) * self._h + self._l <= self._total_in:
            self._submit(self._next_submit)
            self._next_submit += 1

    def _submit(self, k: int) -> None:
        try:
            self._plan[k] = self._plan_region(k)
        except Exception:
            logger.exception("Audio preprocessing window submit failed", extra=self._log_extra())
            self._plan[k] = "error"

    def _plan_region(self, k: int) -> str:
        if self.bypassed:
            return "bypass"
        a = k * self._h
        _, mask = self._cut(a, a + self._h + self._x, dither=False)
        if mask.all():
            return "silent"
        window, _ = self._cut(a - self._cl, a + self._h + self._l, dither=True)
        status = self._submit_fn(AudioPreprocessJob(key=self.key, region_id=k, window=window))
        if status is SubmitStatus.ACCEPTED:
            return "job"
        return "shed" if status is SubmitStatus.OVERLOADED else "error"

    def _take_content(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        enhanced = np.empty(count, dtype=np.float32)
        pos = 0
        while pos < count:
            c = self._emitted + pos
            k = c // self._h
            if self._cur is None or self._cur[0] != k:
                self._cur = (k, self._materialize(k))
            offset = c - k * self._h
            take = min(count - pos, self._h - offset)
            enhanced[pos : pos + take] = self._cur[1][offset : offset + take]
            pos += take
        raw = self._track.view(self._emitted, self._emitted + count)[0].copy()
        return enhanced, raw

    def _materialize(self, k: int) -> np.ndarray:
        h, x = self._h, self._x
        raw, mask = self._cut(k * h, k * h + h + x, dither=False)
        with self._lock:
            needed = self._clock()
            self._next_materialize = k + 1
            plan = self._plan.pop(k, "bypass")
            result = self._inbox.pop(k, None)
            arrived = self._arrived.pop(k, None)
            # flush-time regions have an artificial deadline, so they are not measured
            if plan == "job" and result is None and not self._flushed:
                self._late_since[k] = needed
                while len(self._late_since) > _LATE_TRACK_REGIONS:
                    del self._late_since[next(iter(self._late_since))]
        body = raw
        outcome = plan
        if plan == "job":
            if result is None:
                outcome = "late"
            elif result.audio is None:
                outcome = "error"
            else:
                try:
                    body = result.audio[self._cl : self._cl + h + x].copy()
                    body[mask] = 0.0
                    outcome = "enhanced"
                    if arrived is not None and not self._flushed:
                        AUDIO_PREPROC_SLACK_SECONDS.labels(backend=AUDIO_PREPROC_BACKEND).observe(
                            max(0.0, needed - arrived)
                        )
                except Exception:
                    logger.exception("Audio preprocessing region assembly failed", extra=self._log_extra())
                    body, outcome = raw, "error"
        self._record(k, outcome)
        out = body[:h].copy()
        if self._xfade is not None and self._prev_tail is not None:
            out[:x] = (1.0 - self._xfade) * self._prev_tail + self._xfade * body[:x]
        self._prev_tail = body[h : h + x].copy()
        return out

    def _record(self, k: int, outcome: str) -> None:
        self.stats[outcome] += 1
        AUDIO_PREPROC_REGIONS.labels(backend=AUDIO_PREPROC_BACKEND, outcome=outcome).inc()
        reason = _BAD_OUTCOMES.get(outcome)
        if reason is None:
            if outcome in ("enhanced", "silent"):
                self._consec_bad = 0
            return
        AUDIO_PREPROC_FALLBACK.labels(backend=AUDIO_PREPROC_BACKEND, reason=reason).inc()
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
            AUDIO_PREPROC_BYPASS.labels(backend=AUDIO_PREPROC_BACKEND).inc()
            logger.warning(
                "AUDIO_PREPROCESSING_BYPASS",
                extra=self._log_extra(reason=reason, stage="AUDIO_PREPROCESSING_BYPASS"),
            )
