import asyncio
import threading
import unittest
from unittest import mock

import numpy as np
from prometheus_client import REGISTRY

from nemotron_testbed.audio_preprocess import SAMPLE_RATE, StreamAudioPreprocessor
from nemotron_testbed.config import AudioPreprocessSettings
from nemotron_testbed.domain import (
    AudioPreprocessJob,
    AudioPreprocessResult,
    StreamKey,
    SubmitStatus,
)

KEY = StreamKey("rpc", "sess", "stream")
MS = SAMPLE_RATE // 1000


def make_settings(**overrides) -> AudioPreprocessSettings:
    base = dict(
        enabled=True,
        left_context_ms=100,
        hop_ms=100,
        right_context_ms=100,
        crossfade_ms=10,
        compute_budget_ms=50,
        dither_sigma_lsb=3.0,
        fallback_bad_regions=3,
        fallback_bypass_s=10.0,
    )
    base.update(overrides)
    return AudioPreprocessSettings(**base)


class FakePool:
    """Stands in for the runtime pool. Modes: inline (answer at once), hold, shed, unavailable."""

    def __init__(self, model=lambda w: w, mode="inline", only=None):
        self.model = model
        self.mode = mode
        self.only = only
        self.jobs: list[AudioPreprocessJob] = []
        self.held: list[AudioPreprocessJob] = []
        self.pre: StreamAudioPreprocessor | None = None

    def __call__(self, job: AudioPreprocessJob) -> SubmitStatus:
        self.jobs.append(job)
        if self.mode == "shed":
            return SubmitStatus.OVERLOADED
        if self.mode == "unavailable":
            return SubmitStatus.UNAVAILABLE
        if self.mode == "hold" or (self.only is not None and not self.only(job.region_id)):
            self.held.append(job)
            return SubmitStatus.ACCEPTED
        self.reply(job)
        return SubmitStatus.ACCEPTED

    def reply(self, job: AudioPreprocessJob, audio="model") -> None:
        if isinstance(audio, str):
            audio = self.model(job.window)
        self.pre.deliver(AudioPreprocessResult(key=job.key, region_id=job.region_id, audio=audio))


def build(pool: FakePool, **overrides) -> StreamAudioPreprocessor:
    pre = StreamAudioPreprocessor(KEY, make_settings(**overrides), pool)
    pool.pre = pre
    return pre


def speech(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.uniform(-0.5, 0.5, n) + 0.01).astype(np.float32)


def run(pre: StreamAudioPreprocessor, x: np.ndarray, chunks) -> tuple[np.ndarray, np.ndarray]:
    enh, raw, pos = [], [], 0
    for size in chunks:
        if pos >= len(x):
            break
        part = x[pos : pos + size]
        out = pre.process(part)
        assert len(out.enhanced) == len(part) and len(out.raw_aligned) == len(part)
        enh.append(out.enhanced)
        raw.append(out.raw_aligned)
        pos += len(part)
    return np.concatenate(enh), np.concatenate(raw)


def cyclic(sizes):
    while True:
        yield from sizes


def streams_active() -> float:
    return REGISTRY.get_sample_value("audio_preproc_streams_active", {"backend": "resemble"}) or 0.0


def histogram(name: str) -> tuple[float, float]:
    labels = {"backend": "resemble"}
    return (
        REGISTRY.get_sample_value(f"{name}_count", labels) or 0.0,
        REGISTRY.get_sample_value(f"{name}_sum", labels) or 0.0,
    )


class StreamPreprocessorTest(unittest.TestCase):
    def test_length_and_constant_delay_for_irregular_chunks(self):
        x = speech(30_000)
        pool = FakePool()
        pre = build(pool)
        delay = pre.delay_samples
        self.assertEqual(delay, (100 + 100 + 50) * MS)
        enh, raw = run(pre, x, cyclic([1280, 1, 333, 80, 4001, 17]))
        self.assertEqual(len(enh), len(x))
        np.testing.assert_array_equal(enh[:delay], 0)
        np.testing.assert_array_equal(raw[:delay], 0)
        np.testing.assert_allclose(enh[delay:], x[: len(x) - delay], atol=1e-6)
        np.testing.assert_array_equal(raw[delay:], x[: len(x) - delay])

    def test_output_does_not_depend_on_chunking(self):
        x = speech(24_000)
        outs = []
        for sizes in ([1280], [7, 1000, 3, 2222]):
            pre = build(FakePool(lambda w: w * 0.5))
            outs.append(run(pre, x, cyclic(sizes)))
        np.testing.assert_allclose(outs[0][0], outs[1][0], atol=1e-6)
        np.testing.assert_array_equal(outs[0][1], outs[1][1])

    def test_model_output_is_played_and_raw_is_untouched(self):
        x = speech(20_000)
        pre = build(FakePool(lambda w: w * 0.5))
        enh, raw = run(pre, x, cyclic([1280]))
        d = pre.delay_samples
        np.testing.assert_allclose(enh[d:], 0.5 * x[: len(x) - d], atol=1e-6)
        np.testing.assert_array_equal(raw[d:], x[: len(x) - d])
        self.assertEqual(pre.stats["late"] + pre.stats["shed"] + pre.stats["error"], 0)
        self.assertGreater(pre.stats["enhanced"], 5)

    def test_flush_releases_exactly_the_held_back_content(self):
        x = speech(20_000)
        pre = build(FakePool())
        enh, raw = run(pre, x, cyclic([1280]))
        tail = asyncio.run(pre.flush())
        d = pre.delay_samples
        self.assertEqual(len(tail.enhanced), d)
        np.testing.assert_allclose(tail.enhanced, x[-d:], atol=1e-6)
        np.testing.assert_array_equal(tail.raw_aligned, x[-d:])
        full = np.concatenate((enh, tail.enhanced))
        np.testing.assert_allclose(full[d:], x, atol=1e-6)

    def test_flush_of_stream_shorter_than_delay(self):
        x = speech(1280 * 3)
        pre = build(FakePool())
        enh, _ = run(pre, x, [1280, 1280, 1280])
        self.assertEqual(float(np.abs(enh).max()), 0.0)
        tail = asyncio.run(pre.flush())
        self.assertEqual(len(tail.enhanced), len(x))
        np.testing.assert_allclose(tail.enhanced, x, atol=1e-6)

    def test_process_after_flush_passes_raw_through(self):
        pre = build(FakePool())
        run(pre, speech(8000), [1280])
        asyncio.run(pre.flush())
        x = speech(500, seed=9)
        out = pre.process(x)
        np.testing.assert_array_equal(out.enhanced, x)
        np.testing.assert_array_equal(out.raw_aligned, x)

    def test_digital_zero_runs_are_dithered_for_the_model_and_rezeroed_after(self):
        x = speech(24_000)
        x[8200:9100] = 0.0  # long run: masked
        x[3000:3040] = 0.0  # short run: untouched
        pool = FakePool()
        pre = build(pool)
        enh, raw = run(pre, x, cyclic([1280]))
        d = pre.delay_samples
        windows = {job.region_id: job.window for job in pool.jobs}
        starts = {k: k * 100 * MS - 100 * MS for k in windows}
        seen_gap = False
        for k, window in windows.items():
            lo = starts[k]
            for pos in range(max(lo, 0), min(lo + len(window), len(x))):
                value = window[pos - lo]
                if 8200 <= pos < 9100:
                    seen_gap = True
                    self.assertLessEqual(abs(float(value)) * 32768.0, 20.0)
                elif pos < 24_000 - 1280:
                    self.assertEqual(float(value), float(x[pos]))
            if lo <= 8300 < lo + len(window):
                self.assertTrue(np.any(window[8200 - lo : 9000 - lo] != 0.0))
        self.assertTrue(seen_gap)
        np.testing.assert_array_equal(enh[8200 + d : 9100 + d], 0.0)
        np.testing.assert_array_equal(enh[3000 + d : 3040 + d], 0.0)
        np.testing.assert_array_equal(raw[8200 + d : 9100 + d], 0.0)

    def test_dither_is_deterministic_and_shared_by_overlapping_windows(self):
        x = speech(24_000)
        x[8200:9100] = 0.0
        pools = []
        for _ in range(2):
            pool = FakePool()
            run(build(pool), x, cyclic([1280, 640]))
            pools.append(pool)
        for a, b in zip(pools[0].jobs, pools[1].jobs):
            np.testing.assert_array_equal(a.window, b.window)
        by_id = {job.region_id: job.window for job in pools[0].jobs}
        overlap = (100 + 100) * MS
        shared = 0
        for k in by_id:
            if k + 1 in by_id:
                np.testing.assert_array_equal(by_id[k][-overlap:], by_id[k + 1][:overlap])
                shared += 1
        self.assertGreater(shared, 5)

    def test_dither_off_with_zero_sigma(self):
        x = speech(16_000)
        x[5000:6000] = 0.0
        pool = FakePool()
        run(build(pool, dither_sigma_lsb=0.0), x, cyclic([1280]))
        window = next(job.window for job in pool.jobs if job.region_id == 3)
        lo = 3 * 100 * MS - 100 * MS
        np.testing.assert_array_equal(window[5000 - lo : 6000 - lo], 0.0)

    def test_all_zero_regions_skip_the_model_and_stay_zero(self):
        x = np.concatenate((speech(6400), np.zeros(16_000, np.float32), speech(6400, seed=5)))
        pool = FakePool(lambda w: w + 1.0)
        pre = build(pool)
        enh, raw = run(pre, x, cyclic([1280]))
        self.assertGreater(pre.stats["silent"], 3)
        self.assertLess(len(pool.jobs), len(x) // (100 * MS))
        d = pre.delay_samples
        np.testing.assert_array_equal(enh[8000 + d : 20_000 + d - 1280], 0.0)

    def test_pure_silence_never_reaches_the_model(self):
        pool = FakePool()
        pre = build(pool)
        enh, _ = run(pre, np.zeros(16_000, np.float32), cyclic([1280]))
        self.assertEqual(pool.jobs, [])
        self.assertEqual(float(np.abs(enh).max()), 0.0)
        self.assertGreater(pre.stats["silent"], 0)

    def test_late_regions_play_raw_and_late_results_are_dropped(self):
        x = speech(24_000)
        pool = FakePool(mode="hold")
        pre = build(pool, fallback_bad_regions=100)
        enh, raw = run(pre, x, cyclic([1280]))
        d = pre.delay_samples
        np.testing.assert_allclose(enh[d:], x[: len(x) - d], atol=1e-6)
        self.assertGreater(pre.stats["late"], 3)
        self.assertEqual(pre.stats["enhanced"], 0)
        for job in pool.held:
            pool.reply(job, audio=job.window * 0.0)
        out = pre.process(speech(1280, seed=3))
        self.assertGreater(float(np.abs(out.enhanced).max()), 0.0)

    def test_late_result_arriving_in_time_is_used(self):
        x = speech(16_000)
        pool = FakePool(lambda w: w * 0.5, mode="hold")
        pre = build(pool, fallback_bad_regions=100, compute_budget_ms=200)
        outs, pos = [], 0
        while pos < len(x):
            outs.append(pre.process(x[pos : pos + 1280]).enhanced)
            pos += 1280
            while pool.held:
                pool.reply(pool.held.pop(0))
        enh = np.concatenate(outs)
        d = pre.delay_samples
        np.testing.assert_allclose(enh[d:], 0.5 * x[: len(x) - d], atol=1e-6)

    def test_results_in_any_order(self):
        x = speech(20_000)
        pool = FakePool(lambda w: w * 0.5, mode="hold")
        pre = build(pool, fallback_bad_regions=100, compute_budget_ms=400)
        outs, pos = [], 0
        while pos < len(x):
            outs.append(pre.process(x[pos : pos + 1280]).enhanced)
            pos += 1280
            if len(pool.held) >= 2:
                pool.reply(pool.held.pop())
                pool.reply(pool.held.pop())
        for job in pool.held:
            pool.reply(job)
        self.assertEqual(sum(len(o) for o in outs), len(x))
        self.assertGreater(pre.stats["enhanced"], 0)

    def test_shed_and_unavailable_play_raw(self):
        for mode, key in (("shed", "shed"), ("unavailable", "error")):
            x = speech(16_000)
            pre = build(FakePool(mode=mode), fallback_bad_regions=100)
            enh, _ = run(pre, x, cyclic([1280]))
            d = pre.delay_samples
            np.testing.assert_allclose(enh[d:], x[: len(x) - d], atol=1e-6)
            self.assertGreater(pre.stats[key], 2, mode)

    def test_model_error_result_plays_raw(self):
        x = speech(16_000)
        pool = FakePool()
        pre = build(pool, fallback_bad_regions=100)
        pool.reply = lambda job, audio=None: pre.deliver(
            AudioPreprocessResult(key=job.key, region_id=job.region_id, error="boom")
        )
        enh, _ = run(pre, x, cyclic([1280]))
        d = pre.delay_samples
        np.testing.assert_allclose(enh[d:], x[: len(x) - d], atol=1e-6)
        self.assertGreater(pre.stats["error"], 2)

    def test_bypass_after_consecutive_bad_regions_then_recovers(self):
        x = speech(64_000)
        pool = FakePool()
        state = {"n": 0}
        inline = pool.__call__

        def flaky(job):
            state["n"] += 1
            if state["n"] <= 3:
                pool.jobs.append(job)
                return SubmitStatus.OVERLOADED
            return inline(job)

        pre = StreamAudioPreprocessor(
            KEY, make_settings(fallback_bad_regions=2, fallback_bypass_s=0.5), flaky
        )
        pool.pre = pre
        enh, _ = run(pre, x, cyclic([1280]))
        d = pre.delay_samples
        self.assertEqual(pre.stats["bypass_events"], 1)
        self.assertGreater(pre.stats["bypass"], 0)
        self.assertGreater(pre.stats["enhanced"], 3)
        np.testing.assert_allclose(enh[d:], x[: len(x) - d], atol=1e-6)

    def test_bypass_until_end_of_stream_when_seconds_is_zero(self):
        x = speech(64_000)
        pre = build(FakePool(mode="shed"), fallback_bad_regions=2, fallback_bypass_s=0.0)
        run(pre, x, cyclic([1280]))
        self.assertTrue(pre.bypassed)
        self.assertEqual(pre.stats["bypass_events"], 1)
        self.assertEqual(pre.stats["enhanced"], 0)

    def test_flush_times_out_on_missing_results_and_returns_raw(self):
        x = speech(12_000)
        pool = FakePool(mode="hold")
        pre = build(pool, fallback_bad_regions=100)
        enh, _ = run(pre, x, cyclic([1280]))
        tail = asyncio.run(pre.flush(timeout_s=0.05))
        d = pre.delay_samples
        np.testing.assert_allclose(tail.enhanced, x[-d:], atol=1e-6)

    def test_flush_picks_up_results_that_arrive_while_waiting(self):
        x = speech(12_000)
        pool = FakePool(lambda w: w * 0.5, mode="hold")
        pre = build(pool, fallback_bad_regions=100)
        run(pre, x, cyclic([1280]))

        async def scenario():
            flushing = asyncio.ensure_future(pre.flush(timeout_s=2.0))
            await asyncio.sleep(0.02)
            for job in list(pool.held):
                threading.Thread(target=pool.reply, args=(job,)).start()
            return await flushing

        tail = asyncio.run(scenario())
        d = pre.delay_samples
        self.assertGreater(float(np.abs(tail.enhanced - x[-d:]).max()), 0.01)

    def test_crossfade_between_enhanced_and_raw_regions_is_a_linear_ramp(self):
        x = np.full(16_000, 0.2, dtype=np.float32)
        pool = FakePool(lambda w: w * 2.0, only=lambda k: k % 2 == 0)
        pre = build(pool, fallback_bad_regions=100, crossfade_ms=10)
        enh, _ = run(pre, x, cyclic([1280]))
        d, h, xf = pre.delay_samples, 100 * MS, 10 * MS
        # region 1 (raw) follows region 0 (enhanced): ramp 0.4 -> 0.2 over the first xf samples
        seg = enh[d + h : d + h + xf]
        self.assertTrue(np.all(np.diff(seg) < 0))
        self.assertGreater(seg[0], 0.39)
        self.assertLess(seg[-1], 0.21)
        np.testing.assert_allclose(enh[d + h + xf : d + 2 * h], 0.2, atol=1e-6)
        np.testing.assert_allclose(enh[d : d + h], 0.4, atol=1e-6)

    def test_close_is_idempotent_and_drops_late_results(self):
        before = streams_active()
        pool = FakePool(mode="hold")
        pre = build(pool)
        self.assertEqual(streams_active(), before + 1)
        run(pre, speech(8000), cyclic([1280]))
        pre.close()
        pre.close()
        self.assertEqual(streams_active(), before)
        pool.reply(pool.held[0])
        self.assertEqual(pre._inbox, {})
        x = speech(100)
        np.testing.assert_array_equal(pre.process(x).enhanced, x)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            StreamAudioPreprocessor(KEY, make_settings(right_context_ms=10, crossfade_ms=10), FakePool())
        with self.assertRaises(ValueError):
            StreamAudioPreprocessor(KEY, make_settings(hop_ms=20, crossfade_ms=30), FakePool())
        self.assertEqual(streams_active(), streams_active())

    def test_submit_failure_degrades_to_raw(self):
        def broken(job):
            raise RuntimeError("pool exploded")

        x = speech(16_000)
        pre = StreamAudioPreprocessor(KEY, make_settings(fallback_bad_regions=100), broken)
        enh, _ = run(pre, x, cyclic([1280]))
        d = pre.delay_samples
        np.testing.assert_allclose(enh[d:], x[: len(x) - d], atol=1e-6)
        self.assertGreater(pre.stats["error"], 2)

    def test_memory_is_bounded_on_a_long_stream(self):
        pre = build(FakePool())
        run(pre, speech(SAMPLE_RATE * 60), cyclic([1280]))
        self.assertLess(pre._track.end - pre._track.start, SAMPLE_RATE * 3)
        self.assertLessEqual(len(pre._track._data), 1 << 18)

    def test_unreleased_raw_is_exactly_the_held_back_content(self):
        x = speech(20_000)
        pre = build(FakePool(lambda w: w * 0.5))
        run(pre, x, cyclic([1280, 333]))
        self.assertEqual(pre.samples_received, len(x))
        held = pre.unreleased_raw()
        self.assertEqual(len(held), pre.delay_samples)
        np.testing.assert_array_equal(held, x[-pre.delay_samples :])

    def test_unreleased_raw_upto_excludes_the_chunk_whose_process_failed(self):
        x = speech(20_000)
        pre = build(FakePool())
        run(pre, x[:15_000], cyclic([1280]))
        before = pre.samples_received
        with mock.patch.object(pre, "_submit_ready", side_effect=RuntimeError("bug")):
            with self.assertRaises(RuntimeError):
                pre.process(x[15_000:])
        self.assertEqual(pre.samples_received, len(x))
        held = pre.unreleased_raw(before)
        self.assertEqual(len(held), pre.delay_samples)
        np.testing.assert_array_equal(held, x[before - pre.delay_samples : before])

    def test_unreleased_raw_of_stream_shorter_than_delay_is_everything(self):
        x = speech(1280 * 3)
        pre = build(FakePool())
        run(pre, x, [1280, 1280, 1280])
        np.testing.assert_array_equal(pre.unreleased_raw(), x)
        self.assertEqual(len(pre.unreleased_raw(1000)), 1000)

    def test_unreleased_raw_is_empty_when_nothing_is_held_and_never_raises(self):
        pre = build(FakePool())
        self.assertEqual(len(pre.unreleased_raw()), 0)
        run(pre, speech(8000), cyclic([1280]))
        asyncio.run(pre.flush())
        self.assertEqual(len(pre.unreleased_raw()), 0)
        pre._track = None
        self.assertEqual(len(pre.unreleased_raw()), 0)


class TimingMetricsTest(unittest.TestCase):
    """Chunks of one hop (1600 samples) at t = 0.1 * i: region k is submitted after chunk k + 1 and needed at chunk k + 2."""

    def setUp(self):
        self.t = [0.0]
        self.pool = FakePool(mode="hold")
        self.pre = build(self.pool, fallback_bad_regions=10_000)
        self.pre._clock = lambda: self.t[0]

    def feed(self, chunks: int, reply_after_s: float | None = None) -> None:
        for i in range(chunks):
            self.t[0] = 0.1 * i
            self.pre.process(speech(1600, seed=i))
            if reply_after_s is not None:
                self.t[0] += reply_after_s
                while self.pool.held:
                    self.pool.reply(self.pool.held.pop(0))

    def held_by_region(self) -> dict[int, AudioPreprocessJob]:
        return {job.region_id: job for job in self.pool.held}

    def test_slack_is_needed_minus_arrival_of_enhanced_regions(self):
        count0, sum0 = histogram("audio_preproc_slack_seconds")
        late0 = histogram("audio_preproc_late_seconds")[0]
        self.feed(12, reply_after_s=0.02)
        count, total = histogram("audio_preproc_slack_seconds")
        enhanced = self.pre.stats["enhanced"]
        self.assertGreater(enhanced, 5)
        self.assertEqual(count - count0, enhanced)
        self.assertAlmostEqual((total - sum0) / enhanced, 0.08, places=6)
        self.assertEqual(histogram("audio_preproc_late_seconds")[0], late0)

    def test_late_is_arrival_minus_the_moment_the_region_was_needed(self):
        count0, sum0 = histogram("audio_preproc_late_seconds")
        slack0 = histogram("audio_preproc_slack_seconds")[0]
        self.feed(12)
        self.assertEqual(self.pre.stats["enhanced"], 0)
        held = self.held_by_region()
        self.t[0] = 2.0
        self.pool.reply(held[3])  # needed at 0.5 s
        self.pool.reply(held[6])  # needed at 0.8 s
        count, total = histogram("audio_preproc_late_seconds")
        self.assertEqual(count - count0, 2)
        self.assertAlmostEqual(total - sum0, 1.5 + 1.2, places=6)
        self.assertEqual(histogram("audio_preproc_slack_seconds")[0], slack0)

    def test_late_error_duplicate_and_not_yet_needed_results_are_not_counted(self):
        count0 = histogram("audio_preproc_late_seconds")[0]
        self.feed(12)
        held = self.held_by_region()
        self.t[0] = 2.0
        self.pre.deliver(AudioPreprocessResult(key=KEY, region_id=8, error="boom"))
        self.pool.reply(held[8])  # already consumed by the error
        self.pool.reply(held[10])  # region 10 is still in the future
        self.assertEqual(histogram("audio_preproc_late_seconds")[0], count0)
        self.pool.reply(held[2])
        self.pool.reply(held[2])
        self.assertEqual(histogram("audio_preproc_late_seconds")[0], count0 + 1)

    def test_flush_time_regions_are_not_measured(self):
        self.feed(6)
        tracked = set(self.pre._late_since)
        self.assertEqual(tracked, {0, 1, 2, 3})
        slack0 = histogram("audio_preproc_slack_seconds")[0]
        late0 = histogram("audio_preproc_late_seconds")[0]
        asyncio.run(self.pre.flush(timeout_s=0.02))
        self.assertEqual(set(self.pre._late_since), tracked)
        self.t[0] = 5.0
        for job in list(self.pool.held):
            self.pool.reply(job)
        self.assertEqual(histogram("audio_preproc_late_seconds")[0], late0 + len(tracked))
        self.assertEqual(histogram("audio_preproc_slack_seconds")[0], slack0)

    def test_flushed_enhanced_regions_have_no_slack(self):
        pool = FakePool()
        pre = build(pool, fallback_bad_regions=10_000)
        pre._clock = lambda: self.t[0]
        for i in range(6):
            self.t[0] = 0.1 * i
            pre.process(speech(1600, seed=i))
        slack0, _ = histogram("audio_preproc_slack_seconds")
        enhanced0 = pre.stats["enhanced"]
        asyncio.run(pre.flush())
        self.assertGreater(pre.stats["enhanced"], enhanced0)
        self.assertEqual(histogram("audio_preproc_slack_seconds")[0], slack0)

    def test_tracking_tables_are_bounded_and_cleared_on_close(self):
        self.feed(120)
        self.assertGreater(self.pre.stats["late"], 100)
        self.assertLessEqual(len(self.pre._late_since), 64)
        self.assertLessEqual(len(self.pre._arrived), 1)
        count0 = histogram("audio_preproc_late_seconds")[0]
        evicted = self.held_by_region()[0]
        self.pool.reply(evicted)
        self.assertEqual(histogram("audio_preproc_late_seconds")[0], count0)
        self.pre.close()
        self.assertEqual(self.pre._late_since, {})
        self.assertEqual(self.pre._arrived, {})
        self.pool.reply(self.held_by_region()[100])
        self.assertEqual(histogram("audio_preproc_late_seconds")[0], count0)


if __name__ == "__main__":
    unittest.main()
