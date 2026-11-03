import asyncio
import unittest

import numpy as np
from prometheus_client import REGISTRY

from nemotron_testbed.audio_preprocess_stateful import MAX_BACKLOG_S, StatefulStreamPreprocessor
from nemotron_testbed.config import AudioPreprocessSettings
from nemotron_testbed.domain import AudioPreprocessJob, AudioPreprocessResult, StreamKey, SubmitStatus

KEY = StreamKey("rpc", "sess", "stream")
MS = 16
GAIN = 0.5
LAG = 1000  # samples the fake model holds back, like the AudX look-ahead


def make_settings(**overrides) -> AudioPreprocessSettings:
    base = dict(
        enabled=True,
        mode="audx",
        left_context_ms=0,
        hop_ms=80,
        right_context_ms=64,
        crossfade_ms=0,
        compute_budget_ms=100,
        fallback_bad_regions=3,
        fallback_bypass_s=10.0,
    )
    base.update(overrides)
    return AudioPreprocessSettings(**base)


class FakeState:
    def __init__(self):
        self.held = np.zeros(0, dtype=np.float32)
        self.received = 0
        self.finished = False


class FakeStreamModel:
    """step(): returns its input times GAIN, held back by LAG samples until final (like AudxModel.step)."""

    def __init__(self, gain=GAIN, offset=0.0, lag=LAG):
        self.gain, self.offset, self.lag = gain, offset, lag

    def step(self, state, samples, final=False):
        assert not state.finished
        state.held = np.concatenate((state.held, samples * self.gain + self.offset)).astype(np.float32)
        state.received += len(samples)
        keep = 0 if final else min(self.lag, state.held.size)
        out, state.held = state.held[: state.held.size - keep], state.held[state.held.size - keep :]
        state.finished = final
        return out


class FakePool:
    """Modes: inline (answer at once), hold (answer when told), shed, unavailable."""

    def __init__(self, model=None, mode="inline"):
        self.model = model or FakeStreamModel()
        self.mode = mode
        self.jobs: list[AudioPreprocessJob] = []
        self.held: list[AudioPreprocessJob] = []
        self.pre: StatefulStreamPreprocessor | None = None

    def __call__(self, job):
        self.jobs.append(job)
        if self.mode == "shed":
            return SubmitStatus.OVERLOADED
        if self.mode == "unavailable":
            return SubmitStatus.UNAVAILABLE
        if self.mode == "hold":
            self.held.append(job)
            return SubmitStatus.ACCEPTED
        self.reply(job)
        return SubmitStatus.ACCEPTED

    def reply(self, job, audio="model"):
        if isinstance(audio, str):
            audio = self.model.step(job.state, job.window, job.final)
        self.pre.deliver(AudioPreprocessResult(key=job.key, region_id=job.region_id, audio=audio))

    def release(self):
        held, self.held = self.held, []
        for job in held:
            self.reply(job)


def build(pool, **overrides):
    pre = StatefulStreamPreprocessor(KEY, make_settings(**overrides), pool, new_state=FakeState)
    pool.pre = pre
    return pre


def speech(n, seed=1):
    return (np.random.default_rng(seed).uniform(-0.5, 0.5, n) + 0.01).astype(np.float32)


def run(pre, x, size=320):
    enh, raw = [], []
    for pos in range(0, len(x), size):
        part = x[pos : pos + size]
        out = pre.process(part)
        assert len(out.enhanced) == len(part) == len(out.raw_aligned)
        enh.append(out.enhanced)
        raw.append(out.raw_aligned)
    return np.concatenate(enh), np.concatenate(raw)


def finish(pre, timeout_s=0.5):
    out = asyncio.run(pre.flush(timeout_s))
    return out.enhanced, out.raw_aligned


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, {"backend": "audx", **labels}) or 0.0


class StatefulPreprocessorTest(unittest.TestCase):
    def test_delay_is_hop_plus_lookahead_plus_budget(self):
        pre = build(FakePool())
        self.assertEqual(pre.delay_samples, (80 + 64 + 100) * MS)

    def test_enhanced_stream_with_constant_delay_and_all_samples_on_flush(self):
        x = speech(16_000)
        pool = FakePool()
        pre = build(pool)
        d = pre.delay_samples
        enh, raw = run(pre, x, 320)
        np.testing.assert_array_equal(enh[:d], 0.0)
        np.testing.assert_array_equal(raw[:d], 0.0)
        np.testing.assert_allclose(enh[d:], GAIN * x[: x.size - d], atol=1e-7)
        np.testing.assert_array_equal(raw[d:], x[: x.size - d])
        tail_enh, tail_raw = finish(pre)
        np.testing.assert_allclose(tail_enh, GAIN * x[x.size - d :], atol=1e-7)
        np.testing.assert_array_equal(tail_raw, x[x.size - d :])
        self.assertTrue(pool.jobs[-1].final)
        self.assertEqual(pre.stats["late"] + pre.stats["error"] + pre.stats["shed"], 0)
        self.assertEqual(pre.stats["resets"], 0)

    def test_jobs_cover_the_input_once_in_order_with_one_state(self):
        x = speech(9_000)
        pool = FakePool()
        pre = build(pool)
        sizes = [1, 333, 4001, 80, 17, 1280]
        pos = i = 0
        while pos < x.size:
            pre.process(x[pos : pos + sizes[i % len(sizes)]])
            pos += sizes[i % len(sizes)]
            i += 1
        finish(pre)
        np.testing.assert_array_equal(np.concatenate([job.window for job in pool.jobs]), x)
        self.assertEqual(len({id(job.state) for job in pool.jobs}), 1)
        self.assertEqual([job.region_id for job in pool.jobs], list(range(len(pool.jobs))))
        self.assertTrue(all(job.window.size >= 80 * MS for job in pool.jobs[:-1]))

    def test_one_job_in_flight_and_the_next_carries_everything_waiting(self):
        pool = FakePool(mode="hold")
        pre = build(pool)
        x = speech(320 * 10)
        run(pre, x[: 320 * 4], 320)  # 80 ms: first job
        self.assertEqual(len(pool.jobs), 1)
        run(pre, x[320 * 4 :], 320)  # more audio while the job is in flight
        self.assertEqual(len(pool.jobs), 1)
        pool.release()
        pre.process(x[:1])  # next chunk: result drained, next job sent with all waiting audio
        self.assertEqual(len(pool.jobs), 2)
        self.assertEqual(pool.jobs[1].window.size, 320 * 6 + 1)

    def test_late_job_plays_raw_and_keeps_the_state(self):
        x = speech(16_000)
        pool = FakePool(mode="hold")
        pre = build(pool, fallback_bad_regions=100)
        d = pre.delay_samples
        enh, _ = run(pre, x[:8_000], 320)  # nothing answered: content played so far is raw
        np.testing.assert_array_equal(enh[d:], x[: 8_000 - d])
        pool.mode = "inline"
        pool.release()
        enh2, _ = run(pre, x[8_000:], 320)
        # the state went on (no reset) and catches up: the end of the stream is enhanced again
        self.assertEqual(pre.stats["resets"], 0)
        self.assertGreater(pre.stats["late"], 0)
        np.testing.assert_allclose(enh2[-1000:], GAIN * x[16_000 - d - 1000 : 16_000 - d], atol=1e-7)
        self.assertEqual(len({id(job.state) for job in pool.jobs}), 1)

    def test_error_restarts_the_state_and_recovers(self):
        x = speech(24_000)
        pool = FakePool()
        pre = build(pool, fallback_bad_regions=100)
        resets = sample("audio_preproc_state_resets_total", reason="error")
        run(pre, x[:8_000], 320)
        pool.mode = "hold"
        run(pre, x[8_000:9_600], 320)
        job = pool.held.pop()
        pool.reply(job, audio=None)  # the model failed
        pool.mode = "inline"
        enh, _ = run(pre, x[9_600:], 320)
        d = pre.delay_samples
        self.assertEqual(pre.stats["resets"], 1)
        self.assertEqual(sample("audio_preproc_state_resets_total", reason="error") - resets, 1)
        self.assertGreater(pre.stats["error"], 0)
        self.assertGreater(len({id(job.state) for job in pool.jobs}), 1)
        np.testing.assert_allclose(enh[-1000:], GAIN * x[24_000 - d - 1000 : 24_000 - d], atol=1e-7)

    def test_too_many_samples_from_the_model_is_an_error(self):
        pool = FakePool(mode="hold")
        pre = build(pool)
        run(pre, speech(1280), 320)
        pool.reply(pool.held.pop(), audio=np.ones(5000, dtype=np.float32))
        pre.process(speech(320))
        self.assertEqual(pre.stats["resets"], 1)

    def test_full_queue_keeps_the_audio_then_gives_up_after_the_backlog_limit(self):
        pool = FakePool(mode="shed")
        pre = build(pool, fallback_bad_regions=1000)
        x = speech(int(MAX_BACKLOG_S * 16_000) + 3_200)
        run(pre, x[:4_000], 320)
        self.assertGreater(len(pool.jobs), 1)
        self.assertEqual(pool.jobs[-1].window.size, 4_000)  # everything since the start, retried
        self.assertEqual(pre.stats["resets"], 0)
        run(pre, x[4_000:], 320)
        self.assertEqual(pre.stats["resets"], 1)
        given_up_at = 16_160  # first chunk end (4_000 + k * 320) with more than MAX_BACKLOG_S waiting
        pool.mode = "inline"
        pre.process(speech(1280))
        # the given-up backlog is not sent again: only what came after it
        self.assertEqual(pool.jobs[-1].window.size, x.size - given_up_at + 1280)
        finish(pre)
        self.assertGreater(pre.stats["shed"], 0)

    def test_unavailable_pool_restarts_the_state_as_error(self):
        pool = FakePool(mode="unavailable")
        pre = build(pool, fallback_bad_regions=1000)
        run(pre, speech(12_000), 320)
        self.assertGreaterEqual(pre.stats["resets"], 1)
        self.assertGreater(pre.stats["error"], 0)

    def test_bad_regions_switch_to_bypass_then_a_fresh_state(self):
        pool = FakePool(mode="hold")
        pre = build(pool, fallback_bad_regions=3, fallback_bypass_s=0.5)
        x = speech(48_000)
        run(pre, x[:12_000], 320)
        self.assertEqual(pre.stats["bypass_events"], 1)
        jobs_at_bypass = len(pool.jobs)
        pool.mode = "inline"
        pool.held.clear()
        run(pre, x[12_000:16_000], 320)  # inside the bypass: nothing sent
        self.assertEqual(len(pool.jobs), jobs_at_bypass)
        enh, _ = run(pre, x[16_000:], 320)
        d = pre.delay_samples
        self.assertGreater(len(pool.jobs), jobs_at_bypass)
        self.assertIsNot(pool.jobs[-1].state, pool.jobs[0].state)
        self.assertGreater(pre.stats["bypass"], 0)
        np.testing.assert_allclose(enh[-1000:], GAIN * x[48_000 - d - 1000 : 48_000 - d], atol=1e-7)

    def test_digital_zero_runs_stay_zero(self):
        pool = FakePool(model=FakeStreamModel(offset=0.01))
        pre = build(pool)
        x = speech(16_000)
        x[4_000:6_000] = 0.0
        x[9_000:9_010] = 0.0  # too short to count as a zero run
        enh, _ = run(pre, x, 320)
        tail, _ = finish(pre)
        out = np.concatenate((enh, tail))[pre.delay_samples :]
        np.testing.assert_array_equal(out[4_000:6_000], 0.0)
        np.testing.assert_allclose(out[9_000:9_010], 0.01, atol=1e-7)

    def test_crossfade_ramps_between_enhanced_and_raw(self):
        x = np.full(32_000, 0.4, dtype=np.float32)
        pool = FakePool()
        pre = build(pool, crossfade_ms=10, fallback_bad_regions=100)
        x_ms = 10 * MS
        first, _ = run(pre, x[:8_000], 320)
        pool.mode = "hold"
        middle, _ = run(pre, x[8_000:16_000], 320)  # enhanced stops, raw takes over
        pool.mode = "inline"
        pool.release()
        last, _ = run(pre, x[16_000:], 320)
        out = np.concatenate((first, middle, last))[pre.delay_samples + 1 :]  # after the stream start
        self.assertTrue(np.any(np.isclose(out, 0.4)) and np.any(np.isclose(out, 0.2)))
        steps = np.abs(np.diff(out))
        # GAIN 0.5: enhanced 0.2, raw 0.4; a ramp over x_ms samples moves at most 0.2 / x_ms per sample
        self.assertLessEqual(float(steps.max()), 0.2 / x_ms + 1e-6)

    def test_enhanced_from_the_first_to_the_last_sample(self):
        x = np.full(8_000, 0.4, dtype=np.float32)
        pre = build(FakePool(), crossfade_ms=10)
        enh, _ = run(pre, x, 320)
        tail, _ = finish(pre)
        out = np.concatenate((enh, tail))[pre.delay_samples :]
        self.assertEqual(out.size, x.size)
        # GAIN 0.5 -> enhanced 0.2: no fade in from raw at the stream start, no fade out at the end
        np.testing.assert_allclose(out, 0.2, atol=1e-7)

    def test_late_start_plays_raw_then_fades_in(self):
        x = np.full(16_000, 0.4, dtype=np.float32)
        pool = FakePool(mode="hold")
        pre = build(pool, crossfade_ms=10, fallback_bad_regions=100)
        first, _ = run(pre, x[:6_000], 320)  # first job held: the stream start has no enhanced audio yet
        pool.mode = "inline"
        pool.release()
        rest, _ = run(pre, x[6_000:], 320)
        out = np.concatenate((first, rest))[pre.delay_samples :]
        self.assertAlmostEqual(float(out[0]), 0.4, places=6)  # raw while nothing came back
        self.assertAlmostEqual(float(out[-1]), 0.2, places=6)
        self.assertLessEqual(float(np.abs(np.diff(out)).max()), 0.2 / (10 * MS) + 1e-6)  # faded, no jump

    def test_flush_still_plays_raw_where_the_final_job_did_not_come_back(self):
        x = np.full(8_000, 0.4, dtype=np.float32)
        pool = FakePool()
        pre = build(pool, crossfade_ms=10)
        enh, _ = run(pre, x, 320)
        pool.mode = "hold"  # the final job never answers
        tail, _ = finish(pre, timeout_s=0.05)
        out = np.concatenate((enh, tail))[pre.delay_samples :]
        self.assertEqual(out.size, x.size)
        missing_from = pre._out_pos  # enhanced samples stop here: the model still held LAG samples
        np.testing.assert_allclose(out[missing_from:], 0.4, atol=1e-7)
        np.testing.assert_allclose(out[: missing_from - 10 * MS], 0.2, atol=1e-7)
        # fade out over the last 10 ms of the enhanced run, no jump
        self.assertLessEqual(float(np.abs(np.diff(out)).max()), 0.2 / (10 * MS) + 1e-6)

    def test_flush_timeout_plays_the_rest_raw(self):
        pool = FakePool(mode="hold")
        pre = build(pool)
        x = speech(4_000)
        run(pre, x, 320)
        enh, raw = finish(pre, timeout_s=0.05)
        np.testing.assert_array_equal(enh, raw)
        self.assertEqual(enh.size, pre.delay_samples)

    def test_close_ignores_late_results_and_updates_the_gauge(self):
        before = sample("audio_preproc_streams_active")
        pool = FakePool(mode="hold")
        pre = build(pool)
        self.assertEqual(sample("audio_preproc_streams_active") - before, 1)
        run(pre, speech(1280), 320)
        pre.close()
        pool.release()
        self.assertEqual(sample("audio_preproc_streams_active") - before, 0)
        out = pre.process(speech(10))
        np.testing.assert_array_equal(out.enhanced, out.raw_aligned)

    def test_fallback_reasons_are_the_audx_codes(self):
        def fallback(reason):
            return sample("audio_preproc_fallback_total", reason=reason)

        codes = ("AUDX_LATE", "AUDX_QUEUE_FULL", "AUDX_ERROR")
        before = {code: fallback(code) for code in codes}
        resemble_before = REGISTRY.get_sample_value(
            "audio_preproc_fallback_total", {"backend": "resemble", "reason": "DENOISER_LATE"}
        ) or 0.0
        for mode, expected in (("hold", "AUDX_LATE"), ("shed", "AUDX_QUEUE_FULL"), ("unavailable", "AUDX_ERROR")):
            with self.subTest(mode=mode):
                pre = build(FakePool(mode=mode), fallback_bad_regions=1000)
                with self.assertLogs("nemotron_testbed.audio_preprocess_stateful", level="DEBUG") as cm:
                    run(pre, speech(12_000), 320)
                self.assertGreater(fallback(expected) - before[expected], 0)
                reasons = {getattr(r, "reason", None) for r in cm.records} - {None}
                self.assertIn(expected, reasons)
                self.assertFalse(any(str(r).startswith("DENOISER_") for r in reasons))
        after_resemble = REGISTRY.get_sample_value(
            "audio_preproc_fallback_total", {"backend": "resemble", "reason": "DENOISER_LATE"}
        ) or 0.0
        self.assertEqual(after_resemble, resemble_before)

    def test_unreleased_raw_is_the_content_not_played_yet(self):
        pre = build(FakePool())
        x = speech(5_000)
        run(pre, x, 500)
        np.testing.assert_array_equal(pre.unreleased_raw(), x[5_000 - pre.delay_samples :])
        np.testing.assert_array_equal(pre.unreleased_raw(4_000), x[5_000 - pre.delay_samples : 4_000])


if __name__ == "__main__":
    unittest.main()
