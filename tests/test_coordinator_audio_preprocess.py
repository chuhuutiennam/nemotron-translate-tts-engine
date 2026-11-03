import asyncio
import dataclasses
import time
import unittest
from unittest import mock

import numpy as np

from nemotron_testbed.audio_preprocess_stateful import StatefulStreamPreprocessor
from nemotron_testbed.audx_model import AudxStreamState
from nemotron_testbed.config import AudioPreprocessSettings
from nemotron_testbed.coordinator import EngineCoordinator
from nemotron_testbed.domain import AudioPreprocessResult, SubmitStatus
from nemotron_testbed.metrics import (
    AUDIO_PREPROC_BACKEND,
    AUDIO_PREPROC_BACKEND_AUDX,
    AUDIO_PREPROC_POOL_WORKERS,
    AUDIO_PREPROC_SHED_STREAMS,
    AUDIO_PREPROC_STREAMS_ACTIVE,
)
from nemotron_testbed.stats_logger import _labeled_counter, _labeled_gauge
from nemotron_testbed.vad import VadEvent
from test_coordinator_flow import (
    FakeRuntime,
    FakeTranslationRuntime,
    FakeTtsRuntime,
    _TailVad,
    make_settings,
    request,
)

LOGGER = "nemotron_testbed.coordinator"
AMP = 1000 / 32768.0  # one test chunk is 320 samples of int16 1000
GAIN = 0.5
DELAY = 1600  # hop 40 + right context 40 + budget 20 ms at 16 kHz

SMALL_WINDOW = dict(
    left_context_ms=20,
    hop_ms=40,
    right_context_ms=40,
    crossfade_ms=10,
    compute_budget_ms=20,
    dither_sigma_lsb=0.0,
)


class FakePreprocPool:
    """Answers every window immediately with window * gain, like an infinitely fast model."""

    def __init__(self, status=SubmitStatus.ACCEPTED, gain=GAIN):
        self.coordinator = None
        self.status = status
        self.gain = gain
        self.jobs = []

    def try_submit_nowait(self, job):
        self.jobs.append(job)
        if self.status is SubmitStatus.ACCEPTED:
            self.coordinator._on_audio_preprocess_result_from_thread(
                AudioPreprocessResult(key=job.key, region_id=job.region_id, audio=job.window * self.gain)
            )
        return self.status

    def start(self, timeout):
        pass

    def shutdown(self):
        pass


class StartableRuntime(FakeRuntime):
    def start(self, timeout):
        pass

    def shutdown(self):
        pass


class SpyVad(_TailVad):
    def __init__(self, events=(), end_tail_ms=0.0):
        super().__init__(events, end_tail_ms=end_tail_ms)
        self.audio = []

    def process(self, audio, duration_ms):
        self.audio.append(audio.copy())
        return super().process(audio, duration_ms)


def patched_vad(vad):
    return mock.patch("nemotron_testbed.coordinator.build_vad_controller", return_value=vad)


def ramp(chunks: int) -> np.ndarray:
    """Content of `chunks` ramp requests: sample i has value i + 1 (int16 scale), so every position is unique."""
    return (np.arange(1, chunks * 320 + 1) / 32768.0).astype(np.float32)


def ramp_request(index, end=False, **fields):
    req = request(index * 160, end=end, **fields)
    req.pcm = np.arange(index * 320 + 1, (index + 1) * 320 + 1, dtype="<i2").tobytes()
    return req


def preproc_streams() -> float:
    return _labeled_gauge(AUDIO_PREPROC_STREAMS_ACTIVE).get(AUDIO_PREPROC_BACKEND, 0.0)


def shed_streams() -> float:
    return _labeled_counter(AUDIO_PREPROC_SHED_STREAMS, "backend").get(AUDIO_PREPROC_BACKEND, 0.0)


class AudioPreprocessCoordinatorTest(unittest.IsolatedAsyncioTestCase):
    def _coordinator(
        self, *, enabled=True, status=SubmitStatus.ACCEPTED, translate=False, settings_fields=None, **preproc_fields
    ):
        preproc = dataclasses.replace(
            AudioPreprocessSettings(), enabled=enabled, **{**SMALL_WINDOW, **preproc_fields}
        )
        settings = dataclasses.replace(
            make_settings(translate), audio_preprocessing=preproc, **(settings_fields or {})
        )
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        if translate:
            coordinator.translation_runtime = FakeTranslationRuntime()
            coordinator.tts_runtime = FakeTtsRuntime()
        if enabled:
            pool = FakePreprocPool(status)
            pool.coordinator = coordinator
            coordinator.audio_preproc_pool = pool
        return coordinator

    @staticmethod
    async def _feed(coordinator, rpc, count, *, end=False, **fields):
        for index in range(count):
            await coordinator.accept_chunk(
                rpc.rpc_id, request(index * 160, end=end and index == count - 1, **fields)
            )

    @staticmethod
    async def _feed_ramp(coordinator, rpc, first, count, **fields):
        for index in range(first, first + count):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(index, **fields))

    @staticmethod
    def _stream(coordinator):
        return next(iter(coordinator._streams.values()))

    def _utterance(self, coordinator):
        stream = self._stream(coordinator)
        return stream.active_utterance or coordinator._finishing[stream.key][1]

    def _model_audio(self, coordinator):
        return self._utterance(coordinator).model_audio.to_array()

    async def test_disabled_keeps_the_raw_flow(self):
        coordinator = self._coordinator(enabled=False)
        calls = []
        original = coordinator._process_audio

        async def spy(**kwargs):
            calls.append(kwargs)
            return await original(**kwargs)

        coordinator._process_audio = spy
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 3)

        stream = self._stream(coordinator)
        self.assertIsNone(coordinator.audio_preproc_pool)
        self.assertIsNone(stream.preprocessor)
        self.assertIsNone(stream.raw_pre_roll)
        self.assertEqual(coordinator._preprocessors, {})
        self.assertTrue(calls)
        self.assertTrue(all(call["raw_audio"] is None for call in calls))
        np.testing.assert_allclose(self._model_audio(coordinator), np.full(960, AMP), atol=1e-7)

    async def test_enabled_delays_the_stream_and_feeds_enhanced_audio(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 10)

        stream = self._stream(coordinator)
        self.assertIsNotNone(stream.preprocessor)
        self.assertIsNotNone(stream.raw_pre_roll)
        audio = self._model_audio(coordinator)
        self.assertEqual(audio.size, 3200)
        np.testing.assert_array_equal(audio[:DELAY], 0.0)
        np.testing.assert_allclose(audio[DELAY:], GAIN * AMP, atol=1e-7)

    async def _emitted_delays(self, coordinator, rpc, key):
        now = time.monotonic()
        base = dict(key=key, utterance_id=1, text="x", is_final=True, ts_start=0, ts_end=1, language="vi-VN")
        coordinator._emit_recognition(**base, audio_payload=b"\x00" * 160, speech_end_wall=now)
        coordinator._emit_recognition(**base, audio_payload=b"\x00" * 160)
        coordinator._emit_recognition(**base)
        items = [await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2) for _ in range(3)]
        return [item.preproc_delay_s for item in items]

    async def test_emitted_audio_carries_the_preprocessing_delay_only_with_speech_end(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 3)

        delays = await self._emitted_delays(coordinator, rpc, self._stream(coordinator).key)
        self.assertEqual(delays, [DELAY / 16_000, 0.0, 0.0])

    async def test_emitted_audio_has_no_delay_without_a_preprocessor(self):
        coordinator = self._coordinator(enabled=False)
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 3)

        delays = await self._emitted_delays(coordinator, rpc, self._stream(coordinator).key)
        self.assertEqual(delays, [0.0, 0.0, 0.0])

    async def test_emitted_audio_has_no_delay_after_the_preprocessor_is_detached(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 3)
        key = self._stream(coordinator).key
        await coordinator.close_input(rpc.rpc_id)
        self.assertIsNone(self._stream(coordinator).preprocessor)

        delays = await self._emitted_delays(coordinator, rpc, key)
        self.assertEqual(delays, [0.0, 0.0, 0.0])

    async def test_translation_passthrough_audio_is_the_raw_delayed_signal(self):
        coordinator = self._coordinator(translate=True)
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 10, task="realtime_translation", language="en")

        utt = self._stream(coordinator).active_utterance
        source = np.concatenate(utt.source_audio)
        model = utt.model_audio.to_array()
        self.assertEqual(source.size, model.size)
        np.testing.assert_array_equal(source[:DELAY], 0.0)
        np.testing.assert_allclose(source[DELAY:], AMP, atol=1e-7)
        np.testing.assert_allclose(model[DELAY:], GAIN * AMP, atol=1e-7)

    async def test_vad_input_selects_raw_or_enhanced_audio(self):
        for vad_input, level in (("enhanced", GAIN * AMP), ("raw", AMP)):
            with self.subTest(vad_input=vad_input):
                coordinator = self._coordinator(vad_input=vad_input)
                vad = SpyVad()
                with patched_vad(vad):
                    rpc = coordinator.open_rpc()
                    await self._feed(coordinator, rpc, 10)
                seen = np.concatenate(vad.audio)
                self.assertEqual(seen.size, 3200)
                np.testing.assert_array_equal(seen[:DELAY], 0.0)
                np.testing.assert_allclose(seen[DELAY:], level, atol=1e-7)

    async def test_vad_and_asr_see_the_content_delayed_by_exactly_d_sample_for_sample(self):
        def delayed(level):
            return np.concatenate((np.zeros(DELAY), level * ramp(10)[: 3200 - DELAY]))

        for vad_input, vad_level in (("enhanced", GAIN), ("raw", 1.0)):
            with self.subTest(vad_input=vad_input):
                coordinator = self._coordinator(vad_input=vad_input, translate=True)
                vad = SpyVad([VadEvent.START])
                with patched_vad(vad):
                    rpc = coordinator.open_rpc()
                    await self._feed_ramp(coordinator, rpc, 0, 10, task="realtime_translation", language="en")
                utt = self._stream(coordinator).active_utterance
                np.testing.assert_allclose(np.concatenate(vad.audio), delayed(vad_level), atol=1e-7)
                np.testing.assert_allclose(utt.model_audio.to_array(), delayed(GAIN), atol=1e-7)
                np.testing.assert_allclose(np.concatenate(utt.source_audio), delayed(1.0), atol=1e-7)

    async def test_failure_mid_stream_plays_the_held_back_raw_audio_before_the_next_chunk(self):
        coordinator = self._coordinator()
        vad = SpyVad([VadEvent.START])
        with patched_vad(vad):
            rpc = coordinator.open_rpc()
            await self._feed_ramp(coordinator, rpc, 0, 6)
            stream = self._stream(coordinator)

            def boom(audio):
                raise RuntimeError("bug in preprocessor")

            stream.preprocessor.process = boom
            with self.assertLogs(LOGGER, level="ERROR") as cm:
                await self._feed_ramp(coordinator, rpc, 6, 2)

        self.assertIsNone(stream.preprocessor)
        failures = [r for r in cm.records if hasattr(r, "reason")]
        self.assertEqual([r.salvaged_ms for r in failures], [DELAY / 16])
        # 6 chunks went through the delay line (zeros, then the first 320 enhanced samples); the 1600 samples it
        # still held come out raw, followed by the two chunks sent after the failure: nothing lost, nothing repeated.
        expected = np.concatenate((np.zeros(DELAY), GAIN * ramp(1), ramp(8)[320:]))
        self.assertEqual(expected.size, 8 * 320 + DELAY)
        utt = stream.active_utterance
        np.testing.assert_allclose(utt.model_audio.to_array(), expected, atol=1e-7)
        np.testing.assert_allclose(np.concatenate(vad.audio), expected, atol=1e-7)
        self.assertEqual(utt.ts_end, 7 * 160 + 160)

    async def test_flush_failure_still_releases_the_held_back_raw_audio(self):
        coordinator = self._coordinator()
        with patched_vad(SpyVad([VadEvent.START])):
            rpc = coordinator.open_rpc()
            await self._feed_ramp(coordinator, rpc, 0, 10)
            stream = self._stream(coordinator)
            with mock.patch.object(stream.preprocessor, "flush", side_effect=RuntimeError("bug")):
                with self.assertLogs(LOGGER, level="ERROR") as cm:
                    await coordinator.close_input(rpc.rpc_id)

        self.assertIsNone(stream.preprocessor)
        self.assertEqual([r.salvaged_ms for r in cm.records if hasattr(r, "reason")], [DELAY / 16])
        audio = self._model_audio(coordinator)
        self.assertEqual(audio.size, 3200 + DELAY)
        np.testing.assert_allclose(audio[:3200], np.concatenate((np.zeros(DELAY), GAIN * ramp(10)[:DELAY])), atol=1e-7)
        np.testing.assert_allclose(audio[3200:], ramp(10)[3200 - DELAY :], atol=1e-7)

    async def test_vad_silence_warning_does_not_count_the_padding_zeros(self):
        coordinator = self._coordinator(settings_fields=dict(vad_no_start_warn_ms=100))
        with patched_vad(SpyVad()):
            rpc = coordinator.open_rpc()
            with self.assertLogs(LOGGER, level="DEBUG") as cm:
                await self._feed_ramp(coordinator, rpc, 0, 9)
                early = [r for r in cm.records if hasattr(r, "vad_max_score")]
                await self._feed_ramp(coordinator, rpc, 9, 1)
        # 9 chunks = 180 ms fed, of which 100 ms are the delay line's zeros: only 80 ms of real audio
        self.assertEqual(early, [])
        warnings = [r for r in cm.records if hasattr(r, "vad_max_score")]
        self.assertEqual([r.audio_ms for r in warnings], [100.0])

    async def test_pre_roll_and_raw_pre_roll_stay_in_lockstep(self):
        coordinator = self._coordinator()
        with patched_vad(SpyVad()):
            rpc = coordinator.open_rpc()
            await self._feed(coordinator, rpc, 10)
        stream = self._stream(coordinator)
        enhanced = stream.pre_roll.to_array()
        raw = stream.raw_pre_roll.to_array()
        self.assertEqual(enhanced.size, raw.size)
        np.testing.assert_allclose(enhanced[DELAY:], GAIN * AMP, atol=1e-7)
        np.testing.assert_allclose(raw[DELAY:], AMP, atol=1e-7)

    async def test_vad_end_tail_splits_raw_and_enhanced_together(self):
        coordinator = self._coordinator(translate=True)
        vad = SpyVad([VadEvent.START] + [VadEvent.NONE] * 7 + [VadEvent.END, VadEvent.NONE], end_tail_ms=5.0)
        with patched_vad(vad):
            rpc = coordinator.open_rpc()
            await self._feed(coordinator, rpc, 9, task="realtime_translation", language="en")

        stream = self._stream(coordinator)
        self.assertIsNone(stream.active_utterance)
        utt = coordinator._finishing[stream.key][1]
        source = np.concatenate(utt.source_audio)
        model = utt.model_audio.to_array()
        self.assertEqual(source.size, model.size)
        self.assertEqual(model.size, 8 * 320 + 240)
        np.testing.assert_allclose(source[-240:], AMP, atol=1e-7)
        np.testing.assert_allclose(model[-240:], GAIN * AMP, atol=1e-7)
        self.assertEqual(stream.pre_roll.samples, 80)
        self.assertEqual(stream.raw_pre_roll.samples, 80)
        np.testing.assert_allclose(stream.pre_roll.to_array(), GAIN * AMP, atol=1e-7)
        np.testing.assert_allclose(stream.raw_pre_roll.to_array(), AMP, atol=1e-7)

    async def test_sample_percent_gates_attachment(self):
        cases = ((0.3, 50, True), (0.7, 50, False), (0.0, 0, False), (0.99, 100, True))
        for roll, percent, attached in cases:
            with self.subTest(roll=roll, percent=percent):
                coordinator = self._coordinator(sample_percent=percent)
                with mock.patch("nemotron_testbed.coordinator.random.random", return_value=roll):
                    rpc = coordinator.open_rpc()
                    await self._feed(coordinator, rpc, 1)
                stream = self._stream(coordinator)
                self.assertEqual(stream.preprocessor is not None, attached)
                self.assertEqual(stream.raw_pre_roll is not None, attached)
                await coordinator.cancel_rpc(rpc.rpc_id, "done")

    async def test_max_streams_sheds_extra_streams_to_raw(self):
        coordinator = self._coordinator(max_streams=1)
        shed_before = shed_streams()
        first, second = coordinator.open_rpc(), coordinator.open_rpc()
        await coordinator.accept_chunk(first.rpc_id, request(0))
        other = request(0)
        other.session_id = "call-2"
        with self.assertLogs(LOGGER, level="WARNING") as cm:
            await coordinator.accept_chunk(second.rpc_id, other)

        streams = list(coordinator._streams.values())
        self.assertEqual([s.preprocessor is not None for s in streams], [True, False])
        self.assertEqual(len(coordinator._preprocessors), 1)
        self.assertEqual(shed_streams() - shed_before, 1)
        shed = [r for r in cm.records if getattr(r, "reason", "") == "DENOISER_MAX_STREAMS"]
        self.assertEqual(len(shed), 1)
        await coordinator.cancel_rpc(first.rpc_id, "done")
        await coordinator.cancel_rpc(second.rpc_id, "done")

    async def test_queue_full_regions_play_raw_audio(self):
        coordinator = self._coordinator(status=SubmitStatus.OVERLOADED)
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 10)

        audio = self._model_audio(coordinator)
        np.testing.assert_array_equal(audio[:DELAY], 0.0)
        np.testing.assert_allclose(audio[DELAY:], AMP, atol=1e-7)
        self.assertGreater(self._stream(coordinator).preprocessor.stats["shed"], 0)

    async def test_end_of_stream_flushes_the_held_back_audio(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        before = preproc_streams()
        await self._feed(coordinator, rpc, 10, end=True)

        stream = self._stream(coordinator)
        audio = self._model_audio(coordinator)
        self.assertEqual(audio.size, 3200 + DELAY)
        np.testing.assert_array_equal(audio[:DELAY], 0.0)
        np.testing.assert_allclose(audio[DELAY:], GAIN * AMP, atol=1e-7)
        self.assertTrue(stream.close_after_stream)
        self.assertIsNone(stream.preprocessor)
        self.assertEqual(coordinator._preprocessors, {})
        self.assertEqual(preproc_streams(), before)

    async def test_close_input_flushes_the_held_back_audio(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 10)
        await coordinator.close_input(rpc.rpc_id)

        stream = self._stream(coordinator)
        audio = self._model_audio(coordinator)
        self.assertEqual(audio.size, 3200 + DELAY)
        np.testing.assert_allclose(audio[DELAY:], GAIN * AMP, atol=1e-7)
        self.assertIsNone(stream.preprocessor)
        self.assertEqual(coordinator._preprocessors, {})

    async def test_cancel_rpc_detaches_the_preprocessor(self):
        coordinator = self._coordinator()
        base = preproc_streams()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 3)
        self.assertEqual(preproc_streams(), base + 1)
        self.assertEqual(len(coordinator._preprocessors), 1)

        await coordinator.cancel_rpc(rpc.rpc_id, "client gone")
        self.assertEqual(coordinator._preprocessors, {})
        self.assertEqual(preproc_streams(), base)

    async def test_late_result_for_a_closed_stream_is_ignored(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 1)
        key = self._stream(coordinator).key
        await coordinator.cancel_rpc(rpc.rpc_id, "client gone")
        coordinator._on_audio_preprocess_result_from_thread(
            AudioPreprocessResult(key=key, region_id=0, audio=np.zeros(10, dtype=np.float32))
        )

    async def test_preprocessor_exception_degrades_to_raw_and_detaches(self):
        coordinator = self._coordinator()
        base = preproc_streams()
        rpc = coordinator.open_rpc()
        await self._feed(coordinator, rpc, 3)
        stream = self._stream(coordinator)

        def boom(audio):
            raise RuntimeError("bug in preprocessor")

        stream.preprocessor.process = boom
        with self.assertLogs(LOGGER, level="ERROR") as cm:
            await coordinator.accept_chunk(rpc.rpc_id, request(3 * 160))

        self.assertIsNone(stream.preprocessor)
        self.assertEqual(coordinator._preprocessors, {})
        self.assertEqual(preproc_streams(), base)
        self.assertEqual([r.reason for r in cm.records if hasattr(r, "reason")], ["DENOISER_ERROR"])
        audio = self._model_audio(coordinator)
        # 960 delay-line zeros, the 960 samples it still held (raw), then the chunk sent after the failure
        self.assertEqual(audio.size, 960 + 960 + 320)
        np.testing.assert_array_equal(audio[:960], 0.0)
        np.testing.assert_allclose(audio[960:], AMP, atol=1e-7)

        await coordinator.accept_chunk(rpc.rpc_id, request(4 * 160))
        audio = self._model_audio(coordinator)
        self.assertEqual(audio.size, 960 + 960 + 640)
        np.testing.assert_allclose(audio[-320:], AMP, atol=1e-7)

    async def test_pool_start_failure_runs_without_preprocessing(self):
        coordinator = self._coordinator()
        coordinator.runtime = StartableRuntime()

        class BrokenPool(FakePreprocPool):
            def start(self, timeout):
                raise RuntimeError("denoiser.onnx not found")

        coordinator.audio_preproc_pool = BrokenPool()
        with self.assertLogs(LOGGER, level="ERROR") as cm:
            await coordinator.start()
        try:
            self.assertIsNone(coordinator.audio_preproc_pool)
            self.assertEqual(
                [r.reason for r in cm.records if hasattr(r, "reason")], ["DENOISER_START_FAILED"]
            )
            self.assertEqual(
                _labeled_gauge(AUDIO_PREPROC_POOL_WORKERS).get(AUDIO_PREPROC_BACKEND), 0.0
            )

            rpc = coordinator.open_rpc()
            await self._feed(coordinator, rpc, 3)
            stream = self._stream(coordinator)
            self.assertIsNone(stream.preprocessor)
            np.testing.assert_allclose(self._model_audio(coordinator), np.full(960, AMP), atol=1e-7)
        finally:
            await coordinator.shutdown()



class FakeStreamPool:
    """Answers stream jobs at once like a model without look-ahead: input * gain; checks one job in flight."""

    def __init__(self, gain=GAIN):
        self.coordinator = None
        self.gain = gain
        self.jobs = []

    def try_submit_nowait(self, job):
        self.jobs.append(job)
        self.coordinator._on_audio_preprocess_result_from_thread(
            AudioPreprocessResult(key=job.key, region_id=job.region_id, audio=job.window * self.gain)
        )
        return SubmitStatus.ACCEPTED

    def start(self, timeout):
        pass

    def shutdown(self):
        pass


AUDX_DELAY = (80 + 64 + 100) * 16


class AudxCoordinatorTest(unittest.IsolatedAsyncioTestCase):
    def _coordinator(self, **preproc_fields):
        preproc = dataclasses.replace(
            AudioPreprocessSettings(),
            enabled=True,
            mode="audx",
            left_context_ms=0,
            hop_ms=80,
            right_context_ms=64,
            crossfade_ms=0,
            compute_budget_ms=100,
            **preproc_fields,
        )
        settings = dataclasses.replace(make_settings(False), audio_preprocessing=preproc)
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        pool = FakeStreamPool()
        pool.coordinator = coordinator
        coordinator.audio_preproc_pool = pool
        return coordinator, pool

    @staticmethod
    def _audx_streams() -> float:
        return _labeled_gauge(AUDIO_PREPROC_STREAMS_ACTIVE).get(AUDIO_PREPROC_BACKEND_AUDX, 0.0)

    async def test_audx_mode_runs_the_stateful_preprocessor_with_one_state_per_stream(self):
        coordinator, pool = self._coordinator()
        before = self._audx_streams()
        rpc = coordinator.open_rpc()
        for index in range(20):
            await coordinator.accept_chunk(rpc.rpc_id, request(index * 160))

        stream = next(iter(coordinator._streams.values()))
        self.assertIsInstance(stream.preprocessor, StatefulStreamPreprocessor)
        self.assertEqual(stream.preprocessor.delay_samples, AUDX_DELAY)
        self.assertEqual(self._audx_streams() - before, 1)
        utt = stream.active_utterance or coordinator._finishing[stream.key][1]
        audio = utt.model_audio.to_array()
        self.assertEqual(audio.size, 20 * 320)
        np.testing.assert_array_equal(audio[:AUDX_DELAY], 0.0)
        np.testing.assert_allclose(audio[AUDX_DELAY:], GAIN * AMP, atol=1e-7)
        self.assertTrue(pool.jobs)
        self.assertTrue(all(isinstance(job.state, AudxStreamState) for job in pool.jobs))
        self.assertEqual(len({id(job.state) for job in pool.jobs}), 1)

        await coordinator.close_input(rpc.rpc_id)
        self.assertIsNone(stream.preprocessor)
        self.assertTrue(pool.jobs[-1].final)
        self.assertEqual(sum(job.window.size for job in pool.jobs), 20 * 320)
        self.assertEqual(self._audx_streams() - before, 0)


    async def test_audx_logs_its_own_reason_codes(self):
        # max_streams reached -> AUDX_MAX_STREAMS
        coordinator, _ = self._coordinator(max_streams=1)
        first, second = coordinator.open_rpc(), coordinator.open_rpc()
        await coordinator.accept_chunk(first.rpc_id, request(0))
        other = request(0)
        other.session_id = "call-2"
        with self.assertLogs(LOGGER, level="WARNING") as cm:
            await coordinator.accept_chunk(second.rpc_id, other)
        self.assertEqual([r.reason for r in cm.records if hasattr(r, "reason")], ["AUDX_MAX_STREAMS"])

        # a bug inside the preprocessor -> AUDX_ERROR
        stream = next(s for s in coordinator._streams.values() if s.preprocessor is not None)

        def boom(audio):
            raise RuntimeError("bug in preprocessor")

        stream.preprocessor.process = boom
        with self.assertLogs(LOGGER, level="ERROR") as cm:
            await coordinator.accept_chunk(first.rpc_id, request(160))
        self.assertEqual([r.reason for r in cm.records if hasattr(r, "reason")], ["AUDX_ERROR"])
        await coordinator.cancel_rpc(first.rpc_id, "done")
        await coordinator.cancel_rpc(second.rpc_id, "done")

    async def test_audx_pool_start_failure_logs_audx_start_failed(self):
        coordinator, _ = self._coordinator()
        coordinator.runtime = StartableRuntime()

        class BrokenPool(FakeStreamPool):
            def start(self, timeout):
                raise RuntimeError("AudX checkpoint not found")

        coordinator.audio_preproc_pool = BrokenPool()
        with self.assertLogs(LOGGER, level="ERROR") as cm:
            await coordinator.start()
        try:
            self.assertIsNone(coordinator.audio_preproc_pool)
            self.assertEqual([r.reason for r in cm.records if hasattr(r, "reason")], ["AUDX_START_FAILED"])
            self.assertEqual(_labeled_gauge(AUDIO_PREPROC_POOL_WORKERS).get(AUDIO_PREPROC_BACKEND_AUDX), 0.0)
        finally:
            await coordinator.shutdown()


if __name__ == "__main__":
    unittest.main()
