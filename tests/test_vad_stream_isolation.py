import asyncio
import contextlib
import dataclasses
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nemotron_testbed import vad as vad_module
from nemotron_testbed.coordinator import EngineCoordinator
from nemotron_testbed.domain import RuntimeCommandType, StreamKey
from nemotron_testbed.vad import build_vad_controller
from test_coordinator_flow import FakeRuntime, make_settings

FAKE_MODEL_PATH = "fake-silero.jit"
CHUNK_SAMPLES = 320  # 20 ms at 16 kHz; deliberately not a multiple of the 512-sample Silero window
CHUNK_MS = 20.0
SPEECH = 0.5


class _Score:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _StatefulModel:
    """Like the Silero LSTM: the score of a window depends on every window seen before it."""

    def __init__(self):
        self.state = 0.0

    def eval(self):
        return self

    def reset_states(self):
        self.state = 0.0

    def __call__(self, tensor, _rate):
        self.state = min(1.0, 0.6 * self.state + float(np.abs(tensor.array).max()))
        return _Score(self.state)


class _FakeTensor:
    def __init__(self, array):
        self.array = array

    def to(self, _device):
        return self


class _FakeTorch:
    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def install_fake_silero(testcase: unittest.TestCase) -> _StatefulModel:
    """Make backend="silero" build stateful fake models; returns the cached template."""
    previous = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorch

    def restore():
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous

    testcase.addCleanup(restore)
    template = _StatefulModel()
    testcase.enterContext(mock.patch.dict(vad_module._silero_model_cache, {f"{FAKE_MODEL_PATH}:cpu": template}))
    return template


def pattern(on: int, off: int, total: int, lead: int = 0) -> list[float]:
    levels = [0.0] * lead
    while len(levels) < total:
        levels += [SPEECH] * on + [0.0] * off
    return levels[:total]


def chunk_audio(level: float) -> np.ndarray:
    return np.full(CHUNK_SAMPLES, level, dtype=np.float32)


class SileroPerStreamIsolationTest(unittest.TestCase):
    def setUp(self):
        self.template = install_fake_silero(self)
        self.signal_a = pattern(on=14, off=14, total=240)
        self.signal_b = pattern(on=9, off=17, total=240, lead=5)

    def _controller(self):
        return build_vad_controller(
            backend="silero",
            enabled=True,
            threshold=0.5,
            neg_threshold=0.5,
            energy_db_threshold=-42.0,
            min_speech_ms=64,
            end_silence_ms=100,
            silero_model_path=FAKE_MODEL_PATH,
            device="cpu",
        )

    @staticmethod
    def _step(controller, level):
        event = controller.process(chunk_audio(level), CHUNK_MS)
        return (
            event.name,
            controller.active,
            controller.speech_candidate_ms,
            controller.silence_ms,
            controller.probability.model.state,
        )

    def _run(self, controller, signal):
        return [self._step(controller, level) for level in signal]

    @staticmethod
    def _mismatches(trace, reference):
        return sum(1 for got, want in zip(trace, reference, strict=True) if got != want)

    def _isolated_traces(self):
        ref_a = self._run(self._controller(), self.signal_a)
        ref_b = self._run(self._controller(), self.signal_b)
        for ref in (ref_a, ref_b):
            self.assertLessEqual({"START", "END"}, {step[0] for step in ref})
        self.assertNotEqual(ref_a, ref_b)
        return ref_a, ref_b

    def test_each_stream_gets_its_own_model_copy(self):
        a, b = self._controller(), self._controller()
        self.assertIsNot(a.probability, b.probability)
        self.assertIsNot(a.probability.model, b.probability.model)
        self.assertIsNot(a.probability.model, self.template)
        self.assertIsNot(b.probability.model, self.template)

    def test_streams_never_touch_the_cached_template(self):
        a, b = self._controller(), self._controller()
        self._run(a, self.signal_a)
        self._run(b, self.signal_b)
        self.assertEqual(self.template.state, 0.0)

    def test_a_new_stream_starts_clean_while_another_is_mid_utterance(self):
        busy = self._controller()
        self._run(busy, [SPEECH] * 10)
        self.assertGreater(busy.probability.model.state, 0.0)
        fresh = self._controller()
        self.assertEqual(fresh.probability.model.state, 0.0)
        self.assertFalse(fresh.active)
        self.assertTrue(busy.active)

    def test_interleaved_streams_match_isolated_runs(self):
        ref_a, ref_b = self._isolated_traces()
        a, b = self._controller(), self._controller()
        trace_a, trace_b = [], []
        for level_a, level_b in zip(self.signal_a, self.signal_b, strict=True):
            trace_a.append(self._step(a, level_a))
            trace_b.append(self._step(b, level_b))
        self.assertEqual(self._mismatches(trace_a, ref_a), 0)
        self.assertEqual(self._mismatches(trace_b, ref_b), 0)

    def test_streams_on_parallel_threads_match_isolated_runs(self):
        ref_a, ref_b = self._isolated_traces()
        a, b = self._controller(), self._controller()
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_a = pool.submit(self._run, a, self.signal_a)
                future_b = pool.submit(self._run, b, self.signal_b)
                trace_a, trace_b = future_a.result(), future_b.result()
        finally:
            sys.setswitchinterval(previous)
        self.assertEqual(self._mismatches(trace_a, ref_a), 0)
        self.assertEqual(self._mismatches(trace_b, ref_b), 0)

    def test_a_controller_shared_by_two_streams_would_interfere(self):
        ref_a, _ = self._isolated_traces()
        shared = self._controller()
        trace_a = []
        for level_a, level_b in zip(self.signal_a, self.signal_b, strict=True):
            trace_a.append(self._step(shared, level_a))
            self._step(shared, level_b)
        self.assertGreater(self._mismatches(trace_a, ref_a), len(ref_a) // 2)

    def test_reset_restores_a_fresh_state(self):
        ref_a, _ = self._isolated_traces()
        controller = self._controller()
        self._run(controller, self.signal_a[:100])
        controller.reset()
        self.assertEqual(controller.probability.model.state, 0.0)
        self.assertEqual(self._mismatches(self._run(controller, self.signal_a), ref_a), 0)

    def test_reset_of_one_stream_does_not_touch_the_other(self):
        _, ref_b = self._isolated_traces()
        a, b = self._controller(), self._controller()
        trace_b = []
        for index, (level_a, level_b) in enumerate(zip(self.signal_a, self.signal_b, strict=True)):
            self._step(a, level_a)
            if index % 7 == 0:
                a.reset()
            trace_b.append(self._step(b, level_b))
        self.assertEqual(self._mismatches(trace_b, ref_b), 0)
        self.assertEqual(self.template.state, 0.0)
        self.assertEqual(self._controller().probability.model.state, 0.0)


class CoordinatorStreamVadIsolationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.template = install_fake_silero(self)
        settings = dataclasses.replace(
            make_settings(False),
            vad_enabled=True,
            vad_backend="silero",
            silero_model_path=FAKE_MODEL_PATH,
            vad_min_speech_ms=64,
            vad_end_silence_ms=100,
        )
        self.coordinator = EngineCoordinator(settings)
        self.runtime = FakeRuntime()
        self.coordinator.runtime = self.runtime

    @staticmethod
    def _chunk(stream_id, index, level):
        return SimpleNamespace(
            session_id="call-1",
            stream_id=stream_id,
            pcm=np.full(CHUNK_SAMPLES, int(level * 32768), dtype="<i2").tobytes(),
            sample_rate=16000,
            channels=1,
            rtp_timestamp=index * CHUNK_SAMPLES,
            rtp_clock_rate=16000,
            duration_ms=20,
            end_of_stream=False,
            language="vi",
            task="transcribe",
        )

    def _stream(self, rpc, stream_id):
        return self.coordinator._streams[StreamKey(rpc.rpc_id, "call-1", stream_id)]

    def _starts(self):
        return [c.key.stream_id for c in self.runtime.commands if c.type is RuntimeCommandType.START]

    async def _feed(self, rpc, stream_id, levels, first_index=0):
        for offset, level in enumerate(levels):
            await self.coordinator.accept_chunk(rpc.rpc_id, self._chunk(stream_id, first_index + offset, level))

    async def _feed_interleaved(self, rpc, levels_a, levels_b, first_index=0):
        for offset, (level_a, level_b) in enumerate(zip(levels_a, levels_b, strict=True)):
            await self.coordinator.accept_chunk(rpc.rpc_id, self._chunk("A", first_index + offset, level_a))
            await self.coordinator.accept_chunk(rpc.rpc_id, self._chunk("B", first_index + offset, level_b))

    async def test_streams_of_one_call_own_separate_vad_state(self):
        rpc = self.coordinator.open_rpc()
        await self._feed_interleaved(rpc, [0.0], [0.0])
        a, b = self._stream(rpc, "A"), self._stream(rpc, "B")
        self.assertIsNot(a.vad, b.vad)
        self.assertIsNot(a.vad.probability, b.vad.probability)
        self.assertIsNot(a.vad.probability.model, b.vad.probability.model)
        self.assertIsNot(a.vad.probability.model, self.template)
        self.assertIsNot(a.pre_roll, b.pre_roll)

    async def test_speech_on_one_stream_does_not_start_the_other(self):
        rpc = self.coordinator.open_rpc()
        await self._feed_interleaved(rpc, [SPEECH] * 10, [0.0] * 10)
        a, b = self._stream(rpc, "A"), self._stream(rpc, "B")
        self.assertEqual(self._starts(), ["A"])
        self.assertTrue(a.vad.active)
        self.assertIsNotNone(a.active_utterance)
        self.assertFalse(b.vad.active)
        self.assertIsNone(b.active_utterance)
        self.assertEqual(b.vad.probability.model.state, 0.0)
        self.assertEqual(self.template.state, 0.0)

    async def test_utterance_end_on_one_stream_leaves_the_other_utterance_running(self):
        rpc = self.coordinator.open_rpc()
        await self._feed_interleaved(rpc, [SPEECH] * 10, [0.0] * 10)
        a, b = self._stream(rpc, "A"), self._stream(rpc, "B")
        for index in range(10, 40):
            await self.coordinator.accept_chunk(rpc.rpc_id, self._chunk("A", index, 0.0))
            if a.active_utterance is None:
                break
            await self.coordinator.accept_chunk(rpc.rpc_id, self._chunk("B", index, SPEECH))
        else:
            self.fail("stream A never reached VAD END")
        # checked right after A's END: B must not need further audio to look unaffected
        self.assertEqual(self._starts(), ["A", "B"])
        self.assertIn(1, self.coordinator._finishing[a.key])
        self.assertFalse(a.vad.active)
        self.assertEqual(a.vad.probability.model.state, 0.0)
        self.assertIsNotNone(b.active_utterance)
        self.assertTrue(b.vad.active)
        self.assertGreater(b.vad.probability.model.state, 0.5)
        self.assertNotIn(b.key, self.coordinator._finishing)

    async def test_concurrent_rpcs_do_not_share_vad_state(self):
        rpc_speech = self.coordinator.open_rpc()
        rpc_silence = self.coordinator.open_rpc()
        await asyncio.gather(
            self._feed(rpc_speech, "A", [SPEECH] * 30),
            self._feed(rpc_silence, "A", [0.0] * 30),
        )
        speech, silence = self._stream(rpc_speech, "A"), self._stream(rpc_silence, "A")
        self.assertIsNot(speech.vad.probability.model, silence.vad.probability.model)
        self.assertEqual([c.key.rpc_id for c in self.runtime.commands], [rpc_speech.rpc_id])
        self.assertTrue(speech.vad.active)
        self.assertFalse(silence.vad.active)
        self.assertEqual(silence.vad.probability.model.state, 0.0)


if __name__ == "__main__":
    unittest.main()
