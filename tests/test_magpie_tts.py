import unittest
from unittest.mock import patch

import numpy as np

from nemotron_testbed.domain import StreamKey, SubmitStatus, TtsJob
from nemotron_testbed.magpie_tts_runtime import MagpieTtsRuntime
from nemotron_testbed.pools import MagpieTtsRuntimePool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(text="xin chào", lang="vi-VN"):
    key = StreamKey("rpc", "call", "MO")
    return TtsJob(key, utterance_id=1, text=text, target_language=lang,
                  ts_start=0, ts_end=1000, audio_pt=0)


def _make_runtime(**overrides):
    defaults = dict(
        model_path="fake",
        speaker_index=0,
        cfg_scale=2.5,
        temperature=0.6,
        top_k=80,
        volume=1.0,
        target_sample_rate=8000,
        queue_size=8,
        max_text_chars=20,
        result_callback=lambda r: None,
    )
    defaults.update(overrides)
    return MagpieTtsRuntime(**defaults)


class _FakeTensor:
    """Numpy-backed stand-in for the (audio, length) tensors returned by do_tts."""

    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def __getitem__(self, idx):
        return _FakeTensor(self._data[idx])

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._data

    def item(self):
        return float(self._data.flat[0])


class _FakeTorch:
    """Minimal torch stand-in for the `import torch` inside _synthesize."""

    class inference_mode:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False


def _runtime_with_stub_model(audio_samples=None, **overrides):
    """Return an unstarted runtime whose _model is pre-stubbed."""
    from unittest.mock import MagicMock
    rt = _make_runtime(**overrides)
    rt._sample_rate = 22050
    rt._model = MagicMock()
    if audio_samples is None:
        audio_samples = np.full(22050, 0.1, dtype=np.float32)
    audio_t = _FakeTensor(audio_samples[np.newaxis, :])         # shape (1, N)
    audio_len_t = _FakeTensor(np.array([len(audio_samples)]))
    rt._model.do_tts.return_value = (audio_t, audio_len_t)
    return rt


# ---------------------------------------------------------------------------
# _synthesize — unit tests (no thread, no model loading)
# ---------------------------------------------------------------------------

class SynthesizeTest(unittest.TestCase):
    def test_empty_text_returns_silence_without_calling_model(self):
        rt = _runtime_with_stub_model()
        with patch.dict("sys.modules", {"torch": _FakeTorch()}):
            payload, rate, duration = rt._synthesize(_make_job(text="   "))
        self.assertEqual(payload, b"")
        self.assertEqual(rate, 0)
        self.assertEqual(duration, 0.0)
        rt._model.do_tts.assert_not_called()

    def test_text_too_long_raises_value_error(self):
        rt = _runtime_with_stub_model(max_text_chars=5)
        with self.assertRaises(ValueError):
            with patch.dict("sys.modules", {"torch": _FakeTorch()}):
                rt._synthesize(_make_job(text="this is too long"))

    def test_unsupported_language_raises_value_error(self):
        rt = _runtime_with_stub_model()
        with self.assertRaises(ValueError):
            with patch.dict("sys.modules", {"torch": _FakeTorch()}):
                rt._synthesize(_make_job(lang="fr-FR"))

    def test_valid_synthesis_returns_non_empty_payload(self):
        rt = _runtime_with_stub_model()
        with patch.dict("sys.modules", {"torch": _FakeTorch()}):
            payload, rate, duration = rt._synthesize(_make_job())
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)
        self.assertEqual(rate, 22050)
        self.assertGreater(duration, 0.0)

    def test_apply_tn_true_forwarded_to_model(self):
        rt = _runtime_with_stub_model(apply_tn=True)
        with patch.dict("sys.modules", {"torch": _FakeTorch()}):
            rt._synthesize(_make_job())
        self.assertTrue(rt._model.do_tts.call_args.kwargs["apply_TN"])

    def test_apply_tn_default_false(self):
        rt = _runtime_with_stub_model()
        with patch.dict("sys.modules", {"torch": _FakeTorch()}):
            rt._synthesize(_make_job())
        self.assertFalse(rt._model.do_tts.call_args.kwargs["apply_TN"])

    def test_volume_above_one_clips_without_crashing(self):
        audio = np.full(22050, 0.9, dtype=np.float32)
        rt = _runtime_with_stub_model(audio_samples=audio, volume=2.0)
        with patch.dict("sys.modules", {"torch": _FakeTorch()}):
            payload, _, duration = rt._synthesize(_make_job())
        self.assertGreater(len(payload), 0)
        self.assertGreater(duration, 0.0)

    def test_english_target_language_accepted(self):
        rt = _runtime_with_stub_model()
        with patch.dict("sys.modules", {"torch": _FakeTorch()}):
            payload, _, _ = rt._synthesize(_make_job(lang="en-US"))
        self.assertGreater(len(payload), 0)


# ---------------------------------------------------------------------------
# Lifecycle — startup error, queue states
# ---------------------------------------------------------------------------

class LifecycleTest(unittest.TestCase):
    def test_startup_error_propagates_through_start(self):
        rt = _make_runtime()
        with patch.object(rt, "_load", side_effect=RuntimeError("model missing")):
            with self.assertRaises(RuntimeError):
                rt.start(timeout=5.0)

    def test_local_path_with_extension_not_found_raises_file_not_found(self):
        rt = _make_runtime(model_path="nonexistent/model.nemo")
        # FileNotFoundError raised before nemo import — no patch needed
        with self.assertRaises(FileNotFoundError):
            rt._load()

    def test_try_submit_when_stopped_returns_unavailable(self):
        rt = _make_runtime()
        rt._stopped.set()
        self.assertEqual(rt.try_submit_nowait(_make_job()), SubmitStatus.UNAVAILABLE)

    def test_try_submit_when_queue_full_returns_overloaded(self):
        rt = _make_runtime(queue_size=1)
        rt._jobs.put_nowait(_make_job())        # fill the single slot
        self.assertEqual(rt.try_submit_nowait(_make_job()), SubmitStatus.OVERLOADED)

    def test_submit_nowait_when_stopped_raises(self):
        rt = _make_runtime()
        rt._stopped.set()
        with self.assertRaises(RuntimeError):
            rt.submit_nowait(_make_job())

    def test_submit_nowait_when_queue_full_raises(self):
        rt = _make_runtime(queue_size=1)
        rt._jobs.put_nowait(_make_job())
        with self.assertRaises(RuntimeError):
            rt.submit_nowait(_make_job())


# ---------------------------------------------------------------------------
# MagpieTtsRuntimePool — routing and param forwarding
# ---------------------------------------------------------------------------

class _FakeMagpieTtsRuntime:
    instances: list = []

    def __init__(self, *, result_callback, worker_id=0,
                 suppress_model_init_output=True, apply_tn=False,
                 device="cpu", **kwargs):
        self.result_callback = result_callback
        self.worker_id = worker_id
        self.suppress_model_init_output = suppress_model_init_output
        self.apply_tn = apply_tn
        self.device = device
        self.jobs: list = []
        self._queue_depth = 0
        type(self).instances.append(self)

    @property
    def queue_depth(self):
        return self._queue_depth

    def start(self, timeout=0):
        pass

    def shutdown(self, timeout=0):
        pass

    def submit_nowait(self, job):
        self.jobs.append(job)

    def try_submit_nowait(self, job):
        self.jobs.append(job)
        return SubmitStatus.ACCEPTED


def _make_pool(**overrides):
    defaults = dict(
        workers=2,
        model_path="fake",
        speaker_index=0,
        cfg_scale=2.5,
        temperature=0.6,
        top_k=80,
        volume=1.0,
        target_sample_rate=8000,
        queue_size=8,
        max_text_chars=1000,
        devices=["cpu"],
        result_callback=lambda r: None,
        runtime_factory=_FakeMagpieTtsRuntime,
    )
    defaults.update(overrides)
    return MagpieTtsRuntimePool(**defaults)


class PoolRoutingTest(unittest.TestCase):
    def setUp(self):
        _FakeMagpieTtsRuntime.instances.clear()

    def test_dispatches_jobs_across_workers(self):
        pool = _make_pool()
        key = StreamKey("rpc", "call", "MO")
        for utt in (1, 2):
            pool.submit_nowait(TtsJob(key, utt, "hello", "en-US", 0, 1, 0))
        self.assertEqual([len(r.jobs) for r in _FakeMagpieTtsRuntime.instances], [1, 1])

    def test_routes_to_least_loaded_worker(self):
        pool = _make_pool(workers=2)
        _FakeMagpieTtsRuntime.instances[0]._queue_depth = 5
        _FakeMagpieTtsRuntime.instances[1]._queue_depth = 0
        key = StreamKey("rpc", "call", "MO")
        pool.submit_nowait(TtsJob(key, 1, "hello", "en-US", 0, 1, 0))
        self.assertEqual(len(_FakeMagpieTtsRuntime.instances[1].jobs), 1)
        self.assertEqual(len(_FakeMagpieTtsRuntime.instances[0].jobs), 0)

    def test_device_assigned_round_robin(self):
        _make_pool(workers=4, devices=["cuda:2", "cuda:3"])
        devices = [rt.device for rt in _FakeMagpieTtsRuntime.instances]
        self.assertEqual(devices, ["cuda:2", "cuda:3", "cuda:2", "cuda:3"])

    def test_all_queues_full_raises(self):
        pool = _make_pool()
        for rt in _FakeMagpieTtsRuntime.instances:
            def _raise(job):
                raise RuntimeError("queue full")
            rt.submit_nowait = _raise
        with self.assertRaises(RuntimeError):
            pool.submit_nowait(TtsJob(StreamKey("r", "c", "s"), 1, "x", "en-US", 0, 1, 0))


class PoolParamForwardingTest(unittest.TestCase):
    def setUp(self):
        _FakeMagpieTtsRuntime.instances.clear()

    def test_forwards_suppress_model_init_output_false(self):
        _make_pool(suppress_model_init_output=False)
        for rt in _FakeMagpieTtsRuntime.instances:
            self.assertFalse(rt.suppress_model_init_output)

    def test_forwards_suppress_model_init_output_true(self):
        _make_pool(suppress_model_init_output=True)
        for rt in _FakeMagpieTtsRuntime.instances:
            self.assertTrue(rt.suppress_model_init_output)

    def test_forwards_apply_tn_true(self):
        _make_pool(apply_tn=True)
        for rt in _FakeMagpieTtsRuntime.instances:
            self.assertTrue(rt.apply_tn)

    def test_forwards_apply_tn_default_false(self):
        _make_pool()
        for rt in _FakeMagpieTtsRuntime.instances:
            self.assertFalse(rt.apply_tn)


if __name__ == "__main__":
    unittest.main()
