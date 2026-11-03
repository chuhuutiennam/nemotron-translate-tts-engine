import queue
import threading
import time
import unittest

import numpy as np

from nemotron_testbed.audio_preprocess_runtime import (
    AudioPreprocessRuntime,
    AudioPreprocessRuntimePool,
)
from nemotron_testbed.domain import (
    AudioPreprocessJob,
    AudioPreprocessResult,
    StreamKey,
    SubmitStatus,
)

KEY = StreamKey("rpc", "sess", "stream")


def job(region_id: int = 0, n: int = 160) -> AudioPreprocessJob:
    window = (np.arange(n, dtype=np.float32) / n) - 0.5
    return AudioPreprocessJob(key=KEY, region_id=region_id, window=window)


class Collector:
    def __init__(self) -> None:
        self.results: queue.Queue[AudioPreprocessResult] = queue.Queue()

    def __call__(self, result: AudioPreprocessResult) -> None:
        self.results.put(result)

    def get(self, timeout: float = 5.0) -> AudioPreprocessResult:
        return self.results.get(timeout=timeout)


def make_runtime(model, collector, *, queue_size=4, warmup_samples=0, device="cpu", worker_id=0):
    return AudioPreprocessRuntime(
        model_loader=lambda dev: model,
        device=device,
        queue_size=queue_size,
        warmup_samples=warmup_samples,
        result_callback=collector,
        worker_id=worker_id,
    )


class BlockingModel:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, window: np.ndarray) -> np.ndarray:
        self.entered.set()
        self.release.wait(10.0)
        return window


class RuntimeTest(unittest.TestCase):
    def test_result_carries_model_output_and_timings(self):
        collector = Collector()
        runtime = make_runtime(lambda w: w * 0.5, collector)
        runtime.start()
        try:
            request = job(region_id=7)
            self.assertIs(runtime.try_submit_nowait(request), SubmitStatus.ACCEPTED)
            result = collector.get()
        finally:
            runtime.shutdown()
        self.assertEqual(result.key, KEY)
        self.assertEqual(result.region_id, 7)
        self.assertEqual(result.error, "")
        self.assertEqual(result.audio.dtype, np.float32)
        np.testing.assert_allclose(result.audio, request.window * 0.5)
        self.assertGreaterEqual(result.compute_s, 0.0)
        self.assertGreaterEqual(result.queue_s, 0.0)

    def test_model_exception_becomes_error_result(self):
        def boom(window):
            raise RuntimeError("cuda out of memory")

        collector = Collector()
        runtime = make_runtime(boom, collector)
        runtime.start()
        try:
            runtime.try_submit_nowait(job())
            result = collector.get()
            self.assertIsNone(result.audio)
            self.assertIn("RuntimeError", result.error)
            self.assertIn("cuda out of memory", result.error)
            runtime.try_submit_nowait(job(region_id=1))
            self.assertEqual(collector.get().region_id, 1)
        finally:
            runtime.shutdown()

    def test_non_finite_output_becomes_error_result(self):
        def nan_model(window):
            out = window.copy()
            out[3] = np.nan
            return out

        collector = Collector()
        runtime = make_runtime(nan_model, collector)
        runtime.start()
        try:
            runtime.try_submit_nowait(job())
            result = collector.get()
        finally:
            runtime.shutdown()
        self.assertIsNone(result.audio)
        self.assertIn("bad model output", result.error)

    def test_wrong_output_shape_becomes_error_result(self):
        collector = Collector()
        runtime = make_runtime(lambda w: w[:-1], collector)
        runtime.start()
        try:
            runtime.try_submit_nowait(job())
            result = collector.get()
        finally:
            runtime.shutdown()
        self.assertIsNone(result.audio)
        self.assertIn("bad model output", result.error)

    def test_stream_job_runs_step_with_its_state_and_any_output_length(self):
        class StreamModel:
            def __init__(self):
                self.calls = []

            def __call__(self, window):  # warm-up only
                return window

            def step(self, state, samples, final):
                self.calls.append((state, samples.size, final))
                state["seen"] += samples.size
                return samples[: samples.size // 2] * 2.0

        model = StreamModel()
        collector = Collector()
        runtime = make_runtime(model, collector, warmup_samples=64)
        state = {"seen": 0}
        runtime.start()
        try:
            for region_id, final in ((0, False), (1, True)):
                window = np.ones(100, dtype=np.float32)
                runtime.try_submit_nowait(
                    AudioPreprocessJob(key=KEY, region_id=region_id, window=window, state=state, final=final)
                )
                result = collector.get()
                self.assertEqual(result.error, "")
                np.testing.assert_array_equal(result.audio, np.full(50, 2.0, dtype=np.float32))
        finally:
            runtime.shutdown()
        self.assertEqual(state["seen"], 200)
        self.assertEqual([(s is state, n, f) for s, n, f in model.calls], [(True, 100, False), (True, 100, True)])

    def test_stream_job_with_bad_output_becomes_error_result(self):
        class BadStream:
            def __call__(self, window):
                return window

            def step(self, state, samples, final):
                return np.full((2, 3), np.float32(1.0)) if final else np.array([np.nan], dtype=np.float32)

        collector = Collector()
        runtime = make_runtime(BadStream(), collector)
        runtime.start()
        try:
            for final in (False, True):
                runtime.try_submit_nowait(
                    AudioPreprocessJob(key=KEY, region_id=0, window=np.ones(4, np.float32), state={}, final=final)
                )
                result = collector.get()
                self.assertIsNone(result.audio)
                self.assertIn("bad model output", result.error)
        finally:
            runtime.shutdown()

    def test_callback_exception_does_not_kill_worker(self):
        calls = []

        def callback(result):
            calls.append(result.region_id)
            if len(calls) == 1:
                raise ValueError("callback bug")

        runtime = AudioPreprocessRuntime(
            model_loader=lambda dev: (lambda w: w),
            device="cpu",
            queue_size=4,
            warmup_samples=0,
            result_callback=callback,
        )
        runtime.start()
        try:
            runtime.try_submit_nowait(job(region_id=0))
            runtime.try_submit_nowait(job(region_id=1))
            deadline = time.monotonic() + 5.0
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            runtime.shutdown()
        self.assertEqual(calls, [0, 1])

    def test_queue_full_is_overloaded(self):
        model = BlockingModel()
        collector = Collector()
        runtime = make_runtime(model, collector, queue_size=1)
        runtime.start()
        try:
            self.assertIs(runtime.try_submit_nowait(job(0)), SubmitStatus.ACCEPTED)
            self.assertTrue(model.entered.wait(5.0))
            self.assertIs(runtime.try_submit_nowait(job(1)), SubmitStatus.ACCEPTED)
            self.assertEqual(runtime.queue_depth, 1)
            self.assertIs(runtime.try_submit_nowait(job(2)), SubmitStatus.OVERLOADED)
            model.release.set()
            self.assertEqual({collector.get().region_id, collector.get().region_id}, {0, 1})
        finally:
            model.release.set()
            runtime.shutdown()

    def test_unavailable_after_shutdown(self):
        collector = Collector()
        runtime = make_runtime(lambda w: w, collector)
        runtime.start()
        runtime.shutdown()
        self.assertIs(runtime.try_submit_nowait(job()), SubmitStatus.UNAVAILABLE)
        runtime.shutdown()

    def test_startup_failure_raises(self):
        def loader(device):
            raise FileNotFoundError("denoiser ONNX not found")

        runtime = AudioPreprocessRuntime(
            model_loader=loader,
            device="cpu",
            queue_size=1,
            warmup_samples=0,
            result_callback=Collector(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            runtime.start()
        self.assertIsInstance(ctx.exception.__cause__, FileNotFoundError)
        self.assertIs(runtime.try_submit_nowait(job()), SubmitStatus.UNAVAILABLE)

    def test_warmup_runs_model_on_requested_window_size(self):
        seen = []

        def model(window):
            seen.append((len(window), window.dtype))
            return window

        runtime = make_runtime(model, Collector(), warmup_samples=480)
        runtime.start()
        runtime.shutdown()
        self.assertEqual(seen, [(480, np.dtype(np.float32))])

    def test_failing_warmup_fails_startup(self):
        def model(window):
            raise RuntimeError("no kernel for this device")

        runtime = make_runtime(model, Collector(), warmup_samples=480)
        with self.assertRaises(RuntimeError):
            runtime.start()


class PoolTest(unittest.TestCase):
    def make_pool(self, loader, *, workers=2, devices=("cpu",), queue_size=2, collector=None):
        collector = collector or Collector()
        pool = AudioPreprocessRuntimePool(
            workers=workers,
            devices=list(devices),
            queue_size=queue_size,
            warmup_samples=0,
            model_loader=loader,
            result_callback=collector,
        )
        return pool, collector

    def test_workers_get_devices_round_robin(self):
        loaded = []
        lock = threading.Lock()

        def loader(device):
            with lock:
                loaded.append(device)
            return lambda w: w

        pool, _ = self.make_pool(loader, workers=3, devices=("cuda:0", "cuda:1"))
        pool.start()
        pool.shutdown()
        self.assertEqual(sorted(loaded), ["cuda:0", "cuda:0", "cuda:1"])

    def test_jobs_are_served_by_any_worker(self):
        pool, collector = self.make_pool(lambda dev: (lambda w: w * 2), workers=2, queue_size=8)
        pool.start()
        try:
            for index in range(6):
                self.assertIs(pool.try_submit_nowait(job(index)), SubmitStatus.ACCEPTED)
            got = sorted(collector.get().region_id for _ in range(6))
        finally:
            pool.shutdown()
        self.assertEqual(got, list(range(6)))

    def test_full_worker_is_skipped(self):
        busy = BlockingModel()
        loads = []

        def loader(device):
            loads.append(device)
            return busy if len(loads) == 1 else (lambda w: w)

        pool, collector = self.make_pool(loader, workers=2, queue_size=1)
        pool.start()
        try:
            blocked = pool._runtimes[0]
            self.assertIs(blocked.try_submit_nowait(job(100)), SubmitStatus.ACCEPTED)
            self.assertTrue(busy.entered.wait(5.0))
            self.assertIs(blocked.try_submit_nowait(job(101)), SubmitStatus.ACCEPTED)
            self.assertIs(blocked.try_submit_nowait(job(102)), SubmitStatus.OVERLOADED)
            for index in range(5):
                self.assertIs(pool.try_submit_nowait(job(index)), SubmitStatus.ACCEPTED)
                self.assertEqual(collector.get().region_id, index)
        finally:
            busy.release.set()
            pool.shutdown()

    def test_all_workers_full_is_overloaded(self):
        models = [BlockingModel(), BlockingModel()]
        counter = iter(models)
        pool, _ = self.make_pool(lambda dev: next(counter), workers=2, queue_size=1)
        pool.start()
        try:
            accepted = 0
            for index in range(2):
                self.assertIs(pool.try_submit_nowait(job(index)), SubmitStatus.ACCEPTED)
                accepted += 1
            for model in models:
                self.assertTrue(model.entered.wait(5.0))
            for index in range(2, 4):
                self.assertIs(pool.try_submit_nowait(job(index)), SubmitStatus.ACCEPTED)
            self.assertEqual(pool.queue_depth, 2)
            self.assertIs(pool.try_submit_nowait(job(9)), SubmitStatus.OVERLOADED)
        finally:
            for model in models:
                model.release.set()
            pool.shutdown()

    def test_unavailable_after_shutdown(self):
        pool, _ = self.make_pool(lambda dev: (lambda w: w))
        pool.start()
        pool.shutdown()
        self.assertIs(pool.try_submit_nowait(job()), SubmitStatus.UNAVAILABLE)

    def test_start_failure_stops_workers_already_started(self):
        calls = []

        def loader(device):
            calls.append(device)
            if len(calls) == 2:
                raise RuntimeError("second worker cannot load")
            return lambda w: w

        pool, _ = self.make_pool(loader, workers=2)
        with self.assertRaises(RuntimeError):
            pool.start()
        self.assertIs(pool.try_submit_nowait(job()), SubmitStatus.UNAVAILABLE)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.make_pool(lambda dev: (lambda w: w), workers=0)
        with self.assertRaises(ValueError):
            self.make_pool(lambda dev: (lambda w: w), devices=())


if __name__ == "__main__":
    unittest.main()
