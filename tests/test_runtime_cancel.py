"""ASR CANCEL (runtime.py _CommandQueue): recorded on submit, own path that wakes the worker even with the command
queue full, queued commands of the utterance dropped, state freed, then CANCELLED and the pool affinity released.

Real NemotronRuntimePool / NemotronRuntime worker threads; only the model is replaced. _run_one_step blocks while
the runtime's gate is closed, which keeps the worker busy at a known point."""

import random
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from nemotron_testbed.domain import RuntimeCommand, RuntimeCommandType, RuntimeResultType, StreamKey, SubmitStatus
from nemotron_testbed.pools import NemotronRuntimePool
from nemotron_testbed.runtime import NemotronRuntime

BUSY = StreamKey("rpc-busy", "call-busy", "MO")
KEY = StreamKey("rpc", "call", "MO")
AUDIO = np.zeros(160, dtype=np.float32)


class GatedRuntime(NemotronRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate = threading.Event()
        self.gate.set()
        self.in_step = threading.Event()
        self.steps = []  # (StreamKey, utterance_id, flush) of each model step run
        self.step_delay = 0.0

    def _load_model(self):
        self.model = SimpleNamespace(encoder=SimpleNamespace(get_initial_cache_state=lambda batch_size=1: (0, 0, 0)))

    def _warmup(self):
        pass

    def _run_one_step(self, state, flush):
        self.steps.append((state.key, state.utterance_id, flush))
        self.in_step.set()
        if not self.gate.wait(10):
            raise RuntimeError("gate never opened")
        if self.step_delay:
            time.sleep(self.step_delay)
        state.last_text = "x"
        return "x"


def command(kind, key=KEY, utterance_id=1):
    audio = AUDIO if kind is RuntimeCommandType.STEP else np.empty(0, dtype=np.float32)
    return RuntimeCommand(type=kind, key=key, utterance_id=utterance_id, audio=audio)


START, STEP, FINALIZE, CANCEL = (
    RuntimeCommandType.START,
    RuntimeCommandType.STEP,
    RuntimeCommandType.FINALIZE,
    RuntimeCommandType.CANCEL,
)


class RuntimeCancelTest(unittest.TestCase):
    def make_pool(self, workers=1, queue_size=64):
        self.results = []
        pool = NemotronRuntimePool(
            workers=workers, model_path="", devices=["cpu"], language="vi-VN", left_context=56, lookahead_tokens=6,
            queue_size=queue_size, warmup_seconds=0.0, result_callback=self.results.append,
            runtime_factory=GatedRuntime,
        )
        pool.start(timeout=5)

        def stop():
            for runtime in pool._runtimes:
                runtime.gate.set()
            pool.shutdown(timeout=5)

        self.addCleanup(stop)
        return pool

    def wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("timed out")
            time.sleep(0.002)

    def types(self, key=KEY):
        return [result.type for result in list(self.results) if result.key == key]

    def wait_result(self, kind, key=KEY, count=1):
        self.wait_until(lambda: self.types(key).count(kind) >= count)

    def steps(self, runtime, key=KEY):
        return [step for step in runtime.steps if step[0] == key]

    def assert_clean(self, pool):
        self.wait_until(
            lambda: pool.active_owners == 0
            and pool.pending_cancellations == 0
            and pool.queue_depth == 0
            and all(not runtime._states for runtime in pool._runtimes)
        )

    def hold_worker(self, pool):
        """Worker 0 busy in a model step of BUSY; returns the runtime. Open runtime.gate to free it."""
        runtime = pool._runtimes[0]
        pool.try_submit_nowait(command(START, BUSY))
        self.wait_result(RuntimeResultType.STARTED, BUSY)
        runtime.gate.clear()
        runtime.in_step.clear()
        pool.try_submit_nowait(command(STEP, BUSY))
        self.assertTrue(runtime.in_step.wait(5))
        return runtime

    def release_busy(self, pool, runtime):
        runtime.gate.set()
        pool.submit_nowait(command(CANCEL, BUSY))
        self.wait_result(RuntimeResultType.CANCELLED, BUSY)

    def test_cancel_before_start_drops_the_start(self):
        pool = self.make_pool()
        runtime = self.hold_worker(pool)
        self.assertIs(pool.try_submit_nowait(command(START)), SubmitStatus.ACCEPTED)
        pool.submit_nowait(command(CANCEL))
        self.assertEqual(pool.pending_cancellations, 1)  # recorded on submit, before the worker sees it

        self.release_busy(pool, runtime)
        self.wait_result(RuntimeResultType.CANCELLED)
        self.assertEqual(self.types(), [RuntimeResultType.CANCELLED])  # the START never ran: no STARTED
        self.assert_clean(pool)

    def test_cancel_during_step_drops_queued_commands(self):
        pool = self.make_pool()
        runtime = pool._runtimes[0]
        pool.try_submit_nowait(command(START))
        self.wait_result(RuntimeResultType.STARTED)
        runtime.gate.clear()
        pool.try_submit_nowait(command(STEP))
        self.assertTrue(runtime.in_step.wait(5))
        pool.try_submit_nowait(command(STEP))
        pool.try_submit_nowait(command(FINALIZE))
        pool.submit_nowait(command(CANCEL))

        runtime.gate.set()
        self.wait_result(RuntimeResultType.CANCELLED)
        self.assertEqual(
            self.types(), [RuntimeResultType.STARTED, RuntimeResultType.PARTIAL, RuntimeResultType.CANCELLED]
        )
        self.assertEqual(len(self.steps(runtime)), 1)  # the queued STEP and FINALIZE were dropped
        self.assert_clean(pool)

    def test_cancel_after_final(self):
        pool = self.make_pool()
        runtime = pool._runtimes[0]
        # one command at a time, as the coordinator does (FINALIZE would run before a queued START)
        pool.try_submit_nowait(command(START))
        self.wait_result(RuntimeResultType.STARTED)
        pool.try_submit_nowait(command(STEP))
        self.wait_result(RuntimeResultType.PARTIAL)
        pool.try_submit_nowait(command(FINALIZE))
        self.wait_result(RuntimeResultType.FINAL)
        self.assert_clean(pool)

        pool.submit_nowait(command(CANCEL))  # affinity already released: the pool confirms it
        self.wait_result(RuntimeResultType.CANCELLED)
        self.assert_clean(pool)

        # CANCEL that reached the worker after FINAL removed the state (the pool had not released the affinity
        # yet): the idle worker is woken, nothing is left behind.
        runtime.submit_nowait(command(CANCEL))
        self.wait_result(RuntimeResultType.CANCELLED, count=2)
        self.assertEqual(
            self.types(),
            [
                RuntimeResultType.STARTED,
                RuntimeResultType.PARTIAL,
                RuntimeResultType.FINAL,
                RuntimeResultType.CANCELLED,
                RuntimeResultType.CANCELLED,
            ],
        )
        self.assert_clean(pool)

    def test_repeated_cancel(self):
        pool = self.make_pool()
        runtime = self.hold_worker(pool)
        pool.try_submit_nowait(command(START))
        for _ in range(3):
            pool.submit_nowait(command(CANCEL))
        self.assertEqual(pool.pending_cancellations, 1)

        self.release_busy(pool, runtime)
        self.wait_result(RuntimeResultType.CANCELLED)
        self.assert_clean(pool)
        self.assertEqual(self.types(), [RuntimeResultType.CANCELLED])  # cleaned up once, confirmed once

        pool.submit_nowait(command(CANCEL))  # after cleanup: nothing owned, the pool confirms
        self.wait_result(RuntimeResultType.CANCELLED, count=2)
        self.assert_clean(pool)

    def test_cancel_with_full_queue(self):
        pool = self.make_pool(queue_size=4)
        runtime = pool._runtimes[0]
        pool.try_submit_nowait(command(START))
        self.wait_result(RuntimeResultType.STARTED)
        runtime.gate.clear()
        pool.try_submit_nowait(command(STEP))
        self.assertTrue(runtime.in_step.wait(5))
        while pool.try_submit_nowait(command(STEP)) is SubmitStatus.ACCEPTED:
            pass
        self.assertEqual(runtime.queue_depth, 4)
        self.assertIs(pool.try_submit_nowait(command(FINALIZE)), SubmitStatus.OVERLOADED)

        pool.submit_nowait(command(CANCEL))  # must not raise QueueFullError
        self.assertEqual(pool.pending_cancellations, 1)
        self.assertIs(pool.try_submit_nowait(command(STEP)), SubmitStatus.ACCEPTED)  # cancelled: dropped
        self.assertEqual(runtime.queue_depth, 4)

        runtime.gate.set()
        self.wait_result(RuntimeResultType.CANCELLED)
        self.assertEqual(len(self.steps(runtime)), 1)
        self.assertEqual(
            self.types(), [RuntimeResultType.STARTED, RuntimeResultType.PARTIAL, RuntimeResultType.CANCELLED]
        )
        self.assert_clean(pool)

    def test_cancel_wakes_idle_worker(self):
        pool = self.make_pool()
        pool.try_submit_nowait(command(START))
        self.wait_result(RuntimeResultType.STARTED)
        self.assertEqual(len(pool._runtimes[0]._states), 1)
        pool.submit_nowait(command(CANCEL))
        self.wait_result(RuntimeResultType.CANCELLED)
        self.assert_clean(pool)

    def test_open_cancel_cycles_return_to_zero(self):
        pool = self.make_pool(workers=2, queue_size=16)
        for runtime in pool._runtimes:
            runtime.step_delay = 0.0005
        rng = random.Random(20261112)
        patterns = [
            [START, CANCEL],
            [START, STEP, CANCEL],
            [START, STEP, STEP, FINALIZE, CANCEL],
            [START, CANCEL, CANCEL],
            [START, "wait", STEP, CANCEL],
            [START, "wait", STEP, FINALIZE, "final", CANCEL],
        ]
        keys = []
        for index in range(300):
            key = StreamKey(f"rpc-{index}", "call", "MO" if index % 2 else "MT")
            keys.append(key)
            for step in rng.choice(patterns):
                if step == "wait":
                    self.wait_result(RuntimeResultType.STARTED, key)
                elif step == "final":
                    self.wait_result(RuntimeResultType.FINAL, key)
                elif step is CANCEL:
                    pool.submit_nowait(command(CANCEL, key))
                elif pool.try_submit_nowait(command(step, key)) is not SubmitStatus.ACCEPTED:
                    pool.submit_nowait(command(CANCEL, key))  # as fail_rpc -> cancel_rpc
                    break
            if rng.random() < 0.2:
                time.sleep(0.001)

        self.assert_clean(pool)
        for key in keys:
            self.assertIn(RuntimeResultType.CANCELLED, self.types(key))
            self.assertLessEqual(self.types(key).count(RuntimeResultType.STARTED), 1)
        self.assertNotIn(RuntimeResultType.ERROR, [result.type for result in self.results])


if __name__ == "__main__":
    unittest.main()
