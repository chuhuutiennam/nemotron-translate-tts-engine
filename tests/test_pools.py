import unittest

from nemotron_testbed.domain import (
    RuntimeCommand,
    RuntimeCommandType,
    RuntimeResult,
    RuntimeResultType,
    StreamKey,
    TranslationJob,
    TtsJob,
)
from nemotron_testbed.pools import NemotronRuntimePool, TranslationRuntimePool, TtsRuntimePool


class FakeAsrRuntime:
    instances = []

    def __init__(self, *, result_callback, worker_id=0, **kwargs):
        self.result_callback = result_callback
        self.worker_id = worker_id
        self.commands = []
        self._queue_depth = 0
        FakeAsrRuntime.instances.append(self)

    @property
    def queue_depth(self):
        return self._queue_depth

    def start(self, timeout=0):
        pass

    def shutdown(self, timeout=0):
        pass

    def submit_nowait(self, command):
        self.commands.append(command)

    def submit(self, command, timeout=0):
        self.commands.append(command)

    def emit(self, result):
        self.result_callback(result)


class FakeJobRuntime:
    instances = []

    def __init__(self, *, result_callback, worker_id=0, **kwargs):
        self.result_callback = result_callback
        self.worker_id = worker_id
        self.jobs = []
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


class FakeTranslationRuntime(FakeJobRuntime):
    instances = []


class FakeTtsRuntime(FakeJobRuntime):
    instances = []


class PoolTest(unittest.TestCase):
    def setUp(self):
        FakeAsrRuntime.instances.clear()
        FakeTranslationRuntime.instances.clear()
        FakeTtsRuntime.instances.clear()

    def test_asr_stream_affinity_until_final(self):
        results = []
        pool = NemotronRuntimePool(
            workers=2,
            model_path="x",
            devices=["cpu"],
            language="vi-VN",
            left_context=56,
            lookahead_tokens=6,
            queue_size=8,
            warmup_seconds=0,
            result_callback=results.append,
            runtime_factory=FakeAsrRuntime,
        )
        key = StreamKey("rpc", "call", "MO")
        pool.submit_nowait(RuntimeCommand(RuntimeCommandType.START, key=key, utterance_id=1))
        owner = next(i for i, r in enumerate(FakeAsrRuntime.instances) if r.commands)
        pool.submit_nowait(RuntimeCommand(RuntimeCommandType.STEP, key=key, utterance_id=1))
        pool.submit_nowait(RuntimeCommand(RuntimeCommandType.FINALIZE, key=key, utterance_id=1))
        self.assertEqual(len(FakeAsrRuntime.instances[owner].commands), 3)
        self.assertEqual(pool.active_owners, 1)

        FakeAsrRuntime.instances[owner].emit(
            RuntimeResult(RuntimeResultType.FINAL, key=key, utterance_id=1, text="x")
        )
        self.assertEqual(pool.active_owners, 0)
        self.assertEqual(results[-1].type, RuntimeResultType.FINAL)

    def test_translation_pool_dispatches_jobs_across_workers(self):
        pool = TranslationRuntimePool(
            workers=2,
            model_path="x",
            tokenizer_path="x",
            devices=["cpu"],
            compute_type="int8",
            inter_threads=1,
            intra_threads=1,
            queue_size=8,
            beam_size=1,
            max_decoding_length=32,
            vi_prefix="vi: ",
            en_prefix="en: ",
            result_callback=lambda result: None,
            runtime_factory=FakeTranslationRuntime,
        )
        key = StreamKey("rpc", "call", "MO")
        for utt in (1, 2):
            pool.submit_nowait(
                TranslationJob(key, utt, "xin chao", "vi-VN", "en-US", 0, 1)
            )
        self.assertEqual([len(r.jobs) for r in FakeTranslationRuntime.instances], [1, 1])

    def test_tts_pool_dispatches_jobs_across_workers(self):
        pool = TtsRuntimePool(
            workers=2,
            en_model_path="en",
            en_config_path=None,
            vi_model_path="vi",
            vi_config_path=None,
            target_sample_rate=8000,
            queue_size=8,
            max_text_chars=1000,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=1.0,
            devices=["cpu"],
            result_callback=lambda result: None,
            runtime_factory=FakeTtsRuntime,
        )
        key = StreamKey("rpc", "call", "MO")
        for utt in (1, 2):
            pool.submit_nowait(TtsJob(key, utt, "hello", "en-US", 0, 1, 0))
        self.assertEqual([len(r.jobs) for r in FakeTtsRuntime.instances], [1, 1])


if __name__ == "__main__":
    unittest.main()
