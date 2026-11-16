"""task=keyword: ASR-final text -> KeywordRuntime (topic detection) -> FINAL with topic_label/score/info.

The keyword model (multilingual E5) is not needed: the coordinator tests use a fake runtime, the
KeywordRuntime tests replace only the model calls (_load/_detect).
"""

import asyncio
import dataclasses
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from prometheus_client import REGISTRY

from nemotron_testbed.config import Settings
from nemotron_testbed.coordinator import EngineCoordinator, FeatureUnavailableError, RecognitionEnvelope
from nemotron_testbed.domain import (
    KeywordJob,
    KeywordResult,
    RuntimeResultType,
    StreamKey,
    SubmitStatus,
    TopicLabel,
)
from nemotron_testbed.keyword_runtime import KeywordRuntime
from nemotron_testbed.pb import speech_pb2
from nemotron_testbed.tasks import TASK_KEYWORD, normalize_task

from test_coordinator_flow import FakeRuntime, RuntimeResult, make_settings, request

TOPICS = {"WEATHER": {"rain": ["trời mưa to"]}, "FOOD": ["ăn phở"]}


class FakeKeywordRuntime:
    def __init__(self, status=SubmitStatus.ACCEPTED):
        self.jobs: list[KeywordJob] = []
        self.status = status
        self.queue_depth = 0

    def try_submit_nowait(self, job):
        if self.status is SubmitStatus.ACCEPTED:
            self.jobs.append(job)
        return self.status


def keyword_settings(**fields):
    settings = make_settings(False, max_utterance_ms=40)
    return dataclasses.replace(
        settings,
        keyword_enabled=True,
        keyword_model_path="models/multilingual-e5-small",
        keyword_topics={"WEATHER": {"rain": ["trời mưa"]}},
        **fields,
    )


def kw_request(timestamp, end=False, language="vi"):
    return request(timestamp, end=end, task="keyword", language=language)


def topic_result(job: KeywordJob, label=TopicLabel.WEATHER, info="rain", score=0.86, error=""):
    return KeywordResult(
        key=job.key,
        utterance_id=job.utterance_id,
        source_text=job.source_text,
        source_language=job.source_language,
        ts_start=job.ts_start,
        ts_end=job.ts_end,
        asr_confidence=job.asr_confidence,
        topic_label=label if not error else TopicLabel.UNKNOWN,
        topic_info=info if not error else "",
        topic_score=score if not error else 0.0,
        wall_start=job.wall_start,
        latency_ms=7.0,
        queue_ms=0.5,
        error=error,
    )


def envs(rpc):
    return [item for item in rpc.response_buffer._items if isinstance(item, RecognitionEnvelope)]


class KeywordFlowTest(unittest.IsolatedAsyncioTestCase):
    def _coordinator(self, kw=None, **fields):
        with mock.patch("nemotron_testbed.coordinator.KeywordRuntime"):
            coordinator = EngineCoordinator(keyword_settings(**fields))
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.keyword_runtime = kw if kw is not None else FakeKeywordRuntime()
        return coordinator, asr

    async def _one_utterance(self, coordinator, asr, final_text="Trời mưa to quá.", partial="trời mưa"):
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, kw_request(0))
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        if partial:
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.PARTIAL, key=key, utterance_id=utt, text=partial, confidence=0.5)
            )
        await coordinator.accept_chunk(rpc.rpc_id, kw_request(160, end=True))
        await coordinator.close_input(rpc.rpc_id)
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=utt, text=final_text, confidence=0.9)
        )
        return rpc, key, utt

    async def test_final_waits_for_topic_and_carries_it(self):
        coordinator, asr = self._coordinator()
        kw = coordinator.keyword_runtime
        rpc, key, utt = await self._one_utterance(coordinator, asr)

        self.assertEqual([(e.text, e.is_final) for e in envs(rpc)], [("trời mưa", False)])
        self.assertEqual(len(kw.jobs), 1)
        self.assertEqual(kw.jobs[0].source_text, "Trời mưa to quá.")
        self.assertEqual(kw.jobs[0].source_language, "vi-VN")
        self.assertNotIn(key, coordinator._streams)  # ASR side done; the RPC waits for the topic
        self.assertEqual(rpc.pending_postprocess, 1)
        self.assertFalse(rpc.finished)

        await coordinator._handle_keyword_result(topic_result(kw.jobs[0]))
        final = envs(rpc)[-1]
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, "Trời mưa to quá.")
        self.assertEqual(final.topic_label, int(TopicLabel.WEATHER))
        self.assertEqual(final.topic_info, "rain")
        self.assertAlmostEqual(final.topic_score, 0.86, places=6)
        self.assertAlmostEqual(final.confidence, 0.9, places=6)
        self.assertEqual([e.seq for e in envs(rpc)], [1, 2])  # seq survives the stream closing
        self.assertTrue(rpc.finished)
        self.assertEqual(coordinator._keyword_jobs, {})
        self.assertNotIn(key, coordinator._stream_seq)

    async def test_partials_have_no_topic(self):
        coordinator, asr = self._coordinator()
        rpc, _, _ = await self._one_utterance(coordinator, asr)
        partial = envs(rpc)[0]
        self.assertEqual((partial.topic_label, partial.topic_info, partial.topic_score), (0, "", 0.0))

    async def test_keyword_without_runtime_is_rejected(self):
        coordinator = EngineCoordinator(make_settings(False))
        coordinator.runtime = FakeRuntime()
        rpc = coordinator.open_rpc()
        with self.assertRaises(FeatureUnavailableError):
            await coordinator.accept_chunk(rpc.rpc_id, kw_request(0))

    async def test_empty_final_is_not_classified(self):
        coordinator, asr = self._coordinator()
        rpc, _, _ = await self._one_utterance(coordinator, asr, final_text="", partial="")
        self.assertEqual(coordinator.keyword_runtime.jobs, [])
        self.assertEqual(envs(rpc), [])  # empty result skipped, as for transcribe
        self.assertTrue(rpc.finished)

    async def test_detection_error_degrades_to_unknown(self):
        coordinator, asr = self._coordinator()
        rpc, _, _ = await self._one_utterance(coordinator, asr)
        before = REGISTRY.get_sample_value("keyword_errors_total") or 0.0
        await coordinator._handle_keyword_result(
            topic_result(coordinator.keyword_runtime.jobs[0], error="RuntimeError: boom")
        )
        final = envs(rpc)[-1]
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, "Trời mưa to quá.")
        self.assertEqual(final.topic_label, int(TopicLabel.UNKNOWN))
        self.assertEqual(REGISTRY.get_sample_value("keyword_errors_total") - before, 1)
        self.assertTrue(rpc.finished)  # RPC completes normally, no failure item
        self.assertFalse(rpc.cancelled)

    async def test_queue_full_degrades_to_unknown_at_once(self):
        coordinator, asr = self._coordinator(kw=FakeKeywordRuntime(SubmitStatus.OVERLOADED))
        rpc, _, _ = await self._one_utterance(coordinator, asr)
        final = envs(rpc)[-1]
        self.assertTrue(final.is_final)
        self.assertEqual(final.topic_label, int(TopicLabel.UNKNOWN))
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertTrue(rpc.finished)

    async def test_pending_limit_fails_rpc(self):
        coordinator, asr = self._coordinator(max_postprocess_pending_per_rpc=0)
        rpc, _, _ = await self._one_utterance(coordinator, asr)
        self.assertTrue(rpc.cancelled)
        failure = rpc.response_buffer._items[-1]
        self.assertEqual(getattr(failure, "code_name", None), "RESOURCE_EXHAUSTED")

    async def test_cancel_drops_pending_job_and_late_result_is_stale(self):
        coordinator, asr = self._coordinator()
        rpc, key, _ = await self._one_utterance(coordinator, asr)
        job = coordinator.keyword_runtime.jobs[0]
        await coordinator.cancel_rpc(rpc.rpc_id, "client gone")
        self.assertEqual(coordinator._keyword_jobs, {})
        self.assertNotIn(key, coordinator._stream_seq)
        await coordinator._handle_keyword_result(topic_result(job))  # no crash, nothing emitted
        self.assertEqual([e.is_final for e in envs(rpc)], [False])


class KeywordOrderTest(unittest.IsolatedAsyncioTestCase):
    """With several ASR workers utt2 can reach FINAL (and its topic) first; FINALs still go out 1 -> 2."""

    async def test_finals_in_utterance_order(self):
        with mock.patch("nemotron_testbed.coordinator.KeywordRuntime"):
            coordinator = EngineCoordinator(keyword_settings())
        asr = FakeRuntime()
        kw = FakeKeywordRuntime()
        coordinator.runtime = asr
        coordinator.keyword_runtime = kw
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, kw_request(0))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1))
        await coordinator.accept_chunk(rpc.rpc_id, kw_request(160))
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 1))
        await coordinator.accept_chunk(rpc.rpc_id, kw_request(320))
        await coordinator._handle_runtime_result(RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2))
        await coordinator.accept_chunk(rpc.rpc_id, kw_request(480, end=True))
        await coordinator.close_input(rpc.rpc_id)

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="ăn phở")
        )
        await coordinator._handle_keyword_result(topic_result(kw.jobs[0], TopicLabel.FOOD, "", 0.8))
        self.assertEqual(envs(rpc), [])  # held: utt1 has no FINAL yet

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="trời mưa")
        )
        self.assertEqual(envs(rpc), [])  # utt1 still in topic detection
        await coordinator._handle_keyword_result(topic_result(kw.jobs[1]))
        out = [(e.utterance_id, e.text, TopicLabel(e.topic_label).name, e.seq) for e in envs(rpc)]
        self.assertEqual(out, [(1, "trời mưa", "WEATHER", 1), (2, "ăn phở", "FOOD", 2)])
        self.assertTrue(rpc.finished)


_BASE_YAML = (
    "asr:\n  model_path: models/asr.nemo\n  lookahead_tokens: 6\n"
    "translation:\n  enabled: false\n"
    "tts:\n  enabled: false\n"
)


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(("KEYWORD_", "ENGINE_CONFIG_FILE", "APP_DIR"))}


class KeywordConfigTest(unittest.TestCase):
    def _yaml(self, body: str) -> Settings:
        body = _BASE_YAML + body.replace("asr:\n  lookahead_tokens: 6\n", "")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "engine.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            with mock.patch.dict(os.environ, _clean_env(), clear=True):
                return Settings.from_file_or_env(path)

    def test_default_is_off(self):
        settings = self._yaml("asr:\n  lookahead_tokens: 6\n")
        self.assertFalse(settings.keyword_enabled)
        self.assertEqual(settings.keyword_topics, {})

    def test_yaml_topics_list_and_mapping(self):
        settings = self._yaml(
            "asr:\n  lookahead_tokens: 6\n"
            "keyword:\n  enabled: true\n  model_path: models/e5\n  threshold: 0.8\n  device: cuda:1\n"
            "  topics:\n    weather:\n      rain: [trời mưa, mưa to]\n    FOOD: [ăn phở]\n"
        )
        self.assertTrue(settings.keyword_enabled)
        self.assertEqual(settings.keyword_device, "cuda:1")
        self.assertAlmostEqual(settings.keyword_threshold, 0.8)
        self.assertEqual(
            settings.keyword_topics,
            {"WEATHER": {"rain": ["trời mưa", "mưa to"]}, "FOOD": {"": ["ăn phở"]}},
        )

    def test_env_json_topics(self):
        env = {
            **_clean_env(),
            "KEYWORD_ENABLED": "true",
            "KEYWORD_MODEL_PATH": "models/e5",
            "KEYWORD_TOPICS_JSON": json.dumps(TOPICS, ensure_ascii=False),
            "MODEL_PATH": "models/asr.nemo",
            "MODEL_LOOKAHEAD_TOKENS": "6",
            "TRANSLATION_ENABLED": "false",
            "TTS_ENABLED": "false",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.keyword_topics["FOOD"], {"": ["ăn phở"]})

    def test_enabled_requires_model_path_and_topics(self):
        with self.assertRaisesRegex(ValueError, "model_path"):
            self._yaml("asr:\n  lookahead_tokens: 6\nkeyword:\n  enabled: true\n  topics: {FOOD: [phở]}\n")
        with self.assertRaisesRegex(ValueError, "topics"):
            self._yaml("asr:\n  lookahead_tokens: 6\nkeyword:\n  enabled: true\n  model_path: m\n")

    def test_unknown_label_rejected_at_config_time(self):
        with self.assertRaisesRegex(ValueError, "not a TopicLabel"):
            self._yaml("asr:\n  lookahead_tokens: 6\nkeyword:\n  topics: {SPORT: [bóng đá]}\n")
        with self.assertRaisesRegex(ValueError, "not a TopicLabel"):
            self._yaml("asr:\n  lookahead_tokens: 6\nkeyword:\n  topics: {UNKNOWN: [x]}\n")

    def test_bad_threshold_and_device(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            self._yaml("asr:\n  lookahead_tokens: 6\nkeyword:\n  threshold: 1.5\n")
        with self.assertRaisesRegex(ValueError, "device"):
            self._yaml("asr:\n  lookahead_tokens: 6\nkeyword:\n  device: gpu0\n")


class KeywordRuntimeContractTest(unittest.TestCase):
    """Submit/emit/shutdown of KeywordRuntime with the model calls replaced (no torch/transformers needed)."""

    def _runtime(self, detect, queue_size=4):
        results: list[KeywordResult] = []
        done = threading.Event()

        def callback(result):
            results.append(result)
            done.set()

        runtime = KeywordRuntime(
            model_path="unused", device="cpu", threshold=0.75, topics=TOPICS,
            queue_size=queue_size, result_callback=callback,
        )
        runtime._load = lambda: None  # type: ignore[method-assign]
        runtime._detect = detect  # type: ignore[method-assign]
        runtime.start(timeout=5)
        return runtime, results, done

    @staticmethod
    def _job(text="trời mưa"):
        return KeywordJob(
            key=StreamKey("rpc", "s", "MO"), utterance_id=1, source_text=text,
            source_language="vi-VN", ts_start=0, ts_end=160, asr_confidence=0.9,
        )

    def test_result_carries_topic(self):
        runtime, results, done = self._runtime(lambda text: (TopicLabel.WEATHER, "rain", 0.91))
        try:
            self.assertIs(runtime.try_submit_nowait(self._job()), SubmitStatus.ACCEPTED)
            self.assertTrue(done.wait(5))
            r = results[0]
            self.assertEqual((r.topic_label, r.topic_info, r.error), (TopicLabel.WEATHER, "rain", ""))
            self.assertAlmostEqual(r.topic_score, 0.91)
            self.assertGreaterEqual(r.queue_ms, 0.0)
        finally:
            runtime.shutdown(timeout=5)
        self.assertIs(runtime.try_submit_nowait(self._job()), SubmitStatus.UNAVAILABLE)

    def test_detect_exception_becomes_error_result(self):
        def boom(text):
            raise RuntimeError("cuda oom")

        runtime, results, done = self._runtime(boom)
        try:
            runtime.try_submit_nowait(self._job())
            self.assertTrue(done.wait(5))
            self.assertIn("cuda oom", results[0].error)
            self.assertIs(results[0].topic_label, TopicLabel.UNKNOWN)
        finally:
            runtime.shutdown(timeout=5)

    def test_queue_full_is_overloaded(self):
        gate = threading.Event()

        def slow(text):
            gate.wait(5)
            return TopicLabel.FOOD, "", 0.8

        runtime, _, _ = self._runtime(slow, queue_size=1)
        try:
            statuses = [runtime.try_submit_nowait(self._job()) for _ in range(4)]
            self.assertIn(SubmitStatus.OVERLOADED, statuses)
        finally:
            gate.set()
            runtime.shutdown(timeout=5)


class KeywordProtoTest(unittest.TestCase):
    def test_task_and_enum_aligned(self):
        self.assertEqual(normalize_task("keyword"), TASK_KEYWORD)
        for label in TopicLabel:
            self.assertEqual(speech_pb2.TopicLabel.Value(f"TOPIC_{label.name}"), int(label))

    def test_result_round_trip_keeps_master_fields(self):
        msg = speech_pb2.RecognitionResult(
            text="ăn phở", is_final=True, audio_pt=8,
            topic_label=speech_pb2.TOPIC_FOOD, topic_score=0.75, topic_info="pho",
        )
        back = speech_pb2.RecognitionResult.FromString(msg.SerializeToString())
        self.assertEqual((back.topic_label, back.topic_info, back.audio_pt), (speech_pb2.TOPIC_FOOD, "pho", 8))
        self.assertAlmostEqual(back.topic_score, 0.75, places=6)
        chunk = speech_pb2.AudioChunk(task="keyword", language_source="vi")  # master field 13 kept
        self.assertEqual(speech_pb2.AudioChunk.FromString(chunk.SerializeToString()).language_source, "vi")


if __name__ == "__main__":
    unittest.main()
