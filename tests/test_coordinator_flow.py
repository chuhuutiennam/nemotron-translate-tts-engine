import asyncio
import dataclasses
import logging
import math
import time
import unittest
from types import SimpleNamespace

import numpy as np

from nemotron_testbed.config import Settings
from nemotron_testbed.coordinator import EngineCoordinator, RecognitionEnvelope
from nemotron_testbed.metrics import (
    EMPTY_RESULTS_SKIPPED,
    SEGMENT_ERRORS,
    SEGMENT_PHASE_MS,
    UTTERANCE_E2E_MS,
)
from nemotron_testbed.vad import VadEvent
from nemotron_testbed.domain import (
    RuntimeResult as _RuntimeResult,
    RuntimeResultType,
    StreamKey,
    StreamPhase,
    TranslationResult,
    TtsResult,
)


def RuntimeResult(*args, **kwargs):
    kwargs.setdefault("detected_language", "vi-VN")
    return _RuntimeResult(*args, **kwargs)


class FakeRuntime:
    def __init__(self):
        self.commands = []
        self.reject_next = False

    def submit_nowait(self, command):
        self.commands.append(command)

    def try_submit_nowait(self, command):
        from nemotron_testbed.domain import SubmitStatus
        if self.reject_next:
            self.reject_next = False
            return SubmitStatus.OVERLOADED
        self.commands.append(command)
        return SubmitStatus.ACCEPTED


class FakeTranslationRuntime:
    def __init__(self):
        self.jobs = []

    def submit_nowait(self, job):
        self.jobs.append(job)

    def try_submit_nowait(self, job):
        from nemotron_testbed.domain import SubmitStatus
        self.jobs.append(job)
        return SubmitStatus.ACCEPTED


class FakeTtsRuntime:
    def __init__(self):
        self.jobs = []

    def submit_nowait(self, job):
        self.jobs.append(job)

    def try_submit_nowait(self, job):
        from nemotron_testbed.domain import SubmitStatus
        self.jobs.append(job)
        return SubmitStatus.ACCEPTED


def make_settings(
    translate=False,
    max_utterance_ms=20_000,
    segmentation_mode="off",
    use_language_source=False,
):
    return Settings(
        config_source="env",
        grpc_address="127.0.0.1:0",
        max_rpc_message_mb=8,
        response_queue_size=8,
        metrics_port=0,
        log_level="DEBUG",
        console_log_level="ERROR",
        third_party_log_level="ERROR",
        log_file="",
        log_backup_count=1,
        pipeline_trace=True,
        pipeline_trace_text=True,
        pipeline_trace_partial=False,
        pipeline_trace_level="DEBUG",
        suppress_model_init_output=True,
        grpc_keepalive_time_ms=60_000,
        grpc_keepalive_timeout_ms=20_000,
        grpc_keepalive_permit_without_calls=True,
        grpc_min_ping_interval_without_data_ms=60_000,
        grpc_max_ping_strikes=2,
        grpc_stop_grace_s=5.0,
        result_queue_maxsize=512,
        shutdown_timeout_s=30.0,
        stats_log_interval_s=0,
        gpu_monitor_interval_s=0,
        model_loading_timeout_s=1800.0,
        asr_backlog_soft_ms=320.0,
        asr_backlog_hard_ms=800.0,
        short_utterance_ms=300.0,
        model_path="unused.nemo",
        device="cpu",
        language="vi-VN",
        use_language_source=use_language_source,
        left_context=56,
        lookahead_tokens=0,
        model_chunk_ms=80,
        warmup_seconds=0.0,
        runtime_queue_size=32,
        asr_workers=1,
        asr_beam_size=1,
        cpu_threads=1,
        cpu_interop_threads=1,
        cuda_memory_fraction=0.0,
        vad_backend="energy",
        vad_enabled=False,
        vad_threshold=0.5,
        vad_neg_threshold=0.5,
        vad_energy_db_threshold=-42.0,
        vad_min_speech_ms=0,
        vad_end_silence_ms=100,
        vad_pre_roll_ms=0,
        vad_no_start_warn_ms=0,
        silero_model_path=None,
        vad_device="cpu",
        max_utterance_ms=max_utterance_ms,
        segmentation_mode=segmentation_mode,
        segmentation_stable_n=2,
        segmentation_stable_ms=0.0,
        segmentation_punct_stable_n=2,
        segmentation_punct_stable_ms=0.0,
        segmentation_min_tokens=4,
        segmentation_max_segment_ms=0.0,
        segmentation_comma_after_ms=0.0,
        segmentation_speech_to_text=False,
        segmentation_stt_punct_stable_n=2,
        segmentation_stt_punct_stable_ms=0.0,
        max_ingress_audio_ms=3_000,
        max_postprocess_pending_per_rpc=8,
        max_concurrent_rpcs=0,
        max_concurrent_utterances=0,
        rpc_max_idle_s=0,
        rtp_gap_policy="warn",
        rtp_max_gap_ms=200,
        rtp_out_of_order_policy="drop",
        rtp_jitter_tolerance_ms=20,
        translation_enabled=translate,
        translation_model_path="dummy-ct2" if translate else None,
        translation_tokenizer_path="dummy-tokenizer" if translate else None,
        translation_devices=["cpu"],
        translation_compute_type="int8",
        translation_inter_threads=1,
        translation_intra_threads=1,
        translation_queue_size=16,
        translation_workers=1,
        translation_beam_size=1,
        translation_max_decoding_length=128,
        translation_batch_size=1,
        translation_batch_wait_ms=0,
        translation_vi_prefix="vi: ",
        translation_en_prefix="en: ",
        translate_emit_source_partials=False,
        tts_enabled=translate,
        tts_backend="piper",
        tts_queue_size=16,
        tts_workers=1,
        tts_en_model_path="en.onnx" if translate else None,
        tts_en_config_path=None,
        tts_vi_model_path="vi.onnx" if translate else None,
        tts_vi_config_path=None,
        tts_output_pt=0,
        tts_target_sample_rate=8000,
        tts_max_text_chars=1000,
        tts_continuation_retry_ms=2000.0,
        tts_max_audio_chunk_bytes=0,
        tts_length_scale=1.0,
        tts_noise_scale=0.667,
        tts_noise_w_scale=0.8,
        tts_volume=1.0,
        tts_intra_threads=2,
        tts_devices=["cpu"],
        tts_magpie_model_path=None,
        tts_magpie_codec_model_path=None,
        tts_speaker_index=0,
        tts_cfg_scale=2.5,
        tts_temperature=0.6,
        tts_top_k=80,
        tts_magpie_apply_tn=False,
        asr_devices=["cpu"],
    )


def request(timestamp, end=False, task="transcribe", language=None, language_source=""):
    if language is None:
        # translation: `language` is the TARGET (the source is detected); transcription: the source
        language = "en" if task in ("translate", "realtime_translation") else "vi"
    samples = np.full(320, 1000, dtype="<i2")
    return SimpleNamespace(
        session_id="call-1",
        stream_id="MO",
        pcm=samples.tobytes(),
        sample_rate=16000,
        channels=1,
        rtp_timestamp=timestamp,
        duration_ms=20,
        end_of_stream=end,
        language=language,
        language_source=language_source,
        task=task,
    )


class CoordinatorFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_start_partial_final(self):
        coordinator = EngineCoordinator(make_settings(False))
        fake = FakeRuntime()
        coordinator.runtime = fake
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0))
        self.assertEqual(fake.commands[-1].type.name, "START")
        key = fake.commands[-1].key
        utterance_id = fake.commands[-1].utterance_id

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utterance_id)
        )
        for i in range(1, 4):
            await coordinator.accept_chunk(rpc.rpc_id, request(i * 160))
        self.assertEqual(fake.commands[-1].type.name, "STEP")

        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.PARTIAL,
                key=key,
                utterance_id=utterance_id,
                text="xin chào",
                confidence=0.73,
            )
        )
        partial = await rpc.response_buffer.get()
        self.assertIsInstance(partial, RecognitionEnvelope)
        self.assertFalse(partial.is_final)
        self.assertEqual(partial.language, "vi-VN")
        self.assertAlmostEqual(partial.confidence, 0.73, places=6)

        await coordinator.accept_chunk(rpc.rpc_id, request(4 * 160, end=True))
        self.assertEqual(fake.commands[-1].type.name, "FINALIZE")
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL,
                key=key,
                utterance_id=utterance_id,
                text="Xin chào.",
                confidence=0.91,
            )
        )
        final = await rpc.response_buffer.get()
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, "Xin chào.")
        self.assertEqual(final.audio_payload, b"")
        self.assertAlmostEqual(final.confidence, 0.91, places=6)

    async def test_translate_final_runs_background_pipeline(self):
        coordinator = EngineCoordinator(make_settings(True))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))

        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL,
                key=key,
                utterance_id=utt,
                text="Xin chào",
                confidence=0.84,
            )
        )
        self.assertEqual(len(tr.jobs), 1)
        self.assertEqual(tr.jobs[0].target_language, "en-US")
        self.assertNotIn(key, coordinator._streams)  # ASR state released immediately.
        self.assertEqual(rpc.pending_postprocess, 1)

        await coordinator._handle_translation_result(
            TranslationResult(
                key=key,
                utterance_id=utt,
                source_text="Xin chào",
                translated_text="Hello",
                source_language="vi-VN",
                target_language="en-US",
                ts_start=0,
                ts_end=640,
            )
        )
        self.assertEqual(len(tts.jobs), 1)
        self.assertEqual(tts.jobs[0].text, "Hello")

        await coordinator._handle_tts_result(
            TtsResult(
                key=key,
                utterance_id=utt,
                text="Hello",
                target_language="en-US",
                ts_start=0,
                ts_end=640,
                audio_payload=b"\xff" * 160,
                audio_pt=0,
            )
        )
        final = await rpc.response_buffer.get()
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, "Hello")
        self.assertEqual(final.language, "en-US")
        self.assertEqual(final.audio_pt, 0)
        # Translate/TTS does not invent a translation-quality score. The public
        # confidence remains the source ASR utterance confidence.
        self.assertAlmostEqual(final.confidence, 0.84, places=6)
        self.assertEqual(len(final.audio_payload), 160)
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_tts_serialization_holds_second_utterance(self):
        """TTS 2 must not be submitted until TTS 1 completes.

        Two utterances share one stream (same StreamKey) so they use the same
        PipelineOrderState.  max_utterance_ms=40 forces utt1 to FINALIZE after
        2 chunks (640 samples / 16 kHz = 40 ms), freeing the active slot so
        utt2 can start immediately on the same stream.
        """
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=40))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        # Chunk 0: stream opens, VAD disabled → START immediately, utt1 begins.
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        self.assertEqual(asr.commands[-1].type.name, "START")
        self.assertEqual(asr.commands[-1].utterance_id, 1)
        key = asr.commands[-1].key

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )

        # Chunk 1: utt1 accumulates 640 samples (≥ max_utterance_ms=40 ms) →
        # FINALIZE submitted, active slot freed, VAD reset.
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        self.assertEqual(asr.commands[-1].type.name, "FINALIZE")
        self.assertEqual(asr.commands[-1].utterance_id, 1)

        # Chunk 2: VAD reset → START again on the same stream, utt2 begins.
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        self.assertEqual(asr.commands[-1].type.name, "START")
        self.assertEqual(asr.commands[-1].utterance_id, 2)

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )

        # Chunk 3 (end_of_stream): utt2 also reaches 640 samples → FINALIZE,
        # stream enters CLOSING.
        await coordinator.accept_chunk(rpc.rpc_id, request(480, end=True, task="translate", language="en"))
        self.assertEqual(asr.commands[-1].type.name, "FINALIZE")
        self.assertEqual(asr.commands[-1].utterance_id, 2)

        # Both utterances produce FINAL → translation submitted for each.
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Tạm biệt")
        )
        self.assertEqual(len(tr.jobs), 2)

        # Translation for utt1 arrives → TTS 1 submitted (it's next_tts_slot).
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1,
                source_text="Xin chào", translated_text="Hello",
                source_language="vi-VN", target_language="en-US",
                ts_start=0, ts_end=320,
            )
        )
        self.assertEqual(len(tts.jobs), 1)
        self.assertEqual(tts.jobs[0].text, "Hello")

        # Translation for utt2 arrives BEFORE TTS 1 completes → must be held.
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=2,
                source_text="Tạm biệt", translated_text="Goodbye",
                source_language="vi-VN", target_language="en-US",
                ts_start=640, ts_end=960,
            )
        )
        self.assertEqual(len(tts.jobs), 1, "TTS 2 must be held until TTS 1 completes")

        # TTS 1 completes → drain pending → TTS 2 submitted automatically.
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1,
                text="Hello", target_language="en-US",
                ts_start=0, ts_end=320, audio_payload=b"\xff" * 80, audio_pt=0,
            )
        )
        self.assertEqual(len(tts.jobs), 2, "TTS 2 submitted after TTS 1 completed")
        self.assertEqual(tts.jobs[1].text, "Goodbye")


    async def test_tts_slot_not_stuck_when_translation_fails_mid_stream(self):
        """TTS slot must not deadlock when utterance N's translation fails while TTS N-1 is in-flight.

        Scenario:
          utt1 → Translation OK → TTS 1 submitted (next_tts=1)
          utt2 → Translation FAIL while TTS 1 still in-flight
          utt3 → Translation OK → should eventually reach TTS

        Without the tts_skip fix, next_tts advances to 2 when TTS 1 completes,
        finds nothing in translation_pending[2], and stalls — utt3 is never submitted.
        """
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=40))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        # Three utterances on the same stream (max_utterance_ms=40 forces boundaries after 2 chunks)
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        # utt1 finalized; utt2 starts
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(480, task="translate", language="en"))
        # utt2 finalized; utt3 starts
        await coordinator.accept_chunk(rpc.rpc_id, request(640, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=3)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(800, end=True, task="translate", language="en"))

        # All three utterances produce FINAL
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Tạm biệt")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=3, text="Hẹn gặp lại")
        )
        self.assertEqual(len(tr.jobs), 3)

        # utt1 translation OK → TTS 1 submitted (next_tts=1)
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1,
                source_text="Xin chào", translated_text="Hello",
                source_language="vi-VN", target_language="en-US",
                ts_start=0, ts_end=320,
            )
        )
        self.assertEqual(len(tts.jobs), 1)  # TTS 1 in-flight

        # utt2 translation FAILS while TTS 1 is still in-flight
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=2,
                source_text="Tạm biệt", translated_text="",
                source_language="vi-VN", target_language="en-US",
                ts_start=320, ts_end=640,
                error="translation timeout",
            )
        )
        self.assertEqual(len(tts.jobs), 1, "TTS 2 must not be submitted (translation failed)")

        # utt3 translation OK → held because TTS 1 still in-flight
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=3,
                source_text="Hẹn gặp lại", translated_text="See you later",
                source_language="vi-VN", target_language="en-US",
                ts_start=640, ts_end=960,
            )
        )
        self.assertEqual(len(tts.jobs), 1, "TTS 3 must be held until TTS 1 completes")

        # TTS 1 completes → slot advances to 2 → tts_skip has 2 → skip → slot=3 → drain pending[3]
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1,
                text="Hello", target_language="en-US",
                ts_start=0, ts_end=320, audio_payload=b"\xff" * 80, audio_pt=0,
            )
        )
        self.assertEqual(len(tts.jobs), 2, "TTS 3 must be submitted after TTS 1 completes (slot 2 skipped)")
        self.assertEqual(tts.jobs[1].text, "See you later")

    async def test_asr_start_queue_full_fails_rpc(self):
        """START rejected by ASR runtime → RPC fails with RESOURCE_EXHAUSTED."""
        coordinator = EngineCoordinator(make_settings(False))
        fake = FakeRuntime()
        coordinator.runtime = fake
        rpc = coordinator.open_rpc()

        fake.reject_next = True
        await coordinator.accept_chunk(rpc.rpc_id, request(0))

        self.assertTrue(rpc.cancelled)
        # No START was queued (rejected before appending)
        self.assertEqual(len(fake.commands), 0)

        failure = await rpc.response_buffer.get()
        from nemotron_testbed.coordinator import RpcFailure
        self.assertIsInstance(failure, RpcFailure)
        self.assertEqual(failure.code_name, "RESOURCE_EXHAUSTED")

    async def test_asr_step_queue_full_retries_on_next_chunk(self):
        """STEP rejected → audio prepended back, submitted successfully on next chunk."""
        import dataclasses
        # model_chunk_ms=40 → 640 samples per STEP, so two 320-sample RTP chunks fill one STEP
        settings = dataclasses.replace(make_settings(False), model_chunk_ms=40)
        coordinator = EngineCoordinator(settings)
        fake = FakeRuntime()
        coordinator.runtime = fake
        rpc = coordinator.open_rpc()

        # Chunk 0 → START; pre_roll (320 samples) folded into model_audio → 320 samples
        await coordinator.accept_chunk(rpc.rpc_id, request(0))
        key = fake.commands[-1].key
        utt = fake.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        # After STARTED: inference_pending=False, model_audio=320 < 640 → no STEP yet

        # Chunk 1: model_audio grows to 640 ≥ 640 → STEP attempted, rejected → audio prepended
        fake.reject_next = True
        await coordinator.accept_chunk(rpc.rpc_id, request(160))
        self.assertEqual(len([c for c in fake.commands if c.type.name == "STEP"]), 0)
        self.assertFalse(rpc.cancelled)

        # Chunk 2: model_audio = 640 (prepend) + 320 (new) = 960 ≥ 640 → STEP accepted
        await coordinator.accept_chunk(rpc.rpc_id, request(320))
        self.assertEqual(len([c for c in fake.commands if c.type.name == "STEP"]), 1)
        self.assertFalse(rpc.cancelled)

    async def test_tts_worker_error_degrades_gracefully(self):
        """TtsResult with error → text-only result emitted, RPC not cancelled."""
        coordinator = EngineCoordinator(make_settings(True))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL, key=key, utterance_id=utt,
                text="Xin chào", confidence=0.9,
            )
        )
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=utt,
                source_text="Xin chào", translated_text="Hello",
                source_language="vi-VN", target_language="en-US",
                ts_start=0, ts_end=640,
            )
        )
        # TTS worker fails
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=utt,
                text="Hello", target_language="en-US",
                ts_start=0, ts_end=640,
                audio_payload=b"",
                audio_pt=0,
                error="onnx runtime crash",
            )
        )

        self.assertFalse(rpc.cancelled)
        self.assertEqual(rpc.pending_postprocess, 0)
        result = await rpc.response_buffer.get()
        self.assertIsInstance(result, RecognitionEnvelope)
        self.assertTrue(result.is_final)
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.audio_payload, b"")

    async def test_ingress_backlog_exceeds_limit_fails_rpc(self):
        """Ingress audio exceeding max_ingress_audio_ms causes RESOURCE_EXHAUSTED."""
        import dataclasses
        # max_ingress_audio_ms=40 → 640 samples limit at 16 kHz
        settings = dataclasses.replace(make_settings(False), max_ingress_audio_ms=40)
        coordinator = EngineCoordinator(settings)
        fake = FakeRuntime()
        coordinator.runtime = fake
        rpc = coordinator.open_rpc()

        # Chunk 0 → START; model_audio = 320 samples, inference_pending = True
        await coordinator.accept_chunk(rpc.rpc_id, request(0))
        # STARTED not acked → inference_pending stays True; audio queues in model_audio

        # Chunk 1 → model_audio = 640, exactly at limit (640 ≤ 640), no fail yet
        await coordinator.accept_chunk(rpc.rpc_id, request(160))
        self.assertFalse(rpc.cancelled)

        # Chunk 2 → 640 + 320 = 960 > 640 → fail_rpc triggered
        await coordinator.accept_chunk(rpc.rpc_id, request(320))

        self.assertTrue(rpc.cancelled)
        from nemotron_testbed.coordinator import RpcFailure
        failure = await rpc.response_buffer.get()
        self.assertIsInstance(failure, RpcFailure)
        self.assertEqual(failure.code_name, "RESOURCE_EXHAUSTED")


class FinalizeWhileStepPendingTest(unittest.IsolatedAsyncioTestCase):
    async def test_audio_after_forced_finalize_is_not_dropped(self):
        """Finalize requested while a STEP is in flight must free the slot at once.

        Audio arriving before the STEP result returns has to reach the next
        utterance instead of being discarded, and FINALIZE must still be
        submitted (with the full remainder) once the pending result arrives.
        """
        # 60 ms → 960 samples: forced finalize on the third 20 ms chunk.
        coordinator = EngineCoordinator(make_settings(False, max_utterance_ms=60))
        fake = FakeRuntime()
        coordinator.runtime = fake
        rpc = coordinator.open_rpc()

        for i in range(3):
            await coordinator.accept_chunk(rpc.rpc_id, request(i * 160))

        # STARTED never acked → inference_pending stays True at forced finalize.
        self.assertEqual([c.type.name for c in fake.commands], ["START"])
        key = fake.commands[0].key
        stream = coordinator._streams[key]
        self.assertIn(1, coordinator._finishing[key])
        self.assertNotEqual(getattr(stream.active_utterance, "utterance_id", None), 1)

        # Chunk arriving inside the window is kept, not dropped.
        await coordinator.accept_chunk(rpc.rpc_id, request(3 * 160))
        self.assertEqual(fake.commands[-1].type.name, "START")
        self.assertEqual(fake.commands[-1].utterance_id, 2)
        self.assertEqual(stream.active_utterance.model_audio.samples, 320)

        # Pending result returns → deferred FINALIZE carries the whole remainder.
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        finalize = [c for c in fake.commands if c.type.name == "FINALIZE"]
        self.assertEqual(len(finalize), 1)
        self.assertEqual(finalize[0].utterance_id, 1)
        self.assertEqual(finalize[0].audio.size, 960)


class OutOfOrderFinalTest(unittest.IsolatedAsyncioTestCase):
    async def test_later_utterance_final_first_does_not_strand_earlier_one(self):
        """With several ASR workers utt2 can reach FINAL before utt1.

        The pipeline order must still start at the lowest unresolved utterance,
        emit both in order, and let the RPC drain (pending_postprocess == 0).
        """
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=40))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        def tr_result(uid, src, dst, t0, t1):
            return TranslationResult(
                key=key, utterance_id=uid, source_text=src, translated_text=dst,
                source_language="vi-VN", target_language="en-US", ts_start=t0, ts_end=t1,
            )

        def tts_result(uid, text, t0, t1):
            return TtsResult(
                key=key, utterance_id=uid, text=text, target_language="en-US",
                ts_start=t0, ts_end=t1, audio_payload=b"\xff" * 80, audio_pt=0,
            )

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        self.assertEqual(asr.commands[-1].type.name, "FINALIZE")
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        self.assertEqual(asr.commands[-1].utterance_id, 2)
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(
            rpc.rpc_id, request(480, end=True, task="translate", language="en")
        )
        self.assertEqual(asr.commands[-1].type.name, "FINALIZE")
        self.assertEqual(asr.commands[-1].utterance_id, 2)

        # utt2 finishes first (e.g. it ran on a less loaded ASR worker).
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Tạm biệt")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào")
        )
        self.assertEqual(len(tr.jobs), 2)

        await coordinator._handle_translation_result(tr_result(2, "Tạm biệt", "Goodbye", 640, 960))
        await coordinator._handle_translation_result(tr_result(1, "Xin chào", "Hello", 0, 320))

        # utt1 is the earliest: it must be first (and only) in TTS; utt2 is held.
        self.assertEqual([j.text for j in tts.jobs], ["Hello"])

        await coordinator._handle_tts_result(tts_result(1, "Hello", 0, 320))
        self.assertEqual([j.text for j in tts.jobs], ["Hello", "Goodbye"])
        await coordinator._handle_tts_result(tts_result(2, "Goodbye", 640, 960))

        texts = []
        for _ in range(2):
            try:
                env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
            except asyncio.TimeoutError:
                self.fail(f"utterance stranded in pipeline; emitted so far: {texts}")
            texts.append(env.text)
        self.assertEqual(texts, ["Hello", "Goodbye"])
        self.assertEqual(rpc.pending_postprocess, 0)


    async def test_end_of_stream_does_not_strand_earlier_utterance_still_finishing(self):
        """End of stream while utt2 is active must not close the stream before utt1's FINAL."""
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=60))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        def tr_result(uid, src, dst, t0, t1):
            return TranslationResult(
                key=key, utterance_id=uid, source_text=src, translated_text=dst,
                source_language="vi-VN", target_language="en-US", ts_start=t0, ts_end=t1,
            )

        def tts_result(uid, text, t0, t1):
            return TtsResult(
                key=key, utterance_id=uid, text=text, target_language="en-US",
                ts_start=t0, ts_end=t1, audio_payload=b"\xff" * 80, audio_pt=0,
            )

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 1))
        await coordinator.accept_chunk(rpc.rpc_id, request(480, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(
            rpc.rpc_id, request(640, end=True, task="translate", language="en")
        )
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 2))

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Tạm biệt")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào")
        )
        self.assertEqual([j.utterance_id for j in tr.jobs], [2, 1])

        await coordinator._handle_translation_result(tr_result(2, "Tạm biệt", "Goodbye", 640, 960))
        await coordinator._handle_translation_result(tr_result(1, "Xin chào", "Hello", 0, 320))
        await coordinator._handle_tts_result(tts_result(1, "Hello", 0, 320))
        await coordinator._handle_tts_result(tts_result(2, "Goodbye", 640, 960))

        texts = []
        for _ in range(2):
            try:
                env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
            except asyncio.TimeoutError:
                self.fail(f"utterance stranded in pipeline; emitted so far: {texts}")
            texts.append(env.text)
        self.assertEqual(texts, ["Hello", "Goodbye"])
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_empty_earlier_utterance_does_not_hold_later_tts(self):
        """An earlier utterance with empty ASR text must free its TTS slot."""
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=40))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(
            rpc.rpc_id, request(480, end=True, task="translate", language="en")
        )

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Tạm biệt")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="")
        )
        self.assertEqual(len(tr.jobs), 1)

        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=2, source_text="Tạm biệt", translated_text="Goodbye",
                source_language="vi-VN", target_language="en-US", ts_start=640, ts_end=960,
            )
        )
        self.assertEqual([j.text for j in tts.jobs], ["Goodbye"])


class RejectOnceTtsRuntime(FakeTtsRuntime):
    """Reports a full queue the first time it sees `reject_phrase_id`, accepts afterwards."""

    def __init__(self, reject_phrase_id):
        super().__init__()
        self.reject_phrase_id = reject_phrase_id
        self.rejected = False

    def _should_reject(self, job):
        if not self.rejected and job.phrase_id == self.reject_phrase_id:
            self.rejected = True
            return True
        return False

    def submit_nowait(self, job):
        from nemotron_testbed.domain import QueueFullError
        if self._should_reject(job):
            raise QueueFullError("all TTS worker queues are full or stopped")
        self.jobs.append(job)

    def try_submit_nowait(self, job):
        from nemotron_testbed.domain import SubmitStatus
        if self._should_reject(job):
            return SubmitStatus.OVERLOADED
        self.jobs.append(job)
        return SubmitStatus.ACCEPTED


class TranscribeOrderTest(unittest.IsolatedAsyncioTestCase):
    """Transcribe results go out in utterance order: with several ASR workers utt2 can reach FINAL before utt1.

    The proto result has no utterance_id; clients and eval_wer.py join FINALs in the order received.
    """

    SENT = "This is a complete sentence here."

    async def _two_utterances(self, *, segmentation=False, utt1_partials=(), utt2_partials=()):
        settings = make_settings(False, max_utterance_ms=40, segmentation_mode="on" if segmentation else "off")
        settings = dataclasses.replace(settings, segmentation_speech_to_text=segmentation)
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        rpc = coordinator.open_rpc()

        async def partials(utterance_id, texts):
            for text in texts:
                await coordinator._handle_runtime_result(
                    RuntimeResult(
                        type=RuntimeResultType.PARTIAL, key=key, utterance_id=utterance_id, text=text,
                        metadata={"model_step_executed": True},
                    )
                )

        await coordinator.accept_chunk(rpc.rpc_id, request(0, language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1))
        await partials(1, utt1_partials)
        await coordinator.accept_chunk(rpc.rpc_id, request(160, language="en"))
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 1))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, language="en"))
        await coordinator._handle_runtime_result(RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2))
        await partials(2, utt2_partials)
        await coordinator.accept_chunk(rpc.rpc_id, request(480, end=True, language="en"))
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 2))
        return coordinator, rpc, key

    @staticmethod
    def _envs(rpc):
        return [item for item in rpc.response_buffer._items if isinstance(item, RecognitionEnvelope)]

    async def _final(self, coordinator, key, utterance_id, text):
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=utterance_id, text=text)
        )

    async def test_later_final_waits_for_the_earlier_one(self):
        coordinator, rpc, key = await self._two_utterances()
        await self._final(coordinator, key, 2, "cau hai")
        self.assertEqual(self._envs(rpc), [])  # held: utt1 has not emitted its FINAL
        await self._final(coordinator, key, 1, "cau mot")
        envs = self._envs(rpc)
        self.assertEqual([(e.utterance_id, e.text, e.is_final) for e in envs], [(1, "cau mot", True), (2, "cau hai", True)])
        self.assertEqual([e.seq for e in envs], [1, 2])
        stream = coordinator._streams.get(key)
        self.assertTrue(stream is None or not stream.stt_held)

    async def test_held_partials_keep_only_the_latest(self):
        coordinator, rpc, key = await self._two_utterances(utt2_partials=["cau", "cau hai"])
        self.assertEqual(self._envs(rpc), [])  # utt2 partials held while utt1 is unfinished
        await self._final(coordinator, key, 1, "cau mot")
        self.assertEqual(
            [(e.utterance_id, e.text, e.is_final) for e in self._envs(rpc)],
            [(1, "cau mot", True), (2, "cau hai", False)],
        )
        await self._final(coordinator, key, 2, "cau hai xong")
        self.assertEqual(
            [(e.utterance_id, e.text, e.is_final) for e in self._envs(rpc)],
            [(1, "cau mot", True), (2, "cau hai", False), (2, "cau hai xong", True)],
        )

    async def test_empty_earlier_final_still_releases_the_later_one(self):
        coordinator, rpc, key = await self._two_utterances()
        await self._final(coordinator, key, 2, "cau hai")
        await self._final(coordinator, key, 1, "")
        self.assertEqual([(e.utterance_id, e.text) for e in self._envs(rpc)], [(2, "cau hai")])

    async def test_held_segments_keep_their_order(self):
        coordinator, rpc, key = await self._two_utterances(
            segmentation=True, utt2_partials=[self.SENT, self.SENT, self.SENT + " and more"]
        )
        self.assertEqual(self._envs(rpc), [])  # utt2's committed sentence is held too
        await self._final(coordinator, key, 1, "cau mot.")
        await self._final(coordinator, key, 2, self.SENT + " and more words")
        envs = self._envs(rpc)
        self.assertEqual(
            [(e.utterance_id, e.is_final, e.text) for e in envs],
            [(1, True, "cau mot."), (2, True, self.SENT), (2, False, "and more"), (2, True, "and more words")],
        )
        self.assertEqual([e.seq for e in envs], [1, 2, 3, 4])

    async def test_final_2_then_1_emits_1_then_2_with_segmentation_on_and_off(self):
        for segmentation in (False, True):
            with self.subTest(segmentation=segmentation):
                coordinator, rpc, key = await self._two_utterances(segmentation=segmentation)
                await self._final(coordinator, key, 2, "cau hai.")
                self.assertEqual(self._envs(rpc), [])
                await self._final(coordinator, key, 1, "cau mot.")
                envs = self._envs(rpc)
                self.assertEqual(
                    [(e.utterance_id, e.is_final, e.text) for e in envs], [(1, True, "cau mot."), (2, True, "cau hai.")]
                )
                self.assertEqual([e.seq for e in envs], [1, 2])

    async def test_empty_first_utterance_with_segmentation_on_and_off(self):
        for segmentation in (False, True):
            with self.subTest(segmentation=segmentation):
                coordinator, rpc, key = await self._two_utterances(segmentation=segmentation)
                await self._final(coordinator, key, 2, "cau hai.")
                await self._final(coordinator, key, 1, "")
                envs = self._envs(rpc)
                self.assertEqual([(e.utterance_id, e.is_final, e.text) for e in envs], [(2, True, "cau hai.")])
                self.assertEqual([e.seq for e in envs], [1])

    async def test_end_of_stream_while_waiting_closes_only_after_the_held_results_are_out(self):
        coordinator, rpc, key = await self._two_utterances()  # the last chunk carried end_of_stream
        await coordinator.close_input(rpc.rpc_id)
        await self._final(coordinator, key, 2, "cau hai")
        stream = coordinator._streams.get(key)
        self.assertIsNotNone(stream)  # utt1 still finishing: the stream stays open
        self.assertIs(stream.phase, StreamPhase.CLOSING)
        self.assertEqual(self._envs(rpc), [])
        self.assertFalse(rpc.response_buffer._closed)
        self.assertIn(rpc.rpc_id, coordinator._rpcs)

        await self._final(coordinator, key, 1, "cau mot")
        self.assertEqual([(e.utterance_id, e.text) for e in self._envs(rpc)], [(1, "cau mot"), (2, "cau hai")])
        self.assertNotIn(key, coordinator._streams)  # closed after both were emitted
        self.assertTrue(rpc.response_buffer._closed)
        self.assertNotIn(rpc.rpc_id, coordinator._rpcs)

    async def test_mo_and_mt_streams_are_ordered_independently(self):
        coordinator = EngineCoordinator(make_settings(False, max_utterance_ms=40))
        asr = FakeRuntime()
        coordinator.runtime = asr
        rpc = coordinator.open_rpc()

        def chunk(ts, stream_id, end=False):
            item = request(ts, end=end, language="en")
            item.stream_id = stream_id
            return item

        async def started(key, utterance_id):
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utterance_id)
            )

        await coordinator.accept_chunk(rpc.rpc_id, chunk(0, "MO"))
        mo = asr.commands[-1].key
        await started(mo, 1)
        await coordinator.accept_chunk(rpc.rpc_id, chunk(160, "MO"))  # MO utt1 -> FINALIZE
        await coordinator.accept_chunk(rpc.rpc_id, chunk(320, "MO"))
        await started(mo, 2)
        await coordinator.accept_chunk(rpc.rpc_id, chunk(480, "MO", end=True))  # MO utt2 -> FINALIZE
        await coordinator.accept_chunk(rpc.rpc_id, chunk(0, "MT"))
        mt = asr.commands[-1].key
        self.assertEqual((mo.stream_id, mt.stream_id), ("MO", "MT"))
        await started(mt, 1)
        await coordinator.accept_chunk(rpc.rpc_id, chunk(160, "MT", end=True))  # MT utt1 -> FINALIZE

        await self._final(coordinator, mo, 2, "mo hai")  # held behind MO utt1
        await self._final(coordinator, mt, 1, "mt mot")  # MT has nothing earlier: out at once
        self.assertEqual([(e.stream_id, e.utterance_id, e.text) for e in self._envs(rpc)], [("MT", 1, "mt mot")])
        await self._final(coordinator, mo, 1, "mo mot")

        envs = self._envs(rpc)
        self.assertEqual(
            [(e.stream_id, e.utterance_id, e.text) for e in envs],
            [("MT", 1, "mt mot"), ("MO", 1, "mo mot"), ("MO", 2, "mo hai")],
        )
        self.assertEqual([(e.stream_id, e.seq) for e in envs], [("MT", 1), ("MO", 1), ("MO", 2)])

    async def test_translate_source_partials_are_not_held(self):
        """The transcribe order does not touch translate: its source partials still go out at once."""
        settings = dataclasses.replace(
            make_settings(True, max_utterance_ms=40, use_language_source=True), translate_emit_source_partials=True
        )
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()

        def chunk(ts):
            return request(ts, task="translate", language="en", language_source="vi")

        await coordinator.accept_chunk(rpc.rpc_id, chunk(0))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1))
        await coordinator.accept_chunk(rpc.rpc_id, chunk(160))  # utt1 -> FINALIZE, not FINAL yet
        await coordinator.accept_chunk(rpc.rpc_id, chunk(320))
        await coordinator._handle_runtime_result(RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2))
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.PARTIAL, key=key, utterance_id=2, text="xin chao",
                metadata={"model_step_executed": True},
            )
        )
        self.assertEqual([(e.utterance_id, e.is_final, e.text) for e in self._envs(rpc)], [(2, False, "xin chao")])
        stream = coordinator._streams[key]
        self.assertEqual((stream.stt_emit_next, stream.stt_held), (1, {}))

    async def test_earlier_final_with_nothing_left_after_its_segments_releases_the_later_one(self):
        """utt1 already sent its sentence as a segment and its FINAL adds nothing: utt2 must still be released."""
        coordinator, rpc, key = await self._two_utterances(segmentation=True, utt1_partials=[self.SENT, self.SENT])
        self.assertEqual([(e.utterance_id, e.is_final, e.text) for e in self._envs(rpc)][-1], (1, True, self.SENT))
        await self._final(coordinator, key, 2, "cau hai.")
        self.assertNotIn(2, [e.utterance_id for e in self._envs(rpc)])
        await self._final(coordinator, key, 1, self.SENT)  # nothing after the committed sentence
        envs = self._envs(rpc)
        self.assertEqual([(e.utterance_id, e.is_final, e.text) for e in envs if e.is_final], [(1, True, self.SENT), (2, True, "cau hai.")])


class TtsPhraseContinuationTest(unittest.IsolatedAsyncioTestCase):
    async def test_continuation_phrase_survives_transient_full_queue(self):
        """A transiently full TTS queue must not drop the rest of a multi-phrase utterance.

        Phrase 0 is synthesised, then phrase 1 hits a full queue once. The
        utterance must still get all its audio (phrase 1 retried) instead of
        degrading to a text-only final after phrase 0 audio was already sent.
        """
        coordinator = EngineCoordinator(make_settings(True))
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = RejectOnceTtsRuntime(reject_phrase_id=1)
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào bạn")
        )

        translated = "Hello there! How are you?"
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1, source_text="Xin chào bạn", translated_text=translated,
                source_language="vi-VN", target_language="en-US", ts_start=0, ts_end=640,
            )
        )
        self.assertEqual([j.text for j in tts.jobs], ["Hello there!"])

        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="Hello there!", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=0, is_last_phrase=False, translated_text=translated,
            )
        )

        # Phrase 1 must eventually be submitted despite the first rejection.
        for _ in range(20):
            if len(tts.jobs) == 2:
                break
            await asyncio.sleep(0.05)
        self.assertTrue(tts.rejected)
        self.assertEqual(
            [j.text for j in tts.jobs], ["Hello there!", "How are you?"],
            "phrase 1 was dropped after a transient full TTS queue",
        )

        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="How are you?", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=1, is_last_phrase=True, translated_text=translated,
            )
        )

        first = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertFalse(first.is_final)
        self.assertEqual(len(first.audio_payload), 80)
        final = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, translated)
        self.assertEqual(len(final.audio_payload), 80)
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_tts_partial_carries_asr_confidence(self):
        coordinator = EngineCoordinator(make_settings(True))
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = RejectOnceTtsRuntime(reject_phrase_id=-1)
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL, key=key, utterance_id=1,
                text="Xin chào bạn", confidence=0.77,
            )
        )
        translated = "Hello there! How are you?"
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1, source_text="Xin chào bạn", translated_text=translated,
                source_language="vi-VN", target_language="en-US", ts_start=0, ts_end=640,
            )
        )
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="Hello there!", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=0, is_last_phrase=False, translated_text=translated,
            )
        )
        for _ in range(20):
            if len(coordinator.tts_runtime.jobs) == 2:
                break
            await asyncio.sleep(0.05)
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="How are you?", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=1, is_last_phrase=True, translated_text=translated,
            )
        )

        partial = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        final = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertFalse(partial.is_final)
        self.assertAlmostEqual(partial.confidence, 0.77, places=6)
        self.assertAlmostEqual(final.confidence, 0.77, places=6)

    async def test_continuation_degrades_after_retry_budget_exhausted(self):
        import dataclasses

        settings = dataclasses.replace(make_settings(True), tts_continuation_retry_ms=100.0)
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        tr = FakeTranslationRuntime()
        tts = RejectOnceTtsRuntime(reject_phrase_id=1)
        tts._should_reject = lambda job: job.phrase_id == 1  # queue never frees up
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào bạn")
        )
        translated = "Hello there! How are you?"
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1, source_text="Xin chào bạn", translated_text=translated,
                source_language="vi-VN", target_language="en-US", ts_start=0, ts_end=640,
            )
        )
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="Hello there!", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=0, is_last_phrase=False, translated_text=translated,
            )
        )

        first = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertFalse(first.is_final)
        final = await asyncio.wait_for(rpc.response_buffer.get(), timeout=1.0)
        self.assertTrue(final.is_final)
        self.assertEqual(final.text, translated)
        self.assertEqual(final.audio_payload, b"")
        self.assertEqual(len(tts.jobs), 1)


class PipelineTraceTest(unittest.IsolatedAsyncioTestCase):
    LOGGER = "nemotron_testbed.coordinator"

    async def _run(
        self,
        settings,
        translated="Hello",
        translate_error="",
        tts_error="",
        tts_empty=False,
    ):
        coordinator = EngineCoordinator(settings)
        asr, tr, tts = FakeRuntime(), FakeTranslationRuntime(), FakeTtsRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = tr
        coordinator.tts_runtime = tts
        self.coordinator = coordinator
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL,
                key=key,
                utterance_id=utt,
                text="Xin chào",
                confidence=0.84,
            )
        )
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key,
                utterance_id=utt,
                source_text="Xin chào",
                translated_text="" if translate_error else translated,
                source_language="vi-VN",
                target_language="en-US",
                ts_start=0,
                ts_end=640,
                latency_ms=12.34,
                error=translate_error,
            )
        )
        if tts.jobs:
            await coordinator._handle_tts_result(
                TtsResult(
                    key=key,
                    utterance_id=utt,
                    text=translated,
                    target_language="en-US",
                    ts_start=0,
                    ts_end=640,
                    audio_payload=b"" if (tts_error or tts_empty) else b"\xff" * 160,
                    audio_pt=0,
                    latency_ms=5.0,
                    error=tts_error,
                )
            )
        return key, utt

    @staticmethod
    def _by_stage(records):
        out = {}
        for rec in records:
            stage = getattr(rec, "stage", None)
            if stage:
                out.setdefault(stage, []).append(rec)
        return out

    async def test_each_stage_is_logged_with_stream_identity(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            key, utt = await self._run(make_settings(True))
        stages = self._by_stage(cm.records)
        self.assertEqual(
            set(stages),
            {
                "ASR_FINAL",
                "TRANSLATE_SUBMIT",
                "TRANSLATE",
                "TTS_SUBMIT",
                "TTS_PHRASE",
                "SEGMENT_LATENCY",
                "EMIT",
                "GRPC_OPEN",
                "STREAM_OPEN",
                "VAD_START",
                "STREAM_CLOSE",
            },
        )
        for stage, recs in stages.items():
            for rec in recs:
                self.assertEqual(rec.rpc_id, key.rpc_id)
                if stage == "GRPC_OPEN":
                    continue
                self.assertEqual(rec.session_id, key.session_id)
                self.assertEqual(rec.stream_id, key.stream_id)
                if stage not in ("STREAM_OPEN", "STREAM_CLOSE"):
                    self.assertEqual(rec.utterance_id, utt)

        asr = stages["ASR_FINAL"][0]
        self.assertEqual(asr.text, "Xin chào")
        self.assertEqual(asr.text_len, len("Xin chào"))
        self.assertEqual(asr.source_language, "vi-VN")
        self.assertEqual(asr.target_language, "en-US")
        self.assertAlmostEqual(asr.confidence, 0.84, places=6)

        tr = stages["TRANSLATE"][0]
        self.assertEqual(tr.source_text, "Xin chào")
        self.assertEqual(tr.translated_text, "Hello")
        self.assertEqual(tr.latency_ms, 12.3)
        self.assertFalse(hasattr(tr, "error"))

        tts = stages["TTS_PHRASE"][0]
        self.assertEqual(tts.phrase_id, 0)
        self.assertTrue(tts.is_last_phrase)
        self.assertEqual(tts.audio_bytes, 160)

        emit = stages["EMIT"][0]
        self.assertTrue(emit.is_final)
        self.assertEqual(emit.text, "Hello")
        self.assertEqual(emit.audio_bytes, 160)
        self.assertEqual(emit.seq, 1)
        self.assertTrue(emit.queued)

    async def test_trace_disabled_logs_nothing(self):
        settings = dataclasses.replace(make_settings(True), pipeline_trace=False)
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._run(settings)
        self.assertEqual(self._by_stage(cm.records), {})

    async def test_trace_level_setting_applies_to_stage_lines(self):
        settings = dataclasses.replace(make_settings(True), pipeline_trace_level="INFO")
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._run(settings)
        stage_records = [r for r in cm.records if getattr(r, "stage", None)]
        self.assertTrue(stage_records)
        self.assertEqual({r.levelno for r in stage_records}, {logging.INFO})

    async def test_trace_text_disabled_keeps_only_lengths(self):
        settings = dataclasses.replace(make_settings(True), pipeline_trace_text=False)
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._run(settings)
        stages = self._by_stage(cm.records)
        self.assertFalse(hasattr(stages["ASR_FINAL"][0], "text"))
        self.assertEqual(stages["ASR_FINAL"][0].text_len, len("Xin chào"))
        self.assertFalse(hasattr(stages["TRANSLATE"][0], "source_text"))
        self.assertFalse(hasattr(stages["TRANSLATE"][0], "translated_text"))
        self.assertEqual(stages["TRANSLATE"][0].translated_text_len, len("Hello"))
        self.assertFalse(hasattr(stages["EMIT"][0], "text"))
        self.assertEqual(stages["EMIT"][0].text_len, len("Hello"))

    async def test_degrade_reasons_are_logged_as_warnings(self):
        cases = {
            "translate_error": {"translate_error": "RuntimeError: boom"},
            "translate_empty": {"translated": ""},
            "tts_error": {"tts_error": "RuntimeError: onnx"},
        }
        for reason, kwargs in cases.items():
            with self.subTest(reason=reason):
                with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
                    await self._run(make_settings(True), **kwargs)
                degrade = self._by_stage(cm.records).get("DEGRADE", [])
                self.assertEqual([r.reason for r in degrade], [reason])
                self.assertEqual(degrade[0].levelname, "WARNING")
                self.assertTrue(degrade[0].text)

    async def test_submit_stages_carry_queue_and_phrase_info(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._run(make_settings(True), translated="Hello. World")
        stages = self._by_stage(cm.records)
        tr_submit = stages["TRANSLATE_SUBMIT"][0]
        self.assertEqual(tr_submit.source_text, "Xin chào")
        self.assertEqual(tr_submit.source_language, "vi-VN")
        self.assertEqual(tr_submit.target_language, "en-US")
        self.assertEqual(tr_submit.pending_postprocess, 1)
        self.assertGreaterEqual(tr_submit.asr_to_submit_ms, 0.0)
        tts_submit = stages["TTS_SUBMIT"][0]
        self.assertEqual(tts_submit.phrases, 2)
        self.assertEqual(tts_submit.text, "Hello.")
        self.assertEqual(tts_submit.target_language, "en-US")

    async def test_tts_empty_audio_is_reported_but_result_still_emitted(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._run(make_settings(True), tts_empty=True)
        stages = self._by_stage(cm.records)
        degrade = stages["DEGRADE"]
        self.assertEqual([r.reason for r in degrade], ["tts_empty_audio"])
        self.assertEqual(degrade[0].levelname, "WARNING")
        self.assertEqual(degrade[0].phrase_id, 0)
        self.assertEqual(stages["EMIT"][0].audio_bytes, 0)

    async def _partials(self, settings, results):
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        for text, confidence, model_step in results:
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL,
                    key=key,
                    utterance_id=utt,
                    text=text,
                    confidence=confidence,
                    metadata={"model_step_executed": model_step},
                )
            )

    async def test_asr_partial_trace_reports_stability_fields(self):
        settings = dataclasses.replace(make_settings(True), pipeline_trace_partial=True)
        results = [
            ("Xin chào", 0.5, True),
            ("Xin chào", 0.5, False),
            ("Xin chào các bạn. Hôm", 0.7, True),
            ("Xin chao các bạn. Hôm nay", 0.8, True),
        ]
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._partials(settings, results)
        partials = self._by_stage(cm.records)["ASR_PARTIAL"]
        self.assertEqual([r.step for r in partials], [1, 2, 3, 4])
        self.assertEqual([r.changed for r in partials], [True, False, True, True])
        self.assertEqual([r.model_step for r in partials], [True, False, True, True])
        self.assertFalse(hasattr(partials[1], "text"))
        self.assertEqual(partials[0].text, "Xin chào")
        self.assertEqual(partials[0].common_prefix_words, 0)
        self.assertEqual(partials[0].revised_words, 0)
        self.assertEqual(partials[2].common_prefix_words, 2)
        self.assertEqual(partials[2].revised_words, 0)
        self.assertEqual(partials[2].sentence_ends, 1)
        self.assertAlmostEqual(partials[2].confidence, 0.7, places=6)
        self.assertEqual(partials[3].common_prefix_words, 1)
        self.assertEqual(partials[3].revised_words, 4)
        self.assertGreaterEqual(partials[3].t_ms, partials[0].t_ms)

    async def test_asr_partial_trace_reports_infer_and_queue_ms(self):
        settings = dataclasses.replace(make_settings(True), pipeline_trace_partial=True)
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL,
                    key=key,
                    utterance_id=utt,
                    text="Xin chào",
                    latency_ms=27.34,
                    metadata={"model_step_executed": True, "queue_ms": 4.56},
                )
            )
        partial = self._by_stage(cm.records)["ASR_PARTIAL"][0]
        self.assertEqual(partial.infer_ms, 27.3)
        self.assertEqual(partial.queue_ms, 4.6)

    async def test_asr_partial_trace_is_off_by_default(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._partials(make_settings(True), [("Xin chào", 0.5, True)])
        self.assertNotIn("ASR_PARTIAL", self._by_stage(cm.records))

    async def test_asr_partial_trace_respects_text_flag(self):
        settings = dataclasses.replace(
            make_settings(True), pipeline_trace_partial=True, pipeline_trace_text=False
        )
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._partials(settings, [("Xin chào", 0.5, True)])
        partial = self._by_stage(cm.records)["ASR_PARTIAL"][0]
        self.assertFalse(hasattr(partial, "text"))
        self.assertEqual(partial.text_len, len("Xin chào"))

    async def test_asr_final_trace_carries_elapsed_time(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await self._run(make_settings(True))
        self.assertGreaterEqual(self._by_stage(cm.records)["ASR_FINAL"][0].t_ms, 0.0)

    async def _asr_final_only(self, *, text, language="", detected=""):
        coordinator = EngineCoordinator(make_settings(True))
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(
            rpc.rpc_id, request(0, task="translate", language=language)
        )
        key = asr.commands[-1].key
        utt = asr.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt)
        )
        await coordinator.accept_chunk(
            rpc.rpc_id, request(160, end=True, task="translate", language=language)
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL,
                key=key,
                utterance_id=utt,
                text=text,
                detected_language=detected,
            )
        )
        return coordinator

    async def _auto_utterances(self, *items, target="en"):
        """Run consecutive translate utterances toward `target`; items are (text, detected)."""
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=40))
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        ts = 0
        for index, (text, detected) in enumerate(items, start=1):
            await coordinator.accept_chunk(rpc.rpc_id, request(ts, task="translate", language=target))
            key = asr.commands[-1].key
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=index)
            )
            await coordinator.accept_chunk(
                rpc.rpc_id, request(ts + 160, task="translate", language=target)
            )
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.FINAL,
                    key=key,
                    utterance_id=index,
                    text=text,
                    detected_language=detected,
                )
            )
            ts += 320
        return coordinator

    def _lang_of_jobs(self, coordinator):
        return [(j.source_language, j.target_language) for j in coordinator.translation_runtime.jobs]

    async def test_unsupported_language_without_history_is_passed_through(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator = await self._auto_utterances(("konnichiwa", "ja-JP"))
        stages = self._by_stage(cm.records)
        self.assertNotIn("DEGRADE", stages)
        fallback = stages["LANG_FALLBACK"][0]
        self.assertEqual(fallback.fallback, "default")
        self.assertEqual(fallback.detected_language, "ja-JP")
        self.assertEqual([r.reason for r in stages["PASSTHROUGH"]], ["language_undetected"])
        self.assertEqual(self._lang_of_jobs(coordinator), [])

    async def test_unsupported_language_falls_back_to_last_stream_language(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator = await self._auto_utterances(
                ("It is late.", "en-US"),
                ("Actually, este. It should have arrived.", "pt-PT"),
                target="vi",
            )
        stages = self._by_stage(cm.records)
        self.assertNotIn("DEGRADE", stages)
        self.assertNotIn("PASSTHROUGH", stages)
        self.assertEqual([r.fallback for r in stages["LANG_FALLBACK"]], ["last_language"])
        self.assertEqual(
            self._lang_of_jobs(coordinator), [("en-US", "vi-VN"), ("en-US", "vi-VN")]
        )

    async def test_empty_detection_uses_last_stream_language(self):
        coordinator = await self._auto_utterances(
            ("It is late.", "en-US"), ("It should have arrived.", ""), target="vi"
        )
        self.assertEqual(
            self._lang_of_jobs(coordinator), [("en-US", "vi-VN"), ("en-US", "vi-VN")]
        )

    async def test_language_switch_between_vi_and_en_is_followed(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator = await self._auto_utterances(
                ("It is late.", "en-US"), ("Xin chào", "vi-VN"), ("Nó đã muộn", "pt-PT")
            )
        stages = self._by_stage(cm.records)
        self.assertEqual([r.reason for r in stages["PASSTHROUGH"]], ["source_equals_target"])
        self.assertEqual(
            self._lang_of_jobs(coordinator), [("vi-VN", "en-US"), ("vi-VN", "en-US")]
        )

    async def test_fallback_does_not_become_last_language(self):
        coordinator = await self._auto_utterances(("konnichiwa", "ja-JP"), ("Hello", "pt-PT"))
        stream = next(iter(coordinator._streams.values()))
        self.assertEqual(stream.last_language, "")

    async def test_empty_asr_text_is_traced_and_skips_translation(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator = await self._asr_final_only(text="", language="vi")
        stages = self._by_stage(cm.records)
        self.assertEqual([r.reason for r in stages["DEGRADE"]], ["asr_empty"])
        self.assertNotIn("TRANSLATE_SUBMIT", stages)
        self.assertEqual(coordinator.translation_runtime.jobs, [])

    async def test_translation_held_behind_inflight_tts_is_traced(self):
        from nemotron_testbed.coordinator import PipelineJobState, PipelineOrderState
        from nemotron_testbed.domain import StreamKey

        coordinator = EngineCoordinator(make_settings(True))
        coordinator.runtime = FakeRuntime()
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        key = StreamKey(rpc.rpc_id, "call-1", "MO")
        # Utterance 1 still owns the TTS slot, so utterance 2's translation must wait.
        coordinator._pipeline_order[key] = PipelineOrderState(
            next_emit_slot=(1, 0), next_tts_slot=(1, 0)
        )
        coordinator._pipeline_job_add(
            (key, 2, 0),
            PipelineJobState(
                key=key,
                utterance_id=2,
                source_text="Tạm biệt",
                source_language="vi-VN",
                target_language="en-US",
                ts_start=0,
                ts_end=640,
            ),
        )
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await coordinator._handle_translation_result(
                TranslationResult(
                    key=key,
                    utterance_id=2,
                    source_text="Tạm biệt",
                    translated_text="Goodbye",
                    source_language="vi-VN",
                    target_language="en-US",
                    ts_start=0,
                    ts_end=640,
                    latency_ms=1.0,
                )
            )
        hold = self._by_stage(cm.records)["ORDER_HOLD"][0]
        self.assertEqual(hold.utterance_id, 2)
        self.assertEqual(hold.waiting_for, 1)
        self.assertEqual(hold.held_count, 1)
        self.assertNotIn("TTS_SUBMIT", self._by_stage(cm.records))

    async def test_stale_results_are_traced_with_origin_and_reason(self):
        coordinator = EngineCoordinator(make_settings(True))
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        from nemotron_testbed.domain import StreamKey

        key = StreamKey(rpc.rpc_id, "call-1", "MO")
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            await coordinator._handle_translation_result(
                TranslationResult(
                    key=key,
                    utterance_id=7,
                    source_text="x",
                    translated_text="y",
                    source_language="vi-VN",
                    target_language="en-US",
                    ts_start=0,
                    ts_end=1,
                    latency_ms=1.0,
                )
            )
            await coordinator._handle_tts_result(
                TtsResult(
                    key=key,
                    utterance_id=7,
                    text="y",
                    target_language="en-US",
                    ts_start=0,
                    ts_end=1,
                    audio_payload=b"\x00",
                    audio_pt=0,
                    latency_ms=1.0,
                )
            )
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.FINAL, key=key, utterance_id=7, text="x"
                )
            )
        stale = self._by_stage(cm.records)["STALE"]
        self.assertEqual(
            [(r.origin, r.reason) for r in stale],
            [
                ("translate", "job_missing"),
                ("tts", "job_missing"),
                ("asr", "stream_gone"),
            ],
        )


class RtpClockRateTest(unittest.IsolatedAsyncioTestCase):
    """RTP clock-rate decoupling: rtp_clock_rate may differ from source_rate.

    G.711 example: PCM at 16 kHz, RTP clock at 8 kHz.
    A 20 ms chunk has 320 PCM samples but only 160 RTP ticks.
    The old code used source_samples as the RTP increment, so chunk #101
    at ts=T+160 was misclassified as out-of-order because the code expected
    ts=T+320.
    """

    def _coordinator(self, **overrides):
        import dataclasses
        settings = make_settings(False)
        if overrides:
            settings = dataclasses.replace(settings, **overrides)
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        return coordinator

    def _chunk(self, ts, seq=-1, clock=8000, end=False):
        """20 ms PCM chunk (320 samples @ 16 kHz) with an 8 kHz RTP clock by default."""
        return SimpleNamespace(
            session_id="s1",
            stream_id="A",
            pcm=np.zeros(320, dtype="<i2").tobytes(),
            sample_rate=16000,
            channels=1,
            rtp_timestamp=ts,
            rtp_clock_rate=clock,
            chunk_seq=seq,
            duration_ms=20,
            end_of_stream=end,
            language="vi",
            task="transcribe",
        )

    def _key(self, rpc):
        from nemotron_testbed.domain import StreamKey
        return StreamKey(rpc.rpc_id, "s1", "A")

    async def test_consecutive_chunks_8khz_clock_not_ooo(self):
        """ts increments by 160 ticks (20 ms × 8 kHz) — must NOT trigger out-of-order."""
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1160))
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1320))

        self.assertFalse(rpc.cancelled, "Consecutive 8 kHz-clock chunks must not be flagged as OOO")

    async def test_expected_rtp_timestamp_advances_by_rtp_ticks(self):
        """After a 20 ms chunk at 8 kHz clock: expected_next = ts + 160, not ts + 320."""
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))

        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.expected_rtp_timestamp, 1160)  # 1000 + 160, not 1000 + 320

    async def test_ts_start_and_ts_end_use_rtp_clock_rate(self):
        """ts_start / ts_end in the recognition result must be in RTP clock ticks.

        Setup (VAD disabled, vad_pre_roll_ms=0 so pre_roll = exactly 1 chunk):
          chunk 0 ts=1000 → START.  pre_roll=320 PCM → 160 RTP ticks, and it holds chunk 0 itself.
            ts_end   = 1000 + 160 = 1160
            ts_start = ts_end − 160 = 1000 (the pre-roll ends where chunk 0 ends)
          chunk 1 ts=1160, end_of_stream → ts_end = 1160 + 160 = 1320

        Counting the clock rate wrong gave ts_start = 680, ts_end = 1480; subtracting the pre-roll
        from the start of chunk 0 instead of its end gave ts_start = 840.
        """
        coordinator = self._coordinator()
        fake = coordinator.runtime
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))
        key = fake.commands[-1].key
        utt_id = fake.commands[-1].utterance_id
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=utt_id)
        )
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1160, end=True))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=utt_id,
                          text="x", confidence=0.9)
        )

        result = await rpc.response_buffer.get()
        self.assertIsInstance(result, RecognitionEnvelope)
        self.assertEqual(result.ts_start, 1000)  # start of chunk 0, not 1000 − 160
        self.assertEqual(result.ts_end,   1320)  # 1160 + 160, not 1160 + 320

    async def test_ts_start_of_a_first_chunk_at_zero_does_not_wrap(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(0))
        utt = coordinator._streams[self._key(rpc)].active_utterance
        self.assertEqual((utt.ts_start, utt.ts_end), (0, 160))  # was 2**32 − 160

    async def test_ts_start_is_the_first_chunk_in_a_multi_chunk_pre_roll(self):
        """Energy VAD, pre-roll 40 + min speech 40 = 80 ms = 4 chunks of 20 ms (16 kHz clock: 320 ticks each).

        At VAD START the pre-roll ends at the end of the START chunk k and holds chunks k-3..k, so
        ts_start is the timestamp of chunk k-3.
        """
        coordinator = self._coordinator(
            vad_enabled=True, vad_backend="energy", vad_min_speech_ms=40, vad_pre_roll_ms=40
        )
        rpc = coordinator.open_rpc()

        def chunk(index, amplitude):
            return SimpleNamespace(
                session_id="s1", stream_id="A",
                pcm=np.full(320, amplitude, dtype="<i2").tobytes(),
                sample_rate=16000, channels=1,
                rtp_timestamp=10_000 + index * 320, rtp_clock_rate=16000, chunk_seq=-1,
                duration_ms=20, end_of_stream=False, language="vi", task="transcribe",
            )

        start_index = None
        for index in range(12):
            await coordinator.accept_chunk(rpc.rpc_id, chunk(index, 0 if index < 6 else 1000))
            stream = coordinator._streams[self._key(rpc)]
            if stream.active_utterance is not None:
                start_index = index
                break
        self.assertIsNotNone(start_index)
        utt = stream.active_utterance
        self.assertEqual(utt.model_audio.samples, 4 * 320)
        self.assertEqual(utt.ts_end, 10_000 + (start_index + 1) * 320)
        self.assertEqual(utt.ts_start, 10_000 + (start_index - 3) * 320)

    async def test_rtp_gap_silence_pcm_samples_from_rtp_clock(self):
        """Gap of 160 RTP ticks @ 8 kHz = 20 ms → 320 PCM silence samples (not 160).

        Without fix: gap_ms = 160 ticks / 16000 = 10 ms → only 160 silence samples.
        """
        coordinator = self._coordinator(rtp_gap_policy="insert_silence", rtp_max_gap_ms=200)
        rpc = coordinator.open_rpc()

        # Chunk 0 (ts=1000): utterance starts; model_audio = 320 samples (pre_roll).
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))
        # Chunk 1 (ts=1160) missing.  Chunk 2 arrives at ts=1320.
        # RTP gap = 1320 − 1160 = 160 ticks @ 8 kHz → gap_ms = 20 ms → 320 PCM silence.
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1320))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        # model_audio: 320 (chunk 0) + 320 (silence) + 320 (chunk 2) = 960
        # Without fix:  320 + 160 (silence) + 320 = 800
        self.assertEqual(stream.active_utterance.model_audio.samples, 960)

    async def test_rtp_gap_with_8khz_input_counts_source_samples(self):
        """8 kHz input, 20 ms chunks, 1000 then 1320 (1160 missing, 20 ms gap): the chunk at 1320 ends at 1480.

        The silence added to the 16 kHz audio is 320 samples, but source_samples counts 8 kHz samples, so the gap
        adds 160 there. Before the fix it added 320, and ts_end came out 1640.
        """
        coordinator = self._coordinator(rtp_gap_policy="insert_silence", rtp_max_gap_ms=200)
        rpc = coordinator.open_rpc()

        def chunk_8k(ts):
            return SimpleNamespace(
                session_id="s1", stream_id="A",
                pcm=np.zeros(160, dtype="<i2").tobytes(),
                sample_rate=8000, channels=1,
                rtp_timestamp=ts, rtp_clock_rate=8000, chunk_seq=-1,
                duration_ms=20, end_of_stream=False, language="vi", task="transcribe",
            )

        await coordinator.accept_chunk(rpc.rpc_id, chunk_8k(1000))
        await coordinator.accept_chunk(rpc.rpc_id, chunk_8k(1320))

        self.assertFalse(rpc.cancelled)
        utt = coordinator._streams[self._key(rpc)].active_utterance
        self.assertEqual(utt.model_audio.samples, 960)  # 16 kHz: chunk + 20 ms silence + chunk
        self.assertEqual(utt.ts_end, 1480)

    async def test_rtp_gap_100ms_chunk_800_tick_gap(self):
        """User's example: chunk_seq 100→101→102 at 8 kHz RTP clock.

        chunk_seq   rtp_timestamp   rtp_clock
          100       188468360       8000
          101       188469160       8000   ← missing
          102       188469960       8000   ← arrives after gap

        Gap = 800 ticks @ 8 kHz = 100 ms → 1600 PCM silence samples inserted.
        Without fix: gap_ms = 800 / 16000 = 50 ms → only 800 silence samples.
        """
        coordinator = self._coordinator(rtp_gap_policy="insert_silence", rtp_max_gap_ms=500)
        rpc = coordinator.open_rpc()

        def chunk_100ms(ts):
            return SimpleNamespace(
                session_id="s1", stream_id="A",
                pcm=np.zeros(1600, dtype="<i2").tobytes(),
                sample_rate=16000, channels=1,
                rtp_timestamp=ts, rtp_clock_rate=8000, chunk_seq=-1,
                duration_ms=100, end_of_stream=False, language="vi", task="transcribe",
            )

        T = 188468360
        await coordinator.accept_chunk(rpc.rpc_id, chunk_100ms(T))          # chunk 100
        await coordinator.accept_chunk(rpc.rpc_id, chunk_100ms(T + 1600))   # chunk 102 (101 missing)

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        # model_audio: 1600 (chunk 100) + 1600 (silence) + 1600 (chunk 102) = 4800
        # Without fix:  1600 + 800 (silence, 50 ms) + 1600 = 4000
        self.assertEqual(stream.active_utterance.model_audio.samples, 4800)

    async def test_chunk_seq_gap_is_informational(self):
        """seq=100 then seq=103: 2 missing chunks logged, RPC not cancelled."""
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000, seq=100))
        # seq 101 and 102 missing; seq 103 arrives 3 chunks later (ts gap also expected).
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1480, seq=103))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.expected_chunk_seq, 104)

    async def test_chunk_seq_ooo_does_not_reset_expected(self):
        """seq=99 arriving after seq=101: expected_chunk_seq must not roll back."""
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000, seq=100))
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1160, seq=101))
        # seq=99 arrives late; also OOO in RTP (ts=840 < expected 1320 → dropped).
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(840, seq=99))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.expected_chunk_seq, 102)  # not reset by OOO

    async def test_rtp_clock_rate_change_raises(self):
        """rtp_clock_rate must remain constant within one logical stream."""
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000, clock=8000))
        with self.assertRaises(ValueError):
            await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1160, clock=16000))

    async def test_no_rtp_clock_rate_backward_compat(self):
        """Chunks without rtp_clock_rate default to 8000 (G.711 standard clock)."""
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()

        # request() has no rtp_clock_rate field → defaults to 8000 Hz.
        # ts must increment by 160 (= 20 ms × 8000 Hz) for continuity to hold.
        await coordinator.accept_chunk(rpc.rpc_id, request(0))
        await coordinator.accept_chunk(rpc.rpc_id, request(160))

        self.assertFalse(rpc.cancelled)
        from nemotron_testbed.domain import StreamKey
        stream = coordinator._streams[StreamKey(rpc.rpc_id, "call-1", "MO")]
        self.assertEqual(stream.rtp_clock_rate, 8000)
        self.assertEqual(stream.expected_rtp_timestamp, 320)  # 160 + 160

    async def test_jitter_tolerance_accepts_small_ooo(self):
        """Chunk arriving ≤ rtp_jitter_tolerance_ms behind expected is accepted, not dropped.

        Scenario: chunk 1 ts=1000 (expected_next=1160). Chunk 2 arrives at ts=1144,
        which is 16 ticks behind expected (= 2 ms at 8 kHz) — within the 20 ms
        tolerance (20 ms × 8 kHz / 1000 = 160 ticks). Must be accepted, and
        expected_rtp_timestamp moves to the end of chunk 2 (1144 + 160 = 1304),
        so the next chunk, contiguous with it, is not misread as a forward gap.
        """
        coordinator = self._coordinator(rtp_jitter_tolerance_ms=20)
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1144))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.expected_rtp_timestamp, 1304)
        self.assertEqual(stream.last_rtp_timestamp, 1144)

    async def test_jitter_chunk_then_contiguous_chunk_inserts_no_silence(self):
        """1000 -> 1144 (2 ms behind, within jitter) -> 1304 (contiguous with chunk 2): 60 ms reach ASR.

        Before the fix expected stayed 1160, so 1304 looked like an 18 ms gap and 18 ms of
        silence were inserted (78 ms in total).
        """
        coordinator = self._coordinator(rtp_jitter_tolerance_ms=20, rtp_gap_policy="insert_silence")
        rpc = coordinator.open_rpc()

        for ts in (1000, 1144, 1304):
            await coordinator.accept_chunk(rpc.rpc_id, self._chunk(ts))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.active_utterance.model_audio.samples, 3 * 320)
        self.assertEqual(stream.expected_rtp_timestamp, 1464)

    async def test_duplicate_within_jitter_does_not_move_expected_back(self):
        """A repeated chunk (its end is not past expected) leaves expected where it was."""
        coordinator = self._coordinator(rtp_jitter_tolerance_ms=20, rtp_gap_policy="insert_silence")
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1160))
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1160))  # repeated: accepted as jitter
        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.expected_rtp_timestamp, 1320)
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1320))

        self.assertFalse(rpc.cancelled)
        self.assertEqual(stream.active_utterance.model_audio.samples, 4 * 320)  # no silence inserted
        self.assertEqual(stream.expected_rtp_timestamp, 1480)

    async def test_ooo_accept_chunk_past_expected_moves_expected(self):
        """out_of_order_policy=accept, 80 ms chunks (640 ticks): 1000 -> 1400 -> 2040.

        1400 is 240 ticks (30 ms) behind expected 1640, beyond the 20 ms jitter tolerance, so it
        goes to the accept branch; it ends at 2040, past 1640, so 2040 is contiguous. Before the
        fix expected stayed 1640 and 50 ms of silence (800 samples) were inserted before 2040.
        """
        coordinator = self._coordinator(
            rtp_jitter_tolerance_ms=20, rtp_gap_policy="insert_silence", rtp_out_of_order_policy="accept"
        )
        rpc = coordinator.open_rpc()

        def chunk_80ms(ts):
            return SimpleNamespace(
                session_id="s1", stream_id="A",
                pcm=np.zeros(1280, dtype="<i2").tobytes(),
                sample_rate=16000, channels=1,
                rtp_timestamp=ts, rtp_clock_rate=8000, chunk_seq=-1,
                duration_ms=80, end_of_stream=False, language="vi", task="transcribe",
            )

        for ts in (1000, 1400, 2040):
            await coordinator.accept_chunk(rpc.rpc_id, chunk_80ms(ts))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        self.assertEqual(stream.active_utterance.model_audio.samples, 3 * 1280)
        self.assertEqual(stream.expected_rtp_timestamp, 2680)

    async def test_jitter_tolerance_drops_large_ooo(self):
        """Chunk arriving beyond jitter tolerance is still dropped (OOO policy=drop)."""
        coordinator = self._coordinator(rtp_jitter_tolerance_ms=20)
        rpc = coordinator.open_rpc()

        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1000))
        # Arrive 200 ticks (25 ms) early — exceeds 160-tick (20 ms) jitter window.
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(960))

        self.assertFalse(rpc.cancelled)
        stream = coordinator._streams[self._key(rpc)]
        # expected_rtp_timestamp must not advance past chunk 1's end.
        self.assertEqual(stream.expected_rtp_timestamp, 1160)  # unchanged after drop


class VadSilenceDiagnosticsTest(unittest.IsolatedAsyncioTestCase):
    LOGGER = "nemotron_testbed.coordinator"

    def _coordinator(self, **overrides):
        import dataclasses
        fields = dict(
            vad_enabled=True,
            vad_backend="energy",
            vad_min_speech_ms=40,
            vad_no_start_warn_ms=60,
        )
        fields.update(overrides)
        settings = dataclasses.replace(make_settings(False), **fields)
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        return coordinator

    def _chunk(self, index, amplitude):
        return SimpleNamespace(
            session_id="s1",
            stream_id="A",
            pcm=np.full(320, amplitude, dtype="<i2").tobytes(),
            sample_rate=16000,
            channels=1,
            rtp_timestamp=index * 320,
            rtp_clock_rate=16000,
            duration_ms=20,
            end_of_stream=False,
            language="vi",
            task="transcribe",
        )

    @staticmethod
    def _vad_warnings(cm):
        return [r for r in cm.records if r.levelname == "WARNING" and hasattr(r, "vad_max_score")]

    async def test_warns_once_when_vad_never_starts(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            for i in range(6):
                await coordinator.accept_chunk(rpc.rpc_id, self._chunk(i, 5))
        warnings = self._vad_warnings(cm)
        self.assertEqual(len(warnings), 1)
        record = warnings[0]
        self.assertEqual(record.stream_id, "A")
        self.assertEqual(record.audio_ms, 60.0)
        self.assertLess(record.vad_max_score, 0.01)
        self.assertAlmostEqual(record.peak_dbfs, 20 * math.log10(5 / 32768), places=1)
        self.assertEqual(record.sample_rate, 16000)

    async def test_no_warning_when_vad_starts(self):
        coordinator = self._coordinator()
        rpc = coordinator.open_rpc()
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            for i in range(6):
                await coordinator.accept_chunk(rpc.rpc_id, self._chunk(i, 1000))
        self.assertEqual(self._vad_warnings(cm), [])

    async def test_warning_disabled_by_zero(self):
        coordinator = self._coordinator(vad_no_start_warn_ms=0)
        rpc = coordinator.open_rpc()
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            for i in range(6):
                await coordinator.accept_chunk(rpc.rpc_id, self._chunk(i, 5))
        self.assertEqual(self._vad_warnings(cm), [])


class SpeechOnsetTest(unittest.IsolatedAsyncioTestCase):
    """first_audio_from_speech_ms starts at the first chunk of the VAD speech run."""

    def _coordinator(self, **overrides):
        import dataclasses
        fields = dict(vad_enabled=True, vad_backend="energy", vad_min_speech_ms=40)
        fields.update(overrides)
        coordinator = EngineCoordinator(dataclasses.replace(make_settings(False), **fields))
        coordinator.runtime = FakeRuntime()
        return coordinator

    @staticmethod
    def _chunk(index, amplitude):
        return SimpleNamespace(
            session_id="s1",
            stream_id="A",
            pcm=np.full(320, amplitude, dtype="<i2").tobytes(),
            sample_rate=16000,
            channels=1,
            rtp_timestamp=index * 320,
            rtp_clock_rate=16000,
            duration_ms=20,
            end_of_stream=False,
            language="vi",
            task="transcribe",
        )

    def _stream(self, coordinator):
        return next(iter(coordinator._streams.values()))

    async def test_lead_covers_the_chunks_before_vad_start(self):
        coordinator = self._coordinator(vad_min_speech_ms=60)
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(0, 1000))
        stream = self._stream(coordinator)
        self.assertGreater(stream.speech_onset_wall, 0.0)
        await asyncio.sleep(0.05)
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000))
        await asyncio.sleep(0.05)
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(2, 1000))
        utt = stream.active_utterance
        self.assertIsNotNone(utt)
        self.assertGreaterEqual(utt.speech_lead_ms, 90.0)
        self.assertEqual(stream.speech_onset_wall, 0.0)

    async def test_silence_resets_the_onset(self):
        coordinator = self._coordinator(vad_min_speech_ms=60)
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(0, 1000))
        stream = self._stream(coordinator)
        self.assertGreater(stream.speech_onset_wall, 0.0)
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 0))
        self.assertEqual(stream.speech_onset_wall, 0.0)

    async def test_lead_is_zero_when_vad_is_disabled(self):
        coordinator = self._coordinator(vad_enabled=False)
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(0, 1000))
        utt = self._stream(coordinator).active_utterance
        self.assertIsNotNone(utt)
        self.assertEqual(utt.speech_lead_ms, 0.0)


class _TailVad:
    needs_executor = False
    speech_candidate_ms = 0.0
    max_score = 0.0

    def __init__(self, events, end_tail_ms=0.0, end_silence_observed_ms=0.0):
        self.events = list(events)
        self.end_tail_ms = end_tail_ms
        self.end_silence_observed_ms = end_silence_observed_ms
        self.seen = []

    def process(self, audio, duration_ms):
        self.seen.append((audio.size, duration_ms))
        return self.events.pop(0) if self.events else VadEvent.NONE

    def reset(self):
        self.reset_calls = getattr(self, "reset_calls", 0) + 1


class EndTailTest(unittest.IsolatedAsyncioTestCase):
    """Audio after the VAD END window goes to the next utterance, not the finished one."""

    async def _setup(self):
        import dataclasses

        settings = dataclasses.replace(
            make_settings(False), vad_enabled=True, vad_backend="energy", vad_min_speech_ms=0, vad_pre_roll_ms=200
        )
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(0, 1000))
        stream = next(iter(coordinator._streams.values()))
        self.assertIsNotNone(stream.active_utterance)
        return coordinator, rpc, stream

    @staticmethod
    def _chunk(index, amplitude, end=False):
        return SimpleNamespace(
            session_id="s1",
            stream_id="A",
            pcm=np.full(320, amplitude, dtype="<i2").tobytes(),
            sample_rate=16000,
            channels=1,
            rtp_timestamp=index * 320,
            rtp_clock_rate=16000,
            duration_ms=20,
            end_of_stream=end,
            language="vi",
            task="transcribe",
        )

    async def test_tail_goes_to_pre_roll_and_new_vad(self):
        coordinator, rpc, stream = await self._setup()
        utt = stream.active_utterance
        samples_before = utt.samples
        ts_end_before = utt.ts_end
        vad = _TailVad([VadEvent.END, VadEvent.NONE], end_tail_ms=5.0)
        stream.vad = vad
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000))

        self.assertTrue(utt.final_requested)
        self.assertIsNone(stream.active_utterance)
        self.assertEqual(utt.samples - samples_before, 240)
        self.assertEqual(utt.ts_end, 320 + 240)
        self.assertEqual(vad.seen, [(320, 20), (80, 5.0)])
        self.assertEqual(stream.pre_roll.samples, 80)
        self.assertGreater(utt.ts_end, ts_end_before)

    async def test_speech_end_wall_is_vad_end_minus_silence_and_tail(self):
        coordinator, rpc, stream = await self._setup()
        utt = stream.active_utterance
        stream.vad = _TailVad([VadEvent.END, VadEvent.NONE], end_tail_ms=5.0, end_silence_observed_ms=250.0)
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000))

        self.assertGreater(utt.vad_end_wall, 0.0)
        self.assertAlmostEqual(utt.vad_end_wall - utt.speech_end_wall, 0.255, places=6)

    async def test_vad_end_is_traced_with_silence_length(self):
        coordinator, rpc, stream = await self._setup()
        stream.vad = _TailVad([VadEvent.END, VadEvent.NONE], end_tail_ms=5.0, end_silence_observed_ms=250.0)
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000))

        ends = [r for r in cm.records if getattr(r, "stage", None) == "VAD_END"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0].utterance_id, 1)
        self.assertEqual(ends[0].session_id, "s1")
        self.assertAlmostEqual(ends[0].vad_silence_ms, 255.0, places=1)
        self.assertGreater(ends[0].utterance_audio_ms, 0.0)

    async def test_lifecycle_lines_carry_stage_only_when_trace_enabled(self):
        import dataclasses

        for trace in (True, False):
            with self.subTest(pipeline_trace=trace):
                coordinator = EngineCoordinator(dataclasses.replace(make_settings(False), pipeline_trace=trace))
                coordinator.runtime = FakeRuntime()
                with self.assertLogs("nemotron_testbed.coordinator", level="INFO") as cm:
                    rpc = coordinator.open_rpc()
                    await coordinator.accept_chunk(rpc.rpc_id, self._chunk(0, 0))
                    await coordinator.cancel_rpc(rpc.rpc_id, "test cancel")
                stages = {r.getMessage().split(":")[0]: getattr(r, "stage", None) for r in cm.records}
                if trace:
                    self.assertEqual(stages["RPC opened"], "GRPC_OPEN")
                    self.assertEqual(stages["Logical stream opened"], "STREAM_OPEN")
                    self.assertEqual(stages["RPC cancelled"], "GRPC_CANCEL")
                    cancel = next(r for r in cm.records if r.getMessage().startswith("RPC cancelled"))
                    self.assertEqual(cancel.reason, "test cancel")
                else:
                    self.assertTrue(all(stage is None for stage in stages.values()), stages)

    async def test_speech_end_wall_stays_zero_without_vad_end(self):
        coordinator, rpc, stream = await self._setup()
        utt = stream.active_utterance
        stream.vad = _TailVad([VadEvent.SPEECH], end_silence_observed_ms=250.0)
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000))

        self.assertEqual(utt.speech_end_wall, 0.0)

    async def test_no_tail_keeps_the_whole_chunk_in_the_finished_utterance(self):
        coordinator, rpc, stream = await self._setup()
        utt = stream.active_utterance
        samples_before = utt.samples
        vad = _TailVad([VadEvent.END], end_tail_ms=0.0)
        stream.vad = vad
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000))

        self.assertEqual(utt.samples - samples_before, 320)
        self.assertEqual(vad.seen, [(320, 20)])
        self.assertEqual(stream.pre_roll.samples, 0)

    async def test_end_of_stream_chunk_is_not_split(self):
        coordinator, rpc, stream = await self._setup()
        utt = stream.active_utterance
        samples_before = utt.samples
        vad = _TailVad([VadEvent.END], end_tail_ms=5.0)
        stream.vad = vad
        await coordinator.accept_chunk(rpc.rpc_id, self._chunk(1, 1000, end=True))

        self.assertEqual(utt.samples - samples_before, 320)
        self.assertEqual(vad.seen, [(320, 20)])


class EmptyResultSkipTest(unittest.IsolatedAsyncioTestCase):
    async def _setup(self):
        coordinator = EngineCoordinator(make_settings(False))
        coordinator.runtime = FakeRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0))
        return coordinator, rpc, coordinator.runtime.commands[-1].key

    def _emit(self, coordinator, key, **kw):
        base = dict(
            key=key, utterance_id=1, text="", is_final=True, ts_start=0, ts_end=1, language="vi-VN"
        )
        base.update(kw)
        coordinator._emit_recognition(**base)

    async def test_empty_text_and_audio_is_not_sent(self):
        coordinator, rpc, key = await self._setup()
        self._emit(coordinator, key)
        self._emit(coordinator, key, is_final=False)
        self.assertEqual(len(rpc.response_buffer._items), 0)

    async def test_text_only_and_audio_only_are_sent_with_contiguous_seq(self):
        coordinator, rpc, key = await self._setup()
        self._emit(coordinator, key, text="xin chào")
        self._emit(coordinator, key)
        self._emit(coordinator, key, is_final=False, audio_payload=b"" * 160, audio_pt=0)
        self._emit(coordinator, key, text="tạm biệt", audio_payload=b"" * 160)
        items = [await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2) for _ in range(3)]
        self.assertEqual([i.seq for i in items], [1, 2, 3])
        self.assertEqual([bool(i.text) for i in items], [True, False, True])
        self.assertEqual([bool(i.audio_payload) for i in items], [False, True, True])
        self.assertEqual(len(rpc.response_buffer._items), 0)

    async def test_skip_is_traced_and_counted(self):
        coordinator, rpc, key = await self._setup()
        before = EMPTY_RESULTS_SKIPPED._value.get()
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            self._emit(coordinator, key)
        self.assertEqual(EMPTY_RESULTS_SKIPPED._value.get(), before + 1)
        emits = [r for r in cm.records if getattr(r, "stage", None) == "EMIT"]
        self.assertEqual(len(emits), 1)
        self.assertEqual(emits[0].skipped, "empty")


class DownstreamTraceTimingTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_stages_carry_t_ms_and_every_phrase_is_submitted_in_trace(self):
        coordinator = EngineCoordinator(make_settings(True))
        asr, tr = FakeRuntime(), FakeTranslationRuntime()
        tts = RejectOnceTtsRuntime(reject_phrase_id=1)
        coordinator.runtime, coordinator.translation_runtime, coordinator.tts_runtime = asr, tr, tts
        rpc = coordinator.open_rpc()
        translated = "Hello there! How are you?"

        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
            key = asr.commands[-1].key
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
            )
            await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào bạn")
            )
            await coordinator._handle_translation_result(
                TranslationResult(
                    key=key, utterance_id=1, source_text="Xin chào bạn", translated_text=translated,
                    source_language="vi-VN", target_language="en-US", ts_start=0, ts_end=640,
                )
            )
            await coordinator._handle_tts_result(
                TtsResult(
                    key=key, utterance_id=1, text="Hello there!", target_language="en-US",
                    ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                    phrase_id=0, is_last_phrase=False, translated_text=translated,
                )
            )
            for _ in range(20):
                if len(tts.jobs) == 2:
                    break
                await asyncio.sleep(0.05)
            await coordinator._handle_tts_result(
                TtsResult(
                    key=key, utterance_id=1, text="How are you?", target_language="en-US",
                    ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                    phrase_id=1, is_last_phrase=True, translated_text=translated,
                )
            )

        records = [r for r in cm.records if getattr(r, "stage", None)]
        submits = [r for r in records if r.stage == "TTS_SUBMIT"]
        self.assertEqual([r.phrase_id for r in submits], [0, 1])
        self.assertEqual([r.phrases for r in submits], [2, 2])
        self.assertEqual(submits[0].phrase_texts, ["Hello there!", "How are you?"])
        for stage in ("TRANSLATE_SUBMIT", "TRANSLATE", "TTS_SUBMIT", "TTS_PHRASE", "EMIT"):
            stage_records = [r for r in records if r.stage == stage]
            self.assertTrue(stage_records, stage)
            for rec in stage_records:
                self.assertGreaterEqual(rec.t_ms, 0.0, f"{stage} missing t_ms")
        emits = [r for r in records if r.stage == "EMIT"]
        self.assertEqual([r.is_final for r in emits], [False, True])


    async def test_phrase_texts_hidden_when_text_tracing_is_off(self):
        coordinator = EngineCoordinator(dataclasses.replace(make_settings(True), pipeline_trace_text=False))
        key = StreamKey("r1", "s1", "MO")
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            coordinator._trace("TTS_SUBMIT", key, 1, phrases=2, phrase_texts=["a b", "c d"], text="a b")
        record = cm.records[0]
        self.assertEqual(record.phrases, 2)
        self.assertEqual(record.text_len, 3)
        self.assertFalse(hasattr(record, "phrase_texts"))
        self.assertFalse(hasattr(record, "text"))


class SegmentLatencyTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, settings):
        coordinator = EngineCoordinator(settings)
        asr, tr, tts = FakeRuntime(), FakeTranslationRuntime(), FakeTtsRuntime()
        coordinator.runtime, coordinator.translation_runtime, coordinator.tts_runtime = asr, tr, tts
        rpc = coordinator.open_rpc()
        translated = "Hello there! How are you?"
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào bạn")
        )
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1, source_text="Xin chào bạn", translated_text=translated,
                source_language="vi-VN", target_language="en-US", ts_start=0, ts_end=640,
                latency_ms=11.0, queue_ms=4.0,
            )
        )
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="Hello there!", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=0, is_last_phrase=False, translated_text=translated,
                latency_ms=7.0, queue_ms=3.0,
            )
        )
        await asyncio.sleep(0)
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="How are you?", target_language="en-US",
                ts_start=0, ts_end=640, audio_payload=b"\xff" * 80, audio_pt=0,
                phrase_id=1, is_last_phrase=True, translated_text=translated,
                latency_ms=99.0, queue_ms=88.0,
            )
        )

    @staticmethod
    def _count(phase):
        return sum(b.get() for b in SEGMENT_PHASE_MS.labels(phase=phase)._buckets)

    async def test_line_emitted_once_with_first_phrase_tts_timing(self):
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await self._run(make_settings(True))
        lines = [r for r in cm.records if getattr(r, "stage", None) == "SEGMENT_LATENCY"]
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertEqual(line.phrases, 2)
        self.assertEqual(line.translate_ms, 11.0)
        self.assertEqual(line.translate_queue_ms, 4.0)
        self.assertEqual(line.tts_ms, 7.0)
        self.assertEqual(line.tts_queue_ms, 3.0)
        self.assertGreaterEqual(line.order_hold_ms, 0.0)
        self.assertGreaterEqual(line.total_ms, 0.0)
        self.assertFalse(hasattr(line, "asr_ms"))
        stages = [getattr(r, "stage", None) for r in cm.records]
        self.assertLess(stages.index("SEGMENT_LATENCY"), stages.index("EMIT"))

    async def test_asr_phase_reported_when_known_and_once_per_segment(self):
        from nemotron_testbed.coordinator import PipelineJobState

        coordinator = EngineCoordinator(make_settings(True))
        key = StreamKey("r1", "s1", "M0")
        now = time.monotonic()
        job = PipelineJobState(
            key=key, utterance_id=1, source_text="a", source_language="vi-VN",
            target_language="en-US", ts_start=0, ts_end=1, segment_idx=2,
            submit_wall=now - 0.5, asr_ms=420.5, translate_ms=10.0, tts_ms=20.0,
        )
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            coordinator._observe_segment_latency(job)
            coordinator._observe_segment_latency(job)
        lines = [r for r in cm.records if getattr(r, "stage", None) == "SEGMENT_LATENCY"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].asr_ms, 420.5)
        self.assertEqual(lines[0].segment_idx, 2)
        self.assertGreaterEqual(lines[0].total_ms, 500.0)

    async def test_histogram_observed_without_trace(self):
        before = {p: self._count(p) for p in ("translate", "tts", "total", "order_hold")}
        settings = dataclasses.replace(make_settings(True), pipeline_trace=False)
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await self._run(settings)
        self.assertFalse(any(getattr(r, "stage", None) == "SEGMENT_LATENCY" for r in cm.records))
        for phase, count in before.items():
            self.assertEqual(self._count(phase), count + 1, phase)


class AudioChunkSplitTest(unittest.IsolatedAsyncioTestCase):
    async def _setup(self, chunk_bytes):
        settings = dataclasses.replace(make_settings(False), tts_max_audio_chunk_bytes=chunk_bytes)
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0))
        return coordinator, rpc, coordinator.runtime.commands[-1].key

    def _emit(self, coordinator, key, **kw):
        base = dict(key=key, utterance_id=1, text="", is_final=True, ts_start=0, ts_end=1, language="vi-VN", audio_pt=0)
        base.update(kw)
        coordinator._emit_recognition(**base)

    async def _drain(self, rpc):
        items = []
        while rpc.response_buffer._items:
            items.append(await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2))
        return items

    async def test_large_final_is_split_and_text_and_final_only_on_last_chunk(self):
        coordinator, rpc, key = await self._setup(1600)
        payload = bytes(range(256)) * 13  # 3328 bytes -> 1600 + 1600 + 128
        self._emit(coordinator, key, text="xin chào", audio_payload=payload)
        items = await self._drain(rpc)
        self.assertEqual([len(i.audio_payload) for i in items], [1600, 1600, 128])
        self.assertEqual([i.seq for i in items], [1, 2, 3])
        self.assertEqual([i.is_final for i in items], [False, False, True])
        self.assertEqual([i.text for i in items], ["", "", "xin chào"])
        self.assertEqual(b"".join(i.audio_payload for i in items), payload)

    async def test_large_intermediate_phrase_stays_non_final(self):
        coordinator, rpc, key = await self._setup(1600)
        self._emit(coordinator, key, is_final=False, audio_payload=b"a" * 3200)
        items = await self._drain(rpc)
        self.assertEqual([len(i.audio_payload) for i in items], [1600, 1600])
        self.assertEqual([i.is_final for i in items], [False, False])

    async def test_payload_at_or_below_limit_is_not_split(self):
        coordinator, rpc, key = await self._setup(1600)
        self._emit(coordinator, key, text="a", audio_payload=b"a" * 1600)
        items = await self._drain(rpc)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].is_final)

    async def test_zero_disables_splitting(self):
        coordinator, rpc, key = await self._setup(0)
        self._emit(coordinator, key, text="a", audio_payload=b"a" * 100_000)
        items = await self._drain(rpc)
        self.assertEqual(len(items), 1)

    async def test_e2e_histogram_counts_final_once(self):
        coordinator, rpc, key = await self._setup(1600)
        before = UTTERANCE_E2E_MS._sum.get()
        count_before = sum(b.get() for b in UTTERANCE_E2E_MS._buckets)
        self._emit(coordinator, key, text="a", audio_payload=b"a" * 4800, wall_start=1.0)
        self.assertEqual(sum(b.get() for b in UTTERANCE_E2E_MS._buckets), count_before + 1)

    async def test_each_chunk_is_traced_with_audio_bytes(self):
        coordinator, rpc, key = await self._setup(1600)
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            self._emit(coordinator, key, text="a", audio_payload=b"a" * 3200)
        emits = [r for r in cm.records if getattr(r, "stage", None) == "EMIT"]
        self.assertEqual([r.audio_bytes for r in emits], [1600, 1600])
        self.assertEqual([r.is_final for r in emits], [False, True])


class ShadowSegmentationTest(unittest.IsolatedAsyncioTestCase):
    SENT = "This is a complete sentence here."

    async def _run(self, mode, *, task="translate", partials=None, final=None):
        coordinator = EngineCoordinator(make_settings(True, segmentation_mode=mode))
        asr, tr = FakeRuntime(), FakeTranslationRuntime()
        coordinator.runtime, coordinator.translation_runtime = asr, tr
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        partials = partials if partials is not None else [self.SENT, self.SENT, self.SENT + " and more"]
        final = final if final is not None else self.SENT + " and more words"
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await coordinator.accept_chunk(rpc.rpc_id, request(0, task=task, language="en"))
            key = asr.commands[-1].key
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
            )
            for text in partials:
                await coordinator._handle_runtime_result(
                    RuntimeResult(
                        type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                        metadata={"model_step_executed": True},
                    )
                )
            await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task=task, language="en"))
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=final)
            )
        return coordinator, tr, [r for r in cm.records if getattr(r, "stage", "").startswith("SEGMENT_")]

    async def test_off_emits_no_segment_events(self):
        _, tr, records = await self._run("off")
        self.assertEqual(records, [])
        self.assertEqual(len(tr.jobs), 1)

    async def test_shadow_logs_commits_and_summary_without_changing_output(self):
        _, tr, records = await self._run("shadow")
        commits = [r for r in records if r.stage == "SEGMENT_COMMIT"]
        self.assertEqual([c.kind for c in commits], ["punct", "utterance_end"])
        self.assertEqual(commits[0].text, self.SENT)
        self.assertFalse(commits[0].forced_cut)
        summaries = [r for r in records if r.stage == "SEGMENT_SUMMARY"]
        self.assertEqual(len(summaries), 1)
        self.assertFalse(summaries[0].premature)
        self.assertGreaterEqual(summaries[0].gain_ms, 0.0)
        self.assertEqual(len(tr.jobs), 1)
        self.assertEqual(tr.jobs[0].source_text, self.SENT + " and more words")

    async def test_shadow_reports_premature_when_final_rewrites_committed_text(self):
        _, _, records = await self._run("shadow", final="This was a complete sentence here. And more")
        summary = [r for r in records if r.stage == "SEGMENT_SUMMARY"][0]
        self.assertTrue(summary.premature)
        self.assertTrue(summary.revised)
        self.assertTrue([r for r in records if r.stage == "SEGMENT_REVISED"])

    async def test_on_mode_submit_reports_seen_to_submit_ms_for_committed_sentence(self):
        coordinator = EngineCoordinator(make_settings(True, segmentation_mode="on"))
        asr, tr = FakeRuntime(), FakeTranslationRuntime()
        coordinator.runtime, coordinator.translation_runtime = asr, tr
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await coordinator.accept_chunk(rpc.rpc_id, request(0, task="translate", language="en"))
            key = asr.commands[-1].key
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
            )
            for text in (self.SENT, self.SENT, self.SENT + " and more"):
                await coordinator._handle_runtime_result(
                    RuntimeResult(
                        type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                        metadata={"model_step_executed": True},
                    )
                )
            await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, task="translate", language="en"))
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=self.SENT + " and more words")
            )
        submits = [r for r in cm.records if getattr(r, "stage", "") == "TRANSLATE_SUBMIT"]
        self.assertEqual(len(submits), 2)
        self.assertGreaterEqual(submits[0].seen_to_submit_ms, 0.0)
        self.assertFalse(hasattr(submits[1], "seen_to_submit_ms"))

    async def test_shadow_ignores_speech_to_text_task(self):
        _, _, records = await self._run("shadow", task="transcribe")
        self.assertEqual(records, [])

    async def test_segmenter_error_disables_it_without_breaking_pipeline(self):
        from unittest.mock import patch

        before = SEGMENT_ERRORS._value.get()
        with patch(
            "nemotron_testbed.coordinator.StableSentenceSegmenter.update",
            side_effect=RuntimeError("boom"),
        ):
            _, tr, records = await self._run("shadow")
        self.assertEqual([r for r in records if r.stage == "SEGMENT_COMMIT"], [])
        self.assertEqual(len(tr.jobs), 1)
        self.assertEqual(SEGMENT_ERRORS._value.get(), before + 1)


class SpeechToTextSegmentationTest(unittest.IsolatedAsyncioTestCase):
    SENT = "This is a complete sentence here."

    async def _run(self, mode, flag, *, partials=None, final=None):
        settings = dataclasses.replace(
            make_settings(False, segmentation_mode=mode), segmentation_speech_to_text=flag
        )
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        rpc = coordinator.open_rpc()
        partials = partials if partials is not None else [self.SENT, self.SENT, self.SENT + " and more"]
        final = final if final is not None else self.SENT + " and more words"
        await coordinator.accept_chunk(rpc.rpc_id, request(0, language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        for text in partials:
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                    metadata={"model_step_executed": True},
                )
            )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=final)
        )
        return [item for item in rpc.response_buffer._items if isinstance(item, RecognitionEnvelope)]

    async def test_flag_off_keeps_single_final_per_utterance(self):
        envs = await self._run("on", False)
        self.assertEqual(
            [(e.is_final, e.text) for e in envs],
            [(False, self.SENT), (False, self.SENT + " and more"), (True, self.SENT + " and more words")],
        )

    async def test_mode_off_ignores_flag(self):
        envs = await self._run("off", True)
        self.assertEqual([e.is_final for e in envs], [False, False, True])
        self.assertEqual(envs[-1].text, self.SENT + " and more words")

    async def test_shadow_mode_does_not_change_output(self):
        envs = await self._run("shadow", True)
        self.assertEqual([e.is_final for e in envs], [False, False, True])
        self.assertEqual(envs[-1].text, self.SENT + " and more words")

    async def test_on_emits_final_per_committed_sentence_and_trims_remainder(self):
        envs = await self._run("on", True)
        self.assertEqual(
            [(e.is_final, e.text) for e in envs],
            [
                (False, self.SENT),
                (True, self.SENT),
                (False, "and more"),
                (True, "and more words"),
            ],
        )
        self.assertEqual(envs[2].ts_start, envs[1].ts_end)
        self.assertEqual(envs[3].ts_start, envs[1].ts_end)
        self.assertEqual([e.seq for e in envs], [1, 2, 3, 4])

    async def test_on_skips_final_when_nothing_is_left_after_commit(self):
        envs = await self._run("on", True, partials=[self.SENT, self.SENT], final=self.SENT)
        self.assertEqual([(e.is_final, e.text) for e in envs], [(False, self.SENT), (True, self.SENT)])

    async def test_on_without_commit_emits_whole_text_as_final(self):
        envs = await self._run("on", True, partials=["short text"], final="short text.")
        self.assertEqual([(e.is_final, e.text) for e in envs], [(False, "short text"), (True, "short text.")])

    REVISED_FIRST = "I'm going to the store today."
    REVISED_PARTIALS = [REVISED_FIRST, REVISED_FIRST, "I am going to the store today. And then"]
    REVISED_FINAL = "I am going to the store today. And then home."

    async def test_on_remainder_follows_asr_revision_of_committed_sentence(self):
        envs = await self._run("on", True, partials=self.REVISED_PARTIALS, final=self.REVISED_FINAL)
        # A token count would send "today. And then" and "today. And then home." (repeated word).
        self.assertEqual(
            [(e.is_final, e.text) for e in envs],
            [
                (False, self.REVISED_FIRST),
                (True, self.REVISED_FIRST),
                (False, "And then"),
                (True, "And then home."),
            ],
        )

    async def test_on_revision_is_counted_and_traced_once_per_partial_utterance_and_at_final(self):
        from nemotron_testbed.metrics import SEGMENT_ALIGN

        partial = SEGMENT_ALIGN.labels(how="aligned", phase="partial")
        final = SEGMENT_ALIGN.labels(how="aligned", phase="final")
        before = (partial._value.get(), final._value.get())
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await self._run(
                "on", True,
                partials=self.REVISED_PARTIALS + ["I am going to the store today. And then ho"],
                final=self.REVISED_FINAL,
            )
        self.assertEqual((partial._value.get(), final._value.get()), (before[0] + 1, before[1] + 1))
        aligns = [r for r in cm.records if getattr(r, "stage", None) == "SEGMENT_ALIGN"]
        self.assertEqual([(r.how, r.phase) for r in aligns], [("aligned", "partial"), ("aligned", "final")])
        self.assertEqual(aligns[0].committed_text, self.REVISED_FIRST)
        self.assertEqual(aligns[0].remainder, "And then")
        self.assertEqual(aligns[1].text, self.REVISED_FINAL)
        self.assertEqual(aligns[1].remainder, "And then home.")

    async def test_on_unrevised_utterance_has_no_alignment_trace(self):
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await self._run("on", True)
        self.assertEqual([r for r in cm.records if getattr(r, "stage", None) == "SEGMENT_ALIGN"], [])

    async def test_alignment_trace_hides_text_when_trace_text_is_off(self):
        coordinator_settings = dataclasses.replace(
            make_settings(False, segmentation_mode="on"),
            segmentation_speech_to_text=True,
            pipeline_trace_text=False,
        )
        coordinator = EngineCoordinator(coordinator_settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        rpc = coordinator.open_rpc()
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await coordinator.accept_chunk(rpc.rpc_id, request(0, language="en"))
            key = asr.commands[-1].key
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
            )
            for text in self.REVISED_PARTIALS:
                await coordinator._handle_runtime_result(
                    RuntimeResult(
                        type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                        metadata={"model_step_executed": True},
                    )
                )
        align = [r for r in cm.records if getattr(r, "stage", None) == "SEGMENT_ALIGN"][0]
        self.assertFalse(hasattr(align, "committed_text"))
        self.assertFalse(hasattr(align, "text"))
        self.assertFalse(hasattr(align, "remainder"))
        self.assertEqual(align.committed_text_len, len(self.REVISED_FIRST))
        self.assertEqual(align.remainder_len, len("And then"))

    async def test_stt_uses_its_own_punct_stability(self):
        settings = dataclasses.replace(
            make_settings(False, segmentation_mode="on"),
            segmentation_speech_to_text=True,
            segmentation_punct_stable_n=1,
            segmentation_stt_punct_stable_n=3,
        )
        coordinator = EngineCoordinator(settings)
        self.assertEqual(coordinator._segmenter_params.sentence_stable_n, 1)
        self.assertEqual(coordinator._stt_segmenter_params.sentence_stable_n, 3)


class PipelineOrderStateTest(unittest.TestCase):
    def _state(self, segmented):
        from nemotron_testbed.coordinator import PipelineOrderState

        return PipelineOrderState(next_emit_slot=(1, 0), next_tts_slot=(1, 0), segmented=segmented)

    def test_unsegmented_steps_straight_to_next_utterance(self):
        state = self._state(False)
        state.advance_tts()
        state.advance_emit()
        self.assertEqual(state.next_tts_slot, (2, 0))
        self.assertEqual(state.next_emit_slot, (2, 0))

    def test_segmented_cursor_waits_inside_open_utterance(self):
        state = self._state(True)
        state.advance_tts()
        self.assertEqual(state.next_tts_slot, (1, 1))
        state.advance_tts()
        self.assertEqual(state.next_tts_slot, (1, 2))

    def test_set_total_settles_cursors_past_end(self):
        state = self._state(True)
        state.advance_tts()
        state.advance_tts()
        state.advance_emit()
        state.open_utts.add(1)
        state.set_total(1, 2)
        self.assertEqual(state.next_tts_slot, (2, 0))
        self.assertEqual(state.next_emit_slot, (1, 1))
        self.assertNotIn(1, state.open_utts)
        self.assertIn(1, state.utt_total)

    def test_set_total_zero_skips_empty_utterance(self):
        state = self._state(True)
        state.set_total(1, 0)
        self.assertEqual(state.next_tts_slot, (2, 0))
        self.assertEqual(state.next_emit_slot, (2, 0))

    def test_total_pruned_once_both_cursors_pass(self):
        state = self._state(True)
        state.set_total(1, 1)
        state.advance_tts()
        self.assertIn(1, state.utt_total)
        state.advance_emit()
        self.assertEqual(state.next_emit_slot, (2, 0))
        self.assertEqual(state.next_tts_slot, (2, 0))
        self.assertNotIn(1, state.utt_total)


class SegmentedPlaybackTest(unittest.IsolatedAsyncioTestCase):
    SENT = "This is a complete sentence here."
    REST = "and more words"
    FULL = "This is a complete sentence here. and more words"

    async def _open(
        self,
        *,
        mode="on",
        task="translate",
        language="en",
        max_utterance_ms=20_000,
        max_pending=None,
        tr_runtime=None,
        tts_runtime=None,
    ):
        self.language = language
        settings = make_settings(True, max_utterance_ms=max_utterance_ms, segmentation_mode=mode)
        if max_pending is not None:
            settings = dataclasses.replace(settings, max_postprocess_pending_per_rpc=max_pending)
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        tr = tr_runtime or FakeTranslationRuntime()
        tts = tts_runtime or FakeTtsRuntime()
        coordinator.runtime, coordinator.translation_runtime = asr, tr
        coordinator.tts_runtime = tts
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task=task, language=language))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        return coordinator, rpc, key, tr, tts

    async def _partials(self, coordinator, key, texts, uid=1):
        for text in texts:
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL, key=key, utterance_id=uid, text=text,
                    metadata={"model_step_executed": True},
                )
            )

    async def _final(self, coordinator, rpc, key, text, uid=1, ts=160):
        await coordinator.accept_chunk(rpc.rpc_id, request(ts, end=True, task="translate", language=self.language))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=uid, text=text)
        )

    @staticmethod
    def _tr(key, uid, seg, src, dst, t0=0, t1=320):
        return TranslationResult(
            key=key, utterance_id=uid, source_text=src, translated_text=dst,
            source_language="en-US", target_language="vi-VN", ts_start=t0, ts_end=t1,
            segment_idx=seg,
        )

    @staticmethod
    def _tts(key, uid, seg, text, t0=0, t1=320, **extra):
        fields = dict(
            key=key, utterance_id=uid, text=text, target_language="vi-VN",
            ts_start=t0, ts_end=t1, audio_payload=b"\xff" * 80, audio_pt=0,
            segment_idx=seg,
        )
        fields.update(extra)
        return TtsResult(**fields)

    @staticmethod
    async def _drain(rpc, count):
        out = []
        for _ in range(count):
            out.append(await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2))
        return out

    async def test_segment_is_translated_before_final_and_remainder_is_last(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, self.SENT)])

        await self._final(coordinator, rpc, key, self.FULL)
        self.assertEqual(
            [(j.segment_idx, j.source_text) for j in tr.jobs],
            [(0, self.SENT), (1, self.REST)],
        )

    async def test_remainder_sent_to_translation_follows_asr_revision(self):
        first = "We will meet at nine thirty."
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [first, first, "We'll meet at nine thirty. See"])
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, first)])

        await self._final(coordinator, rpc, key, "We'll meet at nine thirty. See you there.")
        # A token count would translate "you there." and lose "See".
        self.assertEqual(
            [(j.segment_idx, j.source_text) for j in tr.jobs],
            [(0, first), (1, "See you there.")],
        )

    async def test_asr_final_trace_shows_committed_tokens_and_remainder(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await self._final(coordinator, rpc, key, self.FULL)
        final = [r for r in cm.records if getattr(r, "stage", None) == "ASR_FINAL"][0]
        self.assertEqual(final.text, self.FULL)
        self.assertEqual(final.committed_tokens, len(self.SENT.split()))
        self.assertEqual(final.remainder, self.REST)

    async def test_asr_final_trace_has_no_segment_fields_when_off(self):
        coordinator, rpc, key, tr, tts = await self._open(mode="off")
        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await self._final(coordinator, rpc, key, self.FULL)
        final = [r for r in cm.records if getattr(r, "stage", None) == "ASR_FINAL"][0]
        self.assertFalse(hasattr(final, "committed_tokens"))
        self.assertFalse(hasattr(final, "remainder"))

    async def test_out_of_order_results_are_emitted_in_segment_order(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)

        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        self.assertEqual(tts.jobs, [])
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        self.assertEqual([(j.segment_idx, j.text) for j in tts.jobs], [(0, "day la")])

        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        self.assertEqual([(j.segment_idx, j.text) for j in tts.jobs], [(0, "day la"), (1, "va them")])
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))

        envs = await self._drain(rpc, 2)
        self.assertEqual([e.text for e in envs], ["day la", "va them"])
        self.assertTrue(all(e.is_final for e in envs))
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_tts_result_for_later_segment_waits_for_earlier_emit(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))

        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        envs = await self._drain(rpc, 2)
        self.assertEqual([e.text for e in envs], ["day la", "va them"])

    async def test_empty_remainder_closes_without_extra_message(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.SENT)
        self.assertEqual([j.segment_idx for j in tr.jobs], [0])

        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        envs = await self._drain(rpc, 1)
        self.assertEqual([e.text for e in envs], ["day la"])
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_punctuation_only_tail_is_not_a_segment(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.SENT + " .")
        self.assertEqual([j.segment_idx for j in tr.jobs], [0])

        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        envs = await self._drain(rpc, 1)
        self.assertEqual([e.text for e in envs], ["day la"])
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_punctuation_only_final_without_commit_is_empty(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._final(coordinator, rpc, key, ".")
        self.assertEqual(tr.jobs, [])
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_short_word_tail_is_still_a_segment(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " ok"])
        await self._final(coordinator, rpc, key, self.SENT + " ok")
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, self.SENT), (1, "ok")])

    async def test_no_commit_falls_back_to_single_segment(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, ["hello there"])
        await self._final(coordinator, rpc, key, "hello there friend")
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, "hello there friend")])

        await coordinator._handle_translation_result(self._tr(key, 1, 0, "hello there friend", "chao"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "chao"))
        envs = await self._drain(rpc, 1)
        self.assertEqual([e.text for e in envs], ["chao"])
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_auto_language_does_not_segment(self):
        coordinator, rpc, key, tr, tts = await self._open()
        for text in [self.SENT, self.SENT, self.SENT + " and more"]:
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                    detected_language="", metadata={"model_step_executed": True},
                )
            )
        self.assertEqual(tr.jobs, [])
        await self._final(coordinator, rpc, key, self.FULL)
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, self.FULL)])

    async def test_auto_language_segments_once_detected_on_partial(self):
        coordinator, rpc, key, tr, tts = await self._open()
        for text in [self.SENT, self.SENT, self.SENT + " and more"]:
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                    detected_language="vi-VN", metadata={"model_step_executed": True},
                )
            )
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, self.SENT)])

    async def test_segmenter_error_keeps_sent_segments_and_sends_rest_at_final(self):
        from unittest.mock import patch

        from nemotron_testbed.metrics import SEGMENT_ERRORS

        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        before = SEGMENT_ERRORS._value.get()
        with patch(
            "nemotron_testbed.coordinator.StableSentenceSegmenter.update",
            side_effect=RuntimeError("boom"),
        ):
            await self._partials(coordinator, key, [self.SENT + " and more words"])
        self.assertEqual(SEGMENT_ERRORS._value.get(), before + 1)

        await self._final(coordinator, rpc, key, self.FULL)
        self.assertEqual(
            [(j.segment_idx, j.source_text) for j in tr.jobs],
            [(0, self.SENT), (1, self.REST)],
        )
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        envs = await self._drain(rpc, 2)
        self.assertEqual([e.text for e in envs], ["day la", "va them"])
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_empty_earlier_utterance_does_not_hold_later_segments(self):
        coordinator, rpc, key, tr, tts = await self._open(max_utterance_ms=40)
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(480, end=True, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Tạm biệt")
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="")
        )
        self.assertEqual([(j.utterance_id, j.segment_idx) for j in tr.jobs], [(2, 0)])
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=2, source_text="Tạm biệt", translated_text="Goodbye",
                source_language="vi-VN", target_language="en-US", ts_start=640, ts_end=960,
            )
        )
        self.assertEqual([j.text for j in tts.jobs], ["Goodbye"])

    async def test_off_mode_never_segments(self):
        coordinator, rpc, key, tr, tts = await self._open(mode="off")
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        self.assertEqual(tr.jobs, [])
        await self._final(coordinator, rpc, key, self.FULL)
        self.assertEqual([(j.segment_idx, j.source_text) for j in tr.jobs], [(0, self.FULL)])
        self.assertFalse(coordinator._pipeline_order[key].segmented)

    async def test_cancel_mid_utterance_leaves_no_state(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        self.assertIn(key, coordinator._pipeline_order)
        await coordinator.cancel_rpc(rpc.rpc_id, "test")
        self.assertNotIn(key, coordinator._pipeline_order)
        self.assertEqual(coordinator._pipeline_jobs, {})

    async def test_metrics_per_segment(self):
        from nemotron_testbed.metrics import FIRST_AUDIO_MS, SEGMENT_EMITTED

        before_seg = SEGMENT_EMITTED._value.get()
        before_first = sum(b.get() for b in FIRST_AUDIO_MS._buckets)
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        await self._drain(rpc, 2)
        self.assertEqual(SEGMENT_EMITTED._value.get(), before_seg + 2)
        self.assertEqual(sum(b.get() for b in FIRST_AUDIO_MS._buckets), before_first + 1)

    async def test_first_audio_from_speech_adds_the_lead_to_first_audio(self):
        from nemotron_testbed.metrics import FIRST_AUDIO_FROM_SPEECH_MS, FIRST_AUDIO_MS

        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        coordinator._pipeline_jobs[(key, 1, 0)].speech_lead_ms = 500.0
        before_speech = FIRST_AUDIO_FROM_SPEECH_MS._sum.get()
        before_first = FIRST_AUDIO_MS._sum.get()
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        self.assertAlmostEqual(
            (FIRST_AUDIO_FROM_SPEECH_MS._sum.get() - before_speech)
            - (FIRST_AUDIO_MS._sum.get() - before_first),
            500.0,
            delta=50.0,
        )

    async def test_post_speech_latency_measured_from_vad_end_once_per_utterance(self):
        from nemotron_testbed.metrics import POST_SPEECH_LATENCY_MS

        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        vad_end = time.monotonic() - 0.4
        coordinator._pipeline_jobs[(key, 1, 0)].vad_end_wall = vad_end
        coordinator._pipeline_jobs[(key, 1, 1)].vad_end_wall = vad_end
        count_before = sum(b.get() for b in POST_SPEECH_LATENCY_MS._buckets)
        sum_before = POST_SPEECH_LATENCY_MS._sum.get()
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        await self._drain(rpc, 2)
        self.assertEqual(
            sum(b.get() for b in POST_SPEECH_LATENCY_MS._buckets), count_before + 1
        )
        self.assertAlmostEqual(
            POST_SPEECH_LATENCY_MS._sum.get() - sum_before, 400.0, delta=150.0
        )

    async def test_post_speech_latency_skipped_when_vad_end_not_fired(self):
        from nemotron_testbed.metrics import POST_SPEECH_LATENCY_MS

        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        self.assertEqual(coordinator._pipeline_jobs[(key, 1, 0)].vad_end_wall, 0.0)
        count_before = sum(b.get() for b in POST_SPEECH_LATENCY_MS._buckets)
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        self.assertEqual(
            sum(b.get() for b in POST_SPEECH_LATENCY_MS._buckets), count_before
        )

    async def test_envelope_carries_speech_end_wall_for_audio_only(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        speech_end = time.monotonic() - 0.5
        for seg in (0, 1):
            coordinator._pipeline_jobs[(key, 1, seg)].speech_end_wall = speech_end
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them", audio_payload=b""))
        envs = await self._drain(rpc, 2)
        self.assertTrue(envs[0].audio_payload)
        self.assertEqual(envs[0].speech_end_wall, speech_end)
        self.assertFalse(envs[1].audio_payload)
        self.assertEqual(envs[1].speech_end_wall, 0.0)

    NEXT = "Another full sentence goes here."

    async def _two_utterances(self, **kwargs):
        """Utterance 1 has committed segment 0 but no FINAL; utterance 2 has committed its own segment 0."""
        coordinator, rpc, key, tr, tts = await self._open(language="en", max_utterance_ms=40, **kwargs)
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await self._partials(coordinator, key, [self.NEXT, self.NEXT, self.NEXT + " and"], uid=2)
        return coordinator, rpc, key, tr, tts

    async def test_next_utterance_is_held_until_open_utterance_closes(self):
        coordinator, rpc, key, tr, tts = await self._two_utterances()
        self.assertEqual([(j.utterance_id, j.segment_idx) for j in tr.jobs], [(1, 0), (2, 0)])

        with self.assertLogs("nemotron_testbed.coordinator", level="DEBUG") as cm:
            await coordinator._handle_translation_result(self._tr(key, 2, 0, self.NEXT, "tiep", 640, 960))
        holds = [r for r in cm.records if getattr(r, "stage", None) == "ORDER_HOLD"]
        self.assertEqual([(r.waiting_for, r.waiting_segment) for r in holds], [(1, 0)])
        self.assertEqual(tts.jobs, [])

        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        # Utterance 1 is still open (no FINAL yet), so utterance 2 must keep waiting.
        self.assertEqual([(j.utterance_id, j.segment_idx) for j in tts.jobs], [(1, 0)])
        self.assertEqual(coordinator._pipeline_order[key].next_tts_slot, (1, 1))

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=self.FULL)
        )
        self.assertEqual([(j.utterance_id, j.segment_idx) for j in tr.jobs][-1], (1, 1))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        self.assertEqual([(j.utterance_id, j.segment_idx) for j in tts.jobs], [(1, 0), (1, 1)])

        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        self.assertEqual([(j.utterance_id, j.segment_idx) for j in tts.jobs][-1], (2, 0))
        await coordinator._handle_tts_result(self._tts(key, 2, 0, "tiep"))

        envs = await self._drain(rpc, 3)
        self.assertEqual([e.text for e in envs], ["day la", "va them", "tiep"])
        self.assertEqual([e.utterance_id for e in envs], [1, 1, 2])

    async def test_end_of_stream_does_not_strand_earlier_utterance_still_finishing(self):
        """End of stream while utterance 2 is active closes the stream on ITS final.

        Utterance 1 (already finalizing, with a committed segment) can still reach FINAL later
        on another ASR worker; its remainder must not be dropped and utterance 2 must not be
        left waiting for it.
        """
        coordinator, rpc, key, tr, tts = await self._open(max_utterance_ms=60)
        asr = coordinator.runtime
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task="translate", language="en"))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task="translate", language="en"))
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 1))
        await coordinator.accept_chunk(rpc.rpc_id, request(480, task="translate", language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(640, end=True, task="translate", language="en"))
        self.assertEqual((asr.commands[-1].type.name, asr.commands[-1].utterance_id), ("FINALIZE", 2))

        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=2, text=self.NEXT)
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=self.FULL)
        )
        self.assertEqual(
            [(j.utterance_id, j.segment_idx) for j in tr.jobs], [(1, 0), (2, 0), (1, 1)]
        )

        await coordinator._handle_translation_result(self._tr(key, 2, 0, self.NEXT, "tiep", 640, 960))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la"))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 2, 0, "tiep", 640, 960))

        envs = await self._drain(rpc, 3)
        self.assertEqual([e.text for e in envs], ["day la", "va them", "tiep"])
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_pending_limit_is_counted_per_segment(self):
        from nemotron_testbed.coordinator import RpcFailure

        coordinator, rpc, key, tr, tts = await self._open(max_pending=1)
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        self.assertEqual(rpc.pending_postprocess, 1)
        await self._final(coordinator, rpc, key, self.FULL)

        self.assertEqual([j.segment_idx for j in tr.jobs], [0])
        self.assertTrue(rpc.cancelled)
        failure = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertIsInstance(failure, RpcFailure)
        self.assertEqual(failure.code_name, "RESOURCE_EXHAUSTED")
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_pending_limit_allows_one_slot_per_segment(self):
        coordinator, rpc, key, tr, tts = await self._open(max_pending=2)
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        self.assertFalse(rpc.cancelled)
        self.assertEqual([j.segment_idx for j in tr.jobs], [0, 1])
        self.assertEqual(rpc.pending_postprocess, 2)

    async def test_translate_error_on_segment_emits_source_text_in_order(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)

        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1, source_text=self.SENT, error="boom", segment_idx=0,
                source_language="en-US", target_language="vi-VN",
            )
        )
        self.assertEqual([(j.segment_idx, j.text) for j in tts.jobs], [(1, "va them")])
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))

        envs = await self._drain(rpc, 2)
        self.assertEqual([e.text for e in envs], [self.SENT, "va them"])
        self.assertEqual(envs[0].audio_payload, b"")
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_tts_error_on_segment_emits_translated_text_in_order(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))

        await coordinator._handle_tts_result(self._tts(key, 1, 0, "day la", audio_payload=b"", error="boom"))
        self.assertEqual([(j.segment_idx, j.text) for j in tts.jobs], [(0, "day la"), (1, "va them")])
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))

        envs = await self._drain(rpc, 2)
        self.assertEqual([e.text for e in envs], ["day la", "va them"])
        self.assertEqual(envs[0].audio_payload, b"")
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_translation_queue_full_on_segment_emits_source_text(self):
        from nemotron_testbed.domain import SubmitStatus

        class RejectFirstTranslation(FakeTranslationRuntime):
            def try_submit_nowait(self, job):
                if not self.jobs:
                    self.jobs.append(job)
                    return SubmitStatus.OVERLOADED
                return super().try_submit_nowait(job)

        coordinator, rpc, key, tr, tts = await self._open(tr_runtime=RejectFirstTranslation())
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        first = await self._drain(rpc, 1)
        self.assertEqual([e.text for e in first], [self.SENT])

        await self._final(coordinator, rpc, key, self.FULL)
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))
        second = await self._drain(rpc, 1)
        self.assertEqual([e.text for e in second], ["va them"])
        self.assertEqual(rpc.pending_postprocess, 0)

    async def test_tts_queue_full_degrades_every_segment_in_order(self):
        from nemotron_testbed.domain import SubmitStatus

        class FullTts(FakeTtsRuntime):
            def try_submit_nowait(self, job):
                return SubmitStatus.OVERLOADED

        coordinator, rpc, key, tr, tts = await self._open(tts_runtime=FullTts())
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)

        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, "day la"))

        envs = await self._drain(rpc, 2)
        self.assertEqual([e.text for e in envs], ["day la", "va them"])
        self.assertTrue(all(e.audio_payload == b"" for e in envs))
        self.assertEqual(rpc.pending_postprocess, 0)
        self.assertNotIn(key, coordinator._pipeline_order)

    async def test_multi_phrase_segment_completes_before_next_segment(self):
        coordinator, rpc, key, tr, tts = await self._open()
        await self._partials(coordinator, key, [self.SENT, self.SENT, self.SENT + " and more"])
        await self._final(coordinator, rpc, key, self.FULL)
        two = "Xin chao! Tam biet"
        await coordinator._handle_translation_result(self._tr(key, 1, 1, self.REST, "va them"))
        await coordinator._handle_translation_result(self._tr(key, 1, 0, self.SENT, two))
        self.assertEqual([(j.segment_idx, j.phrase_id) for j in tts.jobs], [(0, 0)])

        await coordinator._handle_tts_result(
            self._tts(key, 1, 0, "Xin chao!", phrase_id=0, is_last_phrase=False, translated_text=two)
        )
        self.assertEqual([(j.segment_idx, j.phrase_id) for j in tts.jobs], [(0, 0), (0, 1)])
        await coordinator._handle_tts_result(
            self._tts(key, 1, 0, "Tam biet", phrase_id=1, is_last_phrase=True, translated_text=two)
        )
        self.assertEqual([(j.segment_idx, j.phrase_id) for j in tts.jobs][-1], (1, 0))
        await coordinator._handle_tts_result(self._tts(key, 1, 1, "va them"))

        envs = await self._drain(rpc, 3)
        self.assertEqual([(e.text, e.is_final) for e in envs], [("", False), (two, True), ("va them", True)])
        self.assertEqual(rpc.pending_postprocess, 0)


class TargetLanguageTest(unittest.IsolatedAsyncioTestCase):
    LOGGER = "nemotron_testbed.coordinator"

    async def _open(self, language="en", task="realtime_translation"):
        coordinator = EngineCoordinator(make_settings(True))
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task=task, language=language))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        return coordinator, rpc, key

    async def _final(self, coordinator, rpc, key, text, detected, language="en", uid=1, ts=160):
        await coordinator.accept_chunk(
            rpc.rpc_id, request(ts, end=True, task="realtime_translation", language=language)
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL, key=key, utterance_id=uid, text=text,
                detected_language=detected,
            )
        )

    async def test_language_is_the_target_and_source_is_auto(self):
        coordinator, rpc, key = await self._open("EN")
        stream = coordinator._streams[key]
        self.assertEqual(stream.source_language, "auto")
        self.assertEqual(stream.target_language, "en-US")

    async def test_transcribe_language_is_still_the_source(self):
        coordinator, rpc, key = await self._open("vi", task="transcribe")
        stream = coordinator._streams[key]
        self.assertEqual(stream.source_language, "vi-VN")
        self.assertEqual(stream.target_language, "")

    async def test_unsupported_target_is_rejected_at_open(self):
        coordinator = EngineCoordinator(make_settings(True))
        coordinator.runtime = FakeRuntime()
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        with self.assertRaises(ValueError):
            await coordinator.accept_chunk(
                rpc.rpc_id, request(0, task="realtime_translation", language="ja-JP")
            )

    async def test_changing_the_target_inside_a_stream_is_rejected(self):
        coordinator, rpc, key = await self._open("en")
        with self.assertRaises(ValueError):
            await coordinator.accept_chunk(
                rpc.rpc_id, request(160, task="realtime_translation", language="vi")
            )

    async def test_detected_source_is_translated_toward_the_target(self):
        coordinator, rpc, key = await self._open("en")
        await self._final(coordinator, rpc, key, "Xin chào", "vi-VN")
        jobs = coordinator.translation_runtime.jobs
        self.assertEqual([(j.source_language, j.target_language) for j in jobs], [("vi-VN", "en-US")])

    async def test_source_equal_to_target_returns_text_and_original_audio(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator, rpc, key = await self._open("en")
            await self._final(coordinator, rpc, key, "Hello there", "en-US")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        self.assertEqual(coordinator.tts_runtime.jobs, [])
        env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertEqual((env.text, env.is_final, env.language), ("Hello there", True, "en-US"))
        self.assertGreater(len(env.audio_payload), 0)
        self.assertEqual(env.audio_pt, coordinator.settings.tts_output_pt)
        passthrough = [r for r in cm.records if getattr(r, "stage", "") == "PASSTHROUGH"]
        self.assertEqual([r.reason for r in passthrough], ["source_equals_target"])
        self.assertEqual(passthrough[0].audio_bytes, len(env.audio_payload))

    async def test_passthrough_audio_is_the_utterance_audio_as_g711(self):
        coordinator, rpc, key = await self._open("en")
        await self._final(coordinator, rpc, key, "Hello there", "en-US")
        env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        samples = 2 * 320  # two 20 ms chunks at 16 kHz, resampled to 8 kHz G.711 = 1 byte/sample
        self.assertEqual(len(env.audio_payload), samples // 2)

    async def test_empty_target_returns_text_and_original_audio(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator, rpc, key = await self._open("")
            await self._final(coordinator, rpc, key, "Xin chào", "vi-VN", language="")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertEqual(env.text, "Xin chào")
        self.assertGreater(len(env.audio_payload), 0)
        reasons = [r.reason for r in cm.records if getattr(r, "stage", "") == "PASSTHROUGH"]
        self.assertEqual(reasons, ["no_target_language"])

    async def test_undetected_language_without_history_is_passed_through(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator, rpc, key = await self._open("en")
            await self._final(coordinator, rpc, key, "konnichiwa", "ja-JP")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertGreater(len(env.audio_payload), 0)
        reasons = [r.reason for r in cm.records if getattr(r, "stage", "") == "PASSTHROUGH"]
        self.assertEqual(reasons, ["language_undetected"])

    async def test_empty_asr_text_emits_nothing(self):
        coordinator, rpc, key = await self._open("en")
        await self._final(coordinator, rpc, key, "", "en-US")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        self.assertTrue(len(rpc.response_buffer._items) == 0)

    async def test_passthrough_waits_for_an_earlier_utterance_in_translation(self):
        coordinator = EngineCoordinator(make_settings(True, max_utterance_ms=40))
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        task = "realtime_translation"
        await coordinator.accept_chunk(rpc.rpc_id, request(0, task=task, language="en"))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(160, task=task, language="en"))
        await coordinator.accept_chunk(rpc.rpc_id, request(320, task=task, language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=2)
        )
        await coordinator.accept_chunk(rpc.rpc_id, request(480, end=True, task=task, language="en"))
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL, key=key, utterance_id=1, text="Xin chào",
                detected_language="vi-VN",
            )
        )
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL, key=key, utterance_id=2, text="Hello again",
                detected_language="en-US",
            )
        )
        self.assertEqual(len(coordinator.translation_runtime.jobs), 1)
        self.assertEqual(len(rpc.response_buffer._items), 0)
        await coordinator._handle_translation_result(
            TranslationResult(
                key=key, utterance_id=1, source_text="Xin chào", translated_text="Hello",
                source_language="vi-VN", target_language="en-US", ts_start=0, ts_end=320,
            )
        )
        await coordinator._handle_tts_result(
            TtsResult(
                key=key, utterance_id=1, text="Hello", target_language="en-US",
                ts_start=0, ts_end=320, audio_payload=bytes(80), audio_pt=0,
            )
        )
        first = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        second = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertEqual([first.text, second.text], ["Hello", "Hello again"])
        self.assertEqual(rpc.pending_postprocess, 0)


class LanguageSourceTest(unittest.IsolatedAsyncioTestCase):
    LOGGER = "nemotron_testbed.coordinator"
    SENT = "This is a complete sentence here."
    TRANSLATE = "realtime_translation"

    async def _open(
        self,
        *,
        task=TRANSLATE,
        language="en",
        language_source="",
        switch=True,
        segmentation_mode="off",
    ):
        self.task, self.language, self.language_source = task, language, language_source
        coordinator = EngineCoordinator(
            make_settings(
                True, segmentation_mode=segmentation_mode, use_language_source=switch
            )
        )
        asr = FakeRuntime()
        coordinator.runtime = asr
        coordinator.translation_runtime = FakeTranslationRuntime()
        coordinator.tts_runtime = FakeTtsRuntime()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, self._request(0))
        key = asr.commands[-1].key
        await coordinator._handle_runtime_result(
            RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1)
        )
        return coordinator, rpc, key, asr

    def _request(self, ts, end=False):
        return request(
            ts,
            end=end,
            task=self.task,
            language=self.language,
            language_source=self.language_source,
        )

    async def _final(self, coordinator, rpc, key, text, detected="", ts=160):
        await coordinator.accept_chunk(rpc.rpc_id, self._request(ts, end=True))
        await coordinator._handle_runtime_result(
            RuntimeResult(
                type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=text,
                detected_language=detected,
            )
        )

    async def test_translate_uses_language_source_as_fixed_source(self):
        coordinator, _, key, asr = await self._open(language="en", language_source="vi")
        stream = coordinator._streams[key]
        self.assertEqual((stream.source_language, stream.target_language), ("vi-VN", "en-US"))
        self.assertEqual(asr.commands[0].language, "vi-VN")

    async def test_translate_accepts_region_tags_and_canonicalizes(self):
        coordinator, _, key, asr = await self._open(language="vi", language_source="en-GB")
        self.assertEqual(coordinator._streams[key].source_language, "en-US")
        self.assertEqual(asr.commands[0].language, "en-US")

    async def test_translate_without_language_source_stays_auto(self):
        coordinator, _, key, asr = await self._open(language="en", language_source="")
        self.assertEqual(coordinator._streams[key].source_language, "auto")
        self.assertEqual(asr.commands[0].language, "auto")

    async def test_translate_ignores_unsupported_language_source_with_warning(self):
        with self.assertLogs(self.LOGGER, level="WARNING") as cm:
            coordinator, _, key, asr = await self._open(language="en", language_source="ja-JP")
        self.assertEqual(coordinator._streams[key].source_language, "auto")
        self.assertEqual(asr.commands[0].language, "auto")
        warned = [r for r in cm.records if "language_source" in r.getMessage()]
        self.assertEqual(len(warned), 1)
        self.assertEqual(warned[0].language_source, "ja-JP")

    async def test_switch_off_ignores_language_source_for_translate(self):
        coordinator, _, key, asr = await self._open(
            language="en", language_source="vi", switch=False
        )
        self.assertEqual(coordinator._streams[key].source_language, "auto")
        self.assertEqual(asr.commands[0].language, "auto")

    async def test_stream_open_log_reports_the_source_mode(self):
        cases = [
            ({"language_source": "vi"}, "explicit"),
            ({"language_source": ""}, "auto"),
            ({"language_source": "ja-JP"}, "ignored"),
            ({"language_source": "vi", "switch": False}, "disabled"),
        ]
        for kwargs, expected in cases:
            with self.subTest(expected=expected):
                with self.assertLogs(self.LOGGER, level="INFO") as cm:
                    await self._open(language="en", **kwargs)
                opened = [r for r in cm.records if r.getMessage() == "Logical stream opened"]
                self.assertEqual([r.source_mode for r in opened], [expected])

    async def test_fixed_source_translates_without_waiting_for_detection(self):
        coordinator, rpc, key, _ = await self._open(language="en", language_source="vi")
        await self._final(coordinator, rpc, key, "Xin chào", detected="")
        jobs = coordinator.translation_runtime.jobs
        self.assertEqual([(j.source_language, j.target_language) for j in jobs], [("vi-VN", "en-US")])

    async def test_fixed_source_wins_over_a_conflicting_detection(self):
        coordinator, rpc, key, _ = await self._open(language="en", language_source="vi")
        await self._final(coordinator, rpc, key, "Hello", detected="en-US")
        jobs = coordinator.translation_runtime.jobs
        self.assertEqual([(j.source_language, j.target_language) for j in jobs], [("vi-VN", "en-US")])

    async def test_fixed_source_equal_to_target_is_passed_through(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator, rpc, key, _ = await self._open(language="en", language_source="en")
            await self._final(coordinator, rpc, key, "Hello there")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        self.assertEqual(coordinator.tts_runtime.jobs, [])
        env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertEqual((env.text, env.language), ("Hello there", "en-US"))
        self.assertGreater(len(env.audio_payload), 0)
        reasons = [r.reason for r in cm.records if getattr(r, "stage", "") == "PASSTHROUGH"]
        self.assertEqual(reasons, ["source_equals_target"])

    async def test_fixed_source_without_target_is_passed_through(self):
        with self.assertLogs(self.LOGGER, level="DEBUG") as cm:
            coordinator, rpc, key, _ = await self._open(language="", language_source="vi")
            await self._final(coordinator, rpc, key, "Xin chào")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        reasons = [r.reason for r in cm.records if getattr(r, "stage", "") == "PASSTHROUGH"]
        self.assertEqual(reasons, ["no_target_language"])

    async def test_fixed_source_segments_from_the_first_partial_without_detection(self):
        coordinator, _, key, _ = await self._open(
            language="vi", language_source="en", segmentation_mode="on"
        )
        for text in (self.SENT, self.SENT, self.SENT + " and more"):
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                    detected_language="", metadata={"model_step_executed": True},
                )
            )
        jobs = coordinator.translation_runtime.jobs
        self.assertEqual(
            [(j.segment_idx, j.source_language, j.target_language) for j in jobs],
            [(0, "en-US", "vi-VN")],
        )

    async def test_fixed_source_equal_to_target_never_translates_a_segment(self):
        coordinator, rpc, key, _ = await self._open(
            language="en", language_source="en", segmentation_mode="on"
        )
        for text in (self.SENT, self.SENT, self.SENT + " and more"):
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                    detected_language="", metadata={"model_step_executed": True},
                )
            )
        await self._final(coordinator, rpc, key, self.SENT + " and more words")
        self.assertEqual(coordinator.translation_runtime.jobs, [])
        env = await asyncio.wait_for(rpc.response_buffer.get(), timeout=0.2)
        self.assertEqual(env.text, self.SENT + " and more words")

    async def test_transcribe_language_source_overrides_language(self):
        coordinator, _, key, asr = await self._open(
            task="transcribe", language="vi", language_source="en"
        )
        self.assertEqual(coordinator._streams[key].source_language, "en-US")
        self.assertEqual(asr.commands[0].language, "en-US")

    async def test_transcribe_without_language_source_uses_language(self):
        coordinator, _, key, asr = await self._open(
            task="transcribe", language="vi", language_source=""
        )
        self.assertEqual(coordinator._streams[key].source_language, "vi-VN")
        self.assertEqual(asr.commands[0].language, "vi-VN")

    async def test_transcribe_language_source_alone_sets_the_source(self):
        coordinator, _, key, asr = await self._open(
            task="transcribe", language="", language_source="en"
        )
        self.assertEqual(coordinator._streams[key].source_language, "en-US")

    async def test_transcribe_with_neither_is_auto(self):
        coordinator, _, key, asr = await self._open(
            task="transcribe", language="", language_source=""
        )
        self.assertEqual(coordinator._streams[key].source_language, "auto")

    async def test_transcribe_switch_off_uses_language(self):
        coordinator, _, key, asr = await self._open(
            task="transcribe", language="vi", language_source="en", switch=False
        )
        self.assertEqual(coordinator._streams[key].source_language, "vi-VN")
        self.assertEqual(asr.commands[0].language, "vi-VN")

    async def test_transcribe_resending_both_fields_on_later_chunks_is_accepted(self):
        coordinator, rpc, key, _ = await self._open(
            task="transcribe", language="vi", language_source="en"
        )
        await coordinator.accept_chunk(rpc.rpc_id, self._request(160))
        self.assertEqual(coordinator._streams[key].source_language, "en-US")

    async def test_changing_language_source_inside_a_stream_is_rejected(self):
        coordinator, rpc, key, _ = await self._open(language="en", language_source="vi")
        self.language_source = "en"
        with self.assertRaises(ValueError):
            await coordinator.accept_chunk(rpc.rpc_id, self._request(160))

    async def test_changing_language_source_is_rejected_for_transcribe_too(self):
        coordinator, rpc, key, _ = await self._open(
            task="transcribe", language="vi", language_source="en"
        )
        self.language_source = "vi"
        with self.assertRaises(ValueError):
            await coordinator.accept_chunk(rpc.rpc_id, self._request(160))

    async def test_empty_language_source_on_a_later_chunk_is_accepted(self):
        coordinator, rpc, key, _ = await self._open(language="en", language_source="vi")
        self.language_source = ""
        await coordinator.accept_chunk(rpc.rpc_id, self._request(160))
        self.assertEqual(coordinator._streams[key].source_language, "vi-VN")

    async def test_language_source_is_not_checked_when_the_switch_is_off(self):
        coordinator, rpc, key, _ = await self._open(
            language="en", language_source="vi", switch=False
        )
        self.language_source = "en"
        await coordinator.accept_chunk(rpc.rpc_id, self._request(160))


if __name__ == "__main__":
    unittest.main()
