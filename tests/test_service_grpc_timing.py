import asyncio
import logging
import time
import types
import unittest

from nemotron_testbed.coordinator import (
    _RPC_END,
    AsyncResponseBuffer,
    RecognitionEnvelope,
)
from nemotron_testbed.metrics import (
    GRPC_BUFFER_WAIT_MS,
    GRPC_SEND_MS,
    SPEECH_END_TO_AUDIO_SENT_MS,
    SPEECH_END_TO_AUDIO_SENT_PREPROC_MS,
)
from nemotron_testbed.service import SpeechStreamService


class _Coordinator:
    def __init__(
        self,
        buffer: AsyncResponseBuffer,
        pipeline_trace: bool = True,
        pipeline_trace_level: str = "DEBUG",
    ) -> None:
        self.settings = types.SimpleNamespace(
            rpc_max_idle_s=0,
            pipeline_trace=pipeline_trace,
            pipeline_trace_level=pipeline_trace_level,
        )
        self.rpc = types.SimpleNamespace(
            rpc_id="rpc1", response_buffer=buffer, finished=True, cancelled=False
        )

    def open_rpc(self):
        return self.rpc

    async def accept_chunk(self, rpc_id, request):
        return None

    async def close_input(self, rpc_id):
        return None

    async def cancel_rpc(self, rpc_id, reason):
        return None


async def _no_requests():
    return
    yield


class _Context:
    def cancelled(self) -> bool:
        return False


class GrpcTimingLogTest(unittest.IsolatedAsyncioTestCase):
    async def test_grpc_out_reports_buffer_wait_and_sent_reports_send_ms(self):
        buffer = AsyncResponseBuffer(soft_limit=8)
        envelope = RecognitionEnvelope(
            session_id="s",
            stream_id="M0",
            text="xin chao",
            is_final=True,
            ts_start=0,
            ts_end=1,
            confidence=0.5,
            language="vi-VN",
            seq=7,
            utterance_id=1,
            audio_payload=b"\x00" * 16,
        )
        buffer.put(envelope)
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))

        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            results = [r async for r in service.Recognize(_no_requests(), _Context())]

        self.assertEqual(len(results), 1)
        out = [m for m in cm.output if "GRPC_OUT" in m]
        sent = [m for m in cm.output if "GRPC_SENT" in m]
        self.assertEqual(len(out), 1)
        self.assertIn("buffer_wait_ms=", out[0])
        self.assertEqual(len(sent), 1)
        self.assertIn("seq=7", sent[0])
        self.assertIn("audio_bytes=16", sent[0])
        self.assertIn("send_ms=", sent[0])

    async def test_metrics_observed_without_debug_logging(self):
        before_wait = sum(b.get() for b in GRPC_BUFFER_WAIT_MS._buckets)
        before_send = sum(b.get() for b in GRPC_SEND_MS._buckets)
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(
            RecognitionEnvelope(
                session_id="s",
                stream_id="M0",
                text="xin chao",
                is_final=True,
                ts_start=0,
                ts_end=1,
                confidence=0.5,
                language="vi-VN",
                seq=1,
                utterance_id=1,
                audio_payload=b"\x00" * 16,
            )
        )
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))
        service_logger = logging.getLogger("nemotron_testbed.service")
        self.addCleanup(service_logger.setLevel, service_logger.level)
        service_logger.setLevel(logging.INFO)

        results = [r async for r in service.Recognize(_no_requests(), _Context())]

        self.assertEqual(len(results), 1)
        self.assertEqual(sum(b.get() for b in GRPC_BUFFER_WAIT_MS._buckets), before_wait + 1)
        self.assertEqual(sum(b.get() for b in GRPC_SEND_MS._buckets), before_send + 1)

    @staticmethod
    def _envelope(seq, utterance_id, speech_end_wall, audio=b"\x00" * 16, preproc_delay_s=0.0):
        return RecognitionEnvelope(
            session_id="s",
            stream_id="M0",
            text="xin chao",
            is_final=True,
            ts_start=0,
            ts_end=1,
            confidence=0.5,
            language="vi-VN",
            seq=seq,
            utterance_id=utterance_id,
            audio_payload=audio,
            speech_end_wall=speech_end_wall,
            preproc_delay_s=preproc_delay_s,
        )

    async def _run(self, *envelopes):
        buffer = AsyncResponseBuffer(soft_limit=8)
        for envelope in envelopes:
            buffer.put(envelope)
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))
        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            results = [r async for r in service.Recognize(_no_requests(), _Context())]
        self.assertEqual(len(results), len(envelopes))
        return cm.output

    async def test_sent_line_has_join_fields_for_gateway_log(self):
        output = await self._run(self._envelope(7, 1, 0.0))
        sent = [m for m in output if "GRPC_SENT" in m]
        self.assertEqual(len(sent), 1)
        for field in ("session=s", "stream=M0", "seq=7", "final=True", "ts_start=0",
                      "ts_end=1", "text_len=8", "audio_bytes=16"):
            self.assertIn(field, sent[0])

    async def test_sent_record_has_structured_trace_fields(self):
        now = time.monotonic()
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(self._envelope(1, 4, now - 0.5))
        buffer.put(self._envelope(2, 4, now - 0.5))
        buffer.put(self._envelope(3, 5, 0.0, audio=b""))
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))
        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            [r async for r in service.Recognize(_no_requests(), _Context())]
        sent = [r for r in cm.records if getattr(r, "stage", None) == "GRPC_SENT"]
        self.assertEqual([r.seq for r in sent], [1, 2, 3])
        self.assertEqual([r.utterance_id for r in sent], [4, 4, 5])
        self.assertEqual([r.first_audio for r in sent], [True, False, False])
        self.assertEqual({r.rpc_id for r in sent}, {"rpc1"})
        self.assertEqual({r.stream_id for r in sent}, {"M0"})
        self.assertEqual(sent[0].audio_bytes, 16)
        self.assertGreaterEqual(sent[0].speech_end_to_sent_ms, 500.0)
        self.assertIsNone(sent[1].speech_end_to_sent_ms)
        self.assertGreaterEqual(sent[0].send_ms, 0.0)
        self.assertGreaterEqual(sent[0].buffer_wait_ms, 0.0)

    async def test_first_audio_flag_does_not_need_speech_end_wall(self):
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(self._envelope(1, 1, 0.0))
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))
        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            [r async for r in service.Recognize(_no_requests(), _Context())]
        sent = [r for r in cm.records if getattr(r, "stage", None) == "GRPC_SENT"]
        self.assertTrue(sent[0].first_audio)
        self.assertIsNone(sent[0].speech_end_to_sent_ms)

    @staticmethod
    def _service(trace, level):
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(GrpcTimingLogTest._envelope(1, 1, 0.0))
        buffer.put(_RPC_END)
        return SpeechStreamService(_Coordinator(buffer, trace, level))

    async def test_sent_line_logged_at_trace_level_without_debug_lines(self):
        service = self._service(True, "INFO")
        with self.assertLogs("nemotron_testbed.service", level="INFO") as cm:
            [r async for r in service.Recognize(_no_requests(), _Context())]
        self.assertEqual(len([m for m in cm.output if "GRPC_SENT" in m]), 1)
        self.assertTrue(all(r.levelno == logging.INFO for r in cm.records if "GRPC_SENT" in r.getMessage()))
        self.assertFalse(any("GRPC_OUT" in m for m in cm.output))

    async def test_sent_line_not_logged_when_trace_disabled(self):
        service = self._service(False, "DEBUG")
        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            [r async for r in service.Recognize(_no_requests(), _Context())]
        self.assertTrue(any("GRPC_OUT" in m for m in cm.output))
        self.assertFalse(any("GRPC_SENT" in m for m in cm.output))

    async def test_sent_line_dropped_when_trace_level_below_logger_level(self):
        service = self._service(True, "DEBUG")
        service_logger = logging.getLogger("nemotron_testbed.service")
        self.addCleanup(service_logger.setLevel, service_logger.level)
        service_logger.setLevel(logging.INFO)
        with self.assertNoLogs("nemotron_testbed.service", level="INFO"):
            [r async for r in service.Recognize(_no_requests(), _Context())]

    async def test_speech_end_metric_observed_once_per_utterance(self):
        before = sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_MS._buckets)
        now = time.monotonic()
        output = await self._run(
            self._envelope(1, 1, now - 0.5),
            self._envelope(2, 1, now - 0.5),
            self._envelope(3, 2, now - 0.3),
        )
        self.assertEqual(
            sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_MS._buckets), before + 2
        )
        sent = [m for m in output if "GRPC_SENT" in m]
        self.assertEqual(len(sent), 3)
        self.assertIn("speech_end_to_sent_ms=", sent[0])
        self.assertNotIn("speech_end_to_sent_ms=", sent[1])
        self.assertIn("speech_end_to_sent_ms=", sent[2])

    async def test_speech_end_metric_skipped_without_wall_or_audio(self):
        before = sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_MS._buckets)
        now = time.monotonic()
        output = await self._run(
            self._envelope(1, 1, 0.0),
            self._envelope(2, 2, now - 0.5, audio=b""),
        )
        self.assertEqual(
            sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_MS._buckets), before
        )
        self.assertFalse(any("speech_end_to_sent_ms=" in m for m in output))

    async def test_preproc_metric_adds_delay_for_preprocessed_stream(self):
        before_old = sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_MS._buckets)
        before = sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_PREPROC_MS._buckets)
        now = time.monotonic()
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(self._envelope(1, 1, now - 0.5, preproc_delay_s=1.5))
        buffer.put(self._envelope(2, 1, now - 0.5, preproc_delay_s=1.5))
        buffer.put(self._envelope(3, 2, now - 0.3, preproc_delay_s=1.5))
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))
        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            [r async for r in service.Recognize(_no_requests(), _Context())]

        self.assertEqual(
            sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_MS._buckets), before_old + 2
        )
        self.assertEqual(
            sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_PREPROC_MS._buckets), before + 2
        )
        sent = [r for r in cm.records if getattr(r, "stage", None) == "GRPC_SENT"]
        for first in (sent[0], sent[2]):
            self.assertAlmostEqual(
                first.speech_end_to_sent_preproc_ms - first.speech_end_to_sent_ms, 1500.0, delta=0.2
            )
        self.assertIsNone(sent[1].speech_end_to_sent_preproc_ms)
        lines = [m for m in cm.output if "GRPC_SENT" in m]
        self.assertIn("speech_end_to_sent_preproc_ms=", lines[0])
        self.assertNotIn("speech_end_to_sent_preproc_ms=", lines[1])

    async def test_preproc_metric_skipped_without_preprocessor(self):
        before = sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_PREPROC_MS._buckets)
        now = time.monotonic()
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(self._envelope(1, 1, now - 0.5))
        buffer.put(_RPC_END)
        service = SpeechStreamService(_Coordinator(buffer))
        with self.assertLogs("nemotron_testbed.service", level="DEBUG") as cm:
            [r async for r in service.Recognize(_no_requests(), _Context())]

        self.assertEqual(
            sum(b.get() for b in SPEECH_END_TO_AUDIO_SENT_PREPROC_MS._buckets), before
        )
        sent = [r for r in cm.records if getattr(r, "stage", None) == "GRPC_SENT"]
        self.assertGreaterEqual(sent[0].speech_end_to_sent_ms, 500.0)
        self.assertIsNone(sent[0].speech_end_to_sent_preproc_ms)
        self.assertFalse(any("speech_end_to_sent_preproc_ms=" in m for m in cm.output))


class _EventLog(logging.Handler):
    def __init__(self, events: list) -> None:
        super().__init__(logging.DEBUG)
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "stage", None) == "GRPC_CALL_ENDED":
            self.events.append(("log", record))


class _EndingCoordinator(_Coordinator):
    def __init__(self, buffer, events: list, finished=False, cancelled=False) -> None:
        super().__init__(buffer)
        self.rpc.finished = finished
        self.rpc.cancelled = cancelled
        self.events = events
        self.cancel_called = asyncio.Event()

    async def cancel_rpc(self, rpc_id, reason):
        self.events.append(("cancel", reason))
        self.cancel_called.set()


class _PeerContext(_Context):
    def __init__(self, peer="ipv4:10.0.0.5:41234", code=None, remaining=None) -> None:
        self._peer = peer
        self._code = code
        self._remaining = remaining

    def peer(self):
        return self._peer

    def code(self):
        return self._code

    def time_remaining(self):
        return self._remaining


class _BrokenContext(_Context):
    def peer(self):
        raise RuntimeError("no peer")

    def code(self):
        raise RuntimeError("no code")

    def time_remaining(self):
        raise RuntimeError("no deadline")


class GrpcCallEndedLogTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.events: list = []
        handler = _EventLog(self.events)
        service_logger = logging.getLogger("nemotron_testbed.service")
        service_logger.addHandler(handler)
        self.addCleanup(service_logger.removeHandler, handler)
        previous = service_logger.level
        service_logger.setLevel(logging.INFO)
        self.addCleanup(service_logger.setLevel, previous)

    async def _run(self, context, finished=False, cancelled=False):
        buffer = AsyncResponseBuffer(soft_limit=8)
        buffer.put(_RPC_END)
        coordinator = _EndingCoordinator(buffer, self.events, finished, cancelled)
        service = SpeechStreamService(coordinator)
        [r async for r in service.Recognize(_no_requests(), context)]

    async def test_early_end_logs_peer_and_code_before_cancel(self):
        await self._run(_PeerContext("ipv4:10.0.0.5:41234", None, 12.3456))

        self.assertEqual([kind for kind, _ in self.events], ["log", "cancel"])
        record = self.events[0][1]
        self.assertEqual(record.levelno, logging.INFO)
        self.assertEqual(record.rpc_id, "rpc1")
        self.assertEqual(record.peer, "ipv4:10.0.0.5:41234")
        self.assertEqual(record.grpc_code, "None")
        self.assertEqual(record.time_remaining_s, 12.346)
        self.assertIn("peer=ipv4:10.0.0.5:41234", record.getMessage())
        self.assertIn("grpc_code=None", record.getMessage())
        self.assertIn("time_remaining_s=12.346", record.getMessage())
        self.assertEqual(self.events[1][1], "gRPC call ended before engine pipeline completed")

    async def test_code_set_by_server_is_reported_by_name(self):
        import grpc

        await self._run(_PeerContext(code=grpc.StatusCode.CANCELLED))

        self.assertEqual(self.events[0][1].grpc_code, "CANCELLED")

    async def test_empty_peer_is_reported_as_dash(self):
        await self._run(_PeerContext(peer=""))

        self.assertEqual(self.events[0][1].peer, "-")

    async def test_context_errors_do_not_block_cancel(self):
        await self._run(_BrokenContext())

        self.assertEqual([kind for kind, _ in self.events], ["log", "cancel"])
        record = self.events[0][1]
        self.assertEqual(record.peer, "-")
        self.assertEqual(record.grpc_code, "unknown")
        self.assertIsNone(record.time_remaining_s)

    async def test_no_deadline_is_none_and_expired_deadline_is_zero(self):
        await self._run(_PeerContext(remaining=None))
        await self._run(_PeerContext(remaining=0))
        await self._run(_PeerContext(remaining=-0.2))

        remaining = [r.time_remaining_s for kind, r in self.events if kind == "log"]
        self.assertEqual(remaining, [None, 0, 0])
        self.assertIn("time_remaining_s=None", self.events[0][1].getMessage())

    async def test_time_remaining_is_written_to_json_even_when_zero(self):
        import json

        from nemotron_testbed.logging_config import JsonFormatter

        await self._run(_PeerContext(remaining=0))
        await self._run(_PeerContext(remaining=None))

        zero, none = (JsonFormatter().format(r) for kind, r in self.events if kind == "log")
        self.assertEqual(json.loads(zero)["time_remaining_s"], 0)
        self.assertEqual(json.loads(zero)["peer"], "ipv4:10.0.0.5:41234")
        self.assertEqual(json.loads(zero)["grpc_code"], "None")
        self.assertNotIn("time_remaining_s", json.loads(none))

    async def test_no_log_when_pipeline_finished(self):
        await self._run(_PeerContext(), finished=True)

        self.assertEqual(self.events, [])

    async def test_no_log_when_engine_cancelled_first(self):
        await self._run(_PeerContext(), cancelled=True)

        self.assertEqual(self.events, [])

    async def _real_call(self, timeout, cancel):
        import grpc
        import grpc.aio

        from nemotron_testbed.pb import speech_pb2, speech_pb2_grpc

        buffer = AsyncResponseBuffer(soft_limit=8)
        coordinator = _EndingCoordinator(buffer, self.events)
        server = grpc.aio.server()
        speech_pb2_grpc.add_SpeechStreamServicer_to_server(
            SpeechStreamService(coordinator), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        self.addAsyncCleanup(server.stop, None)

        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = speech_pb2_grpc.SpeechStreamStub(channel).Recognize(
                iter([speech_pb2.AudioChunk()]), timeout=timeout
            )
            if cancel:
                await asyncio.sleep(0.2)
                call.cancel()
            await asyncio.wait_for(coordinator.cancel_called.wait(), timeout=5)

        self.assertEqual([kind for kind, _ in self.events], ["log", "cancel"])
        return self.events[0][1]

    async def test_real_client_cancel_reports_peer_address(self):
        record = await self._real_call(timeout=None, cancel=True)

        self.assertRegex(record.peer, r"^ipv[46]:")
        self.assertEqual(record.grpc_code, "None")
        self.assertIsNone(record.time_remaining_s)

    async def test_real_client_cancel_with_deadline_reports_time_left(self):
        record = await self._real_call(timeout=30, cancel=True)

        self.assertRegex(record.peer, r"^ipv[46]:")
        self.assertGreater(record.time_remaining_s, 20)
        self.assertLess(record.time_remaining_s, 30)

    async def test_real_client_deadline_expiry_reports_almost_no_time_left(self):
        record = await self._real_call(timeout=0.3, cancel=False)

        self.assertIsNotNone(record.time_remaining_s, record.getMessage())
        self.assertLess(record.time_remaining_s, 0.1, record.getMessage())


if __name__ == "__main__":
    unittest.main()
