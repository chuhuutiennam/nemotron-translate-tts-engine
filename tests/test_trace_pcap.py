"""trace_pcap (docs/trace-pcap-design.md): pcap of the audio received ("in") and after AudX ("audx")."""

import asyncio
import dataclasses
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nemotron_testbed import trace_pcap
from nemotron_testbed.client_pcap import build_chunks, describe_flows, read_pcap, select_flow
from nemotron_testbed.config import AudioPreprocessSettings, TracePcapSettings
from nemotron_testbed.coordinator import EngineCoordinator
from nemotron_testbed.metrics import TRACE_PCAP_STREAMS
from nemotron_testbed.stats_logger import _labeled_counter
from nemotron_testbed.trace_pcap import PcapTracer
from test_coordinator_audio_preprocess import FakeStreamPool, ramp_request
from test_coordinator_flow import FakeRuntime, make_settings

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trace_pcap_to_wav.py"
_spec = importlib.util.spec_from_file_location("trace_pcap_to_wav", _SCRIPT)
to_wav = importlib.util.module_from_spec(_spec)
sys.modules["trace_pcap_to_wav"] = to_wav  # dataclasses look their module up
_spec.loader.exec_module(to_wav)

AUDX_DELAY = (80 + 64 + 100) * 16  # D = hop 80 + look-ahead 64 + budget 100 ms
LOGGER = "nemotron_testbed.trace_pcap"


def outcomes() -> dict:
    return dict(_labeled_counter(TRACE_PCAP_STREAMS, "outcome"))


def outcome_delta(before: dict, name: str) -> float:
    return outcomes().get(name, 0.0) - before.get(name, 0.0)


def sent_pcm(chunks: int, start: int = 0) -> np.ndarray:
    """PCM of ramp_request(start .. start + chunks - 1)."""
    return np.arange(start * 320 + 1, (start + chunks) * 320 + 1, dtype=np.int16)


def make_trace_file(directory: Path, chunks: int = 12) -> tuple[Path, np.ndarray]:
    """A trace written directly through PcapTracer: 20 ms chunks at 8 kHz RTP timestamps 1000 + 160 * i, and an
    "audx" flow that returns the same audio 640 samples late (like a preprocessor with D = 640)."""
    tracer = PcapTracer(TracePcapSettings(enabled=True, dir=str(directory), min_free_mb=0))
    trace = tracer.open_stream(
        rpc_id="rpc-1", session_id="call/1", stream_id="MO", task="speech_to_text", rtp_clock_rate=8000,
        first_rtp_timestamp=1000, audx_delay_samples=640,
    )
    pcm = sent_pcm(chunks)
    audio = pcm.astype(np.float32) / 32768.0
    delayed = np.concatenate((np.zeros(640, dtype=np.float32), audio))
    for i in range(chunks):
        piece = audio[i * 320 : (i + 1) * 320]
        trace.write_in(piece, 1000 + 160 * i, 100.0 + 0.02 * i)
        trace.note_processed(1000 + 160 * i, 320, 100.0 + 0.02 * i)
        trace.write_audx(delayed[i * 320 : (i + 1) * 320], lag=640)
    trace.write_audx(audio[-640:], lag=0)  # flush
    trace.close()
    tracer.close()
    return next(directory.glob("*.pcap")), pcm


class _TraceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.dir = self.tmp / "pcap"
        self.coordinators: list[EngineCoordinator] = []

    def tearDown(self):
        for coordinator in self.coordinators:
            if coordinator.pcap_tracer is not None:
                coordinator.pcap_tracer.close()

    def coordinator(self, *, audx=True, trace=True, settings_fields=None, **trace_fields):
        preproc = dataclasses.replace(
            AudioPreprocessSettings(), enabled=audx, mode="audx", left_context_ms=0, hop_ms=80,
            right_context_ms=64, crossfade_ms=0, compute_budget_ms=100,
        )
        trace_settings = TracePcapSettings(**{"enabled": trace, "dir": str(self.dir), "min_free_mb": 0, **trace_fields})
        settings = dataclasses.replace(
            make_settings(False), audio_preprocessing=preproc, trace_pcap=trace_settings, **(settings_fields or {})
        )
        coordinator = EngineCoordinator(settings)
        coordinator.runtime = FakeRuntime()
        if audx:
            pool = FakeStreamPool(gain=1.0)  # AudX that returns its input after the delay D
            pool.coordinator = coordinator
            coordinator.audio_preproc_pool = pool
        else:
            coordinator.audio_preproc_pool = None
        self.coordinators.append(coordinator)
        return coordinator

    async def run_call(self, coordinator, requests, *, flush=True):
        rpc = coordinator.open_rpc()
        for req in requests:
            await coordinator.accept_chunk(rpc.rpc_id, req)
        if flush:
            await coordinator.close_input(rpc.rpc_id)
        model_audio = [
            s.active_utterance.model_audio.samples for s in coordinator._streams.values() if s.active_utterance
        ]
        await coordinator.cancel_rpc(rpc.rpc_id, "test done")
        coordinator.pcap_tracer.close()
        return rpc, model_audio

    def trace_file(self) -> Path:
        files = sorted(self.dir.glob("*.pcap"))
        self.assertEqual(len(files), 1, files)
        return files[0]

    def flows(self) -> dict:
        return {flow.name: flow for flow in to_wav.flow_audio(read_pcap(self.trace_file()))}


class TraceOffTest(_TraceTestCase):
    async def test_off_creates_no_tracer_and_no_file(self):
        coordinator = self.coordinator(trace=False)
        self.assertIsNone(coordinator.pcap_tracer)
        rpc = coordinator.open_rpc()
        for i in range(10):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
        self.assertIsNone(next(iter(coordinator._streams.values())).pcap_trace)
        await coordinator.cancel_rpc(rpc.rpc_id, "done")
        self.assertFalse(self.dir.exists())


class TraceFlowsTest(_TraceTestCase):
    async def test_in_is_the_chunks_as_sent_and_audx_lines_up_with_it(self):
        coordinator = self.coordinator()
        with self.assertLogs(LOGGER, level="INFO") as logs:
            await self.run_call(coordinator, [ramp_request(i) for i in range(40)])
        stages = [r.stage for r in logs.records if hasattr(r, "stage")]
        self.assertIn("TRACE_PCAP_OPEN", stages)
        self.assertIn("TRACE_PCAP_CLOSE", stages)
        close = next(r for r in logs.records if getattr(r, "stage", "") == "TRACE_PCAP_CLOSE")
        self.assertEqual((close.packets_in, close.stopped_reason), (40, ""))
        self.assertGreater(close.packets_audx, 0)
        self.assertEqual(close.seconds_in, 0.8)
        self.assertIsNotNone(close.audx_stats)

        path = self.trace_file()
        self.assertRegex(path.name, r"^\d{8}-\d{6}_call-1_MO_[0-9a-f]{8}\.pcap$")
        flows = read_pcap(path)
        self.assertEqual(sorted((f.label, f.codec) for f in flows), [("audx", "L16"), ("in", "L16")])
        incoming = select_flow(flows, "in")
        payload = np.concatenate([np.frombuffer(p.payload, dtype=">i2") for p in incoming.packets])
        np.testing.assert_array_equal(payload, sent_pcm(40))  # byte for byte, big-endian
        self.assertEqual([p.timestamp for p in incoming.packets], [320 * i for i in range(40)])
        self.assertTrue(all(p.marker for p in incoming.packets))  # one 20 ms chunk = one packet

        lined_up = self.flows()
        np.testing.assert_array_equal(lined_up["audx"].samples, lined_up["in"].samples)
        self.assertEqual(lined_up["audx"].seconds, 0.8)  # the flush released the last D
        self.assertEqual((lined_up["audx"].gaps, lined_up["audx"].duplicate_packets), (0, 0))

    async def test_80ms_chunks_give_20ms_packets_on_the_in_grid(self):
        coordinator = self.coordinator()
        requests = []
        for i in range(20):  # rtpgw: 80 ms = 1280 samples, +640 ticks at 8 kHz
            req = ramp_request(0)
            req.rtp_timestamp, req.duration_ms = 640 * i, 80
            req.pcm = np.arange(i * 1280 + 1, (i + 1) * 1280 + 1, dtype="<i2").tobytes()
            requests.append(req)
        await self.run_call(coordinator, requests)
        flows = read_pcap(self.trace_file())
        incoming, audx = select_flow(flows, "in"), select_flow(flows, "audx")
        self.assertEqual([len(p.payload) for p in incoming.packets], [640] * 80)
        self.assertEqual([p.marker for p in incoming.packets], [True, False, False, False] * 20)
        self.assertTrue(all(len(p.payload) <= 640 for p in audx.packets))
        self.assertTrue(all(p.timestamp % 320 == 0 for p in audx.packets if not p.marker))
        lined_up = self.flows()
        np.testing.assert_array_equal(lined_up["audx"].samples, lined_up["in"].samples)
        np.testing.assert_array_equal(lined_up["in"].samples, np.arange(1, 20 * 1280 + 1, dtype=np.int16))

    async def test_dropped_chunk_is_only_in_in_and_gap_silence_lines_up(self):
        coordinator = self.coordinator(
            settings_fields={"rtp_gap_policy": "insert_silence", "rtp_max_gap_ms": 200}
        )
        requests = [ramp_request(i) for i in range(12)]
        late = ramp_request(4)  # 160 ms behind: dropped by the out-of-order policy
        late.pcm = np.full(320, 7, dtype="<i2").tobytes()
        requests.append(late)
        requests += [ramp_request(i) for i in range(13, 30)]  # chunk 12 lost: 20 ms gap, engine inserts silence
        await self.run_call(coordinator, requests)

        flows = self.flows()
        self.assertEqual((flows["in"].duplicate_packets, flows["in"].gaps), (1, 1))
        self.assertEqual((flows["audx"].duplicate_packets, flows["audx"].gaps), (0, 0))
        expected = sent_pcm(30)
        expected[12 * 320 : 13 * 320] = 0
        np.testing.assert_array_equal(flows["in"].samples, expected)
        np.testing.assert_array_equal(flows["audx"].samples, expected)  # silence sits in the gap, nothing shifted

    async def test_preprocessor_failure_keeps_audx_lined_up(self):
        coordinator = self.coordinator()
        rpc = coordinator.open_rpc()
        for i in range(20):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
        stream = next(iter(coordinator._streams.values()))

        def boom(audio):
            raise RuntimeError("bug in preprocessor")

        stream.preprocessor.process = boom
        with self.assertLogs("nemotron_testbed.coordinator", level="ERROR"):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(20))
        self.assertIsNone(stream.preprocessor)
        for i in range(21, 30):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
        await coordinator.cancel_rpc(rpc.rpc_id, "done")
        coordinator.pcap_tracer.close()

        flows = self.flows()
        np.testing.assert_array_equal(flows["audx"].samples, sent_pcm(30))
        self.assertEqual((flows["audx"].gaps, flows["audx"].duplicate_packets), (0, 0))

    async def test_cancel_without_flush_lacks_only_the_last_d(self):
        coordinator = self.coordinator()
        await self.run_call(coordinator, [ramp_request(i) for i in range(40)], flush=False)
        flows = self.flows()
        content = 40 * 320 - AUDX_DELAY
        np.testing.assert_array_equal(flows["audx"].samples[:content], sent_pcm(40)[:content])
        self.assertEqual(flows["audx"].seconds, content / 16000)
        np.testing.assert_array_equal(flows["in"].samples, sent_pcm(40))

    async def test_stream_without_audx_has_only_the_in_flow(self):
        coordinator = self.coordinator(audx=False)
        await self.run_call(coordinator, [ramp_request(i) for i in range(10)])
        flows = read_pcap(self.trace_file())
        self.assertEqual([(f.label, f.codec) for f in flows], [("in", "L16")])
        self.assertIn(b"a=label:in", self.trace_file().read_bytes())
        self.assertNotIn(b"a=label:audx", self.trace_file().read_bytes())

    async def test_tshark_decodes_both_flows_as_l16(self):
        tshark = shutil.which("tshark") or next(
            (p for p in ("C:/Program Files/Wireshark/tshark.exe",) if Path(p).is_file()), None
        )
        if tshark is None:
            self.skipTest("tshark is not installed")
        await self.run_call(self.coordinator(), [ramp_request(i) for i in range(20)])
        out = subprocess.run(
            [tshark, "-r", str(self.trace_file()), "-q", "-z", "rtp,streams"], capture_output=True, text=True,
            timeout=60,
        ).stdout
        streams = [line for line in out.splitlines() if "L16" in line]
        self.assertEqual(len(streams), 2, out)
        self.assertTrue(any(" 40000 " in line for line in streams) and any(" 40002 " in line for line in streams))


class TraceLimitsTest(_TraceTestCase):
    async def test_max_streams_skips_new_streams(self):
        coordinator = self.coordinator(max_streams=1)
        before = outcomes()
        first, second = coordinator.open_rpc(), coordinator.open_rpc()
        await coordinator.accept_chunk(first.rpc_id, ramp_request(0))
        other = ramp_request(0)
        other.session_id = "call-2"
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            await coordinator.accept_chunk(second.rpc_id, other)
        self.assertEqual([r.reason for r in logs.records if hasattr(r, "reason")], ["MAX_STREAMS"])
        self.assertEqual(outcome_delta(before, "skipped_max_streams"), 1)
        streams = {s.key.session_id: s for s in coordinator._streams.values()}
        self.assertIsNotNone(streams["call-1"].pcap_trace)
        self.assertIsNone(streams["call-2"].pcap_trace)
        await coordinator.cancel_rpc(first.rpc_id, "done")
        await coordinator.cancel_rpc(second.rpc_id, "done")

    async def test_max_seconds_stops_the_trace_not_the_call(self):
        coordinator = self.coordinator(max_seconds=1)
        before = outcomes()
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            rpc, model_audio = await self.run_call(coordinator, [ramp_request(i) for i in range(60)], flush=False)
        self.assertIn("MAX_SECONDS", [getattr(r, "reason", "") for r in logs.records])
        self.assertEqual(outcome_delta(before, "stopped_max_seconds"), 1)
        self.assertEqual(len(select_flow(read_pcap(self.trace_file()), "in").packets), 50)  # 1 s of 20 ms chunks
        self.assertEqual(model_audio, [60 * 320])  # the call itself got every chunk

    async def test_low_disk_skips_new_streams_and_stops_open_ones(self):
        coordinator = self.coordinator(min_free_mb=100)
        free = [50.0]
        coordinator.pcap_tracer.close()
        coordinator.pcap_tracer = PcapTracer(coordinator.settings.trace_pcap, disk_free_mb=lambda _path: free[0])
        before = outcomes()
        rpc = coordinator.open_rpc()
        await coordinator.accept_chunk(rpc.rpc_id, ramp_request(0))
        self.assertIsNone(next(iter(coordinator._streams.values())).pcap_trace)
        self.assertEqual(outcome_delta(before, "skipped_disk_low"), 1)
        await coordinator.cancel_rpc(rpc.rpc_id, "done")

        free[0] = 500.0
        rpc = coordinator.open_rpc()
        with mock.patch.object(trace_pcap, "DISK_CHECK_INTERVAL_S", 0.0):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(0))
            stream = next(iter(coordinator._streams.values()))
            self.assertIsNotNone(stream.pcap_trace)
            for _ in range(200):  # let the writer put the first packet on disk
                if stream.pcap_trace._state.packets["in"]:
                    break
                await asyncio.sleep(0.01)
            free[0] = 50.0
            for i in range(1, 200):
                await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i % 100))
                if not stream.pcap_trace.writing:
                    break
                await asyncio.sleep(0.01)
        self.assertFalse(stream.pcap_trace.writing)
        self.assertEqual(outcome_delta(before, "stopped_disk_low"), 1)
        await coordinator.cancel_rpc(rpc.rpc_id, "done")
        coordinator.pcap_tracer.close()
        self.assertTrue(read_pcap(self.trace_file()))  # what was written stays readable

    async def test_full_queue_stops_only_the_trace(self):
        coordinator = self.coordinator()
        coordinator.pcap_tracer.close()
        tracer = PcapTracer(coordinator.settings.trace_pcap, queue_max=3)
        release = threading.Event()
        handle = tracer._handle

        def slow_handle(item, open_files):
            release.wait(5)
            handle(item, open_files)

        tracer._handle = slow_handle
        coordinator.pcap_tracer = tracer
        before = outcomes()
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            rpc, model_audio = await self.run_call_released(coordinator, release, 20)
        self.assertIn("QUEUE_FULL", [getattr(r, "reason", "") for r in logs.records])
        self.assertEqual(outcome_delta(before, "stopped_queue_full"), 1)
        self.assertEqual(model_audio, [20 * 320])
        read_pcap(self.trace_file())  # what was written (at least the SIP/SDP header) stays a valid pcap

    async def run_call_released(self, coordinator, release, chunks):
        rpc = coordinator.open_rpc()
        for i in range(chunks):
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
        model_audio = [
            s.active_utterance.model_audio.samples for s in coordinator._streams.values() if s.active_utterance
        ]
        release.set()
        await coordinator.cancel_rpc(rpc.rpc_id, "done")
        coordinator.pcap_tracer.close()
        return rpc, model_audio

    async def test_write_error_stops_only_the_trace(self):
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.dir.write_text("not a directory")  # the trace directory cannot be created
        coordinator = self.coordinator()
        before = outcomes()
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            rpc = coordinator.open_rpc()
            for i in range(10):
                await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
            stream = next(iter(coordinator._streams.values()))
            for _ in range(100):
                if not stream.pcap_trace.writing:
                    break
                await asyncio.sleep(0.01)
            model_audio = stream.active_utterance.model_audio.samples if stream.active_utterance else None
            await coordinator.cancel_rpc(rpc.rpc_id, "done")
            coordinator.pcap_tracer.close()
        self.assertIn("IO_ERROR", [getattr(r, "reason", "") for r in logs.records])
        self.assertEqual(outcome_delta(before, "stopped_io_error"), 1)
        self.assertEqual(model_audio, 10 * 320)

    async def test_stopped_trace_stops_recording_what_entered_audx(self):
        coordinator = self.coordinator(max_seconds=1)
        rpc = coordinator.open_rpc()
        with self.assertLogs(LOGGER, level="WARNING"):
            for i in range(50):  # 1 s of 20 ms chunks: the trace stops on the last one
                await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
        trace = next(iter(coordinator._streams.values())).pcap_trace
        self.assertFalse(trace.writing)
        kept = len(trace._segments)
        for i in range(50, 100):  # the call goes on
            await coordinator.accept_chunk(rpc.rpc_id, ramp_request(i))
        self.assertEqual(len(trace._segments), kept)
        await coordinator.cancel_rpc(rpc.rpc_id, "done")

    async def test_stop_before_the_writer_opened_the_file_leaves_no_file_open(self):
        tracer = PcapTracer(TracePcapSettings(enabled=True, dir=str(self.dir), min_free_mb=0), queue_max=2)
        release = threading.Event()
        self.addCleanup(tracer.close)
        self.addCleanup(release.set)
        handle = tracer._handle

        def slow_handle(item, open_files):
            release.wait(5)
            handle(item, open_files)

        tracer._handle = slow_handle
        chunk = np.full(320, 0.1, dtype=np.float32)

        def open_stream(name):
            return tracer.open_stream(
                rpc_id=name, session_id=name, stream_id="MO", task="speech_to_text", rtp_clock_rate=8000,
                first_rtp_timestamp=0, audx_delay_samples=None,
            )

        first = open_stream("a")  # the writer takes this "open" and waits
        while tracer._queue.qsize():
            await asyncio.sleep(0.01)
        late = open_stream("b")  # its "open" waits in the queue (1/2)
        with self.assertLogs(LOGGER, level="WARNING"):
            late.write_in(chunk, 0, 0.0)  # 2/2
            late.write_in(chunk, 160, 0.0)  # queue full: the trace stops and its close cannot be queued
        self.assertEqual(late._state.stopped_reason, "QUEUE_FULL")
        release.set()
        while tracer._queue.qsize() >= 2:
            await asyncio.sleep(0.01)
        marker = open_stream("c")  # queued after b's items: once c is open, they were handled
        for _ in range(300):
            if marker._state.fh is not None:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(marker._state.fh)
        self.assertIsNone(late._state.fh)
        self.assertFalse(late._state.path.exists())  # stopped before its file was opened: no file
        for trace in (first, late, marker):
            trace.close()


class TraceToolsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.path, self.pcm = make_trace_file(self.tmp / "pcap")

    def test_flows_are_labelled_and_selectable(self):
        flows = read_pcap(self.path)
        listing = "\n".join(describe_flows(flows))
        self.assertIn("label=in", listing)
        self.assertIn("label=audx", listing)
        self.assertEqual(select_flow(flows, "audx").dst_port, 40002)
        self.assertEqual(select_flow(flows, "in").dst_port, 40000)

    def test_replaying_in_gives_back_the_chunks_as_sent(self):
        chunks, stats = build_chunks(select_flow(read_pcap(self.path), "in"), jitter="sim", resampler="linear")
        self.assertEqual(len(chunks), 12)
        self.assertEqual([c.rtp_timestamp for c in chunks], [160 * i for i in range(12)])
        self.assertTrue(all(c.rtp_clock_rate == 8000 and c.duration_ms == 20 for c in chunks))
        self.assertEqual(b"".join(c.pcm for c in chunks), self.pcm.astype("<i2").tobytes())
        self.assertEqual([c.end_of_stream for c in chunks], [False] * 11 + [True])
        self.assertAlmostEqual(chunks[3].due_s, 0.06)
        self.assertEqual(stats.packets, 12)

    def test_replaying_audx_gives_the_audio_asr_received(self):
        chunks, _ = build_chunks(select_flow(read_pcap(self.path), "audx"))
        self.assertEqual(b"".join(c.pcm for c in chunks), self.pcm.astype("<i2").tobytes())
        self.assertEqual(chunks[0].rtp_timestamp, 0)

    def test_replaying_a_chunk_older_than_the_first_goes_back_in_time(self):
        directory = self.tmp / "late"
        tracer = PcapTracer(TracePcapSettings(enabled=True, dir=str(directory), min_free_mb=0))
        trace = tracer.open_stream(
            rpc_id="rpc-2", session_id="call", stream_id="MO", task="speech_to_text", rtp_clock_rate=8000,
            first_rtp_timestamp=10_000, audx_delay_samples=None,
        )
        audio = np.zeros(320, dtype=np.float32)
        trace.write_in(audio, 10_000, 100.0)
        trace.write_in(audio, 10_000 - 640, 100.02)  # the gateway sent a chunk 80 ms older than the first one
        trace.close()
        tracer.close()
        chunks, _ = build_chunks(select_flow(read_pcap(next(directory.glob("*.pcap"))), "in"))
        self.assertEqual([c.rtp_timestamp for c in chunks], [0, -640 & 0xFFFFFFFF])

    def test_trace_pcap_to_wav_writes_lined_up_files(self):
        out = self.tmp / "wav"
        self.assertEqual(to_wav.main([str(self.path), "--out", str(out)]), 0)
        files = sorted(out.glob("*.wav"))
        self.assertEqual([f.name.rsplit("_", 1)[1] for f in files], ["audx.wav", "in.wav"])
        import wave

        data = []
        for f in files:
            with wave.open(str(f)) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
                data.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2"))
        np.testing.assert_array_equal(data[0], data[1])
        np.testing.assert_array_equal(data[1], self.pcm)

    def test_trace_pcap_to_wav_rejects_a_pcap_without_trace(self):
        other = self.tmp / "empty.pcap"
        other.write_bytes(trace_pcap.PCAP_FILE_HEADER)
        self.assertEqual(to_wav.main([str(other), "--out", str(self.tmp)]), 1)


class SdpTest(unittest.TestCase):
    def test_sdp_declares_l16_16k_with_labels(self):
        body = trace_pcap.sdp_body({"session": "a b\r\nc"}, ["in", "audx"])
        self.assertIn("a=rtpmap:96 L16/16000", body)
        self.assertIn("m=audio 40000 RTP/AVP 96", body)
        self.assertIn("m=audio 40002 RTP/AVP 96", body)
        self.assertIn("i=session=a_b__c", body)  # no line break can leak into the SDP

    def test_rtp_payload_is_big_endian(self):
        packet = trace_pcap.rtp_packet("in", 7, 640, True, np.array([1, -2], dtype=np.int16))
        self.assertEqual(packet[1], 0x80 | 96)
        self.assertEqual(packet[12:], b"\x00\x01\xff\xfe")


if __name__ == "__main__":
    unittest.main()
