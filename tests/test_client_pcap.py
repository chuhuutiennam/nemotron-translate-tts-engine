import contextlib
import hashlib
import importlib.util
import io
import json
import random
import socket
import struct
import sys
import tempfile
import time
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from nemotron_testbed.client_pcap import (
    HALF_BAND_COEF,
    HALF_BAND_DELAY,
    GatewayChunker,
    GatewayResampler,
    PcapError,
    RtpFlow,
    RtpPacket,
    build_chunks,
    decode_g711,
    describe_flows,
    read_pcap,
    select_flow,
    simulate_jitter_buffer,
    upsample_2x,
)
from nemotron_testbed.pb import speech_pb2

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "client_wav.py"
_spec = importlib.util.spec_from_file_location("client_wav_cli", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "rtpgw_golden.json").read_text(encoding="utf-8"))

MS = 1_000_000


# ---------------------------------------------------------------------------------------------------------
# Synthetic pcap / flow builders
# ---------------------------------------------------------------------------------------------------------


def random_payload(rng: random.Random, size: int = 160) -> bytes:
    return bytes(rng.randrange(256) for _ in range(size))


def rtp_frame(
    src: str,
    sport: int,
    dst: str,
    dport: int,
    seq: int,
    timestamp: int,
    payload_type: int,
    payload: bytes,
    *,
    ssrc: int = 0x11223344,
    marker: bool = False,
    vlan: bool = False,
    csrc: int = 0,
    extension: bool = False,
    padding: int = 0,
    fragment: bool = False,
    ip_protocol: int = 17,
    rtp_version: int = 2,
) -> bytes:
    first = (rtp_version << 6) | (0x20 if padding else 0) | (0x10 if extension else 0) | csrc
    rtp = bytes([first, (0x80 if marker else 0) | payload_type])
    rtp += struct.pack(">HII", seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc) + b"\x00\x00\x00\x01" * csrc
    if extension:
        rtp += struct.pack(">HH", 0xBEDE, 1) + b"\xaa\xbb\xcc\xdd"
    rtp += payload
    if padding:
        rtp += b"\x00" * (padding - 1) + bytes([padding])
    udp = struct.pack(">HHHH", sport, dport, 8 + len(rtp), 0) + rtp
    ip = struct.pack(
        ">BBHHHBBH4s4s",
        0x45,
        0,
        20 + len(udp),
        0,
        0x2000 if fragment else 0,
        64,
        ip_protocol,
        0,
        socket.inet_aton(src),
        socket.inet_aton(dst),
    )
    ethernet = b"\x00" * 12 + (b"\x81\x00\x00\x64" if vlan else b"") + b"\x08\x00"
    return ethernet + ip + udp


def write_pcap(
    path: Path,
    records: list[tuple[int, bytes]],
    *,
    endian: str = "<",
    nano: bool = False,
    linktype: int = 1,
) -> None:
    """records: (capture time in ns, Ethernet frame). Frames are re-wrapped for raw IP / Linux cooked link types."""
    magic = {("<", False): "d4c3b2a1", ("<", True): "4d3cb2a1", (">", False): "a1b2c3d4", (">", True): "a1b23c4d"}
    data = bytes.fromhex(magic[(endian, nano)]) + struct.pack(endian + "HHiIII", 2, 4, 0, 0, 65535, linktype)
    for time_ns, frame in records:
        if linktype == 101:
            frame = frame[18:] if frame[12:14] == b"\x81\x00" else frame[14:]
        elif linktype == 113:
            frame = b"\x00" * 14 + b"\x08\x00" + (frame[18:] if frame[12:14] == b"\x81\x00" else frame[14:])
        seconds, remainder = divmod(time_ns, 1_000_000_000)
        fraction = remainder if nano else remainder // 1000
        data += struct.pack(endian + "IIII", seconds, fraction, len(frame), len(frame)) + frame
    path.write_bytes(data)


def stream_records(
    dport: int,
    count: int,
    *,
    start_ns: int = 1_700_000_000 * 1_000_000_000,
    start_seq: int = 100,
    start_ts: int = 16_000,
    payload_type: int = 0,
    seed: int = 1,
    sport: int = 4000,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    **frame_options,
) -> list[tuple[int, bytes]]:
    rng = random.Random(seed)
    return [
        (
            start_ns + index * 20 * MS,
            rtp_frame(
                src, sport, dst, dport, start_seq + index, start_ts + index * 160, payload_type, random_payload(rng),
                **frame_options,
            ),
        )
        for index in range(count)
    ]


def packet(index: int, arrival_ms: float, *, seq: int | None = None, payload: bytes = b"\x00" * 160, pt: int = 0):
    seq = index if seq is None else seq
    return RtpPacket(round(arrival_ms * MS), seq & 0xFFFF, (16_000 + index * 160) & 0xFFFFFFFF, pt, False, payload)


def clean_flow(count: int, *, codec: str = "PCMU", pt: int = 0, seed: int = 7) -> RtpFlow:
    rng = random.Random(seed)
    packets = [packet(i, i * 20, payload=random_payload(rng), pt=pt) for i in range(count)]
    return RtpFlow("10.0.0.1:4000->10.0.0.2:20000", 20000, 0xABCD, codec, packets)


def int16_bytes(array: np.ndarray) -> bytes:
    return array.astype("<i2").tobytes()


# ---------------------------------------------------------------------------------------------------------
# Gateway algorithms vs. the real Go implementation (tests/fixtures/rtpgw_golden.json)
# ---------------------------------------------------------------------------------------------------------


class GoldenFixtureTest(unittest.TestCase):
    def test_fixture_declares_its_provenance(self):
        provenance = GOLDEN["provenance"]
        self.assertIn("rtpgw_rev", provenance)
        self.assertEqual(
            set(provenance["source_sha256"]),
            {"internal/audio/resampler.go", "internal/audio/chunker.go", "internal/audio/pcm.go", "internal/codec/g711.go"},
        )

    def test_g711_tables_match_gateway(self):
        wire = bytes(range(256))
        self.assertEqual(decode_g711(wire, "PCMU").tolist(), GOLDEN["g711_tables"]["pcmu"])
        self.assertEqual(decode_g711(wire, "PCMA").tolist(), GOLDEN["g711_tables"]["pcma"])

    def test_linear_upsample_matches_gateway_resampler(self):
        for case in GOLDEN["resample_linear"]:
            with self.subTest(case["name"]):
                out = upsample_2x(np.array(case["input"], dtype=np.int16))
                self.assertEqual(out.dtype, np.int16)
                self.assertEqual(out.tolist(), case["output"])

    def test_sinc_taps_match_gateway_bit_for_bit(self):
        self.assertEqual(HALF_BAND_COEF.tolist(), GOLDEN["sinc_half_band_coef"])

    def test_sinc_resampler_matches_gateway(self):
        for case in GOLDEN["resample_sinc"]:
            with self.subTest(case["name"]):
                resampler = GatewayResampler("sinc")
                self.assertEqual(resampler.delay_samples, case["delay_samples"])
                outputs = [resampler.process(np.array(p, dtype=np.int16)) for p in case["packets"]]
                self.assertTrue(all(out.dtype == np.int16 for out in outputs))
                self.assertEqual([out.tolist() for out in outputs], case["outputs"])
                tail = resampler.flush()
                self.assertEqual(tail.tolist() if tail.size else None, case["flush"])

    def test_pipeline_matches_gateway_decoder_resampler_and_chunker(self):
        # each case: packets through decode -> resampler -> chunker like rtpgw audio_pipeline.go, both resamplers
        self.assertEqual({case["resampler"] for case in GOLDEN["chunker"]}, {"linear", "sinc"})
        for case in GOLDEN["chunker"]:
            with self.subTest(case["name"]):
                packets = [
                    RtpPacket(i * 20 * MS, i, item["timestamp"], 8 if item["codec"] == "PCMA" else 0, False,
                              bytes.fromhex(item["payload_hex"]))
                    for i, item in enumerate(case["packets"])
                ]
                flow = RtpFlow("a:1->b:2", 2, 1, case["packets"][0]["codec"], packets)
                chunks, _ = build_chunks(flow, jitter="off", chunk_ms=case["chunk_ms"], resampler=case["resampler"])

                want = case["chunks"] + case["flush_chunks"]
                self.assertEqual(len(chunks), len(want))
                for got, expected in zip(chunks, want, strict=True):
                    self.assertEqual(got.chunk_seq, expected["chunk_seq"])
                    self.assertEqual(got.rtp_timestamp, expected["rtp_timestamp"])
                    self.assertEqual(got.duration_ms, expected["duration_ms"])
                    self.assertEqual(len(got.pcm), expected["bytes"])
                    self.assertEqual(hashlib.sha256(got.pcm).hexdigest(), expected["sha256"])
                    self.assertEqual(got.rtp_clock_rate, 8000)
                self.assertEqual([c.end_of_stream for c in chunks], [False] * (len(chunks) - 1) + [True])


class G711Test(unittest.TestCase):
    def test_pcmu_polarity_is_itu_like_the_gateway(self):
        # rtpgw g711.go decodes with ITU polarity since 2026-11 (wire 0x00 -> -32124); it was inverted before
        self.assertEqual(decode_g711(b"\x00\x80", "PCMU").tolist(), [-32124, 32124])

    def test_pcmu_silence_is_zero(self):
        self.assertEqual(decode_g711(b"\xff\x7f", "PCMU").tolist(), [0, 0])

    def test_pcma_is_standard(self):
        self.assertEqual(decode_g711(b"\xd5\x55", "PCMA").tolist(), [8, -8])

    def test_empty_payload(self):
        self.assertEqual(decode_g711(b"", "PCMU").size, 0)


class UpsampleTest(unittest.TestCase):
    def test_interpolates_and_holds_last_sample(self):
        self.assertEqual(upsample_2x(np.array([0, 10], dtype=np.int16)).tolist(), [0, 5, 10, 10])

    def test_midpoint_truncates_toward_zero_like_go(self):
        # floor would give -2 here, Go integer division gives -1
        self.assertEqual(upsample_2x(np.array([0, -3], dtype=np.int16)).tolist(), [0, -1, -3, -3])
        self.assertEqual(upsample_2x(np.array([0, 3], dtype=np.int16)).tolist(), [0, 1, 3, 3])

    def test_single_sample_is_held(self):
        self.assertEqual(upsample_2x(np.array([7], dtype=np.int16)).tolist(), [7, 7])

    def test_full_scale_does_not_overflow(self):
        out = upsample_2x(np.array([-32768, 32767], dtype=np.int16))
        self.assertEqual(out.tolist(), [-32768, -1, 32767, 32767])  # delta 65535 halves to 32767 (toward zero)

    def test_no_state_between_packets(self):
        first = upsample_2x(np.array([0, 100], dtype=np.int16))
        second = upsample_2x(np.array([200, 300], dtype=np.int16))
        self.assertEqual(first[-1], 100)
        self.assertEqual(second[0], 200)

    def test_linear_mode_of_the_gateway_resampler_is_upsample_2x(self):
        resampler = GatewayResampler("linear")
        pcm = np.array([0, 10, -7, 300], dtype=np.int16)
        self.assertEqual(resampler.process(pcm).tolist(), upsample_2x(pcm).tolist())
        self.assertEqual(resampler.delay_samples, 0)
        self.assertEqual(resampler.flush().size, 0)


class SincResamplerTest(unittest.TestCase):
    def test_default_is_sinc_with_a_26_sample_delay(self):
        self.assertEqual(GatewayResampler().mode, "sinc")
        self.assertEqual(GatewayResampler().delay_samples, HALF_BAND_DELAY)
        self.assertEqual(HALF_BAND_DELAY, 26)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            GatewayResampler("cubic")

    def test_output_does_not_depend_on_packet_split(self):
        rng = np.random.default_rng(1)
        pcm = rng.integers(-9000, 9000, 1000).astype(np.int16)
        whole = GatewayResampler("sinc").process(pcm)
        split = GatewayResampler("sinc")
        parts = [split.process(pcm[a:b]) for a, b in ((0, 1), (1, 160), (160, 161), (161, 700), (700, 1000))]
        self.assertEqual(np.concatenate(parts).tolist(), whole.tolist())

    def test_even_samples_are_the_input_delayed(self):
        pcm = np.arange(1, 101, dtype=np.int16)
        out = GatewayResampler("sinc").process(pcm)
        self.assertEqual(out[0::2][:HALF_BAND_DELAY].tolist(), [0] * HALF_BAND_DELAY)
        self.assertEqual(out[0::2][HALF_BAND_DELAY:].tolist(), pcm[: 100 - HALF_BAND_DELAY].tolist())

    def test_flush_returns_the_tail_once(self):
        resampler = GatewayResampler("sinc")
        self.assertEqual(resampler.flush().size, 0)  # nothing processed yet
        resampler.process(np.full(160, 1000, dtype=np.int16))
        tail = resampler.flush()
        self.assertEqual(tail.size, 2 * HALF_BAND_DELAY)
        self.assertEqual(tail[0::2][0], 1000)  # the delayed input comes out first
        self.assertEqual(resampler.flush().size, 0)

    def test_suppresses_the_image_above_4khz(self):
        t = np.arange(8000) / 8000
        tone = np.round(10000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
        out = GatewayResampler("sinc").process(tone).astype(np.float64)[200:]
        spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
        freqs = np.fft.rfftfreq(out.size, 1 / 16_000)
        image_db = 20 * np.log10(spectrum[np.argmin(abs(freqs - 7000))] / spectrum[np.argmin(abs(freqs - 1000))])
        self.assertLess(image_db, -70.0)
        linear = upsample_2x(tone).astype(np.float64)[200:]
        spectrum = np.abs(np.fft.rfft(linear * np.hanning(linear.size)))
        self.assertGreater(20 * np.log10(spectrum[np.argmin(abs(freqs - 7000))] / spectrum[np.argmin(abs(freqs - 1000))]), -40.0)


class GatewayChunkerTest(unittest.TestCase):
    def test_default_chunk_is_80ms(self):
        chunker = GatewayChunker()
        self.assertEqual(chunker.chunk_samples, 1280)

    def test_partial_data_waits(self):
        chunker = GatewayChunker(80)
        self.assertEqual(chunker.push(np.zeros(640, dtype=np.int16), 0, 0.0), [])
        chunks = chunker.push(np.zeros(640, dtype=np.int16), 320, 0.02)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].rtp_timestamp, 0)
        self.assertEqual(chunks[0].duration_ms, 80)
        self.assertEqual(chunks[0].due_s, 0.02)

    def test_remainder_gets_offset_timestamp(self):
        chunker = GatewayChunker(20)
        chunks = chunker.push(np.zeros(480, dtype=np.int16), 1000, 0.0)
        self.assertEqual([chunk.rtp_timestamp for chunk in chunks], [1000])
        tail = chunker.flush(0.0)
        self.assertEqual(tail.rtp_timestamp, 1000 + 160)
        self.assertEqual(tail.duration_ms, 10)

    def test_flush_of_empty_buffer_is_none(self):
        self.assertIsNone(GatewayChunker().flush(0.0))

    def test_empty_push_is_ignored(self):
        self.assertEqual(GatewayChunker().push(np.empty(0, dtype=np.int16), 0, 0.0), [])


# ---------------------------------------------------------------------------------------------------------
# pcap parsing
# ---------------------------------------------------------------------------------------------------------


class ReadPcapTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def pcap(self, records, name="capture.pcap", **options) -> Path:
        path = self.directory / name
        write_pcap(path, records, **options)
        return path

    def test_single_flow(self):
        flows = read_pcap(self.pcap(stream_records(20000, 5)))
        self.assertEqual(len(flows), 1)
        flow = flows[0]
        self.assertEqual(flow.key, "10.0.0.1:4000->10.0.0.2:20000")
        self.assertEqual(flow.dst_port, 20000)
        self.assertEqual(flow.codec, "PCMU")
        self.assertEqual(flow.ssrc, 0x11223344)
        self.assertEqual([p.seq for p in flow.packets], [100, 101, 102, 103, 104])
        self.assertEqual(flow.packets[1].timestamp, 16_160)
        self.assertEqual(len(flow.packets[0].payload), 160)
        self.assertAlmostEqual(flow.duration_s, 0.08)

    def test_flows_are_grouped_by_4_tuple_and_ordered_by_first_packet(self):
        late = stream_records(30000, 3, start_ns=1_700_000_005 * 1_000_000_000, payload_type=8, sport=5000)
        early = stream_records(20000, 4)
        flows = read_pcap(self.pcap(sorted(late + early, key=lambda r: r[0])))
        self.assertEqual([flow.dst_port for flow in flows], [20000, 30000])
        self.assertEqual([flow.codec for flow in flows], ["PCMU", "PCMA"])
        self.assertEqual([len(flow.packets) for flow in flows], [4, 3])

    def test_same_destination_port_from_two_sources_is_two_flows(self):
        records = stream_records(20000, 3) + stream_records(20000, 3, src="10.0.0.9", sport=4100, seed=2)
        flows = read_pcap(self.pcap(sorted(records, key=lambda r: r[0])))
        self.assertEqual(len(flows), 2)

    def test_interleaved_packets_keep_arrival_order_per_flow(self):
        a = stream_records(20000, 3)
        b = stream_records(30000, 3, seed=2)
        flows = read_pcap(self.pcap([a[0], b[0], a[1], b[1], a[2], b[2]]))
        self.assertEqual([p.seq for p in flows[0].packets], [100, 101, 102])

    def test_capture_variants(self):
        reference = None
        for options in (
            {"endian": "<", "nano": False},
            {"endian": "<", "nano": True},
            {"endian": ">", "nano": False},
            {"endian": ">", "nano": True},
            {"linktype": 101},
            {"linktype": 113},
        ):
            with self.subTest(options):
                flows = read_pcap(self.pcap(stream_records(20000, 4), **options))
                self.assertEqual(len(flows), 1)
                times = [p.recv_ns for p in flows[0].packets]
                self.assertEqual(times, [times[0] + i * 20 * MS for i in range(4)])
                payloads = [p.payload for p in flows[0].packets]
                if reference is None:
                    reference = payloads
                self.assertEqual(payloads, reference)

    def test_vlan_tagged_frames(self):
        flows = read_pcap(self.pcap(stream_records(20000, 3, vlan=True)))
        self.assertEqual(len(flows[0].packets), 3)

    def test_rtp_header_csrc_extension_and_padding_are_stripped(self):
        plain = read_pcap(self.pcap(stream_records(20000, 2, seed=3), name="plain.pcap"))[0]
        for options in ({"csrc": 2}, {"extension": True}, {"padding": 4}, {"csrc": 1, "extension": True, "padding": 3}):
            with self.subTest(options):
                flow = read_pcap(self.pcap(stream_records(20000, 2, seed=3, **options), name="variant.pcap"))[0]
                self.assertEqual([p.payload for p in flow.packets], [p.payload for p in plain.packets])

    def test_marker_bit(self):
        flow = read_pcap(self.pcap(stream_records(20000, 1, marker=True)))[0]
        self.assertTrue(flow.packets[0].marker)
        self.assertEqual(flow.packets[0].payload_type, 0)

    def test_non_rtp_traffic_is_ignored(self):
        junk = [
            (1_700_000_000 * 1_000_000_000, rtp_frame("10.0.0.1", 4000, "10.0.0.2", 20000, 1, 0, 0, b"x" * 160, ip_protocol=6)),
            (1_700_000_000 * 1_000_000_000, rtp_frame("10.0.0.1", 4000, "10.0.0.2", 20001, 1, 0, 0, b"x" * 160, rtp_version=1)),
            (1_700_000_000 * 1_000_000_000, rtp_frame("10.0.0.1", 4000, "10.0.0.2", 20002, 1, 0, 0, b"x" * 160, fragment=True)),
            (1_700_000_000 * 1_000_000_000, b"\x00" * 10),
        ]
        flows = read_pcap(self.pcap(junk + stream_records(20000, 3)))
        self.assertEqual([flow.dst_port for flow in flows], [20000])
        self.assertEqual(len(flows[0].packets), 3)

    def test_truncated_tail_record_is_ignored(self):
        path = self.pcap(stream_records(20000, 4))
        path.write_bytes(path.read_bytes()[:-40])
        flows = read_pcap(path)
        self.assertEqual(len(flows[0].packets), 3)

    def test_capture_without_rtp_has_no_flows(self):
        self.assertEqual(read_pcap(self.pcap([])), [])

    def test_pcapng_is_rejected_with_conversion_hint(self):
        path = self.directory / "capture.pcapng"
        path.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 60)
        with self.assertRaisesRegex(PcapError, "pcapng.*editcap"):
            read_pcap(path)

    def test_bad_magic(self):
        path = self.directory / "junk.pcap"
        path.write_bytes(b"NOTAPCAP" * 8)
        with self.assertRaisesRegex(PcapError, "bad magic"):
            read_pcap(path)

    def test_too_short(self):
        path = self.directory / "short.pcap"
        path.write_bytes(b"\xd4\xc3\xb2\xa1")
        with self.assertRaisesRegex(PcapError, "too short"):
            read_pcap(path)

    def test_unsupported_link_type(self):
        with self.assertRaisesRegex(PcapError, "link type"):
            read_pcap(self.pcap(stream_records(20000, 1), linktype=127))

    def test_missing_file(self):
        with self.assertRaises(OSError):
            read_pcap(self.directory / "missing.pcap")


class SelectFlowTest(unittest.TestCase):
    def setUp(self):
        self.flows = [
            RtpFlow("10.0.0.1:4000->10.0.0.2:20000", 20000, 1, "PCMU", [packet(0, 0)]),
            RtpFlow("10.0.0.1:4001->10.0.0.2:30000", 30000, 2, "PCMA", [packet(0, 5)]),
            RtpFlow("10.0.0.9:4002->10.0.0.2:30000", 30000, 3, "PCMU", [packet(0, 9)]),
        ]

    def test_single_flow_needs_no_selector(self):
        self.assertIs(select_flow(self.flows[:1]), self.flows[0])

    def test_several_flows_need_a_selector_and_are_listed(self):
        with self.assertRaisesRegex(PcapError, r"(?s)3 flows.*#0.*#1.*#2"):
            select_flow(self.flows)

    def test_by_index(self):
        self.assertIs(select_flow(self.flows, "#1"), self.flows[1])

    def test_by_destination_port(self):
        self.assertIs(select_flow(self.flows, "20000"), self.flows[0])

    def test_ambiguous_port_asks_for_the_full_key(self):
        with self.assertRaisesRegex(PcapError, "matches 2 flows"):
            select_flow(self.flows, "30000")

    def test_by_full_key(self):
        self.assertIs(select_flow(self.flows, "10.0.0.9:4002->10.0.0.2:30000"), self.flows[2])

    def test_unknown_selector(self):
        for selector in ("#7", "12345", "nonsense"):
            with self.subTest(selector), self.assertRaisesRegex(PcapError, "no flow matches"):
                select_flow(self.flows, selector)

    def test_no_flows(self):
        with self.assertRaisesRegex(PcapError, "no RTP flows"):
            select_flow([])

    def test_describe_flows(self):
        lines = describe_flows(self.flows)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("#1  10.0.0.1:4001->10.0.0.2:30000  PCMA  packets=1"))
        self.assertIn("pt={0: 1}", lines[0])
        self.assertIn("ssrc=0x00000002", lines[1])


# ---------------------------------------------------------------------------------------------------------
# Jitter buffer simulation
# ---------------------------------------------------------------------------------------------------------


def run_jitter(arrivals: list[tuple[int, float]]):
    """arrivals: (seq, arrival ms) in arrival order."""
    packets = [packet(seq, ms, seq=seq) for seq, ms in arrivals]
    released, stats = simulate_jitter_buffer(packets)
    return [(p.seq, at / MS) for p, at in released], stats


class JitterSimulationTest(unittest.TestCase):
    def test_in_order_stream_is_held_60ms_then_follows_arrivals(self):
        released, stats = run_jitter([(i, i * 20) for i in range(10)])
        self.assertEqual([seq for seq, _ in released], list(range(10)))
        self.assertEqual([at for _, at in released[:4]], [60, 60, 60, 60])
        self.assertEqual([at for _, at in released[4:]], [i * 20 for i in range(4, 10)])
        self.assertEqual((stats.received, stats.released, stats.dropped, stats.lost), (10, 10, 0, 0))

    def test_reordered_packets_are_released_in_sequence(self):
        released, stats = run_jitter([(0, 0), (2, 20), (1, 40), (3, 60), (4, 80)])
        self.assertEqual([seq for seq, _ in released], [0, 1, 2, 3, 4])
        self.assertEqual(stats.lost, 0)

    def test_missing_packet_is_declared_lost_after_the_hold_time(self):
        arrivals = [(i, i * 20) for i in range(10) if i != 3]
        released, stats = run_jitter(arrivals)
        self.assertEqual([seq for seq, _ in released], [0, 1, 2, 4, 5, 6, 7, 8, 9])
        self.assertEqual(stats.lost, 1)
        times = dict(released)
        self.assertEqual(times[2], 60)
        self.assertEqual(times[4], 140)  # held until 60 ms after the last release (60), on the 20 ms tick grid

    def test_late_packet_after_loss_is_dropped(self):
        arrivals = [(i, i * 20) for i in range(10) if i != 3] + [(3, 400)]
        released, stats = run_jitter(arrivals)
        self.assertEqual([seq for seq, _ in released], [0, 1, 2, 4, 5, 6, 7, 8, 9])
        self.assertEqual((stats.lost, stats.dropped), (1, 1))

    def test_duplicate_before_release_is_dropped(self):
        released, stats = run_jitter([(0, 0), (1, 20), (1, 25), (2, 40), (3, 60), (4, 80)])
        self.assertEqual([seq for seq, _ in released], [0, 1, 2, 3, 4])
        self.assertEqual(stats.dropped, 1)

    def test_duplicate_after_release_is_dropped(self):
        released, stats = run_jitter([(i, i * 20) for i in range(8)] + [(2, 200)])
        self.assertEqual([seq for seq, _ in released], list(range(8)))
        self.assertEqual(stats.dropped, 1)

    def test_sequence_number_wraparound(self):
        arrivals = [((65532 + i) & 0xFFFF, i * 20) for i in range(10)]
        released, stats = run_jitter(arrivals)
        self.assertEqual([seq for seq, _ in released], [(65532 + i) & 0xFFFF for i in range(10)])
        self.assertEqual((stats.lost, stats.dropped), (0, 0))

    def test_reorder_across_the_wrap(self):
        arrivals = [(65534, 0), (0, 20), (65535, 40), (1, 60), (2, 80)]
        released, _ = run_jitter(arrivals)
        self.assertEqual([seq for seq, _ in released], [65534, 65535, 0, 1, 2])

    def test_long_silence_does_not_stall_the_simulation(self):
        arrivals = [(0, 0), (1, 20), (2, 40), (3, 60), (4, 10_000), (5, 10_020), (6, 10_040), (7, 10_060)]
        released, stats = run_jitter(arrivals)
        self.assertEqual(len(released), 8)
        self.assertEqual(stats.lost, 0)

    def test_empty_input(self):
        released, stats = simulate_jitter_buffer([])
        self.assertEqual(released, [])
        self.assertEqual(stats.released, 0)


# ---------------------------------------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------------------------------------


def expected_pcm(flow: RtpFlow, packets=None, codec: str | None = None) -> bytes:
    return b"".join(
        int16_bytes(upsample_2x(decode_g711(p.payload, codec or flow.codec))) for p in (packets or flow.packets)
    )


class BuildChunksTest(unittest.TestCase):
    def test_clean_stream_reproduces_gateway_chunks(self):
        flow = clean_flow(50)  # 1 s
        chunks, stats = build_chunks(flow, resampler="linear")
        self.assertEqual(len(chunks), 13)
        self.assertEqual(b"".join(c.pcm for c in chunks), expected_pcm(flow))
        self.assertEqual([c.chunk_seq for c in chunks], list(range(13)))
        self.assertEqual([c.rtp_timestamp for c in chunks], [16_000 + i * 640 for i in range(13)])
        self.assertTrue(all(c.rtp_clock_rate == 8000 for c in chunks))
        self.assertTrue(all(c.duration_ms == 80 and len(c.pcm) == 2560 for c in chunks[:-1]))
        self.assertEqual((chunks[-1].duration_ms, len(chunks[-1].pcm)), (40, 1280))
        self.assertEqual([c.end_of_stream for c in chunks], [False] * 12 + [True])
        self.assertEqual((stats.packets, stats.jitter.released, stats.jitter.lost, stats.non_audio_skipped), (50, 50, 0, 0))

    def test_exact_multiple_marks_the_last_full_chunk_without_an_empty_tail(self):
        chunks, _ = build_chunks(clean_flow(80), resampler="linear")
        self.assertEqual(len(chunks), 20)
        self.assertEqual([c.end_of_stream for c in chunks], [False] * 19 + [True])
        self.assertEqual(chunks[-1].duration_ms, 80)

    def test_single_short_packet_is_one_partial_end_of_stream_chunk(self):
        chunks, _ = build_chunks(clean_flow(1), resampler="linear")
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].duration_ms, chunks[0].end_of_stream, chunks[0].chunk_seq), (20, True, 0))

    def test_chunk_ms_is_configurable(self):
        chunks, _ = build_chunks(clean_flow(8), chunk_ms=20, resampler="linear")
        self.assertEqual(len(chunks), 8)
        self.assertTrue(all(len(c.pcm) == 640 for c in chunks))
        self.assertEqual(chunks[1].rtp_timestamp - chunks[0].rtp_timestamp, 160)

    def test_due_time_is_when_the_gateway_has_the_chunk(self):
        chunks, _ = build_chunks(clean_flow(50))
        self.assertAlmostEqual(chunks[0].due_s, 0.060)  # fourth packet arrives at 60 ms, chunk needs four packets
        self.assertAlmostEqual(chunks[1].due_s, 0.140)
        self.assertEqual([c.due_s for c in chunks], sorted(c.due_s for c in chunks))

    def test_lost_packet_leaves_a_timestamp_gap_and_no_zero_fill(self):
        flow = clean_flow(50)
        lossy = RtpFlow(flow.key, flow.dst_port, flow.ssrc, flow.codec, [p for p in flow.packets if p.seq != 10])
        chunks, stats = build_chunks(lossy, resampler="linear")
        self.assertEqual(stats.jitter.lost, 1)
        total_samples = sum(len(c.pcm) for c in chunks) // 2
        self.assertEqual(total_samples, 49 * 320)
        self.assertEqual(b"".join(c.pcm for c in chunks), expected_pcm(lossy))
        steps = [b.rtp_timestamp - a.rtp_timestamp for a, b in zip(chunks, chunks[1:])]
        self.assertEqual(steps[:2], [640, 640])
        self.assertEqual(steps[2], 5 * 160)  # chunk 2 = packets 8, 9, 11, 12; chunk 3 starts at packet 13
        self.assertEqual(chunks[3].rtp_timestamp, 16_000 + 13 * 160)

    def test_reordered_capture_is_sorted_by_the_jitter_buffer(self):
        flow = clean_flow(4)
        a, b, c, d = flow.packets
        shuffled = RtpFlow(flow.key, flow.dst_port, flow.ssrc, flow.codec, [a, c, b, d])
        chunks, _ = build_chunks(shuffled, chunk_ms=20, resampler="linear")
        self.assertEqual([ch.rtp_timestamp for ch in chunks], [p.timestamp for p in (a, b, c, d)])

    def test_jitter_off_keeps_file_order_like_pcap_replay(self):
        flow = clean_flow(4)
        a, b, c, d = flow.packets
        shuffled = RtpFlow(flow.key, flow.dst_port, flow.ssrc, flow.codec, [a, c, b, d])
        chunks, stats = build_chunks(shuffled, jitter="off", chunk_ms=20, resampler="linear")
        self.assertEqual([ch.rtp_timestamp for ch in chunks], [p.timestamp for p in (a, c, b, d)])
        self.assertEqual((stats.jitter.released, stats.jitter.dropped, stats.jitter.lost), (4, 0, 0))

    def test_jitter_off_on_a_clean_stream_gives_the_same_audio(self):
        flow = clean_flow(50)
        sim, _ = build_chunks(flow, jitter="sim")
        off, _ = build_chunks(flow, jitter="off")
        self.assertEqual([(c.pcm, c.rtp_timestamp, c.chunk_seq, c.end_of_stream) for c in sim],
                         [(c.pcm, c.rtp_timestamp, c.chunk_seq, c.end_of_stream) for c in off])
        self.assertAlmostEqual(off[0].due_s, 0.060)

    def test_comfort_noise_and_dtmf_are_skipped(self):
        flow = clean_flow(10)
        rng = random.Random(3)
        extra = [
            packet(10, 200, payload=b"\x10", pt=13),
            packet(11, 220, payload=b"\x01\x0a\x00\xa0", pt=101),
        ]
        voice = [packet(12 + i, 240 + i * 20, payload=random_payload(rng)) for i in range(4)]
        mixed = RtpFlow(flow.key, flow.dst_port, flow.ssrc, flow.codec, flow.packets + extra + voice)
        chunks, stats = build_chunks(mixed, resampler="linear")
        self.assertEqual(stats.non_audio_skipped, 2)
        self.assertEqual(b"".join(c.pcm for c in chunks), expected_pcm(mixed, flow.packets + voice))

    def test_pcma_flow_uses_a_law(self):
        flow = clean_flow(8, codec="PCMA", pt=8)
        chunks, _ = build_chunks(flow, resampler="linear")
        self.assertEqual(b"".join(c.pcm for c in chunks), expected_pcm(flow))
        self.assertNotEqual(expected_pcm(flow), expected_pcm(flow, codec="PCMU"))

    def test_payload_type_switches_the_decoder_mid_stream(self):
        flow = clean_flow(8)
        rng = random.Random(5)
        a_law = [packet(8 + i, (8 + i) * 20, payload=random_payload(rng), pt=8) for i in range(4)]
        mixed = RtpFlow(flow.key, flow.dst_port, flow.ssrc, "PCMU", flow.packets + a_law)
        chunks, _ = build_chunks(mixed, resampler="linear")
        want = expected_pcm(flow) + expected_pcm(flow, a_law, codec="PCMA")
        self.assertEqual(b"".join(c.pcm for c in chunks), want)

    def test_non_g711_flow_is_rejected(self):
        flow = RtpFlow("a:1->b:2", 2, 1, "opus", [packet(0, 0, pt=111)])
        with self.assertRaisesRegex(PcapError, "only G.711"):
            build_chunks(flow)

    def test_unknown_jitter_mode(self):
        with self.assertRaises(ValueError):
            build_chunks(clean_flow(2), jitter="bogus")

    def test_flow_with_only_comfort_noise_has_no_chunks(self):
        flow = RtpFlow("a:1->b:2", 2, 1, "PCMU", [packet(i, i * 20, payload=b"\x10", pt=13) for i in range(5)])
        chunks, stats = build_chunks(flow)
        self.assertEqual(chunks, [])
        self.assertEqual(stats.non_audio_skipped, 5)

    def test_timestamp_wraparound(self):
        rng = random.Random(1)
        packets = [
            RtpPacket(i * 20 * MS, i, (0xFFFFFFFF - 330 + i * 160) & 0xFFFFFFFF, 0, False, random_payload(rng))
            for i in range(9)
        ]
        chunks, _ = build_chunks(RtpFlow("a:1->b:2", 2, 1, "PCMU", packets), resampler="linear")
        self.assertTrue(all(0 <= c.rtp_timestamp <= 0xFFFFFFFF for c in chunks))
        self.assertEqual(chunks[1].rtp_timestamp, (packets[4].timestamp) & 0xFFFFFFFF)

    def test_default_sinc_shifts_timestamps_by_the_filter_delay_and_flushes_its_tail(self):
        flow = clean_flow(50)
        chunks, _ = build_chunks(flow)
        resampler = GatewayResampler("sinc")
        want = b"".join(int16_bytes(resampler.process(decode_g711(p.payload, "PCMU"))) for p in flow.packets)
        want += int16_bytes(resampler.flush())
        self.assertEqual(b"".join(c.pcm for c in chunks), want)
        self.assertEqual(len(want) // 2, 50 * 320 + 2 * HALF_BAND_DELAY)
        self.assertEqual([c.rtp_timestamp for c in chunks], [16_000 - HALF_BAND_DELAY + i * 640 for i in range(13)])
        self.assertEqual((chunks[-1].duration_ms, len(chunks[-1].pcm)), (43, 1384))  # 40 ms + 3.25 ms of filter tail
        self.assertEqual([c.end_of_stream for c in chunks], [False] * 12 + [True])

    def test_sinc_filter_tail_alone_becomes_the_end_of_stream_chunk(self):
        chunks, _ = build_chunks(clean_flow(80))  # 1.6 s = 20 full chunks, then only the 52-sample tail
        self.assertEqual(len(chunks), 21)
        self.assertEqual((len(chunks[-1].pcm), chunks[-1].end_of_stream), (2 * 2 * HALF_BAND_DELAY, True))
        self.assertEqual(chunks[-1].rtp_timestamp, 16_000 + 80 * 160 - HALF_BAND_DELAY)

    def test_sinc_state_runs_across_a_lost_packet(self):
        flow = clean_flow(20)
        lossy = RtpFlow(flow.key, flow.dst_port, flow.ssrc, flow.codec, [p for p in flow.packets if p.seq != 5])
        chunks, stats = build_chunks(lossy)
        self.assertEqual(stats.jitter.lost, 1)
        resampler = GatewayResampler("sinc")  # no reset at the gap, like rtpgw
        want = b"".join(int16_bytes(resampler.process(decode_g711(p.payload, "PCMU"))) for p in lossy.packets)
        self.assertEqual(b"".join(c.pcm for c in chunks), want + int16_bytes(resampler.flush()))

    def test_unknown_resampler(self):
        with self.assertRaises(ValueError):
            build_chunks(clean_flow(2), resampler="cubic")


# ---------------------------------------------------------------------------------------------------------
# client_wav.py command line (fake gRPC)
# ---------------------------------------------------------------------------------------------------------


class _FakeCall:
    def __init__(self, request_iterator, results):
        self._requests = request_iterator
        self._results = results
        self.received = []

    def __iter__(self):
        for request in self._requests:  # blocks until the client closes the request stream
            self.received.append(request)
        yield from self._results


class _FakeChannel:
    def close(self):
        pass


def write_wav(path: Path, samples: np.ndarray, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())


class ClientCliTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_main(self, argv, results=()):
        holder = {}

        def fake_stub(_channel):
            def recognize(request_iterator):
                holder["call"] = _FakeCall(request_iterator, results)
                return holder["call"]

            return types.SimpleNamespace(Recognize=recognize)

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["client_wav.py", *argv]),
            mock.patch.object(cli.grpc, "insecure_channel", return_value=_FakeChannel()) as channel,
            mock.patch.object(cli.speech_pb2_grpc, "SpeechStreamStub", side_effect=fake_stub),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                cli.main()
            finally:
                self.stdout, self.stderr, self.channel = stdout.getvalue(), stderr.getvalue(), channel
        return holder["call"].received if "call" in holder else None

    def two_flow_pcap(self) -> Path:
        path = self.directory / "two.pcap"
        a = stream_records(20000, 45, seed=1)
        b = stream_records(30000, 30, start_ns=1_700_000_002 * 1_000_000_000, payload_type=8, sport=5000, seed=2)
        write_pcap(path, sorted(a + b, key=lambda r: r[0]))
        return path

    def single_flow_pcap(self, count=45) -> Path:
        path = self.directory / "one.pcap"
        write_pcap(path, stream_records(20000, count))
        return path

    # --- pcap mode -------------------------------------------------------------------------------------

    def test_list_flows_prints_and_does_not_connect(self):
        self.run_main(["--pcap", str(self.two_flow_pcap()), "--list-flows"])
        lines = self.stdout.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("->10.0.0.2:20000", lines[0])
        self.assertIn("PCMA", lines[1])
        self.channel.assert_not_called()

    def test_several_flows_without_flow_option_fails_with_the_list(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--pcap", str(self.two_flow_pcap()), "--no-realtime"])
        self.assertIn("2 flows", str(ctx.exception.code))
        self.assertIn("20000", str(ctx.exception.code))
        self.channel.assert_not_called()

    def test_sends_the_chunks_the_gateway_would(self):
        path = self.two_flow_pcap()
        received = self.run_main(
            [
                "--pcap", str(path), "--flow", "30000", "--no-realtime", "--session-id", "s1", "--stream-id", "MT",
                "--task", "translate", "--language", "en", "--language-source", "vi",
            ]
        )
        flow = select_flow(read_pcap(path), "30000")
        expected, _ = build_chunks(flow)
        self.assertEqual(len(received), len(expected))
        for sent, want in zip(received, expected, strict=True):
            self.assertEqual(sent.pcm, want.pcm)
            self.assertEqual(sent.rtp_timestamp, want.rtp_timestamp)
            self.assertEqual(sent.chunk_seq, want.chunk_seq)
            self.assertEqual(sent.duration_ms, want.duration_ms)
            self.assertEqual(sent.end_of_stream, want.end_of_stream)
            self.assertEqual(sent.rtp_clock_rate, 8000)
            self.assertEqual((sent.sample_rate, sent.channels), (16000, 1))
            self.assertEqual((sent.session_id, sent.stream_id), ("s1", "MT"))
            self.assertEqual((sent.task, sent.language, sent.language_source), ("translate", "en", "vi"))
        self.assertEqual([c.end_of_stream for c in received], [False] * (len(received) - 1) + [True])
        self.assertIn("flow=10.0.0.1:5000->10.0.0.2:30000 codec=PCMA jitter=sim chunk_ms=80", self.stdout)

    def test_flow_by_index_and_default_session_id(self):
        received = self.run_main(["--pcap", str(self.two_flow_pcap()), "--flow", "#0", "--no-realtime"])
        self.assertTrue(received[0].session_id.startswith("pcap-"))
        self.assertEqual(len(received), 12)  # 45 packets = 900 ms = 11 full chunks + a 20 ms tail...

    def test_jitter_off_and_chunk_ms_options_reach_the_engine(self):
        received = self.run_main(
            ["--pcap", str(self.single_flow_pcap(10)), "--jitter", "off", "--chunk-ms", "20", "--no-realtime"]
        )
        self.assertEqual(len(received), 11)  # 10 packets of 20 ms, then the 3.25 ms sinc filter tail
        self.assertTrue(all(len(c.pcm) == 640 for c in received[:-1]))
        self.assertEqual(len(received[-1].pcm), 104)
        self.assertIn("jitter=off chunk_ms=20 resampler=sinc", self.stdout)

    def test_gw_resampler_linear_with_pcap(self):
        path = self.single_flow_pcap(10)
        received = self.run_main(["--pcap", str(path), "--chunk-ms", "20", "--no-realtime", "--gw-resampler", "linear"])
        expected, _ = build_chunks(select_flow(read_pcap(path)), chunk_ms=20, resampler="linear")
        self.assertEqual([c.pcm for c in received], [c.pcm for c in expected])
        self.assertEqual([c.rtp_timestamp for c in received], [16_000 + i * 160 for i in range(10)])
        self.assertIn("resampler=linear", self.stdout)

    def test_gw_resampler_reaches_the_wav_narrowband_path(self):
        try:
            import soxr  # noqa: F401
        except ImportError:
            self.skipTest("soxr is not installed")
        path = self.sine_wav(0.2)
        for mode in ("sinc", "linear"):
            with self.subTest(mode):
                received = self.run_main([str(path), "--no-realtime", "--narrowband", "--gw-resampler", mode])
                sent = np.frombuffer(b"".join(c.pcm for c in received), dtype="<i2")
                want = (np.clip(cli.to_narrowband(cli.read_wav(path), mode), -1.0, 0.9999695) * 32768.0).astype("<i2")
                np.testing.assert_array_equal(sent, want)
                self.assertIn(f"narrowband=True resampler={mode}", self.stdout)

    def test_gw_resampler_without_narrowband_is_reported_as_ignored(self):
        self.run_main([str(self.sine_wav(0.2)), "--no-realtime", "--gw-resampler", "linear"])
        self.assertIn("--gw-resampler ignored without --narrowband", self.stderr)

    # --- trace_pcap file of the engine (docs/trace-pcap-design.md) -----------------------------------------

    def test_trace_pcap_flow_is_sent_as_recorded(self):
        from test_trace_pcap import make_trace_file

        path, pcm = make_trace_file(self.directory / "trace")
        self.run_main(["--pcap", str(path), "--list-flows"])
        self.assertIn("label=in", self.stdout)
        self.assertIn("label=audx", self.stdout)
        for label in ("in", "audx"):
            with self.subTest(label):
                received = self.run_main(
                    ["--pcap", str(path), "--flow", label, "--no-realtime", "--jitter", "off", "--chunk-ms", "40"]
                )
                expected, _ = build_chunks(select_flow(read_pcap(path), label))
                self.assertEqual([c.pcm for c in received], [c.pcm for c in expected])
                self.assertEqual(b"".join(c.pcm for c in received), pcm.astype("<i2").tobytes())
                self.assertEqual([c.rtp_timestamp for c in received], [c.rtp_timestamp for c in expected])
                self.assertTrue(all(c.rtp_clock_rate == 8000 and c.sample_rate == 16000 for c in received))
                self.assertEqual([c.end_of_stream for c in received], [False] * (len(received) - 1) + [True])
                self.assertIn(f"codec=L16 label={label} (trace_pcap)", self.stdout)
                self.assertIn("--jitter, --chunk-ms ignored for a trace_pcap flow", self.stderr)

    def test_pace_capture_waits_for_the_gateway_timeline(self):
        path = self.single_flow_pcap(25)  # 500 ms of audio, last chunk ready at ~0.46 s, first at 0.06 s
        started = time.monotonic()
        received = self.run_main(["--pcap", str(path), "--pace", "capture"])
        elapsed = time.monotonic() - started
        self.assertEqual(len(received), 7)
        self.assertGreaterEqual(elapsed, 0.35)

    def test_pace_chunk_sleeps_one_chunk_duration_per_chunk(self):
        path = self.single_flow_pcap(8)  # 2 chunks of 80 ms
        started = time.monotonic()
        self.run_main(["--pcap", str(path)])
        self.assertGreaterEqual(time.monotonic() - started, 0.15)

    def test_timeline_works_without_a_wav(self):
        results = [
            speech_pb2.RecognitionResult(text="hello", is_final=True, seq=1, stream_id="MO", language="en",
                                         ts_start=16_000, ts_end=16_640)
        ]
        self.run_main(["--pcap", str(self.single_flow_pcap(10)), "--no-realtime", "--timeline"], results)
        self.assertIn("[FINAL]", self.stdout)
        self.assertIn("lag=", self.stdout)
        self.assertIn("Timeline (seconds since the first chunk was sent):", self.stdout)
        self.assertIn("sent_to_result", self.stdout)  # random G.711 noise counts as speech for the energy marker

    def test_timeline_out_is_written(self):
        out = self.directory / "timeline.txt"
        self.run_main(["--pcap", str(self.single_flow_pcap(10)), "--no-realtime", "--timeline-out", str(out)])
        self.assertIn("input_end", out.read_text(encoding="utf-8"))

    def test_wav_only_options_are_reported_as_ignored(self):
        self.run_main(["--pcap", str(self.single_flow_pcap(10)), "--no-realtime", "--profile", "rtpgw", "--rtp-start", "5"])
        self.assertIn("--profile, --rtp-start ignored with --pcap", self.stderr)

    def test_bad_capture_is_a_clean_error(self):
        path = self.directory / "junk.pcap"
        path.write_bytes(b"x" * 100)
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--pcap", str(path)])
        self.assertTrue(str(ctx.exception.code).startswith("error: "))

    def test_missing_capture_is_a_clean_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--pcap", str(self.directory / "missing.pcap")])
        self.assertTrue(str(ctx.exception.code).startswith("error: "))

    def test_capture_without_rtp_is_a_clean_error(self):
        path = self.directory / "empty.pcap"
        write_pcap(path, [])
        for argv in (["--list-flows"], ["--no-realtime"]):
            with self.subTest(argv), self.assertRaises(SystemExit) as ctx:
                self.run_main(["--pcap", str(path), *argv])
            self.assertIn("no RTP flows", str(ctx.exception.code))

    def test_non_g711_capture_is_a_clean_error(self):
        path = self.directory / "opus.pcap"
        write_pcap(path, stream_records(20000, 5, payload_type=111))
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--pcap", str(path), "--no-realtime"])
        self.assertIn("only G.711", str(ctx.exception.code))

    # --- argument validation ---------------------------------------------------------------------------

    def test_needs_exactly_one_input(self):
        wav = self.directory / "a.wav"
        write_wav(wav, np.zeros(1600))
        for argv in ([], [str(wav), "--pcap", str(self.single_flow_pcap())]):
            with self.subTest(argv), self.assertRaises(SystemExit) as ctx:
                self.run_main(argv)
            self.assertEqual(ctx.exception.code, 2)
        self.assertIn("either a WAV file or --pcap", self.stderr)

    def test_pcap_only_options_need_pcap(self):
        wav = self.directory / "a.wav"
        write_wav(wav, np.zeros(1600))
        for extra in (["--flow", "1"], ["--list-flows"], ["--jitter", "off"], ["--pace", "capture"]):
            with self.subTest(extra), self.assertRaises(SystemExit) as ctx:
                self.run_main([str(wav), *extra])
            self.assertEqual(ctx.exception.code, 2)

    # --- WAV regression --------------------------------------------------------------------------------

    def sine_wav(self, seconds: float = 1.0) -> Path:
        path = self.directory / "tone.wav"
        t = np.arange(int(16_000 * seconds)) / 16_000
        write_wav(path, 0.5 * np.sin(2 * np.pi * 440 * t))
        return path

    def test_wav_defaults_are_unchanged(self):
        received = self.run_main([str(self.sine_wav()), "--no-realtime", "--session-id", "w"])
        self.assertEqual(len(received), 50)
        self.assertTrue(all(len(c.pcm) == 640 and c.duration_ms == 20 for c in received))
        self.assertEqual([c.rtp_timestamp for c in received], [i * 320 for i in range(50)])
        self.assertTrue(all(c.rtp_clock_rate == 16_000 for c in received))
        self.assertEqual([c.chunk_seq for c in received], list(range(50)))
        self.assertEqual([c.end_of_stream for c in received], [False] * 49 + [True])
        self.assertTrue(all((c.sample_rate, c.channels, c.language, c.task) == (16_000, 1, "vi", "speech_to_text") for c in received))
        self.assertIn("chunk_ms=20 rtp_clock_rate=16000 rtp_start=0 narrowband=False", self.stdout)

    def test_wav_sends_the_samples_unmodified(self):
        path = self.sine_wav()
        received = self.run_main([str(path), "--no-realtime"])
        original = cli.read_wav(path)
        sent = np.frombuffer(b"".join(c.pcm for c in received), dtype="<i2")
        np.testing.assert_array_equal(sent, (np.clip(original, -1.0, 0.9999695) * 32768.0).astype("<i2"))

    def test_wav_rtp_options(self):
        received = self.run_main(
            [str(self.sine_wav(0.2)), "--no-realtime", "--chunk-ms", "50", "--rtp-clock-rate", "8000", "--rtp-start", "1000"]
        )
        self.assertEqual([c.rtp_timestamp for c in received], [1000, 1400, 1800, 2200])
        self.assertTrue(all(c.rtp_clock_rate == 8000 for c in received))

    def test_wav_last_chunk_is_partial_when_the_length_is_not_a_multiple(self):
        received = self.run_main([str(self.sine_wav(0.25)), "--no-realtime", "--chunk-ms", "80"])
        self.assertEqual([c.duration_ms for c in received], [80, 80, 80, 10])
        self.assertEqual([len(c.pcm) for c in received], [2560, 2560, 2560, 320])
        self.assertTrue(received[-1].end_of_stream)

    def test_wav_profile_rtpgw(self):
        try:
            import soxr  # noqa: F401
        except ImportError:
            self.skipTest("soxr is not installed")
        received = self.run_main([str(self.sine_wav()), "--no-realtime", "--profile", "rtpgw"])
        self.assertTrue(all(len(c.pcm) == 2560 for c in received[:-1]))
        self.assertTrue(all(c.rtp_clock_rate == 8000 for c in received))
        self.assertEqual(received[1].rtp_timestamp - received[0].rtp_timestamp, 640)
        self.assertIn("narrowband=True", self.stdout)

    def test_wav_profile_values_can_be_overridden(self):
        received = self.run_main(
            [str(self.sine_wav(0.2)), "--no-realtime", "--profile", "rtpgw", "--no-narrowband", "--chunk-ms", "40"]
        )
        self.assertEqual(len(received[0].pcm), 1280)
        self.assertEqual(received[0].rtp_clock_rate, 8000)
        self.assertIn("narrowband=False", self.stdout)

    def test_wav_chunks_helper_marks_end_of_stream_on_an_exact_multiple(self):
        chunks = cli.wav_chunks(np.zeros(1280, dtype=np.float32), 80, 8000, 0)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].end_of_stream)
        chunks = cli.wav_chunks(np.zeros(2560, dtype=np.float32), 80, 8000, 0)
        self.assertEqual([c.end_of_stream for c in chunks], [False, True])
        self.assertEqual([c.due_s for c in chunks], [0.0, 0.08])

    def test_wav_timeline_output(self):
        results = [speech_pb2.RecognitionResult(text="x", is_final=False, seq=1, ts_start=0, ts_end=3200)]
        self.run_main([str(self.sine_wav(0.2)), "--no-realtime", "--timeline"], results)
        self.assertIn("[PARTIAL]", self.stdout)
        self.assertIn("speech_onset", self.stdout)


if __name__ == "__main__":
    unittest.main()
