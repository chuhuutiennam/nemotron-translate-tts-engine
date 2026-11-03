"""Rebuild the AudioChunk stream rtpgw sends from a captured RTP pcap (G.711 8 kHz -> 16 kHz PCM)."""

from __future__ import annotations

import heapq
import math
import socket
import struct
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from nemotron_testbed.client_audio import RTPGW_CHUNK_MS, SAMPLE_RATE

G711_RATE = 8_000
JITTER_BUFFER_MS = 60
PACKET_TIME_MS = 20
NON_AUDIO_PAYLOAD_TYPES = frozenset({13, 101})

JitterMode = Literal["sim", "off"]
ResamplerMode = Literal["sinc", "linear"]
RESAMPLER_MODES: tuple[str, ...] = ("sinc", "linear")
RTPGW_RESAMPLER: ResamplerMode = "sinc"  # rtpgw audio.resampler default

_PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1),
    b"\xa1\xb2\x3c\x4d": (">", 1),
}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
_LINK_ETHERNET, _LINK_RAW, _LINK_LINUX_SLL = 1, 101, 113
_PT_CODEC = {0: "PCMU", 8: "PCMA", 111: "opus"}


class PcapError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RtpPacket:
    recv_ns: int
    seq: int
    timestamp: int
    payload_type: int
    marker: bool
    payload: bytes


@dataclass(slots=True)
class RtpFlow:
    key: str
    dst_port: int
    ssrc: int
    codec: str  # PCMU / PCMA / opus, or L16 (16 kHz, declared by the SDP of a trace_pcap file)
    packets: list[RtpPacket] = field(default_factory=list)
    label: str = ""  # SDP a=label of the flow (trace_pcap: in / audx)

    @property
    def duration_s(self) -> float:
        return (self.packets[-1].recv_ns - self.packets[0].recv_ns) / 1e9


@dataclass(slots=True)
class JitterStats:
    received: int = 0
    released: int = 0
    dropped: int = 0
    lost: int = 0


@dataclass(slots=True)
class PcapStats:
    packets: int
    jitter: JitterStats
    non_audio_skipped: int


@dataclass(frozen=True, slots=True)
class PcapChunk:
    pcm: bytes
    rtp_timestamp: int
    rtp_clock_rate: int
    chunk_seq: int
    duration_ms: int
    end_of_stream: bool
    due_s: float


def _ipv4_offset(frame: bytes, linktype: int) -> int | None:
    if linktype == _LINK_RAW:
        return 0
    if linktype == _LINK_LINUX_SLL:
        return 16 if frame[14:16] == b"\x08\x00" else None
    offset = 12
    ethertype = int.from_bytes(frame[offset : offset + 2], "big")
    while ethertype in (0x8100, 0x88A8):
        offset += 4
        ethertype = int.from_bytes(frame[offset : offset + 2], "big")
    return offset + 2 if ethertype == 0x0800 else None


def _udp_datagram(frame: bytes, linktype: int) -> tuple[str, int, str, int, bytes] | None:
    offset = _ipv4_offset(frame, linktype)
    if offset is None:
        return None
    ip = frame[offset:]
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 17:
        return None
    header_len = (ip[0] & 0x0F) * 4
    total_len, _ident, flags_fragment = struct.unpack_from(">HHH", ip, 2)
    if header_len < 20 or total_len < header_len + 8 or flags_fragment & 0x3FFF:
        return None
    udp = ip[header_len : min(total_len, len(ip))]
    if len(udp) < 8:
        return None
    src_port, dst_port, udp_len = struct.unpack_from(">HHH", udp)
    if udp_len < 8:
        return None
    return socket.inet_ntoa(ip[12:16]), src_port, socket.inet_ntoa(ip[16:20]), dst_port, udp[8:udp_len]


def _parse_rtp(datagram: bytes) -> tuple[int, bool, int, int, int, bytes] | None:
    """(payload_type, marker, seq, timestamp, ssrc, payload) for an RTP v2 packet, else None."""
    if len(datagram) < 12 or datagram[0] >> 6 != 2:
        return None
    offset = 12 + 4 * (datagram[0] & 0x0F)
    if datagram[0] & 0x10:
        if len(datagram) < offset + 4:
            return None
        offset += 4 + 4 * int.from_bytes(datagram[offset + 2 : offset + 4], "big")
    end = len(datagram)
    if datagram[0] & 0x20:
        end -= datagram[-1]
    if end < offset:
        return None
    seq, timestamp, ssrc = struct.unpack_from(">HII", datagram, 2)
    return datagram[1] & 0x7F, bool(datagram[1] & 0x80), seq, timestamp, ssrc, datagram[offset:end]


def _parse_sdp(text: str) -> dict[tuple[str, int], tuple[dict[int, tuple[str, int]], str]]:
    """{(connection ip, media port): ({payload type: (encoding, clock rate)}, a=label)} for every m=audio line."""
    media: dict[tuple[str, int], tuple[dict[int, tuple[str, int]], str]] = {}
    session_ip = ""
    current: dict | None = None
    entries: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("c=IN IP4 "):
            ip = line.split()[-1].split("/")[0]
            if current is None:
                session_ip = ip
            else:
                current["ip"] = ip
        elif line.startswith("m=audio "):
            parts = line.split()
            current = {"ip": "", "port": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                       "rtpmap": {}, "label": ""}
            entries.append(current)
        elif current is not None and line.startswith("a=rtpmap:"):
            pt_text, _, encoding = line[len("a=rtpmap:") :].partition(" ")
            name, _, rest = encoding.partition("/")
            rate = rest.split("/")[0]
            if pt_text.isdigit() and rate.isdigit():
                current["rtpmap"][int(pt_text)] = (name.upper(), int(rate))
        elif current is not None and line.startswith("a=label:"):
            current["label"] = line[len("a=label:") :]
    for entry in entries:
        media[(entry["ip"] or session_ip, entry["port"])] = (entry["rtpmap"], entry["label"])
    return media


def read_pcap(path: str | Path) -> list[RtpFlow]:
    """Group RTP-over-UDP-over-IPv4 packets by 4-tuple, ordered by first packet; mirrors rtpgw ReadRTPStreams.
    An SDP found in the capture (SIP over UDP, e.g. the one a trace_pcap file starts with) names L16/16000 flows
    and their a=label."""
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise PcapError(f"{path}: not a pcap file (too short)")
    if data[:4] == _PCAPNG_MAGIC:
        raise PcapError(f"{path}: pcapng is not supported (rtpgw cannot read it either); convert: editcap -F pcap in out")
    if data[:4] not in _PCAP_MAGIC:
        raise PcapError(f"{path}: not a pcap file (bad magic {data[:4].hex()})")
    endian, ns_per_unit = _PCAP_MAGIC[data[:4]]
    linktype = struct.unpack_from(endian + "I", data, 20)[0]
    if linktype not in (_LINK_ETHERNET, _LINK_RAW, _LINK_LINUX_SLL):
        raise PcapError(f"{path}: unsupported link type {linktype} (need Ethernet, raw IP or Linux cooked)")

    record = struct.Struct(endian + "IIII")
    flows: dict[str, RtpFlow] = {}
    destinations: dict[str, tuple[str, int]] = {}
    sdp_media: dict[tuple[str, int], tuple[dict[int, tuple[str, int]], str]] = {}
    position = 24
    while position + record.size <= len(data):
        seconds, fraction, captured, _original = record.unpack_from(data, position)
        position += record.size
        frame = data[position : position + captured]
        position += captured
        if len(frame) < captured:
            break
        datagram = _udp_datagram(frame, linktype)
        if datagram is not None and b"\r\n\r\nv=0" in datagram[4] and datagram[4][:7] in (b"INVITE ", b"SIP/2.0"):
            sdp_media.update(_parse_sdp(datagram[4].split(b"\r\n\r\n", 1)[1].decode("utf-8", "replace")))
            continue
        rtp = _parse_rtp(datagram[4]) if datagram else None
        if datagram is None or rtp is None:
            continue
        src_ip, src_port, dst_ip, dst_port, _ = datagram
        payload_type, marker, seq, timestamp, ssrc, payload = rtp
        key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        flow = flows.get(key)
        if flow is None:
            flow = flows[key] = RtpFlow(key, dst_port, ssrc, _PT_CODEC.get(payload_type, "PCMU"))
            destinations[key] = (dst_ip, dst_port)
        flow.packets.append(
            RtpPacket(seconds * 1_000_000_000 + fraction * ns_per_unit, seq, timestamp, payload_type, marker, payload)
        )
    for key, flow in flows.items():
        described = sdp_media.get(destinations[key])
        if described is None:
            continue
        rtpmap, flow.label = described
        encoding = rtpmap.get(flow.packets[0].payload_type)
        if encoding is not None and encoding[0] == "L16":
            flow.codec = "L16" if encoding[1] == SAMPLE_RATE else f"L16/{encoding[1]}"
    return sorted(flows.values(), key=lambda flow: flow.packets[0].recv_ns)


def describe_flows(flows: list[RtpFlow]) -> list[str]:
    lines = []
    for index, flow in enumerate(flows):
        payload_types = dict(sorted(Counter(packet.payload_type for packet in flow.packets).items()))
        lines.append(
            f"#{index}  {flow.key}  {flow.codec}  packets={len(flow.packets)}  pt={payload_types}  "
            f"duration={flow.duration_s:.1f}s  ssrc=0x{flow.ssrc:08x}" + (f"  label={flow.label}" if flow.label else "")
        )
    return lines


def select_flow(flows: list[RtpFlow], selector: str | None = None) -> RtpFlow:
    """selector: None (only valid for a single flow), '#N' (index in the list), a destination port, the full key,
    or an SDP label (trace_pcap: in / audx)."""
    if not flows:
        raise PcapError("no RTP flows found in the pcap")
    listing = "\n".join(describe_flows(flows))
    if selector is None:
        if len(flows) == 1:
            return flows[0]
        raise PcapError(f"the pcap has {len(flows)} flows, choose one with --flow:\n{listing}")
    selector = selector.strip()
    if selector.startswith("#") and selector[1:].isdigit() and int(selector[1:]) < len(flows):
        return flows[int(selector[1:])]
    for flow in flows:
        if flow.key == selector:
            return flow
    if selector.isdigit():
        by_port = [flow for flow in flows if flow.dst_port == int(selector)]
        if len(by_port) == 1:
            return by_port[0]
        if len(by_port) > 1:
            raise PcapError(f"destination port {selector} matches {len(by_port)} flows, use the full key:\n{listing}")
    by_label = [flow for flow in flows if flow.label and flow.label == selector]
    if len(by_label) == 1:
        return by_label[0]
    raise PcapError(f"no flow matches {selector!r}; available flows:\n{listing}")


def _seq_diff(a: int, b: int) -> int:
    return ((a - b + 0x8000) & 0xFFFF) - 0x8000


class _JitterSim:
    """Deterministic replay of rtpgw internal/jitter/buffer.go; sequence numbers are unwrapped into one integer line."""

    def __init__(self, buffer_ns: int) -> None:
        self.buffer_ns = buffer_ns
        self.heap: list[tuple[int, int, RtpPacket]] = []
        self.out: list[tuple[RtpPacket, int]] = []
        self.stats = JitterStats()
        self.started = False
        self.first_ns = 0
        self.next_set = False
        self.next_expected = 0
        self.hold_until = 0
        self.last_processed: int | None = None
        self.reference: int | None = None
        self.arrivals = 0

    def push(self, packet: RtpPacket) -> None:
        if self.reference is None:
            extended = packet.seq
            self.reference = extended
        else:
            extended = self.reference + _seq_diff(packet.seq, self.reference & 0xFFFF)
            self.reference = max(self.reference, extended)
        if self.last_processed is not None and extended <= self.last_processed:
            self.stats.dropped += 1
            return
        heapq.heappush(self.heap, (extended, self.arrivals, packet))
        self.arrivals += 1
        self.stats.received += 1
        if not self.started:
            self.first_ns = packet.recv_ns
            self.started = True

    def flush(self, now_ns: int) -> None:
        if not self.started or not self.heap:
            return
        if not self.next_set:
            if now_ns < self.first_ns + self.buffer_ns:
                return
            self.next_expected = self.heap[0][0]
            self.next_set = True
            self.hold_until = now_ns + self.buffer_ns
        while self.heap:
            extended, _, packet = self.heap[0]
            diff = extended - self.next_expected
            if diff == 0:
                heapq.heappop(self.heap)
                self.out.append((packet, now_ns))
                self.stats.released += 1
                self.last_processed = self.next_expected
                self.next_expected += 1
                self.hold_until = now_ns + self.buffer_ns
            elif diff < 0:
                heapq.heappop(self.heap)
                self.stats.dropped += 1
            elif now_ns > self.hold_until:
                self.last_processed = self.next_expected
                self.stats.lost += 1
                self.next_expected += 1
                self.hold_until = now_ns + self.buffer_ns
            else:
                break


def simulate_jitter_buffer(
    packets: list[RtpPacket],
    buffer_ms: int = JITTER_BUFFER_MS,
    packet_time_ms: int = PACKET_TIME_MS,
) -> tuple[list[tuple[RtpPacket, int]], JitterStats]:
    """(packet, release_ns) in release order. The flush ticker is assumed to start at the first arrival; the
    max-late wall-clock drop is not modelled (capture timestamps carry no queueing delay)."""
    sim = _JitterSim(buffer_ms * 1_000_000)
    tick_ns = packet_time_ms * 1_000_000
    next_tick: int | None = None
    for packet in packets:
        if next_tick is None:
            next_tick = packet.recv_ns + tick_ns
        while next_tick < packet.recv_ns:
            if not sim.heap:
                next_tick += ((packet.recv_ns - next_tick - 1) // tick_ns + 1) * tick_ns
                break
            sim.flush(next_tick)
            next_tick += tick_ns
        sim.push(packet)
    while sim.heap and next_tick is not None:
        sim.flush(next_tick)
        next_tick += tick_ns
    return sim.out, sim.stats


def _decode_ulaw(wire: int) -> int:
    u = ~wire & 0xFF
    exponent = (u >> 4) & 0x07
    magnitude = (0, 132, 396, 924, 1980, 4092, 8316, 16764)[exponent] + ((u & 0x0F) << (exponent + 3))
    # ITU polarity (wire 0x00 -> -32124), as rtpgw g711.go since 2026-11; it decoded the inverted sign before.
    return -magnitude if u & 0x80 else magnitude


def _decode_alaw(wire: int) -> int:
    a = wire ^ 0x55
    exponent = (a >> 4) & 0x07
    mantissa = a & 0x0F
    magnitude = mantissa * 16 + 8 if exponent == 0 else (mantissa * 16 + 264) << (exponent - 1)
    return magnitude if a & 0x80 else -magnitude


_ULAW_TABLE = np.array([_decode_ulaw(wire) for wire in range(256)], dtype=np.int16)
_ALAW_TABLE = np.array([_decode_alaw(wire) for wire in range(256)], dtype=np.int16)


def decode_g711(payload: bytes, codec: str) -> np.ndarray:
    table = _ULAW_TABLE if codec == "PCMU" else _ALAW_TABLE
    return table[np.frombuffer(payload, dtype=np.uint8)]


def upsample_2x(pcm: np.ndarray) -> np.ndarray:
    """rtpgw linear 8 -> 16 kHz (audio.resampler: linear): x[i], then x[i] + trunc((x[i+1]-x[i])/2); the last
    sample of the packet is held twice; no state between packets."""
    count = pcm.size
    out = np.empty(2 * count, dtype=np.int16)
    out[0::2] = pcm
    if count > 1:
        wide = pcm.astype(np.int32)
        delta = wide[1:] - wide[:-1]
        half = np.where(delta >= 0, delta >> 1, -((-delta) >> 1))
        out[1 : 2 * count - 1 : 2] = wide[:-1] + half
    out[-1] = pcm[-1]
    return out


# rtpgw internal/audio/resampler.go, audio.resampler: sinc. Half-band FIR, 103 taps h[-51..51], Kaiser window
# (beta 7.86): the centre tap is 1 and the other even taps are 0, so the even output samples are the input delayed
# by HALF_BAND_DELAY samples and the odd ones convolve HALF_BAND_K inputs with the odd taps (symmetric).
_HALF_BAND_M = 51
_HALF_BAND_BETA = 7.86
_HALF_BAND_K = _HALF_BAND_M + 1
HALF_BAND_DELAY = _HALF_BAND_K // 2  # 26 input samples = 3.25 ms at 8 kHz


def _bessel_i0(x: float) -> float:
    s = t = 1.0
    for k in range(1, 50):
        t *= x / (2 * k)
        s += t * t
    return s


def _half_band_coef() -> np.ndarray:
    """First half of the odd taps (n = M, M-2, ..., 1), normalised so the odd taps sum to 1; same float64 operations
    in the same order as makeHalfBandCoef, so the taps are bit-identical (checked against the golden fixture)."""
    taps = []
    total = 0.0
    for j in range(_HALF_BAND_K // 2):
        n = float(_HALF_BAND_M - 2 * j)
        r = n / _HALF_BAND_M
        w = _bessel_i0(_HALF_BAND_BETA * math.sqrt(1 - r * r)) / _bessel_i0(_HALF_BAND_BETA)
        tap = 2 * math.sin(math.pi * n / 2) / (math.pi * n) * w
        taps.append(tap)
        total += 2 * tap
    return np.array([tap / total for tap in taps], dtype=np.float64)


HALF_BAND_COEF = _half_band_coef()


def _round_sat16(values: np.ndarray) -> np.ndarray:
    """Go roundSat16: round half away from zero, saturate to int16."""
    shifted = np.where(values >= 0, values + 0.5, values - 0.5)
    return np.where(shifted >= 32767, 32767, np.where(shifted <= -32768, -32768, np.trunc(shifted))).astype(np.int16)


class GatewayResampler:
    """rtpgw 8 -> 16 kHz resampler of one session. sinc keeps the last HALF_BAND_K-1 input samples across packets,
    so the output does not depend on how the input is split; linear is upsample_2x (stateless)."""

    def __init__(self, mode: ResamplerMode = RTPGW_RESAMPLER) -> None:
        if mode not in RESAMPLER_MODES:
            raise ValueError(f"unknown resampler {mode!r}, expected one of {RESAMPLER_MODES}")
        self.mode = mode
        self._history = np.zeros(_HALF_BAND_K - 1, dtype=np.float64)
        self._pending = False

    @property
    def delay_samples(self) -> int:
        """Delay in input samples (= 8 kHz RTP ticks); rtpgw subtracts it from the timestamp given to the chunker."""
        return HALF_BAND_DELAY if self.mode == "sinc" else 0

    def process(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.size == 0:
            return np.empty(0, dtype=np.int16)
        if self.mode == "linear":
            return upsample_2x(pcm)
        count = pcm.size
        buffer = np.concatenate((self._history, pcm.astype(np.float64)))
        windows = np.lib.stride_tricks.sliding_window_view(buffer, _HALF_BAND_K)[:count]
        acc = np.zeros(count, dtype=np.float64)
        for j, tap in enumerate(HALF_BAND_COEF):  # Go's summation order, one tap at a time
            acc += tap * (windows[:, j] + windows[:, _HALF_BAND_K - 1 - j])
        out = np.empty(2 * count, dtype=np.int16)
        out[0::2] = windows[:, HALF_BAND_DELAY - 1].astype(np.int16)
        out[1::2] = _round_sat16(acc)
        self._history = buffer[count:].copy()
        self._pending = True
        return out

    def flush(self) -> np.ndarray:
        """The 2 * delay_samples output samples still inside the filter (end of stream), then reset; empty for
        linear or when nothing was processed since the last flush."""
        if self.mode == "linear" or not self._pending:
            return np.empty(0, dtype=np.int16)
        out = self.process(np.zeros(HALF_BAND_DELAY, dtype=np.int16))
        self._history[:] = 0.0
        self._pending = False
        return out


class GatewayChunker:
    """Port of rtpgw internal/audio/chunker.go: fixed-size chunks stamped with the first contributing RTP timestamp."""

    def __init__(
        self,
        chunk_ms: int = RTPGW_CHUNK_MS,
        sample_rate: int = SAMPLE_RATE,
        rtp_clock_rate: int = G711_RATE,
    ) -> None:
        self.chunk_samples = max(1, sample_rate * chunk_ms // 1000)
        self.sample_rate = sample_rate
        self.rtp_clock_rate = rtp_clock_rate
        self._buffer = np.empty(0, dtype=np.int16)
        self._start_ts: int | None = None
        self._seq = 0

    def _ticks(self, samples: int) -> int:
        return samples * self.rtp_clock_rate // self.sample_rate if samples > 0 else 0

    def _emit(self, samples: np.ndarray, due_s: float) -> PcapChunk:
        chunk = PcapChunk(
            pcm=samples.astype("<i2").tobytes(),
            rtp_timestamp=self._start_ts or 0,
            rtp_clock_rate=self.rtp_clock_rate,
            chunk_seq=self._seq,
            duration_ms=samples.size * 1000 // self.sample_rate,
            end_of_stream=False,
            due_s=due_s,
        )
        self._seq += 1
        return chunk

    def push(self, pcm: np.ndarray, rtp_timestamp: int, due_s: float) -> list[PcapChunk]:
        if pcm.size == 0:
            return []
        if self._start_ts is None:
            self._start_ts = rtp_timestamp
        self._buffer = np.concatenate((self._buffer, pcm))
        chunks = []
        while self._buffer.size >= self.chunk_samples:
            chunks.append(self._emit(self._buffer[: self.chunk_samples], due_s))
            self._buffer = self._buffer[self.chunk_samples :]
            if self._buffer.size == 0:
                self._start_ts = None
            else:
                self._start_ts = (rtp_timestamp + self._ticks(pcm.size - self._buffer.size)) & 0xFFFFFFFF
        return chunks

    def flush(self, due_s: float) -> PcapChunk | None:
        if self._buffer.size == 0:
            return None
        chunk = self._emit(self._buffer, due_s)
        self._buffer = self._buffer[:0]
        self._start_ts = None
        return chunk


def build_chunks(
    flow: RtpFlow,
    *,
    jitter: JitterMode = "sim",
    chunk_ms: int = RTPGW_CHUNK_MS,
    jitter_buffer_ms: int = JITTER_BUFFER_MS,
    packet_time_ms: int = PACKET_TIME_MS,
    resampler: ResamplerMode = RTPGW_RESAMPLER,
) -> tuple[list[PcapChunk], PcapStats]:
    """Chunks as the gateway would emit them; the last one carries end_of_stream (no extra empty chunk).

    jitter="sim" runs the live path (jitter buffer, then per-packet pipeline); "off" feeds packets in file order
    like pcap_replay, except that CN/telephone-event packets are still skipped as in the live pipeline.
    resampler is rtpgw audio.resampler; like internal/pipeline/audio_pipeline.go, the chunker gets the packet
    timestamp minus the resampler delay, and at the end the resampler tail goes through the chunker before its
    remainder (pcap_replay does the same; the live gateway cancels the stream, so it never delivers that tail).
    due_s is when the gateway would have the chunk ready, in seconds after the flow's first packet.
    An L16 flow (trace_pcap file) is already the gateway's output: see trace_chunks; the other options do not apply.
    """
    if flow.codec == "L16":
        return trace_chunks(flow)
    if flow.codec not in ("PCMU", "PCMA"):
        raise PcapError(f"flow {flow.key} carries {flow.codec}; only G.711 (PCMU/PCMA) is supported")
    if jitter == "sim":
        released, jitter_stats = simulate_jitter_buffer(flow.packets, jitter_buffer_ms, packet_time_ms)
    elif jitter == "off":
        released = [(packet, packet.recv_ns) for packet in flow.packets]
        jitter_stats = JitterStats(len(released), len(released))
    else:
        raise ValueError(f"unknown jitter mode {jitter!r}")

    origin_ns = flow.packets[0].recv_ns
    chunker = GatewayChunker(chunk_ms)
    upsampler = GatewayResampler(resampler)
    delay = upsampler.delay_samples
    codec = flow.codec
    chunks: list[PcapChunk] = []
    skipped = 0
    last_due = 0.0
    next_timestamp = 0
    for packet, at_ns in released:
        if packet.payload_type in NON_AUDIO_PAYLOAD_TYPES:
            skipped += 1
            continue
        if packet.payload_type == 0:
            codec = "PCMU"
        elif packet.payload_type == 8:
            codec = "PCMA"
        pcm = decode_g711(packet.payload, codec)
        if pcm.size == 0:
            continue
        last_due = (at_ns - origin_ns) / 1e9
        chunks.extend(chunker.push(upsampler.process(pcm), (packet.timestamp - delay) & 0xFFFFFFFF, last_due))
        next_timestamp = (packet.timestamp + pcm.size) & 0xFFFFFFFF
    filter_tail = upsampler.flush()
    if filter_tail.size:
        chunks.extend(chunker.push(filter_tail, (next_timestamp - delay) & 0xFFFFFFFF, last_due))
    tail = chunker.flush(last_due)
    if tail is not None:
        chunks.append(replace(tail, end_of_stream=True))
    elif chunks:
        chunks[-1] = replace(chunks[-1], end_of_stream=True)
    return chunks, PcapStats(len(flow.packets), jitter_stats, skipped)


def trace_chunks(flow: RtpFlow) -> tuple[list[PcapChunk], PcapStats]:
    """Chunks of a trace_pcap flow (L16 16 kHz, docs/trace-pcap-design.md): each run of packets started by a marker
    is one chunk the engine received ("in") or one piece AudX returned ("audx"). The 16 kHz RTP timestamps go back
    to the gateway's 8 kHz clock; due_s comes from the capture times (when the engine got the chunk)."""
    if not flow.packets:
        return [], PcapStats(0, JitterStats(), 0)
    origin_ns = flow.packets[0].recv_ns
    groups: list[list[RtpPacket]] = []
    for packet in flow.packets:
        if packet.marker or not groups:
            groups.append([packet])
        else:
            groups[-1].append(packet)
    chunks = []
    for index, group in enumerate(groups):
        pcm = np.concatenate(
            [np.frombuffer(p.payload[: len(p.payload) // 2 * 2], dtype=">i2") for p in group]
        ).astype("<i2")
        ts16 = group[0].timestamp - (1 << 32 if group[0].timestamp & 0x80000000 else 0)  # < 0: before the first chunk
        chunks.append(
            PcapChunk(
                pcm=pcm.tobytes(),
                rtp_timestamp=(ts16 // 2) & 0xFFFFFFFF,
                rtp_clock_rate=G711_RATE,
                chunk_seq=index,
                duration_ms=pcm.size * 1000 // SAMPLE_RATE,
                end_of_stream=index == len(groups) - 1,
                due_s=(group[0].recv_ns - origin_ns) / 1e9,
            )
        )
    count = len(flow.packets)
    return chunks, PcapStats(count, JitterStats(received=count, released=count), 0)
