"""Trace pcap: per-stream pcap of the audio received in AudioChunk ("in") and of the audio after AudX ("audx").

Design: docs/trace-pcap-design.md. One file per stream: a synthetic SIP INVITE whose SDP declares the flows as
L16/16000 (so Wireshark decodes them), then RTP payload type 96, L16 big-endian, 20 ms packets. The "audx" flow
carries the RTP timestamps and capture times of the matching "in" content, so both flows line up despite the AudX
delay. The event loop only converts and queues packets; one writer thread builds the frames and writes the files.
Any failure stops the trace of that stream (or of every stream when the disk is low); the call is never affected.
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

from .audio import float32_to_pcm16
from .config import TracePcapSettings
from .metrics import TRACE_PCAP_BYTES, TRACE_PCAP_STREAMS, TRACE_PCAP_STREAMS_ACTIVE

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
PACKET_SAMPLES = SAMPLE_RATE * 20 // 1000
PAYLOAD_TYPE = 96
GATEWAY_IP = "10.0.0.1"
ENGINE_IP = "10.0.0.2"
SIP_PORT = 5060
FLOW_IN = "in"
FLOW_AUDX = "audx"
FLOW_PORTS = {FLOW_IN: 40000, FLOW_AUDX: 40002}
FLOW_SSRC = {FLOW_IN: 0x00000001, FLOW_AUDX: 0x00000002}
QUEUE_MAX = 4096
FLUSH_INTERVAL_S = 1.0
DISK_CHECK_INTERVAL_S = 5.0
_MASK32 = 0xFFFFFFFF
_STOP = object()


# ── pcap / packet building ────────────────────────────────────────────────────

PCAP_FILE_HEADER = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)  # microsecond pcap, Ethernet


def udp_frame(src_ip: str, src_port: int, dst_ip: str, dst_port: int, payload: bytes) -> bytes:
    """Ethernet / IPv4 / UDP frame with zero checksums (Wireshark accepts them)."""
    udp = struct.pack(">HHHH", src_port, dst_port, 8 + len(payload), 0) + payload
    ip = struct.pack(
        ">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, socket.inet_aton(src_ip), socket.inet_aton(dst_ip)
    )
    return b"\x00" * 12 + b"\x08\x00" + ip + udp


def pcap_record(time_s: float, frame: bytes) -> bytes:
    micros = int(round(time_s * 1_000_000))
    seconds, fraction = divmod(max(0, micros), 1_000_000)
    return struct.pack("<IIII", seconds & _MASK32, fraction, len(frame), len(frame)) + frame


def rtp_packet(flow: str, seq: int, timestamp: int, marker: bool, pcm: np.ndarray) -> bytes:
    header = struct.pack(
        ">BBHII", 0x80, (0x80 if marker else 0) | PAYLOAD_TYPE, seq & 0xFFFF, timestamp & _MASK32, FLOW_SSRC[flow]
    )
    return header + pcm.astype(">i2").tobytes()  # L16 is network byte order (RFC 3551)


def _sdp_value(value: object) -> str:
    return re.sub(r"[^\x21-\x7e]", "_", str(value)) or "-"


def sdp_body(info: dict[str, object], flows: list[str]) -> str:
    lines = [
        "v=0",
        f"o=nemotron 0 0 IN IP4 {ENGINE_IP}",
        "s=nemotron trace pcap",
        "i=" + " ".join(f"{key}={_sdp_value(value)}" for key, value in info.items()),
        f"c=IN IP4 {ENGINE_IP}",
        "t=0 0",
    ]
    for flow in flows:
        lines += [
            f"m=audio {FLOW_PORTS[flow]} RTP/AVP {PAYLOAD_TYPE}",
            f"a=rtpmap:{PAYLOAD_TYPE} L16/{SAMPLE_RATE}",
            f"a=label:{flow}",
            "a=recvonly",
        ]
    return "\r\n".join(lines) + "\r\n"


def sip_invite(call_id: str, sdp: str) -> bytes:
    head = (
        f"INVITE sip:nemotron-trace@{ENGINE_IP} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {GATEWAY_IP}:{SIP_PORT};branch=z9hG4bK-trace\r\n"
        f"From: <sip:rtpgw@{GATEWAY_IP}>;tag=trace\r\n"
        f"To: <sip:nemotron@{ENGINE_IP}>\r\n"
        f"Call-ID: {_sdp_value(call_id)}\r\n"
        "CSeq: 1 INVITE\r\n"
        f"Contact: <sip:rtpgw@{GATEWAY_IP}>\r\n"
        "Content-Type: application/sdp\r\n"
        f"Content-Length: {len(sdp.encode())}\r\n\r\n"
    )
    return (head + sdp).encode()


def _safe_name(value: str, limit: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:limit] or "_"


def _disk_free_mb(directory: Path) -> float:
    path = directory
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free / (1024 * 1024)


# ── shared per-file state ─────────────────────────────────────────────────────


@dataclass(eq=False)
class _FileState:
    path: Path
    header: bytes
    log_extra: dict[str, str]
    stopped_reason: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    # writer thread only
    fh: BinaryIO | None = None
    closed: bool = False
    seq: dict[str, int] = field(default_factory=lambda: {FLOW_IN: 0, FLOW_AUDX: 0})
    packets: dict[str, int] = field(default_factory=lambda: {FLOW_IN: 0, FLOW_AUDX: 0})
    bytes_written: int = 0

    def try_stop(self, reason: str) -> bool:
        """Set the stop reason once; True for the caller that set it."""
        with self.lock:
            if self.stopped_reason is not None:
                return False
            self.stopped_reason = reason
            return True


def _count_stop(state: _FileState, reason: str) -> None:
    TRACE_PCAP_STREAMS.labels(outcome=f"stopped_{reason.lower()}").inc()
    logger.warning(
        "TRACE_PCAP_STOPPED",
        extra={**state.log_extra, "reason": reason, "path": str(state.path), "stage": "TRACE_PCAP_STOPPED"},
    )


@dataclass(slots=True)
class _Segment:
    """Audio that entered AudX: positions [start, start + samples) carry RTP timestamps ts.. (16 kHz clock)."""

    start: int
    samples: int
    ts: int
    arrival: float

    @property
    def end(self) -> int:
        return self.start + self.samples


# ── per-stream handle (event loop side) ───────────────────────────────────────


class StreamTrace:
    """Trace of one stream. Called from the event loop; every method swallows its own errors."""

    def __init__(
        self,
        tracer: PcapTracer,
        state: _FileState,
        *,
        rtp_clock_rate: int,
        first_rtp_timestamp: int,
        audx_delay_samples: int | None,
        preproc_stats: dict | None,
        max_samples: int,
    ) -> None:
        self._tracer = tracer
        self._state = state
        self._clock = rtp_clock_rate
        self._first = int(first_rtp_timestamp) & _MASK32
        self.delay = audx_delay_samples  # None: the stream does not run AudX, no "audx" flow
        self._preproc_stats = preproc_stats
        self._max_samples = max_samples
        self._segments: deque[_Segment] = deque()
        self._proc_total = 0
        self._in_samples = 0
        self._audx_samples = 0
        self._closed = False

    @property
    def writing(self) -> bool:
        return not self._closed and self._state.stopped_reason is None

    def _ts16(self, rtp_timestamp: int) -> int:
        delta = ((int(rtp_timestamp) - self._first + 0x80000000) & _MASK32) - 0x80000000
        return (delta * SAMPLE_RATE // self._clock) & _MASK32

    def write_in(self, audio: np.ndarray, rtp_timestamp: int, arrival: float) -> None:
        """The chunk as the gateway sent it (16 kHz mono), before the RTP policy."""
        if not self.writing or audio.size == 0:
            return
        try:
            pcm = float32_to_pcm16(audio)
            ts = self._ts16(rtp_timestamp)
            packets = [
                ((ts + offset) & _MASK32, arrival, offset == 0, pcm[offset : offset + PACKET_SAMPLES])
                for offset in range(0, pcm.size, PACKET_SAMPLES)
            ]
            self._send(FLOW_IN, packets)
            self._in_samples += pcm.size
            if self._max_samples and self._in_samples >= self._max_samples:
                self._stop("MAX_SECONDS")
        except Exception:
            self._failed()

    def note_processed(self, rtp_timestamp: int, samples: int, arrival: float) -> None:
        """Audio the RTP policy passed on to AudX (silence for a gap included), starting at rtp_timestamp."""
        if self.delay is None or samples <= 0 or not self.writing:
            return
        self._segments.append(_Segment(self._proc_total, samples, self._ts16(rtp_timestamp), arrival))
        self._proc_total += samples

    def write_audx(self, audio: np.ndarray, lag: int) -> None:
        """Audio going to VAD/ASR; its content ends `lag` samples before the end of what entered AudX."""
        if self.delay is None or not self.writing or audio.size == 0:
            return
        try:
            end = self._proc_total - lag
            start = end - audio.size
            if end <= 0:
                return
            if start < 0:  # AudX's leading padding has no content
                audio = audio[-start:]
                start = 0
            pcm = float32_to_pcm16(audio)
            while self._segments and self._segments[0].end <= start:
                self._segments.popleft()
            packets = []
            for seg in self._segments:
                if seg.start >= end:
                    break
                pos = max(start, seg.start)
                stop = min(end, seg.end)
                while pos < stop:  # cut on the 20 ms grid of the chunk, like the "in" packets
                    cut = min(stop, seg.start + ((pos - seg.start) // PACKET_SAMPLES + 1) * PACKET_SAMPLES)
                    ts = (seg.ts + pos - seg.start) & _MASK32
                    packets.append((ts, seg.arrival, not packets, pcm[pos - start : cut - start]))
                    pos = cut
            if packets:
                self._send(FLOW_AUDX, packets)
                self._audx_samples += sum(p[3].size for p in packets)
        except Exception:
            self._failed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        summary = {
            "seconds_in": round(self._in_samples / SAMPLE_RATE, 2),
            "seconds_audx": round(self._audx_samples / SAMPLE_RATE, 2) if self.delay is not None else None,
            "audx_stats": dict(self._preproc_stats) if self._preproc_stats is not None else None,
        }
        self._tracer._stream_closed(self, self._state, summary)

    # internals

    def _send(self, flow: str, packets: list) -> None:
        if not self._tracer._put(("pkt", self._state, flow, packets)):
            self._stop("QUEUE_FULL")

    def _stop(self, reason: str) -> None:
        if self._state.try_stop(reason):
            _count_stop(self._state, reason)
            self._tracer._request_file_close(self._state)

    def _failed(self) -> None:
        logger.exception("TRACE_PCAP_ERROR", extra={**self._state.log_extra, "stage": "TRACE_PCAP_ERROR"})
        self._stop("IO_ERROR")


# ── process-wide tracer ───────────────────────────────────────────────────────


class PcapTracer:
    """Owns the writer thread; open_stream() is called by the coordinator for every new stream."""

    def __init__(
        self,
        settings: TracePcapSettings,
        *,
        queue_max: int = QUEUE_MAX,
        disk_free_mb: Callable[[Path], float] = _disk_free_mb,
    ) -> None:
        self.settings = settings
        self.directory = Path(settings.dir)
        self._disk_free_mb = disk_free_mb
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._control: deque = deque()  # file closes that did not fit in the queue
        self._active: dict[int, StreamTrace] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="trace-pcap", daemon=True)
        self._thread.start()
        logger.info(
            "TRACE_PCAP_ENABLED",
            extra={
                "path": str(self.directory),
                "max_streams": settings.max_streams,
                "max_seconds": settings.max_seconds,
                "min_free_mb": settings.min_free_mb,
                "stage": "TRACE_PCAP_ENABLED",
            },
        )

    # event loop side

    def open_stream(
        self,
        *,
        rpc_id: str,
        session_id: str,
        stream_id: str,
        task: str,
        rtp_clock_rate: int,
        first_rtp_timestamp: int,
        audx_delay_samples: int | None,
        preproc_stats: dict | None = None,
    ) -> StreamTrace | None:
        if self._closed:
            return None
        log_extra = {"rpc_id": rpc_id, "session_id": session_id, "stream_id": stream_id}
        try:
            limit = self.settings.max_streams
            if limit > 0 and sum(1 for trace in self._active.values() if trace.writing) >= limit:
                return self._skip("MAX_STREAMS", log_extra)
            if self._disk_low():
                return self._skip("DISK_LOW", log_extra)
            now = time.time()
            name = "_".join(
                (
                    time.strftime("%Y%m%d-%H%M%S", time.localtime(now)),
                    _safe_name(session_id),
                    _safe_name(stream_id, 40),
                    _safe_name(rpc_id, 8),
                )
            )
            path = self.directory / f"{name}.pcap"
            flows = [FLOW_IN] if audx_delay_samples is None else [FLOW_IN, FLOW_AUDX]
            info = {
                "session": session_id,
                "stream": stream_id,
                "rpc": rpc_id,
                "task": task,
                "audx": "off" if audx_delay_samples is None else "on",
                "audx_delay_ms": 0 if audx_delay_samples is None else audx_delay_samples * 1000 // SAMPLE_RATE,
                "rtp_clock_rate": rtp_clock_rate,
                "rtp_start": int(first_rtp_timestamp) & _MASK32,
            }
            invite = sip_invite(name, sdp_body(info, flows))
            header = PCAP_FILE_HEADER + pcap_record(now, udp_frame(GATEWAY_IP, SIP_PORT, ENGINE_IP, SIP_PORT, invite))
            state = _FileState(path=path, header=header, log_extra=log_extra)
            if not self._put(("open", state)):
                return self._skip("QUEUE_FULL", log_extra)
            trace = StreamTrace(
                self,
                state,
                rtp_clock_rate=rtp_clock_rate,
                first_rtp_timestamp=first_rtp_timestamp,
                audx_delay_samples=audx_delay_samples,
                preproc_stats=preproc_stats,
                max_samples=self.settings.max_seconds * SAMPLE_RATE,
            )
            self._active[id(trace)] = trace
            TRACE_PCAP_STREAMS.labels(outcome="traced").inc()
            logger.info(
                "TRACE_PCAP_OPEN",
                extra={**log_extra, "path": str(path), "flows": ",".join(flows), "stage": "TRACE_PCAP_OPEN"},
            )
            return trace
        except Exception:
            logger.exception("TRACE_PCAP_ERROR", extra={**log_extra, "stage": "TRACE_PCAP_ERROR"})
            return None

    def close(self, timeout_s: float = 5.0) -> None:
        """Close every trace and stop the writer thread (engine shutdown)."""
        if self._closed:
            return
        for trace in list(self._active.values()):
            trace.close()
        self._closed = True
        try:
            self._queue.put(_STOP, timeout=timeout_s)
        except queue.Full:
            logger.error("TRACE_PCAP_ERROR: writer queue full at shutdown", extra={"stage": "TRACE_PCAP_ERROR"})
        self._thread.join(timeout_s)

    def _skip(self, reason: str, log_extra: dict[str, str]) -> None:
        TRACE_PCAP_STREAMS.labels(outcome=f"skipped_{reason.lower()}").inc()
        logger.warning("TRACE_PCAP_SKIPPED", extra={**log_extra, "reason": reason, "stage": "TRACE_PCAP_SKIPPED"})
        return None

    def _disk_low(self) -> bool:
        if self.settings.min_free_mb <= 0:
            return False
        return self._disk_free_mb(self.directory) < self.settings.min_free_mb

    def _put(self, item: tuple) -> bool:
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _request_file_close(self, state: _FileState) -> None:
        if not self._put(("stop", state)):
            self._control.append(state)

    def _stream_closed(self, trace: StreamTrace, state: _FileState, summary: dict) -> None:
        self._active.pop(id(trace), None)
        if not self._put(("close", state, summary)):
            self._control.append(state)

    # writer thread

    def _run(self) -> None:
        open_files: set[_FileState] = set()
        last_flush = last_disk = time.monotonic()
        while True:
            while self._control:
                self._close_file(self._control.popleft(), open_files)
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            if item is not None:
                try:
                    self._handle(item, open_files)
                except Exception:
                    state = item[1]
                    logger.exception("TRACE_PCAP_ERROR", extra={**state.log_extra, "stage": "TRACE_PCAP_ERROR"})
                    if state.try_stop("IO_ERROR"):
                        _count_stop(state, "IO_ERROR")
                    self._close_file(state, open_files)
            now = time.monotonic()
            if now - last_flush >= FLUSH_INTERVAL_S:
                last_flush = now
                for state in list(open_files):
                    try:
                        state.fh.flush()  # type: ignore[union-attr]
                    except OSError:
                        if state.try_stop("IO_ERROR"):
                            _count_stop(state, "IO_ERROR")
                        self._close_file(state, open_files)
            if open_files and now - last_disk >= DISK_CHECK_INTERVAL_S:
                last_disk = now
                try:
                    low = self._disk_low()
                except OSError:
                    low = False
                if low:
                    for state in list(open_files):
                        if state.try_stop("DISK_LOW"):
                            _count_stop(state, "DISK_LOW")
                        self._close_file(state, open_files)
        while self._control:
            self._close_file(self._control.popleft(), open_files)
        for state in list(open_files):
            self._close_file(state, open_files)

    def _handle(self, item: tuple, open_files: set[_FileState]) -> None:
        kind, state = item[0], item[1]
        if kind == "open":
            if state.closed:  # a close from _control overtook this "open" in the queue
                return
            state.path.parent.mkdir(parents=True, exist_ok=True)
            state.fh = open(state.path, "wb")
            open_files.add(state)
            TRACE_PCAP_STREAMS_ACTIVE.inc()
            self._write(state, state.header)
        elif kind == "pkt":
            if state.fh is None:  # packets queued before a stop are still written; the "stop" item closes the file
                return
            flow, packets = item[2], item[3]
            records = []
            for timestamp, time_s, marker, pcm in packets:
                payload = rtp_packet(flow, state.seq[flow], timestamp, marker, pcm)
                state.seq[flow] += 1
                frame = udp_frame(GATEWAY_IP, FLOW_PORTS[flow], ENGINE_IP, FLOW_PORTS[flow], payload)
                records.append(pcap_record(time_s, frame))
            self._write(state, b"".join(records))
            state.packets[flow] += len(packets)
        elif kind == "stop":
            self._close_file(state, open_files)
        elif kind == "close":
            self._close_file(state, open_files)
            summary = item[2]
            logger.info(
                "TRACE_PCAP_CLOSE",
                extra={
                    **state.log_extra,
                    "path": str(state.path),
                    "packets_in": state.packets[FLOW_IN],
                    "packets_audx": state.packets[FLOW_AUDX],
                    "bytes": state.bytes_written,
                    "stopped_reason": state.stopped_reason or "",
                    **summary,
                    "stage": "TRACE_PCAP_CLOSE",
                },
            )

    def _write(self, state: _FileState, data: bytes) -> None:
        state.fh.write(data)  # type: ignore[union-attr]
        state.bytes_written += len(data)
        TRACE_PCAP_BYTES.inc(len(data))

    @staticmethod
    def _close_file(state: _FileState, open_files: set[_FileState]) -> None:
        open_files.discard(state)
        state.closed = True
        fh, state.fh = state.fh, None
        if fh is not None:
            TRACE_PCAP_STREAMS_ACTIVE.dec()
            try:
                fh.close()
            except OSError:
                logger.warning("TRACE_PCAP_ERROR: close failed", extra={**state.log_extra, "path": str(state.path)})
