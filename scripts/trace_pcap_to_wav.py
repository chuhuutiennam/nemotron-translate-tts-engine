#!/usr/bin/env python3
"""Split an engine trace_pcap file (docs/trace-pcap-design.md) into one 16 kHz WAV per flow, lined up.

  PYTHONPATH=src python scripts/trace_pcap_to_wav.py logs/pcap/20261102-101500_call_MO_ab12cd34.pcap --out /tmp/trace

writes <name>_in.wav and <name>_audx.wav: every sample sits at its RTP timestamp counted from the smallest timestamp
in the file, gaps are zeros and all files have the same length, so they can be compared side by side (Audacity).
A packet whose timestamps are already filled (a chunk the engine dropped as out of order) is not written again.
"""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nemotron_testbed.client_pcap import PcapError, RtpFlow, read_pcap

SAMPLE_RATE = 16_000


@dataclass(slots=True)
class FlowAudio:
    name: str
    samples: np.ndarray
    seconds: float
    gaps: int
    gap_seconds: float
    duplicate_packets: int


def _offset(timestamp: int, reference: int) -> int:
    """Signed distance in samples (16 kHz clock) from reference, wrap-around aware."""
    return ((timestamp - reference + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def flow_audio(flows: list[RtpFlow]) -> list[FlowAudio]:
    traced = [flow for flow in flows if flow.codec == "L16" and flow.packets]
    if not traced:
        raise PcapError("no L16 flow in the pcap: not a trace_pcap file")
    reference = traced[0].packets[0].timestamp
    origin = min(_offset(p.timestamp, reference) for flow in traced for p in flow.packets)
    length = max(
        _offset(p.timestamp, reference) - origin + len(p.payload) // 2 for flow in traced for p in flow.packets
    )
    result = []
    for flow in traced:
        samples = np.zeros(length, dtype=np.int16)
        filled = np.zeros(length, dtype=bool)
        duplicates = 0
        for packet in flow.packets:
            pcm = np.frombuffer(packet.payload[: len(packet.payload) // 2 * 2], dtype=">i2")
            start = _offset(packet.timestamp, reference) - origin
            if filled[start : start + pcm.size].any():
                duplicates += 1
                continue
            samples[start : start + pcm.size] = pcm
            filled[start : start + pcm.size] = True
        covered = np.flatnonzero(filled)
        gaps = 0
        gap_samples = 0
        if covered.size:
            inner = ~filled[covered[0] : covered[-1] + 1]
            gap_samples = int(inner.sum())
            gaps = int(np.count_nonzero(inner[1:] & ~inner[:-1]))  # inner[0] is a filled sample, never a gap
        name = flow.label or str(flow.dst_port)
        result.append(
            FlowAudio(name, samples, int(filled.sum()) / SAMPLE_RATE, gaps, gap_samples / SAMPLE_RATE, duplicates)
        )
    return result


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.astype("<i2").tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="trace_pcap file -> one lined-up 16 kHz WAV per flow (in / audx)")
    parser.add_argument("pcap", type=Path, nargs="+", help="trace_pcap file(s) written by the engine")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: next to each pcap)")
    args = parser.parse_args(argv)
    status = 0
    for path in args.pcap:
        try:
            flows = flow_audio(read_pcap(path))
        except (OSError, PcapError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            status = 1
            continue
        out_dir = args.out or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{path.name}: {flows[0].samples.size / SAMPLE_RATE:.2f} s")
        for flow in flows:
            target = out_dir / f"{path.stem}_{flow.name}.wav"
            write_wav(target, flow.samples)
            print(
                f"  {flow.name}: audio {flow.seconds:.2f} s, gaps {flow.gaps} ({flow.gap_seconds:.2f} s), "
                f"duplicate packets {flow.duplicate_packets} -> {target}"
            )
    return status


if __name__ == "__main__":
    sys.exit(main())
