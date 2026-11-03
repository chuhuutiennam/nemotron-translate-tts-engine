#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import grpc

from nemotron_testbed.client_audio import (
    RTPGW_CHUNK_MS,
    RTPGW_CLOCK_RATE,
    RtpTimeline,
    chunk_ms_warning,
    parse_rtp_start,
    pcm16_bytes,
    read_wav,
    to_narrowband,
)
from nemotron_testbed.pb import speech_pb2, speech_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send MO and MT WAV files interleaved in one Recognize RPC"
    )
    parser.add_argument("mo_wav", type=Path)
    parser.add_argument("mt_wav", type=Path)
    parser.add_argument("--address", default="127.0.0.1:50052")
    parser.add_argument("--session-id", default=f"dual-{uuid.uuid4().hex[:8]}")
    parser.add_argument(
        "--profile",
        choices=("default", "rtpgw"),
        default="default",
        help="rtpgw = --chunk-ms 80 --rtp-clock-rate 8000 --narrowband; flags given explicitly still win",
    )
    parser.add_argument("--chunk-ms", type=int, default=None, help="default 20 (profile rtpgw: 80)")
    parser.add_argument(
        "--rtp-clock-rate",
        type=int,
        default=None,
        help="rtp_clock_rate sent with every chunk (default 16000; profile rtpgw: 8000)",
    )
    parser.add_argument(
        "--narrowband",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass each side through 8 kHz G.711 PCMU and back to 16 kHz (needs soxr; profile rtpgw: on)",
    )
    parser.add_argument(
        "--rtp-start",
        default="random",
        help="First rtp_timestamp of each stream: an integer (same for both) or 'random' (independent per stream)",
    )
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument(
        "--mo-language",
        default="vi",
        help="transcribe: MO source language; translate: TARGET language of the MO stream (e.g. en)",
    )
    parser.add_argument(
        "--mt-language",
        default="en",
        help="transcribe: MT source language; translate: TARGET language of the MT stream (e.g. vi)",
    )
    parser.add_argument(
        "--mo-language-source",
        default="",
        help="SOURCE language hint (vi|en) of the MO stream (AudioChunk.language_source); empty = not sent",
    )
    parser.add_argument(
        "--mt-language-source",
        default="",
        help="SOURCE language hint (vi|en) of the MT stream (AudioChunk.language_source); empty = not sent",
    )
    args = parser.parse_args()
    rtpgw = args.profile == "rtpgw"
    if args.chunk_ms is None:
        args.chunk_ms = RTPGW_CHUNK_MS if rtpgw else 20
    if args.rtp_clock_rate is None:
        args.rtp_clock_rate = RTPGW_CLOCK_RATE if rtpgw else 16_000
    if args.narrowband is None:
        args.narrowband = rtpgw
    warning = chunk_ms_warning(args.chunk_ms)
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)

    def load(path: Path):
        audio = read_wav(path)
        return to_narrowband(audio) if args.narrowband else audio

    streams = {
        "MO": (load(args.mo_wav), args.mo_language, args.mo_language_source),
        "MT": (load(args.mt_wav), args.mt_language, args.mt_language_source),
    }
    sample_rate = 16_000
    chunk_samples = sample_rate * args.chunk_ms // 1000

    def requests():
        positions = {name: 0 for name in streams}
        timelines = {
            name: RtpTimeline(sample_rate, args.rtp_clock_rate, parse_rtp_start(args.rtp_start))
            for name in streams
        }
        print(
            f"chunk_ms={args.chunk_ms} rtp_clock_rate={args.rtp_clock_rate} narrowband={args.narrowband} "
            + " ".join(f"{name}.rtp_start={t.start}" for name, t in timelines.items()),
            flush=True,
        )
        chunk_seqs = {name: 0 for name in streams}
        active = set(streams)

        while active:
            cycle_started = time.perf_counter()
            for stream_id in list(active):
                audio, language, language_source = streams[stream_id]
                start = positions[stream_id]
                chunk = audio[start : start + chunk_samples]
                if chunk.size == 0:
                    active.remove(stream_id)
                    continue
                end = start + chunk.size
                is_last = end >= audio.size
                yield speech_pb2.AudioChunk(
                    session_id=args.session_id,
                    stream_id=stream_id,
                    pcm=pcm16_bytes(chunk),
                    sample_rate=sample_rate,
                    channels=1,
                    rtp_timestamp=timelines[stream_id].next_timestamp(chunk.size),
                    rtp_clock_rate=args.rtp_clock_rate,
                    chunk_seq=chunk_seqs[stream_id],
                    duration_ms=round(chunk.size * 1000 / sample_rate),
                    end_of_stream=is_last,
                    language=language,
                    language_source=language_source,
                    task=args.task,
                )
                positions[stream_id] = end
                chunk_seqs[stream_id] += 1
                if is_last:
                    active.remove(stream_id)

            elapsed = time.perf_counter() - cycle_started
            time.sleep(max(0.0, args.chunk_ms / 1000.0 - elapsed))

    channel = grpc.insecure_channel(args.address)
    stub = speech_pb2_grpc.SpeechStreamStub(channel)
    try:
        for result in stub.Recognize(requests()):
            kind = "FINAL" if result.is_final else "PARTIAL"
            audio_info = (
                f" audio_bytes={len(result.audio_payload)} pt={result.audio_pt}"
                if result.audio_payload
                else ""
            )
            print(
                f"[{kind}] seq={result.seq} {result.stream_id} "
                f"lang={result.language}{audio_info}: {result.text}",
                flush=True,
            )
    finally:
        channel.close()


if __name__ == "__main__":
    main()
