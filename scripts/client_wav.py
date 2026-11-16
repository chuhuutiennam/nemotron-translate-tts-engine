#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import grpc
import numpy as np

from nemotron_testbed.client_audio import (  # noqa: F401  (re-exported for tests)
    RTPGW_CHUNK_MS,
    RTPGW_CLOCK_RATE,
    SAMPLE_RATE,
    RtpTimeline,
    chunk_ms_warning,
    parse_rtp_start,
    pcm16_bytes,
    read_wav,
    rtp_ticks,
    to_narrowband,
)
from nemotron_testbed.client_pcap import (
    RESAMPLER_MODES,
    RTPGW_RESAMPLER,
    PcapChunk,
    PcapError,
    build_chunks,
    describe_flows,
    read_pcap,
    select_flow,
)
from nemotron_testbed.pb import speech_pb2, speech_pb2_grpc


def first_speech_seconds(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    rms_threshold: float = 0.01,
    min_frames: int = 2,
) -> float | None:
    """Audio position of the first energy-based speech (RMS >= -40 dBFS for min_frames frames); a rough marker, not VAD."""
    frame = max(1, sample_rate * frame_ms // 1000)
    run = 0
    for index in range(audio.size // frame):
        rms = float(np.sqrt(np.mean(np.square(audio[index * frame : (index + 1) * frame]))))
        run = run + 1 if rms >= rms_threshold else 0
        if run >= min_frames:
            return (index - min_frames + 1) * frame / sample_rate
    return None


def wav_chunks(audio: np.ndarray, chunk_ms: int, rtp_clock_rate: int, rtp_start: int) -> list[PcapChunk]:
    """Split a 16 kHz float32 signal into the chunks sent to the engine; due_s is the chunk's start position."""
    chunk_samples = max(1, SAMPLE_RATE * chunk_ms // 1000)
    rtp = RtpTimeline(SAMPLE_RATE, rtp_clock_rate, rtp_start)
    chunks = []
    for chunk_seq, start in enumerate(range(0, audio.size, chunk_samples)):
        chunk = audio[start : start + chunk_samples]
        chunks.append(
            PcapChunk(
                pcm=pcm16_bytes(chunk),
                rtp_timestamp=rtp.next_timestamp(chunk.size),
                rtp_clock_rate=rtp_clock_rate,
                chunk_seq=chunk_seq,
                duration_ms=round(chunk.size * 1000 / SAMPLE_RATE),
                end_of_stream=start + chunk_samples >= audio.size,
                due_s=start / SAMPLE_RATE,
            )
        )
    return chunks


def chunks_signal(chunks: list[PcapChunk]) -> np.ndarray:
    return np.frombuffer(b"".join(chunk.pcm for chunk in chunks), dtype="<i2").astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime WAV / pcap client for Nemotron engine")
    parser.add_argument("wav", type=Path, nargs="?", help="16-bit PCM WAV to send (or use --pcap)")
    parser.add_argument(
        "--pcap",
        type=Path,
        default=None,
        help=(
            "Classic pcap captured at rtpgw (RTP/G.711 over UDP/IPv4): rebuild the AudioChunks the gateway "
            "would send (jitter buffer, 8->16 kHz resampler, 80 ms chunks, original RTP timestamps). "
            "A trace_pcap file of the engine (L16 flows 'in' / 'audx') is sent chunk by chunk as recorded"
        ),
    )
    parser.add_argument(
        "--gw-resampler",
        choices=RESAMPLER_MODES,
        default=None,
        help=(
            "rtpgw audio.resampler emulated by --narrowband and --pcap: sinc = half-band FIR, 3.25 ms delay "
            f"(rtpgw default); linear = the older interpolation without filter (default {RTPGW_RESAMPLER})"
        ),
    )
    parser.add_argument(
        "--flow",
        default=None,
        help=(
            "pcap flow to send: '#N' (index from --list-flows), a destination port, 'src_ip:port->dst_ip:port', "
            "or the label of a trace_pcap flow (in / audx)"
        ),
    )
    parser.add_argument("--list-flows", action="store_true", help="Print the RTP flows found in --pcap and exit")
    parser.add_argument(
        "--jitter",
        choices=("sim", "off"),
        default=None,
        help=(
            "pcap only. sim (default) = replay the gateway jitter buffer (60 ms, 20 ms drain, reorders, drops "
            "lost/late/duplicate packets); off = packets in file order, like the gateway's pcap_replay"
        ),
    )
    parser.add_argument(
        "--pace",
        choices=("chunk", "capture"),
        default=None,
        help=(
            "pcap only. chunk (default) = one chunk duration between sends; capture = send each chunk when the "
            "gateway would have it ready, so lost-packet gaps and jitter-buffer delay are kept (use with --timeline)"
        ),
    )
    parser.add_argument("--address", default="127.0.0.1:50052")
    parser.add_argument("--session-id", default=None, help="default wav-<random> or pcap-<random>")
    parser.add_argument("--stream-id", default="MO")
    parser.add_argument(
        "--profile",
        choices=("default", "rtpgw"),
        default=None,
        help=(
            "WAV only. rtpgw = what the real gateway sends: --chunk-ms 80 --rtp-clock-rate 8000 --narrowband, "
            "mono int16 LE 16 kHz PCM; flags given explicitly still win"
        ),
    )
    parser.add_argument("--chunk-ms", type=int, default=None, help="default 20 (profile rtpgw and --pcap: 80)")
    parser.add_argument(
        "--rtp-clock-rate",
        type=int,
        default=None,
        help=(
            "WAV only. rtp_clock_rate sent with every chunk; rtp_timestamp advances by chunk duration in this clock "
            "(default 16000; profile rtpgw: 8000)"
        ),
    )
    parser.add_argument(
        "--narrowband",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "WAV only. Pass the audio through 8 kHz G.711 PCMU and back to 16 kHz, like the rtpgw path "
            "(needs soxr; profile rtpgw: on)"
        ),
    )
    parser.add_argument(
        "--rtp-start",
        default=None,
        help="WAV only. First rtp_timestamp: an integer or 'random' (real streams start at a random value); default 0",
    )
    parser.add_argument(
        "--language",
        default="vi",
        help="transcribe: source language; translate: TARGET language (the source is detected by the ASR)",
    )
    parser.add_argument(
        "--language-source",
        default="",
        help="SOURCE language hint (vi|en), sent as AudioChunk.language_source; empty = not sent. "
        "Honoured only when the server runs with asr.use_language_source on",
    )
    parser.add_argument(
        "--task",
        choices=("speech_to_text", "transcribe", "realtime_translation", "translate", "keyword"),
        default="speech_to_text",
    )
    parser.add_argument(
        "--audio-out",
        type=Path,
        default=None,
        help="Write FINAL audio_payload bytes for realtime_translation mode (raw G.711, no RTP header)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send chunks as fast as possible instead of pacing by chunk duration",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help=(
            "Print the receive time of every result (t=, seconds since the first chunk was sent) "
            "and lag= (t minus the media position of ts_end relative to the first chunk; meaningful with "
            "real-time pacing only, and for a pcap with lost packets only with --pace capture), "
            "then a summary: first speech in the file (energy-based) and when its chunk was sent, "
            "first result and its lag, first audio, last chunk sent (input_end), last result, "
            "sent_to_result / sent_to_audio (first speech chunk sent -> first result / first audio)"
        ),
    )
    parser.add_argument(
        "--timeline-out",
        type=Path,
        default=None,
        help="Also write the Timeline summary to this file (implies --timeline)",
    )
    args = parser.parse_args()

    if (args.wav is None) == (args.pcap is None):
        parser.error("give either a WAV file or --pcap")
    if args.pcap is None and (args.flow is not None or args.list_flows or args.jitter or args.pace):
        parser.error("--flow, --list-flows, --jitter and --pace need --pcap")
    gw_resampler = args.gw_resampler or RTPGW_RESAMPLER

    if args.pcap is not None:
        ignored = [
            flag
            for flag, value in (
                ("--profile", args.profile),
                ("--narrowband", args.narrowband),
                ("--rtp-start", args.rtp_start),
                ("--rtp-clock-rate", args.rtp_clock_rate),
            )
            if value is not None
        ]
        if ignored:
            print(
                f"WARNING: {', '.join(ignored)} ignored with --pcap (the capture already is the gateway path)",
                file=sys.stderr,
            )
        try:
            flows = read_pcap(args.pcap)
            if args.list_flows:
                if not flows:
                    raise PcapError("no RTP flows found in the pcap")
                print("\n".join(describe_flows(flows)))
                return
            flow = select_flow(flows, args.flow)
            jitter = args.jitter or "sim"
            chunk_ms = args.chunk_ms if args.chunk_ms is not None else RTPGW_CHUNK_MS
            if flow.codec == "L16":
                # a trace_pcap flow already is what the engine received: its own chunks and timestamps
                trace_ignored = [
                    flag
                    for flag, value in (
                        ("--jitter", args.jitter),
                        ("--chunk-ms", args.chunk_ms),
                        ("--gw-resampler", args.gw_resampler),
                    )
                    if value is not None
                ]
                if trace_ignored:
                    print(
                        f"WARNING: {', '.join(trace_ignored)} ignored for a trace_pcap flow (L16): it is sent as recorded",
                        file=sys.stderr,
                    )
                chunk_ms = RTPGW_CHUNK_MS
            chunks, stats = build_chunks(flow, jitter=jitter, chunk_ms=chunk_ms, resampler=gw_resampler)
        except (OSError, PcapError) as exc:
            sys.exit(f"error: {exc}")
        signal = None
        if flow.codec == "L16":
            description = (
                f"pcap={args.pcap.name} flow={flow.key} codec=L16 label={flow.label or '-'} (trace_pcap) "
                f"packets={stats.packets} chunks={len(chunks)}"
            )
        else:
            description = (
                f"pcap={args.pcap.name} flow={flow.key} codec={flow.codec} jitter={jitter} chunk_ms={chunk_ms} "
                f"resampler={gw_resampler} "
                f"packets={stats.packets} released={stats.jitter.released} dropped={stats.jitter.dropped} "
                f"lost={stats.jitter.lost} non_audio={stats.non_audio_skipped} chunks={len(chunks)}"
            )
    else:
        profile = args.profile or "default"
        rtpgw = profile == "rtpgw"
        chunk_ms = args.chunk_ms if args.chunk_ms is not None else (RTPGW_CHUNK_MS if rtpgw else 20)
        rtp_clock_rate = args.rtp_clock_rate if args.rtp_clock_rate is not None else (RTPGW_CLOCK_RATE if rtpgw else 16_000)
        narrowband = args.narrowband if args.narrowband is not None else rtpgw
        rtp_start = parse_rtp_start(args.rtp_start or "0")
        audio = read_wav(args.wav)
        if narrowband:
            audio = to_narrowband(audio, gw_resampler)
        elif args.gw_resampler is not None:
            print("WARNING: --gw-resampler ignored without --narrowband", file=sys.stderr)
        chunks = wav_chunks(audio, chunk_ms, rtp_clock_rate, rtp_start)
        signal = audio
        description = (
            f"chunk_ms={chunk_ms} rtp_clock_rate={rtp_clock_rate} rtp_start={rtp_start} narrowband={narrowband}"
            + (f" resampler={gw_resampler}" if narrowband else "")
        )

    warning = chunk_ms_warning(chunk_ms)
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.timeline_out is not None:
        args.timeline = True
    if not chunks:
        sys.exit("error: no audio chunks to send")
    if signal is None and args.timeline:
        signal = chunks_signal(chunks)

    rtp_clock_rate = chunks[0].rtp_clock_rate
    rtp_origin = chunks[0].rtp_timestamp
    session_id = args.session_id or f"{'pcap' if args.pcap is not None else 'wav'}-{uuid.uuid4().hex[:8]}"
    capture_pace = args.pace == "capture"

    request_queue: queue.Queue[object] = queue.Queue()
    sentinel = object()

    def requests():
        while True:
            item = request_queue.get()
            if item is sentinel:
                return
            yield item

    channel = grpc.insecure_channel(args.address)
    stub = speech_pb2_grpc.SpeechStreamStub(channel)
    call = stub.Recognize(requests())

    audio_parts: list[bytes] = []
    t0 = time.monotonic()
    timeline: dict[str, float | None] = {
        "speech_onset": first_speech_seconds(signal, SAMPLE_RATE) if args.timeline else None,
        "speech_sent": None,
        "first_result": None,
        "first_result_lag": None,
        "first_audio": None,
        "last_result": None,
        "input_end": None,
    }

    def receive() -> None:
        try:
            for result in call:
                kind = "FINAL" if result.is_final else "PARTIAL"
                audio_info = ""
                if result.audio_payload:
                    audio_info = f" audio_bytes={len(result.audio_payload)} pt={result.audio_pt}"
                    audio_parts.append(bytes(result.audio_payload))
                topic_info = ""
                if result.is_final and args.task == "keyword":
                    topic_info = (
                        f" topic={speech_pb2.TopicLabel.Name(result.topic_label)}"
                        f" score={result.topic_score:.4f} info={result.topic_info!r}"
                    )
                timing = ""
                if args.timeline:
                    now = time.monotonic() - t0
                    lag = now - ((result.ts_end - rtp_origin) & 0xFFFFFFFF) / rtp_clock_rate
                    timing = f"t={now:.3f}s lag={lag:.3f}s "
                    if timeline["first_result"] is None:
                        timeline["first_result"] = now
                        timeline["first_result_lag"] = lag
                    if result.audio_payload and timeline["first_audio"] is None:
                        timeline["first_audio"] = now
                    timeline["last_result"] = now
                print(
                    f"[{kind}] {timing}seq={result.seq} stream={result.stream_id} "
                    f"lang={result.language} ts={result.ts_start}-{result.ts_end}{audio_info}{topic_info}\n"
                    f"{result.text}\n",
                    flush=True,
                )
        except grpc.RpcError as exc:
            print(f"RPC failed: {exc.code()} {exc.details()}", file=sys.stderr)

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()

    print(description, flush=True)
    capture_origin = time.monotonic() - chunks[0].due_s
    sent_samples = 0
    for chunk in chunks:
        if capture_pace and not args.no_realtime:
            delay = capture_origin + chunk.due_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        request_queue.put(
            speech_pb2.AudioChunk(
                session_id=session_id,
                stream_id=args.stream_id,
                pcm=chunk.pcm,
                sample_rate=SAMPLE_RATE,
                channels=1,
                rtp_timestamp=chunk.rtp_timestamp,
                rtp_clock_rate=chunk.rtp_clock_rate,
                chunk_seq=chunk.chunk_seq,
                duration_ms=chunk.duration_ms,
                end_of_stream=chunk.end_of_stream,
                language=args.language,
                language_source=args.language_source,
                task=args.task,
            )
        )
        chunk_samples = len(chunk.pcm) // 2
        sent_samples += chunk_samples
        if (
            timeline["speech_onset"] is not None
            and timeline["speech_sent"] is None
            and sent_samples > int(timeline["speech_onset"] * SAMPLE_RATE)
        ):
            timeline["speech_sent"] = time.monotonic() - t0
        if chunk.end_of_stream:
            timeline["input_end"] = time.monotonic() - t0
        if not capture_pace and not args.no_realtime:
            time.sleep(chunk_samples / SAMPLE_RATE)

    request_queue.put(sentinel)
    receiver.join(timeout=180)
    channel.close()

    if args.timeline:
        lines = ["Timeline (seconds since the first chunk was sent):"]
        for name, value in timeline.items():
            lines.append(f"  {name:<16} {'-' if value is None else f'{value:.3f}'}")
        sent = timeline["speech_sent"]
        for label, key in (("sent_to_result", "first_result"), ("sent_to_audio", "first_audio")):
            if sent is not None and timeline[key] is not None:
                lines.append(f"  {label:<16} {timeline[key] - sent:.3f}")
        if timeline["last_result"] is not None and timeline["input_end"] is not None:
            lines.append(f"  {'after_input':<16} {timeline['last_result'] - timeline['input_end']:.3f}")
        print("\n".join(lines))
        if args.timeline_out is not None:
            args.timeline_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Wrote timeline to {args.timeline_out}")

    if args.audio_out and audio_parts:
        args.audio_out.write_bytes(b"".join(audio_parts))
        print(f"Wrote raw G.711 payload to {args.audio_out}")


if __name__ == "__main__":
    main()
