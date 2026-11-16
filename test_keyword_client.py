#!/usr/bin/env python3
import argparse
import sys
import time
import wave
from pathlib import Path

import grpc

# Allow running from project root without installing the package.
ROOT = Path(__file__).resolve().parent
PROJECT_SRC = ROOT / "src"
if PROJECT_SRC.is_dir():
    sys.path.insert(0, str(PROJECT_SRC))

from nemotron_testbed.pb import speech_pb2, speech_pb2_grpc


def build_requests(wav_path: str, session_id: str, stream_id: str,
                   language: str, chunk_ms: int, realtime: bool):
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()

        if sample_width != 2:
            raise ValueError(
                f"WAV must be PCM 16-bit (sample width=2). Got {sample_width} bytes/sample"
            )

        frames_per_chunk = max(1, sample_rate * chunk_ms // 1000)
        seq = 0
        rtp_timestamp = 0

        while True:
            pcm = wf.readframes(frames_per_chunk)
            if not pcm:
                break

            # Number of frames actually read in this chunk.
            bytes_per_frame = channels * sample_width
            frames = len(pcm) // bytes_per_frame
            duration_ms = int(round(frames * 1000 / sample_rate))
            is_last = wf.tell() >= wf.getnframes()

            yield speech_pb2.AudioChunk(
                session_id=session_id,
                stream_id=stream_id,
                pcm=pcm,
                sample_rate=sample_rate,
                channels=channels,
                rtp_timestamp=rtp_timestamp,
                duration_ms=duration_ms,
                end_of_stream=is_last,
                language=language,
                task="keyword",
                rtp_clock_rate=sample_rate,
                chunk_seq=seq,
            )

            seq += 1
            rtp_timestamp += frames

            if realtime and not is_last:
                time.sleep(frames / sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Test gRPC ASR + keyword flow with a WAV file")
    parser.add_argument("wav", help="Input WAV file (PCM16)")
    parser.add_argument("--server", default="127.0.0.1:50052")
    parser.add_argument("--session-id", default="keyword-test-001")
    parser.add_argument("--stream-id", default="caller")
    parser.add_argument("--language", default="vi-VN")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Send chunks as fast as possible instead of real-time pacing",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        raise SystemExit(f"WAV not found: {wav_path}")

    print(f"[CLIENT] server={args.server}")
    print(f"[CLIENT] wav={wav_path}")
    print(f"[CLIENT] task=keyword language={args.language} chunk_ms={args.chunk_ms}")

    options = [
        ("grpc.max_send_message_length", 8 * 1024 * 1024),
        ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    ]

    with grpc.insecure_channel(args.server, options=options) as channel:
        stub = speech_pb2_grpc.SpeechStreamStub(channel)
        requests = build_requests(
            str(wav_path),
            args.session_id,
            args.stream_id,
            args.language,
            args.chunk_ms,
            realtime=not args.fast,
        )

        try:
            for result in stub.Recognize(requests):
                label = speech_pb2.TopicLabel.Name(result.topic_label)
                state = "FINAL" if result.is_final else "PARTIAL"

                print(
                    f"[{state}] seq={result.seq} text={result.text!r} "
                    f"label={label} score={result.topic_score:.4f} info={result.topic_info!r} "
                    f"ts=({result.ts_start},{result.ts_end})"
                )
        except grpc.RpcError as exc:
            print(f"[gRPC ERROR] code={exc.code()} details={exc.details()}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
