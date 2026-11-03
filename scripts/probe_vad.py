#!/usr/bin/env python3
"""Run the engine's VadController over a WAV and print the START/END timeline and score stats.

Chunks are fed exactly like the engine does (default 20 ms; rtpgw sends 80) and the
controller scores Silero in exact 512-sample (32 ms) windows, carrying the remainder
between chunks. Stats and --trace are per window. Use --mode gated to
reproduce the audio the engine sees when the client's rtp_clock_rate does not
match its RTP timestamps (20 ms of zeros prepended to every chunk but the first).

Examples:
  python scripts/probe_vad.py scripts/source/speech_16k.wav --backend silero --mode both
  python scripts/probe_vad.py scripts/source/speech_16k.wav --trace 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nemotron_testbed.client_audio import read_wav, to_narrowband
from nemotron_testbed.vad import VadEvent, build_vad_controller


class _ScoreRecorder:
    def __init__(self, inner):
        self._inner = inner
        self.scores: list[float] = []
        self.last_frames: list[tuple[float, float]] = []
        self.needs_executor = False

    def score(self, audio: np.ndarray) -> float:
        value = self._inner.score(audio)
        self.scores.append(value)
        self.last_frames = [(value, 0.0)]
        return value

    def frame_scores(self, audio: np.ndarray, duration_ms: float) -> list[tuple[float, float]]:
        inner = getattr(self._inner, "frame_scores", None)
        frames = inner(audio, duration_ms) if inner is not None else [(self._inner.score(audio), duration_ms)]
        self.scores.extend(score for score, ms in frames if ms > 0.0)
        self.last_frames = frames
        return frames

    def reset(self) -> None:
        self._inner.reset()


def run(audio: np.ndarray, args: argparse.Namespace, gap_ms: float) -> None:
    controller = build_vad_controller(
        backend=args.backend,
        enabled=True,
        threshold=args.threshold,
        energy_db_threshold=args.energy_db_threshold,
        min_speech_ms=args.min_speech_ms,
        end_silence_ms=args.end_silence_ms,
        silero_model_path=args.silero_model_path,
        device=args.device,
    )
    recorder = _ScoreRecorder(controller.probability)
    controller.probability = recorder

    chunk = max(1, 16_000 * args.chunk_ms // 1000)
    gap_samples = int(round(16_000 * gap_ms / 1000.0))
    total_chunks = audio.size // chunk
    utterances: list[list[float | None]] = []
    active_samples = 0
    trace_lines: list[str] = []

    for i in range(total_chunks):
        piece = audio[i * chunk : (i + 1) * chunk]
        duration_ms = args.chunk_ms
        if gap_samples and i > 0:
            piece = np.concatenate([np.zeros(gap_samples, dtype=np.float32), piece])
            duration_ms += gap_ms
        event = controller.process(piece, duration_ms)
        t_ms = (i + 1) * args.chunk_ms
        if len(trace_lines) < args.trace:
            windows = [score for score, ms in recorder.last_frames if ms > 0.0]
            shown = " ".join(f"{v:.2f}{'*' if v >= args.threshold else ''}" for v in windows) or "-"
            trace_lines.append(
                f"  chunk {i:5d} t={t_ms / 1000:7.2f}s windows={len(windows)} [{shown}] event={event.name}"
            )
        if event is VadEvent.START:
            utterances.append([t_ms / 1000.0, None])
            active_samples = 0
        elif utterances and utterances[-1][1] is None:
            active_samples += piece.size
            force = active_samples >= 16_000 * args.max_utterance_ms / 1000
            if event is VadEvent.END or force:
                utterances[-1][1] = t_ms / 1000.0
                controller.reset()
    if utterances and utterances[-1][1] is None:
        utterances[-1][1] = total_chunks * args.chunk_ms / 1000.0

    scores = np.asarray(recorder.scores, dtype=np.float64)
    label = "raw audio" if not gap_samples else f"gated (+{gap_ms:g} ms zeros before each chunk)"
    print(f"== {label} | backend={args.backend} threshold={args.threshold} "
          f"min_speech={args.min_speech_ms}ms end_silence={args.end_silence_ms}ms ==")
    pending_ms = getattr(recorder._inner, "pending_ms", None)
    tail = f" unscored_tail={pending_ms:.1f}ms" if pending_ms else ""
    print(f"chunks={total_chunks} windows={scores.size} file={audio.size / 16_000:.2f}s "
          f"score mean={scores.mean():.3f} max={scores.max():.3f} "
          f"windows>=threshold={float((scores >= args.threshold).mean()):.2f}{tail}")
    covered = sum((end or 0.0) - start for start, end in utterances)
    print(f"utterances={len(utterances)} covered={covered:.2f}s of {audio.size / 16_000:.2f}s")
    for index, (start, end) in enumerate(utterances[: args.max_print], start=1):
        print(f"  utt {index:3d}: START@{start:7.2f}s END@{end:7.2f}s dur={end - start:6.2f}s")
    if len(utterances) > args.max_print:
        print(f"  ... {len(utterances) - args.max_print} more")
    for line in trace_lines:
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--backend", choices=("silero", "energy"), default="silero")
    parser.add_argument("--mode", choices=("raw", "gated", "both"), default="both")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--narrowband", action="store_true", help="8 kHz G.711 round trip first, like the rtpgw audio (needs soxr)")
    parser.add_argument("--gap-ms", type=float, default=20.0, help="zeros prepended per chunk in gated mode")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--energy-db-threshold", type=float, default=-42.0)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--end-silence-ms", type=int, default=250)
    parser.add_argument("--max-utterance-ms", type=int, default=10_000)
    parser.add_argument("--silero-model-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace", type=int, default=0, help="print the per-window scores (* = speech) of the first N chunks")
    parser.add_argument("--max-print", type=int, default=30)
    args = parser.parse_args()

    audio = read_wav(args.wav)
    if args.narrowband:
        audio = to_narrowband(audio)
    if args.mode in ("raw", "both"):
        run(audio, args, gap_ms=0.0)
    if args.mode in ("gated", "both"):
        run(audio, args, gap_ms=args.gap_ms)


if __name__ == "__main__":
    main()
