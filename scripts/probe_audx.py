#!/usr/bin/env python3
"""Run the AudX model (nemotron_testbed.audx_model, noise reduction of mode audx) on WAV files and measure what it
does to speech. Not resemble-enhance: its modes denoiser / enhancer have their own probe (probe_preproc_gpu.py).

Offline, on this machine: no engine, no gRPC. For each WAV (read as 16 kHz mono; --narrowband first puts it
through the G.711 path the engine gets from rtpgw) it reports:
  * speed: real-time factor of a whole-file pass on --device,
  * speech level: input -> output level over speech-active 20 ms blocks (within 40 dB of the loudest
    block), and over the quiet blocks (what the noise floor became),
  * bands: input -> output energy below and above 4 kHz (a narrowband input has nothing above 4 kHz),
  * level dependence (--gains): the same file scaled by each gain; AudX has no level normalisation, so a
    speech-level change that moves with the gain means its output depends on how loud the call is,
  * windowed vs whole file (unless --no-window): the engine's StreamAudioPreprocessor with the given
    window (left/hop/right ms), results delivered at once, compared with the whole-file pass,
  * stream (what mode audx runs): AudxModel.step() over --piece-ms pieces with the AGC of
    --agc-target-dbfs / --agc-max-gain-db (0 = off): speed, speech level change, SNR vs the whole-file
    pass (no AGC there, so only close when the AGC does not boost), and the level dependence with the AGC.
--out writes <name>_in.wav, <name>_audx.wav, <name>_audx_windowed.wav and <name>_audx_stream.wav to listen to.

Needs torch (CPU is enough), numpy, soxr (non-16 kHz WAV, --narrowband), prometheus_client and pyyaml
(windowed pass, engine modules). PYTHONPATH=src.

Example:
  PYTHONPATH=src python scripts/probe_audx.py asr_data/friend.wav \\
      --checkpoint models/audx/audx_weights_safe.pth --narrowband --no-window --out /tmp/audx
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    from nemotron_testbed.audx_model import SAMPLE_RATE, AudxModel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nemotron_testbed.audx_model import SAMPLE_RATE, AudxModel

BLOCK = SAMPLE_RATE // 50  # 20 ms
ACTIVE_RANGE_DB = 40.0
FLOOR_DBFS = -60.0


def db(power: float) -> float:
    return 10.0 * math.log10(max(power, 1e-20))


def block_powers(audio: np.ndarray) -> np.ndarray:
    usable = audio[: len(audio) // BLOCK * BLOCK].astype(np.float64)
    return np.mean(usable.reshape(-1, BLOCK) ** 2, axis=1) if usable.size else np.zeros(0)


def activity(audio: np.ndarray) -> np.ndarray:
    """Speech-active 20 ms blocks of the input: within ACTIVE_RANGE_DB of the loudest and above FLOOR_DBFS."""
    powers = block_powers(audio)
    if not powers.size:
        return np.zeros(0, dtype=bool)
    levels = 10.0 * np.log10(np.maximum(powers, 1e-20))
    return (levels >= levels.max() - ACTIVE_RANGE_DB) & (levels >= FLOOR_DBFS)


def level_over(audio: np.ndarray, mask: np.ndarray) -> float | None:
    powers = block_powers(audio)[: mask.size]
    return db(float(np.mean(powers[mask[: powers.size]]))) if mask.any() else None


def band_powers(audio: np.ndarray, split_hz: float = 4000.0) -> tuple[float, float]:
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / SAMPLE_RATE)
    total = max(audio.size, 1)
    return db(float(spectrum[freqs < split_hz].sum()) / total), db(float(spectrum[freqs >= split_hz].sum()) / total)


def snr_db(reference: np.ndarray, other: np.ndarray, mask: np.ndarray) -> float:
    n = min(reference.size, other.size) // BLOCK * BLOCK
    if n == 0:
        return float("nan")
    keep = np.repeat(mask[: n // BLOCK], BLOCK)
    ref, oth = reference[:n][keep].astype(np.float64), other[:n][keep].astype(np.float64)
    noise = float(np.sum((ref - oth) ** 2))
    return float("inf") if noise == 0.0 else 10.0 * math.log10(float(np.sum(ref**2)) / noise)


def windowed(model: AudxModel, audio: np.ndarray, left_ms: int, hop_ms: int, right_ms: int, crossfade_ms: int):
    """The engine's StreamAudioPreprocessor over the whole signal, every window computed at once."""
    from nemotron_testbed.audio_preprocess import StreamAudioPreprocessor
    from nemotron_testbed.config import AudioPreprocessSettings
    from nemotron_testbed.domain import AudioPreprocessResult, StreamKey, SubmitStatus

    settings = AudioPreprocessSettings(
        enabled=True, left_context_ms=left_ms, hop_ms=hop_ms, right_context_ms=right_ms,
        crossfade_ms=crossfade_ms, compute_budget_ms=1,
    )
    key = StreamKey("probe", "probe", "MO")
    pre: StreamAudioPreprocessor

    def submit(job):
        pre.deliver(AudioPreprocessResult(key=key, region_id=job.region_id, audio=model(job.window)))
        return SubmitStatus.ACCEPTED

    pre = StreamAudioPreprocessor(key, settings, submit)
    chunk = SAMPLE_RATE * 20 // 1000
    parts = [pre.process(audio[start : start + chunk]).enhanced for start in range(0, audio.size, chunk)]
    parts.append(asyncio.run(pre.flush()).enhanced)
    stats = dict(pre.stats)
    pre.close()
    out = np.concatenate(parts)
    return out[pre.delay_samples : pre.delay_samples + audio.size], stats, pre.delay_samples


def stream(model: AudxModel, audio: np.ndarray, piece_ms: int) -> tuple[np.ndarray, float]:
    """(output, seconds): AudxModel.step() over pieces of piece_ms, as the engine's mode audx runs it."""
    piece = SAMPLE_RATE * piece_ms // 1000
    state = model.new_state()
    started = time.perf_counter()
    parts = [model.step(state, audio[start : start + piece]) for start in range(0, audio.size, piece)]
    parts.append(model.step(state, np.zeros(0, dtype=np.float32), final=True))
    return np.concatenate(parts), time.perf_counter() - started


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 0.9999695) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/audx/audx_weights_safe.pth"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--narrowband", action="store_true", help="G.711 path like rtpgw before the model")
    parser.add_argument("--gains", default="0 -10 -20", help="input gains in dB for the level-dependence check")
    parser.add_argument("--window", default="1000/200/100", help="left/hop/right context ms for the windowed pass")
    parser.add_argument("--crossfade-ms", type=int, default=40)
    parser.add_argument("--no-window", action="store_true", help="skip the windowed pass")
    parser.add_argument("--piece-ms", type=int, default=80, help="stream pass: audio per step (engine hop_ms)")
    parser.add_argument("--agc-target-dbfs", type=float, default=-28.0)
    parser.add_argument("--agc-max-gain-db", type=float, default=30.0, help="0 = stream pass without AGC")
    parser.add_argument("--out", type=Path, default=None, help="directory for the input/enhanced WAVs")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from nemotron_testbed.client_audio import read_wav, to_narrowband

    gains = [float(g) for g in args.gains.split()]
    left_ms, hop_ms, right_ms = (int(v) for v in args.window.split("/"))
    model = AudxModel(
        args.checkpoint, args.device, agc_target_dbfs=args.agc_target_dbfs, agc_max_gain_db=args.agc_max_gain_db
    )
    agc = f"AGC {args.agc_target_dbfs:g} dBFS +{args.agc_max_gain_db:g} dB" if args.agc_max_gain_db > 0 else "no AGC"
    print(f"checkpoint {args.checkpoint} sha256 {model.sha256} device {args.device}"
          f"{' narrowband (G.711 path)' if args.narrowband else ''}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for path in args.wavs:
        audio = read_wav(path)
        if args.narrowband:
            audio = to_narrowband(audio)
        seconds = audio.size / SAMPLE_RATE
        started = time.perf_counter()
        enhanced = model(audio)
        rtf = (time.perf_counter() - started) / max(seconds, 1e-9)
        active = activity(audio)
        quiet = ~active
        speech_in, speech_out = level_over(audio, active), level_over(enhanced, active)
        quiet_in, quiet_out = level_over(audio, quiet), level_over(enhanced, quiet)
        low_in, high_in = band_powers(audio)
        low_out, high_out = band_powers(enhanced)
        print(f"\n== {path.name}: {seconds:.1f} s, rtf {rtf:.3f}, speech blocks {int(active.sum())}/{active.size}")
        print(f"   speech level  {speech_in:7.1f} -> {speech_out:7.1f} dBFS  ({fmt(speech_out - speech_in)} dB)")
        if quiet_in is not None:
            print(f"   quiet blocks  {quiet_in:7.1f} -> {quiet_out:7.1f} dBFS  ({fmt(quiet_out - quiet_in)} dB)")
        print(f"   < 4 kHz       {low_in:7.1f} -> {low_out:7.1f} dB  ({fmt(low_out - low_in)} dB)")
        print(f"   >= 4 kHz      {high_in:7.1f} -> {high_out:7.1f} dB  ({fmt(high_out - high_in)} dB)")
        changes = []
        for gain in gains:
            scale = 10.0 ** (gain / 20.0)
            out = model(audio * scale)
            changes.append(f"{gain:+.0f} dB in: {fmt(level_over(out, active) - level_over(audio * scale, active))} dB")
        print(f"   speech level change by input gain: {', '.join(changes)}")
        windowed_out = None
        if not args.no_window:
            windowed_out, stats, delay = windowed(model, audio, left_ms, hop_ms, right_ms, args.crossfade_ms)
            print(
                f"   windowed {left_ms}/{hop_ms}/{right_ms} ms (D without budget {delay * 1000 // SAMPLE_RATE} ms): "
                f"SNR vs whole file over speech {snr_db(enhanced, windowed_out, active):.1f} dB; "
                f"regions enhanced {stats['enhanced']} silent {stats['silent']} late {stats['late']}"
            )
        streamed, stream_s = stream(model, audio, args.piece_ms)
        stream_changes = []
        for gain in gains:
            scale = 10.0 ** (gain / 20.0)
            out, _ = stream(model, audio * scale, args.piece_ms)
            stream_changes.append(
                f"{gain:+.0f} dB in: {fmt(level_over(out, active) - level_over(audio * scale, active))} dB"
            )
        print(
            f"   stream ({args.piece_ms} ms pieces, {agc}): rtf {stream_s / max(seconds, 1e-9):.3f}, "
            f"speech level {fmt(level_over(streamed, active) - speech_in)} dB, "
            f"SNR vs whole file over speech {snr_db(enhanced, streamed, active):.1f} dB"
        )
        print(f"   stream speech level change by input gain: {', '.join(stream_changes)}")
        if args.out:
            write_wav(args.out / f"{path.stem}_in.wav", audio)
            write_wav(args.out / f"{path.stem}_audx.wav", enhanced)
            write_wav(args.out / f"{path.stem}_audx_stream.wav", streamed)
            if windowed_out is not None:
                write_wav(args.out / f"{path.stem}_audx_windowed.wav", windowed_out)
    if args.out:
        print(f"\nWAVs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
