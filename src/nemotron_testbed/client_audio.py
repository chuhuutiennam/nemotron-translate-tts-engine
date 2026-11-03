"""Audio and RTP helpers shared by the test clients and probes in scripts/."""

from __future__ import annotations

import random
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
RTPGW_CHUNK_MS = 80
RTPGW_CLOCK_RATE = 8_000


def read_wav(path: Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 in [-1, 1] at target_rate from a 16-bit PCM WAV."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported by the test clients")
    pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
    if sample_rate != target_rate:
        try:
            import soxr

            mono = soxr.resample(mono, sample_rate, target_rate)
        except ImportError as exc:
            raise RuntimeError(
                "WAV is not 16 kHz and soxr is not installed. "
                "Install soxr or convert the file first."
            ) from exc
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


def pcm16_bytes(audio: np.ndarray) -> bytes:
    """Mono int16 little-endian payload, the PCM format the gateway sends."""
    return (np.clip(audio, -1.0, 0.9999695) * 32768.0).astype("<i2").tobytes()


def to_narrowband(audio: np.ndarray, resampler: str = "sinc") -> np.ndarray:
    """16 kHz float32 -> what the engine receives from rtpgw for a G.711 PCMU call.

    The caller side is soxr (anti-alias low-pass) + PCMU encode; the gateway side is client_pcap's decode (ITU
    polarity) and 8 -> 16 kHz resampler, fed per 20 ms packet like rtpgw. resampler is rtpgw audio.resampler:
    sinc (default) = half-band FIR with state across packets, so the output lags the input by 52 samples (3.25 ms;
    rtpgw shifts rtp_timestamp to match); linear = the older path (no filter, last sample of each packet held).
    The output has the input length: the filter tail is dropped, as the live gateway never delivers it.
    """
    try:
        import soxr
    except ImportError as exc:
        raise RuntimeError("--narrowband needs soxr (pip install soxr)") from exc
    from nemotron_testbed.client_pcap import G711_RATE, PACKET_TIME_MS, GatewayResampler, decode_g711
    from nemotron_testbed.g711 import encode_g711

    upsampler = GatewayResampler(resampler)  # type: ignore[arg-type]
    narrow = soxr.resample(audio, SAMPLE_RATE, G711_RATE)
    decoded = decode_g711(encode_g711(narrow.astype(np.float32), 0), "PCMU")
    packet_samples = G711_RATE * PACKET_TIME_MS // 1000
    packets = [upsampler.process(decoded[start : start + packet_samples]) for start in range(0, decoded.size, packet_samples)]
    upsampled = np.concatenate(packets) if packets else np.empty(0, dtype=np.int16)
    return (upsampled / 32768.0).astype(np.float32)


def rtp_ticks(samples: int, sample_rate: int, rtp_clock_rate: int) -> int:
    return round(samples * rtp_clock_rate / sample_rate)


def parse_rtp_start(value: str) -> int:
    """'random' -> random 32-bit RTP timestamp (real streams do not start at 0); else an integer."""
    if value.strip().lower() == "random":
        return random.getrandbits(32)
    return int(value) & 0xFFFFFFFF


@dataclass(slots=True)
class RtpTimeline:
    """rtp_timestamp from the cumulative sample count, so rounding never accumulates."""

    sample_rate: int
    rtp_clock_rate: int
    start: int = 0
    samples_sent: int = 0

    def next_timestamp(self, chunk_samples: int) -> int:
        """Timestamp for a chunk of chunk_samples, then advance past it."""
        timestamp = (self.start + rtp_ticks(self.samples_sent, self.sample_rate, self.rtp_clock_rate)) & 0xFFFFFFFF
        self.samples_sent += chunk_samples
        return timestamp


def chunk_ms_warning(chunk_ms: int) -> str | None:
    if chunk_ms > RTPGW_CHUNK_MS:
        return (
            f"--chunk-ms {chunk_ms} is larger than the {RTPGW_CHUNK_MS} ms rtpgw sends; "
            "VAD and latency numbers will not match real calls"
        )
    return None
