from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AudioFormatError(ValueError):
    pass


def decode_pcm_s16le(payload: bytes, channels: int) -> np.ndarray:
    if channels <= 0:
        raise AudioFormatError(f"channels must be positive, got {channels}")
    if not payload:
        return np.empty(0, dtype=np.float32)
    frame_bytes = 2 * channels
    if len(payload) % frame_bytes:
        raise AudioFormatError(
            f"PCM byte length {len(payload)} is not divisible by frame size {frame_bytes}"
        )
    pcm = np.frombuffer(payload, dtype="<i2").reshape(-1, channels)
    mono = pcm.astype(np.float32).mean(axis=1)
    return mono / 32768.0


def pcm16_bytes_to_float32(payload: bytes, channels: int = 1) -> np.ndarray:
    return decode_pcm_s16le(payload, channels)


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.empty(0, dtype=np.int16)
    # Sanitize non-finite values before clipping: np.clip passes NaN through
    # unchanged, which produces undefined int16 output on astype.
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clip max to 32767/32768 so round(x * 32768) never exceeds int16 max (32767).
    x = np.clip(x, -1.0, 32767.0 / 32768.0)
    return np.round(x * 32768.0).astype("<i2")


def _linear_resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    # Testbed fallback only. Each chunk is resampled independently so there is
    # no phase continuity at chunk boundaries — install soxr for production quality.
    if audio.size == 0 or source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    output_size = int(round(audio.size * target_rate / source_rate))
    if output_size <= 0:
        return np.empty(0, dtype=np.float32)
    source_positions = np.linspace(0.0, audio.size - 1, num=output_size)
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, audio.size - 1)
    frac = source_positions - left
    output = audio[left] * (1.0 - frac) + audio[right] * frac
    return output.astype(np.float32, copy=False)


_soxr_warned = False


def resample(audio: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if source_rate <= 0:
        raise AudioFormatError(f"sample_rate must be positive, got {source_rate}")
    if target_rate <= 0:
        raise AudioFormatError(f"target_rate must be positive, got {target_rate}")
    if source_rate == target_rate:
        return np.array(audio, dtype=np.float32)
    try:
        import soxr  # type: ignore

        return soxr.resample(audio, source_rate, target_rate).astype(np.float32, copy=False)
    except ImportError:
        global _soxr_warned
        if not _soxr_warned:
            _soxr_warned = True
            logger.warning(
                "soxr is not installed; using a linear testbed resampler. "
                "Install soxr for production-quality resampling."
            )
        return _linear_resample(audio, source_rate, target_rate)
    except Exception as exc:
        logger.warning("soxr.resample failed (%s); falling back to linear resampler", exc)
        return _linear_resample(audio, source_rate, target_rate)


def decode_and_resample(
    payload: bytes,
    sample_rate: int,
    channels: int,
    target_rate: int = 16_000,
) -> np.ndarray:
    return resample(decode_pcm_s16le(payload, channels), sample_rate, target_rate)
