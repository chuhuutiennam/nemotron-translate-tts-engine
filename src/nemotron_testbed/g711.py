from __future__ import annotations

import numpy as np

from .audio import float32_to_pcm16


def _linear_to_ulaw_sample(sample: int) -> int:
    # ITU-T G.711 mu-law, 16-bit PCM input.
    BIAS = 0x84
    CLIP = 32635
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    sample = min(sample, CLIP) + BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _linear_to_alaw_sample(sample: int) -> int:
    # ITU-T G.711 A-law, 16-bit PCM input.
    pcm = int(sample)
    if pcm >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        pcm = -pcm - 1

    pcm = min(pcm, 32767)
    if pcm < 256:
        aval = pcm >> 4
    else:
        exponent = 1
        value = pcm >> 8
        while value > 1 and exponent < 7:
            exponent += 1
            value >>= 1
        aval = (exponent << 4) | ((pcm >> (exponent + 3)) & 0x0F)
    return aval ^ mask


# Lookup tables — built once at import. audioop was removed in Python 3.13;
# without these, the fallback would loop over every sample in pure Python.
_ULAW_TABLE: np.ndarray = np.array(
    [_linear_to_ulaw_sample(i - 32768) for i in range(65536)], dtype=np.uint8
)
_ALAW_TABLE: np.ndarray = np.array(
    [_linear_to_alaw_sample(i - 32768) for i in range(65536)], dtype=np.uint8
)


def _fallback_encode(pcm16: np.ndarray, payload_type: int) -> bytes:
    indices = pcm16.astype(np.int32) + 32768  # int16 [-32768,32767] → [0,65535]
    if payload_type == 0:
        return _ULAW_TABLE[indices].tobytes()
    return _ALAW_TABLE[indices].tobytes()


def encode_g711(audio_float32: np.ndarray, payload_type: int) -> bytes:
    """Encode mono float32 [-1, 1] into raw G.711 bytes.

    The returned bytes contain no RTP header. PT=0 is PCMU and PT=8 is PCMA.
    """
    if payload_type not in (0, 8):
        raise ValueError("payload_type must be 0 (PCMU) or 8 (PCMA)")
    pcm16 = float32_to_pcm16(audio_float32)
    if pcm16.size == 0:
        return b""

    try:
        import audioop  # Python <= 3.12

        raw = pcm16.astype("<i2", copy=False).tobytes()
        if payload_type == 0:
            return audioop.lin2ulaw(raw, 2)
        return audioop.lin2alaw(raw, 2)
    except (ImportError, AttributeError):
        return _fallback_encode(pcm16, payload_type)
