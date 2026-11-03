import unittest

import numpy as np

from nemotron_testbed.audio import float32_to_pcm16
from nemotron_testbed.g711 import _fallback_encode, encode_g711


class G711Test(unittest.TestCase):
    def test_one_byte_per_sample(self):
        audio = np.zeros(8000, dtype=np.float32)
        self.assertEqual(len(encode_g711(audio, 0)), 8000)
        self.assertEqual(len(encode_g711(audio, 8)), 8000)

    def test_reject_unknown_pt(self):
        with self.assertRaises(ValueError):
            encode_g711(np.zeros(10, dtype=np.float32), 96)

    def test_ulaw_silence_encodes_to_0xff(self):
        # ITU-T G.711: PCM silence (0x0000) must map to 0xFF for PCMU.
        result = encode_g711(np.zeros(1, dtype=np.float32), 0)
        self.assertEqual(result[0], 0xFF)

    def test_alaw_silence_encodes_to_0xd5(self):
        # ITU-T G.711: PCM silence (0x0000) must map to 0xD5 for PCMA.
        result = encode_g711(np.zeros(1, dtype=np.float32), 8)
        self.assertEqual(result[0], 0xD5)

    def test_empty_audio_returns_empty_bytes(self):
        self.assertEqual(encode_g711(np.zeros(0, dtype=np.float32), 0), b"")
        self.assertEqual(encode_g711(np.zeros(0, dtype=np.float32), 8), b"")

    def test_lut_close_to_audioop_ulaw(self):
        # audioop uses a 14-bit-precision internal implementation; our LUT uses
        # full 16-bit. The two are expected to agree within ±1 G.711 code per sample.
        rng = np.random.default_rng(0)
        audio = rng.uniform(-1.0, 1.0, 200).astype(np.float32)
        pcm16 = float32_to_pcm16(audio)
        lut = np.frombuffer(_fallback_encode(pcm16, 0), dtype=np.uint8)
        try:
            import audioop
            raw = pcm16.astype("<i2", copy=False).tobytes()
            ref = np.frombuffer(audioop.lin2ulaw(raw, 2), dtype=np.uint8)
            diff = np.abs(lut.astype(np.int16) - ref.astype(np.int16))
            self.assertTrue(np.all(diff <= 1), f"max diff={diff.max()}")
        except (ImportError, AttributeError):
            pass  # audioop removed in Python 3.13 — LUT is the only path

    def test_lut_close_to_audioop_alaw(self):
        rng = np.random.default_rng(1)
        audio = rng.uniform(-1.0, 1.0, 200).astype(np.float32)
        pcm16 = float32_to_pcm16(audio)
        lut = np.frombuffer(_fallback_encode(pcm16, 8), dtype=np.uint8)
        try:
            import audioop
            raw = pcm16.astype("<i2", copy=False).tobytes()
            ref = np.frombuffer(audioop.lin2alaw(raw, 2), dtype=np.uint8)
            diff = np.abs(lut.astype(np.int16) - ref.astype(np.int16))
            self.assertTrue(np.all(diff <= 1), f"max diff={diff.max()}")
        except (ImportError, AttributeError):
            pass

    def test_float32_to_pcm16_no_overflow_at_max(self):
        # +1.0 must not wrap to -32768 (int16 overflow).
        pcm = float32_to_pcm16(np.array([1.0, -1.0], dtype=np.float32))
        self.assertEqual(int(pcm[0]), 32767)
        self.assertEqual(int(pcm[1]), -32768)

    def test_float32_to_pcm16_roundtrip_zero(self):
        pcm = float32_to_pcm16(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(pcm, np.zeros(4, dtype=np.int16))


if __name__ == "__main__":
    unittest.main()
