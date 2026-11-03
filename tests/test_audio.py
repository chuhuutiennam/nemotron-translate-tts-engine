import unittest

import numpy as np

from nemotron_testbed.audio import AudioFormatError, decode_pcm_s16le, resample


class AudioTest(unittest.TestCase):
    def test_decode_mono(self):
        raw = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        audio = decode_pcm_s16le(raw, channels=1)
        self.assertEqual(audio.dtype, np.float32)
        self.assertAlmostEqual(float(audio[0]), -1.0, places=5)
        self.assertAlmostEqual(float(audio[1]), 0.0, places=5)

    def test_decode_stereo_downmix(self):
        raw = np.array([[1000, -1000], [2000, 2000]], dtype="<i2").tobytes()
        audio = decode_pcm_s16le(raw, channels=2)
        self.assertAlmostEqual(float(audio[0]), 0.0, places=5)
        self.assertAlmostEqual(float(audio[1]), 2000 / 32768.0, places=5)

    def test_reject_invalid_frame_size(self):
        with self.assertRaises(AudioFormatError):
            decode_pcm_s16le(b"abc", channels=1)

    def test_linear_resample_length(self):
        audio = np.linspace(-1, 1, 800, dtype=np.float32)
        output = resample(audio, 8000, 16000)
        self.assertEqual(output.size, 1600)


if __name__ == "__main__":
    unittest.main()
