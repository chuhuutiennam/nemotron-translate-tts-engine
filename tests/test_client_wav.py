import importlib.util
import unittest
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "client_wav.py"
_spec = importlib.util.spec_from_file_location("client_wav", _SCRIPT)
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


class RtpTicksTest(unittest.TestCase):
    def test_100ms_chunk_at_8k_clock_is_800_ticks(self):
        self.assertEqual(client.rtp_ticks(1600, 16_000, 8_000), 800)

    def test_rtpgw_80ms_chunk_at_8k_clock_is_640_ticks(self):
        self.assertEqual(client.rtp_ticks(1280, 16_000, 8_000), 640)

    def test_20ms_chunk_at_8k_clock_is_160_ticks(self):
        self.assertEqual(client.rtp_ticks(320, 16_000, 8_000), 160)

    def test_same_clock_and_rate_keeps_sample_count(self):
        self.assertEqual(client.rtp_ticks(320, 16_000, 16_000), 320)


class NarrowbandTest(unittest.TestCase):
    def setUp(self):
        try:
            import soxr  # noqa: F401
        except ImportError:
            self.skipTest("soxr is not installed")

    def _tone(self, hz: float) -> np.ndarray:
        t = np.arange(16_000, dtype=np.float32) / 16_000
        return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)

    def test_keeps_length_and_rate(self):
        out = client.to_narrowband(self._tone(1000.0))
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(out.size, 16_000, delta=4)

    def test_passes_telephone_band(self):
        out = client.to_narrowband(self._tone(1000.0))
        self.assertGreater(float(np.sqrt(np.mean(out**2))), 0.3)

    def test_caller_side_filter_removes_input_above_4khz(self):
        out = client.to_narrowband(self._tone(6000.0))
        self.assertLess(float(np.sqrt(np.mean(out**2))), 0.02)

    def _image_db(self, out: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
        freqs = np.fft.rfftfreq(out.size, 1 / 16_000)
        return float(20 * np.log10(spectrum[np.argmin(abs(freqs - 7000))] / spectrum[np.argmin(abs(freqs - 1000))]))

    def test_polarity_follows_the_input_like_the_itu_pcmu_decoder(self):
        tone = self._tone(1000.0)
        for mode, lag in (("sinc", 52), ("linear", 0)):
            with self.subTest(mode):
                out = client.to_narrowband(tone, mode)
                # sinc output lags by the filter delay: 26 samples at 8 kHz
                self.assertGreater(float(np.corrcoef(tone[: tone.size - lag], out[lag:])[0, 1]), 0.99)

    def test_default_is_the_gateway_sinc_resampler(self):
        tone = self._tone(1000.0)
        np.testing.assert_array_equal(client.to_narrowband(tone), client.to_narrowband(tone, "sinc"))

    def test_sinc_removes_the_image_above_4khz(self):
        self.assertLess(self._image_db(client.to_narrowband(self._tone(1000.0), "sinc")), -60.0)

    def test_linear_each_20ms_packet_ends_with_a_held_sample(self):
        # linear upsamples packet by packet: the last output pair of a packet repeats its last sample, nothing is
        # interpolated across the packet boundary
        out = client.to_narrowband(self._tone(1000.0), "linear")
        packets = out.size // 320
        self.assertGreater(packets, 40)
        np.testing.assert_array_equal(out[318 : packets * 320 : 320], out[319 : packets * 320 : 320])

    def test_linear_keeps_the_image_above_4khz_because_it_does_not_filter(self):
        # linear interpolation leaves the 7 kHz image about 28 dB down
        image_db = self._image_db(client.to_narrowband(self._tone(1000.0), "linear"))
        self.assertGreater(image_db, -40.0)
        self.assertLess(image_db, -15.0)

    def test_unknown_resampler(self):
        with self.assertRaises(ValueError):
            client.to_narrowband(self._tone(1000.0), "cubic")

    def test_empty_and_short_input(self):
        for mode in ("sinc", "linear"):
            with self.subTest(mode):
                self.assertEqual(client.to_narrowband(np.zeros(0, dtype=np.float32), mode).size, 0)
                self.assertEqual(client.to_narrowband(np.zeros(10, dtype=np.float32), mode).size, 10)


if __name__ == "__main__":
    unittest.main()
