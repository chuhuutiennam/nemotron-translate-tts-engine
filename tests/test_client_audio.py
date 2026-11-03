import importlib.util
import unittest
from pathlib import Path

import numpy as np

from nemotron_testbed.client_audio import (
    RtpTimeline,
    chunk_ms_warning,
    parse_rtp_start,
    pcm16_bytes,
)
from nemotron_testbed.vad import VadController


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RtpTimelineTest(unittest.TestCase):
    def test_rtpgw_80ms_chunks_advance_640_ticks_from_start(self):
        timeline = RtpTimeline(16_000, 8_000, start=1000)
        stamps = [timeline.next_timestamp(1280) for _ in range(3)]
        self.assertEqual(stamps, [1000, 1640, 2280])

    def test_first_timestamp_is_the_start(self):
        self.assertEqual(RtpTimeline(16_000, 8_000, start=42).next_timestamp(320), 42)

    def test_rounding_does_not_accumulate(self):
        # 3 samples per chunk at a 1:2 clock is 1.5 ticks: per-chunk rounding would drift, cumulative does not
        timeline = RtpTimeline(16_000, 8_000)
        last = [timeline.next_timestamp(3) for _ in range(1000)][-1]
        self.assertEqual(last, round(999 * 3 * 8_000 / 16_000))

    def test_wraps_at_32_bits(self):
        timeline = RtpTimeline(16_000, 16_000, start=0xFFFFFFF0)
        timeline.next_timestamp(32)
        self.assertEqual(timeline.next_timestamp(32), 0x10)


class ParseRtpStartTest(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(parse_rtp_start("12345"), 12345)

    def test_random_is_a_32_bit_value(self):
        values = {parse_rtp_start("random") for _ in range(8)}
        self.assertGreater(len(values), 1)
        self.assertTrue(all(0 <= v <= 0xFFFFFFFF for v in values))

    def test_integer_wraps_to_32_bits(self):
        self.assertEqual(parse_rtp_start(str(2**32 + 5)), 5)


class ChunkWarningTest(unittest.TestCase):
    def test_80ms_and_smaller_are_fine(self):
        self.assertIsNone(chunk_ms_warning(80))
        self.assertIsNone(chunk_ms_warning(20))

    def test_100ms_warns(self):
        self.assertIn("100", chunk_ms_warning(100))

    def test_500ms_warns(self):
        self.assertIn("500", chunk_ms_warning(500))


class Pcm16Test(unittest.TestCase):
    def test_is_mono_int16_little_endian(self):
        payload = pcm16_bytes(np.array([0.0, 0.5, -0.5], dtype=np.float32))
        self.assertEqual(np.frombuffer(payload, dtype="<i2").tolist(), [0, 16384, -16384])


class _WindowedProbability:
    needs_executor = False

    def __init__(self, scores_by_window):
        self.scores = iter(scores_by_window)
        self.window = 512
        self.pending = 0

    def score(self, audio):
        return 0.0

    def frame_scores(self, audio, duration_ms):
        self.pending += audio.size
        out = []
        while self.pending >= self.window:
            self.pending -= self.window
            out.append((next(self.scores, 0.0), 32.0))
        return out or [(0.0, 0.0)]

    def reset(self):
        self.pending = 0


class ProbeVadRecorderTest(unittest.TestCase):
    def test_recorder_keeps_every_32ms_window_not_the_chunk_max(self):
        probe = _load_script("probe_vad")
        recorder = probe._ScoreRecorder(_WindowedProbability([0.1, 0.9, 0.2, 0.8]))
        controller = VadController(recorder, threshold=0.5, min_speech_ms=64, end_silence_ms=64)
        controller.process(np.zeros(2048, dtype=np.float32), 128.0)
        self.assertEqual(recorder.scores, [0.1, 0.9, 0.2, 0.8])

    def test_zero_length_frames_are_not_recorded(self):
        probe = _load_script("probe_vad")
        recorder = probe._ScoreRecorder(_WindowedProbability([]))
        controller = VadController(recorder, threshold=0.5, min_speech_ms=64, end_silence_ms=64)
        controller.process(np.zeros(160, dtype=np.float32), 10.0)
        self.assertEqual(recorder.scores, [])


if __name__ == "__main__":
    unittest.main()
