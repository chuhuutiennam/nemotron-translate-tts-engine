import unittest

import numpy as np

from nemotron_testbed.metrics import VAD_SCORE, VAD_STARTS_TOTAL
from nemotron_testbed.vad import EnergySpeechProbability, VadController, VadEvent


def _score_histogram():
    from nemotron_testbed.stats_logger import _histogram

    return _histogram(VAD_SCORE)


class WindowedProbability:
    """Stub with Silero-like 32 ms windows: 1.0 means speech, 0.0 silence."""

    needs_executor = False

    def __init__(self, windows_per_chunk: int = 3, window_ms: float = 32.0):
        self.pending_ms = 0.0
        self.queue: list[float] = []
        self.windows_per_chunk = windows_per_chunk
        self.window_ms = window_ms

    def push(self, *scores: float) -> np.ndarray:
        self.queue.extend(scores)
        return np.zeros(1, dtype=np.float32)

    def frame_scores(self, audio, duration_ms):
        frames = [(score, self.window_ms) for score in self.queue]
        self.queue = []
        return frames

    def score(self, audio):
        raise AssertionError("controller must use frame_scores")

    def reset(self):
        self.queue = []


def _windowed_controller(end_silence_ms: int, neg_threshold: float | None = None):
    probability = WindowedProbability()
    controller = VadController(
        probability=probability,
        threshold=0.5,
        min_speech_ms=40,
        end_silence_ms=end_silence_ms,
        enabled=True,
        neg_threshold=neg_threshold,
    )
    controller.process(probability.push(1.0, 1.0, 1.0), 100)
    assert controller.active
    return controller, probability


class WindowedEndTest(unittest.TestCase):
    def test_end_fires_after_enough_silent_windows_inside_a_chunk(self):
        controller, p = _windowed_controller(end_silence_ms=250)
        self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.SPEECH)
        self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.SPEECH)
        self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.END)

    def test_threshold_is_not_rounded_to_whole_chunks(self):
        # 8 windows = 256 ms >= 250 ms, reached in the third chunk; 200 ms needs 7 windows
        for end_ms, windows_needed in ((250, 8), (200, 7), (300, 10)):
            controller, p = _windowed_controller(end_silence_ms=end_ms)
            events = [VadEvent.SPEECH]
            fed = 0
            while events[-1] is not VadEvent.END and fed < 30:
                events.append(controller.process(p.push(0, 0, 0), 100))
                fed += 3
            self.assertEqual(fed, ((windows_needed + 2) // 3) * 3, end_ms)

    def test_speech_window_at_chunk_end_restarts_the_count(self):
        controller, p = _windowed_controller(end_silence_ms=300)
        controller.process(p.push(0, 0, 0), 100)
        controller.process(p.push(0, 0, 0), 100)
        # 8 silent windows (256 ms) so far; one speech window resets the count
        self.assertEqual(controller.process(p.push(0, 0, 1.0), 100), VadEvent.SPEECH)
        for _ in range(3):
            self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.SPEECH)
        self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.END)

    def test_silence_before_a_speech_window_in_the_same_chunk_is_reset(self):
        controller, p = _windowed_controller(end_silence_ms=96)
        self.assertEqual(controller.process(p.push(0, 0, 1.0), 100), VadEvent.SPEECH)
        self.assertEqual(controller.silence_ms, 0.0)
        self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.END)

    def test_windows_split_across_chunks_accumulate(self):
        controller, p = _windowed_controller(end_silence_ms=96)
        controller.process(p.push(0, 0), 100)
        self.assertEqual(controller.silence_ms, 64.0)
        self.assertEqual(controller.process(p.push(0), 100), VadEvent.END)

    def _start_controller(self, min_speech_ms: int):
        probability = WindowedProbability()
        controller = VadController(
            probability=probability, threshold=0.5, min_speech_ms=min_speech_ms, end_silence_ms=300, enabled=True
        )
        return controller, probability

    def test_single_speech_window_in_a_large_chunk_does_not_start(self):
        controller, p = self._start_controller(250)
        scores = [0.0] * 7 + [0.9] + [0.0] * 7
        self.assertEqual(controller.process(p.push(*scores), 480), VadEvent.NONE)
        self.assertFalse(controller.active)
        self.assertEqual(controller.speech_candidate_ms, 0.0)

    def test_start_needs_min_speech_of_consecutive_windows(self):
        controller, p = self._start_controller(250)
        # 7 windows = 224 ms < 250 ms, the 8th (256 ms) fires START
        self.assertEqual(controller.process(p.push(*[1.0] * 7), 224), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 224.0)
        self.assertEqual(controller.process(p.push(1.0), 32), VadEvent.START)
        self.assertEqual(controller.speech_candidate_ms, 256.0)

    def test_speech_window_gap_resets_the_start_count(self):
        controller, p = self._start_controller(128)
        self.assertEqual(controller.process(p.push(1.0, 1.0, 1.0, 0.0, 1.0, 1.0), 192), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 64.0)
        self.assertEqual(controller.process(p.push(1.0, 1.0), 64), VadEvent.START)

    def test_start_count_carries_over_chunk_boundaries(self):
        controller, p = self._start_controller(128)
        self.assertEqual(controller.process(p.push(0.0, 1.0, 1.0), 96), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 64.0)
        self.assertEqual(controller.process(p.push(1.0, 1.0), 64), VadEvent.START)

    def test_trailing_silence_in_the_start_chunk_resets_the_count(self):
        controller, p = self._start_controller(128)
        self.assertEqual(controller.process(p.push(1.0, 1.0, 0.0), 96), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 0.0)

    def test_zero_length_frame_is_ignored_by_start(self):
        controller, p = self._start_controller(64)
        controller.process(p.push(1.0), 32)
        self.assertEqual(controller.speech_candidate_ms, 32.0)
        p.frame_scores = lambda audio, duration_ms: [(0.0, 0.0)]
        self.assertEqual(controller.process(np.zeros(1, dtype=np.float32), 10), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 32.0)

    def test_start_delay_metric_observes_real_speech_duration(self):
        from nemotron_testbed.metrics import VAD_SPEECH_START_DELAY_MS
        from nemotron_testbed.stats_logger import _histogram

        before_sum, before_count = _histogram(VAD_SPEECH_START_DELAY_MS)
        controller, p = self._start_controller(250)
        scores = [0.0] * 4 + [1.0] * 8 + [0.0] * 3
        self.assertEqual(controller.process(p.push(*scores), 480), VadEvent.START)
        total, count = _histogram(VAD_SPEECH_START_DELAY_MS)
        self.assertEqual(count - before_count, 1)
        self.assertEqual(total - before_sum, 256.0)

    def test_end_tail_counts_windows_after_the_end_window(self):
        controller, p = _windowed_controller(end_silence_ms=64)
        self.assertEqual(controller.process(p.push(0, 0, 1.0), 100), VadEvent.END)
        self.assertEqual(controller.end_tail_ms, 32.0)

    def test_end_tail_includes_unscored_pending_audio(self):
        controller, p = _windowed_controller(end_silence_ms=64)
        p.pending_ms = 10.0
        self.assertEqual(controller.process(p.push(0, 0), 100), VadEvent.END)
        self.assertEqual(controller.end_tail_ms, 10.0)

    def test_end_tail_is_zero_when_end_is_on_the_last_window_and_cleared_next_chunk(self):
        controller, p = _windowed_controller(end_silence_ms=96)
        self.assertEqual(controller.process(p.push(0, 0, 0), 100), VadEvent.END)
        self.assertEqual(controller.end_tail_ms, 0.0)
        controller, p = _windowed_controller(end_silence_ms=64)
        controller.process(p.push(0, 0, 1.0), 100)
        self.assertEqual(controller.end_tail_ms, 32.0)
        controller.process(p.push(1.0, 1.0, 1.0), 100)
        self.assertEqual(controller.end_tail_ms, 0.0)

    def test_end_silence_observed_is_the_silence_at_end_and_cleared_next_chunk(self):
        controller, p = _windowed_controller(end_silence_ms=64)
        self.assertEqual(controller.process(p.push(0, 0, 1.0), 100), VadEvent.END)
        self.assertEqual(controller.end_silence_observed_ms, 64.0)
        controller.process(p.push(1.0), 100)
        self.assertEqual(controller.end_silence_observed_ms, 0.0)

    def test_end_silence_observed_is_zero_without_end(self):
        controller, p = _windowed_controller(end_silence_ms=300)
        controller.process(p.push(0, 0, 0), 100)
        self.assertEqual(controller.end_silence_observed_ms, 0.0)

    def test_chunk_size_does_not_change_the_end_point(self):
        for chunk_windows in (1, 3, 5):
            probability = WindowedProbability()
            controller = VadController(
                probability=probability, threshold=0.5, min_speech_ms=32, end_silence_ms=250, enabled=True
            )
            controller.process(probability.push(1.0), 32)
            silent = 0
            event = VadEvent.SPEECH
            while event is not VadEvent.END:
                event = controller.process(probability.push(*([0.0] * chunk_windows)), 32 * chunk_windows)
                silent += chunk_windows
            self.assertGreaterEqual(silent, 8)
            self.assertLess(silent, 8 + chunk_windows)


class HysteresisTest(unittest.TestCase):
    def test_neg_threshold_defaults_to_threshold(self):
        controller, _ = _windowed_controller(end_silence_ms=96)
        self.assertEqual(controller.neg_threshold, 0.5)

    def test_without_dead_band_a_mid_score_counts_as_silence(self):
        controller, p = _windowed_controller(end_silence_ms=96)
        self.assertEqual(controller.process(p.push(0.4, 0.4, 0.4), 100), VadEvent.END)

    def test_dead_band_scores_never_end_an_utterance(self):
        controller, p = _windowed_controller(end_silence_ms=96, neg_threshold=0.35)
        for _ in range(10):
            self.assertEqual(controller.process(p.push(0.4, 0.4, 0.4), 100), VadEvent.SPEECH)
        self.assertEqual(controller.silence_ms, 0.0)

    def test_dead_band_window_neither_adds_to_nor_resets_the_silence_run(self):
        controller, p = _windowed_controller(end_silence_ms=96, neg_threshold=0.35)
        self.assertEqual(controller.process(p.push(0.1, 0.4), 100), VadEvent.SPEECH)
        self.assertEqual(controller.silence_ms, 32.0)
        self.assertEqual(controller.process(p.push(0.1, 0.4, 0.1), 100), VadEvent.END)
        self.assertEqual(controller.end_silence_observed_ms, 96.0)

    def test_score_at_neg_threshold_is_dead_band_and_just_below_is_silence(self):
        controller, p = _windowed_controller(end_silence_ms=32, neg_threshold=0.35)
        self.assertEqual(controller.process(p.push(0.35), 32), VadEvent.SPEECH)
        self.assertEqual(controller.process(p.push(0.349), 32), VadEvent.END)

    def test_speech_window_resets_silence_even_with_dead_band(self):
        controller, p = _windowed_controller(end_silence_ms=96, neg_threshold=0.35)
        controller.process(p.push(0.1, 0.1), 100)
        self.assertEqual(controller.process(p.push(0.6), 32), VadEvent.SPEECH)
        self.assertEqual(controller.silence_ms, 0.0)

    def _start_controller(self, min_speech_ms: int, neg_threshold: float | None):
        probability = WindowedProbability()
        controller = VadController(
            probability=probability,
            threshold=0.5,
            min_speech_ms=min_speech_ms,
            end_silence_ms=300,
            enabled=True,
            neg_threshold=neg_threshold,
        )
        return controller, probability

    def test_dead_band_window_neither_adds_to_nor_resets_the_start_count(self):
        controller, p = self._start_controller(128, neg_threshold=0.35)
        self.assertEqual(controller.process(p.push(1.0, 1.0, 0.4), 96), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 64.0)
        self.assertEqual(controller.process(p.push(1.0, 1.0), 64), VadEvent.START)
        self.assertEqual(controller.speech_candidate_ms, 128.0)

    def test_window_below_neg_threshold_still_resets_the_start_count(self):
        controller, p = self._start_controller(128, neg_threshold=0.35)
        self.assertEqual(controller.process(p.push(1.0, 1.0, 0.2, 1.0, 1.0), 160), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 64.0)

    def test_without_dead_band_a_mid_score_resets_the_start_count(self):
        controller, p = self._start_controller(128, neg_threshold=None)
        self.assertEqual(controller.process(p.push(1.0, 1.0, 0.4, 1.0, 1.0), 160), VadEvent.NONE)
        self.assertEqual(controller.speech_candidate_ms, 64.0)


class _FakeTensor:
    def __init__(self, array):
        self.array = array

    def to(self, _device):
        return self


class _FakeTorch:
    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)

    @staticmethod
    def inference_mode():
        import contextlib

        return contextlib.nullcontext()


class _FakeModel:
    def __call__(self, tensor, _rate):
        class _Value:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        return _Value(float(np.abs(tensor.array).max() > 0.01))


class SileroAdapterTest(unittest.TestCase):
    def _adapter(self):
        import threading

        from nemotron_testbed.vad import SileroSpeechProbability

        adapter = SileroSpeechProbability.__new__(SileroSpeechProbability)
        adapter._torch = _FakeTorch()
        adapter._device = "cpu"
        adapter.sample_rate = 16_000
        adapter.window_samples = 512
        adapter._pending = np.empty(0, dtype=np.float32)
        adapter._last_score = 0.0
        adapter._lock = threading.Lock()
        adapter.model = _FakeModel()
        return adapter

    def test_chunk_of_1600_samples_yields_three_32ms_windows_and_carries_the_rest(self):
        adapter = self._adapter()
        chunk = np.zeros(1600, dtype=np.float32)
        chunk[1100:] = 0.5
        frames = adapter.frame_scores(chunk, 100.0)
        self.assertEqual([score for score, _ in frames], [0.0, 0.0, 1.0])
        self.assertTrue(all(ms == 32.0 for _, ms in frames))
        self.assertEqual(adapter._pending.size, 64)

    def test_chunk_smaller_than_a_window_returns_last_score_with_zero_time(self):
        adapter = self._adapter()
        adapter._last_score = 1.0
        self.assertEqual(adapter.frame_scores(np.zeros(320, dtype=np.float32), 20.0), [(1.0, 0.0)])

    def test_pending_ms_reports_the_unscored_remainder(self):
        adapter = self._adapter()
        self.assertEqual(adapter.pending_ms, 0.0)
        adapter.frame_scores(np.zeros(1600, dtype=np.float32), 100.0)
        self.assertEqual(adapter.pending_ms, 4.0)

    def test_score_still_returns_chunk_max(self):
        adapter = self._adapter()
        chunk = np.zeros(1600, dtype=np.float32)
        chunk[1100:] = 0.5
        self.assertEqual(adapter.score(chunk), 1.0)


class VadTest(unittest.TestCase):
    def test_start_and_end(self):
        controller = VadController(
            probability=EnergySpeechProbability(db_threshold=-35),
            threshold=0.5,
            min_speech_ms=40,
            end_silence_ms=60,
            enabled=True,
        )
        speech = np.full(320, 0.2, dtype=np.float32)
        silence = np.zeros(320, dtype=np.float32)

        self.assertEqual(controller.process(speech, 20), VadEvent.NONE)
        self.assertEqual(controller.process(speech, 20), VadEvent.START)
        self.assertEqual(controller.process(speech, 20), VadEvent.SPEECH)
        self.assertEqual(controller.process(silence, 20), VadEvent.SPEECH)
        self.assertEqual(controller.process(silence, 20), VadEvent.SPEECH)
        self.assertEqual(controller.process(silence, 20), VadEvent.END)

    def test_records_starts_and_scores(self):
        controller = VadController(
            probability=EnergySpeechProbability(db_threshold=-35),
            threshold=0.5,
            min_speech_ms=40,
            end_silence_ms=60,
            enabled=True,
        )
        silence = np.zeros(320, dtype=np.float32)
        speech = np.full(320, 0.2, dtype=np.float32)
        starts_before = VAD_STARTS_TOTAL._value.get()
        _, count_before = _score_histogram()

        controller.process(silence, 20)
        self.assertLess(controller.max_score, 0.01)
        controller.process(speech, 20)
        controller.process(speech, 20)

        self.assertGreater(controller.max_score, 0.9)
        self.assertEqual(VAD_STARTS_TOTAL._value.get() - starts_before, 1)
        self.assertEqual(_score_histogram()[1] - count_before, 3)

    def test_disabled_vad_counts_single_start(self):
        controller = VadController(
            probability=EnergySpeechProbability(),
            threshold=0.5,
            min_speech_ms=40,
            end_silence_ms=60,
            enabled=False,
        )
        starts_before = VAD_STARTS_TOTAL._value.get()
        controller.process(np.zeros(320, dtype=np.float32), 20)
        controller.process(np.zeros(320, dtype=np.float32), 20)
        self.assertEqual(VAD_STARTS_TOTAL._value.get() - starts_before, 1)


if __name__ == "__main__":
    unittest.main()
