import logging
import unittest

from nemotron_testbed.metrics import (
    AUDIO_PREPROC_BACKEND,
    AUDIO_PREPROC_BACKEND_AUDX,
    AUDIO_PREPROC_FALLBACK,
    AUDIO_PREPROC_LATE_SECONDS,
    AUDIO_PREPROC_POOL_WORKERS,
    AUDIO_PREPROC_QUEUE,
    AUDIO_PREPROC_QUEUE_WAIT_SECONDS,
    AUDIO_PREPROC_REGIONS,
    AUDIO_PREPROC_SLACK_SECONDS,
    AUDIO_PREPROC_STATE_RESETS,
    AUDIO_PREPROC_STREAMS_ACTIVE,
    AUDIO_PREPROC_WINDOW_SECONDS,
    INGRESS_CHUNKS,
    INGRESS_PEAK,
    VAD_MAX_SCORE,
    VAD_SCORE,
    VAD_STARTS_TOTAL,
    IntervalMax,
)
from nemotron_testbed.stats_logger import StatsLogger


class IntervalMaxTest(unittest.TestCase):
    def test_take_returns_max_then_resets(self):
        tracker = IntervalMax()
        self.assertIsNone(tracker.take())
        tracker.observe(0.2)
        tracker.observe(0.7)
        tracker.observe(0.4)
        self.assertEqual(tracker.take(), 0.7)
        self.assertIsNone(tracker.take())


class StatsIngressVadTest(unittest.TestCase):
    def _tick(self, stats_logger):
        with self.assertLogs("nemotron_testbed.stats_logger", level="INFO") as cm:
            stats_logger._tick()
        return cm.records[-1].stats

    def test_reports_chunks_peak_and_vad_signals(self):
        stats_logger = StatsLogger(interval_s=10)
        self._tick(stats_logger)

        INGRESS_CHUNKS.inc(3)
        INGRESS_PEAK.observe(0.05)
        INGRESS_PEAK.observe(0.1)
        VAD_STARTS_TOTAL.inc()
        VAD_SCORE.observe(0.4)
        VAD_MAX_SCORE.observe(0.4)
        VAD_MAX_SCORE.observe(0.7)
        stats = self._tick(stats_logger)

        self.assertEqual(stats["ingress"]["chunks"], 3)
        self.assertEqual(stats["ingress"]["peak_dbfs"], -20.0)
        self.assertEqual(stats["vad"]["starts"], 1)
        self.assertEqual(stats["vad"]["score_mean"], 0.4)
        self.assertEqual(stats["vad"]["max_score"], 0.7)

    def test_idle_interval_reports_none_for_peaks(self):
        stats_logger = StatsLogger(interval_s=10)
        self._tick(stats_logger)
        stats = self._tick(stats_logger)

        self.assertEqual(stats["ingress"]["chunks"], 0)
        self.assertIsNone(stats["ingress"]["peak_dbfs"])
        self.assertEqual(stats["vad"]["starts"], 0)
        self.assertIsNone(stats["vad"]["max_score"])
        self.assertIsNone(stats["vad"]["score_mean"])


class StatsAudioPreprocessingTest(unittest.TestCase):
    def setUp(self):
        # STATS adds every backend; other tests leave the gauges of both backends behind
        for backend in (AUDIO_PREPROC_BACKEND, AUDIO_PREPROC_BACKEND_AUDX):
            AUDIO_PREPROC_POOL_WORKERS.labels(backend=backend).set(0)
            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=backend).set(0)

    def _tick(self, stats_logger):
        with self.assertLogs("nemotron_testbed.stats_logger", level="INFO") as cm:
            stats_logger._tick()
        return cm.records[-1].stats

    def test_section_absent_while_pool_is_not_running(self):
        AUDIO_PREPROC_POOL_WORKERS.labels(backend=AUDIO_PREPROC_BACKEND).set(0)
        stats = self._tick(StatsLogger(interval_s=10))
        self.assertNotIn("audio_preprocessing", stats)

    def test_section_reports_interval_deltas(self):
        backend = AUDIO_PREPROC_BACKEND
        AUDIO_PREPROC_POOL_WORKERS.labels(backend=backend).set(2)
        try:
            stats_logger = StatsLogger(interval_s=10)
            self._tick(stats_logger)

            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=backend).set(3)
            AUDIO_PREPROC_QUEUE.labels(worker="0").set(1)
            AUDIO_PREPROC_REGIONS.labels(backend=backend, outcome="enhanced").inc(5)
            AUDIO_PREPROC_REGIONS.labels(backend=backend, outcome="late").inc(1)
            AUDIO_PREPROC_FALLBACK.labels(backend=backend, reason="DENOISER_LATE").inc(1)
            AUDIO_PREPROC_WINDOW_SECONDS.labels(backend=backend).observe(0.2)
            AUDIO_PREPROC_WINDOW_SECONDS.labels(backend=backend).observe(0.4)
            AUDIO_PREPROC_QUEUE_WAIT_SECONDS.labels(backend=backend).observe(0.01)
            AUDIO_PREPROC_LATE_SECONDS.labels(backend=backend).observe(0.05)
            AUDIO_PREPROC_LATE_SECONDS.labels(backend=backend).observe(0.15)
            AUDIO_PREPROC_SLACK_SECONDS.labels(backend=backend).observe(0.3)
            stats = self._tick(stats_logger)["audio_preprocessing"]

            self.assertEqual(stats["streams"], 3)
            self.assertEqual(stats["queue"]["0"], 1)
            self.assertEqual(stats["regions"]["enhanced"], 5)
            self.assertEqual(stats["regions"]["late"], 1)
            self.assertEqual(stats["fallback"]["DENOISER_LATE"], 1)
            self.assertAlmostEqual(stats["window_mean_ms"], 300.0, places=1)
            self.assertAlmostEqual(stats["queue_wait_mean_ms"], 10.0, places=1)
            self.assertIsNone(stats["flush_mean_ms"])
            self.assertAlmostEqual(stats["late_mean_ms"], 100.0, places=1)
            self.assertAlmostEqual(stats["slack_mean_ms"], 300.0, places=1)

            idle = self._tick(stats_logger)["audio_preprocessing"]
            self.assertEqual(idle["regions"]["enhanced"], 0)
            self.assertIsNone(idle["window_mean_ms"])
            self.assertIsNone(idle["late_mean_ms"])
            self.assertIsNone(idle["slack_mean_ms"])
        finally:
            AUDIO_PREPROC_POOL_WORKERS.labels(backend=backend).set(0)


    def test_section_reports_the_audx_backend_and_state_resets(self):
        audx = AUDIO_PREPROC_BACKEND_AUDX
        AUDIO_PREPROC_POOL_WORKERS.labels(backend=audx).set(1)
        try:
            stats_logger = StatsLogger(interval_s=10)
            self._tick(stats_logger)
            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=audx).set(2)
            AUDIO_PREPROC_STATE_RESETS.labels(backend=audx, reason="error").inc(2)
            AUDIO_PREPROC_WINDOW_SECONDS.labels(backend=audx).observe(0.004)
            stats = self._tick(stats_logger)["audio_preprocessing"]
            self.assertEqual(stats["streams"], 2)
            self.assertEqual(stats["state_resets"]["error"], 2)
            self.assertAlmostEqual(stats["window_mean_ms"], 4.0, places=1)
        finally:
            AUDIO_PREPROC_POOL_WORKERS.labels(backend=audx).set(0)
            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=audx).set(0)


if __name__ == "__main__":
    unittest.main()
