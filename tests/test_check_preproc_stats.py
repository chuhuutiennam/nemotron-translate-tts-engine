import contextlib
import importlib.util
import io
import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_preproc_stats.py"
_spec = importlib.util.spec_from_file_location("check_preproc_stats", _SCRIPT)
cps = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cps  # dataclasses look the module up while it loads
_spec.loader.exec_module(cps)

T0 = datetime(2026, 10, 31, 8, 0, tzinfo=timezone.utc)


def stats_line(offset_s: float, stats: dict, message: str = "STATS") -> str:
    record = {"timestamp": (T0 + timedelta(seconds=offset_s)).isoformat(), "level": "INFO", "message": message,
              "stats": stats}
    return json.dumps(record)


def preproc(regions, fallback=None, resets=None, window=None, slack=None, streams=1, bypass=0, shed=0):
    block = {"streams": streams, "queue": {"0": 0}, "regions": regions, "fallback": fallback or {},
             "bypass": bypass, "shed_streams": shed, "state_resets": resets or {}, "window_mean_ms": window,
             "queue_wait_mean_ms": None, "flush_mean_ms": None, "late_mean_ms": None, "slack_mean_ms": slack}
    return block


class CheckPreprocStatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "nemotron-engine.log"

    def write(self, lines, path=None):
        (path or self.log).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cps.main([str(self.log), *args])
        return code, out.getvalue(), err.getvalue()

    def test_sums_the_intervals_and_passes_under_the_thresholds(self):
        self.write([
            stats_line(0, {"audio_preprocessing": preproc({"enhanced": 500}, window=5.0, slack=110.0, streams=2),
                           "vad": {"utterances": 10, "empty_asr": 1, "short": 2},
                           "ingress": {"finals": 10, "peak_dbfs": -9.0}, "asr": {"confidence_mean": 0.9},
                           "gpu": {"1": {"util_pct": 20.0, "mem_used_mb": 900.0}}}),
            '{"message": "something else", "stats": {}}',
            "not json at all, and no STATS",
            stats_line(10, {"audio_preprocessing": preproc({"enhanced": 499, "late": 1}, {"AUDX_LATE": 1},
                                                           window=9.0, slack=90.0, streams=4),
                            "vad": {"utterances": 5, "empty_asr": 0, "short": 0},
                            "ingress": {"finals": 30, "peak_dbfs": -6.5}, "asr": {"confidence_mean": 0.8},
                            "gpu": {"1": {"util_pct": 40.0, "mem_used_mb": 1100.0}}}),
        ])
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        s = cps.summarise(cps.select(cps.read_stats([str(self.log)])))
        self.assertEqual(s.lines, 2)
        self.assertEqual(dict(s.regions), {"enhanced": 999, "late": 1})
        self.assertAlmostEqual(s.raw_pct, 0.1)
        self.assertEqual(dict(s.fallback), {"AUDX_LATE": 1})
        self.assertAlmostEqual(s.mean_time("window_mean_ms"), (5.0 * 500 + 9.0 * 500) / 1000)
        self.assertEqual((s.streams_peak, s.vad_utterances, s.vad_empty, s.vad_short), (4, 15, 1, 2))
        self.assertEqual((s.finals, s.peak_dbfs), (40, -6.5))
        self.assertAlmostEqual(s.confidence[0] / s.confidence[1], (0.9 * 10 + 0.8 * 30) / 40)
        self.assertEqual(s.gpu["1"]["mem_peak"], 1100.0)
        self.assertIn("played raw (late+shed+error+bypass): 1 of 1000 (0.10%)", out)
        self.assertIn("AUDX_LATE", out)

    def test_raw_share_or_state_resets_above_the_limit_fail(self):
        self.write([stats_line(0, {"audio_preprocessing": preproc({"enhanced": 90, "late": 5, "error": 5},
                                                                  resets={"error": 2})})])
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("raw share 10.00% > --max-raw-pct 1.0", err)
        self.assertIn("state resets 2 > --max-resets 0", err)
        self.assertEqual(self.run_main("--max-raw-pct", "20", "--max-resets", "5")[0], 0)

    def test_time_range_filters(self):
        self.write([stats_line(i * 600, {"audio_preprocessing": preproc({"enhanced": 1 + i})}) for i in range(6)])
        rows = cps.read_stats([str(self.log)])
        pick = lambda **kw: [sum(s["audio_preprocessing"]["regions"].values()) for _, s in cps.select(rows, **kw)]
        # counted back from the newest line (08:50), not from now; the boundary 08:30 is included
        self.assertEqual(pick(last_minutes=20), [4, 5, 6])
        self.assertEqual(pick(since=cps.parse_time("2026-10-31T08:10"), until=cps.parse_time("2026-10-31T08:30")),
                         [2, 3, 4])
        self.assertEqual(pick(since=cps.parse_time("2026-10-31T08:45:00+00:00"), last_minutes=40), [6])
        self.assertEqual(cps.parse_time("2026-10-31T15:00+07:00"), T0)

    def test_rotated_files_are_read_and_sorted_by_time(self):
        rotated = Path(str(self.log) + ".2026-10-30")
        self.write([stats_line(-10, {"audio_preprocessing": preproc({"enhanced": 1})})], rotated)
        self.write([stats_line(0, {"audio_preprocessing": preproc({"enhanced": 2})})])
        files = cps.expand_paths([str(self.log)])
        self.assertEqual(files, [str(rotated), str(self.log)])
        rows = cps.read_stats(files)
        self.assertEqual([s["audio_preprocessing"]["regions"]["enhanced"] for _, s in rows], [1, 2])

    def test_missing_data_exits_1(self):
        code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("No log file matches", err)
        self.write(['{"message": "RPC opened"}'])
        code, _, err = self.run_main()
        self.assertEqual((code, "No STATS line" in err), (1, True))
        self.write([stats_line(0, {"vad": {"utterances": 3}})])
        code, out, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("utterances 3", out)  # VAD/ASR still reported
        self.assertIn("pool was not running", err)

    def test_json_output(self):
        self.write([stats_line(0, {"audio_preprocessing": preproc({"enhanced": 10}, window=4.0)})])
        code, out, _ = self.run_main("--json")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual((data["regions"], data["raw_regions"], data["times_ms"]["window_mean_ms"]),
                         ({"enhanced": 10}, 0, 4.0))

    def test_reads_real_stats_lines(self):
        from nemotron_testbed.logging_config import JsonFormatter
        from nemotron_testbed.metrics import (
            AUDIO_PREPROC_BACKEND,
            AUDIO_PREPROC_BACKEND_AUDX,
            AUDIO_PREPROC_FALLBACK,
            AUDIO_PREPROC_POOL_WORKERS,
            AUDIO_PREPROC_REGIONS,
            AUDIO_PREPROC_STATE_RESETS,
            AUDIO_PREPROC_STREAMS_ACTIVE,
            AUDIO_PREPROC_WINDOW_SECONDS,
            ASR_CONFIDENCE,
            FINAL_RESULTS,
        )
        from nemotron_testbed.stats_logger import StatsLogger

        audx = AUDIO_PREPROC_BACKEND_AUDX
        for backend in (AUDIO_PREPROC_BACKEND, audx):
            AUDIO_PREPROC_POOL_WORKERS.labels(backend=backend).set(0)
            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=backend).set(0)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        stats_log = logging.getLogger("nemotron_testbed.stats_logger")
        old_level = stats_log.level
        stats_log.addHandler(handler)
        stats_log.setLevel(logging.INFO)
        AUDIO_PREPROC_POOL_WORKERS.labels(backend=audx).set(1)
        try:
            stats_logger = StatsLogger(interval_s=10)
            stats_logger._tick()  # the first line holds totals since start (other tests too): dropped
            stream.truncate(0)
            stream.seek(0)
            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=audx).set(3)
            AUDIO_PREPROC_REGIONS.labels(backend=audx, outcome="enhanced").inc(40)
            AUDIO_PREPROC_REGIONS.labels(backend=audx, outcome="late").inc(2)
            AUDIO_PREPROC_FALLBACK.labels(backend=audx, reason="AUDX_LATE").inc(2)
            AUDIO_PREPROC_STATE_RESETS.labels(backend=audx, reason="shed").inc(1)
            AUDIO_PREPROC_WINDOW_SECONDS.labels(backend=audx).observe(0.006)
            FINAL_RESULTS.inc(4)
            for value in (0.7, 0.8, 0.9, 1.0):
                ASR_CONFIDENCE.observe(value)
            stats_logger._tick()
        finally:
            stats_log.removeHandler(handler)
            stats_log.setLevel(old_level)
            AUDIO_PREPROC_POOL_WORKERS.labels(backend=audx).set(0)
            AUDIO_PREPROC_STREAMS_ACTIVE.labels(backend=audx).set(0)
        self.write(stream.getvalue().splitlines())
        s = cps.summarise(cps.read_stats([str(self.log)]))
        self.assertEqual(s.preproc_lines, 1)
        self.assertEqual((s.regions["enhanced"], s.regions["late"]), (40, 2))
        self.assertEqual((dict(s.fallback)["AUDX_LATE"], dict(s.resets)["shed"], s.streams_peak), (2, 1, 3))
        self.assertAlmostEqual(s.mean_time("window_mean_ms"), 6.0, places=3)
        # finals live in the ingress block of a real STATS line, confidence in the asr block
        self.assertEqual(s.finals, 4)
        self.assertAlmostEqual(s.confidence[0] / s.confidence[1], 0.85, places=6)
        self.assertEqual(self.run_main()[0], 2)  # 2 late of 42 is above 1 %, and one reset


if __name__ == "__main__":
    unittest.main()
