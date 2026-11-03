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

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stats_phases.py"
_spec = importlib.util.spec_from_file_location("stats_phases", _SCRIPT)
sp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sp  # dataclasses look the module up while it loads
_spec.loader.exec_module(sp)

T0 = datetime(2026, 10, 31, 8, 0, tzinfo=timezone.utc)


def line(offset_s: float, stats: dict) -> str:
    return json.dumps({"timestamp": (T0 + timedelta(seconds=offset_s)).isoformat(), "message": "STATS",
                       "stats": {"interval_s": 10.0, **stats}})


class StatsPhasesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "nemotron-engine.log"

    def write(self, lines):
        self.log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = sp.main([str(self.log), *args])
        return code, out.getvalue(), err.getvalue()

    def two_intervals(self):
        return [
            line(0, {
                "active": {"streams": 2, "rpcs": 1},
                "ingress": {"audio_s": 15.0, "partials": 8, "finals": 2, "stale": 0, "peak_dbfs": -9.0,
                            "accept_chunk_mean_ms": 1.0},
                "vad": {"utterances": 4, "empty_asr": 1, "short": 0, "speech_s": 6.0, "end_delay_mean_ms": None},
                "asr": {"first_partial_mean_ms": 400.0, "confidence_source": {"unavailable": 1},
                        "model_step": {"STEP": {"mean_ms": 60.0, "n": 10}, "START": {"mean_ms": None, "n": 0}}},
                "gpu": {"1": {"util_pct": -1.0, "mem_used_mb": 900.0}},
                "audio_preprocessing": {"streams": 2, "regions": {"enhanced": 99, "late": 1}},
            }),
            line(10, {
                "active": {"streams": 4, "rpcs": 2},
                "ingress": {"audio_s": 25.0, "partials": 12, "finals": 3, "stale": 1, "peak_dbfs": -3.5,
                            "accept_chunk_mean_ms": None},
                "vad": {"utterances": 6, "empty_asr": 0, "short": 1, "speech_s": 9.0, "end_delay_mean_ms": 300.0},
                "asr": {"first_partial_mean_ms": 800.0, "confidence_source": {"unavailable": 2},
                        "model_step": {"STEP": {"mean_ms": 90.0, "n": 30}, "START": {"mean_ms": 0.5, "n": 2}}},
                "gpu": {"1": {"util_pct": 40.0, "mem_used_mb": 1100.0}},
                "audio_preprocessing": {"streams": 3, "regions": {"enhanced": 100}},
                "newphase": {"thing": 7},
            }),
        ]

    def test_each_field_is_combined_by_its_kind(self):
        self.write(self.two_intervals())
        phases, present = sp.aggregate(sp.read_stats([str(self.log)]))
        v = lambda phase, *path: phases[phase][path].value()  # noqa: E731
        self.assertEqual(v("ingress", "audio_s"), 40.0)  # counter: sum
        self.assertEqual(v("ingress", "partials"), 20.0)
        self.assertEqual(v("ingress", "peak_dbfs"), -3.5)  # max
        self.assertEqual(v("ingress", "accept_chunk_mean_ms"), 1.0)  # mean of the intervals that have one
        self.assertEqual(v("asr", "first_partial_mean_ms"), 600.0)
        self.assertEqual(phases["asr"][("first_partial_mean_ms",)].describe(),
                         "mean 600 (min 400, max 800, 2 intervals)")
        self.assertEqual(v("asr", "model_step", "STEP"), (60.0 * 10 + 90.0 * 30) / 40)  # weighted by n
        self.assertEqual(phases["asr"][("model_step", "STEP")].weight, 40)
        self.assertEqual(v("asr", "model_step", "START"), 0.5)
        self.assertEqual(v("asr", "confidence_source", "unavailable"), 3.0)  # nested counter: sum
        self.assertEqual(phases["active"][("streams",)].describe(), "max 4, mean 3")  # gauge
        self.assertEqual(phases["audio_preprocessing"][("streams",)].kind, "gauge")
        self.assertEqual(v("gpu", "1", "util_pct"), 40.0)  # -1 (unknown) ignored
        self.assertEqual(v("vad", "end_delay_mean_ms"), 300.0)
        self.assertEqual(present["newphase"], 1)
        self.assertEqual(sp.ordered(phases)[-1], "newphase")  # a block the engine adds later comes last

    def test_summary_lists_phases_in_pipeline_order_with_derived_ratios(self):
        self.write(self.two_intervals())
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        order = [out.index(f"== {p} ") for p in ("active", "ingress", "audio_preprocessing", "vad", "asr", "gpu")]
        self.assertEqual(order, sorted(order))
        self.assertIn("audio received / wall time: 2.00", out)  # 40 s of audio in 20 s
        self.assertIn("empty ASR finals: 10.0% of utterances; short: 10.0%", out)
        self.assertIn("played raw (late+shed+error+bypass): 1 of 200 regions (0.5%)", out)
        code, out, _ = self.run_main("--phase", "vad,rtp")
        self.assertEqual(code, 0)
        self.assertIn("== vad", out)
        self.assertIn("== rtp (0 of 2 lines)\n  (not in the selected lines)", out)
        self.assertNotIn("== asr", out)
        code, _, err = self.run_main("--phase", "bogus")
        self.assertEqual(code, 2)
        self.assertIn("unknown phase", err)

    def test_series_buckets_are_clock_aligned(self):
        self.write([
            line(0, {"vad": {"utterances": 1, "end_delay_mean_ms": 100.0}, "active": {"streams": 1}}),
            line(240, {"vad": {"utterances": 2, "end_delay_mean_ms": 300.0}, "active": {"streams": 5}}),
            line(330, {"vad": {"utterances": 4, "end_delay_mean_ms": 500.0}, "active": {"streams": 2}}),
        ])
        code, out, _ = self.run_main("--series", "vad", "--bucket-minutes", "5")
        self.assertEqual(code, 0)
        rows = [r.split() for r in out.splitlines()[2:]]
        self.assertEqual(rows, [["2026-10-31", "08:00", "2", "200", "3"], ["2026-10-31", "08:05", "1", "500", "4"]])
        code, out, _ = self.run_main("--series", "active", "--bucket-minutes", "10", "--fields", "streams")
        self.assertEqual(out.splitlines()[2].split()[-1], "5")  # gauge: bucket max
        self.assertEqual(self.run_main("--series", "vad", "--fields", "nope")[0], 2)
        self.assertEqual(self.run_main("--series", "bogus")[0], 2)

    def test_list_and_missing_data(self):
        self.write(self.two_intervals())
        code, out, _ = self.run_main("--list")
        self.assertEqual(code, 0)
        self.assertIn("asr (2 lines): confidence_source.unavailable, first_partial_mean_ms, model_step.START,"
                      " model_step.STEP", out)
        self.write(['{"message": "RPC opened"}'])
        self.assertEqual(self.run_main()[0], 1)
        self.log.unlink()
        self.assertEqual(self.run_main()[0], 1)

    def test_reads_every_phase_of_real_stats_lines(self):
        from nemotron_testbed.logging_config import JsonFormatter
        from nemotron_testbed.metrics import (
            AUDIO_PREPROC_BACKEND,
            AUDIO_PREPROC_BACKEND_AUDX,
            AUDIO_PREPROC_POOL_WORKERS,
            AUDIO_PREPROC_STREAMS_ACTIVE,
            INGRESS_CHUNKS,
            MODEL_STEP_SECONDS,
            VAD_UTTERANCES_TOTAL,
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
            stats_logger._tick()  # first line: totals since start (other tests too), dropped
            stream.truncate(0)
            stream.seek(0)
            INGRESS_CHUNKS.inc(50)
            VAD_UTTERANCES_TOTAL.inc(3)
            MODEL_STEP_SECONDS.labels(command="STEP").observe(0.05)
            MODEL_STEP_SECONDS.labels(command="STEP").observe(0.07)
            stats_logger._tick()
        finally:
            stats_log.removeHandler(handler)
            stats_log.setLevel(old_level)
            AUDIO_PREPROC_POOL_WORKERS.labels(backend=audx).set(0)
        self.write(stream.getvalue().splitlines())
        phases, present = sp.aggregate(sp.read_stats([str(self.log)]))
        self.assertTrue(set(sp.PHASE_ORDER) <= set(present), sorted(set(sp.PHASE_ORDER) - set(present)))
        self.assertEqual(phases["ingress"][("chunks",)].value(), 50)
        self.assertEqual(phases["vad"][("utterances",)].value(), 3)
        step = phases["asr"][("model_step", "STEP")]
        self.assertEqual((step.kind, step.weight), ("wmean", 2))
        self.assertAlmostEqual(step.value(), 60.0, places=3)
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
