import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_rtp_log.py"
_spec = importlib.util.spec_from_file_location("check_rtp_log", _SCRIPT)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def chunk(ts, seq=-1, *, clock=8000, rate=16000, pcm=640, dur=20, stream="MO", rpc="r1", session="s1"):
    return {
        "rpc": rpc,
        "session": session,
        "stream": stream,
        "pcm_bytes": pcm,
        "sample_rate": rate,
        "channels": 1,
        "rtp_timestamp": ts,
        "rtp_clock_rate": clock,
        "chunk_seq": seq,
        "duration_ms": float(dur),
    }


def line(c, language="vi", task="realtime_translation"):
    return (
        f"GRPC_IN rpc={c['rpc']} session={c['session']} stream={c['stream']} "
        f"pcm_bytes={c['pcm_bytes']} sample_rate={c['sample_rate']} channels={c['channels']} "
        f"rtp_timestamp={c['rtp_timestamp']} rtp_clock_rate={c['rtp_clock_rate']} "
        f"chunk_seq={c['chunk_seq']} duration_ms={int(c['duration_ms'])} "
        f"end_of_stream=False language={language} task={task}"
    )


def pcmu_stream(n=20, start=0, step=160, seq0=0, **kw):
    return [chunk((start + i * step) & 0xFFFFFFFF, seq0 + i, **kw) for i in range(n)]


class AnalyzeTest(unittest.TestCase):
    def test_pcmu_16k_pcm_with_8k_clock_is_ok(self):
        result = check.analyze_stream(pcmu_stream())
        self.assertEqual(check.verdict(result), "OK")
        self.assertAlmostEqual(result["inferred_clock"], 8000, delta=1)

    def test_timestamp_wraparound_is_ok(self):
        result = check.analyze_stream(pcmu_stream(start=0xFFFFFF00))
        self.assertEqual(check.verdict(result), "OK")

    def test_client_clock_bug_is_detected(self):
        # timestamps advance +320 per 20 ms but declared clock is 8000
        result = check.analyze_stream(pcmu_stream(step=320, clock=8000))
        self.assertIn("clock_mismatch", result["counts"])
        self.assertTrue(check.verdict(result).startswith("PROBLEMS"))
        mismatch = [i for i in result["issues"] if i["kind"] == "clock_mismatch"][0]
        self.assertIn("rtp_clock_rate should be 16000", mismatch["detail"])
        self.assertGreater(result["inserted_silence_ms"], 0)

    def test_unset_clock_uses_engine_default_and_flags_mismatch(self):
        result = check.analyze_stream(pcmu_stream(step=320, clock=0))
        self.assertEqual(result["effective_clock"], 8000)
        self.assertIn("clock_unset", result["counts"])
        self.assertIn("clock_mismatch", result["counts"])

    def test_unset_clock_with_8k_timestamps_only_warns(self):
        result = check.analyze_stream(pcmu_stream(clock=0))
        self.assertEqual(list(result["counts"]), ["clock_unset"])
        self.assertTrue(check.verdict(result).startswith("WARN"))

    def test_small_gap_is_inserted_silence(self):
        chunks = pcmu_stream(5)
        chunks[3]["rtp_timestamp"] += 480  # 60 ms missing
        chunks[4]["rtp_timestamp"] += 480
        result = check.analyze_stream(chunks)
        self.assertEqual(result["counts"], {"gap_inserted": 1})
        self.assertAlmostEqual(result["inserted_silence_ms"], 60.0, delta=0.5)

    def test_large_gap_is_audio_loss(self):
        chunks = pcmu_stream(5)
        chunks[3]["rtp_timestamp"] += 4000  # 500 ms
        chunks[4]["rtp_timestamp"] += 4000
        result = check.analyze_stream(chunks)
        self.assertEqual(result["counts"], {"gap_lost": 1})
        self.assertAlmostEqual(result["lost_ms"], 500.0, delta=0.5)
        self.assertTrue(check.verdict(result).startswith("PROBLEMS"))

    def test_jitter_within_tolerance(self):
        chunks = pcmu_stream(5)
        chunks[3]["rtp_timestamp"] -= 80  # 10 ms behind
        result = check.analyze_stream(chunks)
        self.assertEqual(result["counts"].get("jitter"), 1)
        self.assertNotIn("out_of_order", result["counts"])
        # as the engine: expected moves to the end of chunk 3 (400 + 160), so only the real 10 ms hole before
        # chunk 4 (640) is filled, not 20 ms from the old expected (480)
        self.assertAlmostEqual(result["inserted_silence_ms"], 10.0, delta=0.5)

    def test_jitter_then_contiguous_chunk_has_no_gap(self):
        chunks = pcmu_stream(5)
        for later in chunks[3:]:
            later["rtp_timestamp"] -= 16  # chunk 3 is 2 ms behind, chunk 4 contiguous with it
        result = check.analyze_stream(chunks)
        self.assertEqual(result["counts"], {"jitter": 1})
        self.assertEqual(result["inserted_silence_ms"], 0.0)

    def test_out_of_order_beyond_jitter(self):
        chunks = pcmu_stream(5)
        chunks[3]["rtp_timestamp"] -= 480
        result = check.analyze_stream(chunks)
        self.assertEqual(result["counts"].get("out_of_order"), 1)

    def test_chunk_seq_constant_zero_is_flagged(self):
        chunks = [chunk(i * 160, 0) for i in range(5)]
        self.assertIn("seq_constant_zero", check.analyze_stream(chunks)["counts"])

    def test_chunk_seq_not_provided_is_ignored(self):
        chunks = [chunk(i * 160, -1) for i in range(5)]
        self.assertEqual(check.verdict(check.analyze_stream(chunks)), "OK")

    def test_chunk_seq_gap_and_backwards(self):
        chunks = pcmu_stream(6)
        chunks[3]["chunk_seq"] = 6
        chunks[4]["chunk_seq"] = 7
        chunks[5]["chunk_seq"] = 3
        counts = check.analyze_stream(chunks)["counts"]
        self.assertEqual(counts.get("seq_gap"), 1)
        self.assertEqual(counts.get("seq_backwards"), 1)

    def test_param_change_and_duration_mismatch(self):
        chunks = pcmu_stream(4)
        chunks[2]["sample_rate"] = 8000
        chunks[1]["duration_ms"] = 40.0
        counts = check.analyze_stream(chunks)["counts"]
        self.assertIn("param_change", counts)
        self.assertIn("duration_mismatch", counts)

    def test_40ms_chunks_use_actual_payload_duration(self):
        chunks = [chunk(i * 320, i, pcm=1280, dur=40) for i in range(10)]
        self.assertEqual(check.verdict(check.analyze_stream(chunks)), "OK")


class ParseAndRenderTest(unittest.TestCase):
    def test_parses_json_and_plain_lines_with_empty_fields(self):
        c = chunk(160, 1)
        parsed_plain = check.parse_line(line(c, language="", task=""))
        parsed_json = check.parse_line(json.dumps({"level": "DEBUG", "message": line(c)}))
        self.assertEqual(parsed_plain["rtp_timestamp"], 160)
        self.assertEqual(parsed_json["stream"], "MO")
        self.assertIsNone(check.parse_line(json.dumps({"message": "unrelated"})))
        self.assertIsNone(check.parse_line("GRPC_IN rpc=r1 incomplete"))

    def test_load_filter_group_and_render(self):
        mo = pcmu_stream(5, stream="MO")
        mt = pcmu_stream(5, stream="MT", step=320)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            rotated = Path(tmp) / "engine.log.2026-09-28"
            rotated.write_text(
                "\n".join(json.dumps({"message": line(c)}) for c in mo[:2]) + "\n", encoding="utf-8"
            )
            path.write_text(
                "\n".join(json.dumps({"message": line(c)}) for c in mo[2:] + mt) + "\n", encoding="utf-8"
            )
            chunks = check.load_chunks([str(path)])
        self.assertEqual(len(chunks), 10)
        self.assertEqual([c["rtp_timestamp"] for c in chunks[:5]], [0, 160, 320, 480, 640])

        grouped = check.group_chunks(check.filter_chunks(chunks, stream_id="MT"))
        self.assertEqual(list(grouped), [("r1", "s1", "MT")])

        text = check.render(check.group_chunks(chunks), max_issues=2)
        self.assertIn("session=s1 stream=MO rpc=r1", text)
        self.assertIn("verdict: OK", text)
        self.assertIn("verdict: PROBLEMS: ", text)
        self.assertIn("inferred_rtp_clock=16000", text)

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.log"
            bad = Path(tmp) / "bad.log"
            empty = Path(tmp) / "empty.log"
            good.write_text("\n".join(line(c) for c in pcmu_stream(8)) + "\n", encoding="utf-8")
            bad.write_text("\n".join(line(c) for c in pcmu_stream(8, step=320)) + "\n", encoding="utf-8")
            empty.write_text("nothing here\n", encoding="utf-8")
            self.assertEqual(check.main([str(good)]), 0)
            self.assertEqual(check.main([str(bad)]), 1)
            self.assertEqual(check.main([str(empty)]), 2)


class EmptyResultMessagesTest(unittest.TestCase):
    def _run(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = check.main(argv)
        return code, err.getvalue()

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self._run([str(Path(tmp) / "nope.log")])
        self.assertEqual(code, 2)
        self.assertIn("No log file matches", err)

    def test_no_grpc_in_lines_mentions_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.log"
            path.write_text("nothing here\n", encoding="utf-8")
            code, err = self._run([str(path)])
        self.assertEqual(code, 2)
        self.assertIn("DEBUG", err)

    def test_unparseable_grpc_in_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.log"
            path.write_text("GRPC_IN rpc=r1 session=s1\n", encoding="utf-8")
            code, err = self._run([str(path)])
        self.assertEqual(code, 2)
        self.assertIn("expected fields", err)

    def test_filters_excluded_everything_lists_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.log"
            path.write_text("\n".join(line(c) for c in pcmu_stream(4)) + "\n", encoding="utf-8")
            code, err = self._run([str(path), "--stream-id", "XX"])
        self.assertEqual(code, 2)
        self.assertIn("none match the filters", err)
        self.assertIn("s1", err)


if __name__ == "__main__":
    unittest.main()
