import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trace_stream.py"
_spec = importlib.util.spec_from_file_location("trace_stream", _SCRIPT)
trace_stream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace_stream)


def ev(stage, utt, ts, stream="MO", session="call-1", rpc="r1", **extra):
    return {
        "timestamp": f"2026-09-29T10:00:00.{ts:03d}000+00:00",
        "level": "INFO",
        "stage": stage,
        "rpc_id": rpc,
        "session_id": session,
        "stream_id": stream,
        "utterance_id": utt,
        **extra,
    }


def healthy(utt, base, stream="MO"):
    return [
        ev("ASR_FINAL", utt, base, stream=stream, text="Xin chào", text_len=8, target_language="en-US"),
        ev("TRANSLATE", utt, base + 20, stream=stream, translated_text="Hello", latency_ms=18.0),
        ev("TTS_PHRASE", utt, base + 60, stream=stream, phrase_id=0, audio_bytes=160),
        ev("EMIT", utt, base + 61, stream=stream, seq=1, is_final=True, audio_bytes=160, queued=True),
    ]


def segment(utt, seg, base, **extra):
    return [
        ev("TRANSLATE", utt, base + 20, segment_idx=seg, translated_text="Hello", **extra),
        ev("TTS_PHRASE", utt, base + 60, segment_idx=seg, phrase_id=0, audio_bytes=160),
        ev("EMIT", utt, base + 61, segment_idx=seg, seq=seg + 1, is_final=True, audio_bytes=160, queued=True),
    ]


class SegmentedDiagnoseTest(unittest.TestCase):
    def _final(self, utt=1):
        return ev("ASR_FINAL", utt, 100, text="Xin chào bạn", text_len=12, target_language="en-US")

    def test_all_segments_complete_is_ok(self):
        events = [*segment(1, 0, 0), self._final(), *segment(1, 1, 110)]
        self.assertEqual(trace_stream.diagnose(events), "OK")

    def test_incomplete_segment_is_named(self):
        events = [*segment(1, 0, 0), self._final(), *segment(1, 1, 110)[:1]]
        status = trace_stream.diagnose(events)
        self.assertTrue(status.startswith("segment 1: INCOMPLETE: no TTS_PHRASE"), status)
        self.assertNotIn("segment 0", status)

    def test_segment_events_without_asr_final(self):
        status = trace_stream.diagnose(segment(1, 0, 0))
        self.assertIn("no ASR_FINAL", status)

    def test_render_shows_segment_idx(self):
        grouped = trace_stream.group_events([*segment(1, 0, 0), self._final()])
        self.assertIn("segment_idx=0", trace_stream.render(grouped))


class DiagnoseTest(unittest.TestCase):
    def test_complete_utterance_is_ok(self):
        self.assertEqual(trace_stream.diagnose(healthy(1, 0)), "OK")

    def test_missing_translate_reports_where_it_stopped(self):
        events = healthy(1, 0)[:1]
        self.assertEqual(
            trace_stream.diagnose(events),
            "INCOMPLETE: no TRANSLATE (stopped after ASR_FINAL)",
        )

    def test_missing_tts_reports_stop_after_translate(self):
        events = healthy(1, 0)[:2]
        self.assertEqual(
            trace_stream.diagnose(events),
            "INCOMPLETE: no TTS_PHRASE (stopped after TRANSLATE)",
        )

    def test_empty_asr_text(self):
        events = [ev("ASR_FINAL", 1, 0, text="", text_len=0, target_language="en-US")]
        self.assertTrue(trace_stream.diagnose(events).startswith("EMPTY_ASR"))

    def test_degrade_is_reported_with_reason(self):
        events = [
            healthy(1, 0)[0],
            ev("TRANSLATE", 1, 20, translated_text="", latency_ms=5.0),
            ev("DEGRADE", 1, 21, reason="translate_empty"),
            ev("EMIT", 1, 22, seq=1, is_final=True, audio_bytes=0, queued=True),
        ]
        self.assertEqual(trace_stream.diagnose(events), "DEGRADED (translate_empty)")

    def test_passthrough_is_not_incomplete(self):
        events = [
            ev("ASR_FINAL", 1, 0, text="hello", text_len=5, source_language="en-US"),
            ev("PASSTHROUGH", 1, 1, reason="source_equals_target", audio_bytes=800),
            ev("EMIT", 1, 2, seq=1, is_final=True, audio_bytes=800, queued=True),
        ]
        self.assertFalse(trace_stream.diagnose(events).startswith("INCOMPLETE"))

    def test_translated_emit_without_audio_is_flagged(self):
        events = healthy(1, 0)
        events[-1]["audio_bytes"] = 0
        self.assertTrue(trace_stream.diagnose(events).startswith("NO_AUDIO"))

    def test_submitted_but_no_translate_result(self):
        events = [healthy(1, 0)[0], ev("TRANSLATE_SUBMIT", 1, 1, queue_depth=3)]
        self.assertEqual(
            trace_stream.diagnose(events),
            "INCOMPLETE: no TRANSLATE (stopped after TRANSLATE_SUBMIT)",
        )

    def test_translation_held_behind_previous_tts(self):
        events = healthy(1, 0)[:2] + [ev("ORDER_HOLD", 1, 21, waiting_for=0, held_count=1)]
        self.assertEqual(
            trace_stream.diagnose(events),
            "INCOMPLETE: no TTS_PHRASE (stopped after ORDER_HOLD)",
        )

    def test_tts_submitted_but_no_result(self):
        events = healthy(1, 0)[:2] + [ev("TTS_SUBMIT", 1, 22, phrases=1)]
        self.assertEqual(
            trace_stream.diagnose(events),
            "INCOMPLETE: no TTS_PHRASE (stopped after TTS_SUBMIT)",
        )

    def test_stale_result_is_flagged(self):
        events = healthy(1, 0) + [ev("STALE", 1, 70, origin="tts", reason="job_missing")]
        self.assertEqual(
            trace_stream.diagnose(events), "WARN: stale tts result (job_missing)"
        )

    def test_new_stages_are_loaded_from_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            path.write_text(
                "\n".join(
                    json.dumps(ev(stage, 1, i))
                    for i, stage in enumerate(
                        ("TRANSLATE_SUBMIT", "ORDER_HOLD", "TTS_SUBMIT", "STALE")
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            events = trace_stream.load_events([str(path)])
        self.assertEqual(len(events), 4)

    def test_text_hidden_mode_still_diagnoses(self):
        events = healthy(1, 0)
        del events[0]["text"]
        self.assertEqual(trace_stream.diagnose(events), "OK")


class FilterAndRenderTest(unittest.TestCase):
    def test_filter_by_stream_and_group_by_utterance(self):
        events = healthy(1, 0, "MO") + healthy(1, 5, "MT") + healthy(2, 100, "MO")
        picked = trace_stream.filter_events(events, stream_id="MO")
        grouped = trace_stream.group_events(picked)
        self.assertEqual(list(grouped), [("r1", "call-1", "MO")])
        self.assertEqual(sorted(grouped[("r1", "call-1", "MO")]), [1, 2])

    def test_render_shows_stage_timeline_and_status(self):
        grouped = trace_stream.group_events(healthy(1, 0)[:2])
        text = trace_stream.render(grouped)
        self.assertIn("session=call-1 stream=MO rpc=r1", text)
        self.assertIn("utterance 1: INCOMPLETE: no TTS_PHRASE", text)
        self.assertIn("ASR_FINAL", text)
        self.assertIn("+    20.0ms", text)

    def test_only_problems_hides_ok_utterances(self):
        grouped = trace_stream.group_events(healthy(1, 0) + healthy(2, 100)[:1])
        text = trace_stream.render(grouped, only_problems=True)
        self.assertNotIn("utterance 1:", text)
        self.assertIn("utterance 2:", text)


class SegmentLatencyTest(unittest.TestCase):
    def _events(self):
        events = segment(1, 0, 0)
        latency = ev(
            "SEGMENT_LATENCY", 1, 61, segment_idx=0, asr_ms=420.5, translate_queue_ms=3.0,
            translate_ms=18.0, tts_queue_ms=1.5, tts_ms=40.0, order_hold_ms=0.5, total_ms=63.0,
        )
        events.insert(2, latency)
        return [ev("ASR_FINAL", 1, 100, text="x", text_len=1, target_language="en-US"), *events]

    def test_loaded_from_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            latency = next(e for e in self._events() if e["stage"] == "SEGMENT_LATENCY")
            path.write_text(json.dumps(latency) + "\n", encoding="utf-8")
            events = trace_stream.load_events([str(path)])
        self.assertEqual([e["stage"] for e in events], ["SEGMENT_LATENCY"])

    def test_render_shows_phase_details(self):
        out = trace_stream.render(trace_stream.group_events(self._events()))
        self.assertIn("SEGMENT_LATENCY", out)
        for key in ("asr_ms=420.5", "translate_ms=18.0", "tts_ms=40.0", "order_hold_ms=0.5", "total_ms=63.0"):
            self.assertIn(key, out)

    def test_diagnose_unaffected(self):
        self.assertEqual(trace_stream.diagnose(self._events()), "OK")


class LoadEventsTest(unittest.TestCase):
    def test_reads_json_lines_and_rotated_files_in_time_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "engine.log"
            rotated = Path(tmp) / "engine.log.2026-09-28"
            rotated.write_text(
                "not json\n" + json.dumps(ev("ASR_FINAL", 1, 0, text_len=3)) + "\n",
                encoding="utf-8",
            )
            base.write_text(
                json.dumps({"level": "INFO", "message": "unrelated"}) + "\n"
                + json.dumps(ev("EMIT", 1, 50, is_final=True)) + "\n",
                encoding="utf-8",
            )
            events = trace_stream.load_events([str(base)])
        self.assertEqual([e["stage"] for e in events], ["ASR_FINAL", "EMIT"])


LEGACY_LINE = (
    '{"timestamp":"2026-09-29T04:10:10.100000+00:00","level":"INFO","message":"PIPELINE ASR_FINAL",'
    '"session_id":"s-legacy","stream_id":"MO","utterance_id":1,"text":"xin chao","text_len":8,'
    '"message":"ASR_FINAL"}'
)


def _run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = trace_stream.main(argv)
    return code, out.getvalue(), err.getvalue()


class LegacyAndMessagesTest(unittest.TestCase):
    def _write(self, tmp, lines):
        path = Path(tmp) / "engine.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def test_legacy_line_without_stage_key_is_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [LEGACY_LINE])
            events, stats = trace_stream.scan_events([path])
        self.assertEqual([e["stage"] for e in events], ["ASR_FINAL"])
        self.assertEqual(stats["legacy_events"], 1)
        self.assertEqual(stats["pipeline_lines"], 1)

    def test_legacy_message_with_pipeline_prefix(self):
        record = json.loads(LEGACY_LINE)
        record["message"] = "PIPELINE TRANSLATE"
        self.assertEqual(trace_stream._legacy_stage(record), "TRANSLATE")
        record["message"] = "PIPELINE asr→translation: utterance=1"
        self.assertIsNone(trace_stream._legacy_stage(record))

    def test_main_prints_legacy_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [LEGACY_LINE])
            code, out, err = _run_main([path])
        self.assertEqual(code, 0)
        self.assertIn("older log format", err)
        self.assertIn("ASR_FINAL", out)

    def test_missing_file_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = _run_main([str(Path(tmp) / "nope.log")])
        self.assertEqual(code, 2)
        self.assertIn("No log file matches", err)

    def test_no_trace_lines_message_reports_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, ['{"message":"PIPELINE asr→translation: utterance=1"}', "plain text"])
            code, _, err = _run_main([path])
        self.assertEqual(code, 1)
        self.assertIn("No pipeline trace lines", err)
        self.assertIn("1 line(s) contain 'PIPELINE'", err)

    def test_filters_excluded_everything_lists_available_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [json.dumps(e) for e in healthy(1, 0)])
            code, _, err = _run_main([path, "--session-id", "other"])
        self.assertEqual(code, 1)
        self.assertIn("none match the filters", err)
        self.assertIn("call-1", err)
        self.assertIn("MO", err)


class SkippedEmptyTest(unittest.TestCase):
    def test_skipped_final_is_noted_not_incomplete(self):
        events = [
            ev("ASR_FINAL", 1, 0, text="x", text_len=1, target_language="en-US"),
            ev("TRANSLATE", 1, 20, translated_text="", latency_ms=5.0),
            ev("TTS_PHRASE", 1, 30, phrase_id=0, audio_bytes=0),
            ev("EMIT", 1, 31, is_final=True, audio_bytes=0, skipped="empty"),
        ]
        status = trace_stream.diagnose(events)
        self.assertIn("NOTE: skipped empty result", status)
        self.assertNotIn("INCOMPLETE", status)


def multi_phrase(utt, base, translated, phrases, audio_bytes, emit_bytes=None):
    events = [
        ev("ASR_FINAL", utt, base, text="x", text_len=1, target_language="en-US"),
        ev("TRANSLATE", utt, base + 10, translated_text=translated, translated_text_len=len(translated)),
        ev("TTS_SUBMIT", utt, base + 11, phrase_id=0, phrases=len(phrases), text=phrases[0]),
    ]
    for i, (text, nbytes) in enumerate(zip(phrases, audio_bytes)):
        events.append(
            ev("TTS_PHRASE", utt, base + 20 + i * 10, phrase_id=i, text=text, text_len=len(text), audio_bytes=nbytes)
        )
    emit_bytes = audio_bytes if emit_bytes is None else emit_bytes
    for i, nbytes in enumerate(emit_bytes):
        events.append(
            ev("EMIT", utt, base + 25 + i * 10, seq=i + 1, is_final=i == len(emit_bytes) - 1,
               audio_bytes=nbytes, queued=True)
        )
    return events


class TtsConsistencyTest(unittest.TestCase):
    TRANSLATED = "Hôm nay trời đẹp lắm. Mọi người đều vui vẻ khác nhau. Cho dù mưa vẫn đi."

    def test_matching_phrases_with_plausible_audio_are_ok(self):
        phrases = ["Hôm nay trời đẹp lắm.", "Mọi người đều vui vẻ khác nhau.", "Cho dù mưa vẫn đi."]
        events = multi_phrase(1, 0, self.TRANSLATED, phrases, [8000, 12000, 7000])
        self.assertEqual(trace_stream.diagnose(events), "OK")

    def test_dropped_words_between_translation_and_tts_are_reported(self):
        phrases = ["Hôm nay trời đẹp lắm.", "Mọi người đều vui vẻ.", "cho dù mưa vẫn đi."]
        events = multi_phrase(1, 0, self.TRANSLATED, phrases, [8000, 8000, 7000])
        status = trace_stream.diagnose(events)
        self.assertIn("WARN: TTS text differs from translation", status)
        self.assertIn("khác nhau.", status)

    def test_audio_far_shorter_than_text_is_flagged(self):
        phrases = ["Hôm nay trời đẹp lắm.", "Mọi người đều vui vẻ khác nhau.", "Cho dù mưa vẫn đi."]
        events = multi_phrase(1, 0, self.TRANSLATED, phrases, [8000, 2000, 7000])
        status = trace_stream.diagnose(events)
        self.assertIn("phrase 1 audio too short for its text", status)
        self.assertNotIn("phrase 0", status)
        self.assertEqual(trace_stream.diagnose(events, max_cps=200.0), "OK")

    def test_missing_phrase_is_incomplete(self):
        phrases = ["Hôm nay trời đẹp lắm.", "Mọi người đều vui vẻ khác nhau."]
        events = multi_phrase(1, 0, self.TRANSLATED, phrases, [8000, 12000])
        events[2]["phrases"] = 3
        self.assertIn("INCOMPLETE: only 2 of 3 phrases synthesized", trace_stream.diagnose(events))

    def test_emitted_audio_less_than_synthesized_is_flagged(self):
        phrases = ["Hôm nay trời đẹp lắm.", "Mọi người đều vui vẻ khác nhau.", "Cho dù mưa vẫn đi."]
        events = multi_phrase(1, 0, self.TRANSLATED, phrases, [8000, 12000, 7000], emit_bytes=[8000, 7000])
        self.assertIn("emitted audio 15000 bytes < synthesized 27000 bytes", trace_stream.diagnose(events))

    def test_emit_seq_out_of_order_is_flagged(self):
        phrases = ["Hôm nay trời đẹp lắm.", "Mọi người đều vui vẻ khác nhau.", "Cho dù mưa vẫn đi."]
        events = multi_phrase(1, 0, self.TRANSLATED, phrases, [8000, 12000, 7000])
        emits = [e for e in events if e["stage"] == "EMIT"]
        emits[0]["seq"], emits[1]["seq"] = emits[1]["seq"], emits[0]["seq"]
        self.assertIn("EMIT seq not increasing", trace_stream.diagnose(events))

    def test_text_hidden_logs_skip_text_comparison(self):
        events = multi_phrase(1, 0, self.TRANSLATED, ["a b c d e f g", "h"], [8000, 100])
        for e in events:
            e.pop("text", None)
            e.pop("translated_text", None)
        self.assertEqual(trace_stream.diagnose(events), "OK")

    def test_render_shows_phrase_texts_and_audio_seconds(self):
        events = multi_phrase(1, 0, "Hello. World.", ["Hello.", "World."], [8000, 4000])
        events[2]["phrase_texts"] = ["Hello.", "World."]
        out = trace_stream.render(trace_stream.group_events(events))
        self.assertIn("phrase_texts=['Hello.', 'World.']", out)
        self.assertIn("audio_s=1.00", out)


def partials(utt, base):
    return [
        ev("ASR_PARTIAL", utt, base, step=3, model_step=3, changed=True, audio_ms=1680, t_ms=1700.0,
           text="xin", text_len=3, common_prefix_words=0, revised_words=0, sentence_ends=0),
        ev("ASR_PARTIAL", utt, base + 300, step=4, model_step=4, changed=False, audio_ms=2240, t_ms=2260.0),
    ]


class PartialTest(unittest.TestCase):
    def _log(self, tmp, events):
        path = Path(tmp) / "engine.log"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
        return str(path)

    def test_partials_are_loaded_as_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = trace_stream.load_events([self._log(tmp, partials(1, 0))])
        self.assertEqual([e["stage"] for e in events], ["ASR_PARTIAL", "ASR_PARTIAL"])

    def test_diagnose_ignores_partials(self):
        events = partials(1, 0) + healthy(1, 600)
        self.assertEqual(trace_stream.diagnose(events), "OK")

    def test_partial_only_utterance_is_not_reported_as_ok_pipeline(self):
        self.assertIn("no ASR_FINAL", trace_stream.diagnose(partials(1, 0)))

    def test_render_shows_partials_with_negative_offset_before_final(self):
        out = trace_stream.render(trace_stream.group_events(partials(1, 0) + healthy(1, 600)))
        self.assertIn("utterance 1: OK", out)
        self.assertRegex(out, r"-\s+600\.0ms  ASR_PARTIAL")
        self.assertRegex(out, r"-\s+300\.0ms  ASR_PARTIAL")
        self.assertIn("step=3", out)
        self.assertIn("changed=True", out)

    def test_partials_hidden_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._log(tmp, partials(1, 0) + healthy(1, 600))
            _, out, _ = _run_main([path])
            _, out_with, _ = _run_main([path, "--partials"])
        self.assertNotIn("ASR_PARTIAL", out)
        self.assertIn("ASR_PARTIAL", out_with)

    def test_partial_only_log_without_flag_reports_no_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = _run_main([self._log(tmp, partials(1, 0))])
        self.assertEqual(code, 1)
        self.assertIn("Only ASR_PARTIAL trace lines found (2)", err)
        self.assertIn("--partials", err)
        self.assertNotIn("0 recognised", err)


def _t(ms):
    return f"2026-09-29T10:00:{ms // 1000:02d}.{ms % 1000:03d}000+00:00"


def lev(stage, utt, ms, **extra):
    event = ev(stage, utt, 0, **extra)
    event["timestamp"] = _t(ms)
    if utt is None:
        del event["utterance_id"]
    return event


def lifecycle_call(utt_start=500, with_close=True):
    """One RPC / one stream / one utterance, from GRPC_OPEN to GRPC_CLOSE (timestamps in ms)."""
    events = [
        {"timestamp": _t(0), "stage": "GRPC_OPEN", "rpc_id": "r1"},
        lev("STREAM_OPEN", None, 5),
        lev("VAD_START", 1, utt_start, pre_roll_ms=100.0),
        lev("VAD_END", 1, utt_start + 2000, utterance_audio_ms=2000.0, vad_silence_ms=256.0),
        lev("ASR_FINAL", 1, utt_start + 2030, text="Xin chào", text_len=8, target_language="en-US"),
        lev("TRANSLATE", 1, utt_start + 2060, translated_text="Hello", latency_ms=18.0),
        lev("TTS_PHRASE", 1, utt_start + 2140, phrase_id=0, audio_bytes=160),
        lev("EMIT", 1, utt_start + 2141, seq=1, is_final=True, audio_bytes=160, queued=True),
        lev(
            "GRPC_SENT", 1, utt_start + 2145, seq=1, is_final=True, audio_bytes=160, send_ms=0.4,
            first_audio=True, speech_end_to_sent_ms=420.5,
        ),
    ]
    if with_close:
        events += [
            lev("STREAM_CLOSE", None, utt_start + 3000),
            {"timestamp": _t(utt_start + 3010), "stage": "GRPC_CLOSE", "rpc_id": "r1"},
        ]
    return events


class FullStreamTest(unittest.TestCase):
    def _render(self, events, **kwargs):
        grouped = trace_stream.group_events(events)
        return trace_stream.render(
            grouped, stream_events=trace_stream.group_stream_events(events, grouped), **kwargs
        )

    def test_lifecycle_and_first_audio_are_rendered(self):
        out = self._render(lifecycle_call())
        self.assertIn("first audio to rtpgw: 2645.0ms after GRPC_OPEN, 2640.0ms after STREAM_OPEN (utterance 1, seq=1)", out)
        for stage in ("GRPC_OPEN", "STREAM_OPEN", "STREAM_CLOSE", "GRPC_CLOSE", "VAD_START", "VAD_END", "GRPC_SENT"):
            self.assertIn(stage, out)
        self.assertIn(
            ">> first audio to rtpgw (seq=1): +145.0ms after VAD_END, +115.0ms after ASR_FINAL"
            ", 420.5ms after speech end, +2645.0ms after GRPC_OPEN",
            out,
        )
        self.assertNotIn("log ends before", out)
        self.assertIn("utterance 1: OK", out)

    def test_first_audio_line_shows_speech_end_with_denoiser_delay(self):
        events = lifecycle_call()
        for e in events:
            if e["stage"] == "GRPC_SENT":
                e["speech_end_to_sent_preproc_ms"] = 1920.5
        out = self._render(events)
        self.assertIn(
            "420.5ms after speech end, 1920.5ms after speech end incl. denoiser delay, +2645.0ms after GRPC_OPEN",
            out,
        )

    def test_first_audio_line_omits_denoiser_delay_when_absent(self):
        out = self._render(lifecycle_call())
        self.assertNotIn("denoiser delay", out)

    def test_segmented_first_audio_before_speech_end_is_not_misattributed(self):
        events = lifecycle_call()
        events.insert(
            -3,
            lev(
                "GRPC_SENT", 1, 2700, seq=2, is_final=True, audio_bytes=80, send_ms=0.2,
                first_audio=False, speech_end_to_sent_ms=476.1,
            ),
        )
        for e in events:
            if e["stage"] == "GRPC_SENT" and e.get("seq") == 1:
                e.pop("speech_end_to_sent_ms")
        out = self._render(events)
        line = next(l for l in out.splitlines() if ">> first audio to rtpgw (seq=1)" in l)
        self.assertIn("before speech end (segmented)", line)
        self.assertNotIn("476.1ms after speech end", line)

    def test_info_stages_do_not_change_status_or_baseline(self):
        events = lifecycle_call()
        utterance = [e for e in events if e.get("utterance_id") == 1]
        self.assertEqual(trace_stream.diagnose(utterance), "OK")
        out = self._render(events)
        self.assertIn("+     0.0ms  ASR_FINAL", out)
        self.assertIn("-  2030.0ms  VAD_START", out)

    def test_unclosed_log_is_noted(self):
        out = self._render(lifecycle_call(with_close=False))
        self.assertIn("(log ends before the stream/RPC was closed)", out)

    def test_cancelled_rpc_without_audio_is_reported(self):
        events = [
            {"timestamp": _t(0), "stage": "GRPC_OPEN", "rpc_id": "r1"},
            lev("STREAM_OPEN", None, 5),
            {"timestamp": _t(900), "stage": "GRPC_CANCEL", "rpc_id": "r1",
             "reason": "gRPC call ended before engine pipeline completed"},
        ]
        out = self._render(events, only_problems=True)
        self.assertIn("first audio to rtpgw: unknown", out)
        self.assertIn("GRPC_CANCEL", out)
        self.assertIn("gRPC call ended before engine pipeline completed", out)

    def test_audio_emitted_but_never_sent_is_flagged(self):
        events = [e for e in lifecycle_call() if e["stage"] != "GRPC_SENT"]
        events.append(lev("GRPC_SENT", 1, 3000, seq=2, is_final=True, audio_bytes=0))
        out = self._render(events)
        self.assertIn("first audio to rtpgw: NONE (no GRPC_SENT carried audio)", out)
        self.assertIn("EMIT had audio but no GRPC_SENT with audio", out)

    def test_ok_stream_is_hidden_by_only_problems(self):
        self.assertEqual(self._render(lifecycle_call(), only_problems=True), "")

    def test_filter_keeps_rpc_level_events_of_matching_stream(self):
        events = lifecycle_call()
        kept = trace_stream.filter_events(events, stream_id="MO")
        self.assertIn("GRPC_OPEN", [e["stage"] for e in kept])
        self.assertEqual(trace_stream.filter_events(events, stream_id="other"), [])
        by_utt = trace_stream.filter_events(events, utterance_id=1)
        self.assertNotIn("GRPC_OPEN", [e["stage"] for e in by_utt])
        self.assertNotIn("STREAM_OPEN", [e["stage"] for e in by_utt])

    def test_plain_message_lines_of_older_builds_are_recovered(self):
        base = {"level": "INFO", "rpc_id": "r1", "session_id": "call-1", "stream_id": "MO"}
        lines = [
            {**base, "timestamp": "2026-09-29T10:00:00.000000+00:00", "message": "RPC opened"},
            {**base, "timestamp": "2026-09-29T10:00:00.005000+00:00", "message": "Logical stream opened"},
            {**base, "timestamp": "2026-09-29T10:00:00.800000+00:00", "stage": "EMIT", "utterance_id": 3,
             "seq": 7, "audio_bytes": 160, "is_final": True},
            {**base, "timestamp": "2026-09-29T10:00:00.804000+00:00",
             "message": "GRPC_SENT rpc=r1 session=call-1 stream=MO seq=7 final=True ts_start=0 ts_end=1 "
                         "text_len=5 audio_bytes=160 send_ms=0.4 speech_end_to_sent_ms=410.0"},
            {**base, "timestamp": "2026-09-29T10:00:01.000000+00:00", "message": "RPC cancelled: pipeline idle"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
            events, _ = trace_stream.scan_events([str(path)])
        stages = [e["stage"] for e in events]
        self.assertEqual(stages, ["GRPC_OPEN", "STREAM_OPEN", "EMIT", "GRPC_SENT", "GRPC_CANCEL"])
        sent = events[3]
        self.assertEqual((sent["seq"], sent["audio_bytes"], sent["is_final"], sent["utterance_id"]), (7, 160, True, 3))
        self.assertEqual(sent["speech_end_to_sent_ms"], 410.0)
        self.assertEqual(events[4]["reason"], "pipeline idle")

    def test_main_prints_full_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            path.write_text("\n".join(json.dumps(e) for e in lifecycle_call()) + "\n", encoding="utf-8")
            code, out, _ = _run_main([str(path), "--stream-id", "MO"])
        self.assertEqual(code, 0)
        self.assertIn("stream lifecycle:", out)
        self.assertIn("first audio to rtpgw:", out)


if __name__ == "__main__":
    unittest.main()
