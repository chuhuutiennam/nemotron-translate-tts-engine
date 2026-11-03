import contextlib
import dataclasses
import importlib.util
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_segment_output.py"
_spec = importlib.util.spec_from_file_location("check_segment_output", _SCRIPT)
cso = importlib.util.module_from_spec(_spec)
sys.modules["check_segment_output"] = cso
_spec.loader.exec_module(cso)


def event(stage, session="s1", stream="MO", utt=1, rpc="r1", **fields):
    return json.dumps(
        {"stage": stage, "rpc_id": rpc, "session_id": session, "stream_id": stream, "utterance_id": utt, **fields},
        ensure_ascii=False,
    )


def stt_utterance(session, utt, segments, final, *, seq0=1, committed=None):
    """EMIT is_final per segment (the first before ASR_FINAL, like a mid-utterance commit) and the ASR_FINAL."""
    committed = len(segments[0].split()) if committed is None and len(segments) > 1 else (committed or 0)
    lines = [event("EMIT", session, utt=utt, seq=seq0, segment_idx=0, is_final=True, text=segments[0])]
    lines.append(event("ASR_FINAL", session, utt=utt, text=final, committed_tokens=committed))
    for index, text in enumerate(segments[1:], start=1):
        lines.append(event("EMIT", session, utt=utt, seq=seq0 + index, segment_idx=index, is_final=True, text=text))
    return lines


def run_main(lines, *args, name="engine.log", extra_files=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for extra_name, extra_lines in (extra_files or {}).items():
            (Path(tmp) / extra_name).write_text("\n".join(extra_lines) + "\n", encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cso.main([str(path), *args])
        return code, out.getvalue(), err.getvalue()


class CompareTest(unittest.TestCase):
    def _status(self, segments, final):
        return cso.compare(segments, final)

    def test_segments_adding_up_are_ok(self):
        c = self._status(["This is a complete sentence here.", "and more words"], "This is a complete sentence here. and more words")
        self.assertEqual((c.status, c.cut_loss, c.cut_dup, c.cut_changed, c.revised), (cso.OK, 0, 0, 0, 0))

    def test_case_and_punctuation_are_ignored(self):
        c = self._status(["hello my friend how are you", "Fine"], "Hello, my friend, how are you? Fine.")
        self.assertEqual(c.status, cso.OK)

    def test_word_repeated_at_the_cut_is_cut_dup(self):
        c = self._status(
            ["I'm going to the store today.", "today. And then home."],
            "I am going to the store today. And then home.",
        )
        self.assertEqual((c.status, c.cut_dup, c.revised), (cso.CUT_DUP, 1, 1))
        self.assertIn("[+today]", c.diff)
        self.assertIn("[m -> am]", c.diff)

    def test_word_never_sent_at_the_cut_is_cut_loss(self):
        c = self._status(["We will meet at nine thirty.", "you there."], "We'll meet at nine thirty. See you there.")
        self.assertEqual((c.status, c.cut_loss), (cso.CUT_LOSS, 1))
        self.assertIn("[-see]", c.diff)

    def test_words_after_everything_sent_are_cut_loss(self):
        c = self._status(["One two three four five six."], "One two three four five six. Seven eight.")
        self.assertEqual((c.status, c.cut_loss), (cso.CUT_LOSS, 2))

    def test_word_inserted_at_the_start_is_revised(self):
        c = self._status(["I'm going to the store today.", "Okay."], "Uh I'm going to the store today. Okay.")
        self.assertEqual((c.status, c.revised), (cso.REVISED, 1))

    def test_change_inside_a_segment_is_revised(self):
        c = self._status(
            ["I'm going to the store today.", "And then home."], "I am going to the store today. And then home."
        )
        self.assertEqual((c.status, c.revised, c.cut_loss, c.cut_dup), (cso.REVISED, 1, 0, 0))

    def test_committed_last_word_dropped_by_asr_is_cut_changed(self):
        c = self._status(["Please send me the report now.", "Bye."], "Please send me the report. Bye.")
        self.assertEqual((c.status, c.cut_changed), (cso.CUT_CHANGED, 1))

    def test_word_replaced_at_the_cut_is_cut_changed(self):
        c = self._status(["I will go to the store today.", "day. And then home."], "I will go to the store to day. And then home.")
        self.assertEqual(c.status, cso.CUT_CHANGED)

    def test_worst_status_wins(self):
        c = cso.Comparison(cut_loss=0, cut_dup=1, cut_changed=2, revised=3)
        self.assertEqual(c.status, cso.CUT_DUP)
        self.assertEqual(cso.Comparison(cut_loss=1, cut_dup=1).status, cso.CUT_LOSS)


class MainTest(unittest.TestCase):
    SAMPLE = [
        event("STREAM_OPEN", "stt", utt=None, task="speech_to_text"),
        *stt_utterance("stt", 1, ["I'm going to the store today.", "today. And then home."],
                       "I am going to the store today. And then home."),
        *stt_utterance("stt", 2, ["This is a complete sentence here.", "and more words"],
                       "This is a complete sentence here. and more words", seq0=5),
        event("ASR_FINAL", "stt", utt=3, text="Hello.", committed_tokens=0),
        event("EMIT", "stt", utt=3, seq=9, is_final=True, text="Hello."),
        event("STREAM_OPEN", "tr", utt=None, task="realtime_translation"),
        event("TRANSLATE_SUBMIT", "tr", utt=1, segment_idx=0, source_text="We will meet at nine thirty."),
        event("ASR_FINAL", "tr", utt=1, text="We'll meet at nine thirty. See you there.", committed_tokens=6),
        event("TRANSLATE_SUBMIT", "tr", utt=1, segment_idx=1, source_text="you there."),
        event("EMIT", "tr", utt=1, seq=3, segment_idx=1, is_final=True, text="Hẹn gặp bạn ở đó."),
        event("ASR_FINAL", "stt", utt=4, text="never finished"),
        "not json",
    ]

    def test_summary_counts_and_worst_examples_first(self):
        code, out, _ = run_main(self.SAMPLE)
        self.assertEqual(code, 0)
        self.assertIn("utterances with ASR_FINAL: 5 (speech_to_text 4, realtime_translation 1)", out)
        self.assertIn("segmented (text sent before ASR_FINAL): 3 (speech_to_text 2, realtime_translation 1)", out)
        self.assertIn("cut errors (any CUT_LOSS or CUT_DUP word): 2 of 3 segmented utterances (66.7%)", out)
        self.assertLess(out.index("[CUT_LOSS] session=tr"), out.index("[CUT_DUP] session=stt"))
        self.assertNotIn("[OK]", out)

    def test_verbose_lists_ok_utterances_too(self):
        _, out, _ = run_main(self.SAMPLE, "--verbose")
        self.assertIn("[OK] session=stt stream=MO utterance=2", out)

    def test_examples_limit(self):
        _, out, _ = run_main(self.SAMPLE, "--examples", "1")
        self.assertIn("[CUT_LOSS]", out)
        self.assertNotIn("[CUT_DUP] session", out)

    def test_filters(self):
        code, out, _ = run_main(self.SAMPLE, "--session-id", "tr")
        self.assertEqual(code, 0)
        self.assertIn("segmented (text sent before ASR_FINAL): 1 (speech_to_text 0, realtime_translation 1)", out)
        code, _, err = run_main(self.SAMPLE, "--session-id", "nobody")
        self.assertEqual(code, 1)
        self.assertIn("match the filters", err)

    def test_rotated_files_are_read(self):
        rotated = stt_utterance("old", 1, ["One two three four five six.", "seven"], "One two three four five six. seven")
        _, out, _ = run_main(self.SAMPLE, extra_files={"engine.log.2026-10-29": rotated}, name="engine.log")
        self.assertIn("files: 2", out)
        self.assertIn("segmented (text sent before ASR_FINAL): 4", out)

    def test_csv_has_one_row_per_segmented_utterance(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            code, _, _ = run_main(self.SAMPLE, "--csv", str(csv_path))
            self.assertEqual(code, 0)
            rows = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[0].startswith("rpc_id,session_id,stream_id,utterance_id,task,status"))

    def test_missing_text_exits_2(self):
        lines = [event("ASR_FINAL", "s", text_len=5, committed_tokens=0)]
        code, _, err = run_main(lines)
        self.assertEqual(code, 2)
        self.assertIn("pipeline_trace_text", err)

    def test_no_file_no_final_and_nothing_segmented_exit_1(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(cso.main(["/nonexistent/dir/engine.log"]), 1)
        self.assertIn("No log file matches", err.getvalue())
        code, _, err = run_main([event("EMIT", "s", is_final=True, text="x", seq=1)])
        self.assertEqual(code, 1)
        self.assertIn("No ASR_FINAL", err)
        code, _, err = run_main([event("ASR_FINAL", "s", text="Hello.", committed_tokens=0)])
        self.assertEqual(code, 1)
        self.assertIn("No segmented utterance", err)

    def test_task_inferred_without_stream_open_and_duplicate_submit_counted(self):
        lines = [
            event("TRANSLATE_SUBMIT", "t", segment_idx=0, source_text="One two three four five six."),
            event("TRANSLATE_SUBMIT", "t", segment_idx=0, source_text="One two three four five six."),
            event("ASR_FINAL", "t", text="One two three four five six. Seven.", committed_tokens=6),
            event("TRANSLATE_SUBMIT", "t", segment_idx=1, source_text="Seven."),
            event("EMIT", "t", seq=1, segment_idx=0, is_final=True, text="Một hai ba bốn năm sáu."),
        ]
        code, out, _ = run_main(lines)
        self.assertEqual(code, 0)
        self.assertIn("realtime_translation 1)", out)
        self.assertIn("translation segments submitted more than once: 1", out)
        self.assertIn("OK                    1", out)

    def test_segments_are_joined_in_seq_order_not_file_order(self):
        lines = [
            event("EMIT", "s", seq=4, segment_idx=1, is_final=True, text="And then home."),
            event("EMIT", "s", seq=2, segment_idx=0, is_final=True, text="I am going to the store today."),
            event("ASR_FINAL", "s", text="I am going to the store today. And then home.", committed_tokens=7),
        ]
        code, out, _ = run_main(lines)
        self.assertEqual(code, 0)
        self.assertIn("OK                    1", out)

    def test_resubmitted_segment_keeps_the_first_text(self):
        lines = [
            event("STREAM_OPEN", "t", utt=None, task="realtime_translation"),
            event("TRANSLATE_SUBMIT", "t", segment_idx=0, source_text="One two three four five six."),
            event("TRANSLATE_SUBMIT", "t", segment_idx=0, source_text="One two three four five six seven."),
            event("ASR_FINAL", "t", text="One two three four five six. Seven.", committed_tokens=6),
            event("TRANSLATE_SUBMIT", "t", segment_idx=1, source_text="Seven."),
        ]
        _, out, _ = run_main(lines, "--verbose")
        self.assertIn("sent #0: One two three four five six.\n", out)
        self.assertIn("[OK]", out)

    def test_single_mid_utterance_commit_is_segmented_through_committed_tokens(self):
        # One commit before FINAL and an empty remainder: one submit, so only committed_tokens tells it apart.
        lines = [
            event("STREAM_OPEN", "t", utt=None, task="realtime_translation"),
            event("TRANSLATE_SUBMIT", "t", segment_idx=0, source_text="We will meet at nine thirty."),
            event("ASR_FINAL", "t", text="We'll meet at nine thirty.", committed_tokens=6),
        ]
        code, out, _ = run_main(lines)
        self.assertEqual(code, 0)
        self.assertIn("segmented (text sent before ASR_FINAL): 1", out)
        self.assertIn("REVISED               1", out)

    def test_translation_stream_never_compares_translated_emits(self):
        # STREAM_OPEN says translation: without TRANSLATE_SUBMIT lines nothing is compared (EMIT text is the target).
        lines = [
            event("STREAM_OPEN", "t", utt=None, task="realtime_translation"),
            event("EMIT", "t", seq=1, segment_idx=0, is_final=True, text="Chúng ta gặp lúc chín giờ rưỡi."),
            event("ASR_FINAL", "t", text="We will meet at nine thirty. See you.", committed_tokens=6),
            event("EMIT", "t", seq=2, segment_idx=1, is_final=True, text="Hẹn gặp."),
        ]
        code, out, err = run_main(lines)
        self.assertEqual(code, 1)
        self.assertIn("segmented but no sent text found 1", out)
        self.assertIn("No segmented utterance", err)

    def test_without_committed_tokens_segmentation_is_inferred(self):
        tasks = {}
        stt = cso.Utterance(key=("r", "s", "MO", 1), has_final=True, final_text="a b")
        stt.stt_finals = [(1, 0, "a"), (2, 1, "b")]
        self.assertTrue(cso.is_segmented(stt, cso.utterance_task(stt, tasks)))
        plain = cso.Utterance(key=("r", "s", "MO", 2), has_final=True, final_text="a")
        plain.stt_finals = [(3, None, "a")]
        self.assertFalse(cso.is_segmented(plain, cso.TASK_STT))
        tr = cso.Utterance(key=("r", "t", "MO", 1), has_final=True, final_text="a b")
        tr.submits = [(0, "a")]
        self.assertFalse(cso.is_segmented(tr, cso.TASK_TRANSLATION))
        tr.submits.append((1, "b"))
        self.assertTrue(cso.is_segmented(tr, cso.TASK_TRANSLATION))

    def test_empty_skipped_and_non_final_emits_are_ignored(self):
        lines = [
            event("EMIT", "s", seq=1, segment_idx=0, is_final=True, text="One two three four five six."),
            event("EMIT", "s", seq=2, is_final=False, text="Seven eight"),
            event("ASR_FINAL", "s", text="One two three four five six.", committed_tokens=6),
            event("EMIT", "s", seq=3, segment_idx=1, is_final=True, skipped="empty"),
        ]
        code, out, _ = run_main(lines)
        self.assertEqual(code, 0)
        self.assertIn("OK                    1", out)

    def test_segment_align_lines_are_collected(self):
        lines = [
            *stt_utterance("s", 1, ["I'm going to the store today.", "And then home."],
                           "I am going to the store today. And then home."),
            event("SEGMENT_ALIGN", "s", how="aligned", phase="partial"),
            event("SEGMENT_ALIGN", "s", how="aligned", phase="final"),
        ]
        _, out, _ = run_main(lines, "--verbose")
        self.assertIn("SEGMENT_ALIGN seen in 1 utterance(s) (aligned 2)", out)
        self.assertIn("align=partial:aligned,final:aligned", out)


class EngineLogTest(unittest.IsolatedAsyncioTestCase):
    """Real coordinator, real JSON formatter: the script must read the lines the engine writes."""

    FIRST = "I'm going to the store today."
    PARTIALS = [FIRST, FIRST, "I am going to the store today. And then"]
    FINAL = "I am going to the store today. And then home."

    async def _engine_log(self, path):
        from nemotron_testbed.coordinator import EngineCoordinator
        from nemotron_testbed.domain import RuntimeResult, RuntimeResultType
        from nemotron_testbed.logging_config import JsonFormatter
        from test_coordinator_flow import FakeRuntime, make_settings, request

        settings = dataclasses.replace(make_settings(False, segmentation_mode="on"), segmentation_speech_to_text=True)
        coordinator = EngineCoordinator(settings)
        asr = FakeRuntime()
        coordinator.runtime = asr
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        engine_logger = logging.getLogger("nemotron_testbed.coordinator")
        old_level = engine_logger.level
        engine_logger.addHandler(handler)
        engine_logger.setLevel(logging.DEBUG)
        try:
            rpc = coordinator.open_rpc()
            await coordinator.accept_chunk(rpc.rpc_id, request(0, language="en"))
            key = asr.commands[-1].key
            await coordinator._handle_runtime_result(
                RuntimeResult(type=RuntimeResultType.STARTED, key=key, utterance_id=1, detected_language="en-US")
            )
            for text in self.PARTIALS:
                await coordinator._handle_runtime_result(
                    RuntimeResult(
                        type=RuntimeResultType.PARTIAL, key=key, utterance_id=1, text=text,
                        detected_language="en-US", metadata={"model_step_executed": True},
                    )
                )
            await coordinator.accept_chunk(rpc.rpc_id, request(160, end=True, language="en"))
            await coordinator._handle_runtime_result(
                RuntimeResult(
                    type=RuntimeResultType.FINAL, key=key, utterance_id=1, text=self.FINAL, detected_language="en-US"
                )
            )
        finally:
            engine_logger.removeHandler(handler)
            engine_logger.setLevel(old_level)
            handler.close()

    async def _check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            await self._engine_log(path)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = cso.main([str(path), "--verbose"])
        return code, out.getvalue()

    async def test_current_engine_output_has_no_cut_error(self):
        code, out = await self._check()
        self.assertEqual(code, 0)
        self.assertIn("segmented (text sent before ASR_FINAL): 1 (speech_to_text 1, realtime_translation 0)", out)
        self.assertIn("[REVISED] session=call-1 stream=MO utterance=1 task=speech_to_text", out)
        self.assertIn("cut errors (any CUT_LOSS or CUT_DUP word): 0 of 1", out)
        self.assertIn("align=partial:aligned,final:aligned", out)

    async def test_token_count_remainder_is_reported_as_cut_dup(self):
        def token_count_remainder(committed_text, text):
            return " ".join(text.split()[len(committed_text.split()) :])

        with patch("nemotron_testbed.coordinator.remainder_after", token_count_remainder):
            code, out = await self._check()
        self.assertEqual(code, 0)
        self.assertIn("[CUT_DUP] session=call-1", out)
        self.assertIn("sent #1: today. And then home.", out)


if __name__ == "__main__":
    unittest.main()
