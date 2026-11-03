import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_segmentation.py"
_spec = importlib.util.spec_from_file_location("check_segmentation", _SCRIPT)
seg = importlib.util.module_from_spec(_spec)
sys.modules["check_segmentation"] = seg
_spec.loader.exec_module(seg)


def partial(step, t_ms, text=None, *, changed=True, model_step=True, utt="u1", session="s1", stream="MO", rpc="r1"):
    ev = {
        "stage": "ASR_PARTIAL",
        "rpc_id": rpc,
        "session_id": session,
        "stream_id": stream,
        "utterance_id": utt,
        "step": step,
        "model_step": model_step,
        "changed": changed,
        "t_ms": t_ms,
    }
    if text is not None:
        ev["text"] = text
    return ev


def final(text, t_ms, *, utt="u1", session="s1", stream="MO", rpc="r1", language="vi-VN"):
    return {
        "stage": "ASR_FINAL",
        "rpc_id": rpc,
        "session_id": session,
        "stream_id": stream,
        "utterance_id": utt,
        "source_language": language,
        "text": text,
        "t_ms": t_ms,
    }


def write_log(directory, events, name="engine.log"):
    path = Path(directory) / name
    lines = ["plain text line that is not json"]
    lines.extend(json.dumps(ev, ensure_ascii=False) for ev in events)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


STABLE_EVENTS = [
    partial(1, 560, "Xin chào các bạn"),
    partial(2, 1120, "Xin chào các bạn hôm nay."),
    partial(3, 1680, changed=False),
    partial(4, 2240, "Xin chào các bạn hôm nay. Chúng ta bắt đầu"),
    final("Xin chào các bạn hôm nay. Chúng ta bắt đầu buổi họp", 3000),
]


class LoadTest(unittest.TestCase):
    def test_load_groups_partials_and_final_and_carries_text_over_unchanged_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, STABLE_EVENTS)
            utterances, seen, no_text = seg.load_utterances([path])
        self.assertEqual((seen, no_text), (4, 0))
        self.assertEqual(len(utterances), 1)
        self.assertIsNotNone(utterances[0].final)
        self.assertEqual(utterances[0].observations[2].text, "Xin chào các bạn hôm nay.")

    def test_filters_by_session(self):
        events = STABLE_EVENTS + [
            partial(1, 500, "Hello there", utt="u9", session="other"),
            final("Hello there", 900, utt="u9", session="other"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, events)
            utterances, seen, _ = seg.load_utterances([path], session_id="other")
        self.assertEqual(seen, 1)
        self.assertEqual(utterances[0].key[3], "u9")

    def test_counts_partials_without_text(self):
        events = [partial(1, 500, None), final("a b", 900)]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, events)
            _, seen, no_text = seg.load_utterances([path])
        self.assertEqual((seen, no_text), (1, 1))


class SimulateTest(unittest.TestCase):
    def _utt(self, events):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, events)
            utterances, _, _ = seg.load_utterances([path])
        return utterances[0]

    def test_commits_sentence_after_two_stable_steps_and_rest_at_utterance_end(self):
        utt = self._utt(STABLE_EVENTS)
        commits = seg.simulate(utt, seg.Params())
        self.assertEqual([c.kind for c in commits], ["punct", "utterance_end"])
        self.assertEqual(commits[0].text, "Xin chào các bạn hôm nay.")
        self.assertEqual(commits[0].step, 3)
        self.assertEqual(commits[0].t_ms, 1680)
        self.assertEqual(commits[1].text, "Chúng ta bắt đầu buổi họp")

    def test_stable_n_one_commits_on_first_sight(self):
        utt = self._utt(STABLE_EVENTS)
        commits = seg.simulate(utt, seg.Params(stable_n=1))
        self.assertEqual(commits[0].step, 2)

    def test_punct_stable_n_one_commits_on_first_sight_while_stable_n_stays(self):
        utt = self._utt(STABLE_EVENTS)
        commits = seg.simulate(utt, seg.Params(stable_n=3, punct_stable_n=1))
        self.assertEqual(commits[0].step, 2)

    def test_higher_stable_n_commits_later(self):
        utt = self._utt(STABLE_EVENTS)
        self.assertEqual(seg.simulate(utt, seg.Params(stable_n=3))[0].step, 4)
        commits = seg.simulate(utt, seg.Params(stable_n=4))
        self.assertEqual([c.kind for c in commits], ["utterance_end"])

    def test_non_model_steps_do_not_count_as_stable(self):
        events = [
            partial(1, 560, "Xin chào các bạn hôm nay."),
            partial(2, 700, changed=False, model_step=False),
            final("Xin chào các bạn hôm nay.", 1200),
        ]
        commits = seg.simulate(self._utt(events), seg.Params())
        self.assertEqual([c.kind for c in commits], ["utterance_end"])

    def test_stable_ms_commits_by_elapsed_time(self):
        events = [
            partial(1, 560, "Xin chào các bạn hôm nay."),
            partial(2, 1000, changed=False, model_step=False),
            final("Xin chào các bạn hôm nay.", 3000),
        ]
        commits = seg.simulate(self._utt(events), seg.Params(stable_ms=400))
        self.assertEqual(commits[0].kind, "punct")
        self.assertEqual(commits[0].t_ms, 1000)

    def test_min_tokens_blocks_short_sentence(self):
        events = [
            partial(1, 560, "Xin chào."),
            partial(2, 1120, changed=False),
            final("Xin chào.", 2000),
        ]
        utt = self._utt(events)
        self.assertEqual(seg.simulate(utt, seg.Params())[0].kind, "utterance_end")
        self.assertEqual(seg.simulate(utt, seg.Params(min_tokens=1))[0].kind, "punct")

    def test_guard_word_at_end_blocks_commit(self):
        events = [
            partial(1, 560, "Tôi đi học vì."),
            partial(2, 1120, changed=False),
            final("Tôi đi học vì trời đẹp.", 2000),
        ]
        commits = seg.simulate(self._utt(events), seg.Params())
        self.assertEqual([c.kind for c in commits], ["utterance_end"])

    def test_max_segment_fallback_commits_unpunctuated_text(self):
        events = [
            partial(1, 0, "một hai ba bốn năm"),
            partial(2, 9000, "một hai ba bốn năm sáu"),
            final("một hai ba bốn năm sáu bảy", 9500),
        ]
        commits = seg.simulate(self._utt(events), seg.Params(max_segment_ms=8000))
        self.assertEqual([c.kind for c in commits], ["max_segment", "utterance_end"])
        self.assertEqual(commits[1].text, "bảy")


class EvaluateTest(unittest.TestCase):
    def _run(self, events, params=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, events)
            utterances, _, _ = seg.load_utterances([path])
        results, skipped = seg.analyse(utterances, params or seg.Params())
        return results, skipped

    def test_stable_commit_is_not_premature_nor_revised(self):
        results, skipped = self._run(STABLE_EVENTS)
        commits = results[0][1]
        self.assertEqual(skipped, 0)
        self.assertFalse(commits[0].premature)
        self.assertFalse(commits[0].revised)

    def test_commit_missing_from_final_is_premature(self):
        events = [
            partial(1, 560, "Hôm nay trời đẹp lắm."),
            partial(2, 1120, changed=False),
            final("Hôm qua trời mưa to.", 2000),
        ]
        commit = self._run(events)[0][0][1][0]
        self.assertEqual(commit.kind, "punct")
        self.assertTrue(commit.premature)

    def test_later_change_inside_committed_region_is_revised(self):
        events = [
            partial(1, 560, "Hôm nay trời đẹp lắm."),
            partial(2, 1120, changed=False),
            partial(3, 1680, "Hôm qua trời đẹp lắm."),
            final("Hôm nay trời đẹp lắm.", 2000),
        ]
        commit = self._run(events)[0][0][1][0]
        self.assertTrue(commit.revised)
        self.assertFalse(commit.premature)

    def test_utterance_without_final_is_skipped(self):
        results, skipped = self._run([partial(1, 560, "Xin chào các bạn.")])
        self.assertEqual((results, skipped), ([], 1))


class SummaryTest(unittest.TestCase):
    def test_summary_gain_and_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, STABLE_EVENTS)
            utterances, _, _ = seg.load_utterances([path])
        params = seg.Params()
        results, _ = seg.analyse(utterances, params)
        summary = seg.summarize(results)
        self.assertEqual(summary["punct_commits"], 1)
        self.assertEqual(summary["premature_rate"], 0.0)
        self.assertEqual(summary["gain_mean_ms"], 3000 - 1680)
        self.assertIn("premature commit rate: 0.0%", seg.render_summary(summary, params))
        self.assertIn("PREMATURE" if summary["premature"] else "ok", seg.render_verbose(results))
        sweep = seg.render_sweep(utterances, params)
        self.assertEqual(len(sweep.splitlines()), 19)


class MainTest(unittest.TestCase):
    def _main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = seg.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_main_reports_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, STABLE_EVENTS)
            code, out, _ = self._main([path, "--verbose"])
        self.assertEqual(code, 0)
        self.assertIn("utterances analysed: 1", out)
        self.assertIn("utterance_end", out)

    def test_main_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, STABLE_EVENTS)
            code, out, _ = self._main([path, "--sweep"])
        self.assertEqual(code, 0)
        self.assertIn("stable", out)
        self.assertIn("ms=800", out)

    def test_main_exit_1_without_partials(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, [final("a b c", 100)])
            code, _, err = self._main([path])
        self.assertEqual(code, 1)
        self.assertIn("pipeline_trace_partial", err)

    def test_main_exit_2_when_text_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, [partial(1, 100, None), final("a b c", 200)])
            code, _, err = self._main([path])
        self.assertEqual(code, 2)
        self.assertIn("pipeline_trace_text", err)


    def test_missing_file_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self._main([str(Path(tmp) / "nope.log")])
        self.assertEqual(code, 1)
        self.assertIn("No log file matches", err)

    def test_filters_excluded_everything_lists_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_log(tmp, [partial(1, 100, "a b"), final("a b c", 200)])
            code, _, err = self._main([path, "--stream-id", "XX"])
        self.assertEqual(code, 1)
        self.assertIn("none match the filters", err)
        self.assertIn("s1", err)


if __name__ == "__main__":
    unittest.main()
