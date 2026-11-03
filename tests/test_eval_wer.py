import importlib.util
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "eval_wer.py"
_spec = importlib.util.spec_from_file_location("eval_wer", _SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


class NormalizeTextTest(unittest.TestCase):
    def test_lowercase_and_punctuation_removed(self):
        self.assertEqual(ev.normalize_text("Dạ, thưa giám đốc! Có không?"), ["dạ", "thưa", "giám", "đốc", "có", "không"])

    def test_decomposed_and_composed_vietnamese_match(self):
        composed = "Việt Nam"
        decomposed = "Việt Nam"
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(ev.normalize_text(composed), ev.normalize_text(decomposed))

    def test_digits_are_kept_as_written(self):
        self.assertEqual(ev.normalize_text("ngày 15 tháng 2"), ["ngày", "15", "tháng", "2"])

    def test_underscore_and_hyphen_split_words(self):
        self.assertEqual(ev.normalize_text("a_b c-d"), ["a", "b", "c", "d"])

    def test_empty_and_none(self):
        self.assertEqual(ev.normalize_text(""), [])
        self.assertEqual(ev.normalize_text(None), [])


class EditDistanceTest(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(ev.edit_distance(["a", "b"], ["a", "b"]), 0)

    def test_substitution_insertion_deletion(self):
        self.assertEqual(ev.edit_distance(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(ev.edit_distance(["a", "b"], ["a", "b", "c"]), 1)
        self.assertEqual(ev.edit_distance(["a", "b", "c"], ["a", "c"]), 1)

    def test_empty_sides(self):
        self.assertEqual(ev.edit_distance([], ["a", "b"]), 2)
        self.assertEqual(ev.edit_distance(["a", "b"], []), 2)
        self.assertEqual(ev.edit_distance([], []), 0)


class WerTest(unittest.TestCase):
    def test_wer_pairs_skips_rows_without_reference(self):
        rows = [
            {"id": "a", "ref": "xin chào các bạn", "hyp": "xin chào bạn"},
            {"id": "b", "ref": "", "hyp": "anything"},
        ]
        self.assertEqual(ev.wer_pairs(rows), {"a": (1, 4)})

    def test_wer_pairs_ignores_case_and_punctuation(self):
        self.assertEqual(ev.wer_pairs([{"id": "a", "ref": "Xin chào.", "hyp": "xin chào"}]), {"a": (0, 2)})

    def test_empty_hypothesis_counts_every_reference_word(self):
        self.assertEqual(ev.wer_pairs([{"id": "a", "ref": "một hai ba", "hyp": ""}]), {"a": (3, 3)})

    def test_corpus_wer_weights_by_reference_length(self):
        self.assertAlmostEqual(ev.corpus_wer([(1, 4), (0, 6)]), 0.1)

    def test_corpus_wer_no_words(self):
        self.assertEqual(ev.corpus_wer([]), 0.0)

    def test_build_hypothesis_joins_non_empty_segments(self):
        self.assertEqual(ev.build_hypothesis(["Một câu.", "", "  Câu hai  "]), "Một câu. Câu hai")


class PairedBootstrapTest(unittest.TestCase):
    def test_identical_runs_have_zero_delta_and_zero_ci(self):
        pairs = [(1, 10), (2, 10), (0, 10), (3, 10)]
        self.assertEqual(ev.paired_bootstrap(pairs, list(pairs)), (0.0, 0.0, 0.0))

    def test_clear_improvement_has_negative_interval(self):
        base = [(3, 10)] * 50
        better = [(1, 10)] * 50
        delta, low, high = ev.paired_bootstrap(base, better)
        self.assertAlmostEqual(delta, -0.2)
        self.assertLess(high, 0.0)
        self.assertLessEqual(low, delta)

    def test_noise_interval_contains_zero(self):
        base = [(i % 3, 10) for i in range(60)]
        other = [((i + 1) % 3, 10) for i in range(60)]
        _, low, high = ev.paired_bootstrap(base, other)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_deterministic_for_a_seed(self):
        base = [(i % 4, 10) for i in range(40)]
        other = [((i * 3) % 4, 10) for i in range(40)]
        self.assertEqual(ev.paired_bootstrap(base, other, seed=7), ev.paired_bootstrap(base, other, seed=7))

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            ev.paired_bootstrap([(1, 2)], [(1, 2), (0, 2)])

    def test_empty(self):
        self.assertEqual(ev.paired_bootstrap([], []), (0.0, 0.0, 0.0))


class ManifestTest(unittest.TestCase):
    def test_roundtrip_keeps_tabs_quotes_and_vietnamese(self):
        rows = [
            {
                "id": "vi-000",
                "wav": "/x/vi-000.wav",
                "duration_s": "3.20",
                "content_type": "Phone Call",
                "reference": 'Anh nói "dạ",\ttiếp tục',
                "lang": "vi",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "m.tsv"
            ev.save_manifest(path, rows)
            self.assertEqual(ev.load_manifest(path), rows)

    def test_interleave_alternates_languages_and_numbers_english_rows(self):
        vi = [{"id": f"vi-{i}", "lang": "vi", "reference": "x"} for i in range(3)]
        en = {"id": "en", "lang": "en", "reference": ""}
        mixed = ev.interleave_manifest(vi, en, 2)
        self.assertEqual([r["lang"] for r in mixed], ["vi", "en", "vi", "en"])
        self.assertEqual([r["id"] for r in mixed], ["vi-0", "en-000", "vi-1", "en-001"])

    def test_resolve_language_source(self):
        row = {"lang": "en"}
        self.assertEqual(ev.resolve_language_source("@lang", row), "en")
        self.assertEqual(ev.resolve_language_source("vi", row), "vi")
        self.assertEqual(ev.resolve_language_source("", row), "")


def _wav_bytes(samples: np.ndarray, rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


class DecodeAudioTest(unittest.TestCase):
    def test_mono_wav(self):
        samples = np.array([0, 16384, -16384, 8192], dtype=np.int16)
        audio, rate = ev.decode_audio(_wav_bytes(samples, 16_000))
        self.assertEqual(rate, 16_000)
        np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-6)

    def test_stereo_is_averaged_to_mono(self):
        stereo = np.array([1000, 3000, -2000, -4000], dtype=np.int16)
        audio, rate = ev.decode_audio(_wav_bytes(stereo, 8_000, channels=2))
        self.assertEqual(rate, 8_000)
        np.testing.assert_allclose(audio, np.array([2000, -3000]) / 32768.0, atol=1e-6)


class ClipDiffTest(unittest.TestCase):
    BASE = [
        {"id": "a", "ref": "một hai ba bốn", "hyp": "một hai ba bốn"},
        {"id": "b", "ref": "năm sáu bảy", "hyp": "năm sáu"},
        {"id": "c", "ref": "tám chín", "hyp": "tám chín"},
        {"id": "d", "ref": "", "hyp": "no reference"},
    ]
    RUN = [
        {"id": "a", "ref": "một hai ba bốn", "hyp": "một ba"},
        {"id": "b", "ref": "năm sáu bảy", "hyp": "Năm, sáu, bảy."},
        {"id": "c", "ref": "tám chín", "hyp": "tám chín"},
        {"id": "d", "ref": "", "hyp": "other"},
        {"id": "e", "ref": "only in run", "hyp": "only in run"},
    ]

    def test_errors_per_clip_and_worst_first(self):
        diffs = ev.clip_diffs(self.BASE, self.RUN)
        self.assertEqual(
            [(d["id"], d["base_errors"], d["errors"], d["delta"]) for d in diffs],
            [("a", 0, 2, 2), ("c", 0, 0, 0), ("b", 1, 0, -1)],
        )
        self.assertEqual(diffs[0]["base_hyp"], "một hai ba bốn")
        self.assertEqual(diffs[0]["hyp"], "một ba")

    def test_clips_without_reference_or_missing_in_base_are_skipped(self):
        ids = {d["id"] for d in ev.clip_diffs(self.BASE, self.RUN)}
        self.assertNotIn("d", ids)
        self.assertNotIn("e", ids)


def _log_line(stage, session, **fields):
    return json.dumps({"level": "DEBUG", "message": f"PIPELINE {stage}", "stage": stage, "session_id": session, **fields})


class LogStatsTest(unittest.TestCase):
    LINES = [
        "not json at all VAD_START",
        _log_line("VAD_START", "B-vi-000-aa11"),
        _log_line("ASR_FINAL", "B-vi-000-aa11", text="xin chào", text_len=8, utterance_audio_ms=1200.0),
        _log_line("VAD_START", "B-vi-000-aa11"),
        _log_line("ASR_FINAL", "B-vi-000-aa11", text="", text_len=0, utterance_audio_ms=600.0),
        _log_line("VAD_START", "B-vi-001-bb22"),
        _log_line("VAD_END", "B-vi-001-bb22", reason="max_utterance"),
        _log_line("ASR_FINAL", "B-vi-001-bb22", text_len=12, utterance_audio_ms=10000.0),
        _log_line("VAD_START", "S600-vi-000-cc33"),
        _log_line("VAD_END", "S600-vi-000-cc33"),
        _log_line("ASR_FINAL", "S600-vi-000-cc33", text="xin chào bạn", text_len=12, utterance_audio_ms=2400.0),
        _log_line("VAD_START", "A-vi-000-dd44"),
        _log_line("ASR_FINAL", "A-vi-000-dd44", text="", text_len=0, utterance_audio_ms=500.0),
        _log_line("EMIT", "B-vi-000-aa11", text="xin chào"),
        '{"message": "STATS", "stats": {"VAD_START": 3}}',
        '["VAD_START", "not an object"]',
    ]

    def _by_run(self, tags):
        return {row["run"]: row for row in ev.log_stats(self.LINES, tags)}

    def test_counts_per_run_tag(self):
        rows = self._by_run(["B", "S600"])
        b, s = rows["B"], rows["S600"]
        self.assertEqual((b["clips"], b["utterances"], b["utt_per_clip"]), (2, 3, "1.50"))
        self.assertEqual((b["empty_asr"], b["empty_pct"], b["max_utt_cuts"]), (1, "33.3", 1))
        self.assertEqual(b["utt_ms_p50/p90"], "1200/8240")
        self.assertEqual((s["clips"], s["utterances"], s["empty_asr"], s["max_utt_cuts"]), (1, 1, 0, 0))

    def test_runs_not_asked_for_are_ignored_and_order_is_kept(self):
        table = ev.log_stats(self.LINES, ["S600", "B", "B"])
        self.assertEqual([row["run"] for row in table], ["S600", "B"])

    def test_empty_text_without_text_len_still_counts_as_empty(self):
        lines = [_log_line("VAD_START", "X-1-a"), _log_line("ASR_FINAL", "X-1-a", text="  ")]
        self.assertEqual(ev.log_stats(lines, ["X"])[0]["empty_asr"], 1)

    def test_tag_without_events_reports_dashes(self):
        row = ev.log_stats(self.LINES, ["S800"])[0]
        self.assertEqual((row["clips"], row["utterances"], row["utt_per_clip"], row["empty_pct"]), (0, 0, "-", "-"))
        self.assertEqual(row["utt_ms_p50/p90"], "-")


if __name__ == "__main__":
    unittest.main()
