import math
import unittest
from types import SimpleNamespace

from nemotron_testbed.runtime import NemotronRuntime


class ConfidenceTest(unittest.TestCase):
    def test_prefers_native_token_confidence(self):
        hypothesis = SimpleNamespace(
            text="xin chao",
            token_confidence=[0.9, 0.8, 0.7],
            word_confidence=[0.2],
            score=-20.0,
            y_sequence=[1, 2, 3],
        )
        confidence, source = NemotronRuntime._confidence_from_hypothesis(hypothesis)
        self.assertAlmostEqual(confidence, 0.8, places=6)
        self.assertEqual(source, "nemo_token_confidence")

    def test_uses_non_blank_frame_confidence_when_available(self):
        class Hypothesis:
            text = "hello"
            token_confidence = None
            word_confidence = None
            frame_confidence = [[0.1, 0.2], [0.3]]
            score = -10.0
            y_sequence = [1]

            @property
            def non_blank_frame_confidence(self):
                return [0.95, 0.85]

        confidence, source = NemotronRuntime._confidence_from_hypothesis(Hypothesis())
        self.assertAlmostEqual(confidence, 0.9, places=6)
        self.assertEqual(source, "nemo_non_blank_frame_confidence")

    def test_falls_back_to_length_normalized_rnnt_score(self):
        target = 0.8
        hypothesis = SimpleNamespace(
            text="hello world",
            token_confidence=None,
            word_confidence=None,
            frame_confidence=None,
            score=2 * math.log(target),
            y_sequence=[11, 22],
        )
        confidence, source = NemotronRuntime._confidence_from_hypothesis(hypothesis)
        self.assertAlmostEqual(confidence, target, places=6)
        self.assertEqual(source, "rnnt_score_geomean")

    def test_zero_or_missing_score_is_unavailable_not_perfect(self):
        hypothesis = SimpleNamespace(
            text="hello",
            token_confidence=None,
            word_confidence=None,
            frame_confidence=None,
            score=0.0,
            y_sequence=[1],
        )
        confidence, source = NemotronRuntime._confidence_from_hypothesis(hypothesis)
        self.assertEqual(confidence, 0.0)
        self.assertEqual(source, "unavailable")


if __name__ == "__main__":
    unittest.main()
