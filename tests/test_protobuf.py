import unittest

from nemotron_testbed.pb import speech_pb2


class ProtobufTest(unittest.TestCase):
    def test_roundtrip_new_audio_fields(self):
        value = speech_pb2.RecognitionResult(
            session_id="call-1",
            stream_id="MO",
            text="Hello",
            is_final=True,
            ts_start=1,
            ts_end=2,
            confidence=0.87,
            language="en-US",
            seq=3,
            audio_payload=b"\x01\x02\x03",
            audio_pt=8,
        )
        decoded = speech_pb2.RecognitionResult.FromString(value.SerializeToString())
        self.assertEqual(decoded.session_id, "call-1")
        self.assertEqual(decoded.stream_id, "MO")
        self.assertEqual(decoded.text, "Hello")
        self.assertTrue(decoded.is_final)
        self.assertEqual(decoded.ts_start, 1)
        self.assertEqual(decoded.ts_end, 2)
        self.assertAlmostEqual(decoded.confidence, 0.87, places=6)
        self.assertEqual(decoded.language, "en-US")
        self.assertEqual(decoded.seq, 3)
        self.assertEqual(decoded.audio_payload, b"\x01\x02\x03")
        self.assertEqual(decoded.audio_pt, 8)


if __name__ == "__main__":
    unittest.main()
