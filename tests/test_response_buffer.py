import unittest

from nemotron_testbed.coordinator import AsyncResponseBuffer, RecognitionEnvelope


def envelope(text: str, final: bool, stream: str = "MO") -> RecognitionEnvelope:
    return RecognitionEnvelope(
        session_id="s",
        stream_id=stream,
        text=text,
        is_final=final,
        ts_start=0,
        ts_end=1,
        confidence=0.0,
        language="vi-VN",
        seq=1,
        utterance_id=1,
    )


class ResponseBufferTest(unittest.IsolatedAsyncioTestCase):
    async def test_coalesces_partial_but_keeps_final(self):
        buffer = AsyncResponseBuffer(soft_limit=1)
        self.assertTrue(buffer.put(envelope("old", False)))
        self.assertTrue(buffer.put(envelope("new", False)))
        self.assertTrue(buffer.put(envelope("final", True)))

        first = await buffer.get()
        second = await buffer.get()
        self.assertEqual(first.text, "new")
        self.assertTrue(second.is_final)


if __name__ == "__main__":
    unittest.main()
