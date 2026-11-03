import unittest

import numpy as np

from nemotron_testbed.buffers import SampleBuffer


class SampleBufferTest(unittest.TestCase):
    def test_fifo_and_partial_pop(self):
        buffer = SampleBuffer()
        buffer.append(np.array([1, 2, 3], dtype=np.float32))
        buffer.append(np.array([4, 5], dtype=np.float32))
        np.testing.assert_array_equal(
            buffer.pop_exact(4), np.array([1, 2, 3, 4], dtype=np.float32)
        )
        self.assertEqual(buffer.samples, 1)
        np.testing.assert_array_equal(
            buffer.pop_all(), np.array([5], dtype=np.float32)
        )

    def test_bounded_buffer_keeps_latest_samples(self):
        buffer = SampleBuffer(max_samples=3)
        buffer.append(np.array([1, 2], dtype=np.float32))
        buffer.append(np.array([3, 4], dtype=np.float32))
        np.testing.assert_array_equal(
            buffer.to_array(), np.array([2, 3, 4], dtype=np.float32)
        )

    def test_prepend_restores_front(self):
        # prepend() is used for STEP rollback when the ASR queue is full.
        buffer = SampleBuffer()
        buffer.append(np.array([3, 4, 5], dtype=np.float32))
        buffer.prepend(np.array([1, 2], dtype=np.float32))
        np.testing.assert_array_equal(
            buffer.pop_all(), np.array([1, 2, 3, 4, 5], dtype=np.float32)
        )
        self.assertEqual(buffer.samples, 0)

    def test_prepend_updates_samples_count(self):
        buffer = SampleBuffer()
        buffer.prepend(np.array([10, 20], dtype=np.float32))
        self.assertEqual(buffer.samples, 2)

    def test_append_does_not_share_storage_with_caller(self):
        # Buffer must own its data — mutating the source array must not corrupt buffer.
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buffer = SampleBuffer()
        buffer.append(source)
        source[:] = 99.0
        np.testing.assert_array_equal(
            buffer.pop_all(), np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    def test_pop_exact_raises_on_underflow(self):
        buffer = SampleBuffer()
        buffer.append(np.array([1, 2], dtype=np.float32))
        with self.assertRaises(ValueError):
            buffer.pop_exact(5)

    def test_empty_append_ignored(self):
        buffer = SampleBuffer()
        buffer.append(np.empty(0, dtype=np.float32))
        self.assertEqual(buffer.samples, 0)


if __name__ == "__main__":
    unittest.main()
