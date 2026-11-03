from __future__ import annotations

from collections import deque

import numpy as np


class SampleBuffer:
    """FIFO buffer for mono float32 samples without repeated full-array copies."""

    def __init__(self, max_samples: int | None = None):
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._max_samples = max_samples

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def max_samples(self) -> int | None:
        return self._max_samples

    def clear(self) -> None:
        self._chunks.clear()
        self._samples = 0

    def append(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        # .copy() ensures the buffer owns its storage — callers may reuse the
        # source array (e.g. rolling pre-roll) without corrupting buffered data.
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        self._chunks.append(chunk)
        self._samples += chunk.size
        if self._max_samples is not None:
            self._trim_left(max(0, self._samples - self._max_samples))

    def _trim_left(self, count: int) -> None:
        remaining = count
        while remaining > 0 and self._chunks:
            head = self._chunks[0]
            if head.size <= remaining:
                self._chunks.popleft()
                self._samples -= head.size
                remaining -= head.size
            else:
                self._chunks[0] = head[remaining:].copy()
                self._samples -= remaining
                remaining = 0

    def pop_exact(self, count: int) -> np.ndarray:
        if count < 0 or count > self._samples:
            raise ValueError(f"cannot pop {count} samples from buffer with {self._samples}")
        if count == 0:
            return np.empty(0, dtype=np.float32)
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            head = self._chunks[0]
            if head.size <= remaining:
                pieces.append(self._chunks.popleft())
                self._samples -= head.size
                remaining -= head.size
            else:
                pieces.append(head[:remaining].copy())
                self._chunks[0] = head[remaining:].copy()
                self._samples -= remaining
                remaining = 0
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def prepend(self, audio: np.ndarray) -> None:
        """Return audio to the front of the buffer (used for submit-fail rollback)."""
        if audio.size:
            chunk = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
            self._chunks.appendleft(chunk)
            self._samples += chunk.size

    def pop_all(self) -> np.ndarray:
        return self.pop_exact(self._samples)

    def to_array(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(list(self._chunks)).astype(np.float32, copy=False)
