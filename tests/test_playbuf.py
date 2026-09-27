"""Unit tests for _PlayBuf — the bounded playback deque."""

from __future__ import annotations

import numpy as np

from kowairo.audio.pipeline import _PlayBuf


def _fill(buf: _PlayBuf, n: int) -> np.ndarray:
    out = np.zeros(n, np.float32)
    buf.drain(out)
    return out


class TestPushDrain:
    def test_drain_empty_fills_silence(self):
        buf = _PlayBuf()
        out = _fill(buf, 32)
        assert np.all(out == 0.0)
        assert buf.pending == 0

    def test_roundtrip(self):
        buf = _PlayBuf()
        buf.push(np.ones(10, np.float32), cap=1024)
        out = _fill(buf, 10)
        assert np.all(out == 1.0)
        assert buf.pending == 0

    def test_drain_shorter_than_chunk_then_zero(self):
        buf = _PlayBuf()
        buf.push(np.ones(4, np.float32), cap=1024)
        out = _fill(buf, 8)
        assert np.all(out[:4] == 1.0)
        assert np.all(out[4:] == 0.0)

    def test_chunk_continues_across_drains(self):
        buf = _PlayBuf()
        buf.push(np.arange(10, dtype=np.float32), cap=1024)
        assert np.array_equal(_fill(buf, 4), np.arange(4))
        assert np.array_equal(_fill(buf, 6), np.arange(4, 10))

    def test_multiple_chunks_in_order(self):
        buf = _PlayBuf()
        buf.push(np.full(3, 1.0, np.float32), cap=1024)
        buf.push(np.full(3, 2.0, np.float32), cap=1024)
        out = _fill(buf, 6)
        assert np.array_equal(out, [1, 1, 1, 2, 2, 2])


class TestCapDrop:
    def test_partial_head_trim_keeps_newest(self):
        buf = _PlayBuf()
        dropped = buf.push(np.arange(100, dtype=np.float32), cap=50)
        assert dropped == 50
        assert buf.pending == 50
        out = _fill(buf, 50)
        # drop-to-live-edge: the oldest 50 samples are trimmed
        assert np.array_equal(out, np.arange(50, 100))

    def test_whole_chunks_dropped_first(self):
        buf = _PlayBuf()
        buf.push(np.full(60, 1.0, np.float32), cap=100)
        dropped = buf.push(np.full(60, 2.0, np.float32), cap=100)
        assert dropped == 20
        out = _fill(buf, 100)
        # 40 of the first chunk + all 60 of the second survive
        assert np.array_equal(out[:40], np.ones(40))
        assert np.array_equal(out[40:], np.full(60, 2.0))

    def test_push_after_partial_drain_counts_cur(self):
        # samples already moved into `cur` are not part of `pending`
        buf = _PlayBuf()
        buf.push(np.ones(10, np.float32), cap=1024)
        _fill(buf, 4)  # 10-sample chunk becomes cur, pending=0
        assert buf.pending == 0
        dropped = buf.push(np.ones(10, np.float32), cap=8)
        # pending was 0; new chunk trimmed to the newest 8 samples
        assert dropped == 2
        out = _fill(buf, 14)
        assert np.all(out == 1.0)
