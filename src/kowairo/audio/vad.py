"""Silero VAD based endpointing over a 16 kHz mono PCM stream.

Drives sherpa-onnx's ``VadModel`` frame-by-frame (32 ms windows) and performs
hangover-based segmentation: an utterance ends after ``min_silence_ms`` of
continuous non-speech, or is force-cut at ``max_speech_sec`` so ASR can start
early on long monologues.

While speech is ongoing, ``on_speech`` is invoked every ``interim_every_ms``
(starting at ``interim_after_ms``) with a snapshot of the accumulated segment
so callers can run interim ASR and start synthesis before the endpoint — the
counterpart of chunk-streamed output in direct-conversion voice changers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import sherpa_onnx

WINDOW = 512  # silero v5 window @16kHz (32 ms)


class VadSegmenter:
    def __init__(self, model_path: Path, threshold: float = 0.5,
                 min_silence_ms: int = 450, min_speech_ms: int = 160,
                 max_speech_sec: float = 15.0, sample_rate: int = 16000,
                 on_speech: Callable[[np.ndarray], None] | None = None,
                 interim_after_ms: int = 1600,
                 interim_every_ms: int = 1000) -> None:
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model_path)
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence_ms / 1000.0
        cfg.silero_vad.min_speech_duration = min_speech_ms / 1000.0
        cfg.silero_vad.max_speech_duration = max_speech_sec
        cfg.silero_vad.window_size = WINDOW
        cfg.sample_rate = sample_rate
        if not cfg.validate():
            raise ValueError("invalid VadModelConfig")
        self.vad = sherpa_onnx.VadModel.create(cfg)
        if self.vad is None:
            raise ValueError("failed to create VAD model")
        self.sample_rate = sample_rate
        self.win = int(self.vad.window_size())
        # endpointing state (in samples)
        self._min_sil = int(min_silence_ms / 1000.0 * sample_rate)
        self._min_speech = int(min_speech_ms / 1000.0 * sample_rate)
        self._max_speech = int(max_speech_sec * sample_rate)
        self._in_speech = False
        self._seg_start = 0
        self._sil_start: int | None = None
        self._pos = 0           # absolute stream position (samples)
        self._buf = np.zeros(0, dtype=np.float32)
        self._pending: list[np.ndarray] = []
        self._on_speech = on_speech
        self._interim_after = int(interim_after_ms / 1000.0 * sample_rate)
        self._interim_every = int(interim_every_ms / 1000.0 * sample_rate)
        self._next_interim = 0  # abs sample pos of the next interim snapshot

    def feed(self, pcm: np.ndarray) -> Iterator[np.ndarray]:
        """Accept arbitrary-length PCM; yield completed utterance segments."""
        self._buf = (np.concatenate([self._buf, pcm])
                     if self._buf.size else pcm.astype(np.float32, copy=False))
        n_full = self._buf.size // self.win
        for i in range(n_full):
            frame = self._buf[i * self.win:(i + 1) * self.win]
            is_speech = bool(self.vad.is_speech(frame.tolist()))
            frame_end = self._pos + self.win
            if is_speech:
                if not self._in_speech:
                    self._in_speech = True
                    self._seg_start = self._pos
                    self._pending.clear()
                    self._next_interim = self._pos + self._interim_after
                self._sil_start = None
            else:
                if self._in_speech:
                    if self._sil_start is None:
                        self._sil_start = self._pos
                    if frame_end - self._sil_start >= self._min_sil:
                        seg = self._emit(cut_at=self._sil_start)
                        if seg.size:
                            yield seg
            if self._in_speech and frame_end - self._seg_start >= \
                    self._max_speech:
                seg = self._emit(cut_at=frame_end)
                if seg.size:
                    yield seg
            if self._in_speech:
                self._pending.append(frame)
                if (self._on_speech is not None
                        and frame_end >= self._next_interim):
                    self._next_interim = frame_end + self._interim_every
                    self._on_speech(np.concatenate(self._pending))
            self._pos = frame_end
        self._buf = self._buf[n_full * self.win:]

    def _emit(self, cut_at: int) -> np.ndarray:
        """Emit pending speech up to `cut_at` and reset state."""
        seg = np.concatenate(self._pending) if self._pending else \
            np.zeros(0, np.float32)
        trim = self._pos + self.win - cut_at  # tail silence beyond cut point
        if trim > 0 and trim < seg.size:
            seg = seg[: seg.size - trim]
        self._pending.clear()
        self._in_speech = False
        self._sil_start = None
        self._seg_start = self._pos
        return seg if seg.size >= self._min_speech else np.zeros(0, np.float32)

    def flush(self) -> Iterator[np.ndarray]:
        """Emit any in-flight speech (on stop / end of file)."""
        if self._in_speech and self._pending:
            seg = np.concatenate(self._pending)
            if seg.size >= self._min_speech:
                yield seg
        self._pending.clear()
        self._in_speech = False
        self.vad.reset()
