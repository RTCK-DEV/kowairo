"""Lightweight output effects (VOIDOL "SYNTH mode"-style) on float32 PCM.

Vectorized NumPy implementations — cheap enough for the playback worker.
All state carries across chunks so effects sound continuous.
"""

from __future__ import annotations

import numpy as np

MODES = ("off", "echo", "reverb", "robot")


class FxChain:
    """Apply the configured effect to outgoing PCM chunks."""

    def __init__(self, mode: str, amount: float, rate: int) -> None:
        self.mode = mode
        self.amount = float(np.clip(amount, 0.0, 1.0))
        self.rate = rate
        self._d_echo = max(1, int(0.18 * rate))
        self._hist = np.zeros(self._d_echo, np.float32)  # echo feedback history
        # reverb-lite: feedforward multi-tap early reflections
        tap_ms = (29.7, 37.1, 41.1, 43.7, 51.3, 60.5, 70.1, 81.9)
        self._taps = np.array([int(m * rate / 1000) for m in tap_ms])
        self._tap_g = np.array([0.40, 0.34, 0.30, 0.26, 0.21, 0.17, 0.13, 0.10],
                               np.float32)
        self._r_delay = int(self._taps.max())
        self._r_hist = np.zeros(self._r_delay, np.float32)
        self._t = 0.0  # carrier phase (seconds) for robot

    def set(self, mode: str, amount: float) -> None:
        self.mode = mode
        self.amount = float(np.clip(amount, 0.0, 1.0))

    def apply(self, pcm: np.ndarray) -> np.ndarray:
        if self.mode == "off" or self.amount <= 0.0 or pcm.size == 0:
            return pcm
        if self.mode == "echo":
            return self._echo(pcm)
        if self.mode == "reverb":
            return self._reverb(pcm)
        if self.mode == "robot":
            return self._robot(pcm)
        return pcm

    # ------------------------------------------------------------ effects
    def _echo(self, pcm: np.ndarray) -> np.ndarray:
        n = pcm.size
        D = self._d_echo
        wet = 0.15 + 0.5 * self.amount
        fb = 0.30 + 0.40 * self.amount
        out = np.empty(n, np.float32)
        i = 0
        # f[n] = x[n] + fb * f[n-D]; out[n] = x[n] + wet * f[n-D]
        # _hist holds the last D samples of f ending at i → f[i-D:i-D+m]
        # is its first m entries.
        while i < n:
            m = min(D, n - i)
            delayed = self._hist[:m]
            f = pcm[i:i + m] + delayed * fb
            out[i:i + m] = pcm[i:i + m] + delayed * wet
            self._hist = np.concatenate([self._hist[m:], f])
            i += m
        return np.clip(out, -1.0, 1.0)

    def _reverb(self, pcm: np.ndarray) -> np.ndarray:
        n = pcm.size
        wet = 0.20 + 0.55 * self.amount
        full = np.concatenate([self._r_hist, pcm])
        acc = np.zeros(n, np.float32)
        D = self._r_delay
        for d, g in zip(self._taps, self._tap_g, strict=False):
            acc += full[D - d: D - d + n] * g
        self._r_hist = full[-D:]  # full.size = D + n >= D always
        out = pcm * (1.0 - wet) + acc * wet
        return np.clip(out, -1.0, 1.0)

    def _robot(self, pcm: np.ndarray) -> np.ndarray:
        n = pcm.size
        t = self._t + np.arange(n, dtype=np.float64) / self.rate
        carrier = np.sin(2 * np.pi * 30.0 * t).astype(np.float32)
        self._t = float(t[-1] + 1.0 / self.rate)
        wet = self.amount
        return pcm * ((1.0 - wet) + carrier * wet)


def apply_gain_limiter(pcm: np.ndarray, gain: float,
                       limiter: bool) -> np.ndarray:
    """Output gain then soft clip (tanh) or hard clip."""
    if gain != 1.0:
        pcm = (pcm * np.float32(gain)).astype(np.float32)
    if limiter:
        return np.tanh(pcm).astype(np.float32)
    return np.clip(pcm, -1.0, 1.0)
