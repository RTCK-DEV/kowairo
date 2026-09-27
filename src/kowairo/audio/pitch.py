"""Lightweight autocorrelation f0 estimate for prosody transfer."""

from __future__ import annotations

import math

import numpy as np


def median_f0(pcm: np.ndarray, sample_rate: int,
              fmin: float = 60.0, fmax: float = 400.0) -> float | None:
    """Median voiced f0 (Hz) via windowed autocorrelation. None if unvoiced."""
    if pcm.size < sample_rate // 4:
        return None
    frame = int(0.032 * sample_rate)
    hop = int(0.016 * sample_rate)
    lo = int(sample_rate / fmax)
    hi = int(sample_rate / fmin)
    f0s: list[float] = []
    for start in range(0, pcm.size - frame, hop):
        x = pcm[start:start + frame].astype(np.float64)
        x -= x.mean()
        energy = float(x @ x)
        if energy < 1e-6:
            continue
        ac = np.correlate(x, x, mode="full")[frame - 1 + lo: frame - 1 + hi]
        if ac.size == 0:
            continue
        lag = lo + int(np.argmax(ac))
        # normalized autocorrelation at the best lag (ac[0] == energy)
        norm = ac[lag - lo] / energy if energy > 0 else 0.0
        if norm < 0.3:  # weak periodicity → unvoiced
            continue
        # parabolic refinement
        i = lag - lo
        if 0 < i < ac.size - 1:
            a, b, c = ac[i - 1], ac[i], ac[i + 1]
            denom = (a - 2 * b + c)
            if abs(denom) > 1e-12:
                lag = lag + 0.5 * (a - c) / denom
        if lag > 0:
            f0s.append(sample_rate / lag)
    if len(f0s) < 3:
        return None
    return float(np.median(f0s))


def semitone_shift(src_f0: float | None, target_f0: float) -> float:
    """Semitone difference from src to target, clamped to a sane range."""
    if src_f0 is None or src_f0 <= 0:
        return 0.0
    st = 12.0 * math.log2(target_f0 / src_f0)
    return float(max(-12.0, min(12.0, st)))
