"""Autocorrelation f0 estimate and semitone mapping."""

from __future__ import annotations

import numpy as np

from kowairo.audio.pitch import median_f0, semitone_shift


def test_median_f0_sine():
    rate = 16000
    t = np.arange(rate) / rate
    sig = np.sin(2 * np.pi * 150.0 * t).astype(np.float32)
    f0 = median_f0(sig, rate)
    assert f0 is not None
    assert abs(f0 - 150.0) < 5.0


def test_median_f0_silence_and_noise():
    rate = 16000
    assert median_f0(np.zeros(rate, np.float32), rate) is None
    rng = np.random.default_rng(0)
    # white noise has no stable periodicity
    assert median_f0(rng.standard_normal(rate).astype(np.float32),
                     rate) is None


def test_median_f0_short_input():
    assert median_f0(np.zeros(100, np.float32), 16000) is None


def test_semitone_shift():
    assert semitone_shift(None, 220.0) == 0.0
    assert semitone_shift(0.0, 220.0) == 0.0
    assert abs(semitone_shift(110.0, 220.0) - 12.0) < 0.01  # clamped at ±12
    assert abs(semitone_shift(220.0, 220.0) - 0.0) < 0.01
    # +12 semitones clamps
    assert semitone_shift(55.0, 880.0) == 12.0
