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


def _reference_median_f0(pcm, rate, fmin=60.0, fmax=400.0):
    """Pre-FFT implementation (direct correlate per frame) kept here as the
    golden reference — the FFT path must match it numerically."""
    if pcm.size < rate // 4:
        return None
    frame, hop = int(0.032 * rate), int(0.016 * rate)
    lo, hi = int(rate / fmax), int(rate / fmin)
    f0s = []
    for off in range(0, pcm.size - frame, hop):
        x = pcm[off:off + frame].astype(np.float64)
        x -= x.mean()
        energy = float(np.dot(x, x))
        if energy < 1e-6:
            continue
        ac = np.correlate(x, x, "full")[frame - 1 + lo:frame - 1 + hi]
        i = int(np.argmax(ac))
        norm = ac[i] / energy
        if norm < 0.3:
            continue
        lag = lo + i
        if 0 < i < ac.size - 1:
            a, b, c = ac[i - 1], ac[i], ac[i + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                lag = lag + 0.5 * (a - c) / denom
        if lag > 0:
            f0s.append(rate / lag)
    if len(f0s) < 3:
        return None
    return float(np.median(f0s))


def test_median_f0_fft_matches_direct_correlate():
    rate = 16000
    rng = np.random.default_rng(1)
    t = np.arange(rate * 2) / rate
    # harmonics + drift + noise — a realistic voiced signal
    sig = (np.sin(2 * np.pi * 180.0 * t)
           + 0.5 * np.sin(2 * np.pi * 360.0 * t)
           + 0.2 * np.sin(2 * np.pi * 540.0 * t)
           + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    ref = _reference_median_f0(sig, rate)
    got = median_f0(sig, rate)
    assert ref is not None and got is not None
    assert abs(got - ref) < 2.0


def test_median_f0_forced_gpu_falls_back_cleanly():
    """use_gpu=True must not raise even when CuPy/CUDA is absent."""
    rate = 16000
    t = np.arange(rate) / rate
    sig = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
    f0 = median_f0(sig, rate, use_gpu=True)
    assert f0 is None or abs(f0 - 200.0) < 8.0


def test_semitone_shift():
    assert semitone_shift(None, 220.0) == 0.0
    assert semitone_shift(0.0, 220.0) == 0.0
    assert abs(semitone_shift(110.0, 220.0) - 12.0) < 0.01  # clamped at ±12
    assert abs(semitone_shift(220.0, 220.0) - 0.0) < 0.01
    # +12 semitones clamps
    assert semitone_shift(55.0, 880.0) == 12.0
