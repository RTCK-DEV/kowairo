"""Lightweight autocorrelation f0 estimate for prosody transfer.

Autocorrelation is computed per 32 ms frame. The hot path is vectorized
FFT (|rfft|^2 -> irfft), which is mathematically identical to the direct
O(frame^2) correlate but ~50x faster at 16 kHz. If CuPy with a working
CUDA device is installed, the frame batch can run on the GPU instead —
worthwhile once utterances grow past a few seconds, since interim
snapshots re-estimate f0 over the whole accumulated buffer.
"""

from __future__ import annotations

import math
import os

import numpy as np

_cp = None


def _cupy():
    """Lazily import cupy; returns the module or False when unavailable."""
    global _cp
    if _cp is None:
        if os.environ.get("KOWAIRO_NO_GPU"):
            _cp = False
        else:
            try:
                import cupy
                from cupy.cuda import cufft  # noqa: F401  (probe DLL)

                _cp = cupy if cupy.cuda.runtime.getDeviceCount() > 0 \
                    else False
            except Exception:
                _cp = False
    return _cp


def _autocorr(frames: np.ndarray, xp=np) -> np.ndarray:
    """Per-frame autocorrelation c[lag] = sum_i x[i]*x[i+lag] (biased,
    like np.correlate 'full' — overlap shrinks with lag), via FFT."""
    frame = frames.shape[1]
    n_fft = 1 << (2 * frame - 1).bit_length()  # linear (not circular) corr
    f = xp.asarray(frames)
    spec = xp.fft.rfft(f, n=n_fft, axis=1)
    ac = xp.fft.irfft(spec * spec.conj(), n=n_fft, axis=1)
    if xp is not np:
        ac = xp.asnumpy(ac)
    return np.asarray(ac, dtype=np.float64)


def median_f0(pcm: np.ndarray, sample_rate: int,
              fmin: float = 60.0, fmax: float = 400.0,
              use_gpu: bool | None = None) -> float | None:
    """Median voiced f0 (Hz) via windowed autocorrelation. None if unvoiced.

    use_gpu=None auto-detects CuPy; pass True/False to force.
    """
    if pcm.size < sample_rate // 4:
        return None
    frame = int(0.032 * sample_rate)
    hop = int(0.016 * sample_rate)
    lo = int(sample_rate / fmax)
    hi = int(sample_rate / fmin)
    n_frames = 1 + (pcm.size - frame) // hop
    if n_frames <= 0:
        return None
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    x = pcm[idx].astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    energy = np.einsum("ij,ij->i", x, x)

    xp = np
    if use_gpu or (use_gpu is None and _cupy()):
        got = _cupy()
        if got:
            xp = got
    ac_all = _autocorr(x, xp)[:, lo:hi]

    # normalized autocorrelation at each frame's best lag (ac[0] == energy)
    best = np.argmax(ac_all, axis=1)
    norm = ac_all[np.arange(n_frames), best] / np.maximum(energy, 1e-300)
    voiced = (energy >= 1e-6) & (norm >= 0.3)
    lags = best + lo

    f0s: list[float] = []
    for i in np.flatnonzero(voiced):
        lag: float = float(lags[i])
        # parabolic refinement (skipped at the slice edges, like before)
        bi = int(best[i])
        if 0 < bi < ac_all.shape[1] - 1:
            a, b, c = ac_all[i, bi - 1], ac_all[i, bi], ac_all[i, bi + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                lag += 0.5 * (a - c) / denom
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
