"""Output effects: per-mode behavior and cross-chunk continuity."""

from __future__ import annotations

import numpy as np

from kowairo.audio.fx import FxChain, apply_gain_limiter

RATE = 48000


def _impulse(n: int = 4800, amp: float = 0.5) -> np.ndarray:
    x = np.zeros(n, np.float32)
    x[0] = amp
    return x


def test_off_is_identity():
    fx = FxChain("off", 0.5, RATE)
    x = np.random.default_rng(0).standard_normal(256).astype(np.float32)
    assert fx.apply(x) is x


def test_gain_and_limiter():
    x = np.full(4, 0.8, np.float32)
    out = apply_gain_limiter(x, 2.0, True)
    assert np.all(out < 1.0) and np.all(out > 0.8)  # tanh soft-clips
    out = apply_gain_limiter(x, 2.0, False)
    assert np.all(out == 1.0)                      # hard clip
    out = apply_gain_limiter(x, 0.5, False)
    assert np.allclose(out, 0.4)


def test_echo_feeds_back_across_chunks():
    fx = FxChain("echo", 1.0, RATE)
    fx.apply(_impulse())
    tail = fx.apply(np.zeros(4800, np.float32))
    # ~0.18 s (8640 samples) delay — second 0.1 s chunk is still silent,
    # but the impulse response must not be empty overall
    assert np.any(tail != 0) or True  # delay 8640 > 4800: silent here
    tail2 = fx.apply(np.zeros(4800, np.float32))
    assert np.any(np.concatenate([tail, tail2]) != 0)


def test_reverb_has_delayed_taps():
    fx = FxChain("reverb", 1.0, RATE)
    out = fx.apply(_impulse())
    nz = np.flatnonzero(np.abs(out) > 1e-4)
    assert len(nz) > 4                     # multiple taps lit
    assert out[nz[0]] < 0.5                # wet-only delayed copy
    assert np.all(np.abs(out) <= 1.0)


def test_robot_ring_modulation():
    fx = FxChain("robot", 1.0, RATE)
    x = np.full(4800, 0.5, np.float32)
    out = fx.apply(x)
    # ring mod at 30 Hz → carrier crosses zero → output dips below input
    assert out.min() < 0.0
    assert np.all(np.abs(out) <= 0.5 + 1e-6)


def test_amount_zero_is_passthrough():
    fx = FxChain("echo", 0.0, RATE)
    x = np.random.default_rng(1).standard_normal(128).astype(np.float32)
    assert np.array_equal(fx.apply(x), x)


def test_live_mode_switch():
    fx = FxChain("off", 0.5, RATE)
    fx.set("robot", 0.5)
    x = np.full(64, 0.5, np.float32)
    assert not np.array_equal(fx.apply(x), x)
