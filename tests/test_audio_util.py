"""PCM helpers: WAV roundtrip, metering, resampling, file decode."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from kowairo.util.audio import (
    load_audio_file,
    pcm_to_wav,
    resample,
    rms_db,
    wav_to_pcm,
)


def test_wav_roundtrip_int16():
    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(16000) * 0.2).astype(np.float32)
    data = pcm_to_wav(pcm, 16000)
    back, rate = wav_to_pcm(data)
    assert rate == 16000
    assert back.size == pcm.size
    # int16 quantization + truncation: worst case ~2/32768
    assert np.max(np.abs(back - pcm)) < 2.0 / 32768.0 + 1e-6


def test_wav_to_pcm_stereo_to_mono():
    stereo = np.zeros((100, 2), np.int16)
    stereo[:, 0] = 10000
    stereo[:, 1] = -10000
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(stereo.tobytes())
    pcm, rate = wav_to_pcm(buf.getvalue())
    assert rate == 44100
    assert np.all(np.abs(pcm) < 1e-6)  # L+R cancel → silence


def test_rms_db():
    assert rms_db(np.zeros(16, np.float32)) == -80.0
    assert abs(rms_db(np.ones(1024, np.float32)) - 0.0) < 0.01
    t = np.arange(16000) / 16000
    sine = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    assert abs(rms_db(sine) - (-3.01)) < 0.05


def test_resample_ratio_and_passthrough():
    x = np.zeros(16000, np.float32)
    up = resample(x, 16000, 48000)
    assert up.size == 48000
    same = resample(x, 16000, 16000)
    assert same.size == 16000 and same.dtype == np.float32


def test_load_audio_file(tmp_path):
    x = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100).astype(np.float32)
    p = tmp_path / "tone.flac"
    sf.write(str(p), x, 44100)
    pcm, rate = load_audio_file(p)
    assert rate == 44100
    assert pcm.size == 4410
    assert np.max(np.abs(pcm - x)) < 1e-4
