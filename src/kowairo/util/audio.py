"""PCM helpers: decoding, resampling, metering."""

from __future__ import annotations

import io
import wave

import numpy as np
import soxr

TARGET_ASR_RATE = 16000


def wav_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV byte-string to (float32 mono PCM, sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, rate


def pcm_to_wav(pcm: np.ndarray, rate: int) -> bytes:
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int, quality: str = "HQ") -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.float32, copy=False)
    res = soxr.resample(pcm.astype(np.float64), src_rate, dst_rate,
                        quality=quality)
    return res.astype(np.float32)


def rms_db(pcm: np.ndarray, floor: float = -80.0) -> float:
    if pcm.size == 0:
        return floor
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    return max(floor, 20.0 * np.log10(rms + 1e-12))
