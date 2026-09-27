"""ASR backend factory — settings-driven selection with lazy imports."""

from __future__ import annotations

from kowairo.asr import create
from kowairo.asr.sherpa import SherpaReazonASR
from kowairo.asr.whisper import FasterWhisperASR
from kowairo.settings import Settings


def test_default_backend_is_sherpa():
    asr = create(Settings())
    assert isinstance(asr, SherpaReazonASR)
    assert asr.name == "sherpa-reazonspeech"


def test_whisper_backend_selected():
    s = Settings(asr_backend="faster-whisper", whisper_model="small")
    asr = create(s)
    assert isinstance(asr, FasterWhisperASR)
    assert asr.name == "faster-whisper"
    assert asr.model_size == "small"


def test_whisper_backend_lazy_imports():
    """Constructing/selecting the backend must not require faster-whisper
    to be installed — imports happen in ensure_model/load."""
    asr = create(Settings(asr_backend="faster-whisper"))
    # is_ready may probe the HF cache but must not raise without the package
    assert asr.is_ready() in (True, False)


def test_unknown_backend_falls_back_to_sherpa():
    s = Settings(asr_backend="nonexistent-backend")
    assert isinstance(create(s), SherpaReazonASR)
