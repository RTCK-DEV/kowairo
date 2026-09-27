"""ASR backend factory — `settings.asr_backend` selects the engine."""

from __future__ import annotations

from ..settings import Settings
from .base import ASRBackend


def create(settings: Settings) -> ASRBackend:
    """Return the configured ASR backend (default: sherpa-reazonspeech)."""
    if settings.asr_backend == "faster-whisper":
        from .whisper import FasterWhisperASR

        return FasterWhisperASR(model_size=settings.whisper_model,
                                num_threads=settings.asr_num_threads)
    from .sherpa import SherpaReazonASR

    return SherpaReazonASR(num_threads=settings.asr_num_threads)
