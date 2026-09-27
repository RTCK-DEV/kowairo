"""ASR backend interface — all backends take 16 kHz mono float32 PCM."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ASRBackend(Protocol):
    name: str

    def is_ready(self) -> bool: ...

    def ensure_model(self, progress=None) -> None: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str: ...

    def close(self) -> None: ...
