"""faster-whisper ASR backend — CUDA-capable multilingual Whisper.

Optional extra (``pip install kowairo[whisper]``). Uses CTranslate2 under
the hood: CUDA on Windows/Linux when a GPU is present, int8 CPU otherwise.
On Windows the cuBLAS DLLs shipped by ``nvidia-cublas-cu12`` are registered
via ``os.add_dll_directory`` before ctranslate2 loads.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from .. import paths
from ..util.net import ProgressCb

DEFAULT_MODEL = "turbo"  # large-v3-turbo: ~4x faster than large-v3, ~1.6 GB


def _register_cuda_dll_dirs() -> None:
    """Make pip-installed CUDA libs (nvidia-*/bin) visible to ctranslate2."""
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    root = Path(spec.submodule_search_locations[0])
    for d in root.glob("*/bin"):
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


class FasterWhisperASR:
    """faster-whisper backend — GPU (CUDA) or int8 CPU inference."""

    name = "faster-whisper"

    def __init__(self, model_size: str = DEFAULT_MODEL,
                 num_threads: int = 2, device: str = "auto",
                 compute_type: str | None = None) -> None:
        self.model_size = model_size
        self.num_threads = num_threads
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._model_path: Path | None = None

    # --------------------------------------------------------------- model
    def _cache_root(self) -> Path:
        return paths.asr_models_dir() / "hf"

    def is_ready(self) -> bool:
        try:
            from faster_whisper.utils import download_model

            p = download_model(self.model_size, local_files_only=True,
                               cache_dir=str(self._cache_root()))
            return Path(p).exists()
        except Exception:
            return False

    def ensure_model(self, progress: ProgressCb | None = None) -> None:
        _register_cuda_dll_dirs()
        from faster_whisper.utils import download_model

        if progress:
            progress(0, 0)  # HF snapshot download has no byte-level cb
        self._model_path = Path(
            download_model(self.model_size,
                           cache_dir=str(self._cache_root())))

    def load(self) -> None:
        if self._model is not None:
            return
        _register_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        device = self.device
        if device == "auto":
            device = "cuda" if _cuda_device_count() > 0 else "cpu"
        ct = self.compute_type
        if ct is None:
            ct = "int8_float16" if device == "cuda" else "int8"
        src = str(self._model_path) if self._model_path else self.model_size
        self._model = WhisperModel(src, device=device, compute_type=ct,
                                   cpu_threads=self.num_threads)

    def warmup(self) -> None:
        """One throwaway decode so first real use isn't slowed by cuDNN/CT2
        autotune and arena init."""
        self.transcribe(np.zeros(16000 // 2, np.float32), 16000)

    # ------------------------------------------------------------ inference
    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            self.load()
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if sample_rate != 16000:
            from ..util.audio import resample

            pcm = resample(pcm, sample_rate, 16000)
        segments, _info = self._model.transcribe(
            pcm, language="ja", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True)
        return "".join(s.text for s in segments).strip()

    def close(self) -> None:
        self._model = None
