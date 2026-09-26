"""sherpa-onnx ASR backend — Japanese Zipformer transducer (ReazonSpeech).

Runs fully on CPU with int8-quantized encoder (~148 MB) by default; an fp32
variant is kept for users who prefer maximum accuracy. sherpa-onnx pip wheels
are CPU-only, keeping VRAM usage at zero — GPU acceleration is optional via
the faster-whisper backend instead.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import sherpa_onnx

from .. import paths
from ..util.net import ProgressCb, download

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"
)
MODEL_GLOB = "sherpa-onnx-zipformer-ja-reazonspeech-*"

SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)


class SherpaReazonASR:
    name = "sherpa-reazonspeech"

    def __init__(self, num_threads: int = 2, use_int8: bool = True,
                 provider: str = "cpu") -> None:
        self.num_threads = num_threads
        self.use_int8 = use_int8
        self.provider = provider
        self._recognizer: sherpa_onnx.OfflineRecognizer | None = None

    # --------------------------------------------------------------- model
    def model_dir(self) -> Path:
        d = paths.asr_models_dir()
        for p in sorted(d.glob(MODEL_GLOB)):
            if p.is_dir():
                return p
        return d / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"

    def is_ready(self) -> bool:
        d = self.model_dir()
        return (d / "tokens.txt").exists() and any(d.glob("encoder*.onnx"))

    def ensure_model(self, progress: ProgressCb | None = None) -> None:
        if self.is_ready():
            return
        archive = paths.downloads_dir() / Path(MODEL_URL).name
        if not archive.exists():
            download(MODEL_URL, archive, progress)
        out = paths.asr_models_dir()
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(out, filter="data")

    def _pick(self, stem: str) -> Path:
        d = self.model_dir()
        if self.use_int8:
            cand = d / f"{stem}-epoch-99-avg-1.int8.onnx"
            if cand.exists():
                return cand
        return d / f"{stem}-epoch-99-avg-1.onnx"

    def load(self) -> None:
        if self._recognizer is not None:
            return
        d = self.model_dir()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self._pick("encoder")),
            decoder=str(self._pick("decoder")),
            joiner=str(self._pick("joiner")),
            tokens=str(d / "tokens.txt"),
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=self.provider,
            debug=False,
        )

    # ------------------------------------------------------------ inference
    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        if self._recognizer is None:
            self.load()
        assert self._recognizer is not None
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, pcm)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text).strip()

    def close(self) -> None:
        self._recognizer = None


def ensure_silero_vad(progress: ProgressCb | None = None) -> Path:
    dest = paths.asr_models_dir() / "silero_vad.onnx"
    if not dest.exists():
        download(SILERO_VAD_URL, dest, progress)
    return dest
