"""QThread workers that wrap blocking pipeline/engine operations."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..asr.sherpa import SherpaReazonASR, ensure_silero_vad
from ..engine.client import EngineClient
from ..engine.manager import EngineManager
from ..models import hub
from ..models.aivm import find_aivmx


class EngineSetupWorker(QThread):
    progress = Signal(str, int, int)   # stage label, received, total
    done = Signal(object)              # EngineClient
    failed = Signal(str)

    def __init__(self, manager: EngineManager) -> None:
        super().__init__()
        self.manager = manager

    def run(self) -> None:
        try:
            self.manager.ensure_installed(
                progress=lambda r, t: self.progress.emit(
                    "エンジンをダウンロード中", r, t),
                stage=lambda s: self.progress.emit(s, 0, 0))
            client = self.manager.start()
            self.done.emit(client)
        except Exception as e:
            self.failed.emit(str(e))


class ModelInstallWorker(QThread):
    progress = Signal(str, int, int)
    done = Signal(str)       # installed aivm uuid or name
    failed = Signal(str)

    def __init__(self, client: EngineClient, source: Path | None = None,
                 hub_uuid: str | None = None) -> None:
        super().__init__()
        self.client = client
        self.source = source
        self.hub_uuid = hub_uuid

    def run(self) -> None:
        try:
            if self.hub_uuid:
                path = hub.download_model(
                    self.hub_uuid,
                    progress=lambda r, t: self.progress.emit(
                        "モデルをダウンロード中", r, t))
                self.progress.emit("モデルをインストール中", 0, 0)
                self.client.install_model_file(path)
                self.done.emit(path.stem)
            elif self.source is not None:
                files = find_aivmx(self.source)
                for f in files:
                    self.progress.emit(f"{f.name} をインストール中", 0, 0)
                    self.client.install_model_file(f)
                self.done.emit(files[0].stem)
            else:
                self.failed.emit("モデルが指定されていません")
        except Exception as e:
            self.failed.emit(str(e))


class HubSearchWorker(QThread):
    done = Signal(int, list)     # total count, [HubSearchResult]
    failed = Signal(str)

    def __init__(self, keyword: str = "", sort: str = "download",
                 page: int = 1) -> None:
        super().__init__()
        self.keyword = keyword
        self.sort = sort
        self.page = page

    def run(self) -> None:
        try:
            total, entries = hub.search_models(
                keyword=self.keyword, sort=self.sort, page=self.page)
            self.done.emit(total, entries)
        except Exception as e:
            self.failed.emit(str(e))


class AsrSetupWorker(QThread):
    progress = Signal(str, int, int)
    done = Signal(object)    # SherpaReazonASR
    failed = Signal(str)

    def __init__(self, asr: SherpaReazonASR) -> None:
        super().__init__()
        self.asr = asr

    def run(self) -> None:
        try:
            self.asr.ensure_model(
                progress=lambda r, t: self.progress.emit(
                    "音声認識モデルをダウンロード中", r, t))
            ensure_silero_vad(
                progress=lambda r, t: self.progress.emit(
                    "VADモデルをダウンロード中", r, t))
            self.progress.emit("音声認識を初期化中", 0, 0)
            self.asr.load()
            self.done.emit(self.asr)
        except Exception as e:
            self.failed.emit(str(e))
