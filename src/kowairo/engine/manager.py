"""Download, unpack and supervise the embedded AivisSpeech Engine process."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import py7zr

from .. import paths
from ..util.net import ProgressCb, download, free_port, wait_port
from .client import EngineClient
from .releases import EngineAsset, asset_for

LogCb = Callable[[str], None]


class EngineError(RuntimeError):
    pass


class EngineManager:
    """Owns the engine subprocess lifecycle on localhost."""

    def __init__(self, version: str = "1.2.0", use_gpu: bool = False,
                 log: LogCb | None = None) -> None:
        self.version = version
        self.use_gpu = use_gpu
        self._log = log or (lambda _m: None)
        self._proc: subprocess.Popen[bytes] | None = None
        self._port: int | None = None
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ setup
    def asset(self) -> EngineAsset:
        return asset_for(self.version)

    def install_dir(self) -> Path:
        a = self.asset()
        return paths.engine_dir() / f"{a.tag}-{a.archive_name.removesuffix('.7z.001')}"

    def executable(self) -> Path:
        return self.install_dir() / self.asset().executable

    def is_installed(self) -> bool:
        return self.executable().exists()

    def ensure_installed(self, progress: ProgressCb | None = None,
                         stage: Callable[[str], None] | None = None) -> Path:
        """Download + verify + extract the engine archive. Returns exe path."""
        exe = self.executable()
        if exe.exists():
            return exe
        a = self.asset()
        dl = paths.downloads_dir()
        archive = dl / a.archive_name

        (stage or self._log)(f"エンジンをダウンロード中: {a.archive_name}")
        if not archive.exists():
            download(a.url, archive, progress)
        self._verify_digest(a, archive)

        (stage or self._log)("エンジンを展開中…")
        out_dir = self.install_dir()
        tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(tmp_dir)
        # archives contain a single top-level dir; normalize
        entries = list(tmp_dir.iterdir())
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)
        if len(entries) == 1 and entries[0].is_dir():
            entries[0].replace(out_dir)
            import shutil

            shutil.rmtree(tmp_dir)
        else:
            tmp_dir.replace(out_dir)
        if not exe.exists():
            raise EngineError(f"展開後に実行ファイルが見つかりません: {exe}")
        if not sys.platform.startswith("win"):
            exe.chmod(0o755)
        return exe

    def _verify_digest(self, a: EngineAsset, archive: Path) -> None:
        """Verify the archive against the sha256 digest in the release API."""
        import json

        import httpx

        from .releases import GITHUB_REPO

        try:
            r = httpx.get(
                f"https://api.github.com/repos/{GITHUB_REPO}"
                f"/releases/tags/{a.tag}",
                timeout=httpx.Timeout(15.0, connect=10.0))
            r.raise_for_status()
            assets = {x["name"]: x for x in r.json().get("assets", [])}
            digest = (assets.get(a.archive_name) or {}).get("digest", "")
            if not digest.startswith("sha256:"):
                self._log("チェックサム情報なし — 検証をスキップ")
                return
            got = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
            if got.lower() != digest.lower():
                archive.unlink(missing_ok=True)
                raise EngineError(
                    f"エンジンアーカイブのチェックサム不一致: {got} != {digest}")
            self._log("エンジンアーカイブの整合性を確認しました")
        except EngineError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            self._log(f"チェックサム検証をスキップ ({e})")

    # ---------------------------------------------------------------- runtime
    @property
    def port(self) -> int | None:
        return self._port

    def start(self, timeout: float = 90.0) -> EngineClient:
        """Launch the engine and wait for the HTTP server to accept connections."""
        with self._lock:
            if self.is_running():
                assert self._port is not None
                return EngineClient("127.0.0.1", self._port)
            exe = self.executable()
            if not exe.exists():
                raise EngineError("エンジンがインストールされていません")
            port = free_port()
            cmd = [
                str(exe),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--output_log_utf8",
                "--disable_sentry",
            ]
            if self.use_gpu:
                cmd.append("--use_gpu")
            env = os.environ.copy()
            env.setdefault("VV_OUTPUT_LOG_UTF8", "1")
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._log(f"エンジン起動: {' '.join(cmd)}")
            # engine output must be drained or the child blocks on a full pipe
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(exe.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._port = port
            self._reader_thread = threading.Thread(
                target=self._pump_log, args=(self._proc,), daemon=True)
            self._reader_thread.start()

        if not wait_port("127.0.0.1", port, timeout):
            rc = self._proc.poll() if self._proc else None
            self.stop()
            raise EngineError(f"エンジンの起動がタイムアウトしました (exit={rc})")
        client = EngineClient("127.0.0.1", port)
        try:
            version = client.version()
            self._log(f"エンジン起動完了: v{version} port={port}")
        except Exception as e:
            self._log(f"エンジン応答待機中の警告: {e}")
        return client

    def _pump_log(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                line = repr(raw)
            self._log(f"[engine] {line}")

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            self._port = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
