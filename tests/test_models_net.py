"""Model file discovery (.aivmx / BOOTH zip) and network helpers."""

from __future__ import annotations

import socket
import zipfile
from typing import ClassVar

import httpx
import pytest

from kowairo import paths
from kowairo.engine.releases import asset_for
from kowairo.models.aivm import ModelImportError, find_aivmx
from kowairo.util.net import download, free_port


class TestFindAivmx:
    def test_plain_file(self, tmp_path):
        f = tmp_path / "voice.aivmx"
        f.write_bytes(b"onnx")
        assert find_aivmx(f) == [f]

    def test_directory(self, tmp_path):
        (tmp_path / "sub").mkdir()
        a = tmp_path / "sub" / "b.aivmx"
        a.write_bytes(b"1")
        b = tmp_path / "a.aivmx"
        b.write_bytes(b"2")
        got = find_aivmx(tmp_path)
        assert set(got) == {a, b}
        assert got == sorted(got)

    def test_zip_extracts_to_models_dir(self, tmp_path):
        z = tmp_path / "downer.zip"
        with zipfile.ZipFile(z, "w") as zipf:
            zipf.writestr("model/voice.aivmx", b"payload")
            zipf.writestr("readme.txt", b"hi")
        got = find_aivmx(z)
        assert len(got) == 1
        assert got[0].name == "voice.aivmx"
        assert got[0].read_bytes() == b"payload"
        assert paths.models_dir() in got[0].parents

    def test_zip_without_aivmx(self, tmp_path):
        z = tmp_path / "empty.zip"
        with zipfile.ZipFile(z, "w") as zipf:
            zipf.writestr("readme.txt", b"hi")
        with pytest.raises(ModelImportError):
            find_aivmx(z)

    def test_unsupported(self, tmp_path):
        f = tmp_path / "x.rar"
        f.write_bytes(b"x")
        with pytest.raises(ModelImportError):
            find_aivmx(f)


class TestEngineAsset:
    def test_windows(self):
        a = asset_for("1.2.0", "Windows-x64")
        assert a.archive_name == "AivisSpeech-Engine-Windows-x64-1.2.0.7z.001"
        assert a.executable == "run.exe"
        assert "1.2.0" in a.url

    def test_version_prefix_stripped(self):
        assert asset_for("v1.2.0", "Linux-x64").tag == "1.2.0"

    def test_non_windows_executable(self):
        assert asset_for("1.2.0", "macOS-arm64").executable == "run"


class TestNet:
    def test_free_port_is_bindable(self):
        p = free_port()
        with socket.socket() as s:
            s.bind(("127.0.0.1", p))

    def test_download_fresh(self, tmp_path, monkeypatch):
        body = b"x" * 3000
        seen = {}

        class Resp:
            status_code: ClassVar[int] = 200
            headers: ClassVar[dict] = {"content-length": str(len(body))}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def iter_bytes(self, n):
                for i in range(0, len(body), n):
                    yield body[i:i + n]

        def fake_stream(method, url, **kw):
            seen["headers"] = kw.get("headers", {})
            return Resp()

        monkeypatch.setattr(httpx, "stream", fake_stream)
        dest = tmp_path / "f.bin"
        got = download("https://example.com/f.bin", dest)
        assert got.read_bytes() == body
        assert "Range" not in seen["headers"]

    def test_download_resumes_part(self, tmp_path, monkeypatch):
        body = b"0123456789"
        dest = tmp_path / "f.bin"
        part = dest.with_suffix(".bin.part")
        part.write_bytes(body[:4])

        class Resp:
            def __init__(self, data, status):
                self._data = data
                self.status_code = status
                self.headers = {"content-length": str(len(data))}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def iter_bytes(self, n):
                yield self._data

        def fake_stream(method, url, **kw):
            rng = kw.get("headers", {}).get("Range", "")
            if rng == "bytes=4-":
                return Resp(body[4:], 206)
            return Resp(body, 200)

        monkeypatch.setattr(httpx, "stream", fake_stream)
        got = download("https://example.com/f.bin", dest)
        assert got.read_bytes() == body

    def test_download_restarts_when_range_ignored(self, tmp_path,
                                                  monkeypatch):
        body = b"0123456789"
        dest = tmp_path / "f.bin"
        dest.with_suffix(".bin.part").write_bytes(b"JUNK")

        class Resp:
            status_code: ClassVar[int] = 200
            headers: ClassVar[dict] = {"content-length": str(len(body))}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def iter_bytes(self, n):
                yield body

        monkeypatch.setattr(
            httpx, "stream", lambda m, u, **kw: Resp())
        got = download("https://example.com/f.bin", dest)
        assert got.read_bytes() == body
