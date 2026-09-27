"""AivisHub API client — fetch public .aivmx models without a browser login."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .. import paths
from ..util.net import ProgressCb, download

API_BASE = "https://api.aivis-project.com/v1"


@dataclass
class HubModelFile:
    name: str
    version: str
    file_size: int
    checksum: str
    model_type: str
    license_type: str
    license_text: str
    speakers: list[dict[str, Any]]


@dataclass
class HubModel:
    uuid: str
    name: str
    description: str
    files: list[HubModelFile]
    raw: dict[str, Any]


def fetch_model(uuid: str) -> HubModel:
    r = httpx.get(f"{API_BASE}/aivm-models/{uuid}",
                  timeout=httpx.Timeout(30.0, connect=10.0))
    r.raise_for_status()
    d = r.json()
    files = [
        HubModelFile(
            name=f.get("name", ""),
            version=f.get("version", ""),
            file_size=int(f.get("file_size", 0)),
            checksum=f.get("checksum", ""),
            model_type=f.get("model_type", ""),
            license_type=f.get("license_type", ""),
            license_text=f.get("license_text", ""),
            speakers=f.get("speakers", []),
        )
        for f in d.get("model_files", [])
    ]
    return HubModel(
        uuid=d.get("aivm_model_uuid", uuid),
        name=d.get("name", ""),
        description=d.get("description", ""),
        files=files,
        raw=d,
    )


@dataclass
class HubSearchResult:
    uuid: str
    name: str
    description: str
    author: str
    downloads: int
    likes: int
    timbre: str
    category: str
    license_type: str
    size_mb: float
    styles: list[str]
    sample_url: str
    icon_url: str


def search_models(keyword: str = "", sort: str = "download",
                  page: int = 1, limit: int = 24,
                  ) -> tuple[int, list[HubSearchResult]]:
    params: dict[str, Any] = {"sort": sort, "page": page, "limit": limit}
    if keyword:
        params["keyword"] = keyword
    r = httpx.get(f"{API_BASE}/aivm-models/search", params=params,
                  timeout=httpx.Timeout(30.0, connect=10.0))
    r.raise_for_status()
    d = r.json()
    out: list[HubSearchResult] = []
    for m in d.get("aivm_models", []):
        files = m.get("model_files", [])
        aivmx = next((f for f in files if f.get("model_type") == "AIVMX"),
                     files[0] if files else {})
        speakers = m.get("speakers", [])
        styles = [st.get("name", "") for sp in speakers
                  for st in sp.get("styles", [])]
        sample = ""
        for sp in speakers:
            for st in sp.get("styles", []):
                vs = st.get("voice_samples") or []
                if vs and not sample:
                    sample = vs[0].get("audio_url", "")
        out.append(HubSearchResult(
            uuid=m.get("aivm_model_uuid", ""),
            name=m.get("name", ""),
            description=m.get("description", ""),
            author=(m.get("user") or {}).get("name", ""),
            downloads=int(m.get("total_download_count", 0)),
            likes=int(m.get("like_count", 0)),
            timbre=m.get("voice_timbre", ""),
            category=m.get("category", ""),
            license_type=aivmx.get("license_type", ""),
            size_mb=round(int(aivmx.get("file_size", 0)) / 1e6, 1),
            styles=styles,
            sample_url=sample,
            icon_url=(speakers[0].get("icon_url", "")
                      if speakers else "")))
    return int(d.get("total", 0)), out


def download_url(uuid: str, model_type: str = "AIVMX") -> str:
    """Return the (redirecting) URL that serves the model file."""
    return f"{API_BASE}/aivm-models/{uuid}/download?model_type={model_type}"


def download_model(uuid: str, progress: ProgressCb | None = None,
                   model_type: str = "AIVMX") -> Path:
    info = fetch_model(uuid)
    safe = "".join(c if c.isalnum() or c in "._-" else "_"
                   for c in info.name) or uuid
    dest = paths.models_dir() / "hub" / f"{safe}-{uuid[:8]}.aivmx"
    if not dest.exists():
        download(download_url(uuid, model_type), dest, progress)
        expected = next(
            (f.checksum for f in info.files if f.model_type == model_type),
            "")
        expected = expected.lower().removeprefix("sha256:")
        if expected:
            got = hashlib.sha256(dest.read_bytes()).hexdigest()
            if got != expected:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"モデルファイルのチェックサム不一致: {got} != {expected}")
    return dest
