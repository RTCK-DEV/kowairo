"""AivisHub API client — fetch public .aivmx models without a browser login."""

from __future__ import annotations

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
    return dest
