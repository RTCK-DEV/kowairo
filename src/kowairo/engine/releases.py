"""Resolution of the AivisSpeech Engine binary for the current platform.

The engine ships one 7z archive per OS/arch on GitHub Releases. On Windows the
bundled onnxruntime includes the DirectML provider, so ``--use_gpu`` accelerates
NVIDIA, AMD (Radeon) and Intel GPUs alike. macOS and Linux builds run on CPU.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

GITHUB_REPO = "Aivis-Project/AivisSpeech-Engine"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{{tag}}"
DOWNLOAD_BASE = f"https://github.com/{GITHUB_REPO}/releases/download/{{tag}}"


@dataclass(frozen=True)
class EngineAsset:
    tag: str
    archive_name: str
    checksum_name: str
    url: str
    checksum_url: str
    executable: str  # relative path inside the extracted tree


def platform_key() -> str:
    sysname = sys.platform
    machine = platform.machine().lower()
    if sysname == "win32":
        return "Windows-x64"
    if sysname == "darwin":
        return "macOS-arm64" if machine in ("arm64", "aarch64") else "macOS-x64"
    if sysname.startswith("linux"):
        return "Linux-arm64" if machine in ("arm64", "aarch64") else "Linux-x64"
    raise RuntimeError(f"unsupported platform: {sysname}")


def asset_for(version: str, sys_key: str | None = None) -> EngineAsset:
    key = sys_key or platform_key()
    tag = version.lstrip("v")
    archive = f"AivisSpeech-Engine-{key}-{tag}.7z.001"
    checksum = f"AivisSpeech-Engine-{key}-{tag}.7z.txt"
    exe = "run.exe" if key.startswith("Windows") else "run"
    return EngineAsset(
        tag=tag,
        archive_name=archive,
        checksum_name=checksum,
        url=f"{DOWNLOAD_BASE.format(tag=tag)}/{archive}",
        checksum_url=f"{DOWNLOAD_BASE.format(tag=tag)}/{checksum}",
        executable=exe,
    )
