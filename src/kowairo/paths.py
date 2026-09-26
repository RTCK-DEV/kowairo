"""Per-platform directories for app data, models and the embedded engine."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir

from . import APP_NAME


@lru_cache
def data_dir() -> Path:
    """Writable per-user directory holding downloads, models and settings."""
    override = _env_dir()
    if override is not None:
        return override
    return Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))


def _env_dir() -> Path | None:
    import os

    raw = os.environ.get("KOWAIRO_DATA_DIR", "").strip()
    return Path(raw) if raw else None


def engine_dir() -> Path:
    return data_dir() / "engine"


def models_dir() -> Path:
    """Voice model (.aivmx) staging area; the engine keeps its own installed copy."""
    return data_dir() / "models"


def asr_models_dir() -> Path:
    return data_dir() / "asr"


def downloads_dir() -> Path:
    return data_dir() / "downloads"


def recordings_dir() -> Path:
    d = data_dir() / "recordings"
    return d


def settings_path() -> Path:
    return data_dir() / "settings.json"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def resource_dir() -> Path:
    """Directory that contains read-only bundled resources."""
    if is_frozen():
        return Path(sys._MEIPASS) / "kowairo_assets"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / "assets"
