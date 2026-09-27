"""Shared pytest fixtures — isolate the per-user data dir per test."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# KOWAIRO_DATA_DIR must be set before any kowairo import touches
# paths.data_dir() (lru_cached).
_TMP = tempfile.mkdtemp(prefix="kowairo-test-")
os.environ["KOWAIRO_DATA_DIR"] = _TMP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from kowairo import paths, settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Point every per-user dir at tmp_path and clear caches."""
    monkeypatch.setenv("KOWAIRO_DATA_DIR", str(tmp_path))
    paths.data_dir.cache_clear()
    monkeypatch.setattr(settings, "_cache", None)
    yield
