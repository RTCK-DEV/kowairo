"""Network helpers: free-port discovery and resumable downloads with progress."""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path

import httpx

ProgressCb = Callable[[int, int], None]  # (received_bytes, total_bytes_or_0)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port(host: str, port: int, timeout: float = 60.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def download(url: str, dest: Path, progress: ProgressCb | None = None,
             timeout: float = 600.0) -> Path:
    """Download `url` to `dest` atomically (writes .part then renames).

    A leftover `.part` from an aborted run is resumed via a Range request;
    if the server ignores it (200 instead of 206) the file is restarted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout,
                      headers=headers) as r:
        r.raise_for_status()
        if resume and r.status_code != httpx.codes.PARTIAL_CONTENT:
            resume = 0  # server ignored the Range header — restart
        total = int(r.headers.get("content-length", "0") or 0)
        if total:
            total += resume
        received = resume
        with part.open("ab" if resume else "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                received += len(chunk)
                if progress:
                    progress(received, total)
    part.replace(dest)
    return dest
