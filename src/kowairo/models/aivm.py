"""Voice model file handling: locate .aivmx inside the BOOTH zip or a folder."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from .. import paths


class ModelImportError(RuntimeError):
    pass


def find_aivmx(source: Path) -> list[Path]:
    """Return .aivmx files contained in `source` (file, dir or .zip)."""
    if source.is_dir():
        return sorted(source.rglob("*.aivmx"))
    if source.suffix.lower() == ".aivmx":
        return [source]
    if source.suffix.lower() == ".zip":
        out: list[Path] = []
        extract_dir = paths.models_dir() / "imported" / source.stem
        with zipfile.ZipFile(source) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(".aivmx") and not n.endswith("/")]
            if not names:
                raise ModelImportError(
                    f"{source.name} 内に .aivmx が見つかりません")
            extract_dir.mkdir(parents=True, exist_ok=True)
            for n in names:
                dest = extract_dir / Path(n).name
                with z.open(n) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(dest)
        return out
    raise ModelImportError(f"未対応のファイル形式です: {source.name}")


def aivmx_display_name(path: Path) -> str:
    """Best-effort display name without parsing the ONNX payload."""
    return path.stem
