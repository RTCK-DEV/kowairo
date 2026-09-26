# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Kowairo (standalone GUI build).

Build:  pyinstaller kowairo.spec --noconfirm
Output: dist/Kowairo/Kowairo.exe (onedir)
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("sherpa_onnx", "sounddevice", "soundfile"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Windows: force the CPython-bundled OpenSSL DLLs to win TOC dedup. Qt's
# qopensslbackend links the same generic names and PyInstaller may otherwise
# pick copies from PATH (e.g. Git's mingw64 OpenSSL), which breaks _ssl.pyd.
if sys.platform == "win32":
    for _name in ("libcrypto-3-x64.dll", "libssl-3-x64.dll"):
        _p = Path(sys.base_prefix) / "DLLs" / _name
        if _p.exists():
            binaries.append((str(_p), "."))

a = Analysis(
    ["launcher.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchaudio", "matplotlib", "tkinter", "PyQt5", "PyQt6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Kowairo",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Kowairo",
)

app = BUNDLE(
    coll,
    name="Kowairo.app",
    bundle_identifier="app.kowairo",
) if __import__("sys").platform == "darwin" else None
