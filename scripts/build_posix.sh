#!/usr/bin/env bash
# Build standalone Kowairo on macOS / Linux (run in repo root)
set -euo pipefail
python3 -m pip install -e ".[dev]"
pyinstaller kowairo.spec --noconfirm
echo "Done: dist/Kowairo/"
