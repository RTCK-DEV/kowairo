# Build standalone Kowairo.exe on Windows (run in repo root)
$ErrorActionPreference = "Stop"
python -m pip install -e ".[dev]"
pyinstaller kowairo.spec --noconfirm
Write-Host "Done: dist\Kowairo\Kowairo.exe"
