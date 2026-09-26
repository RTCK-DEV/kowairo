"""PyInstaller entry point."""

import sys

from kowairo.app import main

# In frozen windowed builds sys.stderr is None and hard crashes would be
# silent — keep a persistent crash log under the data dir for diagnostics.
if getattr(sys, "frozen", False):
    import atexit
    import faulthandler
    from pathlib import Path

    from kowairo import paths

    _log = paths.data_dir() / "crash.log"
    _log.parent.mkdir(parents=True, exist_ok=True)
    _fh = _log.open("a", encoding="utf-8", errors="replace")
    faulthandler.enable(_fh)
    sys.stderr = _fh

    @atexit.register
    def _flush() -> None:
        try:
            _fh.flush()
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())
