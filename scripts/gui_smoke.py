"""Offscreen GUI smoke test: construct MainWindow, let bootstrap run briefly, quit."""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Kowairo")
    from kowairo.ui.main_window import MainWindow

    win = MainWindow()
    win.show()
    QTimer.singleShot(10000, win.close)
    QTimer.singleShot(10500, app.quit)
    rc = app.exec()
    print(f"GUI smoke OK rc={rc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
