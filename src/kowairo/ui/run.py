"""Qt bootstrap for the GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Kowairo")
    app.setOrganizationName("Kowairo")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()
