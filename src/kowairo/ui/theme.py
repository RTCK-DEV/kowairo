"""Dark modern theme — palette + application stylesheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#7c6cf0"
ACCENT_HOVER = "#8f7fff"
BG = "#17181e"
PANEL = "#1f2129"
SUNKEN = "#12141a"
BORDER = "#2e3240"
TEXT = "#e6e8ef"
MUTED = "#98a0b3"

_STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Yu Gothic UI", "Hiragino Sans", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {MUTED};
    font-weight: bold;
}}
QPushButton {{
    background: #2b2f3d;
    border: 1px solid #3a3f52;
    border-radius: 8px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: #363b4d; border-color: #4a5170; }}
QPushButton:pressed {{ background: #222633; }}
QPushButton:disabled {{ color: #5a6072; background: #22242e; }}
QPushButton#accent {{
    background: {ACCENT};
    border: none;
    color: #ffffff;
    font-weight: bold;
    font-size: 15px;
    padding: 10px;
}}
QPushButton#accent:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#accent:pressed {{ background: #5a4dd0; }}
QPushButton#accent:disabled {{ background: #3a3d52; color: #8088a0; }}
QPushButton#stop {{ background: #d05663; }}
QPushButton#stop:hover {{ background: #e06572; }}
QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SUNKEN};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {MUTED};
    margin-right: 8px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    top: -1px;
    background: {PANEL};
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 8px 16px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; background: #242834; }}
QListWidget {{ outline: none; }}
QListWidget::item {{ border-radius: 8px; padding: 4px; }}
QListWidget::item:selected {{ background: #3a3357; }}
QListWidget::item:hover:!selected {{ background: #242836; }}
QListWidget#cards {{ background: {SUNKEN}; }}
QListWidget#cards::item {{ color: {TEXT}; }}
QSlider::groove:horizontal {{
    height: 6px; background: #2a2e3c; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT}; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px; background: #ffffff;
    border: 2px solid {ACCENT};
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BORDER}; background: {SUNKEN};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}
QProgressBar {{
    background: {SUNKEN};
    border: none;
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: {MUTED};
    font-size: 10px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #3a3f52; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #4a5170; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #3a3f52; border-radius: 5px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; width: 4px; }}
QToolTip {{
    background: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 4px 8px;
}}
QMessageBox QLabel {{ color: {TEXT}; }}
QMenu {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
}}
QMenu::item:selected {{ background: {ACCENT}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(SUNKEN))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(_STYLESHEET)
