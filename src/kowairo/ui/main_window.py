"""Main window — device selection, conversion controls, model management."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..asr.sherpa import SherpaReazonASR, ensure_silero_vad
from ..audio.devices import list_devices
from ..audio.pipeline import PipelineStats, VoiceChangerPipeline
from ..engine.client import EngineClient
from ..engine.manager import EngineManager
from ..settings import Settings, save
from ..util.audio import load_audio_file, pcm_to_wav
from .workers import (
    AsrSetupWorker,
    EngineSetupWorker,
    HubSearchWorker,
    ModelInstallWorker,
    ThumbFetchWorker,
)


class _Bridge(QObject):
    """Thread-safe signal bridge from pipeline threads into Qt."""

    log = Signal(str)
    text = Signal(str)
    level = Signal(float, float)
    stats = Signal(object)


class LevelMeter(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._db = -80.0
        self.setMinimumHeight(16)
        self.setMaximumHeight(16)

    def set_db(self, db: float) -> None:
        self._db = max(-80.0, min(0.0, db))
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#12141a"))
        p.drawRoundedRect(0, 0, w, h, h // 2, h // 2)
        fw = int(w * max(0.0, (self._db + 60.0) / 60.0))
        if fw > 0:
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor("#3ad06e"))
            grad.setColorAt(0.75, QColor("#e0b83a"))
            grad.setColorAt(1.0, QColor("#e0564f"))
            p.setBrush(grad)
            p.drawRoundedRect(0, 0, fw, h, h // 2, h // 2)
        p.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kowairo — AI Voice Changer")
        self.resize(1100, 780)
        self.setMinimumSize(960, 640)

        self.settings = self._load_settings()
        self._bridge = _Bridge()
        self._bridge.log.connect(self._log)
        # EngineManager log callbacks arrive on worker threads — route them
        # through a queued signal, never touch the UI directly.
        self.engine = EngineManager(version=self.settings.engine_version,
                                    use_gpu=self.settings.use_gpu,
                                    log=self._bridge.log.emit)
        self.client: EngineClient | None = None
        self.asr = SherpaReazonASR(num_threads=self.settings.asr_num_threads)
        self.pipeline: VoiceChangerPipeline | None = None
        self._bridge.text.connect(self._on_text)
        self._bridge.level.connect(self._on_level)
        self._bridge.stats.connect(self._on_stats)
        self._workers: list = []
        self._record_chunks: list[np.ndarray] = []
        self._lib_entries: list = []
        self._lib_page = 0
        self._media: QMediaPlayer | None = None
        self._audio_out: QAudioOutput | None = None
        self._thumb_placeholder = QPixmap(72, 72)
        self._thumb_placeholder.fill(QColor("#22242e"))
        self._pad_paths: list[str] = list(
            self.settings.extra.get("pads", [""] * 8))[:8]
        self._pad_paths += [""] * (8 - len(self._pad_paths))
        self._pad_buttons: list[QPushButton] = []

        self._build_ui()
        self._refresh_devices()
        QTimer.singleShot(100, self._bootstrap)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(root)

        left = QWidget()
        ll = QVBoxLayout(left)

        # --- devices
        dev = QGroupBox("オーディオデバイス")
        form = QFormLayout(dev)
        self.cmb_in = QComboBox()
        self.cmb_out = QComboBox()
        self.cmb_monitor = QComboBox()
        btn_refresh = QPushButton("再読み込み")
        btn_refresh.clicked.connect(self._refresh_devices)
        row = QHBoxLayout()
        row.addWidget(self.cmb_in, 1)
        row.addWidget(btn_refresh)
        form.addRow("入力 (マイク)", row)
        form.addRow("出力 (仮想デバイス/スピーカー)", self.cmb_out)
        row2 = QHBoxLayout()
        row2.addWidget(self.cmb_monitor, 1)
        self.chk_monitor = QCheckBox("モニター")
        row2.addWidget(self.chk_monitor)
        form.addRow("モニター出力", row2)
        btn_vcable = QPushButton("仮想デバイス (VB-CABLE) ガイド")
        btn_vcable.clicked.connect(self._open_vcable_guide)
        form.addRow(btn_vcable)
        ll.addWidget(dev)

        # --- voice
        voice = QGroupBox("ボイス設定")
        vf = QFormLayout(voice)
        prow = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.setMinimumWidth(140)
        prow.addWidget(self.cmb_preset, 1)
        btn_psave = QPushButton("保存")
        btn_psave.clicked.connect(self._preset_save)
        btn_pdel = QPushButton("削除")
        btn_pdel.clicked.connect(self._preset_delete)
        prow.addWidget(btn_psave)
        prow.addWidget(btn_pdel)
        pw = QWidget()
        pw.setLayout(prow)
        vf.addRow("プリセット", pw)
        self.cmb_preset.currentIndexChanged.connect(
            lambda *_: self._preset_apply())
        self.cmb_style = QComboBox()
        vf.addRow("モデル / スタイル", self.cmb_style)
        self.sld_speed = self._slider(50, 200, 100)
        self.sld_pitch = self._slider(-12, 12, 0)
        self.sld_intonation = self._slider(0, 200, 100)
        self.sld_volume = self._slider(0, 200, 100)
        vf.addRow("話速", self.sld_speed)
        vf.addRow("ピッチ補正 (半音)", self.sld_pitch)
        vf.addRow("抑揚", self.sld_intonation)
        vf.addRow("音量", self.sld_volume)
        self.chk_auto_speed = QCheckBox("入力の話速に追従")
        self.chk_auto_speed.setChecked(True)
        self.chk_auto_volume = QCheckBox("入力の音量に追従")
        self.chk_auto_volume.setChecked(True)
        self.chk_auto_pitch = QCheckBox("入力のピッチに自動追従")
        vf.addRow(self.chk_auto_speed)
        vf.addRow(self.chk_auto_volume)
        vf.addRow(self.chk_auto_pitch)
        self.chk_passthrough = QCheckBox("入力をそのまま出力 (パススルー)")
        self.chk_gpu = QCheckBox("GPU で音声合成 (Windows: DirectML / 対応環境: CUDA)")
        vf.addRow(self.chk_passthrough)
        vf.addRow(self.chk_gpu)
        ll.addWidget(voice)

        # --- mixer / FX (competitor parity: gain, noise gate, effects)
        mix = QGroupBox("ミキサー / エフェクト")
        mf = QFormLayout(mix)
        self.sld_in_gain = self._slider(0, 200, 100)
        self.sld_out_gain = self._slider(0, 200, 100)
        self.sld_gate = self._slider(-80, -20, -60)
        self.sld_gate.setToolTip("この dB 未満の入力を無音化 (オフは -80)")
        mf.addRow("入力ゲイン %", self.sld_in_gain)
        mf.addRow("ノイズゲート dB", self.sld_gate)
        mf.addRow("出力ゲイン %", self.sld_out_gain)
        self.chk_limiter = QCheckBox("リミッター (クリップ防止)")
        self.chk_limiter.setChecked(True)
        mf.addRow(self.chk_limiter)
        self.cmb_fx = QComboBox()
        for label, key in (("なし", "off"), ("エコー", "echo"),
                           ("リバーブ", "reverb"), ("ロボット", "robot")):
            self.cmb_fx.addItem(label, key)
        mf.addRow("エフェクト", self.cmb_fx)
        self.sld_fx = self._slider(0, 100, 50)
        mf.addRow("エフェクト量", self.sld_fx)
        self.chk_mute = QCheckBox("ミュート (入力を遮断)")
        mf.addRow(self.chk_mute)
        ll.addWidget(mix)

        # --- transport
        trans = QGroupBox("変換")
        tf = QVBoxLayout(trans)
        self.btn_start = QPushButton("▶ 変換開始")
        self.btn_start.setObjectName("accent")
        self.btn_start.setEnabled(False)
        self.btn_start.setMinimumHeight(48)
        self.btn_start.clicked.connect(self._toggle)
        tf.addWidget(self.btn_start)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("入力"))
        self.meter_in = LevelMeter()
        mrow.addWidget(self.meter_in, 1)
        mrow.addWidget(QLabel("出力"))
        self.meter_out = LevelMeter()
        mrow.addWidget(self.meter_out, 1)
        tf.addLayout(mrow)
        self.lbl_stats = QLabel("遅延: —")
        tf.addWidget(self.lbl_stats)
        ll.addWidget(trans)
        ll.addStretch(1)

        # --- right: tabs
        right = QTabWidget()
        tabs_root = right

        # log tab
        tab_log = QWidget()
        tl = QVBoxLayout(tab_log)
        self.txt_log = QTextEdit(readOnly=True)
        tl.addWidget(self.txt_log)
        tabs_root.addTab(tab_log, "ログ")

        # transcript tab
        tab_tr = QWidget()
        tt = QVBoxLayout(tab_tr)
        self.txt_transcript = QTextEdit(readOnly=True)
        tt.addWidget(self.txt_transcript)
        trow = QHBoxLayout()
        self.txt_manual = QLineEdit()
        self.txt_manual.setPlaceholderText("テキストを入力して読み上げ…")
        btn_speak = QPushButton("読み上げ")
        btn_speak.clicked.connect(self._speak_manual)
        trow.addWidget(self.txt_manual, 1)
        trow.addWidget(btn_speak)
        tt.addLayout(trow)
        tabs_root.addTab(tab_tr, "テキスト / 認識結果")

        # model tab
        tab_model = QWidget()
        tm = QVBoxLayout(tab_model)
        self.list_models = QListWidget()
        tm.addWidget(self.list_models)
        brow = QHBoxLayout()
        btn_import = QPushButton("ファイルから導入 (.aivmx / .zip)")
        btn_import.clicked.connect(self._import_model)
        brow.addWidget(btn_import)
        tm.addLayout(brow)
        hrow = QHBoxLayout()
        self.txt_hub = QLineEdit()
        self.txt_hub.setPlaceholderText("AivisHub モデル UUID (任意)")
        btn_hub = QPushButton("AivisHub から導入")
        btn_hub.clicked.connect(self._import_hub)
        hrow.addWidget(self.txt_hub, 1)
        hrow.addWidget(btn_hub)
        tm.addLayout(hrow)
        btn_reload = QPushButton("モデル一覧を更新")
        btn_reload.clicked.connect(self._refresh_models)
        tm.addWidget(btn_reload)
        self.chk_record = QCheckBox("変換後の音声を recordings/ に保存")
        tm.addWidget(self.chk_record)
        tabs_root.addTab(tab_model, "モデル管理")

        # AivisHub library tab
        tab_lib = QWidget()
        tv = QVBoxLayout(tab_lib)
        srow = QHBoxLayout()
        self.txt_lib_search = QLineEdit()
        self.txt_lib_search.setPlaceholderText("AivisHub のモデルを検索…")
        self.txt_lib_search.returnPressed.connect(self._lib_search)
        self.cmb_lib_sort = QComboBox()
        for label, key in (("人気順", "download"), ("いいね順", "like"),
                           ("新着順", "recent")):
            self.cmb_lib_sort.addItem(label, key)
        self.cmb_lib_sort.currentIndexChanged.connect(
            lambda *_: self._lib_search())
        btn_lib_search = QPushButton("検索")
        btn_lib_search.clicked.connect(self._lib_search)
        srow.addWidget(self.txt_lib_search, 1)
        srow.addWidget(self.cmb_lib_sort)
        srow.addWidget(btn_lib_search)
        tv.addLayout(srow)
        ls = QSplitter(Qt.Orientation.Horizontal)
        self.list_lib = QListWidget()
        self.list_lib.setObjectName("cards")
        self.list_lib.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_lib.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_lib.setIconSize(QSize(72, 72))
        self.list_lib.setGridSize(QSize(140, 120))
        self.list_lib.setSpacing(6)
        self.list_lib.setWordWrap(True)
        self.list_lib.setUniformItemSizes(True)
        self.list_lib.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list_lib.currentRowChanged.connect(self._lib_select)
        ls.addWidget(self.list_lib)
        self.txt_lib_detail = QTextEdit(readOnly=True)
        ls.addWidget(self.txt_lib_detail)
        ls.setStretchFactor(0, 1)
        ls.setStretchFactor(1, 1)
        tv.addWidget(ls, 1)
        arow = QHBoxLayout()
        self.btn_lib_play = QPushButton("▶ 試聴")
        self.btn_lib_play.setEnabled(False)
        self.btn_lib_play.clicked.connect(self._lib_play)
        self.btn_lib_install = QPushButton("このモデルを導入")
        self.btn_lib_install.setEnabled(False)
        self.btn_lib_install.clicked.connect(self._lib_install)
        arow.addWidget(self.btn_lib_play)
        arow.addWidget(self.btn_lib_install)
        arow.addStretch(1)
        self.btn_lib_more = QPushButton("さらに読み込む")
        self.btn_lib_more.setEnabled(False)
        self.btn_lib_more.clicked.connect(lambda: self._lib_search(more=True))
        arow.addWidget(self.btn_lib_more)
        tv.addLayout(arow)
        tabs_root.addTab(tab_lib, "オンラインライブラリ")

        # soundboard tab
        tab_sb = QWidget()
        sv = QVBoxLayout(tab_sb)
        sv.addWidget(QLabel(
            "パッドをクリックで再生 / 未設定ならファイル割り当て。"
            "右クリックで解除。F1〜F8 でも再生できます。"))
        grid = QGridLayout()
        for i in range(8):
            b = QPushButton(f"F{i+1}: —")
            b.setMinimumHeight(56)
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.clicked.connect(lambda _=False, k=i: self._pad_click(k))
            b.customContextMenuRequested.connect(
                lambda pos, k=i, btn=b: self._pad_menu(k, btn, pos))
            self._pad_buttons.append(b)
            grid.addWidget(b, i // 4, i % 4)
        sv.addLayout(grid)
        sv.addStretch(1)
        tabs_root.addTab(tab_sb, "サウンドボード")
        self._pad_refresh_labels()

        # setup progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_progress = QLabel("")
        self.lbl_progress.setVisible(False)
        pv = QVBoxLayout()
        pv.addWidget(self.lbl_progress)
        pv.addWidget(self.progress)
        pw = QWidget()
        pw.setLayout(pv)
        ll.addWidget(pw)

        root.addWidget(left)
        root.addWidget(right)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)

        # --- shortcuts (w-okada VCClient parity)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self._toggle)
        QShortcut(QKeySequence("Ctrl+Shift+P"), self,
                  lambda: self.chk_passthrough.toggle())
        QShortcut(QKeySequence("Ctrl+Shift+M"), self,
                  lambda: self.chk_mute.toggle())
        for i in range(8):
            QShortcut(QKeySequence(f"F{i+1}"), self,
                      lambda k=i: self._pad_play(k))
        self._preset_refresh()
        self._restore_voice_ui()
        self._wire_live()

    def _wire_live(self) -> None:
        """Live-apply mixer controls into the shared settings object so a
        running pipeline picks them up without a restart."""
        s = self.settings
        self.sld_in_gain.valueChanged.connect(
            lambda v: setattr(s, "input_gain", v / 100.0))
        self.sld_out_gain.valueChanged.connect(
            lambda v: setattr(s, "output_gain", v / 100.0))
        self.sld_gate.valueChanged.connect(
            lambda v: setattr(s, "noise_gate_db", float(v)))
        self.sld_fx.valueChanged.connect(
            lambda v: setattr(s, "fx_amount", v / 100.0))
        self.cmb_fx.currentIndexChanged.connect(
            lambda *_: setattr(s, "fx_mode",
                               self.cmb_fx.currentData() or "off"))
        self.chk_limiter.toggled.connect(
            lambda v: setattr(s, "limiter", bool(v)))
        self.chk_passthrough.toggled.connect(
            lambda v: setattr(s, "passthrough", bool(v)))
        self.chk_mute.toggled.connect(self._on_mute_toggled)

    def _on_mute_toggled(self, v: bool) -> None:
        self.settings.muted = bool(v)
        if self.pipeline is not None:
            self.pipeline.muted = bool(v)

    def _slider(self, lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _load_settings(self) -> Settings:
        from ..settings import load

        return load()

    def _refresh_devices(self) -> None:
        try:
            devs = list_devices()
        except Exception as e:
            self._log(f"デバイス列挙に失敗: {e}")
            return
        self.cmb_in.clear()
        self.cmb_out.clear()
        self.cmb_monitor.clear()
        for d in devs:
            label = f"{d.index}: {d.name}"
            if d.max_inputs > 0:
                self.cmb_in.addItem(label, d.index)
            if d.max_outputs > 0:
                self.cmb_out.addItem(label, d.index)
                self.cmb_monitor.addItem(label, d.index)
        for i in range(self.cmb_out.count()):
            if self.cmb_out.itemData(i) == sd_default_out(devs):
                self.cmb_out.setCurrentIndex(i)

    # ------------------------------------------------------------------ boot
    def _bootstrap(self) -> None:
        self._log("Kowairo 起動")
        self._lib_search()
        self._start_engine_setup()

    def _start_engine_setup(self) -> None:
        self.engine.use_gpu = self.settings.use_gpu
        w = EngineSetupWorker(self.engine)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_engine_ready)
        w.failed.connect(lambda m: self._fatal("エンジン起動失敗", m))
        self._keep(w)
        w.start()
        self._show_progress("エンジンを準備中…", True)

    def _keep(self, w) -> None:
        self._workers.append(w)
        w.finished.connect(lambda: self._workers.remove(w)
                         if w in self._workers else None)

    def _on_engine_ready(self, client: EngineClient) -> None:
        self.client = client
        self._log("エンジン接続完了")
        self._refresh_models()
        self._start_asr_setup()

    def _start_asr_setup(self) -> None:
        w = AsrSetupWorker(self.asr)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_asr_ready)
        w.failed.connect(lambda m: self._fatal("音声認識の準備に失敗", m))
        self._keep(w)
        w.start()

    def _on_asr_ready(self, _asr) -> None:
        self._hide_progress()
        self._log("音声認識の準備完了")
        self.btn_start.setEnabled(True)

    def _on_progress(self, stage: str, received: int, total: int) -> None:
        self._show_progress(stage, total == 0)
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(received)
            pct = received / total * 100
            self.lbl_progress.setText(
                f"{stage}: {received/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)")
        else:
            self.lbl_progress.setText(stage)

    def _show_progress(self, label: str, indeterminate: bool) -> None:
        self.lbl_progress.setVisible(True)
        self.progress.setVisible(True)
        self.lbl_progress.setText(label)
        if indeterminate:
            self.progress.setRange(0, 0)

    def _hide_progress(self) -> None:
        self.lbl_progress.setVisible(False)
        self.progress.setVisible(False)

    # ------------------------------------------------------------------ models
    def _refresh_models(self) -> None:
        if not self.client:
            return
        try:
            speakers = self.client.speakers()
        except Exception as e:
            self._log(f"モデル一覧の取得に失敗: {e}")
            return
        self.list_models.clear()
        self.cmb_style.clear()
        for sp in speakers:
            for st in sp.styles:
                label = f"{sp.name} / {st.name} (id={st.id})"
                self.cmb_style.addItem(label, st.id)
                self.list_models.addItem(label)
        if self.settings.style_id is not None:
            for i in range(self.cmb_style.count()):
                if self.cmb_style.itemData(i) == self.settings.style_id:
                    self.cmb_style.setCurrentIndex(i)
                    break
        if not speakers:
            self._log("音声モデルが未インストールです。BOOTH で入手した "
                      "downer.zip または .aivmx を「モデル管理」から導入してください。")

    def _import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "音声モデルを選択", str(Path.home()),
            "Voice models (*.aivmx *.zip);;All files (*)")
        if not path:
            return
        self._install_model(source=Path(path))

    def _import_hub(self) -> None:
        uuid = self.txt_hub.text().strip()
        if not uuid:
            QMessageBox.information(self, "AivisHub",
                                    "モデル UUID を入力してください")
            return
        self._install_model(hub_uuid=uuid)

    def _install_model(self, source: Path | None = None,
                       hub_uuid: str | None = None) -> None:
        if not self.client:
            return
        w = ModelInstallWorker(self.client, source=source, hub_uuid=hub_uuid)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_model_installed)
        w.failed.connect(lambda m: self._fatal("モデル導入に失敗", m))
        self._keep(w)
        w.start()

    def _on_model_installed(self, name: str) -> None:
        self._hide_progress()
        self._log(f"モデル導入完了: {name}")
        self._refresh_models()

    # ------------------------------------------------------------------ library
    def _lib_search(self, more: bool = False) -> None:
        page = self._lib_page + 1 if more else 1
        w = HubSearchWorker(
            keyword=self.txt_lib_search.text().strip(),
            sort=self.cmb_lib_sort.currentData() or "download",
            page=page)
        w.done.connect(lambda total, entries, p=page:
                       self._lib_results(total, entries, p))
        w.failed.connect(lambda m: self._log(f"ライブラリ検索に失敗: {m}"))
        self._keep(w)
        w.start()

    def _lib_results(self, total: int, entries: list, page: int) -> None:
        self._lib_page = page
        if page == 1:
            self._lib_entries = []
            self.list_lib.clear()
        base = len(self._lib_entries)
        self._lib_entries.extend(entries)
        for e in entries:
            it = QListWidgetItem(
                QIcon(self._thumb_placeholder),
                f"{e.name}\nDL {e.downloads:,}")
            it.setToolTip(f"{e.name} — {e.author}")
            self.list_lib.addItem(it)
        fetch = [(base + i, e.icon_url)
                 for i, e in enumerate(entries) if e.icon_url]
        if fetch:
            w = ThumbFetchWorker(fetch)
            w.thumb.connect(self._lib_thumb)
            self._keep(w)
            w.start()
        self.btn_lib_more.setEnabled(len(self._lib_entries) < total)
        self._log(f"ライブラリ: {len(self._lib_entries)}/{total} 件")

    def _lib_thumb(self, row: int, data: bytes) -> None:
        pm = QPixmap()
        if not pm.loadFromData(data):
            return
        item = self.list_lib.item(row)
        if item:
            item.setIcon(QIcon(pm.scaled(
                72, 72,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)))

    def _lib_entry(self):
        row = self.list_lib.currentRow()
        if 0 <= row < len(self._lib_entries):
            return self._lib_entries[row]
        return None

    def _lib_select(self, row: int) -> None:
        e = self._lib_entry()
        self.btn_lib_play.setEnabled(bool(e and e.sample_url))
        self.btn_lib_install.setEnabled(bool(e))
        if not e:
            return
        self.txt_lib_detail.setPlainText(
            f"{e.name}\n"
            f"作者: {e.author}\n"
            f"ライセンス: {e.license_type}   声質: {e.timbre}   "
            f"カテゴリ: {e.category}\n"
            f"サイズ: {e.size_mb} MB   DL数: {e.downloads:,}   いいね: {e.likes}\n"
            f"スタイル: {', '.join(e.styles) or '—'}\n"
            f"UUID: {e.uuid}\n\n{e.description}")

    def _lib_play(self) -> None:
        e = self._lib_entry()
        if not (e and e.sample_url):
            return
        if self._media is None:
            self._audio_out = QAudioOutput()
            self._media = QMediaPlayer()
            self._media.setAudioOutput(self._audio_out)
        self._media.setSource(QUrl(e.sample_url))
        self._media.play()
        self._log(f"試聴: {e.name}")

    def _lib_install(self) -> None:
        e = self._lib_entry()
        if e:
            self._log(f"{e.name} を AivisHub から導入します…")
            self._install_model(hub_uuid=e.uuid)

    # ------------------------------------------------------------------ presets
    _PRESET_KEYS = (
        "style_id", "speed_scale", "pitch_semitones", "intonation_scale",
        "volume_scale", "auto_speed", "auto_volume", "auto_pitch",
        "noise_gate_db", "input_gain", "output_gain", "limiter",
        "fx_mode", "fx_amount", "passthrough")

    def _presets(self) -> dict:
        return self.settings.extra.setdefault("presets", {})

    def _preset_refresh(self) -> None:
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        self.cmb_preset.addItem("(プリセット)", None)
        for name in sorted(self._presets()):
            self.cmb_preset.addItem(name, name)
        self.cmb_preset.blockSignals(False)

    def _collect_voice(self) -> dict:
        return {
            "style_id": self.cmb_style.currentData(),
            "speed_scale": self.sld_speed.value() / 100.0,
            "pitch_semitones": float(self.sld_pitch.value()),
            "intonation_scale": self.sld_intonation.value() / 100.0,
            "volume_scale": self.sld_volume.value() / 100.0,
            "auto_speed": self.chk_auto_speed.isChecked(),
            "auto_volume": self.chk_auto_volume.isChecked(),
            "auto_pitch": self.chk_auto_pitch.isChecked(),
            "noise_gate_db": float(self.sld_gate.value()),
            "input_gain": self.sld_in_gain.value() / 100.0,
            "output_gain": self.sld_out_gain.value() / 100.0,
            "limiter": self.chk_limiter.isChecked(),
            "fx_mode": self.cmb_fx.currentData() or "off",
            "fx_amount": self.sld_fx.value() / 100.0,
            "passthrough": self.chk_passthrough.isChecked(),
        }

    def _preset_save(self) -> None:
        name, ok = QInputDialog.getText(
            self, "プリセット保存", "プリセット名:")
        name = name.strip()
        if not ok or not name:
            return
        self._presets()[name] = self._collect_voice()
        save(self.settings)
        self._preset_refresh()
        for i in range(self.cmb_preset.count()):
            if self.cmb_preset.itemData(i) == name:
                self.cmb_preset.blockSignals(True)
                self.cmb_preset.setCurrentIndex(i)
                self.cmb_preset.blockSignals(False)
                break
        self._log(f"プリセット保存: {name}")

    def _preset_delete(self) -> None:
        name = self.cmb_preset.currentData()
        if not name:
            return
        self._presets().pop(name, None)
        save(self.settings)
        self._preset_refresh()
        self._log(f"プリセット削除: {name}")

    def _preset_apply(self) -> None:
        name = self.cmb_preset.currentData()
        if not name or name not in self._presets():
            return
        p = self._presets()[name]
        s = self.settings
        for k in self._PRESET_KEYS:
            if k in p:
                setattr(s, k, p[k])
        self._restore_voice_ui()
        save(s)
        if self.pipeline is not None:
            self._log(f"プリセット適用: {name}")

    def _restore_voice_ui(self) -> None:
        """Push persisted settings into the widgets (startup / preset apply)."""
        s = self.settings
        self.sld_speed.setValue(round(s.speed_scale * 100))
        self.sld_pitch.setValue(round(s.pitch_semitones))
        self.sld_intonation.setValue(round(s.intonation_scale * 100))
        self.sld_volume.setValue(round(s.volume_scale * 100))
        self.chk_auto_speed.setChecked(s.auto_speed)
        self.chk_auto_volume.setChecked(s.auto_volume)
        self.chk_auto_pitch.setChecked(s.auto_pitch)
        self.sld_in_gain.setValue(round(s.input_gain * 100))
        self.sld_out_gain.setValue(round(s.output_gain * 100))
        self.sld_gate.setValue(round(s.noise_gate_db))
        self.chk_limiter.setChecked(s.limiter)
        self.sld_fx.setValue(round(s.fx_amount * 100))
        for i in range(self.cmb_fx.count()):
            if self.cmb_fx.itemData(i) == s.fx_mode:
                self.cmb_fx.setCurrentIndex(i)
                break
        self.chk_passthrough.setChecked(s.passthrough)
        self.chk_mute.setChecked(s.muted)
        self.chk_monitor.setChecked(s.monitor_enabled)
        self.chk_record.setChecked(s.record_output)

    # ------------------------------------------------------------------ soundboard
    def _pad_refresh_labels(self) -> None:
        for i, b in enumerate(self._pad_buttons):
            p = self._pad_paths[i]
            name = Path(p).stem if p else "—"
            if len(name) > 14:
                name = name[:13] + "…"
            b.setText(f"F{i+1}: {name}")
            b.setToolTip(p or "クリックで音声ファイルを割り当て")

    def _pad_click(self, i: int) -> None:
        if self._pad_paths[i]:
            self._pad_play(i)
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "サウンドを割り当て", str(Path.home()),
            "Audio (*.wav *.mp3 *.flac *.ogg *.aiff);;All files (*)")
        if not path:
            return
        self._pad_paths[i] = path
        self.settings.extra["pads"] = list(self._pad_paths)
        save(self.settings)
        self._pad_refresh_labels()

    def _pad_menu(self, i: int, btn: QPushButton, pos) -> None:
        if not self._pad_paths[i]:
            return
        m = QMenu(self)
        act = m.addAction("割り当て解除")
        if act == m.exec(btn.mapToGlobal(pos)):
            self._pad_paths[i] = ""
            self.settings.extra["pads"] = list(self._pad_paths)
            save(self.settings)
            self._pad_refresh_labels()

    def _pad_play(self, i: int) -> None:
        p = self._pad_paths[i]
        if not p or not Path(p).exists():
            return
        try:
            pcm, rate = load_audio_file(p)
        except Exception as e:
            self._log(f"サウンド読み込み失敗: {e}")
            return
        if self.pipeline is not None:
            self.pipeline.play_external(pcm, rate)
        else:
            from .. import paths

            tmp = paths.data_dir() / "pad_preview.wav"
            tmp.write_bytes(pcm_to_wav(pcm, rate))
            if self._media is None:
                self._audio_out = QAudioOutput()
                self._media = QMediaPlayer()
                self._media.setAudioOutput(self._audio_out)
            self._media.setSource(QUrl.fromLocalFile(str(tmp)))
            self._media.play()

    def _open_vcable_guide(self) -> None:
        QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/"))
        self._log("VB-CABLE をインストール後、「出力」に CABLE Input を"
                  "選び、Discord/ゲーム側のマイクに CABLE Output を"
                  "設定してください")

    # ------------------------------------------------------------------ run
    def _toggle(self) -> None:
        if self.pipeline is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if not self.client:
            return
        s = self.settings
        s.input_device = str(self.cmb_in.currentData()) \
            if self.cmb_in.currentData() is not None else None
        s.output_device = str(self.cmb_out.currentData()) \
            if self.cmb_out.currentData() is not None else None
        s.monitor_enabled = self.chk_monitor.isChecked()
        s.monitor_device = str(self.cmb_monitor.currentData()) \
            if self.cmb_monitor.currentData() is not None else None
        s.style_id = self.cmb_style.currentData()
        s.speed_scale = self.sld_speed.value() / 100.0
        s.pitch_semitones = float(self.sld_pitch.value())
        s.intonation_scale = self.sld_intonation.value() / 100.0
        s.volume_scale = self.sld_volume.value() / 100.0
        s.auto_speed = self.chk_auto_speed.isChecked()
        s.auto_volume = self.chk_auto_volume.isChecked()
        s.auto_pitch = self.chk_auto_pitch.isChecked()
        s.passthrough = self.chk_passthrough.isChecked()
        s.record_output = self.chk_record.isChecked()
        s.input_gain = self.sld_in_gain.value() / 100.0
        s.output_gain = self.sld_out_gain.value() / 100.0
        s.noise_gate_db = float(self.sld_gate.value())
        s.limiter = self.chk_limiter.isChecked()
        s.fx_mode = self.cmb_fx.currentData() or "off"
        s.fx_amount = self.sld_fx.value() / 100.0
        s.muted = self.chk_mute.isChecked()
        gpu_now = self.chk_gpu.isChecked()
        if gpu_now != s.use_gpu:
            s.use_gpu = gpu_now
            save(s)
            QMessageBox.information(
                self, "GPU設定", "GPU設定はエンジン再起動後に有効になります。"
                "アプリを再起動してください。")
        save(s)
        if s.style_id is None:
            self._fatal("モデル未選択", "モデル管理タブでモデルを導入してください")
            return
        self._record_chunks.clear()
        self.pipeline = VoiceChangerPipeline(
            s, self.client, self.asr, ensure_silero_vad(),
            on_log=self._bridge.log.emit,
            on_text=self._bridge.text.emit,
            on_level=self._bridge.level.emit,
            on_stats=self._bridge.stats.emit,
            on_output=self._on_output_chunk)
        try:
            self.pipeline.start()
        except Exception as e:
            self.pipeline = None
            self._fatal("開始に失敗", str(e))
            return
        self.btn_start.setText("■ 停止")
        self.btn_start.setObjectName("stop")
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)
        self.pipeline.muted = self.chk_mute.isChecked()
        self._log("変換を開始しました")

    def _stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        if self._record_chunks:
            from .. import paths

            out = np.concatenate(self._record_chunks)
            paths.recordings_dir().mkdir(parents=True, exist_ok=True)
            name = "kowairo-" + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S") + ".wav"
            p = paths.recordings_dir() / name
            p.write_bytes(pcm_to_wav(out, 48000))
            self._log(f"録音を保存: {p}")
            self._record_chunks.clear()
        self.btn_start.setText("▶ 変換開始")
        self.btn_start.setObjectName("accent")
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)
        self._log("変換を停止しました")

    def _on_output_chunk(self, pcm: np.ndarray, rate: int) -> None:
        if self.chk_record.isChecked():
            self._record_chunks.append(pcm)

    def _speak_manual(self) -> None:
        text = self.txt_manual.text().strip()
        if not text or not self.client:
            return
        style = self.cmb_style.currentData()
        if style is None:
            return
        s = self.settings
        try:
            wav = self.client.synthesize(
                text, int(style),
                speed_scale=s.speed_scale,
                pitch_scale=s.pitch_scale + s.pitch_semitones * 0.0125,
                intonation_scale=s.intonation_scale,
                volume_scale=s.volume_scale,
                output_rate=48000)
        except Exception as e:
            self._log(f"合成エラー: {e}")
            return
        if self.pipeline is not None:
            from ..util.audio import wav_to_pcm

            pcm, r = wav_to_pcm(wav)
            self.pipeline.play_external(pcm, r)
            self._log(f"読み上げ: {text}")
        else:
            # no pipeline running — play through Qt Multimedia
            from .. import paths

            tmp = paths.data_dir() / "preview.wav"
            tmp.write_bytes(wav)
            if self._media is None:
                self._audio_out = QAudioOutput()
                self._media = QMediaPlayer()
                self._media.setAudioOutput(self._audio_out)
            self._media.setSource(QUrl.fromLocalFile(str(tmp)))
            self._media.play()
            self._log(f"読み上げ: {text}")

    # ------------------------------------------------------------------ events
    def _on_text(self, text: str) -> None:
        self.txt_transcript.append(text)

    def _on_level(self, in_db: float, out_db: float) -> None:
        if in_db > -1:
            self.meter_in.set_db(in_db)
        if out_db > -1:
            self.meter_out.set_db(out_db)

    def _on_stats(self, st: PipelineStats) -> None:
        self.lbl_stats.setText(
            f"遅延: 最新 {st.last_latency_ms:.0f}ms / 平均 "
            f"{st.avg_latency_ms:.0f}ms | ASR {st.asr_ms:.0f}ms | "
            f"TTS {st.tts_ms:.0f}ms | 発話 {st.utterances} 回 | "
            f"破棄 {st.dropped_ms:.0f}ms")

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}] {msg}")

    def _fatal(self, title: str, detail: str) -> None:
        self._hide_progress()
        self._log(f"{title}: {detail}")
        QMessageBox.critical(self, title, detail)

    def closeEvent(self, e) -> None:
        self._stop()
        if self._media is not None:
            self._media.stop()
        self.engine.stop()
        if self.client:
            self.client.close()
        for w in list(self._workers):
            w.wait(1500)
        e.accept()


def sd_default_out(devs) -> int:
    for d in devs:
        if d.is_default_output:
            return d.index
    return -1
