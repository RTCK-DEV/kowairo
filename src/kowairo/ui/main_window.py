"""Main window — device selection, conversion controls, model management."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
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
from ..util.audio import pcm_to_wav
from .workers import AsrSetupWorker, EngineSetupWorker, ModelInstallWorker


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
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#202020"))
        frac = (self._db + 60.0) / 60.0
        fw = int(w * max(0.0, frac))
        grad = QColor("#3ad06e") if self._db < -6 else QColor("#e0a030")
        p.fillRect(0, 0, fw, h, grad)
        p.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kowairo — AI Voice Changer")
        self.resize(980, 720)

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
        ll.addWidget(dev)

        # --- voice
        voice = QGroupBox("ボイス設定")
        vf = QFormLayout(voice)
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

        # --- transport
        trans = QGroupBox("変換")
        tf = QVBoxLayout(trans)
        self.btn_start = QPushButton("▶ 変換開始")
        self.btn_start.setEnabled(False)
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setStyleSheet("font-size:16px; font-weight:bold;")
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
        # route through the playback queue when running, else play ad-hoc
        if self.pipeline is not None:
            from ..util.audio import wav_to_pcm

            pcm, r = wav_to_pcm(wav)
            self.pipeline._enqueue_play(pcm, None, src_rate=r)
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
            f"TTS {st.tts_ms:.0f}ms | 発話 {st.utterances} 回")

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}] {msg}")

    def _fatal(self, title: str, detail: str) -> None:
        self._hide_progress()
        self._log(f"{title}: {detail}")
        QMessageBox.critical(self, title, detail)

    def closeEvent(self, e) -> None:
        self._stop()
        self.engine.stop()
        if self.client:
            self.client.close()
        e.accept()


def sd_default_out(devs) -> int:
    for d in devs:
        if d.is_default_output:
            return d.index
    return -1
