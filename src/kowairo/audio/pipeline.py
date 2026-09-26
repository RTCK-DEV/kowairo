"""Realtime voice-changer pipeline: mic → VAD → ASR → TTS → playback.

Threads (capture via sounddevice callback) hand immutable buffers through
queues so ASR and TTS run concurrently with I/O. Playback pulls from a bounded
PCM deque; when synthesis outpaces real-time playback the oldest queued audio
is dropped to keep end-to-end latency under the configured cap.

A file-driven mode (``run_file``) exercises the same path without audio
hardware for development and CI.
"""

from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

from ..asr.base import ASRBackend
from ..engine.client import EngineClient
from ..settings import Settings
from ..util.audio import resample, rms_db, wav_to_pcm
from .pitch import median_f0, semitone_shift
from .vad import VadSegmenter

ASR_RATE = 16000
TTS_RATE = 48000  # requested from engine (outputSamplingRate)

_SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")
_CLAUSE = re.compile(r"(?<=[、,，])")


@dataclass
class Utterance:
    pcm: np.ndarray
    t_end: float  # monotonic time when the speech segment ended


@dataclass
class ClauseJob:
    text: str
    speed: float | None
    volume: float | None
    pitch: float | None
    utterance_end: float
    index: int


@dataclass
class PipelineStats:
    utterances: int = 0
    clauses: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    lat_sum: float = 0.0
    asr_ms: float = 0.0
    tts_ms: float = 0.0
    dropped_ms: float = 0.0


def split_clauses(text: str, max_len: int = 48) -> list[str]:
    """Split ASR text into TTS clauses (sentence → comma clause → length)."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    for sent in _SENTENCE_END.split(text):
        if not sent:
            continue
        if len(sent) <= max_len:
            parts.append(sent)
            continue
        buf = ""
        for c in _CLAUSE.split(sent):
            if buf and len(buf) + len(c) > max_len:
                parts.append(buf)
                buf = c
            else:
                buf += c
        if buf:
            parts.append(buf)
    return [p for p in parts if p.strip("。、！？!?, \n")]


def _chars_per_sec(text: str, seconds: float) -> float:
    n = len(re.sub(r"[、。！？!?,\s]", "", text))
    return n / max(seconds, 1e-3)


class VoiceChangerPipeline:
    """Speech → (ASR) → text → (TTS) → converted speech."""

    def __init__(self, settings: Settings, client: EngineClient,
                 asr: ASRBackend, vad_model: Path,
                 on_log: Callable[[str], None] | None = None,
                 on_text: Callable[[str], None] | None = None,
                 on_output: Callable[[np.ndarray, int], None] | None = None,
                 on_level: Callable[[float, float], None] | None = None,
                 on_stats: Callable[[PipelineStats], None] | None = None) -> None:
        self.settings = settings
        self.client = client
        self.asr = asr
        self.vad_model = vad_model
        self.vad = self._new_vad()
        self.on_log = on_log or (lambda m: None)
        self.on_text = on_text or (lambda t: None)
        self.on_output = on_output or (lambda p, r: None)
        self.on_level = on_level or (lambda i, o: None)
        self.on_stats = on_stats or (lambda s: None)

        self.stats = PipelineStats()
        self._utter_q: queue.Queue[Utterance | None] = queue.Queue(maxsize=8)
        self._clause_q: queue.Queue[ClauseJob | None] = queue.Queue(maxsize=32)
        self._play_q: queue.Queue[tuple[np.ndarray, ClauseJob | None] | None] = \
            queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._monitor_stream: sd.OutputStream | None = None
        self._in_rate = ASR_RATE
        self._playbuf: deque[np.ndarray] = deque()
        self._monbuf: deque[np.ndarray] = deque()
        self._playbuf_samples = 0
        self._monbuf_samples = 0
        self._buf_lock = threading.Lock()
        self._cur = np.zeros(0, np.float32)
        self._cur_pos = 0
        self._cur_mon = np.zeros(0, np.float32)
        self._cur_mon_pos = 0
        self._out_rate = TTS_RATE

    def _new_vad(self) -> VadSegmenter:
        s = self.settings
        return VadSegmenter(
            self.vad_model,
            threshold=s.vad_threshold,
            min_silence_ms=s.vad_min_silence_ms,
            min_speech_ms=s.vad_min_speech_ms,
            max_speech_sec=s.max_speech_sec,
        )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._stop.clear()
        self._spawn(self._asr_worker, "asr")
        self._spawn(self._tts_worker, "tts")
        self._spawn(self._play_worker, "play")
        try:
            self._open_output()
            self._open_capture()
        except Exception:
            # workers/streams are already up — tear them down before
            # propagating so a failed start leaks nothing
            self.stop()
            raise

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=self._guarded, args=(fn, name),
                             daemon=True, name=f"kowairo-{name}")
        self._threads.append(t)
        t.start()

    def _guarded(self, fn, name: str) -> None:
        try:
            fn()
        except Exception as e:
            if not self._stop.is_set():
                self.on_log(f"[{name}] エラー: {e}")

    def stop(self) -> None:
        self._stop.set()
        for st in (self._in_stream, self._out_stream, self._monitor_stream):
            try:
                if st is not None:
                    st.abort()
                    st.close()
            except Exception:
                pass
        self._in_stream = self._out_stream = self._monitor_stream = None
        for q in (self._utter_q, self._clause_q, self._play_q):
            with contextlib.suppress(queue.Full):
                q.put(None, timeout=0.5)
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        self.vad = self._new_vad()

    # ------------------------------------------------------------ capture
    def _open_capture(self) -> None:
        s = self.settings
        dev = s.input_device
        dev_i = int(dev) if dev and str(dev).isdigit() else dev
        try:
            self._in_stream = sd.InputStream(
                device=dev_i, samplerate=ASR_RATE, channels=1,
                dtype="float32", blocksize=int(ASR_RATE * 0.032),
                callback=self._on_mic, latency="low")
            self._in_stream.start()
            self._in_rate = ASR_RATE
            self.on_log("入力: 16 kHz モノラルでキャプチャ中")
        except Exception:
            info = sd.query_devices(dev_i, "input")
            rate = int(info["default_samplerate"])
            self._in_stream = sd.InputStream(
                device=dev_i, samplerate=rate, channels=1, dtype="float32",
                blocksize=int(rate * 0.032), callback=self._on_mic_rs,
                latency="low")
            self._in_stream.start()
            self._in_rate = rate
            self.on_log(
                f"入力: {rate} Hz でキャプチャ中 (内部で 16 kHz へ変換)")

    def _on_mic_rs(self, indata, frames, t, status) -> None:
        pcm = resample(indata[:, 0], self._in_rate, ASR_RATE, "VHQ")
        self._dispatch_mic(pcm)

    def _on_mic(self, indata, frames, t, status) -> None:
        self._dispatch_mic(indata[:, 0].copy())

    def _dispatch_mic(self, pcm: np.ndarray) -> None:
        self.on_level(rms_db(pcm), -1.0)
        if self.settings.passthrough:
            self._enqueue_play(pcm, None, src_rate=ASR_RATE)
            return
        for seg in self.vad.feed(pcm):
            try:
                self._utter_q.put_nowait(Utterance(seg, time.monotonic()))
            except queue.Full:
                self.on_log("発話キュー溢れ: セグメントを破棄")

    def feed_pcm(self, pcm16k: np.ndarray) -> None:
        """External PCM source (file/stream) — same path as the mic callback."""
        self._dispatch_mic(pcm16k)

    def flush(self) -> None:
        for seg in self.vad.flush():
            with contextlib.suppress(queue.Full):
                self._utter_q.put(Utterance(seg, time.monotonic()), timeout=2)

    # ------------------------------------------------------------ ASR
    def _asr_worker(self) -> None:
        while True:
            item = self._utter_q.get()
            if item is None:
                return
            t0 = time.monotonic()
            try:
                text = self.asr.transcribe(item.pcm, ASR_RATE)
            except Exception as e:
                self.on_log(f"ASRエラー: {e}")
                continue
            asr_ms = (time.monotonic() - t0) * 1000
            self.stats.asr_ms = asr_ms
            if not text or len(text.strip("。、 ")) < 2:
                continue
            self.stats.utterances += 1
            self.on_text(text)
            self._queue_text(text, item)

    def _queue_text(self, text: str, utt: Utterance) -> None:
        s = self.settings
        dur = utt.pcm.size / ASR_RATE
        cps = _chars_per_sec(text, dur)
        speed = float(np.clip(cps / 8.0, 0.7, 1.6)) if s.auto_speed else None
        volume = (float(np.clip(10 ** ((rms_db(utt.pcm) + 20.0) / 30.0),
                                0.5, 1.8))
                  if s.auto_volume else None)
        if s.auto_pitch:
            st = semitone_shift(median_f0(utt.pcm, ASR_RATE), 220.0) \
                + s.pitch_semitones
            pitch: float | None = float(np.clip(st * 0.0125, -0.15, 0.15))
        elif s.pitch_semitones:
            pitch = float(np.clip(s.pitch_semitones * 0.0125, -0.15, 0.15))
        else:
            pitch = None
        for i, clause in enumerate(split_clauses(text)):
            job = ClauseJob(clause, speed, volume, pitch, utt.t_end, i)
            try:
                self._clause_q.put_nowait(job)
            except queue.Full:
                self.on_log("TTSキュー溢れ: 文節を破棄")

    # ------------------------------------------------------------ TTS
    def _tts_worker(self) -> None:
        s = self.settings
        while True:
            job = self._clause_q.get()
            if job is None:
                return
            if s.style_id is None:
                continue
            t0 = time.monotonic()
            try:
                wav = self.client.synthesize(
                    job.text, s.style_id,
                    speed_scale=_mul(s.speed_scale, job.speed),
                    pitch_scale=_clamp(_add(s.pitch_scale, job.pitch),
                                       -0.15, 0.15),
                    intonation_scale=s.intonation_scale,
                    volume_scale=_clamp(_mul(s.volume_scale, job.volume),
                                        0.0, 2.0),
                    dynamics_scale=s.dynamics_scale,
                    emotion_scale=s.emotion_scale,
                    output_rate=TTS_RATE)
            except Exception as e:
                self.on_log(f"TTSエラー: {e}")
                continue
            self.stats.tts_ms = (time.monotonic() - t0) * 1000
            self.stats.clauses += 1
            pcm, rate = wav_to_pcm(wav)
            if rate != TTS_RATE:
                pcm = resample(pcm, rate, TTS_RATE)
            try:
                self._play_q.put_nowait((pcm, job))
            except queue.Full:
                self.on_log("再生キュー溢れ: 音声を破棄")

    # ------------------------------------------------------------ playback
    def _play_worker(self) -> None:
        """Move synthesized chunks from _play_q into the callback deques."""
        while True:
            item = self._play_q.get()
            if item is None:
                return
            pcm, job = item
            self.on_output(pcm, self._out_rate)
            self._enqueue_play(pcm, job)

    def _open_output(self) -> None:
        s = self.settings
        dev = s.output_device
        dev_i = int(dev) if dev and str(dev).isdigit() else dev
        rate = TTS_RATE
        try:
            sd.check_output_settings(device=dev_i, samplerate=rate, channels=1,
                                     dtype="float32")
        except Exception:
            info = sd.query_devices(dev_i, "output")
            rate = int(info["default_samplerate"])
        self._out_rate = rate
        self._out_stream = sd.OutputStream(
            device=dev_i, samplerate=rate, channels=1, dtype="float32",
            blocksize=int(rate * 0.02), callback=self._on_out, latency="low")
        self._out_stream.start()
        if s.monitor_enabled and s.monitor_device:
            m_i = int(s.monitor_device) if str(s.monitor_device).isdigit() \
                else s.monitor_device
            self._monitor_stream = sd.OutputStream(
                device=m_i, samplerate=rate, channels=1, dtype="float32",
                blocksize=int(rate * 0.02), callback=self._on_out_monitor,
                latency="low")
            self._monitor_stream.start()
        self.on_log(f"出力: {rate} Hz で再生中")

    def _enqueue_play(self, pcm: np.ndarray, job: ClauseJob | None,
                      src_rate: int = TTS_RATE) -> None:
        if src_rate != self._out_rate:
            pcm = resample(pcm, src_rate, self._out_rate)
        cap = int(self._out_rate * self.settings.drop_when_behind_ms / 1000)
        with self._buf_lock:
            self._playbuf.append(pcm)
            self._playbuf_samples += pcm.size
            self._monbuf.append(pcm)
            self._monbuf_samples += pcm.size
            while self._playbuf_samples > cap and self._playbuf:
                old = self._playbuf.popleft()
                self._playbuf_samples -= old.size
                self.stats.dropped_ms += old.size / self._out_rate * 1000
            while self._monbuf_samples > cap and self._monbuf:
                old = self._monbuf.popleft()
                self._monbuf_samples -= old.size
        if job is not None and job.index == 0:
            lat = (time.monotonic() - job.utterance_end) * 1000
            self.stats.last_latency_ms = lat
            self.stats.lat_sum += lat
            self.stats.avg_latency_ms = self.stats.lat_sum / max(
                1, self.stats.utterances)
            self.on_stats(self.stats)

    def _drain(self, outdata: np.ndarray, cur_attr: str, pos_attr: str,
               buf: deque[np.ndarray], cnt_attr: str) -> float:
        n = outdata.shape[0]
        written = 0
        cur: np.ndarray = getattr(self, cur_attr)
        pos: int = getattr(self, pos_attr)
        while written < n:
            if pos >= cur.size:
                with self._buf_lock:
                    cur = buf.popleft() if buf else np.zeros(0, np.float32)
                    if cur.size:
                        setattr(self, cnt_attr,
                                getattr(self, cnt_attr) - cur.size)
                pos = 0
                if cur.size == 0:
                    break
            take = min(n - written, cur.size - pos)
            outdata[written:written + take, 0] = cur[pos:pos + take]
            pos += take
            written += take
        setattr(self, cur_attr, cur)
        setattr(self, pos_attr, pos)
        if written < n:
            outdata[written:, 0] = 0.0
        return rms_db(outdata[:, 0])

    def _on_out(self, outdata, frames, t, status) -> None:
        db = self._drain(outdata, "_cur", "_cur_pos", self._playbuf,
                         "_playbuf_samples")
        self.on_level(-1.0, db)

    def _on_out_monitor(self, outdata, frames, t, status) -> None:
        self._drain(outdata, "_cur_mon", "_cur_mon_pos", self._monbuf,
                    "_monbuf_samples")

    # ------------------------------------------------------------ file mode
    def run_file(self, in_wav: Path,
                 progress: Callable[[str], None] | None = None
                 ) -> tuple[np.ndarray, list[str], PipelineStats]:
        """Drive the pipeline with a WAV file; returns (pcm@TTS_RATE, texts)."""
        log = progress or self.on_log
        pcm, rate = wav_to_pcm(in_wav.read_bytes())
        pcm = resample(pcm, rate, ASR_RATE, "VHQ")
        texts: list[str] = []
        out_chunks: list[np.ndarray] = []
        vad = self._new_vad()
        s = self.settings

        segments = list(vad.feed(pcm)) + list(vad.flush())
        for seg in segments:
            utt = Utterance(seg, time.monotonic())
            t0 = time.monotonic()
            text = self.asr.transcribe(seg, ASR_RATE)
            asr_ms = (time.monotonic() - t0) * 1000
            if not text or len(text.strip("。、 ")) < 2:
                continue
            texts.append(text)
            log(f"ASR ({asr_ms:.0f}ms): {text}")
            self._queue_text(text, utt)
            while True:
                try:
                    job = self._clause_q.get(timeout=0.05)
                except queue.Empty:
                    break
                if job is None:
                    break
                t0 = time.monotonic()
                wav = self.client.synthesize(
                    job.text, s.style_id or 0,
                    speed_scale=_mul(s.speed_scale, job.speed),
                    pitch_scale=_clamp(_add(s.pitch_scale, job.pitch),
                                       -0.15, 0.15),
                    intonation_scale=s.intonation_scale,
                    volume_scale=_clamp(_mul(s.volume_scale, job.volume),
                                        0.0, 2.0),
                    dynamics_scale=s.dynamics_scale,
                    output_rate=TTS_RATE)
                tts_ms = (time.monotonic() - t0) * 1000
                cp, r = wav_to_pcm(wav)
                cp = resample(cp, r, TTS_RATE)
                out_chunks.append(cp)
                log(f"TTS ({tts_ms:.0f}ms): {job.text}")
        if out_chunks:
            return np.concatenate(out_chunks), texts, self.stats
        return np.zeros(0, np.float32), texts, self.stats


def _mul(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return a * b


def _add(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _clamp(v: float | None, lo: float, hi: float) -> float | None:
    if v is None:
        return None
    return float(np.clip(v, lo, hi))
