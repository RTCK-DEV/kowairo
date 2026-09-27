"""Realtime voice-changer pipeline: mic → VAD → ASR → TTS → playback.

Threads (capture via sounddevice callback) hand immutable buffers through
queues so ASR and TTS run concurrently with I/O. Playback pulls from a bounded
PCM deque; when synthesis outpaces real-time playback the oldest queued audio
is dropped to keep end-to-end latency under the configured cap.

Chunk-streamed competitors (w-okada VCClient, Voidol) emit converted audio
while the user is still speaking. To approximate that with an ASR→TTS
cascade, the VAD periodically snapshots the in-flight utterance and an
interim worker transcribes it; clauses that are stable across consecutive
snapshots (and not the still-growing tail clause) are committed to the TTS
queue early, so output can begin mid-utterance instead of only at endpoint.

A file-driven mode (``run_file``) exercises the same path without audio
hardware for development and CI, and ``start(with_io=False)`` runs the live
threads with a real-time drain for latency benchmarking.
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
import soxr

from ..asr.base import ASRBackend
from ..engine.client import EngineClient
from ..settings import Settings
from ..util.audio import resample, rms_db, wav_to_pcm
from .fx import FxChain, apply_gain_limiter
from .pitch import median_f0, semitone_shift
from .vad import VadSegmenter

ASR_RATE = 16000
TTS_RATE = 48000  # requested from engine (outputSamplingRate)

_SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")
_CLAUSE = re.compile(r"(?<=[、,，])")
_INTERIM_TAIL_CHARS = 8    # chars left uncommitted (a word may grow)
_INTERIM_MAX_DELTA = 16    # per-commit cap → bounds a clause's audio size


@dataclass
class Utterance:
    pcm: np.ndarray
    t_end: float  # monotonic time when the speech segment ended
    seq: int = 0


@dataclass
class ClauseJob:
    text: str
    speed: float | None
    volume: float | None
    pitch: float | None
    utterance_end: float      # origin for last_latency (endpoint or snapshot)
    index: int                # enqueue order within the utterance (0 = first)
    seq: int = 0              # owning utterance number
    interim: bool = False     # committed from an interim ASR snapshot
    speech_start: float = 0.0  # approx. monotonic time speech began


@dataclass
class _InterimJob:
    seq: int                  # sequence number of the in-flight utterance
    pcm: np.ndarray           # snapshot of speech accumulated so far
    t_snap: float             # monotonic time the snapshot was taken


_EMPTY = np.zeros(0, np.float32)
LEVEL_NONE = float("nan")  # on_level sentinel: "this side did not change"


class _PlayBuf:
    """Bounded PCM playback queue drained by an audio callback.

    `pending` counts samples still queued in `chunks` (the in-flight `cur`
    chunk is already subtracted). Both `push` and `drain` take `self._lock`
    around the popleft/append so the callback thread and the producer never
    race.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunks: deque[np.ndarray] = deque()
        self.cur = _EMPTY
        self.pos = 0
        self.pending = 0

    def push(self, pcm: np.ndarray, cap: int) -> int:
        """Append a chunk and drop the oldest samples beyond `cap`
        (sample granularity — dropping a whole clause would silence the
        output whenever one clause exceeds the cap). Returns dropped count."""
        with self._lock:
            self.chunks.append(pcm)
            self.pending += pcm.size
            dropped = 0
            excess = self.pending - cap
            while excess > 0 and self.chunks:
                old = self.chunks[0]
                if old.size <= excess:
                    self.chunks.popleft()
                    self.pending -= old.size
                    dropped += old.size
                    excess -= old.size
                else:
                    self.chunks[0] = old[excess:]
                    self.pending -= excess
                    dropped += excess
                    excess = 0
            return dropped

    def drain(self, out: np.ndarray) -> None:
        """Fill 1-D float32 `out` with queued audio; zero-fill the rest."""
        n = out.shape[0]
        written = 0
        while written < n:
            if self.pos >= self.cur.size:
                with self._lock:
                    self.cur = self.chunks.popleft() if self.chunks else _EMPTY
                    self.pending -= self.cur.size
                self.pos = 0
                if self.cur.size == 0:
                    break
            take = min(n - written, self.cur.size - self.pos)
            out[written:written + take] = self.cur[self.pos:self.pos + take]
            self.pos += take
            written += take
        if written < n:
            out[written:] = 0.0


@dataclass
class PipelineStats:
    utterances: int = 0
    clauses: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    lat_sum: float = 0.0
    lat_n: int = 0
    # speech-start → first converted audio (the responsiveness competitors
    # advertise); lower is better, independent of utterance length
    last_respond_ms: float = 0.0
    respond_sum: float = 0.0
    respond_n: int = 0
    avg_respond_ms: float = 0.0
    asr_ms: float = 0.0
    tts_ms: float = 0.0
    interim_ms: float = 0.0   # last interim ASR decode time
    dropped_ms: float = 0.0


def split_clauses(text: str, max_len: int = 48) -> list[str]:
    """Split ASR text into TTS clauses (sentence → comma clause → length).

    Segments that stay longer than ``max_len`` after the clause split are
    hard-split at ``max_len`` — some ASR models emit no punctuation at all,
    and an unbounded clause would overflow the play buffer.
    """
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
    out: list[str] = []
    for p in parts:
        while len(p) > max_len:
            out.append(p[:max_len])
            p = p[max_len:]
        if p:
            out.append(p)
    # a hard split can leave a separator-only tail (…48字|。); fold it into
    # the previous chunk so the sentence break still reaches the synthesizer
    merged = out[:1]
    for p in out[1:]:
        if p.strip("。、！？!?, \n"):
            merged.append(p)
        else:
            merged[-1] += p
    return [p for p in merged if p.strip("。、！？!?, \n")]


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


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
                 on_stats: Callable[[PipelineStats], None] | None = None,
                 on_utterance: Callable[[Utterance], None] | None = None,
                 on_play: Callable[[ClauseJob | None, np.ndarray], None] | None
                 = None) -> None:
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
        self.on_utterance = on_utterance or (lambda u: None)
        self.on_play = on_play or (lambda j, p: None)

        self.stats = PipelineStats()
        self._utter_q: queue.Queue[Utterance | None] = queue.Queue(maxsize=8)
        self._clause_q: queue.Queue[ClauseJob | None] = queue.Queue(maxsize=32)
        self._play_q: queue.Queue[tuple[np.ndarray, ClauseJob | None] | None] = \
            queue.Queue(maxsize=64)
        self._interim_q: queue.Queue[_InterimJob | None] = queue.Queue(maxsize=1)
        self._seq = 0                      # number of offered utterances
        self._finalized_seq = 0            # utterances fully dispatched
        self._live: dict[int, dict] = {}   # per-utterance interim commit state
        self._commit_lock = threading.Lock()
        self._asr_lock = threading.Lock()  # shared recognizer: serialize decodes
        self._interim_busy = False         # interim worker mid-decode
        self._in_rs: soxr.ResampleStream | None = None
        self._live_rs: soxr.ResampleStream | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._monitor_stream: sd.OutputStream | None = None
        self._in_rate = ASR_RATE
        self._playbuf = _PlayBuf()
        self._monbuf = _PlayBuf()
        self._out_rate = TTS_RATE
        self._fx: FxChain | None = None
        self.muted = False

    def _new_vad(self) -> VadSegmenter:
        s = self.settings
        return VadSegmenter(
            self.vad_model,
            threshold=s.vad_threshold,
            min_silence_ms=s.vad_min_silence_ms,
            min_speech_ms=s.vad_min_speech_ms,
            max_speech_sec=s.max_speech_sec,
            on_speech=self._on_vad_progress,
            interim_after_ms=s.interim_after_ms,
            interim_every_ms=s.interim_every_ms,
        )

    # ------------------------------------------------------------ lifecycle
    def start(self, with_io: bool = True) -> None:
        """Start pipeline threads; with_io=False skips audio devices and
        drains the play buffer at real-time pace (headless benchmarking)."""
        self._stop.clear()
        self._spawn(self._asr_worker, "asr")
        self._spawn(self._tts_worker, "tts")
        self._spawn(self._play_worker, "play")
        # always spawned so toggling interim_asr mid-run takes effect
        self._spawn(self._interim_worker, "interim")
        try:
            if with_io:
                self._open_output()
                self._open_capture()
            else:
                self._spawn(self._drain_worker, "drain")
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
        for q in (self._utter_q, self._clause_q, self._play_q,
                  self._interim_q):
            # a full queue would drop the sentinel and leave the worker
            # parked on get() — evict one item to make room, then retry
            while True:
                try:
                    q.put_nowait(None)
                    break
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        q.get_nowait()
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        with self._commit_lock:
            self._live.clear()
        self._seq = self._finalized_seq = 0
        self._in_rs = self._live_rs = None
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
            # the mic stream is contiguous — keep resampler state across
            # 32 ms chunks so block boundaries carry no resampling artifacts
            self._in_rs = soxr.ResampleStream(
                rate, ASR_RATE, 1, dtype="float32", quality="VHQ")
            self._in_stream = sd.InputStream(
                device=dev_i, samplerate=rate, channels=1, dtype="float32",
                blocksize=int(rate * 0.032), callback=self._on_mic_rs,
                latency="low")
            self._in_stream.start()
            self._in_rate = rate
            self.on_log(
                f"入力: {rate} Hz でキャプチャ中 (内部で 16 kHz へ変換)")

    def _on_mic_rs(self, indata, frames, t, status) -> None:
        assert self._in_rs is not None
        pcm = np.asarray(self._in_rs.resample_chunk(indata[:, 0]), np.float32)
        self._dispatch_mic(pcm)

    def _on_mic(self, indata, frames, t, status) -> None:
        self._dispatch_mic(indata[:, 0].copy())

    def _dispatch_mic(self, pcm: np.ndarray) -> None:
        s = self.settings
        if s.input_gain != 1.0:
            pcm = np.clip(pcm * np.float32(s.input_gain), -1.0, 1.0)
        db = rms_db(pcm)
        self.on_level(db, LEVEL_NONE)
        if self.muted or db < s.noise_gate_db:
            # keep VAD timing alive (feed silence) so in-flight utterances
            # endpoint normally; passthrough emits silence instead of noise
            silent = np.zeros_like(pcm)
            if s.passthrough:
                self._enqueue_live(silent)
            else:
                for _seg in self.vad.feed(silent):
                    self._offer_utterance(_seg)
            return
        if s.passthrough:
            self._enqueue_live(pcm)
            return
        for seg in self.vad.feed(pcm):
            self._offer_utterance(seg)

    def _enqueue_live(self, pcm16k: np.ndarray) -> None:
        """Passthrough enqueue — the mic stream is contiguous, so resample
        with a persistent stream rather than per-chunk (avoids boundary
        ringing every 32 ms)."""
        if self._live_rs is None:
            self._live_rs = soxr.ResampleStream(
                ASR_RATE, self._out_rate, 1, dtype="float32", quality="VHQ")
        out = np.asarray(self._live_rs.resample_chunk(pcm16k), np.float32)
        if out.size:
            self._enqueue_play(out, None, src_rate=self._out_rate)

    def _on_vad_progress(self, pcm: np.ndarray) -> None:
        """VAD in-speech snapshot (capture thread) — queue for interim ASR.
        Only the freshest snapshot matters; drop when one is pending or the
        interim worker is still decoding."""
        if not self.settings.interim_asr or self._stop.is_set() \
                or self._interim_busy or not self._interim_q.empty():
            return
        with contextlib.suppress(queue.Full):
            self._interim_q.put_nowait(
                _InterimJob(self._seq + 1, pcm, time.monotonic()))

    def _offer_utterance(self, seg: np.ndarray) -> None:
        utt = Utterance(seg, time.monotonic(), seq=self._seq + 1)
        try:
            self._utter_q.put_nowait(utt)
        except queue.Full:
            self.on_log("発話キュー溢れ: セグメントを破棄")
            return
        self._seq += 1
        self.on_utterance(utt)

    def feed_pcm(self, pcm16k: np.ndarray) -> None:
        """External PCM source (file/stream) — same path as the mic callback."""
        self._dispatch_mic(pcm16k)

    def flush(self) -> None:
        for seg in self.vad.flush():
            self._offer_utterance(seg)

    # ------------------------------------------------------------ ASR
    def _asr_worker(self) -> None:
        while True:
            item = self._utter_q.get()
            if item is None:
                return
            t0 = time.monotonic()
            try:
                # the sherpa recognizer is shared with the interim worker —
                # serialize decode_stream calls (not thread-safe)
                with self._asr_lock:
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

    # ------------------------------------------------------ interim ASR
    def _interim_worker(self) -> None:
        """Decode in-flight utterance snapshots so stable clauses can be
        synthesized before the utterance endpoints (continuous output like
        chunk-streamed competitors)."""
        while True:
            job = self._interim_q.get()
            if job is None or self._stop.is_set():
                return
            if job.seq <= self._finalized_seq:
                continue  # endpoint decode already dispatched it
            self._interim_busy = True
            t0 = time.monotonic()
            try:
                with self._asr_lock:
                    text = self.asr.transcribe(job.pcm, ASR_RATE)
            except Exception as e:
                self.on_log(f"中間認識エラー: {e}")
                continue
            finally:
                self._interim_busy = False
            self.stats.interim_ms = (time.monotonic() - t0) * 1000
            text = text or ""
            self.on_log(f"中間認識 {job.pcm.size / ASR_RATE:.1f}s → "
                        f"{text[:24]}{'…' if len(text) > 24 else ''}")
            if text:
                self._commit_interim(job, text)

    def _live_state(self, seq: int) -> dict:
        st = self._live.get(seq)
        if st is None:
            st = {"committed_text": "", "prev_stable": "",
                  "next_index": 0, "diverged": False}
            self._live[seq] = st
            while len(self._live) > 4:  # bound bookkeeping
                del self._live[min(self._live)]
        return st

    def _commit_interim(self, job: _InterimJob, text: str) -> None:
        # commit the confirmed text prefix: punctuation is unreliable (some
        # ASR models emit none), so stability is defined at the character
        # level — text seen identically in two consecutive snapshots, minus
        # a tail margin whose word may still be growing
        dur = job.pcm.size / ASR_RATE
        pros = self._prosody(job.pcm, text, dur)
        with self._commit_lock:
            if job.seq <= self._finalized_seq:
                return
            st = self._live_state(job.seq)
            if st["diverged"]:
                return
            prev = st["prev_stable"]
            st["prev_stable"] = text
            if not prev:
                return
            stable = text[:_common_prefix_len(text, prev)]
            stable = stable[:max(0, len(stable) - _INTERIM_TAIL_CHARS)]
            committed: str = st["committed_text"]
            if not stable.startswith(committed):
                # ASR revised already-committed text; let the final decode
                # cover the remainder rather than double-committing guesses
                self.on_log("中間認識で確定済みテキストが修正されたため、"
                            "この発話の先回り合成を中断")
                st["diverged"] = True
                return
            # commit at most _INTERIM_MAX_DELTA per snapshot so a clause's
            # audio stays under the drop-to-live-edge cap; committed_text
            # always stays an exact prefix of stable — separators stay in
            # the delta so sentence breaks reach the synthesizer
            tail = stable[len(committed):]
            delta = tail[:_INTERIM_MAX_DELTA]
            if not delta.strip("、。 "):
                return
            n = len(committed) + len(delta)
            cj = ClauseJob(
                delta, pros[0], pros[1], pros[2],
                job.t_snap, st["next_index"],
                seq=job.seq, interim=True,
                speech_start=job.t_snap - dur)
            if not self._put_clause(cj):
                # full queue: leave committed_text so the next snapshot
                # retries this span instead of silently skipping it
                return
            st["committed_text"] = stable[:n]
            st["next_index"] += 1
            self.on_log(f"先回り合成: {delta}")

    def _queue_text(self, text: str, utt: Utterance) -> None:
        s = self.settings
        dur = utt.pcm.size / ASR_RATE
        pros = self._prosody(utt.pcm, text, dur)
        # t_end includes the endpointing hangover (min_silence) only when the
        # segment ended on silence; approximating with it uniformly is fine
        speech_start = utt.t_end - dur - s.vad_min_silence_ms / 1000.0
        with self._commit_lock:
            st = self._live_state(utt.seq)
            committed: str = st["committed_text"]
            if text.startswith(committed):
                # synthesize only what interim commits did not already queue
                tail = text[len(committed):]
                clauses = split_clauses(tail)
            else:
                self.on_log(
                    f"最終認識が先回り文節と不一致 (先回り "
                    f"{len(committed)} 字) — 全文を合成し直し")
                clauses = split_clauses(text)
            for clause in clauses:
                self._put_clause(ClauseJob(
                    clause, pros[0], pros[1], pros[2],
                    utt.t_end, st["next_index"],
                    seq=utt.seq, speech_start=speech_start))
                st["next_index"] += 1
            self._finalized_seq = max(self._finalized_seq, utt.seq)
            self._live.pop(utt.seq, None)

    def _prosody(self, pcm: np.ndarray, text: str, dur: float
                 ) -> tuple[float | None, float | None, float | None]:
        """(speed, volume, pitch) overrides measured from the input audio."""
        s = self.settings
        cps = _chars_per_sec(text, dur)
        speed = float(np.clip(cps / 8.0, 0.7, 1.6)) if s.auto_speed else None
        volume = (float(np.clip(10 ** ((rms_db(pcm) + 20.0) / 30.0),
                                0.5, 1.8))
                  if s.auto_volume else None)
        if s.auto_pitch:
            st = semitone_shift(median_f0(pcm, ASR_RATE), 220.0) \
                + s.pitch_semitones
            pitch: float | None = float(np.clip(st * 0.0125, -0.15, 0.15))
        elif s.pitch_semitones:
            pitch = float(np.clip(s.pitch_semitones * 0.0125, -0.15, 0.15))
        else:
            pitch = None
        return speed, volume, pitch

    def _put_clause(self, job: ClauseJob) -> bool:
        try:
            self._clause_q.put_nowait(job)
            return True
        except queue.Full:
            self.on_log("TTSキュー溢れ: 文節を破棄")
            return False

    # ------------------------------------------------------------ TTS
    def _tts_worker(self) -> None:
        s = self.settings
        if s.style_id is not None:
            # pre-warm: first engine synthesis pays lazy model/ONNX load
            # (measured ~5 s cold vs ~0.4 s warm) — competitors warm at
            # model selection; do it here so the first real clause is fast
            with contextlib.suppress(Exception):
                self.client.synthesize(
                    "あ", s.style_id, output_rate=TTS_RATE)
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

    def _drain_worker(self) -> None:
        """Headless output: drain the play deque at real-time pace so the
        same drop policy/backlog behavior applies without audio hardware."""
        block = int(self._out_rate * 0.02)
        while not self._stop.is_set():
            t0 = time.monotonic()
            out = np.zeros((block, 1), np.float32)
            self._playbuf.drain(out[:, 0])
            dt = time.monotonic() - t0
            time.sleep(max(0.0, 0.02 - dt))

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

    def play_external(self, pcm: np.ndarray, src_rate: int) -> None:
        """Mix external audio (soundboard / manual playback) into output."""
        self._enqueue_play(pcm, None, src_rate=src_rate)

    def _postprocess(self, pcm: np.ndarray) -> np.ndarray:
        s = self.settings
        if self._fx is None:
            self._fx = FxChain(s.fx_mode, s.fx_amount, self._out_rate)
        self._fx.set(s.fx_mode, s.fx_amount)
        pcm = self._fx.apply(pcm)
        return apply_gain_limiter(pcm, s.output_gain, s.limiter)

    def _enqueue_play(self, pcm: np.ndarray, job: ClauseJob | None,
                      src_rate: int = TTS_RATE) -> None:
        if src_rate != self._out_rate:
            pcm = resample(pcm, src_rate, self._out_rate)
        pcm = self._postprocess(pcm)
        cap = int(self._out_rate * self.settings.drop_when_behind_ms / 1000)
        dropped = self._playbuf.push(pcm, cap)
        if self._monitor_stream is not None:
            self._monbuf.push(pcm, cap)
        if dropped:
            self.stats.dropped_ms += dropped / self._out_rate * 1000
        if job is not None and job.index == 0:
            now = time.monotonic()
            lat = (now - job.utterance_end) * 1000
            self.stats.last_latency_ms = lat
            self.stats.lat_sum += lat
            self.stats.lat_n += 1
            self.stats.avg_latency_ms = self.stats.lat_sum / self.stats.lat_n
            if job.speech_start:
                resp = (now - job.speech_start) * 1000
                self.stats.last_respond_ms = resp
                self.stats.respond_sum += resp
                self.stats.respond_n += 1
                self.stats.avg_respond_ms = (
                    self.stats.respond_sum / self.stats.respond_n)
            self.on_stats(self.stats)
        self.on_play(job, pcm)

    def _on_out(self, outdata, frames, t, status) -> None:
        self._playbuf.drain(outdata[:, 0])
        self.on_level(LEVEL_NONE, rms_db(outdata[:, 0]))

    def _on_out_monitor(self, outdata, frames, t, status) -> None:
        self._monbuf.drain(outdata[:, 0])

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
        s = self.settings
        # file mode is sequential — interim snapshots would never be consumed
        vad = VadSegmenter(
            self.vad_model, threshold=s.vad_threshold,
            min_silence_ms=s.vad_min_silence_ms,
            min_speech_ms=s.vad_min_speech_ms,
            max_speech_sec=s.max_speech_sec)

        segments = list(vad.feed(pcm)) + list(vad.flush())
        for seg in segments:
            self._seq += 1
            utt = Utterance(seg, time.monotonic(), seq=self._seq)
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
                    emotion_scale=s.emotion_scale,
                    output_rate=TTS_RATE)
                tts_ms = (time.monotonic() - t0) * 1000
                cp, r = wav_to_pcm(wav)
                cp = resample(cp, r, TTS_RATE)
                out_chunks.append(self._postprocess(cp))
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
