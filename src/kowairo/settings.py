"""JSON-backed persistent settings."""

from __future__ import annotations

import json
import threading
import types
from dataclasses import asdict, dataclass, field
from typing import Any, get_args, get_origin, get_type_hints

from . import paths


@dataclass
class Settings:
    input_device: str | None = None
    output_device: str | None = None
    monitor_device: str | None = None
    monitor_enabled: bool = False

    style_id: int | None = None
    model_uuid: str | None = None

    # TTS AudioQuery scales
    speed_scale: float = 1.0
    pitch_scale: float = 0.0
    intonation_scale: float = 1.0
    volume_scale: float = 1.0
    dynamics_scale: float = 1.0
    emotion_scale: float = 1.0

    # prosody following
    auto_speed: bool = True      # follow input speaking rate
    auto_volume: bool = True     # follow input loudness
    auto_pitch: bool = False     # map measured f0 -> pitchScale offset
    pitch_semitones: float = 0.0 # fixed semitone offset added on top

    # engine
    use_gpu: bool = False
    engine_version: str = "1.2.0"

    # asr
    asr_backend: str = "sherpa-reazonspeech"
    asr_num_threads: int = 2

    # pipeline
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 450
    vad_min_speech_ms: int = 160
    max_speech_sec: float = 15.0
    # bounded-lag cap: trimmed at sample granularity when the play backlog
    # exceeds this; must comfortably cover one interim clause (~16 chars ≈
    # 3 s of audio) or every clause start would be chopped mid-word
    drop_when_behind_ms: int = 3000

    # interim recognition — chunk-based competitors (w-okada VCClient, Voidol)
    # emit converted audio continuously while the user still speaks; an
    # ASR→TTS cascade can approximate that by decoding the in-flight utterance
    # periodically and committing clauses that are stable across snapshots.
    interim_asr: bool = True        # synthesize stable clauses mid-utterance
    interim_after_ms: int = 1600    # first interim decode after this much speech
    interim_every_ms: int = 1000    # subsequent interim cadence

    record_output: bool = False
    passthrough: bool = False
    muted: bool = False

    # competitor-parity controls (w-okada VCClient / VOIDOL)
    noise_gate_db: float = -60.0  # inputs below this are gated to silence (post-gain RMS)
    input_gain: float = 1.0
    output_gain: float = 1.0
    limiter: bool = True          # tanh soft-clip on output
    fx_mode: str = "off"          # off|echo|reverb|robot
    fx_amount: float = 0.5

    extra: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_cache: Settings | None = None
_HINTS = get_type_hints(Settings)


def _accepts(hint: Any, v: Any) -> bool:
    """Is JSON value `v` acceptable for a field annotated with `hint`?"""
    origin = get_origin(hint)
    if origin is not None and origin is not types.UnionType:
        # e.g. dict[str, Any] — validate the container type only
        return isinstance(v, origin)
    types_ = get_args(hint) or (hint,)
    for t in types_:
        if t is type(None):
            if v is None:
                return True
        elif t is bool:
            if isinstance(v, bool):
                return True
        elif t is int:
            if isinstance(v, int) and not isinstance(v, bool):
                return True
        elif t is float:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return True
        elif isinstance(t, type) and isinstance(v, t):
            return True
    return False


def load() -> Settings:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        s = Settings()
        p = paths.settings_path()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if k in _HINTS:
                            # keep the default on type mismatch — a corrupt
                            # settings.json must not poison the dataclass
                            if _accepts(_HINTS[k], v):
                                setattr(s, k, v)
                        else:
                            s.extra[k] = v
            except Exception:
                pass
        _cache = s
        return s


def save(s: Settings) -> None:
    global _cache
    with _lock:
        _cache = s
        p = paths.settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
