"""JSON-backed persistent settings."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields
from typing import Any

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
    drop_when_behind_ms: int = 900

    record_output: bool = False
    passthrough: bool = False

    extra: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_cache: Settings | None = None


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
                known = {f.name for f in fields(Settings)}
                for k, v in raw.items():
                    if k in known:
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
        p.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
