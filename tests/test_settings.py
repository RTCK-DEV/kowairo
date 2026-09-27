"""Settings load/save roundtrip, forward-compat keys, corrupt input."""

from __future__ import annotations

import json

from kowairo import paths, settings
from kowairo.settings import Settings, load, save


def test_roundtrip(tmp_path):
    s = Settings(style_id=42, noise_gate_db=-45.0,
                 extra={"presets": {"a": {"speed_scale": 1.2}}})
    save(s)
    settings._cache = None
    got = load()
    assert got.style_id == 42
    assert got.noise_gate_db == -45.0
    assert got.extra["presets"]["a"]["speed_scale"] == 1.2


def test_unknown_keys_landed_in_extra():
    paths.data_dir().mkdir(parents=True, exist_ok=True)
    paths.settings_path().write_text(
        json.dumps({"future_option": True, "style_id": 9}), "utf-8")
    s = load()
    assert s.style_id == 9
    assert s.extra["future_option"] is True


def test_wrong_type_keeps_default():
    paths.data_dir().mkdir(parents=True, exist_ok=True)
    paths.settings_path().write_text(json.dumps({
        "vad_threshold": "loud",        # str into float → rejected
        "asr_num_threads": True,        # bool into int → rejected
        "noise_gate_db": -30,           # int into float → coerced
        "extra": "junk",                # non-dict into dict → rejected
        "model_uuid": None,             # null into str|None → ok
    }), "utf-8")
    s = load()
    assert s.vad_threshold == 0.5
    assert s.asr_num_threads == 2
    assert s.noise_gate_db == -30.0
    assert isinstance(s.extra, dict)
    assert s.model_uuid is None


def test_corrupt_file_returns_defaults():
    paths.data_dir().mkdir(parents=True, exist_ok=True)
    paths.settings_path().write_text("{not json", "utf-8")
    assert load() == Settings()


def test_atomic_write(tmp_path):
    s = Settings(style_id=1)
    save(s)
    assert paths.settings_path().exists()
    assert not paths.settings_path().with_suffix(".tmp").exists()
