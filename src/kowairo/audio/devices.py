"""Audio device enumeration via sounddevice (PortAudio)."""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


@dataclass
class AudioDevice:
    index: int
    name: str
    max_inputs: int
    max_outputs: int
    default_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


def list_devices() -> list[AudioDevice]:
    devs = sd.query_devices()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    try:
        default_out = sd.default.device[1]
    except Exception:
        default_out = -1
    out: list[AudioDevice] = []
    for i, d in enumerate(devs):
        out.append(AudioDevice(
            index=i,
            name=str(d["name"]),
            max_inputs=int(d["max_input_channels"]),
            max_outputs=int(d["max_output_channels"]),
            default_rate=float(d["default_samplerate"]),
            is_default_input=(i == default_in),
            is_default_output=(i == default_out),
        ))
    return out


def input_devices() -> list[AudioDevice]:
    return [d for d in list_devices() if d.max_inputs > 0]


def output_devices() -> list[AudioDevice]:
    return [d for d in list_devices() if d.max_outputs > 0]


def find_device(name_or_index: str | None, outputs: bool) -> int | None:
    if name_or_index is None:
        return None
    try:
        return int(name_or_index)
    except (ValueError, TypeError):
        pass
    for d in list_devices():
        if d.name == name_or_index:
            return d.index
    return None
