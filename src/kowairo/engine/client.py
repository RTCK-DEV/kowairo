"""Synchronous HTTP client for a locally running AivisSpeech Engine.

The engine API is VOICEVOX-compatible: ``POST /audio_query`` builds a synthesis
plan from text, ``POST /synthesis`` renders it to 44.1 kHz WAV. Voice models are
managed through the AivisSpeech-specific ``/aivm_models`` endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


@dataclass
class Style:
    name: str
    id: int
    type: str = "talk"


@dataclass
class Speaker:
    name: str
    speaker_uuid: str
    styles: list[Style] = field(default_factory=list)


class EngineClient:
    def __init__(self, host: str, port: int) -> None:
        self.base = f"http://{host}:{port}"
        self._http = httpx.Client(base_url=self.base, timeout=DEFAULT_TIMEOUT)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------- meta
    def version(self) -> str:
        r = self._http.get("/version")
        r.raise_for_status()
        return str(r.json()).strip('"')

    def engine_manifest(self) -> dict[str, Any]:
        r = self._http.get("/engine_manifest")
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    def speakers(self) -> list[Speaker]:
        r = self._http.get("/speakers")
        r.raise_for_status()
        out: list[Speaker] = []
        for sp in r.json():
            styles = [
                Style(name=st.get("name", ""), id=int(st["id"]),
                      type=st.get("type", "talk"))
                for st in sp.get("styles", [])
            ]
            out.append(Speaker(name=sp.get("name", ""),
                               speaker_uuid=sp.get("speakerUuid", ""),
                               styles=styles))
        return out

    def aivm_infos(self) -> dict[str, Any]:
        r = self._http.get("/aivm_models")
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------- models
    def install_model_file(self, path: Path) -> None:
        with path.open("rb") as f:
            r = self._http.post("/aivm_models/install",
                                files={"file": (path.name, f,
                                                "application/octet-stream")})
        r.raise_for_status()

    def install_model_url(self, url: str) -> None:
        r = self._http.post("/aivm_models/install", data={"url": url},
                            timeout=httpx.Timeout(connect=5.0, read=600.0,
                                                  write=30.0, pool=5.0))
        r.raise_for_status()

    def load_model(self, aivm_model_uuid: str) -> None:
        r = self._http.post(f"/aivm_models/{aivm_model_uuid}/load")
        r.raise_for_status()

    def unload_model(self, aivm_model_uuid: str) -> None:
        r = self._http.post(f"/aivm_models/{aivm_model_uuid}/unload")
        r.raise_for_status()

    # ------------------------------------------------------------- synthesis
    def audio_query(self, text: str, style_id: int) -> dict[str, Any]:
        r = self._http.post("/audio_query",
                            params={"text": text, "speaker": style_id})
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    def synthesize_query(self, query: dict[str, Any], style_id: int) -> bytes:
        r = self._http.post("/synthesis", params={"speaker": style_id},
                            json=query,
                            headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return r.content

    def synthesize(self, text: str, style_id: int, *,
                   speed_scale: float | None = None,
                   pitch_scale: float | None = None,
                   intonation_scale: float | None = None,
                   volume_scale: float | None = None,
                   dynamics_scale: float | None = None,
                   emotion_scale: float | None = None,
                   output_rate: int = 48000) -> bytes:
        """text -> WAV bytes. Applies scale overrides onto the audio query."""
        q = self.audio_query(text, style_id)
        overrides: dict[str, Any] = {
            "speedScale": speed_scale,
            "pitchScale": pitch_scale,
            "intonationScale": intonation_scale,
            "volumeScale": volume_scale,
            "dynamicsScale": dynamics_scale,
            "emotionScale": emotion_scale,
            "outputSamplingRate": int(output_rate),
        }
        for key, val in overrides.items():
            if val is not None:
                q[key] = val
        return self.synthesize_query(q, style_id)
