"""CLI mode: convert a WAV file or text without a GUI — used for testing."""

from __future__ import annotations

import time

from .asr.sherpa import SherpaReazonASR, ensure_silero_vad
from .audio.pipeline import VoiceChangerPipeline
from .engine.manager import EngineManager
from .models import hub
from .models.aivm import find_aivmx
from .settings import load
from .util.audio import pcm_to_wav


def run_cli(args) -> int:
    # frozen windowed builds may get a legacy-codepage (or missing) console
    import sys

    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    def log(m: str) -> None:
        print(f"[cli] {m}", flush=True)

    mgr = EngineManager(version=args.engine_version, use_gpu=args.use_gpu,
                        log=log)
    print("[cli] engine setup...", flush=True)
    mgr.ensure_installed(progress=lambda r, t: print(
        f"\r[cli] engine dl {r/1e6:.0f}/{t/1e6:.0f}MB", end="", flush=True))
    client = mgr.start()
    print(f"\n[cli] engine up on :{mgr.port}", flush=True)

    try:
        if args.model:
            for f in find_aivmx(args.model):
                log(f"installing {f.name} …")
                client.install_model_file(f)
        if args.hub_uuid:
            p = hub.download_model(
                args.hub_uuid,
                progress=lambda r, t: print(
                    f"\r[cli] model dl {r/1e6:.0f}/{t/1e6:.0f}MB",
                    end="", flush=True))
            client.install_model_file(p)
            print()

        speakers = client.speakers()
        if not speakers:
            print("[cli] no voice model installed — use --model or --hub-uuid",
                  flush=True)
            return 2
        style_id = args.style_id
        if style_id is None:
            style_id = speakers[0].styles[0].id
        log(f"style_id={style_id} speakers="
            f"{[(s.name, [st.name for st in s.styles]) for s in speakers]}")

        if args.text:
            t0 = time.monotonic()
            wav = client.synthesize(args.text, style_id, output_rate=48000)
            dt = (time.monotonic() - t0) * 1000
            out = args.output or "out.wav"
            out.write_bytes(wav)
            print(f"[cli] synthesized {len(wav)} bytes in {dt:.0f}ms -> {out}",
                  flush=True)
            return 0

        if args.input:
            asr = SherpaReazonASR()
            print("[cli] asr setup...", flush=True)
            asr.ensure_model(progress=lambda r, t: print(
                f"\r[cli] asr dl {r/1e6:.0f}/{t/1e6:.0f}MB", end="",
                flush=True))
            ensure_silero_vad()
            asr.load()
            print("\n[cli] asr ready", flush=True)

            s = load()
            s.style_id = style_id
            pipe = VoiceChangerPipeline(
                s, client, asr, ensure_silero_vad(), on_log=log)
            pcm, texts, _stats = pipe.run_file(args.input, progress=log)
            out = args.output or "out.wav"
            out.write_bytes(pcm_to_wav(pcm, 48000))
            print(f"[cli] done: {len(texts)} utterances, "
                  f"{pcm.size/48000:.1f}s audio -> {out}", flush=True)
            return 0

        print("[cli] nothing to do (pass --text, --input, --model or --hub-uuid)",
              flush=True)
        return 0
    finally:
        mgr.stop()
