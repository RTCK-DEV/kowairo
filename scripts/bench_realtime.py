"""Headless realtime benchmark — drives the live pipeline at real-time pace
and reports responsiveness against competitor baselines.

Metrics:
  respond  = speech start → first converted audio out (what Voidol/w-okada
             advertise: continuous output while the user still speaks)
  e2e      = utterance endpoint → first converted audio out
  interim  = clauses committed from interim ASR snapshots

Usage:
    python scripts/bench_realtime.py [--input voice.wav] [--no-interim]
        [--style-id N] [--repeat N] [--json out.json]

Without --input a multi-clause test sentence is synthesized by the engine
itself, so the benchmark is fully self-contained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kowairo.asr.sherpa import SherpaReazonASR, ensure_silero_vad
from kowairo.audio.pipeline import ASR_RATE, VoiceChangerPipeline
from kowairo.engine.manager import EngineManager
from kowairo.settings import load
from kowairo.util.audio import resample, wav_to_pcm

CHUNK = 512  # 32 ms @16 kHz — same frame size the mic callback delivers
TTS_RATE_F = 48000.0


def synthesize_input(client, style_id: int) -> np.ndarray:
    """Build a multi-clause test utterance (~8 s) via the TTS engine."""
    text = ("こんにちは、これはリアルタイム変換のベンチマークです。"
            "競合製品は話している最中から音声を出力するので、"
            "この製品も同じように振る舞うかを測定します。")
    wav = client.synthesize(text, style_id, speed_scale=1.0,
                            output_rate=24000)
    pcm, rate = wav_to_pcm(wav)
    return resample(pcm, rate, ASR_RATE, "VHQ")


def feed_realtime(pipe: VoiceChangerPipeline, pcm: np.ndarray) -> None:
    """Push PCM in 32 ms chunks paced to wall-clock, like a live mic."""
    t0 = time.monotonic()
    for i in range(0, pcm.size, CHUNK):
        pipe.feed_pcm(pcm[i:i + CHUNK])
        target = t0 + (i + CHUNK) / ASR_RATE
        time.sleep(max(0.0, target - time.monotonic()))
    pipe.flush()


def run_once(args, client, asr, vad_model, pcm_in: np.ndarray) -> dict:
    s = load()
    s.style_id = args.style_id
    s.interim_asr = not args.no_interim
    s.auto_speed = s.auto_volume = s.auto_pitch = False  # isolate timing vars
    s.noise_gate_db = -80.0

    first_play: dict[int, float] = {}
    utt_end: dict[int, tuple[float, float]] = {}  # seq -> (t_end, dur)
    plays: list[dict] = []
    t_feed0 = time.monotonic()

    def on_play(job, pcm) -> None:
        if job is None:
            return
        plays.append({"seq": job.seq, "index": job.index,
                      "interim": job.interim,
                      "t": round(time.monotonic() - t_feed0, 2),
                      "dur_s": round(pcm.size / TTS_RATE_F, 2)})
        if job.seq not in first_play:
            first_play[job.seq] = time.monotonic()

    def on_utterance(utt) -> None:
        utt_end[utt.seq] = (utt.t_end, utt.pcm.size / ASR_RATE)

    pipe = VoiceChangerPipeline(
        s, client, asr, vad_model,
        on_log=lambda m: print(f"  [pipe] {m}", flush=True),
        on_utterance=on_utterance, on_play=on_play)
    t0 = time.monotonic()
    pipe.start(with_io=False)
    try:
        feed_realtime(pipe, pcm_in)
        # wait for the queue to finish synthesizing/playing: idle 3 s or cap
        deadline = time.monotonic() + max(15.0, pcm_in.size / ASR_RATE * 2)
        idle_since = None
        while time.monotonic() < deadline:
            pending = (pipe._clause_q.qsize() + pipe._play_q.qsize()
                       + pipe._playbuf.pending + pipe._utter_q.qsize()
                       + pipe._interim_q.qsize())
            if pending == 0:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > 3.0:
                    break
            else:
                idle_since = None
            time.sleep(0.1)
    finally:
        pipe.stop()
    wall = time.monotonic() - t0

    rows = []
    for seq, (t_end, dur) in sorted(utt_end.items()):
        fp = first_play.get(seq)
        rows.append({
            "seq": seq,
            "speech_s": round(dur, 2),
            "respond_ms": round((fp - (t_end - dur)) * 1000) if fp else None,
            "e2e_ms": round((fp - t_end) * 1000) if fp else None,
        })
    st = pipe.stats
    return {"wall_s": round(wall, 1), "utterances": rows,
            "plays": plays,
            "clauses": st.clauses,
            "avg_respond_ms": round(st.avg_respond_ms),
            "avg_e2e_ms": round(st.avg_latency_ms),
            "asr_ms": round(st.asr_ms), "tts_ms": round(st.tts_ms),
            "interim_asr_ms": round(st.interim_ms),
            "dropped_ms": round(st.dropped_ms),
            "interim": s.interim_asr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--no-interim", action="store_true",
                    help="disable mid-utterance synthesis (baseline mode)")
    ap.add_argument("--style-id", type=int)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    print("[bench] engine setup...", flush=True)
    mgr = EngineManager(log=lambda m: print(f"  [eng] {m}", flush=True))
    mgr.ensure_installed()
    client = mgr.start()
    print(f"[bench] engine up on :{mgr.port}", flush=True)

    try:
        speakers = client.speakers()
        if not speakers:
            print("[bench] no voice model installed", flush=True)
            return 2
        args.style_id = args.style_id or speakers[0].styles[0].id
        print(f"[bench] style_id={args.style_id}", flush=True)

        print("[bench] tts warmup...", flush=True)
        t0 = time.monotonic()
        client.synthesize("あ", args.style_id, output_rate=48000)
        print(f"[bench]   {(time.monotonic() - t0) * 1000:.0f}ms", flush=True)

        print("[bench] asr setup...", flush=True)
        asr = SherpaReazonASR()
        asr.ensure_model()
        ensure_silero_vad()
        asr.load()
        asr.warmup()

        if args.input:
            pcm_in, rate = wav_to_pcm(args.input.read_bytes())
            pcm_in = resample(pcm_in, rate, ASR_RATE, "VHQ")
        else:
            print("[bench] synthesizing input speech...", flush=True)
            pcm_in = synthesize_input(client, args.style_id)
        print(f"[bench] input {pcm_in.size / ASR_RATE:.1f}s", flush=True)

        results = []
        for n in range(args.repeat):
            print(f"[bench] --- run {n + 1}/{args.repeat} "
                  f"(interim={'off' if args.no_interim else 'on'}) ---",
                  flush=True)
            results.append(run_once(args, client, asr, ensure_silero_vad(),
                                    pcm_in))
            for r in results[-1]["utterances"]:
                print(f"  utt#{r['seq']}: speech={r['speech_s']}s "
                      f"respond={r['respond_ms']}ms e2e={r['e2e_ms']}ms",
                      flush=True)
            s = results[-1]
            print(f"  summary: respond_avg={s['avg_respond_ms']}ms "
                  f"e2e_avg={s['avg_e2e_ms']}ms clauses={s['clauses']} "
                  f"asr={s['asr_ms']}ms tts={s['tts_ms']}ms "
                  f"interim_asr={s['interim_asr_ms']}ms "
                  f"dropped={s['dropped_ms']}ms wall={s['wall_s']}s",
                  flush=True)

        if args.json:
            args.json.write_text(json.dumps(results, ensure_ascii=False,
                                            indent=2))
            print(f"[bench] wrote {args.json}", flush=True)
        return 0
    finally:
        mgr.stop()


if __name__ == "__main__":
    raise SystemExit(main())
