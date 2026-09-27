"""Unit tests for the interim-commit / final-reconcile logic in
VoiceChangerPipeline — exercised on a bare instance (no VAD/audio needed).
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from kowairo.audio.pipeline import (
    _INTERIM_TAIL_CHARS,
    ClauseJob,
    PipelineStats,
    Utterance,
    VoiceChangerPipeline,
    _InterimJob,
)
from kowairo.settings import Settings


def _bare_pipe() -> tuple[VoiceChangerPipeline, list[str]]:
    """A pipeline object with only the fields the commit path needs."""
    logs: list[str] = []
    p = VoiceChangerPipeline.__new__(VoiceChangerPipeline)
    p.settings = Settings(auto_speed=False, auto_volume=False,
                          auto_pitch=False)
    p._live = {}
    p._commit_lock = threading.Lock()
    p._finalized_seq = 0
    p._clause_q = queue.Queue()
    p.on_log = logs.append
    p.stats = PipelineStats()
    return p, logs


def _job(seq: int, seconds: float = 1.0) -> _InterimJob:
    return _InterimJob(seq, np.zeros(int(16000 * seconds), np.float32),
                       time.monotonic())


def _drain_clauses(p: VoiceChangerPipeline) -> list[ClauseJob]:
    out = []
    while not p._clause_q.empty():
        out.append(p._clause_q.get_nowait())
    return out


class TestInterimCommit:
    def test_first_snapshot_commits_nothing(self):
        p, _ = _bare_pipe()
        p._commit_interim(_job(1), "あいうえおかきくけこさしすせそ")
        assert _drain_clauses(p) == []
        assert p._live[1]["prev_stable"].startswith("あい")

    def test_stable_prefix_committed_minus_tail(self):
        p, _ = _bare_pipe()
        t1 = "あいうえおかきくけこさしすせそ"          # 15 chars
        p._commit_interim(_job(1), t1)
        p._commit_interim(_job(1), t1 + "たちつてと")
        clauses = _drain_clauses(p)
        # stable = t1 minus the 8-char still-growing tail
        assert len(clauses) == 1
        assert clauses[0].text == t1[: len(t1) - _INTERIM_TAIL_CHARS]
        assert clauses[0].interim
        assert clauses[0].index == 0
        assert clauses[0].seq == 1

    def test_delta_capped_per_snapshot(self):
        p, _ = _bare_pipe()
        t1 = "あ" * 30
        p._commit_interim(_job(1), t1)
        p._commit_interim(_job(1), t1 + "い" * 10)
        clauses = _drain_clauses(p)
        # stable = 30-8=22 chars, but delta capped at 16
        assert clauses[0].text == "あ" * 16
        # next snapshot: stable extends to len(text)-8 = 32, so the next
        # 16 chars after committed=16 commit → stable[16:32] = あ×14 + い×2
        p._commit_interim(_job(1), t1 + "い" * 10)
        clauses = _drain_clauses(p)
        assert clauses[0].text == "あ" * 14 + "い" * 2

    def test_revision_diverges(self):
        p, logs = _bare_pipe()
        t1 = "あいうえおかきくけこさしすせそ"
        p._commit_interim(_job(1), t1)
        p._commit_interim(_job(1), t1 + "たちつてと")
        _drain_clauses(p)
        # ASR rewrites committed text → stop committing for this utterance
        p._commit_interim(_job(1), "全然違うテキストになってしまった場合")
        assert p._live[1]["diverged"]
        assert any("中断" in m for m in logs)
        # subsequent snapshots stay inert
        p._commit_interim(_job(1), "全然違うテキストのその後")
        assert _drain_clauses(p) == []

    def test_old_seq_ignored(self):
        p, _ = _bare_pipe()
        p._finalized_seq = 5
        p._commit_interim(_job(1), "過去の発話のスナップショットです")
        assert _drain_clauses(p) == []
        assert 1 not in p._live


class TestQueueText:
    def test_final_delta_only(self):
        p, _ = _bare_pipe()
        t1 = "あいうえおかきくけこさしすせそ"
        p._commit_interim(_job(1), t1)
        p._commit_interim(_job(1), t1 + "たちつてと")
        _drain_clauses(p)
        final = t1 + "たちつてと。追加です。"
        p._queue_text(final, Utterance(np.zeros(16000, np.float32),
                                       time.monotonic(), seq=1))
        clauses = _drain_clauses(p)
        joined = "".join(c.text for c in clauses)
        # only the part after the interim commits was synthesized
        committed = t1[: len(t1) - _INTERIM_TAIL_CHARS]
        assert final.startswith(committed)
        assert joined == final[len(committed):]
        assert all(c.index > 0 for c in clauses)  # indices continue

    def test_final_mismatch_resynthesizes_all(self):
        p, logs = _bare_pipe()
        t1 = "あいうえおかきくけこさしすせそ"
        p._commit_interim(_job(1), t1)
        p._commit_interim(_job(1), t1 + "たちつてと")
        _drain_clauses(p)
        p._queue_text("完全に違う最終認識結果です。",
                      Utterance(np.zeros(16000, np.float32),
                                time.monotonic(), seq=1))
        clauses = _drain_clauses(p)
        assert "".join(c.text for c in clauses) == "完全に違う最終認識結果です。"
        assert any("不一致" in m for m in logs)
        assert p._finalized_seq == 1
        assert 1 not in p._live

    def test_no_interim_full_text(self):
        p, _ = _bare_pipe()
        text = "普通の発話です。二文目です。"
        p._queue_text(text, Utterance(np.zeros(16000, np.float32),
                                      time.monotonic(), seq=7))
        clauses = _drain_clauses(p)
        assert [c.text for c in clauses] == ["普通の発話です。", "二文目です。"]
        assert clauses[0].index == 0 and not clauses[0].interim
        assert p._finalized_seq == 7
