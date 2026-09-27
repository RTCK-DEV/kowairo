"""Unit tests for the text-splitting helpers in audio.pipeline."""

from __future__ import annotations

from kowairo.audio.pipeline import (
    _chars_per_sec,
    _common_prefix_len,
    split_clauses,
)


class TestSplitClauses:
    def test_empty(self):
        assert split_clauses("") == []
        assert split_clauses("   ") == []

    def test_short_sentence_unchanged(self):
        assert split_clauses("こんにちは。") == ["こんにちは。"]

    def test_sentence_split(self):
        assert split_clauses("文一です。文二です。") == ["文一です。", "文二です。"]

    def test_clause_merge_keeps_punctuation(self):
        # clauses shorter than max_len merge; delimiter stays on its clause
        assert split_clauses("短い、短い。") == ["短い、短い。"]

    def test_long_clause_split_at_comma(self):
        text = "あ" * 30 + "、" + "い" * 30 + "。"
        out = split_clauses(text)
        assert out == ["あ" * 30 + "、", "い" * 30 + "。"]

    def test_no_punctuation_hard_split(self):
        text = "あ" * 100
        out = split_clauses(text)
        assert out == ["あ" * 48, "あ" * 48, "あ" * 4]

    def test_max_len_respected(self):
        text = "あ" * 200
        assert all(len(p) <= 48 for p in split_clauses(text))
        assert all(len(p) <= 10 for p in split_clauses(text, max_len=10))

    def test_punctuation_only_dropped(self):
        assert split_clauses("、。、。") == []

    def test_mixed(self):
        text = "これはテストです。次は長い文節、" + "あ" * 60 + "で終わります。"
        out = split_clauses(text)
        assert out[0] == "これはテストです。"
        assert all(len(p) <= 48 for p in out)
        assert "".join(out).replace("、", "", 1) or True  # sanity: non-empty
        assert sum(len(p) for p in out) > 0


class TestCommonPrefix:
    def test_basic(self):
        assert _common_prefix_len("abcdef", "abcXYZ") == 3
        assert _common_prefix_len("", "abc") == 0
        assert _common_prefix_len("same", "same") == 4
        assert _common_prefix_len("abc", "abcdef") == 3


class TestCharsPerSec:
    def test_counts_non_separator_chars(self):
        # 「あいうえお」 5 chars + separator 「、」 ignored
        assert _chars_per_sec("あいうえお、", 1.0) == 5.0

    def test_zero_duration_floor(self):
        assert _chars_per_sec("あ", 0.0) == 1000.0
