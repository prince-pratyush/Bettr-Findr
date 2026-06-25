"""Tests for M2 — the chunker (findr/chunker.py).

The chunker splits extracted text into overlapping windows sized for
nomic-embed-text's ~512-token window. Tokens are approximated by whitespace-
delimited words, which is conservative (words <= tokens).

User journey:
- As the indexer, I want long documents split into overlapping chunks so each
  piece fits the embedding model and context isn't lost at chunk boundaries.
"""

from __future__ import annotations

import pytest

from findr import chunker


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_returns_no_chunks():
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\t  ") == []


def test_short_text_is_single_chunk():
    out = chunker.chunk("alpha beta gamma", target_tokens=400, overlap=50)
    assert out == ["alpha beta gamma"]


def test_long_text_splits_into_multiple_chunks():
    out = chunker.chunk(_words(1000), target_tokens=400, overlap=50)
    assert len(out) == 3


def test_each_chunk_within_target_size():
    out = chunker.chunk(_words(1000), target_tokens=400, overlap=50)
    for c in out:
        assert len(c.split()) <= 400


def test_consecutive_chunks_overlap_by_overlap_words():
    out = chunker.chunk(_words(1000), target_tokens=400, overlap=50)
    first_tail = out[0].split()[-50:]
    second_head = out[1].split()[:50]
    assert first_tail == second_head


def test_zero_overlap_partitions_without_repeats():
    out = chunker.chunk(_words(900), target_tokens=300, overlap=0)
    joined = " ".join(out).split()
    assert joined == _words(900).split()


def test_all_words_are_covered_in_order():
    out = chunker.chunk(_words(1000), target_tokens=400, overlap=50)
    # Every original word index must appear somewhere, order preserved.
    seen = set()
    for c in out:
        for w in c.split():
            seen.add(w)
    assert seen == set(_words(1000).split())


def test_overlap_not_less_than_target_raises():
    with pytest.raises(ValueError):
        chunker.chunk(_words(100), target_tokens=100, overlap=100)


def test_uses_config_defaults_when_unspecified():
    from findr import config

    out = chunker.chunk(_words(2000))
    step = config.CHUNK_TARGET_TOKENS - config.CHUNK_OVERLAP_TOKENS
    assert len(out) > 1
    assert len(out[0].split()) == config.CHUNK_TARGET_TOKENS
    # second chunk starts `step` words in
    assert out[1].split()[0] == f"w{step}"
