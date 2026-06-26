"""Tests for M6 — the ranker (findr/ranker.py), the differentiator.

The ranker embeds the query, runs semantic + keyword search via the store,
normalizes and blends the two score lists (weights from config), boosts
filename matches, dedupes to one result per file, and returns the top k.

The store is real (temp DB); only the query embedding is mocked.
"""

from __future__ import annotations

import pytest

from findr import config, embedder, ranker, store
from findr.models import Chunk, QueryFilters


def _one_hot(i: int, dim: int = config.EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _chunk(cid, *, text, path, name, ftype="txt", idx=0, emb=None, modified="2026-01-01T00:00:00"):
    return Chunk(
        chunk_id=cid,
        file_path=path,
        file_name=name,
        file_type=ftype,
        modified_at=modified,
        file_hash="h",
        chunk_index=idx,
        chunk_text=text,
        embedding=emb if emb is not None else _one_hot(0),
    )


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "index.db")
    store.close()
    yield
    store.close()


def _mock_query_vec(monkeypatch, vec):
    monkeypatch.setattr(embedder, "embed", lambda texts: [vec])


def test_empty_query_returns_empty(store_db):
    assert ranker.query("") == []
    assert ranker.query("   ") == []


def test_semantic_match_ranks_first(store_db, monkeypatch):
    store.upsert_chunks(
        [
            _chunk("a", text="alpha content", path="/d/a.txt", name="a.txt", emb=_one_hot(0)),
            _chunk("b", text="beta content", path="/d/b.txt", name="b.txt", emb=_one_hot(5)),
        ]
    )
    _mock_query_vec(monkeypatch, _one_hot(0))  # nearest to a.txt

    results = ranker.query("unrelated words", k=10)

    assert results[0].file_path == "/d/a.txt"


def test_keyword_only_match_is_surfaced(store_db, monkeypatch):
    store.upsert_chunks(
        [
            _chunk("a", text="common words", path="/d/a.txt", name="a.txt", emb=_one_hot(0)),
            _chunk("b", text="zebra giraffe", path="/d/b.txt", name="b.txt", emb=_one_hot(700)),
        ]
    )
    _mock_query_vec(monkeypatch, _one_hot(0))  # semantically far from b.txt

    paths = {r.file_path for r in ranker.query("zebra", k=10)}

    assert "/d/b.txt" in paths  # keyword pulled it in despite semantic distance


def test_filename_match_boost(store_db, monkeypatch):
    store.upsert_chunks(
        [
            _chunk("a", text="report", path="/d/budget.txt", name="budget.txt", emb=_one_hot(0)),
            _chunk("b", text="report", path="/d/taxes.txt", name="taxes.txt", emb=_one_hot(0)),
        ]
    )
    _mock_query_vec(monkeypatch, _one_hot(0))

    results = ranker.query("budget", k=10)

    assert results[0].file_name == "budget.txt"


def test_dedupes_to_one_result_per_file(store_db, monkeypatch):
    store.upsert_chunks(
        [
            _chunk("c0", text="needle one", path="/d/c.txt", name="c.txt", idx=0, emb=_one_hot(0)),
            _chunk("c1", text="needle two", path="/d/c.txt", name="c.txt", idx=1, emb=_one_hot(1)),
        ]
    )
    _mock_query_vec(monkeypatch, _one_hot(0))

    results = ranker.query("needle", k=10)

    assert [r.file_path for r in results].count("/d/c.txt") == 1


def test_respects_file_type_filter(store_db, monkeypatch):
    store.upsert_chunks(
        [
            _chunk("p", text="quarterly numbers", path="/d/x.pdf", name="x.pdf", ftype="pdf", emb=_one_hot(0)),
            _chunk("t", text="quarterly numbers", path="/d/x.txt", name="x.txt", ftype="txt", emb=_one_hot(1)),
        ]
    )
    _mock_query_vec(monkeypatch, _one_hot(0))

    results = ranker.query("quarterly", k=10, filters=QueryFilters(file_types=["pdf"]))

    assert {r.file_type for r in results} == {"pdf"}


def test_top_k_limit(store_db, monkeypatch):
    store.upsert_chunks(
        [
            _chunk(f"c{i}", text=f"doc number {i}", path=f"/d/{i}.txt", name=f"{i}.txt", emb=_one_hot(i))
            for i in range(4)
        ]
    )
    _mock_query_vec(monkeypatch, _one_hot(0))

    results = ranker.query("doc", k=2)

    assert len(results) == 2
