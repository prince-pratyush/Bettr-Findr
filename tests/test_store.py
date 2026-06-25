"""Tests for M4 — the store (findr/store.py).

The store wraps sqlite-vec (vectors) + SQLite FTS5 (keyword) behind a single
persisted DB. These tests point the DB at a temp file via the `store_db`
fixture so nothing touches the real index.

User journeys:
- As the indexer, I want to upsert chunks (idempotently per chunk_id) and
  delete all chunks of a file.
- As the ranker, I want vector and keyword searches that honor file-type and
  date filters.
"""

from __future__ import annotations

import pytest

from findr import config, store
from findr.models import Chunk, QueryFilters


def _one_hot(i: int, dim: int = config.EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _chunk(
    chunk_id: str,
    *,
    text: str = "some text",
    path: str = "/docs/a.txt",
    name: str = "a.txt",
    ftype: str = "txt",
    modified_at: str = "2026-01-01T00:00:00",
    chunk_index: int = 0,
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file_path=path,
        file_name=name,
        file_type=ftype,
        modified_at=modified_at,
        file_hash="hash",
        chunk_index=chunk_index,
        chunk_text=text,
        embedding=embedding if embedding is not None else _one_hot(0),
    )


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "index.db")
    store.close()  # drop any cached connection so we reconnect to the temp DB
    yield
    store.close()


def test_empty_db_returns_no_results(store_db):
    assert store.search_semantic(_one_hot(0), k=5) == []
    assert store.search_keyword("anything", k=5) == []


def test_upsert_then_semantic_search_ranks_nearest_first(store_db):
    store.upsert_chunks(
        [
            _chunk("c0", text="zero", embedding=_one_hot(0)),
            _chunk("c1", text="one", embedding=_one_hot(1)),
            _chunk("c2", text="two", embedding=_one_hot(2)),
        ]
    )

    results = store.search_semantic(_one_hot(1), k=3)

    assert [r.snippet for r in results][0] == "one"
    assert results[0].score >= results[-1].score  # sorted best-first


def test_keyword_search_finds_term(store_db):
    store.upsert_chunks(
        [
            _chunk("c0", text="the quarterly tax report for 2025", name="tax.pdf"),
            _chunk("c1", text="grocery shopping list", name="list.txt"),
        ]
    )

    results = store.search_keyword("tax report", k=5)

    assert len(results) == 1
    assert results[0].file_name == "tax.pdf"


def test_upsert_is_idempotent_per_chunk_id(store_db):
    store.upsert_chunks([_chunk("c0", text="old text apple")])
    store.upsert_chunks([_chunk("c0", text="new text banana")])

    assert store.search_keyword("banana", k=5)  # new content found
    assert store.search_keyword("apple", k=5) == []  # old content gone


def test_delete_file_removes_its_chunks(store_db):
    store.upsert_chunks(
        [
            _chunk("a0", text="alpha", path="/docs/a.txt", name="a.txt", embedding=_one_hot(0)),
            _chunk("a1", text="alpha two", path="/docs/a.txt", name="a.txt", chunk_index=1, embedding=_one_hot(1)),
            _chunk("b0", text="beta", path="/docs/b.txt", name="b.txt", embedding=_one_hot(2)),
        ]
    )

    store.delete_file("/docs/a.txt")

    sem = store.search_semantic(_one_hot(0), k=10)
    assert {r.file_path for r in sem} == {"/docs/b.txt"}
    assert store.search_keyword("alpha", k=10) == []


def test_semantic_filter_by_file_type(store_db):
    store.upsert_chunks(
        [
            _chunk("p", text="pdf doc", ftype="pdf", name="x.pdf", embedding=_one_hot(0)),
            _chunk("t", text="txt doc", ftype="txt", name="y.txt", embedding=_one_hot(1)),
        ]
    )

    results = store.search_semantic(
        _one_hot(1), k=10, filters=QueryFilters(file_types=["pdf"])
    )

    assert [r.file_type for r in results] == ["pdf"]


def test_semantic_filter_by_modified_since(store_db):
    store.upsert_chunks(
        [
            _chunk("old", text="older", modified_at="2025-01-01T00:00:00", embedding=_one_hot(0)),
            _chunk("new", text="newer", modified_at="2026-06-01T00:00:00", embedding=_one_hot(1)),
        ]
    )

    results = store.search_semantic(
        _one_hot(0), k=10, filters=QueryFilters(modified_since="2026-01-01T00:00:00")
    )

    assert [r.snippet for r in results] == ["newer"]


def test_keyword_filter_by_file_type(store_db):
    store.upsert_chunks(
        [
            _chunk("p", text="budget summary", ftype="pdf", name="b.pdf"),
            _chunk("t", text="budget summary", ftype="txt", name="b.txt"),
        ]
    )

    results = store.search_keyword(
        "budget", k=10, filters=QueryFilters(file_types=["txt"])
    )

    assert [r.file_type for r in results] == ["txt"]


def test_results_carry_metadata(store_db):
    store.upsert_chunks(
        [_chunk("c0", text="hello world", path="/d/h.md", name="h.md", ftype="md")]
    )

    r = store.search_keyword("hello", k=1)[0]

    assert r.file_path == "/d/h.md"
    assert r.file_name == "h.md"
    assert r.file_type == "md"
    assert r.snippet == "hello world"
    assert r.modified_at == "2026-01-01T00:00:00"


def test_blank_keyword_query_returns_empty(store_db):
    store.upsert_chunks([_chunk("c0", text="content")])
    assert store.search_keyword("   ", k=5) == []
