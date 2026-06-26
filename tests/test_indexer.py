"""Tests for M5 — the indexer (findr/indexer.py), first pass.

The indexer walks folders, then for each supported file runs
parse -> chunk -> embed -> store.upsert_chunks. Embeddings are mocked so these
tests run without Ollama, and the store is pointed at a temp DB.

User journey:
- As a user, when I run `findr index <folder>` every supported file under it
  becomes searchable, unsupported/empty files are skipped, and I see progress.
"""

from __future__ import annotations

import pytest

from findr import config, embedder, indexer, store


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "index.db")
    store.close()
    yield
    store.close()


@pytest.fixture
def fake_embed(monkeypatch):
    """One deterministic 768-dim vector per input text."""

    def _embed(texts):
        return [[float(len(t) % 7)] * config.EMBED_DIM for t in texts]

    monkeypatch.setattr(embedder, "embed", _embed)


def _make_tree(root):
    (root / "a.txt").write_text("the annual budget report", encoding="utf-8")
    (root / "b.md").write_text("# Notes\nmeeting agenda items", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (root / "image.heic").write_bytes(b"\x00\x01binary")  # unsupported
    (root / "empty.txt").write_text("   \n", encoding="utf-8")  # no chunks
    return root


def test_indexes_supported_files(store_db, fake_embed, tmp_path):
    _make_tree(tmp_path)

    indexer.index_folders([str(tmp_path)])

    assert store.search_keyword("budget", k=5)  # a.txt
    assert store.search_keyword("agenda", k=5)  # b.md
    assert store.search_keyword("hello", k=5)  # sub/c.py


def test_skips_unsupported_and_empty_files(store_db, fake_embed, tmp_path):
    _make_tree(tmp_path)

    indexer.index_folders([str(tmp_path)])

    files = {r.file_name for r in store.search_semantic([0.0] * config.EMBED_DIM, k=50)}
    assert "image.heic" not in files
    assert "empty.txt" not in files
    assert {"a.txt", "b.md", "c.py"} <= files


def test_reindexing_is_idempotent(store_db, fake_embed, tmp_path):
    f = tmp_path / "only.txt"
    f.write_text("singular unique content", encoding="utf-8")

    indexer.index_folders([str(tmp_path)])
    indexer.index_folders([str(tmp_path)])

    results = store.search_keyword("singular", k=10)
    assert len(results) == 1  # not duplicated


def test_accepts_a_single_file_path(store_db, fake_embed, tmp_path):
    f = tmp_path / "lonely.txt"
    f.write_text("standalone document text", encoding="utf-8")

    indexer.index_folders([str(f)])

    assert store.search_keyword("standalone", k=5)


def test_reports_progress(store_db, fake_embed, tmp_path):
    _make_tree(tmp_path)
    seen: list[tuple[float, str]] = []

    indexer.index_folders([str(tmp_path)], progress=lambda frac, msg: seen.append((frac, msg)))

    assert seen, "progress callback was never called"
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)  # monotonic non-decreasing
    assert fractions[-1] == pytest.approx(1.0)


def test_empty_folder_does_not_crash(store_db, fake_embed, tmp_path):
    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()

    indexer.index_folders([str(empty_dir)])  # should be a no-op

    assert store.search_keyword("anything", k=5) == []
