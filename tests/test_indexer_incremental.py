"""Tests for M5 / Step 9 — incremental re-indexing.

The indexer must diff against the manifest: skip unchanged files (no re-embed),
re-index changed files, pick up new files, and remove files that were deleted
from disk — without touching files outside the indexed roots.

Embeddings are mocked (and counted) so we can prove unchanged files are not
re-embedded.
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


class CountingEmbed:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return [[float(len(t) % 7)] * config.EMBED_DIM for t in texts]


@pytest.fixture
def counting_embed(monkeypatch):
    spy = CountingEmbed()
    monkeypatch.setattr(embedder, "embed", spy)
    return spy


def test_unchanged_file_is_not_reembedded(store_db, counting_embed, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("stable contents here", encoding="utf-8")

    indexer.index_folders([str(tmp_path)])
    after_first = counting_embed.calls
    assert after_first >= 1

    indexer.index_folders([str(tmp_path)])  # nothing changed
    assert counting_embed.calls == after_first  # no new embedding work


def test_changed_file_is_reindexed(store_db, counting_embed, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("original apple text", encoding="utf-8")
    indexer.index_folders([str(tmp_path)])

    f.write_text("updated banana text", encoding="utf-8")
    indexer.index_folders([str(tmp_path)])

    assert store.search_keyword("banana", k=5)
    assert store.search_keyword("apple", k=5) == []


def test_new_file_picked_up_on_reindex(store_db, counting_embed, tmp_path):
    (tmp_path / "a.txt").write_text("first document", encoding="utf-8")
    indexer.index_folders([str(tmp_path)])

    (tmp_path / "b.txt").write_text("second document", encoding="utf-8")
    indexer.index_folders([str(tmp_path)])

    assert store.search_keyword("first", k=5)
    assert store.search_keyword("second", k=5)


def test_deleted_file_removed_from_index(store_db, counting_embed, tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("keep me around", encoding="utf-8")
    b.write_text("delete me later", encoding="utf-8")
    indexer.index_folders([str(tmp_path)])

    b.unlink()
    indexer.index_folders([str(tmp_path)])

    assert store.search_keyword("keep", k=5)
    assert store.search_keyword("delete", k=5) == []
    assert store.stats().file_count == 1


def test_status_last_indexed_set_after_index(store_db, counting_embed, tmp_path):
    (tmp_path / "a.txt").write_text("some content", encoding="utf-8")

    assert store.stats().last_indexed_at is None
    indexer.index_folders([str(tmp_path)])
    assert store.stats().last_indexed_at is not None


def test_out_of_scope_files_are_not_deleted(store_db, counting_embed, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "a.txt").write_text("alpha document", encoding="utf-8")
    (dir_b / "b.txt").write_text("bravo document", encoding="utf-8")

    indexer.index_folders([str(dir_a)])
    indexer.index_folders([str(dir_b)])

    # Re-indexing only dir_a must not evict dir_b's file.
    indexer.index_folders([str(dir_a)])

    assert store.search_keyword("bravo", k=5)
    assert store.stats().file_count == 2
