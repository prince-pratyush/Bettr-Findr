"""M4 — Store.

Wraps sqlite-vec (vectors) + SQLite FTS5 (keyword) in a single persisted DB.

Implemented in Step 5.
"""

from __future__ import annotations

from .models import Chunk, QueryFilters, SearchResult


def upsert_chunks(chunks: list[Chunk]) -> None:
    """Insert or replace chunks (and their vectors + FTS rows)."""
    raise NotImplementedError("store.upsert_chunks is implemented in Step 5")


def delete_file(path: str) -> None:
    """Remove all chunks belonging to `path`."""
    raise NotImplementedError("store.delete_file is implemented in Step 5")


def search_semantic(
    query_vec: list[float], k: int, filters: QueryFilters | None = None
) -> list[SearchResult]:
    """Vector similarity search."""
    raise NotImplementedError("store.search_semantic is implemented in Step 5")


def search_keyword(
    query: str, k: int, filters: QueryFilters | None = None
) -> list[SearchResult]:
    """FTS5 keyword search."""
    raise NotImplementedError("store.search_keyword is implemented in Step 5")
