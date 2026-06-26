"""M4 — Store.

Wraps sqlite-vec (vectors) + SQLite FTS5 (keyword) in a single persisted DB.

Schema (all in one SQLite file at ``config.DB_PATH``):

    chunks       — one row per chunk, integer ``id`` is the shared rowid
    vec_chunks   — vec0 virtual table: ``embedding float[EMBED_DIM]`` by rowid
    fts_chunks   — FTS5 contentless index over ``chunk_text`` + ``file_name``
    manifest     — one row per indexed file (drives incremental re-indexing)

Vector distance is L2 (sqlite-vec default); keyword rank is FTS5 bm25. Both are
mapped to a "higher is better" score so the ranker can blend them.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Optional

import sqlite_vec

from . import config
from .models import Chunk, IndexStatus, ManifestEntry, QueryFilters, SearchResult

# Over-fetch factor for filtered search: pull more candidates than `k` so that
# post-filtering still leaves enough results.
_CANDIDATE_FLOOR = 50
_CANDIDATE_FACTOR = 5

_conn: Optional[sqlite3.Connection] = None


# --- Connection lifecycle ----------------------------------------------------


def _get_conn() -> sqlite3.Connection:
    """Return the cached connection, opening + initializing it on first use."""
    global _conn
    if _conn is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the FastAPI engine serves sync routes from a
        # threadpool, so the single cached connection is touched from worker
        # threads. SQLite still serializes its own writes; this is a local,
        # single-user engine with no concurrent writers.
        conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _init_schema(conn)
        _conn = conn
    return _conn


def close() -> None:
    """Close and forget the cached connection (test hook / clean shutdown)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id    TEXT UNIQUE NOT NULL,
            file_path   TEXT NOT NULL,
            file_name   TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            file_hash   TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{config.EMBED_DIM}]
        );

        -- Regular (content-owning) FTS5 table so rows can be DELETEd directly
        -- when a chunk is re-indexed or its file is removed.
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            chunk_text, file_name
        );

        CREATE TABLE IF NOT EXISTS manifest (
            file_path   TEXT PRIMARY KEY,
            file_hash   TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            indexed_at  TEXT NOT NULL
        );
        """
    )
    conn.commit()


# --- Writes ------------------------------------------------------------------


def upsert_chunks(chunks: list[Chunk]) -> None:
    """Insert or replace chunks (and their vectors + FTS rows) by chunk_id."""
    conn = _get_conn()
    for chunk in chunks:
        if chunk.embedding is None:
            raise ValueError(f"chunk {chunk.chunk_id} has no embedding to store")

        existing = conn.execute(
            "SELECT id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()
        if existing is not None:
            _delete_rowid(conn, existing[0])

        cur = conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, file_path, file_name, file_type,
                modified_at, file_hash, chunk_index, chunk_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.file_path,
                chunk.file_name,
                chunk.file_type,
                chunk.modified_at,
                chunk.file_hash,
                chunk.chunk_index,
                chunk.chunk_text,
            ),
        )
        rowid = cur.lastrowid
        conn.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(chunk.embedding)),
        )
        conn.execute(
            "INSERT INTO fts_chunks (rowid, chunk_text, file_name) VALUES (?, ?, ?)",
            (rowid, chunk.chunk_text, chunk.file_name),
        )
    conn.commit()


def delete_file(path: str) -> None:
    """Remove all chunks belonging to `path`."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id FROM chunks WHERE file_path = ?", (path,)
    ).fetchall()
    for (rowid,) in rows:
        _delete_rowid(conn, rowid)
    conn.commit()


def _delete_rowid(conn: sqlite3.Connection, rowid: int) -> None:
    conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
    conn.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rowid,))
    conn.execute("DELETE FROM chunks WHERE id = ?", (rowid,))


def stats() -> IndexStatus:
    """Return a snapshot of index size and last-indexed time for `status`."""
    conn = _get_conn()
    file_count = conn.execute(
        "SELECT COUNT(DISTINCT file_path) FROM chunks"
    ).fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    last_indexed = conn.execute("SELECT MAX(indexed_at) FROM manifest").fetchone()[0]
    return IndexStatus(
        file_count=file_count,
        chunk_count=chunk_count,
        last_indexed_at=last_indexed,
    )


# --- Manifest (drives incremental re-indexing) -------------------------------


def get_manifest() -> dict[str, ManifestEntry]:
    """Return all manifest entries keyed by file path."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT file_path, file_hash, modified_at, chunk_count, indexed_at "
        "FROM manifest"
    ).fetchall()
    return {
        r[0]: ManifestEntry(
            file_path=r[0],
            file_hash=r[1],
            modified_at=r[2],
            chunk_count=r[3],
            indexed_at=r[4],
        )
        for r in rows
    }


def upsert_manifest(entry: ManifestEntry) -> None:
    """Insert or update a single manifest entry."""
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO manifest (file_path, file_hash, modified_at, chunk_count, indexed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            file_hash   = excluded.file_hash,
            modified_at = excluded.modified_at,
            chunk_count = excluded.chunk_count,
            indexed_at  = excluded.indexed_at
        """,
        (
            entry.file_path,
            entry.file_hash,
            entry.modified_at,
            entry.chunk_count,
            entry.indexed_at,
        ),
    )
    conn.commit()


def delete_manifest(path: str) -> None:
    """Remove a manifest entry (its chunks are removed via ``delete_file``)."""
    conn = _get_conn()
    conn.execute("DELETE FROM manifest WHERE file_path = ?", (path,))
    conn.commit()


# --- Reads -------------------------------------------------------------------


def search_semantic(
    query_vec: list[float], k: int, filters: QueryFilters | None = None
) -> list[SearchResult]:
    """Vector similarity search (L2 distance -> higher-is-better score)."""
    conn = _get_conn()
    limit = max(k * _CANDIDATE_FACTOR, _CANDIDATE_FLOOR)
    candidates = conn.execute(
        """
        SELECT rowid, distance FROM vec_chunks
        WHERE embedding MATCH ? ORDER BY distance LIMIT ?
        """,
        (sqlite_vec.serialize_float32(query_vec), limit),
    ).fetchall()

    results: list[SearchResult] = []
    for rowid, distance in candidates:
        row = _fetch_chunk(conn, rowid)
        if row is None or not _passes(row, filters):
            continue
        results.append(_to_result(row, score=1.0 / (1.0 + distance)))
        if len(results) >= k:
            break
    return results


def search_keyword(
    query: str, k: int, filters: QueryFilters | None = None
) -> list[SearchResult]:
    """FTS5 keyword search (bm25 rank -> higher-is-better score)."""
    match = _to_fts_query(query)
    if match is None:
        return []

    conn = _get_conn()
    limit = max(k * _CANDIDATE_FACTOR, _CANDIDATE_FLOOR)
    candidates = conn.execute(
        """
        SELECT rowid, rank FROM fts_chunks
        WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?
        """,
        (match, limit),
    ).fetchall()

    results: list[SearchResult] = []
    for rowid, rank in candidates:
        row = _fetch_chunk(conn, rowid)
        if row is None or not _passes(row, filters):
            continue
        # bm25 rank is negative; more negative == more relevant.
        results.append(_to_result(row, score=-float(rank)))
        if len(results) >= k:
            break
    return results


# --- Helpers -----------------------------------------------------------------

_CHUNK_COLS = "file_path, file_name, file_type, modified_at, chunk_text"


def _fetch_chunk(conn: sqlite3.Connection, rowid: int) -> Optional[tuple]:
    return conn.execute(
        f"SELECT {_CHUNK_COLS} FROM chunks WHERE id = ?", (rowid,)
    ).fetchone()


def _passes(row: tuple, filters: QueryFilters | None) -> bool:
    if filters is None:
        return True
    _, _, file_type, modified_at, _ = row
    if filters.file_types is not None and file_type not in filters.file_types:
        return False
    if filters.modified_since is not None and modified_at < filters.modified_since:
        return False
    if filters.modified_until is not None and modified_at > filters.modified_until:
        return False
    return True


def _to_result(row: tuple, score: float) -> SearchResult:
    file_path, file_name, file_type, modified_at, chunk_text = row
    return SearchResult(
        file_path=file_path,
        score=score,
        snippet=chunk_text,
        modified_at=modified_at,
        file_name=file_name,
        file_type=file_type,
    )


def _to_fts_query(query: str) -> Optional[str]:
    """Turn a natural-language query into a safe FTS5 MATCH expression.

    Each alphanumeric term is quoted and OR-ed together. Returns None when the
    query has no usable terms (so callers can short-circuit to no results).
    """
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return None
    quoted = ['"' + t + '"' for t in terms]
    return " OR ".join(quoted)
