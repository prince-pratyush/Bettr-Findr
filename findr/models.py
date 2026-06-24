"""Core data model for Better Findr (spec section 5).

These dataclasses are the shared vocabulary between the parser, chunker,
embedder, store, indexer and ranker. All fields are synthetic/structural.
Datetimes are ISO 8601 strings, e.g. "2026-06-24T14:30:00".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """One indexed chunk of a file, stored in the vector store + SQLite."""

    chunk_id: str  # uuid, primary key
    file_path: str  # absolute path
    file_name: str  # for filename keyword matching
    file_type: str  # extension without dot, e.g. "pdf"
    modified_at: str  # ISO 8601 datetime
    file_hash: str  # content hash; skip re-embedding if unchanged
    chunk_index: int  # position within the file
    chunk_text: str  # extracted snippet (also shown in results)
    embedding: Optional[list[float]] = None  # 768-dim, from nomic-embed-text


@dataclass
class ManifestEntry:
    """One row per indexed file; drives incremental re-indexing."""

    file_path: str
    file_hash: str
    modified_at: str  # ISO 8601 datetime
    chunk_count: int
    indexed_at: str  # ISO 8601 datetime


@dataclass
class SearchResult:
    """A single ranked result returned to the CLI / API / UI."""

    file_path: str
    score: float
    snippet: str
    modified_at: str
    file_name: str = ""
    file_type: str = ""


@dataclass
class QueryFilters:
    """Optional filters applied to a query (spec: file type + date range)."""

    file_types: Optional[list[str]] = None  # e.g. ["pdf", "docx"]
    modified_since: Optional[str] = None  # ISO 8601 datetime, inclusive
    modified_until: Optional[str] = None  # ISO 8601 datetime, inclusive


@dataclass
class IndexStatus:
    """Snapshot of engine/index state for `findr status` and GET /status."""

    file_count: int = 0
    chunk_count: int = 0
    last_indexed_at: Optional[str] = None
    indexing: bool = False
    progress: float = 0.0  # 0.0–1.0 while an index job runs
    errors: list[str] = field(default_factory=list)
