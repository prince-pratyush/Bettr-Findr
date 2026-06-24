"""Central configuration for the Better Findr engine.

Values here are intentionally plain module-level constants so they are easy to
read and override. The semantic/keyword blend weights in particular are config
values (not magic numbers buried in the ranker) so they can be tuned against
real queries.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Storage -----------------------------------------------------------------

# Root directory for all local runtime data. Override with BETTR_FINDR_HOME.
DATA_DIR: Path = Path(
    os.environ.get("BETTR_FINDR_HOME", Path.home() / ".bettr-findr")
).expanduser()

# Single-file SQLite database holding chunks, vectors (sqlite-vec) and the
# FTS5 keyword index. Persists across restarts.
DB_PATH: Path = DATA_DIR / "index.db"


# --- Embeddings --------------------------------------------------------------

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL: str = "nomic-embed-text"
EMBED_DIM: int = 768  # nomic-embed-text output dimensionality
EMBED_BATCH_SIZE: int = 32


# --- Chunking ----------------------------------------------------------------

# nomic-embed-text has a ~512-token window; stay comfortably under it.
CHUNK_TARGET_TOKENS: int = 400
CHUNK_OVERLAP_TOKENS: int = 50


# --- Ranking (the differentiator; tunable) -----------------------------------

# Blend weights for hybrid search. Must sum to 1.0.
SEMANTIC_WEIGHT: float = 0.6
KEYWORD_WEIGHT: float = 0.4

# Extra boost applied when the query matches a file's name/title closely.
FILENAME_MATCH_BOOST: float = 0.15

# Default number of results to return.
DEFAULT_TOP_K: int = 10


# --- Local API ---------------------------------------------------------------

API_HOST: str = "127.0.0.1"  # hard invariant: local only, never bind publicly
API_PORT: int = 8765


# --- Indexing ----------------------------------------------------------------

# Extensions the parser knows how to handle (lower-case, no leading dot).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "pdf",
        "docx",
        "txt",
        "md",
        "markdown",
        "rst",
        "py",
        "js",
        "ts",
        "tsx",
        "jsx",
        "java",
        "go",
        "rs",
        "c",
        "h",
        "cpp",
        "hpp",
        "json",
        "yaml",
        "yml",
        "toml",
        "csv",
        "html",
        "css",
        "sh",
    }
)


def ensure_data_dir() -> Path:
    """Create the data directory if needed and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
