"""M5 — Indexer.

Walks folders, diffs against the manifest, and parses -> chunks -> embeds ->
upserts only changed files (removing deleted ones). Reports progress.

First pass implemented in Step 6; incremental diffing finished in Step 9.
"""

from __future__ import annotations

from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]


def index_folders(
    paths: list[str], progress: Optional[ProgressCallback] = None
) -> None:
    """Index (or re-index) the given folders into the store."""
    raise NotImplementedError("indexer.index_folders is implemented in Step 6")
