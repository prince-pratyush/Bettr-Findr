"""M5 — Indexer.

Walks folders, diffs against the manifest, and parses -> chunks -> embeds ->
upserts only changed files (removing deleted ones). Reports progress.

This module is the first-pass implementation: it indexes every supported file
it finds. Incremental diffing against the manifest is added in Step 9.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from . import chunker, config, embedder, parser, store
from .models import Chunk, ManifestEntry

ProgressCallback = Callable[[float, str], None]

logger = logging.getLogger(__name__)


def index_folders(
    paths: list[str], progress: Optional[ProgressCallback] = None
) -> None:
    """Index (or re-index) the given folders into the store.

    `paths` may contain directories (walked recursively) or individual files.
    Only files whose extension is in ``config.SUPPORTED_EXTENSIONS`` are
    considered; unreadable or empty files are skipped. After each file the
    optional `progress` callback is invoked with ``(fraction, file_path)``.
    """
    files = _discover(paths)
    manifest = store.get_manifest()
    roots = [str(Path(p).expanduser()) for p in paths]

    total = len(files)
    for i, path in enumerate(files, start=1):
        try:
            current_hash = _hash_file(path)
            entry = manifest.get(path)
            if entry is None or entry.file_hash != current_hash:
                _index_file(path, file_hash=current_hash)
        except embedder.EmbeddingError:
            raise  # a dead embedder is fatal for the whole run
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't abort
            logger.warning("Failed to index %s: %s", path, exc)
        if progress is not None and total:
            progress(i / total, path)

    _remove_deleted(manifest, set(files), roots)


def _discover(paths: list[str]) -> list[str]:
    """Expand `paths` into a sorted, de-duplicated list of supported files."""
    found: set[str] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file():
            if _is_supported(p):
                found.add(str(p))
        elif p.is_dir():
            for dirpath, _dirnames, filenames in os.walk(p):
                for name in filenames:
                    fp = Path(dirpath) / name
                    if _is_supported(fp):
                        found.add(str(fp))
    return sorted(found)


def _is_supported(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in config.SUPPORTED_EXTENSIONS


def _index_file(path: str, file_hash: str) -> None:
    text = parser.extract_text(path)
    if text is None:
        return

    chunk_texts = chunker.chunk(text)
    if not chunk_texts:
        return

    embeddings = embedder.embed(chunk_texts)

    p = Path(path)
    modified_at = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    file_type = p.suffix.lower().lstrip(".")

    chunks = [
        Chunk(
            chunk_id=f"{path}#{idx}",
            file_path=path,
            file_name=p.name,
            file_type=file_type,
            modified_at=modified_at,
            file_hash=file_hash,
            chunk_index=idx,
            chunk_text=chunk_text,
            embedding=embedding,
        )
        for idx, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings))
    ]

    # A re-index may produce fewer chunks than before; clear stale chunks first.
    store.delete_file(path)
    store.upsert_chunks(chunks)
    store.upsert_manifest(
        ManifestEntry(
            file_path=path,
            file_hash=file_hash,
            modified_at=modified_at,
            chunk_count=len(chunks),
            indexed_at=datetime.now().isoformat(),
        )
    )


def _remove_deleted(
    manifest: dict[str, ManifestEntry], present: set[str], roots: list[str]
) -> None:
    """Drop indexed files that fall under `roots` but no longer exist on disk."""
    for path in manifest:
        if path in present:
            continue
        if _in_scope(path, roots) and not os.path.exists(path):
            store.delete_file(path)
            store.delete_manifest(path)


def _in_scope(path: str, roots: list[str]) -> bool:
    """True if `path` equals a root file or sits under a root directory."""
    for root in roots:
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def _hash_file(path: str, _bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_bufsize), b""):
            h.update(block)
    return h.hexdigest()
