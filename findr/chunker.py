"""M2 — Chunker.

Splits extracted text into overlapping chunks sized for nomic-embed-text's
~512-token window.

Implemented in Step 3.
"""

from __future__ import annotations

from . import config


def chunk(
    text: str,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split `text` into chunks of ~`target_tokens` with `overlap` token overlap."""
    raise NotImplementedError("chunker.chunk is implemented in Step 3")
