"""M2 — Chunker.

Splits extracted text into overlapping chunks sized for nomic-embed-text's
~512-token window.

Tokens are approximated by whitespace-delimited words. This is deliberately
conservative — a word is on average slightly fewer than one model token — so
chunks comfortably fit the embedding window without a tokenizer dependency.
"""

from __future__ import annotations

from . import config


def chunk(
    text: str,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split `text` into ~`target_tokens`-word chunks overlapping by `overlap`.

    Returns an empty list for empty/whitespace-only input. Raises ``ValueError``
    if `overlap` is not strictly less than `target_tokens` (which would prevent
    forward progress).
    """
    if overlap >= target_tokens:
        raise ValueError(
            f"overlap ({overlap}) must be < target_tokens ({target_tokens})"
        )

    words = text.split()
    if not words:
        return []

    step = target_tokens - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + target_tokens]
        chunks.append(" ".join(window))
        if start + target_tokens >= len(words):
            break
        start += step
    return chunks
