"""M3 — Embedder.

Turns text into 768-dim vectors via local Ollama (nomic-embed-text). Batched.
Surfaces a clear, actionable error if Ollama isn't running.
"""

from __future__ import annotations

import logging
from typing import Optional

import ollama

from . import config

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embeddings can't be produced (e.g. Ollama unreachable)."""


_client: Optional[ollama.Client] = None


def _get_client() -> ollama.Client:
    """Return a cached Ollama client bound to the configured host."""
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Embed `texts`, returning one ``EMBED_DIM``-dim vector per input text.

    Requests are sent to Ollama in batches of ``config.EMBED_BATCH_SIZE``.
    Raises :class:`EmbeddingError` with actionable guidance if the local Ollama
    server can't be reached or the model isn't available.
    """
    if not texts:
        return []

    client = _get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        try:
            response = client.embed(model=config.EMBED_MODEL, input=batch)
        except Exception as exc:  # noqa: BLE001 — normalize into a clear error
            raise EmbeddingError(
                f"Could not reach Ollama at {config.OLLAMA_HOST}. "
                f"Is it running? Start it and pull the model with:\n"
                f"    ollama pull {config.EMBED_MODEL}\n"
                f"Original error: {exc}"
            ) from exc
        vectors.extend(response["embeddings"])

    return vectors
