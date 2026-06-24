"""M3 — Embedder.

Turns text into 768-dim vectors via local Ollama (nomic-embed-text). Batched.
Surfaces a clear, actionable error if Ollama isn't running.

Implemented in Step 4.
"""

from __future__ import annotations


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one 768-dim vector per input text."""
    raise NotImplementedError("embedder.embed is implemented in Step 4")
