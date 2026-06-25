"""Tests for M3 — the embedder (findr/embedder.py).

The embedder turns text into 768-dim vectors via local Ollama
(nomic-embed-text), batched. These tests use a fake Ollama client so they run
without a live Ollama server.

User journeys:
- As the indexer, I want a batch of chunk texts embedded into vectors.
- As a user, if Ollama isn't running I want a clear, actionable error rather
  than a cryptic connection traceback.
"""

from __future__ import annotations

import pytest

from findr import config, embedder


class FakeClient:
    """Records the batches it receives and returns deterministic vectors."""

    def __init__(self, dim: int = config.EMBED_DIM):
        self.dim = dim
        self.batch_sizes: list[int] = []
        self.models: list[str] = []

    def embed(self, model: str, input):  # noqa: A002 — matches ollama API
        items = list(input)
        self.batch_sizes.append(len(items))
        self.models.append(model)
        return {"embeddings": [[float(len(t))] * self.dim for t in items]}


class DownClient:
    def embed(self, model: str, input):  # noqa: A002
        raise ConnectionError("[Errno 61] Connection refused")


def test_empty_input_returns_empty_list():
    # Must not even touch the client.
    assert embedder.embed([]) == []


def test_one_vector_per_text(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(embedder, "_get_client", lambda: fake)

    out = embedder.embed(["alpha", "beta", "gamma"])

    assert len(out) == 3
    for vec in out:
        assert len(vec) == config.EMBED_DIM


def test_uses_configured_embed_model(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(embedder, "_get_client", lambda: fake)

    embedder.embed(["x"])

    assert fake.models == [config.EMBED_MODEL]


def test_batches_respect_batch_size(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(embedder, "_get_client", lambda: fake)

    n = config.EMBED_BATCH_SIZE * 2 + 6
    out = embedder.embed([f"text-{i}" for i in range(n)])

    assert len(out) == n
    assert fake.batch_sizes == [
        config.EMBED_BATCH_SIZE,
        config.EMBED_BATCH_SIZE,
        6,
    ]


def test_preserves_order(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(embedder, "_get_client", lambda: fake)

    texts = ["a", "bb", "ccc", "dddd"]  # distinct lengths
    out = embedder.embed(texts)

    # FakeClient encodes len(text) as the first vector component.
    assert [vec[0] for vec in out] == [1.0, 2.0, 3.0, 4.0]


def test_raises_clear_error_when_ollama_down(monkeypatch):
    monkeypatch.setattr(embedder, "_get_client", lambda: DownClient())

    with pytest.raises(embedder.EmbeddingError) as exc:
        embedder.embed(["anything"])

    assert "ollama" in str(exc.value).lower()
