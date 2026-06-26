"""Tests for M7 — the local API (findr/api.py).

The engine exposes a tiny HTTP API (index / status / query) that the CLI and
menu-bar app consume. Tests drive the ASGI app with FastAPI's TestClient using
a temp DB and mocked embeddings — no network binding, no Ollama.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from findr import api, config, embedder, store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "index.db")
    store.close()
    monkeypatch.setattr(
        embedder,
        "embed",
        lambda texts: [[float(len(t) % 7)] * config.EMBED_DIM for t in texts],
    )
    with TestClient(api.app) as c:
        yield c
    store.close()


def test_status_empty(client):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["file_count"] == 0
    assert body["chunk_count"] == 0


def test_index_then_status(client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("the annual budget report", encoding="utf-8")
    (docs / "b.md").write_text("meeting agenda notes", encoding="utf-8")

    r = client.post("/index", json={"paths": [str(docs)]})
    assert r.status_code == 200, r.text

    s = client.get("/status").json()
    assert s["file_count"] == 2


def test_query_returns_results(client, tmp_path):
    f = tmp_path / "budget.txt"
    f.write_text("the annual budget report for the team", encoding="utf-8")
    client.post("/index", json={"paths": [str(f)]})

    r = client.post("/query", json={"text": "budget"})
    assert r.status_code == 200, r.text
    results = r.json()
    assert any(item["file_path"].endswith("budget.txt") for item in results)
    assert {"file_path", "score", "snippet", "file_type"} <= set(results[0])


def test_query_empty_index_returns_empty_list(client):
    r = client.post("/query", json={"text": "anything"})
    assert r.status_code == 200
    assert r.json() == []


def test_query_requires_text(client):
    r = client.post("/query", json={})
    assert r.status_code == 422  # validation error


def test_query_type_filter(client, tmp_path):
    import fitz

    (tmp_path / "x.txt").write_text("quarterly numbers summary", encoding="utf-8")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "quarterly numbers summary")
    doc.save(str(tmp_path / "x.pdf"))
    doc.close()
    client.post("/index", json={"paths": [str(tmp_path)]})

    r = client.post("/query", json={"text": "quarterly", "file_types": ["pdf"]})
    assert r.status_code == 200, r.text
    types = {item["file_type"] for item in r.json()}
    assert types == {"pdf"}
