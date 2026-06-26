"""Tests for M8 — the CLI (findr/cli.py).

The CLI is a thin client over the engine: `index`, `query`, `status`. Tests
drive it through Typer's CliRunner with a temp DB and mocked embeddings.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from findr import config, embedder, store
from findr.cli import app

runner = CliRunner()


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "index.db")
    store.close()
    yield
    store.close()


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[float(len(t) % 7)] * config.EMBED_DIM for t in texts]

    monkeypatch.setattr(embedder, "embed", _embed)


def test_index_then_status(store_db, fake_embed, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("the annual budget report", encoding="utf-8")
    (docs / "b.md").write_text("meeting agenda notes", encoding="utf-8")

    r = runner.invoke(app, ["index", str(docs)])
    assert r.exit_code == 0, r.output

    s = runner.invoke(app, ["status"])
    assert s.exit_code == 0, s.output
    assert "2" in s.output  # two files indexed


def test_query_outputs_matching_path(store_db, fake_embed, tmp_path):
    f = tmp_path / "budget.txt"
    f.write_text("the annual budget report for the team", encoding="utf-8")
    runner.invoke(app, ["index", str(f)])

    r = runner.invoke(app, ["query", "budget"])
    assert r.exit_code == 0, r.output
    assert "budget.txt" in r.output


def test_query_no_results_message(store_db, fake_embed):
    r = runner.invoke(app, ["query", "nonexistentterm"])
    assert r.exit_code == 0, r.output
    assert "no" in r.output.lower()  # e.g. "No matches"


def test_query_type_filter(store_db, fake_embed, tmp_path):
    (tmp_path / "x.txt").write_text("quarterly numbers summary", encoding="utf-8")
    # A real PDF so the parser handles it.
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "quarterly numbers summary")
    doc.save(str(tmp_path / "x.pdf"))
    doc.close()
    runner.invoke(app, ["index", str(tmp_path)])

    r = runner.invoke(app, ["query", "quarterly", "--type", "pdf"])
    assert r.exit_code == 0, r.output
    assert "x.pdf" in r.output
    assert "x.txt" not in r.output


def test_help_lists_commands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("index", "query", "status"):
        assert cmd in r.output
