"""M8 — CLI.

The engine's primary interface and test harness:

    findr index <folder> [<folder> ...]
    findr query "<text>" [--type pdf] [--since 2026-01-01] [--until ...] [-k N]
    findr status

A thin client over the engine modules (indexer, ranker, store).
"""

from __future__ import annotations

from typing import List, Optional

import typer

from . import config, indexer, store
from .models import QueryFilters

app = typer.Typer(
    add_completion=False,
    help="Better Findr — a fully local semantic file finder.",
)


@app.command()
def index(
    folders: List[str] = typer.Argument(..., help="Folders or files to index."),
) -> None:
    """Index (or re-index) the given folders into the local store."""
    seen = 0

    def _progress(fraction: float, path: str) -> None:
        nonlocal seen
        seen += 1
        typer.echo(f"[{fraction:5.0%}] {path}")

    indexer.index_folders(folders, progress=_progress)
    typer.echo(f"Done. Processed {seen} file(s).")


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language description."),
    types: Optional[List[str]] = typer.Option(
        None, "--type", help="Restrict to file type(s), e.g. --type pdf."
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Only files modified on/after this ISO date."
    ),
    until: Optional[str] = typer.Option(
        None, "--until", help="Only files modified on/before this ISO date."
    ),
    top_k: int = typer.Option(config.DEFAULT_TOP_K, "-k", "--top", help="Max results."),
) -> None:
    """Search the index and print ranked results."""
    from . import ranker  # deferred: imports embedder (ollama) only when querying

    filters = QueryFilters(
        file_types=list(types) if types else None,
        modified_since=since,
        modified_until=until,
    )
    results = ranker.query(text, k=top_k, filters=filters)
    if not results:
        typer.echo("No matches.")
        return

    for r in results:
        typer.echo(f"{r.score:6.3f}  {r.file_path}")
        if r.snippet:
            snippet = r.snippet.strip().replace("\n", " ")
            typer.echo(f"         {snippet[:100]}")


@app.command()
def status() -> None:
    """Show index size and last-indexed time."""
    s = store.stats()
    typer.echo(f"Files indexed: {s.file_count}")
    typer.echo(f"Chunks:        {s.chunk_count}")
    typer.echo(f"Last indexed:  {s.last_indexed_at or 'never'}")
    typer.echo(f"Database:      {config.DB_PATH}")


def main() -> None:
    """CLI entry point (registered as the ``findr`` console script)."""
    app()


if __name__ == "__main__":
    main()
