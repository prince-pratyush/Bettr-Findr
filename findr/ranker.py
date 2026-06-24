"""M6 — Ranker (the differentiator).

Runs semantic + keyword searches, normalizes and blends their scores (weights
from config), boosts exact filename/title matches, dedupes to one result per
file, and returns a ranked list.

Implemented in Step 8.
"""

from __future__ import annotations

from . import config
from .models import QueryFilters, SearchResult


def query(
    text: str,
    k: int = config.DEFAULT_TOP_K,
    filters: QueryFilters | None = None,
) -> list[SearchResult]:
    """Return ranked results for a natural-language query."""
    raise NotImplementedError("ranker.query is implemented in Step 8")
