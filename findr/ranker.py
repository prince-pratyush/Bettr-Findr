"""M6 — Ranker (the differentiator).

Runs semantic + keyword searches, normalizes and blends their scores (weights
from config), boosts exact filename/title matches, dedupes to one result per
file, and returns a ranked list.

Blending:
- Each modality's raw scores are min-max normalized to [0, 1] so the two
  scales are comparable before weighting.
- combined = SEMANTIC_WEIGHT * sem + KEYWORD_WEIGHT * kw
- A flat FILENAME_MATCH_BOOST is added when a query term appears in the file
  name, since a name match is a strong signal for "find that file".
"""

from __future__ import annotations

import re

from . import config, embedder, store
from .models import QueryFilters, SearchResult

# Pull more candidates per modality than the caller asked for, so blending and
# dedup have enough to work with before trimming to k.
_CANDIDATE_FACTOR = 3
_CANDIDATE_FLOOR = 20


def query(
    text: str,
    k: int = config.DEFAULT_TOP_K,
    filters: QueryFilters | None = None,
) -> list[SearchResult]:
    """Return ranked, de-duplicated results for a natural-language query."""
    if not text.strip():
        return []

    candidate_k = max(k * _CANDIDATE_FACTOR, _CANDIDATE_FLOOR)
    query_vec = embedder.embed([text])[0]
    semantic = store.search_semantic(query_vec, candidate_k, filters)
    keyword = store.search_keyword(text, candidate_k, filters)

    sem_scores = _normalized_by_file(semantic)
    kw_scores = _normalized_by_file(keyword)

    # Keep one representative SearchResult per file (best-scoring chunk seen),
    # preferring whichever modality scored that file higher for the snippet.
    rep: dict[str, SearchResult] = {}
    for result in semantic:
        rep.setdefault(result.file_path, result)
    for result in keyword:
        path = result.file_path
        if path not in rep or kw_scores[path] > sem_scores.get(path, 0.0):
            rep[path] = result

    terms = set(re.findall(r"\w+", text.lower()))

    ranked: list[SearchResult] = []
    for path, result in rep.items():
        combined = (
            config.SEMANTIC_WEIGHT * sem_scores.get(path, 0.0)
            + config.KEYWORD_WEIGHT * kw_scores.get(path, 0.0)
        )
        if _filename_matches(result.file_name, terms):
            combined += config.FILENAME_MATCH_BOOST
        ranked.append(
            SearchResult(
                file_path=result.file_path,
                score=combined,
                snippet=result.snippet,
                modified_at=result.modified_at,
                file_name=result.file_name,
                file_type=result.file_type,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:k]


def _normalized_by_file(results: list[SearchResult]) -> dict[str, float]:
    """Min-max normalize raw scores to [0, 1], keeping the best per file."""
    if not results:
        return {}

    best: dict[str, float] = {}
    for r in results:
        if r.file_path not in best or r.score > best[r.file_path]:
            best[r.file_path] = r.score

    values = best.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return {path: 1.0 for path in best}
    return {path: (score - lo) / (hi - lo) for path, score in best.items()}


def _filename_matches(file_name: str, terms: set[str]) -> bool:
    name = file_name.lower()
    return any(term in name for term in terms)
