"""M7 — API.

FastAPI service bound to 127.0.0.1 exposing /index, /status and /query. The
CLI and menu-bar UI are both clients of this engine.

Routes are intentionally synchronous (`def`, not `async def`): indexing and
embedding do blocking I/O, so FastAPI runs them in its threadpool rather than
on the event loop. Handlers stay thin and delegate to the engine modules.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config, indexer, ranker, store
from .models import QueryFilters

app = FastAPI(
    title="Better Findr",
    version="0.1.0",
    summary="A fully local semantic file finder.",
)


# --- Schemas -----------------------------------------------------------------


class IndexRequest(BaseModel):
    paths: List[str] = Field(..., min_length=1, description="Folders or files.")


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Natural-language query.")
    k: int = Field(config.DEFAULT_TOP_K, ge=1, le=100)
    file_types: Optional[List[str]] = None
    modified_since: Optional[str] = None
    modified_until: Optional[str] = None


class StatusResponse(BaseModel):
    file_count: int
    chunk_count: int
    last_indexed_at: Optional[str] = None
    indexing: bool = False
    progress: float = 0.0


class SearchResultResponse(BaseModel):
    file_path: str
    score: float
    snippet: str
    modified_at: str
    file_name: str
    file_type: str


# --- Routes ------------------------------------------------------------------


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    s = store.stats()
    return StatusResponse(
        file_count=s.file_count,
        chunk_count=s.chunk_count,
        last_indexed_at=s.last_indexed_at,
    )


@app.post("/index", response_model=StatusResponse)
def post_index(req: IndexRequest) -> StatusResponse:
    indexer.index_folders(req.paths)
    s = store.stats()
    return StatusResponse(
        file_count=s.file_count,
        chunk_count=s.chunk_count,
        last_indexed_at=s.last_indexed_at,
    )


@app.post("/query", response_model=List[SearchResultResponse])
def post_query(req: QueryRequest) -> List[SearchResultResponse]:
    filters = QueryFilters(
        file_types=req.file_types,
        modified_since=req.modified_since,
        modified_until=req.modified_until,
    )
    results = ranker.query(req.text, k=req.k, filters=filters)
    return [
        SearchResultResponse(
            file_path=r.file_path,
            score=r.score,
            snippet=r.snippet,
            modified_at=r.modified_at,
            file_name=r.file_name,
            file_type=r.file_type,
        )
        for r in results
    ]


def run() -> None:
    """Launch the engine locally (127.0.0.1 only — never bind publicly)."""
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    run()
