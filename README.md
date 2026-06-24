# Better Findr

A fully local, semantic file finder for macOS. Type a plain-English description of a
file you half-remember and get the right file ranked at or near the top — faster and
more accurately than Spotlight, with nothing leaving your machine.

> Phase 1 MVP: a CLI search engine plus a lightweight menu-bar app with a
> Spotlight-style search box. No cloud, no per-use cost.

## How it works

A standalone Python engine does all the work and exposes a tiny local HTTP API
(`127.0.0.1` only). The CLI and the menu-bar app are both just clients of that engine.

```
Menu-bar app + hotkey  ──local HTTP──>  Engine (FastAPI)
                                          ├─ Parser    (PDF / docx / text)
                                          ├─ Chunker
                                          ├─ Embedder  (Ollama nomic-embed-text)
                                          ├─ Store      (sqlite-vec + FTS5)
                                          ├─ Indexer
                                          └─ Ranker     (hybrid semantic + keyword)
```

## Requirements

- macOS
- Python 3.11+
- [Ollama](https://ollama.com) running locally with the embedding model:
  ```bash
  ollama pull nomic-embed-text
  ```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (CLI)

> The CLI is the engine's primary interface and test harness. UI comes later.

```bash
findr index ~/Documents          # index a folder
findr query "that tax pdf from last year" --type pdf --since 2026-01-01
findr status                     # index size + last indexed time
```

## Privacy

Nothing leaves the machine. No telemetry, no external endpoints. The index is stored
locally (default: `~/.bettr-findr/index.db`) and is git-ignored.

## Project status

Phase 1 is being built feature by feature. See the build order in the project spec.
