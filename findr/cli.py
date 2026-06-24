"""M8 — CLI.

The engine's primary interface and test harness:

    findr index <folder>
    findr query "<text>" [--type pdf] [--since 2026-01-01]
    findr status

Implemented in Step 7 (query is semantic-only until the ranker lands in Step 8).
"""

from __future__ import annotations


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError("cli.main is implemented in Step 7")


if __name__ == "__main__":
    main()
