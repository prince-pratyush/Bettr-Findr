"""M1 — Parser.

Extracts clean text from a file. Dispatch by extension. Never raises to the
caller: returns None on failure, and unsupported types are skipped silently
with a log line.

Implemented in Step 2.
"""

from __future__ import annotations

from typing import Optional


def extract_text(path: str) -> Optional[str]:
    """Return clean extracted text for `path`, or None if it can't be parsed."""
    raise NotImplementedError("parser.extract_text is implemented in Step 2")
