"""M1 — Parser.

Extracts clean text from a file. Dispatch by extension. Never raises to the
caller: returns None on failure, and unsupported types are skipped silently
with a log line.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


def extract_text(path: str) -> Optional[str]:
    """Return clean extracted text for `path`, or None if it can't be parsed.

    Dispatch is by file extension. Unsupported extensions are skipped, and any
    extraction error (missing/corrupt/unreadable file) is logged and turned
    into ``None`` so a single bad file never aborts an index run.
    """
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in config.SUPPORTED_EXTENSIONS:
        logger.debug("Skipping unsupported file type: %s", path)
        return None

    try:
        if ext == "pdf":
            return _extract_pdf(path)
        if ext == "docx":
            return _extract_docx(path)
        return _extract_textlike(path)
    except Exception as exc:  # noqa: BLE001 — never propagate to the indexer
        logger.warning("Failed to parse %s: %s", path, exc)
        return None


def _extract_textlike(path: str) -> str:
    """Read a plain-text or source-code file, tolerating non-UTF-8 bytes."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: str) -> str:
    """Extract text from a PDF using pymupdf (fitz)."""
    import fitz

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _extract_docx(path: str) -> str:
    """Extract text from a .docx using python-docx."""
    from docx import Document

    document = Document(path)
    return "\n".join(p.text for p in document.paragraphs)
