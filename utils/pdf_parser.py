"""
utils/pdf_parser.py
-------------------
PDF text extractor for the LLM-Powered Resume Screener.

Primary:  PyMuPDF (fitz) — faster, handles multi-column layouts and
          embedded fonts better than pdfplumber on resume PDFs.
Fallback: pdfplumber — used automatically if PyMuPDF raises any exception
          on a given file. The fallback is logged so you can investigate
          problematic PDFs later.

Public API
----------
parse_pdf(path: str | Path) -> ParsedDocument
    Extract text + metadata from a single PDF file.

extract_sections(text: str) -> dict[str, str]
    Split raw resume text into labelled sections
    (EXPERIENCE, EDUCATION, SKILLS, SUMMARY, etc.).

validate_pdf(path: str | Path, max_mb: float = 5.0) -> None
    Raise ValueError with a clean message if the file is not a valid
    PDF or exceeds the size limit. Call this before parse_pdf on any
    user-uploaded file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    """Result returned by parse_pdf()."""

    path: str                          # original file path
    text: str                          # full extracted text, all pages joined
    pages: list[str]                   # per-page text (index 0 = page 1)
    page_count: int
    sections: dict[str, str]           # keyed by section header, e.g. "EXPERIENCE"
    parser_used: str                   # "pymupdf" | "pdfplumber"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Section headers we recognise in resumes
# ---------------------------------------------------------------------------

# Ordered by priority — earlier patterns win on ambiguous headers.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("SUMMARY",     r"\b(summary|profile|objective|about me|professional summary)\b"),
    ("EXPERIENCE",  r"\b(experience|work experience|employment|work history|career)\b"),
    ("EDUCATION",   r"\b(education|academic|qualifications|degrees?)\b"),
    ("SKILLS",      r"\b(skills|technical skills|core competencies|competencies|technologies)\b"),
    ("PROJECTS",    r"\b(projects?|personal projects?|portfolio)\b"),
    ("CERTIFICATIONS", r"\b(certifications?|certificates?|accreditations?|licenses?)\b"),
    ("PUBLICATIONS", r"\b(publications?|research|papers?)\b"),
    ("AWARDS",      r"\b(awards?|honors?|achievements?|recognition)\b"),
    ("LANGUAGES",   r"\b(languages?)\b"),
    ("INTERESTS",   r"\b(interests?|hobbies|activities)\b"),
]

# A header line is: short (≤ 60 chars), mostly uppercase or title-case,
# no sentence-ending punctuation, optionally followed by a colon or line.
_HEADER_RE = re.compile(
    r"^[ \t]*([A-Z][A-Z &/\-]{1,58})[ \t]*:?\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pdf(path: str | Path, max_mb: float = 5.0) -> None:
    """
    Validate that *path* is a readable PDF file within the size limit.

    Parameters
    ----------
    path:    File path (str or Path).
    max_mb:  Maximum allowed file size in megabytes (default 5.0).

    Raises
    ------
    ValueError  — with a human-readable message on any validation failure.
                  Callers should surface this message directly to the UI.
    FileNotFoundError — if the path does not exist.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file type '{p.suffix}'. Only PDF files are accepted."
        )

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"File '{p.name}' is {size_mb:.1f} MB, which exceeds the "
            f"{max_mb:.0f} MB limit. Please upload a smaller file."
        )

    # Peek at the PDF magic bytes — %PDF
    with p.open("rb") as fh:
        header = fh.read(5)
    if not header.startswith(b"%PDF"):
        raise ValueError(
            f"'{p.name}' does not appear to be a valid PDF file."
        )


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _extract_with_pymupdf(path: Path) -> tuple[list[str], int]:
    """
    Extract per-page text using PyMuPDF (fitz).

    Returns (pages, page_count).
    Each page string preserves paragraph breaks but strips excessive
    whitespace runs that fitz sometimes produces in multi-column layouts.
    """
    import fitz  # pymupdf

    pages: list[str] = []
    doc = fitz.open(str(path))
    try:
        for page in doc:
            # "text" mode gives plain text; "blocks" mode would give
            # bounding boxes — we stay with "text" for simplicity.
            raw = page.get_text("text")
            # Collapse 3+ consecutive blank lines into a single blank line
            cleaned = re.sub(r"\n{3,}", "\n\n", raw)
            pages.append(cleaned.strip())
    finally:
        doc.close()

    return pages, len(pages)


def _extract_with_pdfplumber(path: Path) -> tuple[list[str], int]:
    """
    Extract per-page text using pdfplumber.

    Used as fallback when PyMuPDF raises an exception.
    pdfplumber is slower but handles some edge cases (certain font
    encodings, password-removed PDFs) that fitz struggles with.
    """
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            cleaned = re.sub(r"\n{3,}", "\n\n", raw)
            pages.append(cleaned.strip())

    return pages, len(pages)


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def extract_sections(text: str) -> dict[str, str]:
    """
    Split *text* into labelled resume sections.

    Heuristic: A line is treated as a section header if it matches
    _HEADER_RE *and* one of the known section patterns in
    _SECTION_PATTERNS.  Everything between two consecutive headers is
    assigned to the first header's section.

    If no headers are detected the entire text is stored under "FULL_TEXT"
    so downstream code always has something to work with.

    Parameters
    ----------
    text:  Full resume text (all pages joined).

    Returns
    -------
    dict[str, str]
        Keys are canonical section names (e.g. "EXPERIENCE").
        Values are the raw text belonging to that section.
        A special key "HEADER" captures text before the first section
        (usually the candidate's name and contact info).
    """
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_label = "HEADER"
    buffer: list[str] = []

    def _flush(label: str, buf: list[str]) -> None:
        content = "\n".join(buf).strip()
        if content:
            if label in sections:
                # Append if the same section appears twice (rare but possible)
                sections[label] = sections[label] + "\n" + content
            else:
                sections[label] = content

    for line in lines:
        # Test if this line looks like an all-caps header
        header_match = _HEADER_RE.match(line)
        if header_match:
            candidate = header_match.group(1).strip()
            matched_label: str | None = None
            for label, pattern in _SECTION_PATTERNS:
                if re.search(pattern, candidate, re.IGNORECASE):
                    matched_label = label
                    break
            if matched_label:
                _flush(current_label, buffer)
                current_label = matched_label
                buffer = []
                continue  # don't add the header line itself to content

        buffer.append(line)

    _flush(current_label, buffer)  # flush the last section

    if not sections or (len(sections) == 1 and "HEADER" in sections):
        # No structure detected — return full text so nothing is lost
        logger.warning("No resume sections detected; returning FULL_TEXT.")
        sections["FULL_TEXT"] = text

    return sections


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_pdf(path: str | Path) -> ParsedDocument:
    """
    Extract text and structure from a PDF resume.

    Tries PyMuPDF first; on any exception falls back to pdfplumber and
    records a warning in the returned ParsedDocument.

    Parameters
    ----------
    path:  Path to the PDF file (str or pathlib.Path).

    Returns
    -------
    ParsedDocument
        Contains full text, per-page text, section dict, and metadata.

    Raises
    ------
    RuntimeError  — if both parsers fail (very rare; indicates a
                    fundamentally broken PDF).
    """
    p = Path(path)
    warnings: list[str] = []
    parser_used = "pymupdf"
    pages: list[str] = []
    page_count = 0

    # --- Primary: PyMuPDF ---
    try:
        pages, page_count = _extract_with_pymupdf(p)
        logger.debug("PyMuPDF extracted %d pages from '%s'", page_count, p.name)
    except Exception as primary_exc:
        logger.warning(
            "PyMuPDF failed on '%s' (%s) — falling back to pdfplumber.",
            p.name, primary_exc,
        )
        warnings.append(f"PyMuPDF failed ({primary_exc}); pdfplumber fallback used.")
        parser_used = "pdfplumber"

        # --- Fallback: pdfplumber ---
        try:
            pages, page_count = _extract_with_pdfplumber(p)
            logger.debug(
                "pdfplumber extracted %d pages from '%s'", page_count, p.name
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Both PyMuPDF and pdfplumber failed on '{p.name}'. "
                f"Primary error: {primary_exc}. "
                f"Fallback error: {fallback_exc}."
            ) from fallback_exc

    # Join pages with a clear page separator so section detection works
    # across page boundaries (headers often sit at the top of page 2).
    full_text = "\n\n".join(pages)

    # Warn if extracted text looks suspiciously short (likely a scanned PDF)
    if len(full_text.strip()) < 100:
        msg = (
            f"'{p.name}' produced very little text ({len(full_text.strip())} chars). "
            "It may be a scanned/image-only PDF — OCR is not supported."
        )
        logger.warning(msg)
        warnings.append(msg)

    sections = extract_sections(full_text)

    return ParsedDocument(
        path=str(p),
        text=full_text,
        pages=pages,
        page_count=page_count,
        sections=sections,
        parser_used=parser_used,
        warnings=warnings,
    )
