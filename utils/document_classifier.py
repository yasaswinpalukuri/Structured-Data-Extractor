"""
Document classifier for the Structured Data Extractor pipeline.

Classifies documents as 'job_description', 'resume', 'invoice', or 'unknown'
using a heuristic-first approach (< 10ms, no LLM) with LLM fallback when
heuristic confidence is below 60%.
"""

import logging
import time
from dataclasses import dataclass
from typing import Literal

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

DocumentType = Literal["job_description", "resume", "invoice", "unknown"]


# ---------------------------------------------------------------------------
# Keyword banks — tuned for real-world document language
# ---------------------------------------------------------------------------

_JD_KEYWORDS: list[str] = [
    "responsibilities",
    "qualifications",
    "requirements",
    "we are looking for",
    "you will",
    "we offer",
    "apply now",
    "job description",
    "position",
    "vacancy",
    "role",
    "candidate",
    "hiring",
    "compensation",
    "benefits",
    "pto",
    "equal opportunity",
    "years of experience",
    "preferred skills",
    "nice to have",
    "must have",
    "team player",
    "remote",
    "hybrid",
    "on-site",
]

_RESUME_KEYWORDS: list[str] = [
    "curriculum vitae",
    "objective",
    "summary",
    "work experience",
    "employment history",
    "education",
    "skills",
    "certifications",
    "references",
    "projects",
    "achievements",
    "publications",
    "languages",
    "volunteer",
    "github.com",
    "linkedin.com",
    "gpa",
    "bachelor",
    "master",
    "ph.d",
    "university",
    "college",
    "internship",
    "proficient in",
    "experienced in",
]

_INVOICE_KEYWORDS: list[str] = [
    "invoice",
    "invoice number",
    "invoice date",
    "bill to",
    "ship to",
    "due date",
    "payment terms",
    "subtotal",
    "tax",
    "total amount",
    "amount due",
    "purchase order",
    "p.o.",
    "qty",
    "quantity",
    "unit price",
    "line item",
    "remit to",
    "net 30",
    "net 15",
    "overdue",
    "receipt",
    "vendor",
    "payable",
]

# Weight multipliers for high-signal keywords
_STRONG_JD: set[str] = {"job description", "qualifications", "responsibilities", "we are looking for", "apply now"}
_STRONG_RESUME: set[str] = {"curriculum vitae", "work experience", "employment history", "gpa", "references"}
_STRONG_INVOICE: set[str] = {"invoice", "bill to", "amount due", "invoice number", "subtotal", "total amount"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of a document classification attempt."""

    document_type: DocumentType
    confidence: int          # 0–100
    method: str              # "heuristic" | "llm"
    latency_ms: float
    raw_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

def _score_heuristic(text: str) -> dict[str, float]:
    """
    Count weighted keyword hits for each document type.

    Strong keywords count as 3 hits; regular keywords as 1.
    Returns a dict of raw scores per type.
    """
    lower = text.lower()
    scores: dict[str, float] = {"job_description": 0.0, "resume": 0.0, "invoice": 0.0}

    for kw in _JD_KEYWORDS:
        if kw in lower:
            scores["job_description"] += 3.0 if kw in _STRONG_JD else 1.0

    for kw in _RESUME_KEYWORDS:
        if kw in lower:
            scores["resume"] += 3.0 if kw in _STRONG_RESUME else 1.0

    for kw in _INVOICE_KEYWORDS:
        if kw in lower:
            scores["invoice"] += 3.0 if kw in _STRONG_INVOICE else 1.0

    return scores


def _scores_to_confidence(scores: dict[str, float], winner: str) -> int:
    """
    Convert raw keyword scores into a 0–100 confidence value.

    Confidence reflects how clearly the winner stands above the others.
    """
    total = sum(scores.values())
    if total == 0:
        return 0

    winner_score = scores[winner]
    winner_ratio = winner_score / total  # 0.0 – 1.0

    # Scale: ratio=1.0 → 95, ratio=0.5 → 50, ratio=0.33 → ~20
    raw_confidence = int(winner_ratio * 95)

    # Floor: if winner has ≥ 5 hits, guarantee at least 40
    if winner_score >= 5:
        raw_confidence = max(raw_confidence, 40)

    return min(raw_confidence, 100)


def classify_heuristic(text: str) -> ClassificationResult:
    """
    Classify a document using keyword heuristics only. No LLM call.

    Args:
        text: Raw document text (plain text, not PDF bytes).

    Returns:
        ClassificationResult with method='heuristic'.
    """
    start = time.perf_counter()
    scores = _score_heuristic(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if max(scores.values()) == 0:
        logger.debug("Heuristic: no keyword hits — returning 'unknown'")
        return ClassificationResult(
            document_type="unknown",
            confidence=0,
            method="heuristic",
            latency_ms=elapsed_ms,
            raw_scores=scores,
        )

    winner: DocumentType = max(scores, key=scores.__getitem__)  # type: ignore[assignment]
    confidence = _scores_to_confidence(scores, winner)

    logger.debug(
        "Heuristic scores: %s → winner='%s' confidence=%d (%.2fms)",
        scores, winner, confidence, elapsed_ms,
    )
    return ClassificationResult(
        document_type=winner,
        confidence=confidence,
        method="heuristic",
        latency_ms=elapsed_ms,
        raw_scores=scores,
    )


# ---------------------------------------------------------------------------
# LLM fallback classifier
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a document classifier. Classify the document below into exactly one of these categories:
- job_description
- resume
- invoice
- unknown

Rules:
1. Reply with ONLY the category label — no explanation, no punctuation, no extra words.
2. Use 'unknown' only if the document clearly does not fit any of the three categories.

Document:
\"\"\"
{text}
\"\"\"

Category:"""


def classify_llm(
    text: str,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
) -> ClassificationResult:
    """
    Classify a document using the local LLM (Ollama).

    Only called when heuristic confidence < 60. Takes longer than heuristic
    but handles ambiguous or mixed-content documents.

    Args:
        text:             Raw document text.
        ollama_base_url:  Ollama server URL.
        model:            Ollama model name.

    Returns:
        ClassificationResult with method='llm'.
    """
    start = time.perf_counter()

    llm = ChatOllama(
        base_url=ollama_base_url,
        model=model,
        temperature=0.0,
        num_ctx=2048,   # Small context — we only need the label
    )

    prompt = _LLM_PROMPT.format(text=text[:3000])  # Cap at 3000 chars for speed
    response = llm.invoke(prompt)
    raw_label = response.content.strip().lower().replace('"', "").replace("'", "")

    elapsed_ms = (time.perf_counter() - start) * 1000

    valid: set[DocumentType] = {"job_description", "resume", "invoice", "unknown"}
    doc_type: DocumentType = raw_label if raw_label in valid else "unknown"  # type: ignore[assignment]

    # LLM responses get fixed confidence: 75 if recognised, 30 if unknown/invalid
    confidence = 75 if doc_type != "unknown" else 30

    logger.info(
        "LLM classifier: raw='%s' → type='%s' confidence=%d (%.0fms)",
        raw_label, doc_type, confidence, elapsed_ms,
    )
    return ClassificationResult(
        document_type=doc_type,
        confidence=confidence,
        method="llm",
        latency_ms=elapsed_ms,
        raw_scores={},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_document(
    text: str,
    heuristic_threshold: int = 60,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
) -> ClassificationResult:
    """
    Classify a document using heuristics first, LLM fallback if needed.

    Strategy:
        1. Run keyword heuristic (< 10ms).
        2. If confidence >= heuristic_threshold → return heuristic result.
        3. Otherwise → call LLM and return LLM result.

    Args:
        text:                 Raw document text.
        heuristic_threshold:  Minimum confidence to trust heuristic (default 60).
        ollama_base_url:      Ollama server URL.
        model:                Ollama model name.

    Returns:
        ClassificationResult with document_type, confidence, method, latency_ms.
    """
    if not text or not text.strip():
        logger.warning("classify_document called with empty text")
        return ClassificationResult(
            document_type="unknown",
            confidence=0,
            method="heuristic",
            latency_ms=0.0,
            raw_scores={},
        )

    heuristic_result = classify_heuristic(text)

    if heuristic_result.confidence >= heuristic_threshold:
        logger.info(
            "Classification (heuristic): type='%s' confidence=%d latency=%.2fms",
            heuristic_result.document_type,
            heuristic_result.confidence,
            heuristic_result.latency_ms,
        )
        return heuristic_result

    logger.info(
        "Heuristic confidence %d < threshold %d — falling back to LLM",
        heuristic_result.confidence,
        heuristic_threshold,
    )
    llm_result = classify_llm(text, ollama_base_url=ollama_base_url, model=model)
    logger.info(
        "Classification (llm): type='%s' confidence=%d latency=%.0fms",
        llm_result.document_type,
        llm_result.confidence,
        llm_result.latency_ms,
    )
    return llm_result
