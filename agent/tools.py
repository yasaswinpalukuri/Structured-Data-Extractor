"""
LangChain tools for the Structured Data Extractor pipeline.

Four tools used by the extraction agent:
    - classify_document_tool   : Wraps document_classifier.classify_document
    - extract_fields_tool      : LLM extraction → validated Pydantic model
    - validate_extraction_tool : Re-validates an already-extracted dict
    - notion_push_tool         : Pushes extracted data to the correct Notion database

All tools are plain functions decorated with @tool so they can be bound
to a LangChain agent or called directly in tests (with mocked LLM/Notion).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from agent.prompts import RETRY_PROMPT, get_prompt_for_type, get_system_instruction
from agent.structured_output import (
    AnyDocumentSchema,
    get_schema_for_type,
)
from utils.document_classifier import classify_document as _classify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM from environment variables."""
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        temperature=0.0,
        num_ctx=int(os.getenv("OLLAMA_CTX", "8192")),
    )


def _strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Handles:
        ```json ... ```
        ``` ... ```
        Plain JSON (returned as-is)
    """
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    return fenced.strip()


def _normalise_jd_fields(data: dict) -> dict:
    """Map common LLM field name variations to our JD schema field names."""
    aliases = {
        # role_title
        "job_title": "role_title",
        "title": "role_title",
        "position": "role_title",
        "position_title": "role_title",
        "job_position": "role_title",
        # location
        "work_location": "location",
        "job_location": "location",
        "office_location": "location",
        # remote_type
        "work_type": "remote_type",
        "location_type": "remote_type",
        "remote": "remote_type",
        # salary
        "salary": "salary_min",
        # industry
        "sector": "industry",
        "field": "industry",
        "domain": "industry",
        # company_name
        "company": "company_name",
        "employer": "company_name",
        "organization": "company_name",
        "organisation": "company_name",
    }
    for old_key, new_key in aliases.items():
        if old_key in data and new_key not in data:
            data[new_key] = data.pop(old_key)

    # salary_range: [min, max] → salary_min, salary_max
    if "salary_range" in data and isinstance(data["salary_range"], list):
        rng = data.pop("salary_range")
        if len(rng) >= 1 and "salary_min" not in data:
            data["salary_min"] = rng[0]
        if len(rng) >= 2 and "salary_max" not in data:
            data["salary_max"] = rng[1]

    # currency → salary_currency
    if "currency" in data and "salary_currency" not in data:
        data["salary_currency"] = data.pop("currency")

    # remote_type normalisation
    if "remote_type" in data:
        val = str(data["remote_type"]).strip().lower()
        mapping = {
            "remote": "Remote",
            "hybrid": "Hybrid",
            "on-site": "On-site",
            "onsite": "On-site",
            "on site": "On-site",
            "in-office": "On-site",
            "in office": "On-site",
        }
        data["remote_type"] = mapping.get(val, "Not specified")

    # industry fallback
    if not data.get("industry"):
        data["industry"] = "Not specified"

    # company_name fallback
    if not data.get("company_name"):
        data["company_name"] = "Not specified"

    return data


def _normalise_resume_fields(data: dict) -> dict:
    """Map common LLM field name variations to our Resume schema field names."""
    aliases = {
        "name": "candidate_name",
        "full_name": "candidate_name",
        "applicant_name": "candidate_name",
        "email_address": "email",
        "phone_number": "phone",
        "mobile": "phone",
        "city": "location",
        "address": "location",
        "linkedin": "linkedin_url",
        "github": "github_url",
        "experience_years": "years_of_experience",
        "total_experience": "years_of_experience",
        "current_position": "current_role",
        "current_title": "current_role",
        "current_employer": "current_company",
    }
    for old_key, new_key in aliases.items():
        if old_key in data and new_key not in data:
            data[new_key] = data.pop(old_key)

    if not data.get("candidate_name"):
        data["candidate_name"] = "Unknown Candidate"

    return data


def _normalise_invoice_fields(data: dict) -> dict:
    """Map common LLM field name variations to our Invoice schema field names."""
    aliases = {
        "invoice_no": "invoice_number",
        "inv_number": "invoice_number",
        "bill_to": "client_name",
        "billed_to": "client_name",
        "customer": "client_name",
        "from": "vendor_name",
        "supplier": "vendor_name",
        "seller": "vendor_name",
        "total": "total_amount",
        "amount_due": "total_amount",
        "grand_total": "total_amount",
        "tax_percentage": "tax_rate",
        "vat_rate": "tax_rate",
        "currency_code": "currency",
    }
    for old_key, new_key in aliases.items():
        if old_key in data and new_key not in data:
            data[new_key] = data.pop(old_key)

    if not data.get("vendor_name"):
        data["vendor_name"] = "Unknown Vendor"
    if not data.get("client_name"):
        data["client_name"] = "Unknown Client"
    if not data.get("currency"):
        data["currency"] = "USD"
    if data.get("total_amount") is None:
        data["total_amount"] = 0.0

    return data


_NORMALISERS = {
    "job_description": _normalise_jd_fields,
    "resume": _normalise_resume_fields,
    "invoice": _normalise_invoice_fields,
}


def _parse_and_validate(
    raw_text: str,
    document_type: str,
) -> AnyDocumentSchema:
    """
    Parse LLM output as JSON, normalise field names, and validate against
    the correct Pydantic schema.

    Args:
        raw_text:      Raw string output from the LLM.
        document_type: One of 'job_description', 'resume', 'invoice'.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValueError:        If JSON cannot be parsed.
        ValidationError:   If parsed JSON fails Pydantic validation.
    """
    cleaned = _strip_json_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}\nRaw output:\n{raw_text[:500]}"
        ) from exc

    # Normalise field name variations before Pydantic validation
    normaliser = _NORMALISERS.get(document_type)
    if normaliser:
        data = normaliser(data)

    # Force document_type to match what we classified — LLM may omit or mismatch
    data["document_type"] = document_type

    schema_cls = get_schema_for_type(document_type)
    return schema_cls.model_validate(data)


# ---------------------------------------------------------------------------
# Tool 1: classify_document_tool
# ---------------------------------------------------------------------------

@tool
def classify_document_tool(document_text: str) -> dict[str, Any]:
    """
    Classify a document as job_description, resume, invoice, or unknown.

    Uses keyword heuristics first (< 10ms). Falls back to LLM only when
    heuristic confidence is below 60.

    Args:
        document_text: Raw plain text of the document.

    Returns:
        Dict with keys: document_type, confidence, method, latency_ms.
    """
    result = _classify(
        text=document_text,
        heuristic_threshold=60,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    )
    logger.info(
        "classify_document_tool: type=%s confidence=%d method=%s",
        result.document_type, result.confidence, result.method,
    )
    return {
        "document_type": result.document_type,
        "confidence": result.confidence,
        "method": result.method,
        "latency_ms": round(result.latency_ms, 2),
    }


# ---------------------------------------------------------------------------
# Tool 2: extract_fields_tool
# ---------------------------------------------------------------------------

@tool
def extract_fields_tool(document_text: str, document_type: str) -> dict[str, Any]:
    """
    Extract structured fields from a document using the local LLM.

    Sends the document through the correct extraction prompt, parses the
    JSON response, and validates it against the Pydantic schema.

    If the first attempt returns malformed JSON, retries once with an
    explicit JSON-only instruction before raising an error.

    Args:
        document_text: Raw plain text of the document.
        document_type: One of 'job_description', 'resume', 'invoice'.

    Returns:
        Dict of extracted fields (Pydantic model serialised to dict).

    Raises:
        ValueError: If both attempts fail to produce valid JSON.
        ValueError: If document_type is not recognised.
    """
    if document_type == "unknown":
        raise ValueError(
            "Cannot extract fields from an 'unknown' document type. "
            "Run classify_document_tool first."
        )

    llm = _get_llm()
    system = get_system_instruction()
    user_prompt = get_prompt_for_type(document_type).format(
        document_text=document_text
    )

    # --- Attempt 1 ---
    start = time.perf_counter()
    messages = [
        ("system", system),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)
    raw = response.content
    latency_ms = (time.perf_counter() - start) * 1000
    logger.debug("extract_fields_tool attempt 1: %.0fms", latency_ms)

    try:
        validated = _parse_and_validate(raw, document_type)
        result = validated.model_dump()
        result["_extraction_latency_ms"] = round(latency_ms, 0)
        logger.info(
            "extract_fields_tool: type=%s confidence=%s latency=%.0fms",
            document_type, result.get("confidence_score"), latency_ms,
        )
        return result

    except (ValueError, ValidationError) as exc:
        logger.warning(
            "extract_fields_tool attempt 1 failed (%s) — retrying with JSON instruction",
            exc,
        )

    # --- Attempt 2: retry with explicit JSON reminder ---
    retry_prompt = RETRY_PROMPT.format(
        document_type=document_type,
        document_text=document_text,
    )
    start2 = time.perf_counter()
    retry_response = llm.invoke([("system", system), ("human", retry_prompt)])
    raw2 = retry_response.content
    latency_ms2 = (time.perf_counter() - start2) * 1000
    logger.debug("extract_fields_tool attempt 2: %.0fms", latency_ms2)

    try:
        validated2 = _parse_and_validate(raw2, document_type)
        result2 = validated2.model_dump()
        result2["_extraction_latency_ms"] = round(latency_ms + latency_ms2, 0)
        result2["_retry_used"] = True
        logger.info(
            "extract_fields_tool retry succeeded: type=%s confidence=%s",
            document_type, result2.get("confidence_score"),
        )
        return result2

    except (ValueError, ValidationError) as exc2:
        logger.error(
            "extract_fields_tool: both attempts failed for type=%s. Error: %s\nRaw output:\n%s",
            document_type, exc2, raw2[:500],
        )
        raise ValueError(
            f"Extraction failed after 2 attempts for document_type='{document_type}'. "
            f"Last error: {exc2}\nRaw LLM output:\n{raw2[:300]}"
        ) from exc2


# ---------------------------------------------------------------------------
# Tool 3: validate_extraction_tool
# ---------------------------------------------------------------------------

@tool
def validate_extraction_tool(
    extracted_data: dict[str, Any],
    document_type: str,
) -> dict[str, Any]:
    """
    Re-validate a previously extracted dict against the correct Pydantic schema.

    Useful after manual edits or when loading from storage to confirm
    the data still conforms to the schema.

    Args:
        extracted_data: Dict of field values (as returned by extract_fields_tool).
        document_type:  One of 'job_description', 'resume', 'invoice'.

    Returns:
        Dict with keys:
            - valid (bool)
            - errors (list[str]) — empty if valid
            - data (dict) — re-serialised model if valid, else original dict
    """
    schema_cls = get_schema_for_type(document_type)
    extracted_data["document_type"] = document_type

    try:
        model = schema_cls.model_validate(extracted_data)
        logger.info("validate_extraction_tool: valid=True type=%s", document_type)
        return {
            "valid": True,
            "errors": [],
            "data": model.model_dump(),
        }
    except ValidationError as exc:
        errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        logger.warning(
            "validate_extraction_tool: valid=False type=%s errors=%s",
            document_type, errors,
        )
        return {
            "valid": False,
            "errors": errors,
            "data": extracted_data,
        }


# ---------------------------------------------------------------------------
# Tool 4: notion_push_tool
# ---------------------------------------------------------------------------

@tool
def notion_push_tool(
    extracted_data: dict[str, Any],
    document_type: str,
) -> dict[str, Any]:
    """
    Push an extracted document to the correct Notion database.

    Delegates to notion.client.NotionClient which handles:
        - Routing to the correct database by document_type
        - Deduplication by document hash
        - Rate limiting (3 req/s with exponential backoff)

    Args:
        extracted_data: Validated extraction dict from extract_fields_tool.
        document_type:  One of 'job_description', 'resume', 'invoice'.

    Returns:
        Dict with keys:
            - success (bool)
            - notion_page_url (str or None)
            - action ('created' | 'updated' | 'skipped')
            - error (str or None)
    """
    # Import here to avoid circular imports — notion.client imports nothing from agent
    from notion.client import NotionClient  # noqa: PLC0415

    try:
        client = NotionClient()
        result = client.upsert_document(
            document_type=document_type,
            data=extracted_data,
        )
        logger.info(
            "notion_push_tool: action=%s url=%s",
            result.get("action"), result.get("notion_page_url"),
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("notion_push_tool failed: %s", exc)
        return {
            "success": False,
            "notion_page_url": None,
            "action": "failed",
            "error": str(exc),
        }
