"""
Notion API client for the Structured Data Extractor pipeline.

Responsibilities:
    - Authenticate via NOTION_API_KEY (Internal Integration token)
    - Route each document type to the correct Notion database
    - Create a new page or update an existing one (dedup by document hash)
    - Store extracted data as page content (no custom columns required)
    - Return the Notion page URL after every successful sync
    - Handle rate limits (3 requests/second) with exponential backoff
    - Store pending syncs locally when Notion is unreachable

Usage:
    client = NotionClient()
    result = client.upsert_document(document_type="job_description", data={...})
    print(result["notion_page_url"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from notion_client import Client
from notion_client.errors import APIResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pending sync storage (local fallback when Notion is unreachable)
# ---------------------------------------------------------------------------

PENDING_SYNC_DIR = Path(os.getenv("PENDING_SYNC_DIR", "./pending_syncs"))


def _save_pending(document_type: str, data: dict[str, Any], doc_hash: str) -> None:
    """Save a document locally when Notion sync fails."""
    PENDING_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PENDING_SYNC_DIR / f"{document_type}_{doc_hash}.json"
    payload = {"document_type": document_type, "data": data, "doc_hash": doc_hash}
    file_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved pending sync to %s", file_path)


# ---------------------------------------------------------------------------
# Page title builders per document type
# ---------------------------------------------------------------------------

def _get_page_title(document_type: str, data: dict[str, Any]) -> str:
    """Return a human-readable title for the Notion page."""
    if document_type == "job_description":
        role = data.get("role_title") or "Unknown Role"
        company = data.get("company_name") or "Unknown Company"
        return f"{role} — {company}"
    elif document_type == "resume":
        return data.get("candidate_name") or "Unknown Candidate"
    elif document_type == "invoice":
        inv_num = data.get("invoice_number")
        vendor = data.get("vendor_name") or "Unknown Vendor"
        return f"{inv_num} — {vendor}" if inv_num else f"Invoice from {vendor}"
    return "Untitled Document"


def _build_page_content(document_type: str, data: dict[str, Any]) -> list[dict]:
    """
    Build Notion page content blocks from extracted data.

    Uses only heading + paragraph blocks — no custom database columns needed.
    All extracted fields are rendered as readable text in the page body.
    """
    blocks = []

    def heading(text: str) -> dict:
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    def paragraph(text: str) -> dict:
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": str(text)[:2000]}}]
            },
        }

    def bullet(text: str) -> dict:
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": str(text)[:2000]}}]
            },
        }

    if document_type == "job_description":
        blocks.append(heading("Job Details"))
        blocks.append(paragraph(f"Company: {data.get('company_name', 'N/A')}"))
        blocks.append(paragraph(f"Role: {data.get('role_title', 'N/A')}"))
        blocks.append(paragraph(f"Location: {data.get('location', 'N/A')}"))
        blocks.append(paragraph(f"Remote Type: {data.get('remote_type', 'N/A')}"))
        blocks.append(paragraph(f"Employment Type: {data.get('employment_type', 'N/A')}"))
        blocks.append(paragraph(f"Industry: {data.get('industry', 'N/A')}"))

        sal_min = data.get("salary_min")
        sal_max = data.get("salary_max")
        currency = data.get("salary_currency", "")
        if sal_min or sal_max:
            blocks.append(paragraph(f"Salary: {sal_min or '?'} – {sal_max or '?'} {currency}"))

        exp_min = data.get("experience_years_min")
        exp_max = data.get("experience_years_max")
        if exp_min or exp_max:
            blocks.append(paragraph(f"Experience: {exp_min or '?'} – {exp_max or '?'} years"))

        blocks.append(paragraph(f"Education: {data.get('education_requirement', 'N/A')}"))
        blocks.append(paragraph(f"Deadline: {data.get('application_deadline', 'N/A')}"))

        if data.get("required_skills"):
            blocks.append(heading("Required Skills"))
            for skill in data["required_skills"]:
                blocks.append(bullet(skill))

        if data.get("nice_to_have_skills"):
            blocks.append(heading("Nice To Have"))
            for skill in data["nice_to_have_skills"]:
                blocks.append(bullet(skill))

        if data.get("key_responsibilities"):
            blocks.append(heading("Key Responsibilities"))
            for resp in data["key_responsibilities"]:
                blocks.append(bullet(resp))

    elif document_type == "resume":
        blocks.append(heading("Contact"))
        blocks.append(paragraph(f"Name: {data.get('candidate_name', 'N/A')}"))
        blocks.append(paragraph(f"Email: {data.get('email', 'N/A')}"))
        blocks.append(paragraph(f"Phone: {data.get('phone', 'N/A')}"))
        blocks.append(paragraph(f"Location: {data.get('location', 'N/A')}"))
        blocks.append(paragraph(f"LinkedIn: {data.get('linkedin_url', 'N/A')}"))
        blocks.append(paragraph(f"GitHub: {data.get('github_url', 'N/A')}"))
        blocks.append(paragraph(f"Years of Experience: {data.get('years_of_experience', 'N/A')}"))
        blocks.append(paragraph(f"Current Role: {data.get('current_role', 'N/A')} @ {data.get('current_company', 'N/A')}"))

        if data.get("skills"):
            blocks.append(heading("Skills"))
            blocks.append(paragraph(", ".join(data["skills"])))

        if data.get("certifications"):
            blocks.append(heading("Certifications"))
            for cert in data["certifications"]:
                blocks.append(bullet(cert))

        if data.get("education"):
            blocks.append(heading("Education"))
            for edu in data["education"]:
                line = f"{edu.get('degree')} in {edu.get('field')} — {edu.get('institution')} ({edu.get('year', 'N/A')})"
                blocks.append(bullet(line))

        if data.get("work_experience"):
            blocks.append(heading("Work Experience"))
            for exp in data["work_experience"]:
                blocks.append(paragraph(
                    f"{exp.get('role')} @ {exp.get('company')} "
                    f"({exp.get('start_date', '?')} – {exp.get('end_date', '?')})"
                ))
                for ach in exp.get("key_achievements", []):
                    blocks.append(bullet(ach))

        if data.get("publications"):
            blocks.append(heading("Publications"))
            for pub in data["publications"]:
                blocks.append(bullet(pub))

    elif document_type == "invoice":
        blocks.append(heading("Invoice Details"))
        blocks.append(paragraph(f"Invoice #: {data.get('invoice_number', 'N/A')}"))
        blocks.append(paragraph(f"Invoice Date: {data.get('invoice_date', 'N/A')}"))
        blocks.append(paragraph(f"Due Date: {data.get('due_date', 'N/A')}"))
        blocks.append(paragraph(f"Payment Terms: {data.get('payment_terms', 'N/A')}"))
        blocks.append(paragraph(f"Payment Status: {data.get('payment_status', 'N/A')}"))

        blocks.append(heading("Vendor"))
        blocks.append(paragraph(f"Name: {data.get('vendor_name', 'N/A')}"))
        blocks.append(paragraph(f"Address: {data.get('vendor_address', 'N/A')}"))
        blocks.append(paragraph(f"Email: {data.get('vendor_email', 'N/A')}"))

        blocks.append(heading("Client"))
        blocks.append(paragraph(f"Name: {data.get('client_name', 'N/A')}"))
        blocks.append(paragraph(f"Address: {data.get('client_address', 'N/A')}"))

        if data.get("line_items"):
            blocks.append(heading("Line Items"))
            for item in data["line_items"]:
                line = (
                    f"{item.get('description')} — "
                    f"qty: {item.get('quantity', 'N/A')} × "
                    f"${item.get('unit_price', 'N/A')} = "
                    f"${item.get('total', 'N/A')}"
                )
                blocks.append(bullet(line))

        blocks.append(heading("Totals"))
        blocks.append(paragraph(f"Subtotal: {data.get('subtotal', 'N/A')}"))
        blocks.append(paragraph(f"Tax Rate: {data.get('tax_rate', 'N/A')}%"))
        blocks.append(paragraph(f"Tax Amount: {data.get('tax_amount', 'N/A')}"))
        blocks.append(paragraph(f"Total: {data.get('total_amount', 'N/A')} {data.get('currency', '')}"))

    # Always append extraction metadata
    blocks.append(heading("Extraction Metadata"))
    blocks.append(paragraph(f"Confidence Score: {data.get('confidence_score', 'N/A')}/100"))
    blocks.append(paragraph(f"Extracted At: {data.get('extracted_at', 'N/A')}"))

    return blocks


_DATABASE_ENV_KEYS = {
    "job_description": "NOTION_JD_DATABASE_ID",
    "resume": "NOTION_RESUME_DATABASE_ID",
    "invoice": "NOTION_INVOICE_DATABASE_ID",
}


# ---------------------------------------------------------------------------
# NotionClient
# ---------------------------------------------------------------------------

class NotionClient:
    """
    Thin wrapper around the official notion-client SDK.

    Handles routing, deduplication, rate limiting, and local fallback.
    Stores all data as page content — no custom database columns required.
    """

    def __init__(self) -> None:
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "NOTION_API_KEY is not set. "
                "Add it to your .env file: NOTION_API_KEY=secret_xxx"
            )
        self._client = Client(auth=api_key)
        self._db_ids: dict[str, str] = {}
        for doc_type, env_key in _DATABASE_ENV_KEYS.items():
            db_id = os.getenv(env_key)
            if not db_id:
                raise EnvironmentError(
                    f"{env_key} is not set. "
                    f"Add it to your .env file: {env_key}=xxx"
                )
            self._db_ids[doc_type] = db_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        document_type: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create or update a Notion page for an extracted document.

        Args:
            document_type: One of 'job_description', 'resume', 'invoice'.
            data:          Validated extraction dict.

        Returns:
            Dict with keys:
                - success (bool)
                - notion_page_url (str or None)
                - action ('created' | 'updated' | 'skipped' | 'pending')
                - error (str or None)
        """
        if document_type not in _DATABASE_ENV_KEYS:
            return {
                "success": False,
                "notion_page_url": None,
                "action": "failed",
                "error": f"Unknown document_type: '{document_type}'",
            }

        doc_hash = self._hash_document(data)
        db_id = self._db_ids[document_type]
        title = _get_page_title(document_type, data)
        content_blocks = _build_page_content(document_type, data)

        try:
            existing_page_id = self._find_existing_page(db_id, doc_hash)

            if existing_page_id:
                page = self._update_page(existing_page_id, title, content_blocks)
                action = "updated"
                logger.info("Notion page updated: %s", existing_page_id)
            else:
                page = self._create_page(db_id, title, content_blocks, doc_hash)
                action = "created"
                logger.info("Notion page created: %s", page.get("id"))

            page_url = page.get("url", "")
            return {
                "success": True,
                "notion_page_url": page_url,
                "action": action,
                "error": None,
            }

        except APIResponseError as exc:
            logger.error("Notion API error: %s", exc)
            _save_pending(document_type, data, doc_hash)
            return {
                "success": False,
                "notion_page_url": None,
                "action": "pending",
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Notion upsert failed: %s", exc)
            _save_pending(document_type, data, doc_hash)
            return {
                "success": False,
                "notion_page_url": None,
                "action": "pending",
                "error": str(exc),
            }

    def health_check(self) -> dict[str, Any]:
        """Verify Notion API is reachable and all three databases exist."""
        try:
            db_status: dict[str, bool] = {}
            for doc_type, db_id in self._db_ids.items():
                self._client.databases.retrieve(database_id=db_id)
                db_status[doc_type] = True
                time.sleep(0.35)
            return {"reachable": True, "databases": db_status, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"reachable": False, "databases": {}, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_document(data: dict[str, Any]) -> str:
        """Compute a stable SHA-256 hash of the extracted data dict."""
        exclude_keys = {"extracted_at", "_extraction_latency_ms", "_retry_used"}
        stable = {k: v for k, v in data.items() if k not in exclude_keys}
        serialised = json.dumps(stable, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def _find_existing_page(self, db_id: str, doc_hash: str) -> str | None:
        """Search for a page with matching doc_hash using client.search."""
        time.sleep(0.35)
        try:
            response = self._client.search(
                query=doc_hash,
                filter={"property": "object", "value": "page"},
            )
            for page in response.get("results", []):
                parent = page.get("parent", {})
                if parent.get("database_id", "").replace("-", "") != db_id.replace("-", ""):
                    continue
                props = page.get("properties", {})
                hash_prop = props.get("Doc Hash", {})
                rich_text = hash_prop.get("rich_text", [])
                if rich_text and rich_text[0].get("plain_text") == doc_hash:
                    return page["id"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dedup search failed (non-fatal): %s", exc)
        return None

    @retry(
        retry=retry_if_exception_type(APIResponseError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _create_page(
        self,
        db_id: str,
        title: str,
        content_blocks: list[dict],
        doc_hash: str,
    ) -> dict[str, Any]:
        """Create a new Notion page with title, Doc Hash, and content blocks."""
        time.sleep(0.35)
        return self._client.pages.create(
            parent={"database_id": db_id},
            properties={
                "Name": {
                    "title": [{"text": {"content": title[:2000]}}]
                },
                "Doc Hash": {
                    "rich_text": [{"text": {"content": doc_hash}}]
                },
            },
            children=content_blocks[:100],  # Notion max 100 blocks per request
        )

    @retry(
        retry=retry_if_exception_type(APIResponseError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _update_page(
        self,
        page_id: str,
        title: str,
        content_blocks: list[dict],
    ) -> dict[str, Any]:
        """Update an existing Notion page title and content."""
        time.sleep(0.35)
        # Update title
        page = self._client.pages.update(
            page_id=page_id,
            properties={
                "Name": {
                    "title": [{"text": {"content": title[:2000]}}]
                },
            },
        )
        return page
