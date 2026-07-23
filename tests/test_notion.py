"""
Tests for Notion client in notion/client.py

Uses mocked notion-client SDK — no real Notion API calls.
Covers:
    - Page title generation per document type
    - Page content block building
    - Document hash computation
    - Deduplication logic
    - Pending sync fallback
    - Health check
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from notion.client import (
    NotionClient,
    _get_page_title,
    _build_page_content,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_test_key")
    monkeypatch.setenv("NOTION_JD_DATABASE_ID", "jd_db_id_123")
    monkeypatch.setenv("NOTION_RESUME_DATABASE_ID", "resume_db_id_456")
    monkeypatch.setenv("NOTION_INVOICE_DATABASE_ID", "invoice_db_id_789")


@pytest.fixture
def mock_notion_client():
    """Return a mocked notion Client instance."""
    with patch("notion.client.Client") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        # Mock successful page creation
        mock_instance.pages.create.return_value = {
            "id": "new-page-id-123",
            "url": "https://notion.so/new-page-id-123",
        }

        # Mock successful page update
        mock_instance.pages.update.return_value = {
            "id": "existing-page-id",
            "url": "https://notion.so/existing-page-id",
        }

        # Mock database retrieve (health check)
        mock_instance.databases.retrieve.return_value = {"id": "db_id"}

        # Mock search (no existing pages by default)
        mock_instance.search.return_value = {"results": []}

        yield mock_instance


@pytest.fixture
def notion_client(mock_notion_client):
    """Return a NotionClient with mocked SDK."""
    return NotionClient()


# ---------------------------------------------------------------------------
# _get_page_title
# ---------------------------------------------------------------------------

class TestGetPageTitle:
    def test_jd_title(self):
        data = {"role_title": "Senior Data Engineer", "company_name": "TechNova"}
        title = _get_page_title("job_description", data)
        assert "Senior Data Engineer" in title
        assert "TechNova" in title

    def test_jd_title_missing_fields(self):
        title = _get_page_title("job_description", {})
        assert title == "Unknown Role — Unknown Company"

    def test_resume_title(self):
        data = {"candidate_name": "Yasaswin Palukuri"}
        title = _get_page_title("resume", data)
        assert title == "Yasaswin Palukuri"

    def test_resume_title_missing(self):
        title = _get_page_title("resume", {})
        assert title == "Unknown Candidate"

    def test_invoice_title_with_number(self):
        data = {"invoice_number": "INV-001", "vendor_name": "DataSoft"}
        title = _get_page_title("invoice", data)
        assert "INV-001" in title
        assert "DataSoft" in title

    def test_invoice_title_without_number(self):
        data = {"vendor_name": "DataSoft"}
        title = _get_page_title("invoice", data)
        assert "DataSoft" in title


# ---------------------------------------------------------------------------
# _build_page_content
# ---------------------------------------------------------------------------

class TestBuildPageContent:
    def test_jd_content_has_blocks(self):
        data = {
            "role_title": "Engineer",
            "company_name": "Co",
            "remote_type": "Remote",
            "employment_type": "Full-time",
            "industry": "Tech",
            "required_skills": ["Python", "SQL"],
            "key_responsibilities": ["Build pipelines"],
            "confidence_score": 85,
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        blocks = _build_page_content("job_description", data)
        assert len(blocks) > 0
        block_types = {b["type"] for b in blocks}
        assert "heading_2" in block_types
        assert "paragraph" in block_types

    def test_jd_content_includes_skills(self):
        data = {
            "required_skills": ["Python", "SQL", "Spark"],
            "confidence_score": 80,
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        blocks = _build_page_content("job_description", data)
        all_text = " ".join(
            b.get("bulleted_list_item", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
            for b in blocks if b["type"] == "bulleted_list_item"
        )
        assert "Python" in all_text

    def test_resume_content_has_blocks(self):
        data = {
            "candidate_name": "Jane",
            "email": "jane@email.com",
            "skills": ["Python"],
            "confidence_score": 80,
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        blocks = _build_page_content("resume", data)
        assert len(blocks) > 0

    def test_invoice_content_has_blocks(self):
        data = {
            "invoice_number": "INV-001",
            "vendor_name": "Co",
            "client_name": "Corp",
            "total_amount": 1500.0,
            "currency": "CAD",
            "confidence_score": 90,
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        blocks = _build_page_content("invoice", data)
        assert len(blocks) > 0

    def test_always_includes_metadata_section(self):
        blocks = _build_page_content("resume", {"confidence_score": 80, "extracted_at": "x"})
        headings = [
            b["heading_2"]["rich_text"][0]["text"]["content"]
            for b in blocks if b["type"] == "heading_2"
        ]
        assert "Extraction Metadata" in headings


# ---------------------------------------------------------------------------
# NotionClient.upsert_document
# ---------------------------------------------------------------------------

class TestUpsertDocument:
    JD_DATA = {
        "role_title": "Senior Data Engineer",
        "company_name": "TechNova",
        "confidence_score": 85,
        "extracted_at": "2024-01-01T00:00:00Z",
    }

    def test_creates_new_page(self, notion_client, mock_notion_client):
        result = notion_client.upsert_document("job_description", self.JD_DATA)
        assert result["success"] is True
        assert result["action"] == "created"
        assert result["notion_page_url"] is not None
        mock_notion_client.pages.create.assert_called_once()

    def test_updates_existing_page(self, notion_client, mock_notion_client):
        # Simulate existing page found
        mock_notion_client.search.return_value = {
            "results": [{
                "id": "existing-page-id",
                "parent": {"database_id": "jd_db_id_123"},
                "properties": {
                    "Doc Hash": {
                        "rich_text": [{"plain_text": notion_client._hash_document(self.JD_DATA)}]
                    }
                }
            }]
        }
        result = notion_client.upsert_document("job_description", self.JD_DATA)
        assert result["action"] == "updated"
        mock_notion_client.pages.update.assert_called_once()

    def test_unknown_document_type_fails(self, notion_client):
        result = notion_client.upsert_document("unknown_type", {})
        assert result["success"] is False
        assert "Unknown document_type" in result["error"]

    def test_pending_on_api_error(self, notion_client, mock_notion_client, tmp_path, monkeypatch):
        # Raise a generic exception to simulate API failure (avoids APIResponseError constructor changes)
        mock_notion_client.pages.create.side_effect = Exception("Simulated Notion API failure")
        monkeypatch.setattr("notion.client.PENDING_SYNC_DIR", tmp_path)
        result = notion_client.upsert_document("job_description", self.JD_DATA)
        assert result["action"] == "pending"
        assert result["success"] is False


# ---------------------------------------------------------------------------
# NotionClient._hash_document
# ---------------------------------------------------------------------------

class TestHashDocument:
    def test_same_data_same_hash(self, notion_client):
        data = {"role_title": "Engineer", "company_name": "Co"}
        assert notion_client._hash_document(data) == notion_client._hash_document(data)

    def test_different_data_different_hash(self, notion_client):
        data1 = {"role_title": "Engineer"}
        data2 = {"role_title": "Manager"}
        assert notion_client._hash_document(data1) != notion_client._hash_document(data2)

    def test_extracted_at_excluded(self, notion_client):
        data1 = {"role_title": "Engineer", "extracted_at": "2024-01-01"}
        data2 = {"role_title": "Engineer", "extracted_at": "2025-06-15"}
        assert notion_client._hash_document(data1) == notion_client._hash_document(data2)

    def test_hash_is_16_chars(self, notion_client):
        h = notion_client._hash_document({"key": "value"})
        assert len(h) == 16


# ---------------------------------------------------------------------------
# NotionClient.health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_passes(self, notion_client, mock_notion_client):
        result = notion_client.health_check()
        assert result["reachable"] is True
        assert len(result["databases"]) == 3

    def test_health_check_fails_gracefully(self, notion_client, mock_notion_client):
        mock_notion_client.databases.retrieve.side_effect = Exception("Network error")
        result = notion_client.health_check()
        assert result["reachable"] is False
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Missing env var
# ---------------------------------------------------------------------------

class TestMissingEnvVars:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NOTION_API_KEY", raising=False)
        with patch("notion.client.Client"):
            with pytest.raises(EnvironmentError, match="NOTION_API_KEY"):
                NotionClient()

    def test_missing_db_id_raises(self, monkeypatch):
        monkeypatch.delenv("NOTION_JD_DATABASE_ID", raising=False)
        with patch("notion.client.Client"):
            with pytest.raises(EnvironmentError, match="NOTION_JD_DATABASE_ID"):
                NotionClient()
