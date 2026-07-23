"""
Tests for FastAPI endpoints in api/main.py

Uses TestClient — no live server or Ollama required.
Mocks ExtractionAgent to isolate HTTP layer from LLM/ChromaDB.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agent():
    """Return a fully mocked ExtractionAgent."""
    agent = MagicMock()

    # Default extract response
    agent.extract.return_value = {
        "document_type": "job_description",
        "classification_confidence": 95,
        "classification_method": "heuristic",
        "extracted_data": {
            "role_title": "Senior Data Engineer",
            "company_name": "TechNova Inc",
            "confidence_score": 85,
            "document_type": "job_description",
            "remote_type": "Remote",
            "employment_type": "Full-time",
            "industry": "Technology",
            "required_skills": ["Python", "SQL"],
            "key_responsibilities": ["Build pipelines"],
            "extracted_at": "2024-01-15T00:00:00+00:00",
        },
        "notion_page_url": "https://notion.so/test-page",
        "notion_action": "created",
        "memory_id": "test_session_jd_abc123",
        "total_latency_ms": 5000.0,
        "error": None,
    }

    # Default health response
    agent.health.return_value = {
        "ollama": True,
        "chromadb": True,
        "notion": True,
        "errors": [],
    }

    # Default chat response
    agent.chat.return_value = {
        "answer": "The salary range was $110,000-$140,000 CAD.",
        "context_docs_used": 3,
        "latency_ms": 1500.0,
    }

    # Mock ChromaDB collection
    agent._collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "ids": [[]],
    }
    agent._collection.count.return_value = 0
    agent._collection.get.return_value = {
        "ids": ["test_id"],
        "metadatas": [{
            "document_type": "job_description",
            "session_id": "test_session",
        }],
    }

    return agent


@pytest.fixture
def client(mock_agent):
    """Return a TestClient with mocked agent injected."""
    with patch("api.main._agent", mock_agent):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client, mock_agent):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client, mock_agent):
        data = client.get("/health").json()
        assert "status" in data
        assert "ollama" in data
        assert "chromadb" in data
        assert "notion" in data
        assert "version" in data

    def test_health_status_healthy(self, client, mock_agent):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_degraded_when_ollama_down(self, mock_agent):
        mock_agent.health.return_value = {
            "ollama": False,
            "chromadb": True,
            "notion": True,
            "errors": ["Ollama: connection refused"],
        }
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                data = c.get("/health").json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# POST /extract
# ---------------------------------------------------------------------------

class TestExtractAuto:
    def test_extract_returns_200(self, client):
        response = client.post("/extract", json={
            "text": "We are looking for a Senior Data Engineer. Qualifications: Python, SQL.",
            "session_id": "test",
            "sync_to_notion": False,
        })
        assert response.status_code == 200

    def test_extract_response_structure(self, client):
        data = client.post("/extract", json={
            "text": "Job posting for data engineer position",
            "session_id": "test",
        }).json()
        assert "document_type" in data
        assert "extracted_data" in data
        assert "error" in data

    def test_extract_no_error(self, client):
        data = client.post("/extract", json={
            "text": "Senior engineer job posting",
            "session_id": "test",
        }).json()
        assert data["error"] is None


# ---------------------------------------------------------------------------
# POST /extract/jd
# ---------------------------------------------------------------------------

class TestExtractJD:
    def test_extract_jd_returns_200(self, client):
        response = client.post("/extract/jd", json={
            "text": "Senior Data Engineer at TechNova. Requirements: Python, SQL.",
            "session_id": "test",
            "sync_to_notion": False,
        })
        assert response.status_code == 200

    def test_extract_jd_document_type(self, client):
        data = client.post("/extract/jd", json={
            "text": "Data engineer job",
            "session_id": "test",
        }).json()
        assert data["document_type"] == "job_description"

    def test_extract_jd_has_notion_url(self, mock_agent):
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                data = c.post("/extract/jd", json={
                    "text": "Data engineer job",
                    "session_id": "test",
                    "sync_to_notion": True,
                }).json()
        assert data["notion_page_url"] == "https://notion.so/test-page"


# ---------------------------------------------------------------------------
# POST /extract/resume
# ---------------------------------------------------------------------------

class TestExtractResume:
    def test_extract_resume_returns_200(self, client, mock_agent):
        mock_agent.extract.return_value["document_type"] = "resume"
        response = client.post("/extract/resume", json={
            "text": "John Doe. Skills: Python. Education: BSc CS.",
            "session_id": "test",
        })
        assert response.status_code == 200

    def test_extract_resume_calls_agent_with_type(self, mock_agent):
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                c.post("/extract/resume", json={
                    "text": "resume text here",
                    "session_id": "test",
                })
        call_kwargs = mock_agent.extract.call_args
        args, kwargs = call_kwargs
        doc_type = kwargs.get("document_type") or (args[1] if len(args) > 1 else None)
        assert doc_type == "resume"


# ---------------------------------------------------------------------------
# POST /extract/invoice
# ---------------------------------------------------------------------------

class TestExtractInvoice:
    def test_extract_invoice_returns_200(self, client, mock_agent):
        mock_agent.extract.return_value["document_type"] = "invoice"
        response = client.post("/extract/invoice", json={
            "text": "Invoice from DataSoft. Total: $1500.",
            "session_id": "test",
        })
        assert response.status_code == 200

    def test_extract_invoice_calls_agent_with_type(self, mock_agent):
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                c.post("/extract/invoice", json={
                    "text": "invoice text here",
                    "session_id": "test",
                })
        call_kwargs = mock_agent.extract.call_args
        args, kwargs = call_kwargs
        doc_type = kwargs.get("document_type") or (args[1] if len(args) > 1 else None)
        assert doc_type == "invoice"


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_chat_returns_200(self, client):
        response = client.post("/chat", json={
            "question": "What was the salary on the last JD?",
            "session_id": "test",
        })
        assert response.status_code == 200

    def test_chat_response_structure(self, client):
        data = client.post("/chat", json={
            "question": "What skills were required?",
            "session_id": "test",
        }).json()
        assert "answer" in data
        assert "context_docs_used" in data
        assert "latency_ms" in data

    def test_chat_returns_answer(self, client):
        data = client.post("/chat", json={
            "question": "What was the salary?",
            "session_id": "test",
        }).json()
        assert "salary" in data["answer"].lower()


# ---------------------------------------------------------------------------
# GET /notion/status/{doc_id}
# ---------------------------------------------------------------------------

class TestNotionStatus:
    def test_notion_status_returns_200(self, mock_agent):
        mock_agent._collection.get.return_value = {
            "ids": ["test_id"],
            "metadatas": [{"document_type": "job_description", "session_id": "test"}],
        }
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                response = c.get("/notion/status/test_id")
        assert response.status_code == 200

    def test_notion_status_response_structure(self, mock_agent):
        mock_agent._collection.get.return_value = {
            "ids": ["test_id"],
            "metadatas": [{"document_type": "job_description", "session_id": "test"}],
        }
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                data = c.get("/notion/status/test_id").json()
        assert "doc_id" in data
        assert "found" in data
        assert "document_type" in data

    def test_notion_status_not_found(self, mock_agent):
        mock_agent._collection.get.return_value = {"ids": [], "metadatas": []}
        with patch("api.main._agent", mock_agent):
            from api.main import app
            with TestClient(app) as c:
                response = c.get("/notion/status/nonexistent_id")
        assert response.status_code == 404
