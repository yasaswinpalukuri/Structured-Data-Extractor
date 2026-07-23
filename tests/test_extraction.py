"""
Tests for extraction tools in agent/tools.py

Uses mocked LLM responses — no Ollama required.
Covers:
    - classify_document_tool (heuristic path — no mock needed)
    - extract_fields_tool (mocked LLM)
    - validate_extraction_tool
    - Field normalisation (_normalise_jd_fields etc.)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import (
    classify_document_tool,
    extract_fields_tool,
    validate_extraction_tool,
    _normalise_jd_fields,
    _normalise_resume_fields,
    _normalise_invoice_fields,
    _strip_json_fences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response object."""
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# classify_document_tool
# ---------------------------------------------------------------------------

class TestClassifyDocumentTool:
    def test_jd_heuristic(self):
        text = "We are looking for a Senior Data Engineer. Responsibilities include building pipelines. Qualifications: 5 years experience. Apply now."
        result = classify_document_tool.invoke({"document_text": text})
        assert result["document_type"] == "job_description"
        assert result["method"] == "heuristic"
        assert result["confidence"] >= 60

    def test_resume_heuristic(self):
        text = "John Smith. Work Experience: Data Engineer at TechCo. Education: BSc Computer Science. Skills: Python, SQL. References available."
        result = classify_document_tool.invoke({"document_text": text})
        assert result["document_type"] == "resume"
        assert result["method"] == "heuristic"

    def test_invoice_heuristic(self):
        text = "Invoice #001. Bill To: Acme Corp. Total Amount: $500. Due Date: Feb 2025. Subtotal: $450. Tax: $50. Payment Terms: Net 30."
        result = classify_document_tool.invoke({"document_text": text})
        assert result["document_type"] == "invoice"
        assert result["method"] == "heuristic"

    def test_empty_text_returns_unknown(self):
        result = classify_document_tool.invoke({"document_text": ""})
        assert result["document_type"] == "unknown"

    def test_result_has_required_keys(self):
        result = classify_document_tool.invoke({"document_text": "invoice total amount due"})
        assert "document_type" in result
        assert "confidence" in result
        assert "method" in result
        assert "latency_ms" in result


# ---------------------------------------------------------------------------
# extract_fields_tool — mocked LLM
# ---------------------------------------------------------------------------

class TestExtractFieldsTool:
    JD_VALID_RESPONSE = json.dumps({
        "document_type": "job_description",
        "company_name": "TechNova Inc",
        "role_title": "Senior Data Engineer",
        "location": "Toronto, ON",
        "remote_type": "Hybrid",
        "salary_min": 110000,
        "salary_max": 140000,
        "salary_currency": "CAD",
        "experience_years_min": 5,
        "experience_years_max": None,
        "required_skills": ["Python", "SQL", "Spark"],
        "nice_to_have_skills": ["dbt"],
        "education_requirement": "Bachelor in CS",
        "employment_type": "Full-time",
        "industry": "Technology",
        "key_responsibilities": ["Build pipelines"],
        "application_deadline": "March 31, 2025",
        "confidence_score": 88,
    })

    RESUME_VALID_RESPONSE = json.dumps({
        "document_type": "resume",
        "candidate_name": "Yasaswin Palukuri",
        "email": "yash@email.com",
        "phone": None,
        "location": "Toronto, ON",
        "linkedin_url": None,
        "github_url": None,
        "years_of_experience": 2.0,
        "current_role": "Data Engineer",
        "current_company": "Lambton College",
        "education": [],
        "skills": ["Python", "SQL"],
        "certifications": [],
        "languages": [],
        "publications": [],
        "work_experience": [],
        "confidence_score": 85,
    })

    INVOICE_VALID_RESPONSE = json.dumps({
        "document_type": "invoice",
        "invoice_number": "INV-001",
        "invoice_date": "Jan 15, 2024",
        "due_date": "Feb 14, 2024",
        "vendor_name": "DataSoft Inc",
        "vendor_address": None,
        "vendor_email": None,
        "client_name": "Acme Corp",
        "client_address": None,
        "line_items": [],
        "subtotal": None,
        "tax_rate": None,
        "tax_amount": None,
        "total_amount": 1500.0,
        "currency": "CAD",
        "payment_terms": "Net 30",
        "payment_status": "Unpaid",
        "confidence_score": 90,
    })

    @patch("agent.tools._get_llm")
    def test_extract_jd_success(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(self.JD_VALID_RESPONSE)
        mock_get_llm.return_value = mock_llm

        result = extract_fields_tool.invoke({
            "document_text": "Senior Data Engineer job at TechNova Inc",
            "document_type": "job_description",
        })
        assert result["role_title"] == "Senior Data Engineer"
        assert result["company_name"] == "TechNova Inc"
        assert result["confidence_score"] == 88
        assert "_extraction_latency_ms" in result

    @patch("agent.tools._get_llm")
    def test_extract_resume_success(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(self.RESUME_VALID_RESPONSE)
        mock_get_llm.return_value = mock_llm

        result = extract_fields_tool.invoke({
            "document_text": "Yasaswin Palukuri resume with Python SQL skills",
            "document_type": "resume",
        })
        assert result["candidate_name"] == "Yasaswin Palukuri"
        assert "Python" in result["skills"]

    @patch("agent.tools._get_llm")
    def test_extract_invoice_success(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(self.INVOICE_VALID_RESPONSE)
        mock_get_llm.return_value = mock_llm

        result = extract_fields_tool.invoke({
            "document_text": "Invoice from DataSoft Inc to Acme Corp total $1500",
            "document_type": "invoice",
        })
        assert result["vendor_name"] == "DataSoft Inc"
        assert result["total_amount"] == 1500.0

    @patch("agent.tools._get_llm")
    def test_retry_on_invalid_json(self, mock_get_llm):
        """First response is invalid JSON, second is valid — retry should succeed."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _make_llm_response("This is not JSON at all"),
            _make_llm_response(self.JD_VALID_RESPONSE),
        ]
        mock_get_llm.return_value = mock_llm

        result = extract_fields_tool.invoke({
            "document_text": "Senior Data Engineer at TechNova",
            "document_type": "job_description",
        })
        assert result["role_title"] == "Senior Data Engineer"
        assert result.get("_retry_used") is True

    @patch("agent.tools._get_llm")
    def test_both_attempts_fail_raises(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("not json")
        mock_get_llm.return_value = mock_llm

        with pytest.raises(ValueError, match="Extraction failed after 2 attempts"):
            extract_fields_tool.invoke({
                "document_text": "some document text here",
                "document_type": "job_description",
            })

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            extract_fields_tool.invoke({
                "document_text": "some text",
                "document_type": "unknown",
            })


# ---------------------------------------------------------------------------
# validate_extraction_tool
# ---------------------------------------------------------------------------

class TestValidateExtractionTool:
    def test_valid_jd(self):
        data = {
            "company_name": "Acme",
            "role_title": "Engineer",
            "remote_type": "Remote",
            "employment_type": "Full-time",
            "industry": "Tech",
            "required_skills": ["Python"],
            "key_responsibilities": ["Build stuff"],
            "confidence_score": 80,
        }
        result = validate_extraction_tool.invoke({
            "extracted_data": data,
            "document_type": "job_description",
        })
        assert result["valid"] is True
        assert result["errors"] == []

    def test_invalid_jd_missing_required(self):
        result = validate_extraction_tool.invoke({
            "extracted_data": {"confidence_score": 80},
            "document_type": "job_description",
        })
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_valid_resume(self):
        result = validate_extraction_tool.invoke({
            "extracted_data": {
                "candidate_name": "Jane Doe",
                "confidence_score": 75,
            },
            "document_type": "resume",
        })
        assert result["valid"] is True

    def test_valid_invoice(self):
        result = validate_extraction_tool.invoke({
            "extracted_data": {
                "vendor_name": "Co",
                "client_name": "Corp",
                "total_amount": 500.0,
                "currency": "USD",
                "confidence_score": 80,
            },
            "document_type": "invoice",
        })
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Field normalisers
# ---------------------------------------------------------------------------

class TestFieldNormalisers:
    def test_jd_job_title_mapped(self):
        data = {"job_title": "Engineer", "industry": "Tech", "company_name": "Co", "confidence_score": 80}
        result = _normalise_jd_fields(data)
        assert result["role_title"] == "Engineer"
        assert "job_title" not in result

    def test_jd_salary_range_split(self):
        data = {"salary_range": [90000, 120000], "industry": "Tech", "company_name": "Co",
                "role_title": "Engineer", "confidence_score": 80}
        result = _normalise_jd_fields(data)
        assert result["salary_min"] == 90000
        assert result["salary_max"] == 120000

    def test_jd_currency_mapped(self):
        data = {"currency": "CAD", "industry": "Tech", "company_name": "Co",
                "role_title": "Engineer", "confidence_score": 80}
        result = _normalise_jd_fields(data)
        assert result["salary_currency"] == "CAD"

    def test_jd_remote_type_normalised(self):
        data = {"remote_type": "onsite", "industry": "Tech", "company_name": "Co",
                "role_title": "Engineer", "confidence_score": 80}
        result = _normalise_jd_fields(data)
        assert result["remote_type"] == "On-site"

    def test_jd_industry_fallback(self):
        data = {"role_title": "Engineer", "company_name": "Co", "confidence_score": 80}
        result = _normalise_jd_fields(data)
        assert result["industry"] == "Not specified"

    def test_resume_name_mapped(self):
        data = {"full_name": "Jane Doe", "confidence_score": 75}
        result = _normalise_resume_fields(data)
        assert result["candidate_name"] == "Jane Doe"

    def test_invoice_bill_to_mapped(self):
        data = {"bill_to": "Acme Corp", "vendor_name": "Co",
                "total_amount": 500.0, "currency": "USD", "confidence_score": 80}
        result = _normalise_invoice_fields(data)
        assert result["client_name"] == "Acme Corp"

    def test_invoice_total_fallback(self):
        data = {"vendor_name": "Co", "client_name": "Corp", "currency": "USD", "confidence_score": 80}
        result = _normalise_invoice_fields(data)
        assert result["total_amount"] == 0.0


# ---------------------------------------------------------------------------
# _strip_json_fences
# ---------------------------------------------------------------------------

class TestStripJsonFences:
    def test_strips_json_fence(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        result = _strip_json_fences(text)
        assert result == '{"key": "value"}'

    def test_strips_plain_fence(self):
        text = "```\n{\"key\": \"value\"}\n```"
        result = _strip_json_fences(text)
        assert result == '{"key": "value"}'

    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        result = _strip_json_fences(text)
        assert result == text
