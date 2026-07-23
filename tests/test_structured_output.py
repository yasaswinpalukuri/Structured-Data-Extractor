"""
Tests for Pydantic v2 schemas in agent/structured_output.py

Covers:
    - Valid construction of all three schemas
    - Field validators (salary range, GPA, tax rate, etc.)
    - get_schema_for_type() routing
    - Edge cases: null fields, empty lists, boundary values
"""

import pytest
from pydantic import ValidationError

from agent.structured_output import (
    EducationEntry,
    InvoiceSchema,
    JobDescriptionSchema,
    LineItem,
    ResumeSchema,
    WorkExperienceEntry,
    get_schema_for_type,
)


# ---------------------------------------------------------------------------
# JobDescriptionSchema
# ---------------------------------------------------------------------------

class TestJobDescriptionSchema:
    def _minimal(self, **kwargs) -> dict:
        base = {
            "company_name": "Acme Corp",
            "role_title": "Data Engineer",
            "remote_type": "Remote",
            "employment_type": "Full-time",
            "industry": "Technology",
            "required_skills": ["Python"],
            "key_responsibilities": ["Build pipelines"],
            "confidence_score": 80,
        }
        base.update(kwargs)
        return base

    def test_valid_minimal(self):
        jd = JobDescriptionSchema(**self._minimal())
        assert jd.role_title == "Data Engineer"
        assert jd.document_type == "job_description"

    def test_valid_full(self):
        jd = JobDescriptionSchema(**self._minimal(
            location="Toronto, ON",
            salary_min=90000,
            salary_max=120000,
            salary_currency="CAD",
            experience_years_min=3,
            experience_years_max=7,
            nice_to_have_skills=["dbt", "Airflow"],
            education_requirement="Bachelor in CS",
            application_deadline="March 31, 2025",
        ))
        assert jd.salary_min == 90000
        assert jd.salary_max == 120000
        assert len(jd.nice_to_have_skills) == 2

    def test_salary_range_invalid(self):
        with pytest.raises(ValidationError):
            JobDescriptionSchema(**self._minimal(salary_min=150000, salary_max=90000))

    def test_salary_negative(self):
        with pytest.raises(ValidationError):
            JobDescriptionSchema(**self._minimal(salary_min=-1000))

    def test_key_responsibilities_capped_at_5(self):
        jd = JobDescriptionSchema(**self._minimal(
            key_responsibilities=["A", "B", "C", "D", "E", "F", "G"]
        ))
        assert len(jd.key_responsibilities) == 5

    def test_confidence_score_bounds(self):
        with pytest.raises(ValidationError):
            JobDescriptionSchema(**self._minimal(confidence_score=101))
        with pytest.raises(ValidationError):
            JobDescriptionSchema(**self._minimal(confidence_score=-1))

    def test_remote_type_default(self):
        jd = JobDescriptionSchema(**self._minimal())
        assert jd.remote_type == "Remote"

    def test_null_optional_fields(self):
        jd = JobDescriptionSchema(**self._minimal())
        assert jd.location is None
        assert jd.salary_min is None
        assert jd.application_deadline is None

    def test_extracted_at_auto_set(self):
        jd = JobDescriptionSchema(**self._minimal())
        assert jd.extracted_at is not None
        assert "T" in jd.extracted_at  # ISO format


# ---------------------------------------------------------------------------
# ResumeSchema
# ---------------------------------------------------------------------------

class TestResumeSchema:
    def _minimal(self, **kwargs) -> dict:
        base = {
            "candidate_name": "Jane Doe",
            "confidence_score": 75,
        }
        base.update(kwargs)
        return base

    def test_valid_minimal(self):
        r = ResumeSchema(**self._minimal())
        assert r.candidate_name == "Jane Doe"
        assert r.document_type == "resume"

    def test_valid_full(self):
        r = ResumeSchema(**self._minimal(
            email="jane@email.com",
            phone="+1-416-555-0000",
            location="Toronto, ON",
            linkedin_url="linkedin.com/in/jane",
            github_url="github.com/jane",
            years_of_experience=3.5,
            current_role="Data Engineer",
            current_company="TechCo",
            skills=["Python", "SQL"],
            certifications=["AWS SAA"],
            languages=["English", "French"],
            education=[{
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "institution": "University of Toronto",
                "year": 2021,
                "gpa": 3.7,
            }],
            work_experience=[{
                "role": "Data Engineer",
                "company": "TechCo",
                "start_date": "Jan 2022",
                "end_date": "Present",
                "duration_months": 24,
                "key_achievements": ["Built pipelines", "Reduced latency by 30%"],
            }],
        ))
        assert r.years_of_experience == 3.5
        assert len(r.education) == 1
        assert r.education[0].gpa == 3.7

    def test_gpa_out_of_range(self):
        with pytest.raises(ValidationError):
            ResumeSchema(**self._minimal(education=[{
                "degree": "BSc",
                "field": "CS",
                "institution": "UofT",
                "gpa": 5.0,
            }]))

    def test_key_achievements_capped_at_3(self):
        r = ResumeSchema(**self._minimal(work_experience=[{
            "role": "Engineer",
            "company": "Co",
            "key_achievements": ["A", "B", "C", "D", "E"],
        }]))
        assert len(r.work_experience[0].key_achievements) == 3

    def test_negative_experience(self):
        with pytest.raises(ValidationError):
            ResumeSchema(**self._minimal(years_of_experience=-1.0))

    def test_empty_lists_default(self):
        r = ResumeSchema(**self._minimal())
        assert r.skills == []
        assert r.certifications == []
        assert r.publications == []


# ---------------------------------------------------------------------------
# InvoiceSchema
# ---------------------------------------------------------------------------

class TestInvoiceSchema:
    def _minimal(self, **kwargs) -> dict:
        base = {
            "vendor_name": "DataSoft Inc",
            "client_name": "Acme Corp",
            "total_amount": 1500.0,
            "currency": "CAD",
            "confidence_score": 85,
        }
        base.update(kwargs)
        return base

    def test_valid_minimal(self):
        inv = InvoiceSchema(**self._minimal())
        assert inv.vendor_name == "DataSoft Inc"
        assert inv.document_type == "invoice"

    def test_valid_full(self):
        inv = InvoiceSchema(**self._minimal(
            invoice_number="INV-2024-001",
            invoice_date="January 15, 2024",
            due_date="February 14, 2024",
            subtotal=1300.0,
            tax_rate=13.0,
            tax_amount=169.0,
            payment_terms="Net 30",
            payment_status="Unpaid",
            line_items=[{
                "description": "Consulting",
                "quantity": 10.0,
                "unit_price": 130.0,
                "total": 1300.0,
            }],
        ))
        assert inv.invoice_number == "INV-2024-001"
        assert inv.tax_rate == 13.0
        assert len(inv.line_items) == 1

    def test_negative_total(self):
        with pytest.raises(ValidationError):
            InvoiceSchema(**self._minimal(total_amount=-100.0))

    def test_tax_rate_out_of_range(self):
        with pytest.raises(ValidationError):
            InvoiceSchema(**self._minimal(tax_rate=150.0))

    def test_payment_status_default(self):
        inv = InvoiceSchema(**self._minimal())
        assert inv.payment_status == "Not specified"

    def test_empty_line_items_default(self):
        inv = InvoiceSchema(**self._minimal())
        assert inv.line_items == []


# ---------------------------------------------------------------------------
# get_schema_for_type
# ---------------------------------------------------------------------------

class TestGetSchemaForType:
    def test_job_description(self):
        cls = get_schema_for_type("job_description")
        assert cls is JobDescriptionSchema

    def test_resume(self):
        cls = get_schema_for_type("resume")
        assert cls is ResumeSchema

    def test_invoice(self):
        cls = get_schema_for_type("invoice")
        assert cls is InvoiceSchema

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown document_type"):
            get_schema_for_type("unknown")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            get_schema_for_type("")
