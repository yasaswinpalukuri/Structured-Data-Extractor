"""
Pydantic v2 structured output schemas for the Structured Data Extractor.

Three document schemas:
    - JobDescriptionSchema   : Job postings / JDs
    - ResumeSchema           : Candidate resumes / CVs
    - InvoiceSchema          : Vendor invoices / bills

All schemas share:
    - document_type  : Literal discriminator field
    - extracted_at   : ISO 8601 timestamp (auto-set on instantiation)
    - confidence_score: 0–100 LLM self-rating of extraction quality
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class BaseDocumentSchema(BaseModel):
    """Fields common to every extracted document."""

    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of extraction.",
    )
    confidence_score: Annotated[int, Field(ge=0, le=100)] = Field(
        ...,
        description="LLM self-rated confidence 0–100 for this extraction.",
    )

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }


# ---------------------------------------------------------------------------
# Schema 1: Job Description
# ---------------------------------------------------------------------------

class JobDescriptionSchema(BaseDocumentSchema):
    """
    Structured fields extracted from a job description / job posting.

    Required fields (must always be present):
        document_type, company_name, role_title, remote_type,
        employment_type, industry, required_skills, key_responsibilities,
        confidence_score
    """

    document_type: Literal["job_description"] = "job_description"

    # Core identity
    company_name: str = Field(..., description="Name of the hiring company.")
    role_title: str = Field(..., description="Job title / position name.")
    location: str | None = Field(None, description="City, region, or country of the role.")
    remote_type: Literal["Remote", "Hybrid", "On-site", "Not specified"] = Field(
        "Not specified",
        description="Work arrangement type.",
    )

    # Compensation
    salary_min: int | None = Field(None, description="Minimum salary (numeric only).")
    salary_max: int | None = Field(None, description="Maximum salary (numeric only).")
    salary_currency: str | None = Field(None, description="Currency code e.g. CAD, USD.")

    # Experience
    experience_years_min: int | None = Field(None, description="Minimum years of experience required.")
    experience_years_max: int | None = Field(None, description="Maximum years of experience mentioned.")

    # Skills
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills explicitly required. Empty list if none found.",
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Skills listed as preferred or nice-to-have.",
    )

    # Education & employment
    education_requirement: str | None = Field(
        None,
        description="Minimum education level required e.g. 'Bachelor's in Computer Science'.",
    )
    employment_type: Literal["Full-time", "Part-time", "Contract", "Not specified"] = Field(
        "Not specified",
        description="Employment arrangement.",
    )

    # Context
    industry: str = Field(..., description="Industry sector e.g. 'Technology', 'Finance'.")
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="Top responsibilities — maximum 5 bullet points.",
    )
    application_deadline: str | None = Field(
        None,
        description="Application deadline date string as it appears in the document.",
    )

    @field_validator("key_responsibilities")
    @classmethod
    def cap_responsibilities(cls, v: list[str]) -> list[str]:
        """Enforce maximum 5 key responsibilities."""
        return v[:5]

    @field_validator("salary_min", "salary_max")
    @classmethod
    def salary_must_be_positive(cls, v: int | None) -> int | None:
        """Salary values must be positive integers when present."""
        if v is not None and v < 0:
            raise ValueError("Salary values must be non-negative.")
        return v

    @model_validator(mode="after")
    def salary_range_valid(self) -> "JobDescriptionSchema":
        """salary_min must be <= salary_max when both are present."""
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError(
                    f"salary_min ({self.salary_min}) cannot exceed salary_max ({self.salary_max})."
                )
        return self


# ---------------------------------------------------------------------------
# Schema 2: Resume
# ---------------------------------------------------------------------------

class EducationEntry(BaseModel):
    """A single education entry on a resume."""

    degree: str = Field(..., description="Degree type e.g. 'Bachelor of Science'.")
    field: str = Field(..., description="Field of study e.g. 'Computer Science'.")
    institution: str = Field(..., description="Name of the university or college.")
    year: int | None = Field(None, description="Graduation year.")
    gpa: float | None = Field(None, description="GPA if listed.")

    @field_validator("gpa")
    @classmethod
    def gpa_range(cls, v: float | None) -> float | None:
        """GPA must be between 0.0 and 4.0 when present."""
        if v is not None and not (0.0 <= v <= 4.0):
            raise ValueError(f"GPA {v} is outside expected range 0.0–4.0.")
        return v


class WorkExperienceEntry(BaseModel):
    """A single work experience entry on a resume."""

    role: str = Field(..., description="Job title held.")
    company: str = Field(..., description="Company or organisation name.")
    start_date: str | None = Field(None, description="Start date as written e.g. 'Jan 2022'.")
    end_date: str | None = Field(None, description="End date as written, or 'Present'.")
    duration_months: int | None = Field(None, description="Approximate duration in months.")
    key_achievements: list[str] = Field(
        default_factory=list,
        description="Top achievements in this role — maximum 3.",
    )

    @field_validator("key_achievements")
    @classmethod
    def cap_achievements(cls, v: list[str]) -> list[str]:
        """Enforce maximum 3 key achievements per role."""
        return v[:3]


class ResumeSchema(BaseDocumentSchema):
    """
    Structured fields extracted from a candidate resume or CV.

    Required fields (must always be present):
        document_type, candidate_name, confidence_score
    """

    document_type: Literal["resume"] = "resume"

    # Identity
    candidate_name: str = Field(..., description="Full name of the candidate.")
    email: str | None = Field(None, description="Email address.")
    phone: str | None = Field(None, description="Phone number as written.")
    location: str | None = Field(None, description="City / region the candidate is based in.")
    linkedin_url: str | None = Field(None, description="LinkedIn profile URL.")
    github_url: str | None = Field(None, description="GitHub profile URL.")

    # Experience summary
    years_of_experience: float | None = Field(
        None,
        description="Total years of professional experience (calculated or stated).",
    )
    current_role: str | None = Field(None, description="Most recent or current job title.")
    current_company: str | None = Field(None, description="Most recent or current employer.")

    # Education
    education: list[EducationEntry] = Field(
        default_factory=list,
        description="List of education entries, most recent first.",
    )

    # Skills & credentials
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and soft skills listed on the resume.",
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Professional certifications e.g. 'AWS Solutions Architect'.",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Human languages spoken e.g. 'English', 'French'.",
    )
    publications: list[str] = Field(
        default_factory=list,
        description="Published papers, articles, or books.",
    )

    # Work history
    work_experience: list[WorkExperienceEntry] = Field(
        default_factory=list,
        description="Work history entries, most recent first.",
    )

    @field_validator("years_of_experience")
    @classmethod
    def experience_non_negative(cls, v: float | None) -> float | None:
        """Years of experience must be non-negative."""
        if v is not None and v < 0:
            raise ValueError("years_of_experience must be non-negative.")
        return v


# ---------------------------------------------------------------------------
# Schema 3: Invoice
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(..., description="Description of the product or service.")
    quantity: float | None = Field(None, description="Quantity of units.")
    unit_price: float | None = Field(None, description="Price per unit.")
    total: float | None = Field(None, description="Line total (quantity × unit_price).")


class InvoiceSchema(BaseDocumentSchema):
    """
    Structured fields extracted from a vendor invoice or bill.

    Required fields (must always be present):
        document_type, vendor_name, client_name,
        total_amount, currency, confidence_score
    """

    document_type: Literal["invoice"] = "invoice"

    # Invoice metadata
    invoice_number: str | None = Field(None, description="Invoice reference number.")
    invoice_date: str | None = Field(None, description="Date the invoice was issued.")
    due_date: str | None = Field(None, description="Payment due date.")

    # Vendor
    vendor_name: str = Field(..., description="Name of the vendor / seller.")
    vendor_address: str | None = Field(None, description="Vendor's mailing address.")
    vendor_email: str | None = Field(None, description="Vendor's email address.")

    # Client
    client_name: str = Field(..., description="Name of the client / buyer.")
    client_address: str | None = Field(None, description="Client's billing address.")

    # Line items
    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Individual line items on the invoice.",
    )

    # Totals
    subtotal: float | None = Field(None, description="Pre-tax subtotal.")
    tax_rate: float | None = Field(None, description="Tax rate as a percentage e.g. 13.0 for 13%.")
    tax_amount: float | None = Field(None, description="Calculated tax amount.")
    total_amount: float = Field(..., description="Final total amount due.")
    currency: str = Field(..., description="Currency code e.g. 'CAD', 'USD'.")

    # Payment
    payment_terms: str | None = Field(None, description="Payment terms e.g. 'Net 30'.")
    payment_status: Literal["Paid", "Unpaid", "Overdue", "Not specified"] = Field(
        "Not specified",
        description="Current payment status.",
    )

    @field_validator("total_amount")
    @classmethod
    def total_must_be_positive(cls, v: float) -> float:
        """Total amount must be a positive number."""
        if v < 0:
            raise ValueError("total_amount must be non-negative.")
        return v

    @field_validator("tax_rate")
    @classmethod
    def tax_rate_range(cls, v: float | None) -> float | None:
        """Tax rate must be between 0 and 100 when present."""
        if v is not None and not (0.0 <= v <= 100.0):
            raise ValueError(f"tax_rate {v} is outside expected range 0–100.")
        return v


# ---------------------------------------------------------------------------
# Union type — used by the extraction agent to return any schema
# ---------------------------------------------------------------------------

AnyDocumentSchema = JobDescriptionSchema | ResumeSchema | InvoiceSchema


def get_schema_for_type(document_type: str) -> type[AnyDocumentSchema]:
    """
    Return the correct Pydantic schema class for a given document type string.

    Args:
        document_type: One of 'job_description', 'resume', 'invoice'.

    Returns:
        The corresponding Pydantic model class.

    Raises:
        ValueError: If document_type is not recognised.
    """
    mapping: dict[str, type[AnyDocumentSchema]] = {
        "job_description": JobDescriptionSchema,
        "resume": ResumeSchema,
        "invoice": InvoiceSchema,
    }
    if document_type not in mapping:
        raise ValueError(
            f"Unknown document_type '{document_type}'. "
            f"Must be one of: {list(mapping.keys())}"
        )
    return mapping[document_type]
