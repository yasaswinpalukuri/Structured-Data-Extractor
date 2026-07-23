"""
Extraction prompts for the Structured Data Extractor pipeline.

One prompt template per document type, each with:
    - Clear extraction instructions
    - Field-by-field guidance
    - Few-shot example (input → expected JSON output)
    - Strict JSON-only output instruction

Usage:
    from agent.prompts import get_prompt_for_type
    prompt = get_prompt_for_type("job_description")
    filled = prompt.format(document_text="...")
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System instruction shared across all prompts
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are a precise document data extraction engine.
Your only job is to extract structured fields from documents and return valid JSON.

Rules you must never break:
1. Return ONLY a valid JSON object — no explanation, no markdown, no code fences.
2. If a field is not present in the document, use null (not "N/A", not "unknown", not "").
3. All list fields default to [] when nothing is found — never null for lists.
4. confidence_score: rate your own extraction quality 0–100.
   - 80–100 : all key fields found, high certainty
   - 50–79  : some fields missing or ambiguous
   - 0–49   : significant fields missing or document is unclear
5. Never invent or guess values. Only extract what is explicitly stated.
6. Dates: preserve the exact string as written in the document.
7. Salary: extract numeric values only — strip currency symbols and commas.
"""

# ---------------------------------------------------------------------------
# Prompt 1: Job Description
# ---------------------------------------------------------------------------

JD_EXTRACTION_PROMPT = """\
Extract structured fields from the job description below.

Return a JSON object with exactly these fields:
{{
  "document_type": "job_description",
  "company_name": "string — hiring company name",
  "role_title": "string — exact job title",
  "location": "string or null — city, region, country",
  "remote_type": "Remote | Hybrid | On-site | Not specified",
  "salary_min": "integer or null — minimum salary, numbers only",
  "salary_max": "integer or null — maximum salary, numbers only",
  "salary_currency": "string or null — e.g. CAD, USD, GBP",
  "experience_years_min": "integer or null",
  "experience_years_max": "integer or null",
  "required_skills": ["list of strings — explicitly required skills"],
  "nice_to_have_skills": ["list of strings — preferred or nice-to-have"],
  "education_requirement": "string or null — e.g. Bachelor in Computer Science",
  "employment_type": "Full-time | Part-time | Contract | Not specified",
  "industry": "string — industry sector",
  "key_responsibilities": ["list of strings — max 5, most important duties"],
  "application_deadline": "string or null — deadline as written",
  "extracted_at": "leave this field out — it is auto-generated",
  "confidence_score": "integer 0–100"
}}

--- FEW-SHOT EXAMPLE ---

Input document:
\"\"\"
Senior Data Engineer — Toronto, ON (Hybrid)
TechNova Inc | Full-time | $110,000 – $140,000 CAD

About the Role:
We are looking for a Senior Data Engineer to join our growing data platform team.
You will design and maintain scalable ETL pipelines and work closely with ML teams.

Requirements:
- 5+ years of experience in data engineering
- Proficiency in Python, SQL, and Apache Spark
- Experience with AWS (S3, Glue, Redshift)
- Bachelor's degree in Computer Science or related field

Nice to Have:
- Experience with dbt or Airflow
- Familiarity with Kubernetes

Responsibilities:
- Build and maintain batch and streaming data pipelines
- Collaborate with data scientists to deploy ML models
- Optimize query performance on Redshift
- Document data flows and maintain data catalog

Apply by: March 31, 2025
TechNova is an equal opportunity employer.
\"\"\"

Expected output:
{{
  "document_type": "job_description",
  "company_name": "TechNova Inc",
  "role_title": "Senior Data Engineer",
  "location": "Toronto, ON",
  "remote_type": "Hybrid",
  "salary_min": 110000,
  "salary_max": 140000,
  "salary_currency": "CAD",
  "experience_years_min": 5,
  "experience_years_max": null,
  "required_skills": ["Python", "SQL", "Apache Spark", "AWS", "S3", "Glue", "Redshift"],
  "nice_to_have_skills": ["dbt", "Airflow", "Kubernetes"],
  "education_requirement": "Bachelor's degree in Computer Science or related field",
  "employment_type": "Full-time",
  "industry": "Technology",
  "key_responsibilities": [
    "Build and maintain batch and streaming data pipelines",
    "Collaborate with data scientists to deploy ML models",
    "Optimize query performance on Redshift",
    "Document data flows and maintain data catalog"
  ],
  "application_deadline": "March 31, 2025",
  "confidence_score": 95
}}

--- END EXAMPLE ---

Now extract from this document:
\"\"\"
{document_text}
\"\"\"

JSON output:"""


# ---------------------------------------------------------------------------
# Prompt 2: Resume
# ---------------------------------------------------------------------------

RESUME_EXTRACTION_PROMPT = """\
Extract structured fields from the resume or CV below.

Return a JSON object with exactly these fields:
{{
  "document_type": "resume",
  "candidate_name": "string — full name",
  "email": "string or null",
  "phone": "string or null — as written",
  "location": "string or null — city or region",
  "linkedin_url": "string or null",
  "github_url": "string or null",
  "years_of_experience": "float or null — total years",
  "current_role": "string or null — most recent job title",
  "current_company": "string or null — most recent employer",
  "education": [
    {{
      "degree": "string",
      "field": "string",
      "institution": "string",
      "year": "integer or null",
      "gpa": "float or null"
    }}
  ],
  "skills": ["list of strings"],
  "certifications": ["list of strings"],
  "languages": ["list of strings — human languages only"],
  "publications": ["list of strings"],
  "work_experience": [
    {{
      "role": "string",
      "company": "string",
      "start_date": "string or null",
      "end_date": "string or null",
      "duration_months": "integer or null",
      "key_achievements": ["list of strings — max 3"]
    }}
  ],
  "extracted_at": "leave this field out — it is auto-generated",
  "confidence_score": "integer 0–100"
}}

--- FEW-SHOT EXAMPLE ---

Input document:
\"\"\"
Yasaswin Palukuri
Toronto, ON | yasaswin@email.com | +1-647-555-0123
linkedin.com/in/yasaswinpalukuri | github.com/yasaswinpalukuri

SUMMARY
AI/ML Engineer with 2+ years of experience building production data pipelines and LLM applications.

SKILLS
Python, SQL, PySpark, AWS (S3, Glue, Lambda), Docker, LangChain, FastAPI, ChromaDB, TensorFlow

CERTIFICATIONS
AZ-400 Microsoft DevOps Engineer Expert
Columbia University ML Certificate

WORK EXPERIENCE
Data Engineer — Lambton College (Jan 2024 – Present)
- Built GPT-2 chatbot over TD Bank PDFs using Flask, React, AWS EC2, Docker
- Reduced document retrieval time by 40% using semantic search

Software Engineer Intern — TechSoft India (Jun 2022 – Dec 2022, 7 months)
- Built ETL pipelines using Python and Tableau
- Managed AWS RDS databases for 3 production environments

EDUCATION
Graduate Certificate in AI & Machine Learning — Lambton College, 2024. GPA: 3.15
B.Tech (Hons) Computer Science — KL University, 2022

PUBLICATIONS
IEEE ICICT 2023: "Scalable ML Inference Pipelines for Edge Devices"
\"\"\"

Expected output:
{{
  "document_type": "resume",
  "candidate_name": "Yasaswin Palukuri",
  "email": "yasaswin@email.com",
  "phone": "+1-647-555-0123",
  "location": "Toronto, ON",
  "linkedin_url": "linkedin.com/in/yasaswinpalukuri",
  "github_url": "github.com/yasaswinpalukuri",
  "years_of_experience": 2.0,
  "current_role": "Data Engineer",
  "current_company": "Lambton College",
  "education": [
    {{
      "degree": "Graduate Certificate",
      "field": "AI & Machine Learning",
      "institution": "Lambton College",
      "year": 2024,
      "gpa": 3.15
    }},
    {{
      "degree": "B.Tech (Hons)",
      "field": "Computer Science",
      "institution": "KL University",
      "year": 2022,
      "gpa": null
    }}
  ],
  "skills": ["Python", "SQL", "PySpark", "AWS", "S3", "Glue", "Lambda", "Docker", "LangChain", "FastAPI", "ChromaDB", "TensorFlow"],
  "certifications": ["AZ-400 Microsoft DevOps Engineer Expert", "Columbia University ML Certificate"],
  "languages": [],
  "publications": ["IEEE ICICT 2023: Scalable ML Inference Pipelines for Edge Devices"],
  "work_experience": [
    {{
      "role": "Data Engineer",
      "company": "Lambton College",
      "start_date": "Jan 2024",
      "end_date": "Present",
      "duration_months": null,
      "key_achievements": [
        "Built GPT-2 chatbot over TD Bank PDFs using Flask, React, AWS EC2, Docker",
        "Reduced document retrieval time by 40% using semantic search"
      ]
    }},
    {{
      "role": "Software Engineer Intern",
      "company": "TechSoft India",
      "start_date": "Jun 2022",
      "end_date": "Dec 2022",
      "duration_months": 7,
      "key_achievements": [
        "Built ETL pipelines using Python and Tableau",
        "Managed AWS RDS databases for 3 production environments"
      ]
    }}
  ],
  "confidence_score": 92
}}

--- END EXAMPLE ---

Now extract from this document:
\"\"\"
{document_text}
\"\"\"

JSON output:"""


# ---------------------------------------------------------------------------
# Prompt 3: Invoice
# ---------------------------------------------------------------------------

INVOICE_EXTRACTION_PROMPT = """\
Extract structured fields from the invoice or bill below.

Return a JSON object with exactly these fields:
{{
  "document_type": "invoice",
  "invoice_number": "string or null",
  "invoice_date": "string or null — as written",
  "due_date": "string or null — as written",
  "vendor_name": "string — seller / supplier name",
  "vendor_address": "string or null",
  "vendor_email": "string or null",
  "client_name": "string — buyer / bill-to name",
  "client_address": "string or null",
  "line_items": [
    {{
      "description": "string",
      "quantity": "float or null",
      "unit_price": "float or null",
      "total": "float or null"
    }}
  ],
  "subtotal": "float or null",
  "tax_rate": "float or null — percentage e.g. 13.0 for 13%",
  "tax_amount": "float or null",
  "total_amount": "float — required, final amount due",
  "currency": "string — e.g. CAD, USD",
  "payment_terms": "string or null — e.g. Net 30",
  "payment_status": "Paid | Unpaid | Overdue | Not specified",
  "extracted_at": "leave this field out — it is auto-generated",
  "confidence_score": "integer 0–100"
}}

--- FEW-SHOT EXAMPLE ---

Input document:
\"\"\"
INVOICE

From: DataSoft Solutions Inc
      123 Tech Street, Toronto, ON M5V 1A1
      billing@datasoft.ca

Bill To: Acme Corporation
         456 Bay Street, Toronto, ON M5H 2S3

Invoice #: INV-2024-0042
Invoice Date: January 15, 2024
Due Date: February 14, 2024
Payment Terms: Net 30

Description                     Qty    Unit Price    Total
-------------------------------------------------------
Data Pipeline Development        40     $150.00    $6,000.00
Cloud Infrastructure Setup        8     $200.00    $1,600.00
Monthly Maintenance Fee           1     $500.00      $500.00

Subtotal:                                          $8,100.00
HST (13%):                                         $1,053.00
TOTAL DUE:                                         $9,153.00

Payment Status: Unpaid
\"\"\"

Expected output:
{{
  "document_type": "invoice",
  "invoice_number": "INV-2024-0042",
  "invoice_date": "January 15, 2024",
  "due_date": "February 14, 2024",
  "vendor_name": "DataSoft Solutions Inc",
  "vendor_address": "123 Tech Street, Toronto, ON M5V 1A1",
  "vendor_email": "billing@datasoft.ca",
  "client_name": "Acme Corporation",
  "client_address": "456 Bay Street, Toronto, ON M5H 2S3",
  "line_items": [
    {{"description": "Data Pipeline Development", "quantity": 40.0, "unit_price": 150.00, "total": 6000.00}},
    {{"description": "Cloud Infrastructure Setup", "quantity": 8.0, "unit_price": 200.00, "total": 1600.00}},
    {{"description": "Monthly Maintenance Fee", "quantity": 1.0, "unit_price": 500.00, "total": 500.00}}
  ],
  "subtotal": 8100.00,
  "tax_rate": 13.0,
  "tax_amount": 1053.00,
  "total_amount": 9153.00,
  "currency": "CAD",
  "payment_terms": "Net 30",
  "payment_status": "Unpaid",
  "confidence_score": 97
}}

--- END EXAMPLE ---

Now extract from this document:
\"\"\"
{document_text}
\"\"\"

JSON output:"""


# ---------------------------------------------------------------------------
# Retry prompt — used when first extraction returns malformed JSON
# ---------------------------------------------------------------------------

RETRY_PROMPT = """\
Your previous response was not valid JSON. Try again.

Return ONLY a valid JSON object — no explanation, no markdown, no code fences, no extra text.
The JSON must start with {{ and end with }}.

Document type: {document_type}

Document:
\"\"\"
{document_text}
\"\"\"

JSON output:"""


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------

_PROMPT_MAP: dict[str, str] = {
    "job_description": JD_EXTRACTION_PROMPT,
    "resume": RESUME_EXTRACTION_PROMPT,
    "invoice": INVOICE_EXTRACTION_PROMPT,
}


def get_prompt_for_type(document_type: str) -> str:
    """
    Return the extraction prompt template for a given document type.

    Args:
        document_type: One of 'job_description', 'resume', 'invoice'.

    Returns:
        Prompt string with a {document_text} placeholder for .format().

    Raises:
        ValueError: If document_type is not recognised.
    """
    if document_type not in _PROMPT_MAP:
        raise ValueError(
            f"No prompt found for document_type='{document_type}'. "
            f"Available: {list(_PROMPT_MAP.keys())}"
        )
    return _PROMPT_MAP[document_type]


def get_system_instruction() -> str:
    """Return the shared system instruction for the extraction LLM."""
    return SYSTEM_INSTRUCTION
