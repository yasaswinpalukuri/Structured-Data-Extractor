"""
FastAPI application for the Structured Data Extractor pipeline.

Endpoints:
    POST /extract                  — Auto-classify then extract
    POST /extract/jd               — Extract job description fields
    POST /extract/resume           — Extract resume fields
    POST /extract/invoice          — Extract invoice fields
    POST /extract/jd-and-rank      — Extract JD + call Project 2 /rank
    POST /notion/sync/{doc_id}     — Push a stored doc to Notion
    GET  /notion/status/{doc_id}   — Check Notion sync status
    GET  /health                   — System health check

n8n-compatible: all POST endpoints accept JSON with optional session_id field.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.extractor import ExtractionAgent
from utils.pdf_parser import parse_pdf

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App lifecycle — single agent instance shared across requests
# ---------------------------------------------------------------------------

_agent: ExtractionAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    logger.info("Starting up — initialising ExtractionAgent...")
    _agent = ExtractionAgent()
    logger.info("ExtractionAgent ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Structured Data Extractor",
    description=(
        "Extracts structured fields from Job Descriptions, Resumes, and Invoices "
        "using a local LLM (qwen2.5:7b via Ollama) and syncs to Notion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> ExtractionAgent:
    """Return the shared agent instance."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")
    return _agent


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    """Request body for all /extract endpoints."""
    text: str = Field(..., description="Plain text content of the document.")
    source_url: str | None = Field(None, description="URL where the document was sourced.")
    session_id: str = Field("default", description="Session identifier for memory grouping.")
    sync_to_notion: bool = Field(True, description="Whether to push extracted data to Notion.")


class ExtractResponse(BaseModel):
    """Response body for all /extract endpoints."""
    document_type: str | None
    classification_confidence: int | None
    classification_method: str | None
    extracted_data: dict[str, Any] | None
    notion_page_url: str | None
    notion_action: str
    memory_id: str | None
    total_latency_ms: float | None
    error: str | None


class ChatRequest(BaseModel):
    """Request body for multi-turn chat."""
    question: str = Field(..., description="Natural language question about extracted documents.")
    session_id: str = Field("default", description="Session to search memory in.")
    n_context_docs: int = Field(5, description="Number of context documents to retrieve.")


class JDAndRankRequest(BaseModel):
    """Request body for /extract/jd-and-rank."""
    text: str = Field(..., description="Job description plain text.")
    source_url: str | None = None
    session_id: str = "default"
    sync_to_notion: bool = True


class NotionSyncRequest(BaseModel):
    """Request body for manual Notion sync."""
    document_type: str = Field(..., description="One of: job_description, resume, invoice.")
    extracted_data: dict[str, Any] = Field(..., description="Previously extracted data dict.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """
    Check health of all system dependencies.

    Returns status of Ollama, ChromaDB, Notion, and version info.
    """
    agent = get_agent()
    health = agent.health()
    return {
        "status": "healthy" if all([health["ollama"], health["chromadb"], health["notion"]]) else "degraded",
        "version": "1.0.0",
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "ollama": health["ollama"],
        "chromadb": health["chromadb"],
        "notion": health["notion"],
        "errors": health["errors"],
    }


@app.post("/extract", response_model=ExtractResponse, tags=["Extraction"])
async def extract_auto(request: ExtractRequest) -> ExtractResponse:
    """
    Auto-classify document type then extract structured fields.

    Runs heuristic classifier first (< 10ms), falls back to LLM if confidence < 60.
    Syncs to the appropriate Notion database if sync_to_notion=True.
    """
    agent = get_agent()
    result = agent.extract(
        text=request.text,
        document_type=None,  # Auto-classify
        session_id=request.session_id,
        source_url=request.source_url,
        sync_to_notion=request.sync_to_notion,
    )
    return ExtractResponse(**result)


@app.post("/extract/jd", response_model=ExtractResponse, tags=["Extraction"])
async def extract_jd(request: ExtractRequest) -> ExtractResponse:
    """
    Extract structured fields from a job description.

    Skips classification — processes directly as job_description.
    n8n Career Agent uses this endpoint when a new job posting is found.
    """
    agent = get_agent()
    result = agent.extract(
        text=request.text,
        document_type="job_description",
        session_id=request.session_id,
        source_url=request.source_url,
        sync_to_notion=request.sync_to_notion,
    )
    return ExtractResponse(**result)


@app.post("/extract/resume", response_model=ExtractResponse, tags=["Extraction"])
async def extract_resume(request: ExtractRequest) -> ExtractResponse:
    """
    Extract structured fields from a resume or CV.

    Skips classification — processes directly as resume.
    """
    agent = get_agent()
    result = agent.extract(
        text=request.text,
        document_type="resume",
        session_id=request.session_id,
        source_url=request.source_url,
        sync_to_notion=request.sync_to_notion,
    )
    return ExtractResponse(**result)


@app.post("/extract/invoice", response_model=ExtractResponse, tags=["Extraction"])
async def extract_invoice(request: ExtractRequest) -> ExtractResponse:
    """
    Extract structured fields from an invoice or bill.

    Skips classification — processes directly as invoice.
    """
    agent = get_agent()
    result = agent.extract(
        text=request.text,
        document_type="invoice",
        session_id=request.session_id,
        source_url=request.source_url,
        sync_to_notion=request.sync_to_notion,
    )
    return ExtractResponse(**result)


@app.post("/extract/jd-and-rank", tags=["Extraction"])
async def extract_jd_and_rank(request: JDAndRankRequest) -> dict[str, Any]:
    """
    Extract JD fields + rank all stored resumes against the JD.

    Workflow:
        1. Extract structured fields from the job description
        2. Push JD to Notion JD database
        3. Call Project 2 POST /rank with JD text + stored resume texts
        4. Return extracted JD fields + ranked candidates + Notion page URL

    This is the Career Agent's main workflow trigger in n8n.
    """
    agent = get_agent()

    # Step 1 + 2: Extract JD and sync to Notion
    jd_result = agent.extract(
        text=request.text,
        document_type="job_description",
        session_id=request.session_id,
        source_url=request.source_url,
        sync_to_notion=request.sync_to_notion,
    )

    if jd_result.get("error"):
        raise HTTPException(
            status_code=422,
            detail=f"JD extraction failed: {jd_result['error']}",
        )

    # Step 3: Load stored resumes from ChromaDB and call Project 2 /rank
    project2_url = os.getenv("PROJECT2_API_URL", "http://localhost:8000")
    ranked_candidates: list[dict] = []
    rank_error: str | None = None

    try:
        # Query ChromaDB for all stored resume documents
        resume_results = agent._collection.query(
            query_texts=[request.text],
            n_results=min(10, max(agent._collection.count(), 1)),
            where={"document_type": "resume"},
            include=["documents", "metadatas"],
        )
        resume_docs = resume_results.get("documents", [[]])[0]
        # ChromaDB always returns ids separately from include list
        resume_ids = resume_results.get("ids", [[]])[0] if resume_results.get("ids") else [f"resume_{i}" for i in range(len(resume_docs))]

        if not resume_docs:
            rank_error = "No resumes found in memory. Extract some resumes first via /extract/resume."
            logger.warning(rank_error)
        else:
            resumes_payload = [
                {"id": rid, "text": doc}
                for rid, doc in zip(resume_ids, resume_docs)
                if len(doc) >= 50
            ]
            async with httpx.AsyncClient(timeout=120.0) as client:
                rank_response = await client.post(
                    f"{project2_url}/rank",
                    json={
                        "job_description": request.text,
                        "resumes": resumes_payload,
                        "session_id": request.session_id,
                    },
                )
                if rank_response.status_code == 200:
                    ranked_candidates = rank_response.json()
                    logger.info("Project 2 /rank returned %d candidates", len(ranked_candidates))
                else:
                    rank_error = f"Project 2 /rank returned {rank_response.status_code}: {rank_response.text[:200]}"
                    logger.warning(rank_error)

    except httpx.ConnectError:
        rank_error = (
            f"Project 2 API not reachable at {project2_url}. "
            "Start it with: uvicorn api.main:app --port 8000"
        )
        logger.warning(rank_error)
    except Exception as exc:  # noqa: BLE001
        rank_error = str(exc)
        logger.error("Project 2 /rank call failed: %s", exc)

    return {
        "jd_extraction": jd_result,
        "ranked_candidates": ranked_candidates,
        "rank_error": rank_error,
        "notion_page_url": jd_result.get("notion_page_url"),
    }


@app.post("/extract/pdf", tags=["Extraction"])
async def extract_pdf(
    file: UploadFile = File(...),
    session_id: str = "default",
    sync_to_notion: bool = True,
) -> ExtractResponse:
    """
    Upload a PDF file, parse it, auto-classify, and extract structured fields.

    Uses PyMuPDF (primary) + pdfplumber (fallback) — reused from Project 2.
    Max file size: 5MB.
    """
    max_mb = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
    contents = await file.read()

    if len(contents) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_mb}MB.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported at this endpoint.",
        )

    import tempfile  # noqa: PLC0415
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        parsed = parse_pdf(tmp_path)
        text = parsed.text
    finally:
        import os as _os  # noqa: PLC0415
        _os.unlink(tmp_path)

    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from PDF. File may be scanned or image-only.",
        )

    agent = get_agent()
    result = agent.extract(
        text=text,
        document_type=None,
        session_id=session_id,
        source_url=file.filename,
        sync_to_notion=sync_to_notion,
    )
    return ExtractResponse(**result)


@app.post("/chat", tags=["Memory"])
async def chat(request: ChatRequest) -> dict[str, Any]:
    """
    Answer a natural language question about previously extracted documents.

    Example questions:
        - "What was the salary range on the last JD I uploaded?"
        - "Who is the most recent candidate and what are their skills?"
        - "What is the total amount on the latest invoice?"
    """
    agent = get_agent()
    return agent.chat(
        question=request.question,
        session_id=request.session_id,
        n_context_docs=request.n_context_docs,
    )


@app.post("/notion/sync/{doc_id}", tags=["Notion"])
async def notion_sync(doc_id: str, request: NotionSyncRequest) -> dict[str, Any]:
    """
    Push a previously extracted document to Notion.

    Use this to manually sync a document that failed to sync automatically
    or was extracted with sync_to_notion=False.
    """
    from agent.tools import notion_push_tool  # noqa: PLC0415

    result = notion_push_tool.invoke(
        {
            "extracted_data": request.extracted_data,
            "document_type": request.document_type,
        }
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=502,
            detail=f"Notion sync failed: {result.get('error')}",
        )

    return {
        "doc_id": doc_id,
        "notion_page_url": result.get("notion_page_url"),
        "action": result.get("action"),
    }


@app.get("/notion/status/{doc_id}", tags=["Notion"])
async def notion_status(doc_id: str) -> dict[str, Any]:
    """
    Check whether a document has been synced to Notion.

    Looks up the document by memory_id in ChromaDB and checks
    the notion_synced metadata field.
    """
    agent = get_agent()

    try:
        result = agent._collection.get(
            ids=[doc_id],
            include=["metadatas"],
        )
        if not result["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{doc_id}' not found in memory.",
            )
        metadata = result["metadatas"][0]
        return {
            "doc_id": doc_id,
            "found": True,
            "document_type": metadata.get("document_type"),
            "session_id": metadata.get("session_id"),
            "notion_synced": metadata.get("notion_synced", "false") == "true",
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
