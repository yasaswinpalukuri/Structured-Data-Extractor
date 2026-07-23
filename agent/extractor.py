"""
Core extraction agent for the Structured Data Extractor pipeline.

Wires together:
    - Document classifier (heuristic + LLM fallback)
    - LLM extraction with Pydantic validation + retry
    - ChromaDB for document memory (direct, separate from Project 1 conversation memory)
    - Notion sync via NotionClient
    - Multi-turn chat: answer questions about previously extracted documents

Usage:
    agent = ExtractionAgent()
    result = agent.extract(text="...", document_type="job_description", sync_to_notion=True)
    answer = agent.chat("What was the salary on the last JD I uploaded?", session_id="abc")
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from agent.tools import (
    classify_document_tool,
    extract_fields_tool,
    notion_push_tool,
    validate_extraction_tool,
)
from notion.client import NotionClient

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------

class ExtractionAgent:
    """
    Orchestrates the full extraction pipeline for a single document.

    Pipeline for extract():
        1. Classify document type (heuristic → LLM fallback)
        2. Extract structured fields via LLM + Pydantic validation
        3. Store result in ChromaDB memory
        4. Optionally push to Notion

    Pipeline for chat():
        1. Load relevant extraction context from ChromaDB
        2. Send context + question to LLM
        3. Return natural language answer
    """

    def __init__(self) -> None:
        # ChromaDB — separate collection from Project 1
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        collection_name = os.getenv(
            "CHROMA_COLLECTION_EXTRACTION", "extraction_sessions"
        )
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Notion client
        self._notion = NotionClient()

        # LLM for multi-turn chat
        self._llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=0.0,
            num_ctx=int(os.getenv("OLLAMA_CTX", "8192")),
        )

        logger.info(
            "ExtractionAgent initialised | collection=%s | persist_dir=%s",
            collection_name,
            persist_dir,
        )

    # ------------------------------------------------------------------
    # Public: extract
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        document_type: str | None = None,
        session_id: str = "default",
        source_url: str | None = None,
        sync_to_notion: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full extraction pipeline on a document.

        Args:
            text:            Plain text content of the document.
            document_type:   Override classification — one of 'job_description',
                             'resume', 'invoice'. If None, auto-classified.
            session_id:      Session identifier for ChromaDB memory grouping.
            source_url:      Optional URL where the document was sourced.
            sync_to_notion:  Whether to push extracted data to Notion.

        Returns:
            Dict with keys:
                - document_type (str)
                - classification_confidence (int)
                - classification_method (str)
                - extracted_data (dict)
                - notion_page_url (str or None)
                - notion_action (str)
                - memory_id (str)
                - total_latency_ms (float)
                - error (str or None)
        """
        pipeline_start = time.perf_counter()
        result: dict[str, Any] = {
            "document_type": None,
            "classification_confidence": None,
            "classification_method": None,
            "extracted_data": None,
            "notion_page_url": None,
            "notion_action": "skipped",
            "memory_id": None,
            "total_latency_ms": None,
            "error": None,
        }

        try:
            # ── Step 1: Classify ──────────────────────────────────────
            if document_type:
                logger.info("Document type overridden to '%s'", document_type)
                result["document_type"] = document_type
                result["classification_confidence"] = 100
                result["classification_method"] = "override"
            else:
                logger.info("Classifying document...")
                classification = classify_document_tool.invoke(
                    {"document_text": text}
                )
                result["document_type"] = classification["document_type"]
                result["classification_confidence"] = classification["confidence"]
                result["classification_method"] = classification["method"]

                if result["document_type"] == "unknown":
                    result["error"] = (
                        "Document could not be classified. "
                        "Please specify document_type explicitly."
                    )
                    return result

            logger.info(
                "Document type: '%s' (confidence=%s method=%s)",
                result["document_type"],
                result["classification_confidence"],
                result["classification_method"],
            )

            # ── Step 2: Extract ───────────────────────────────────────
            logger.info("Extracting fields...")
            extracted = extract_fields_tool.invoke(
                {
                    "document_text": text,
                    "document_type": result["document_type"],
                }
            )
            result["extracted_data"] = extracted
            logger.info(
                "Extraction complete: confidence=%s",
                extracted.get("confidence_score"),
            )

            # ── Step 3: Validate ──────────────────────────────────────
            validation = validate_extraction_tool.invoke(
                {
                    "extracted_data": dict(extracted),
                    "document_type": result["document_type"],
                }
            )
            if not validation["valid"]:
                logger.warning(
                    "Extraction validation warnings: %s", validation["errors"]
                )
                result["extracted_data"]["_validation_warnings"] = validation["errors"]

            # ── Step 4: Store in ChromaDB ─────────────────────────────
            logger.info("Storing in ChromaDB memory...")
            memory_id = self._store_in_memory(
                text=text,
                extracted=extracted,
                document_type=result["document_type"],
                session_id=session_id,
                source_url=source_url,
            )
            result["memory_id"] = memory_id

            # ── Step 5: Notion sync ───────────────────────────────────
            if sync_to_notion:
                logger.info("Syncing to Notion...")
                notion_result = notion_push_tool.invoke(
                    {
                        "extracted_data": dict(extracted),
                        "document_type": result["document_type"],
                    }
                )
                result["notion_page_url"] = notion_result.get("notion_page_url")
                result["notion_action"] = notion_result.get("action", "unknown")
                if not notion_result.get("success"):
                    logger.warning(
                        "Notion sync failed: %s", notion_result.get("error")
                    )

        except Exception as exc:  # noqa: BLE001
            logger.error("Extraction pipeline failed: %s", exc, exc_info=True)
            result["error"] = str(exc)

        finally:
            result["total_latency_ms"] = round(
                (time.perf_counter() - pipeline_start) * 1000, 0
            )

        return result

    # ------------------------------------------------------------------
    # Public: chat
    # ------------------------------------------------------------------

    def chat(
        self,
        question: str,
        session_id: str = "default",
        n_context_docs: int = 5,
    ) -> dict[str, Any]:
        """
        Answer a natural language question about previously extracted documents.

        Retrieves relevant extraction records from ChromaDB and passes them
        as context to the LLM to answer the question.

        Args:
            question:        User's question e.g. "What was the salary on the last JD?"
            session_id:      Filter memory to this session (or 'all' for no filter).
            n_context_docs:  Number of ChromaDB results to include as context.

        Returns:
            Dict with keys:
                - answer (str)
                - context_docs_used (int)
                - latency_ms (float)
        """
        start = time.perf_counter()

        # Build ChromaDB query filter
        where: dict[str, Any] | None = None
        if session_id and session_id != "all":
            where = {"session_id": session_id}

        try:
            query_result = self._collection.query(
                query_texts=[question],
                n_results=min(n_context_docs, max(self._collection.count(), 1)),
                where=where,
                include=["documents", "metadatas"],
            )
            docs = query_result.get("documents", [[]])[0]
            metas = query_result.get("metadatas", [[]])[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB query failed: %s", exc)
            docs, metas = [], []

        if not docs:
            return {
                "answer": (
                    "I don't have any extracted documents in memory yet. "
                    "Please upload and extract a document first."
                ),
                "context_docs_used": 0,
                "latency_ms": round((time.perf_counter() - start) * 1000, 0),
            }

        # Build context string
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            context_parts.append(
                f"[Document {i} — type={meta.get('document_type', 'unknown')} "
                f"session={meta.get('session_id', 'unknown')}]\n{doc}"
            )
        context_str = "\n\n".join(context_parts)

        prompt = f"""\
You are a helpful assistant with access to extracted document data.
Answer the user's question using only the document context below.
Be concise and specific. If the answer is not in the context, say so clearly.

Document context:
\"\"\"
{context_str}
\"\"\"

Question: {question}

Answer:"""

        response = self._llm.invoke([("human", prompt)])
        latency_ms = round((time.perf_counter() - start) * 1000, 0)

        logger.info(
            "chat: question='%s' context_docs=%d latency=%.0fms",
            question[:60],
            len(docs),
            latency_ms,
        )

        return {
            "answer": response.content.strip(),
            "context_docs_used": len(docs),
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Public: health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Check the health of all agent dependencies.

        Returns:
            Dict with keys: ollama (bool), chromadb (bool), notion (bool), errors (list)
        """
        errors: list[str] = []

        # Ollama
        try:
            self._llm.invoke([("human", "ping")])
            ollama_ok = True
        except Exception as exc:  # noqa: BLE001
            ollama_ok = False
            errors.append(f"Ollama: {exc}")

        # ChromaDB
        try:
            self._collection.count()
            chroma_ok = True
        except Exception as exc:  # noqa: BLE001
            chroma_ok = False
            errors.append(f"ChromaDB: {exc}")

        # Notion
        try:
            notion_health = self._notion.health_check()
            notion_ok = notion_health["reachable"]
            if not notion_ok:
                errors.append(f"Notion: {notion_health.get('error')}")
        except Exception as exc:  # noqa: BLE001
            notion_ok = False
            errors.append(f"Notion: {exc}")

        return {
            "ollama": ollama_ok,
            "chromadb": chroma_ok,
            "notion": notion_ok,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_in_memory(
        self,
        text: str,
        extracted: dict[str, Any],
        document_type: str,
        session_id: str,
        source_url: str | None,
    ) -> str:
        """
        Store the original text + extracted JSON in ChromaDB.

        The stored document combines the raw text with the extracted fields
        so both are searchable in multi-turn chat.

        Returns:
            The ChromaDB document ID.
        """
        extracted_summary = json.dumps(extracted, indent=2, default=str)
        combined_content = (
            f"DOCUMENT TYPE: {document_type}\n\n"
            f"EXTRACTED FIELDS:\n{extracted_summary}\n\n"
            f"ORIGINAL TEXT:\n{text[:2000]}"
        )

        memory_id = f"{session_id}_{document_type}_{uuid.uuid4().hex[:8]}"

        metadata: dict[str, str] = {
            "document_type": document_type,
            "session_id": session_id,
            "source_url": source_url or "",
            "confidence_score": str(extracted.get("confidence_score", 0)),
        }

        self._collection.upsert(
            ids=[memory_id],
            documents=[combined_content],
            metadatas=[metadata],
        )

        logger.info("Stored in ChromaDB: memory_id=%s", memory_id)
        return memory_id
