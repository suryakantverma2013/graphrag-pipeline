"""Typed ingestion state (FR-0.2).

A single pydantic object threaded through the LangGraph StateGraph. Nodes return
partial dicts that LangGraph merges. Heavy payloads (raw bytes, docling tree,
embeddings) are held only as needed and never logged (NFR-LOG-6).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    """Typed pipeline status driving conditional routing (FR-0.6)."""

    RUNNING = "RUNNING"
    INTAKE_ERROR = "INTAKE_ERROR"     # FR-1.1/1.2 (missing/unreadable/unsupported)
    DUPLICATE = "DUPLICATE"           # FR-1.5
    PARSE_ERROR = "PARSE_ERROR"       # FR-2.6
    EMBED_ERROR = "EMBED_ERROR"       # FR-6.5
    WRITE_ERROR = "WRITE_ERROR"       # FR-7.8
    COMPLETED = "COMPLETED"           # FR-9.1


class IngestionState(BaseModel):
    """Full ingestion run state (FR-0.2)."""

    # Identity / control
    file_path: str
    thread_id: str
    doc_hash: str | None = None                     # SHA-256 = Document id (FR-1.3)
    temp_path: str | None = None                    # transient parse file (FR-2.5)

    # Stage payloads
    docling_document: Any | None = None             # DoclingDocument tree (FR-2.4)
    title: str | None = None
    page_count: int | None = None
    chunks: list[dict[str, Any]] = Field(default_factory=list)   # FR-3.x
    total_tokens: int | None = None
    iirds_tags: dict[str, Any] | None = None        # FR-4.2
    confidence: float | None = None                 # FR-4.3
    human_reviewed: bool = False                    # FR-5.5
    embeddings_meta: dict[str, Any] | None = None   # tokens_used, cost (FR-6.6)

    # Write results
    neo4j_doc_id: str | None = None                 # FR-7.9
    neo4j_chunk_ids: list[str] = Field(default_factory=list)

    # Observability
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)  # FR-0.5
    pipeline_status: IngestStatus = IngestStatus.RUNNING
    error: str | None = None
