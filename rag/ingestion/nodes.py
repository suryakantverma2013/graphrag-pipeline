"""Ingestion node implementations (§3).

Each node takes the typed `IngestionState` and returns a partial dict that
LangGraph merges. The `_node` decorator scopes the structured-log `stage`
(NFR-LOG-7), logs stage entry, and records the node's wall-clock duration into
`stage_timings_ms` (FR-0.5) so every node carries its own timing without
boilerplate. Routing functions read typed status/confidence from state so the
graph wiring in graph.py stays declarative.

Heavy local clients (Neo4j driver, OpenAI seam, Docling converter) are built
once per process and reused (NFR-PERF-4); config comes from the cached
`get_config()` singleton (NFR-MAINT-7) since LangGraph node signatures are fixed
to `(state)`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field

from ..clients.docling_client import build_converter
from ..clients.neo4j_client import Neo4jClient
from ..clients.openai_client import OpenAIClient
from ..config import AppConfig, get_config
from ..iirds import (
    LifecyclePhase,
    InformationType,
    normalize_information_type,
    normalize_lifecycle_phase,
)
from ..logging_config import stage_context
from .state import IngestionState, IngestStatus

logger = logging.getLogger(__name__)

# Node names — single source shared with graph.py (FR-0.1).
INTAKE = "intake"
PARSE = "parse"
CHUNK = "chunk"
TAG_IIRDS = "tag_iirds"
HUMAN_REVIEW = "human_review"
EMBED = "embed"
NEO4J_WRITE = "neo4j_write"
RECEIPT = "receipt"

# Accepted source formats (FR-1.2). Docling selects its backend by extension.
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".xml", ".txt", ".md"}

# Content types attached to chunks (FR-3.5). Structural types are kept whole.
CT_TABLE = "table"
CT_LIST = "list"
CT_WARNING = "warning"
CT_CODE = "code"
CT_TEXT = "text"

# Safety-critical leading markers → one unfragmented chunk each (FR-3.3).
_WARNING_RE = re.compile(r"^\s*(warning|caution|danger|notice|important)\b", re.IGNORECASE)
# Sentence splitter for 50-token sentence-boundary overlap (FR-3.4).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# text-embedding-3-small list price (USD per 1M tokens), OpenAI pricing as of
# June 2026. Used only for the informational cost estimate (FR-6.6 / NFR-COST-1);
# it is a published rate, not a tunable, so it lives here as a named constant
# rather than in the env schema (verify against current OpenAI pricing).
EMBED_USD_PER_1M_TOKENS = 0.02

# Append-only ingestion audit trail (FR-8.1a / NFR-OBS-4); distinct from the app
# log (NFR-LOG-8). Repo-root relative; git-ignored.
INGESTION_LOG_PATH = Path("ingestion_log.json")


# --- Per-process clients / tokenizer (NFR-PERF-4) ----------------------------
@lru_cache(maxsize=1)
def _neo4j() -> Neo4jClient:
    return Neo4jClient(get_config())


@lru_cache(maxsize=1)
def _openai() -> OpenAIClient:
    return OpenAIClient(get_config())


@lru_cache(maxsize=1)
def _encoder():
    """tiktoken cl100k_base — the tokenizer for text-embedding-3-small (FR-3.4)."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _node(stage: str) -> Callable:
    """Scope the log `stage`, log entry, and record wall-clock ms (FR-0.5)."""

    def decorator(fn: Callable) -> Callable:
        def wrapper(state: IngestionState) -> dict:
            with stage_context(stage):
                logger.info("stage start: %s", stage)
                start = time.perf_counter()
                result = fn(state)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                # Accumulate timings: returning a bare {stage: ms} would replace
                # the whole dict (no LangGraph reducer here), so merge prior.
                if isinstance(result, dict):
                    result["stage_timings_ms"] = {
                        **state.stage_timings_ms,
                        stage: round(elapsed_ms, 1),
                    }
                logger.info("stage done: %s (%.1f ms)", stage, elapsed_ms)
                return result

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


def _terminal(status: IngestStatus, message: str) -> dict:
    """Build a terminal partial-state dict and log the reason (FR-0.7 / NFR-USE-1)."""
    logger.error("%s: %s", status.value, message)
    return {"pipeline_status": status, "error": message}


# --- Stage 1: Intake (FR-1.x) ------------------------------------------------
@_node(INTAKE)
def intake(state: IngestionState) -> dict:
    config = get_config()
    path = Path(state.file_path)

    # FR-1.1 exists + readable; FR-1.2 supported format.
    if not path.is_file():
        return _terminal(IngestStatus.INTAKE_ERROR, f"file not found or not readable: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return _terminal(
            IngestStatus.INTAKE_ERROR,
            f"unsupported format '{suffix or '(none)'}'; supported: {sorted(SUPPORTED_SUFFIXES)}",
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return _terminal(IngestStatus.INTAKE_ERROR, f"could not read file: {exc}")
    if not raw:
        return _terminal(IngestStatus.INTAKE_ERROR, f"empty file (zero bytes): {path}")

    # FR-1.3 SHA-256 of raw bytes = canonical Document id.
    doc_hash = hashlib.sha256(raw).hexdigest()

    # FR-1.4/1.5 dedup against the graph; duplicate → terminal DUPLICATE (no
    # OpenAI calls happen before this point — AC-2 / NFR-COST-3).
    with _neo4j().driver.session(database=config.neo4j_database) as session:
        existing = session.run(
            "MATCH (d:Document {id: $id}) RETURN d.id AS id", id=doc_hash
        ).single()
    if existing is not None:
        logger.info("duplicate document hash=%s (re-ingestion not supported)", doc_hash[:12])
        return {"doc_hash": doc_hash, "pipeline_status": IngestStatus.DUPLICATE}

    logger.info("intake ok: hash=%s bytes=%d format=%s", doc_hash[:12], len(raw), suffix)
    return {"doc_hash": doc_hash, "pipeline_status": IngestStatus.RUNNING}


# --- Stage 2: Docling parse (FR-2.x) ----------------------------------------
def _document_title(doc: Any, source: Path) -> str:
    """Prefer a TITLE element, else the Docling name, else the file stem (FR-2.7)."""
    from docling_core.types.doc import DocItemLabel

    for item, _level in doc.iterate_items():
        if getattr(item, "label", None) == DocItemLabel.TITLE:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                return text
    name = (getattr(doc, "name", "") or "").strip()
    return name or source.stem


@_node(PARSE)
def parse(state: IngestionState) -> dict:
    from docling.datamodel.base_models import ConversionStatus

    src = Path(state.file_path)
    # FR-2.5 materialize raw bytes to a private temp file for parsing, then delete
    # immediately (success or failure). mkstemp creates an owner-only file
    # (NFR-SEC-6); raw bytes never leave the device (FR-2.1, local-only Docling).
    fd, temp_name = tempfile.mkstemp(suffix=src.suffix.lower(), prefix="rag_ingest_")
    temp_path = Path(temp_name)
    config = get_config()
    # Optional page cap (config): parse only the first N pages of a PDF (0 = all).
    # Bounds memory/time on very large PDFs; page_range is PDF-only in Docling.
    convert_kwargs: dict[str, Any] = {}
    if config.pdf_max_pages > 0 and src.suffix.lower() == ".pdf":
        convert_kwargs["page_range"] = (1, config.pdf_max_pages)
        logger.info("PDF page cap active: parsing pages 1-%s", config.pdf_max_pages)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(src.read_bytes())
        # FR-2.1/2.2/2.3 local Docling (DocLayNet + TableFormer + EasyOCR, GPU).
        result = build_converter(config).convert(str(temp_path), **convert_kwargs)
    except Exception as exc:  # noqa: BLE001 — any parse failure → terminal (FR-2.6)
        logger.error("Docling parse raised", exc_info=True)
        return _terminal(IngestStatus.PARSE_ERROR, f"parse failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)  # FR-2.5 delete temp immediately

    if result.status == ConversionStatus.FAILURE:
        return _terminal(IngestStatus.PARSE_ERROR, f"Docling reported FAILURE for {src.name}")

    doc = result.document
    if not doc.export_to_text().strip():  # FR-2.6 zero content
        return _terminal(IngestStatus.PARSE_ERROR, f"parsed document has no extractable text: {src.name}")

    title = _document_title(doc, src)
    page_count = doc.num_pages()
    logger.info("parse ok: title=%r pages=%s status=%s", title, page_count, result.status.value)
    return {
        "docling_document": doc,
        "title": title,
        "page_count": page_count,
        "pipeline_status": IngestStatus.RUNNING,
    }


# --- Stage 3: Structural chunking (FR-3.x) ----------------------------------
def _split_long_sentence(tokens: list[int], max_tokens: int, enc) -> Iterator[list[int]]:
    """Hard-split a single oversized sentence into <= max_tokens token windows."""
    for start in range(0, len(tokens), max_tokens):
        yield tokens[start : start + max_tokens]


def _pack_sentences(text: str, max_tokens: int, overlap_tokens: int, enc) -> list[tuple[str, int]]:
    """Greedily pack sentences into <=max_tokens chunks with sentence-boundary
    overlap of ~overlap_tokens tokens carried from the tail of the prior chunk
    (FR-3.4). Returns (chunk_text, token_count) pairs."""
    sentences = [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[tuple[str, int]] = []
    current: list[tuple[str, int]] = []  # (sentence, token_count)
    current_tokens = 0

    def flush() -> list[tuple[str, int]]:
        """Emit the current buffer as a chunk; return the overlap tail to seed next."""
        if not current:
            return []
        chunk_text = " ".join(s for s, _ in current).strip()
        chunks.append((chunk_text, current_tokens))
        tail: list[tuple[str, int]] = []
        tail_tokens = 0
        for sent in reversed(current):
            if tail_tokens + sent[1] > overlap_tokens:
                break
            tail.insert(0, sent)
            tail_tokens += sent[1]
        return tail

    for sentence in sentences:
        n_tok = len(enc.encode(sentence))
        if n_tok > max_tokens:
            # Oversized single sentence: flush buffer, then hard-split it.
            if current:
                current = flush()
                current_tokens = sum(t for _, t in current)
            for window in _split_long_sentence(enc.encode(sentence), max_tokens, enc):
                chunks.append((enc.decode(window), len(window)))
            continue
        if current and current_tokens + n_tok > max_tokens:
            current = flush()
            current_tokens = sum(t for _, t in current)
        current.append((sentence, n_tok))
        current_tokens += n_tok

    flush()
    return chunks


@_node(CHUNK)
def chunk(state: IngestionState) -> dict:
    from docling_core.types.doc import DocItemLabel

    config = get_config()
    enc = _encoder()
    doc = state.docling_document
    if doc is None:  # defensive: parse must have run (FR-2.7)
        return _terminal(IngestStatus.PARSE_ERROR, "no parsed document available for chunking")

    chunks: list[dict[str, Any]] = []
    section_stack: list[tuple[int, str]] = []  # (level, heading text)
    text_buffer: list[str] = []  # accumulated paragraph text within a section

    def section_path() -> str:
        parts = [state.title] if state.title else []
        parts.extend(text for _, text in section_stack)
        return " > ".join(parts)

    def add_chunk(text: str, content_type: str) -> None:
        text = text.strip()
        if not text:
            return
        position = len(chunks)
        chunks.append(
            {
                "id": f"{state.doc_hash}#{position}",
                "text": text,
                "content_type": content_type,
                "section_path": section_path(),
                "document_title": state.title,  # FR-3.5
                "position": position,
                "token_count": len(enc.encode(text)),
            }
        )

    def flush_text() -> None:
        """Sentence-pack the accumulated paragraph text into 512-token chunks (FR-3.4)."""
        if not text_buffer:
            return
        joined = "\n".join(text_buffer)
        for piece, _ntok in _pack_sentences(
            joined, config.chunk_max_tokens, config.chunk_overlap_tokens, enc
        ):
            add_chunk(piece, CT_TEXT)
        text_buffer.clear()

    list_buffer: list[str] = []  # consecutive list items → one chunk (FR-3.2)

    def flush_list() -> None:
        if not list_buffer:
            return
        add_chunk("\n".join(list_buffer), CT_LIST)
        list_buffer.clear()

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        item_text = (getattr(item, "text", "") or "").strip()

        # Section headings → context only, not standalone chunks (FR-3.6).
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            flush_text()
            flush_list()
            if label == DocItemLabel.TITLE:
                continue  # title already feeds section_path()
            level = getattr(item, "level", len(section_stack) + 1) or 1
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            if item_text:
                section_stack.append((level, item_text))
            continue

        # List items group together until the run ends.
        if label == DocItemLabel.LIST_ITEM:
            flush_text()
            if item_text:
                list_buffer.append(item_text)
            continue
        flush_list()  # any non-list item ends a list run

        # Tables → exactly one chunk each, structure preserved (FR-3.1).
        if label == DocItemLabel.TABLE:
            flush_text()
            try:
                table_md = item.export_to_markdown(doc).strip()
            except Exception:  # noqa: BLE001 — fall back to plain text on export quirks
                table_md = item_text
            add_chunk(table_md, CT_TABLE)
            continue

        # Code/formula → kept whole (never fragmented), like tables.
        if label in (DocItemLabel.CODE, DocItemLabel.FORMULA):
            flush_text()
            add_chunk(item_text, CT_CODE)
            continue

        if not item_text:
            continue
        # Safety-critical warnings → one unfragmented chunk each (FR-3.3).
        if _WARNING_RE.match(item_text):
            flush_text()
            add_chunk(item_text, CT_WARNING)
            continue

        # Everything else is paragraph/body text → accumulate, split later.
        text_buffer.append(item_text)

    flush_text()
    flush_list()

    if not chunks:
        return _terminal(IngestStatus.PARSE_ERROR, "chunking produced zero chunks")

    total_tokens = sum(c["token_count"] for c in chunks)
    logger.info("chunk ok: chunks=%d total_tokens=%d", len(chunks), total_tokens)
    # Drop the heavy Docling tree now that chunks are extracted (state hygiene;
    # keeps downstream checkpoints small).
    return {
        "chunks": chunks,
        "total_tokens": total_tokens,
        "docling_document": None,
        "pipeline_status": IngestStatus.RUNNING,
    }


# --- Stage 4: iiRDS tagging (FR-4.x) ----------------------------------------
class _IiRDSTagSchema(BaseModel):
    """Structured tagging output (FR-4.2). Enums are validated post-hoc against
    the closed iiRDS vocabulary (FR-4.2a) rather than constrained here, so the
    model is free to answer and out-of-vocab values normalize to None."""

    product: str | None = None
    components: list[str] = Field(default_factory=list)
    lifecycle_phase: str | None = None
    information_type: str | None = None
    language: str | None = None
    confidence: float = 0.0


_TAG_SYSTEM_PROMPT = (
    "You are a technical-documentation metadata extractor applying the iiRDS "
    "standard. Read the document excerpt and extract its metadata.\n"
    f"- lifecycle_phase MUST be one of {[p.value for p in LifecyclePhase]} or null.\n"
    f"- information_type MUST be one of {[t.value for t in InformationType]} or null.\n"
    "- product: the main product/system the document covers (free text, or null).\n"
    "- components: specific parts/subsystems mentioned (list, possibly empty).\n"
    "- language: ISO 639-1 code of the document text (e.g. 'en').\n"
    "- confidence: your 0.0-1.0 confidence in these tags overall.\n"
    "Answer only with the structured fields."
)


@_node(TAG_IIRDS)
def tag_iirds(state: IngestionState) -> dict:
    # FR-4.1 one call/doc over the first 3000 chars. The Docling tree is already
    # dropped, so reconstruct the head from the leading chunks (cheap, exact).
    head = ""
    for chunk_dict in state.chunks:
        head += chunk_dict["text"] + "\n"
        if len(head) >= 3000:
            break
    head = head[:3000]

    messages = [
        {"role": "system", "content": _TAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpt:\n\n{head}"},
    ]
    try:
        # FR-4.4 retry 3x lives in the client; FR-4.6 malformed = retryable.
        result: _IiRDSTagSchema = _openai().complete_structured(messages, _IiRDSTagSchema)
    except Exception:  # noqa: BLE001 — FR-4.5 all retries failed → empty tags, continue
        logger.warning("iiRDS tagging failed after retries; continuing with empty tags", exc_info=True)
        # confidence=None signals "tagging unavailable" → bypass human review
        # (nothing to review) rather than block the run (FR-4.5 non-blocking).
        return {"iirds_tags": {}, "confidence": None, "pipeline_status": IngestStatus.RUNNING}

    # FR-4.2a normalize closed enums; out-of-vocab → None (dropped).
    lifecycle = normalize_lifecycle_phase(result.lifecycle_phase) if result.lifecycle_phase else None
    info_type = normalize_information_type(result.information_type) if result.information_type else None
    components = [c.strip() for c in result.components if c and c.strip()]
    confidence = max(0.0, min(1.0, float(result.confidence)))

    tags: dict[str, Any] = {
        "product": (result.product or "").strip() or None,
        "components": components,
        "lifecycle_phase": lifecycle,
        "information_type": info_type,
        "language": (result.language or "").strip() or None,
    }
    logger.info(
        "tagging ok: confidence=%.2f product=%s phase=%s info_type=%s components=%d",
        confidence, bool(tags["product"]), lifecycle, info_type, len(components),
    )
    return {"iirds_tags": tags, "confidence": confidence, "pipeline_status": IngestStatus.RUNNING}


# --- Human review (conditional interrupt) (FR-5.x) --------------------------
@_node(HUMAN_REVIEW)
def human_review(state: IngestionState) -> dict:
    # The graph is compiled with interrupt_before=[human_review] (FR-5.1/5.2) and
    # is the resume target for review_tags.py (FR-5.4). In practice review_tags
    # resumes via update_state (which writes the corrected tags AND
    # human_reviewed=True as this pending node, so this body is bypassed). This
    # body remains a safety net: if the run is ever resumed without that write,
    # it still marks the document reviewed and continues to embed (FR-5.5).
    logger.info("human review node ran; marking reviewed, continuing")
    return {"human_reviewed": True, "pipeline_status": IngestStatus.RUNNING}


# --- Stage 5: Embedding (FR-6.x) --------------------------------------------
def _embed_text(chunk_dict: dict[str, Any], title: str | None) -> str:
    """Prefix the chunk text with document/section context (FR-6.3)."""
    section = chunk_dict.get("section_path") or ""
    return f"[Document: {title or ''}] [Section: {section}]\n{chunk_dict['text']}"


@_node(EMBED)
def embed(state: IngestionState) -> dict:
    config = get_config()
    enc = _encoder()
    chunks = [dict(c) for c in state.chunks]  # copy; we attach embeddings
    batch_size = config.embed_batch_size

    embed_texts = [_embed_text(c, state.title) for c in chunks]
    tokens_used = sum(len(enc.encode(t)) for t in embed_texts)  # FR-6.6 (cl100k = billed)

    try:
        for start in range(0, len(embed_texts), batch_size):  # FR-6.2 batches of 100
            batch = embed_texts[start : start + batch_size]
            vectors = _openai().embed(batch)  # FR-6.1, retries inside the client (FR-6.4)
            if len(vectors) != len(batch):
                raise ValueError(f"embedding count mismatch: got {len(vectors)} for {len(batch)} inputs")
            for offset, vector in enumerate(vectors):
                if len(vector) != config.embedding_dimensions:  # NFR-REL-8 fail fast
                    raise ValueError(
                        f"embedding dim {len(vector)} != {config.embedding_dimensions} (index mismatch)"
                    )
                chunks[start + offset]["embedding"] = vector
    except Exception as exc:  # noqa: BLE001 — FR-6.5 all retries failed → terminal, no write
        logger.error("embedding failed", exc_info=True)
        return _terminal(IngestStatus.EMBED_ERROR, f"embedding failed: {exc}")

    cost = tokens_used / 1_000_000 * EMBED_USD_PER_1M_TOKENS
    logger.info("embed ok: chunks=%d tokens=%d est_cost_usd=%.6f", len(chunks), tokens_used, cost)
    return {
        "chunks": chunks,
        "embeddings_meta": {"tokens_used": tokens_used, "estimated_cost_usd": round(cost, 6)},
        "pipeline_status": IngestStatus.RUNNING,
    }


# --- Stage 6: Neo4j atomic write (FR-7.x) -----------------------------------
_DOC_CYPHER = """
MERGE (d:Document {id: $doc_id})
SET d.title = $title,
    d.file_path = $file_path,
    d.ingested_at = $ingested_at,
    d.chunk_count = $chunk_count,
    d.language = $language
"""

_CHUNK_CYPHER = """
MATCH (d:Document {id: $doc_id})
UNWIND $chunks AS ch
CREATE (c:Chunk {
    id: ch.id, text: ch.text, content_type: ch.content_type,
    section_path: ch.section_path, position: ch.position, token_count: ch.token_count
})
WITH d, c, ch
CALL db.create.setNodeVectorProperty(c, 'embedding', ch.embedding)
MERGE (d)-[:HAS_CHUNK]->(c)
"""

# (label, relationship, list?) for the iiRDS nodes MERGE'd cross-doc (FR-7.3/7.5).
_IIRDS_RELS = (
    ("Product", "RELATES_TO_PRODUCT", "product", False),
    ("Component", "RELATES_TO_COMPONENT", "components", True),
    ("LifecyclePhase", "HAS_LIFECYCLE_PHASE", "lifecycle_phase", False),
    ("InformationType", "HAS_INFORMATION_TYPE", "information_type", False),
)


def _write_tx(tx, doc_id, params, chunks, tag_values) -> None:
    """All graph writes for one document in a single transaction (FR-7.1)."""
    tx.run(_DOC_CYPHER, doc_id=doc_id, **params)
    tx.run(_CHUNK_CYPHER, doc_id=doc_id, chunks=chunks)
    for label, rel, key, is_list in _IIRDS_RELS:
        names = tag_values.get(key)
        names = names if is_list else ([names] if names else [])
        for name in names or []:
            tx.run(
                f"MATCH (d:Document {{id: $doc_id}}) "
                f"MERGE (n:{label} {{name: $name}}) "
                f"MERGE (d)-[:{rel}]->(n)",
                doc_id=doc_id, name=name,
            )


@_node(NEO4J_WRITE)
def neo4j_write(state: IngestionState) -> dict:
    config = get_config()
    tags = state.iirds_tags or {}
    params = {
        "title": state.title,
        "file_path": state.file_path,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(state.chunks),
        "language": tags.get("language"),
    }
    try:
        with _neo4j().driver.session(database=config.neo4j_database) as session:
            # execute_write runs the unit in one transaction; any exception rolls
            # the whole thing back → zero partial data (FR-7.1/7.8, NFR-REL-1).
            session.execute_write(_write_tx, state.doc_hash, params, state.chunks, tags)
    except Exception as exc:  # noqa: BLE001 — FR-7.8 rollback → terminal, retryable
        logger.error("Neo4j write failed (rolled back)", exc_info=True)
        return _terminal(IngestStatus.WRITE_ERROR, f"Neo4j write failed: {exc}")

    chunk_ids = [c["id"] for c in state.chunks]
    logger.info("neo4j write ok: doc=%s chunks=%d", state.doc_hash[:12], len(chunk_ids))
    return {
        "neo4j_doc_id": state.doc_hash,
        "neo4j_chunk_ids": chunk_ids,
        "pipeline_status": IngestStatus.RUNNING,
    }


# --- Stage 7: Receipt (FR-8.x) ----------------------------------------------
def _append_ingestion_log(record: dict) -> None:
    """Append one record to the JSON audit trail (FR-8.1a)."""
    existing: list = []
    if INGESTION_LOG_PATH.exists():
        try:
            existing = json.loads(INGESTION_LOG_PATH.read_text(encoding="utf-8")) or []
        except (json.JSONDecodeError, OSError):
            logger.warning("ingestion_log.json unreadable; starting a fresh log")
            existing = []
    existing.append(record)
    INGESTION_LOG_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


@_node(RECEIPT)
def receipt(state: IngestionState) -> dict:
    config = get_config()
    meta = state.embeddings_meta or {}
    low_confidence = state.confidence is not None and state.confidence < config.tag_confidence_threshold
    ingested_at = datetime.now(timezone.utc).isoformat()

    record = {
        "doc_hash": state.doc_hash,
        "file_name": Path(state.file_path).name,
        "doc_title": state.title,
        "ingested_at": ingested_at,
        "chunk_count": len(state.chunks),
        "total_tokens": state.total_tokens,
        "embedding_cost_usd": meta.get("estimated_cost_usd"),
        "stage_timings_ms": state.stage_timings_ms,
        "pipeline_status": IngestStatus.COMPLETED.value,
        "low_confidence_flag": low_confidence,
    }

    # FR-8.2 either receipt write failing is a warning only — the document is
    # already durable in the graph, so the run still succeeds.
    try:
        _append_ingestion_log(record)
    except OSError:
        logger.warning("could not append ingestion_log.json (continuing)", exc_info=True)
    try:
        with _neo4j().driver.session(database=config.neo4j_database) as session:
            session.run(
                "MATCH (d:Document {id: $doc_id}) "
                "CREATE (r:IngestionRecord {ts: $ts, status: $status, chunk_count: $chunk_count, "
                "total_tokens: $total_tokens, cost_usd: $cost_usd, low_confidence: $low_confidence}) "
                "CREATE (r)-[:INGESTION_OF]->(d)",
                doc_id=state.doc_hash, ts=ingested_at, status=IngestStatus.COMPLETED.value,
                chunk_count=len(state.chunks), total_tokens=state.total_tokens,
                cost_usd=meta.get("estimated_cost_usd"), low_confidence=low_confidence,
            )
    except Exception:  # noqa: BLE001 — FR-8.2 warning only
        logger.warning("could not write IngestionRecord to Neo4j (continuing)", exc_info=True)

    logger.info("receipt ok: COMPLETED low_confidence=%s", low_confidence)
    return {"pipeline_status": IngestStatus.COMPLETED}


# === Routing (real) =========================================================
def route_after_intake(state: IngestionState) -> str:
    """Duplicate / intake error → terminal; otherwise continue to parse (FR-1.5)."""
    if state.pipeline_status is IngestStatus.DUPLICATE:
        return "duplicate"
    if state.pipeline_status is IngestStatus.INTAKE_ERROR:
        return "error"
    return PARSE


def route_after_parse(state: IngestionState) -> str:
    """Parse error → terminal; otherwise chunk (FR-2.6)."""
    return "error" if state.pipeline_status is IngestStatus.PARSE_ERROR else CHUNK


def route_after_tag(state: IngestionState, threshold: float) -> str:
    """Successful low-confidence tags (< threshold) → human_review; a tagging
    failure (confidence is None) has nothing to review → bypass to embed; a
    confident tag → embed (FR-4.7 / FR-4.5)."""
    if state.confidence is None:
        return EMBED
    return HUMAN_REVIEW if state.confidence < threshold else EMBED


def route_after_embed(state: IngestionState) -> str:
    """Embed error → terminal (no write); else neo4j_write (FR-6.5)."""
    return "error" if state.pipeline_status is IngestStatus.EMBED_ERROR else NEO4J_WRITE


def route_after_write(state: IngestionState) -> str:
    """Write error → terminal; else receipt (FR-7.8)."""
    return "error" if state.pipeline_status is IngestStatus.WRITE_ERROR else RECEIPT
