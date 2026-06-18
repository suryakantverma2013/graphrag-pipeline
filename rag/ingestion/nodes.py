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

# Per-run ingestion audit trail (FR-8.1a / NFR-OBS-4); distinct from the app log
# (NFR-LOG-8). Each ingestion run writes ONE JSON file into this directory, named
# by timestamp + thread_id + source stem, so runs never overwrite each other and
# the directory sorts chronologically. Repo-root relative; git-ignored.
INGESTION_LOG_DIR = Path("ingestion_logs")


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


# Process-local handoff for the parsed DoclingDocument from `parse` to `chunk`.
# The tree is heavy and NOT msgpack-serializable, so it must never enter the
# checkpointed state (the Postgres checkpointer would fail on it). `parse` and
# `chunk` always run back-to-back in the same process — the only interrupt is at
# human_review, well after chunk — so a per-thread_id stash is safe. Keyed by
# thread_id; popped (and thus freed) in `chunk`.
_DOC_HANDOFF: dict[str, Any] = {}


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
def _pdf_requires_password(path: Path) -> bool:
    """True only when a PDF can't be opened without a password (encrypted).

    Other open failures (corrupt, etc.) return False — they're left for the
    parse stage to report. This exists only to give a clear, early message for
    the common 'password-protected' case (FR-1.2a). Detection is by the pdfium
    error message so it's robust across pypdfium2 versions.
    """
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError:
        return False
    try:
        doc = pdfium.PdfDocument(str(path))
        doc.close()
        return False
    except Exception as exc:  # noqa: BLE001 — classify by error, not type
        # Precise signal: pdfium's password error code; fall back to the message
        # text for any version that doesn't surface err_code.
        if getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
            return True
        return "password" in str(exc).lower()


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

    # FR-1.2a password-protected (encrypted) PDFs can't be parsed — fail here
    # with a clear, actionable message instead of a generic downstream parse error.
    if suffix == ".pdf" and _pdf_requires_password(path):
        return _terminal(
            IngestStatus.INTAKE_ERROR,
            f"password-protected (encrypted) PDF is not supported: {path.name}. "
            "Remove the password (e.g. re-save or print to an unprotected PDF) and re-ingest.",
        )

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
def _pdf_metadata_title(source: Path) -> str:
    """The embedded PDF ``/Title`` (document info dict), if present (best-effort).

    Read locally from the original file via pypdfium2 (a Docling dependency, so
    no new requirement and no egress, NFR-SEC-5). Never fatal — any failure or a
    non-PDF source returns "" so the caller falls back to the file stem.
    """
    if source.suffix.lower() != ".pdf":
        return ""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(source))
        try:
            return (pdf.get_metadata_dict().get("Title") or "").strip()
        finally:
            pdf.close()
    except Exception:  # noqa: BLE001 — metadata is a nicety, never block ingest
        logger.debug("could not read PDF /Title metadata for %s", source, exc_info=True)
        return ""


def _document_title(doc: Any, source: Path) -> str:
    """Resolve a human-readable title (FR-2.7).

    Order: an on-page Docling TITLE element → the embedded PDF ``/Title`` →
    the original file stem. Docling's ``doc.name`` is deliberately NOT used: we
    parse from a private temp file (FR-2.5), so it is always the ``rag_ingest_*``
    temp stem, never anything meaningful.
    """
    from docling_core.types.doc import DocItemLabel

    for item, _level in doc.iterate_items():
        if getattr(item, "label", None) == DocItemLabel.TITLE:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                return text
    return _pdf_metadata_title(source) or source.stem


def _pdf_page_count(path: Path) -> int | None:
    """Total PDF page count via pypdfium2 (cheap, no full parse). None on failure."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:  # noqa: BLE001 — fall back to a single convert if unknown
        logger.debug("could not read PDF page count for %s", path, exc_info=True)
        return None


def _plan_page_ranges(path: Path, config: AppConfig, is_pdf: bool) -> list[tuple[int, int] | None]:
    """Plan the Docling convert calls (FR-2.8c).

    Non-PDF → one convert with no page_range. PDF → respect `PDF_MAX_PAGES` (cap)
    and split into `PDF_PARSE_BATCH_PAGES`-page slices so Docling releases memory
    between slices (avoids the cumulative std::bad_alloc on very large PDFs). A
    `None` entry means "convert the whole file" (legacy single-shot path)."""
    if not is_pdf:
        return [None]
    total = _pdf_page_count(path)
    cap = config.pdf_max_pages
    batch = config.pdf_parse_batch_pages
    if total is None:  # unknown page count → single convert (still honor the cap)
        return [(1, cap)] if cap > 0 else [None]
    last = min(total, cap) if cap > 0 else total
    if batch > 0 and last > batch:
        return [(lo, min(lo + batch - 1, last)) for lo in range(1, last + 1, batch)]
    return [(1, last)] if cap > 0 else [None]


@_node(PARSE)
def parse(state: IngestionState) -> dict:
    from docling.datamodel.base_models import ConversionStatus

    src = Path(state.file_path)
    is_pdf = src.suffix.lower() == ".pdf"
    # FR-2.5 materialize raw bytes to a private temp file for parsing, then delete
    # immediately (success or failure). mkstemp creates an owner-only file
    # (NFR-SEC-6); raw bytes never leave the device (FR-2.1, local-only Docling).
    fd, temp_name = tempfile.mkstemp(suffix=src.suffix.lower(), prefix="rag_ingest_")
    temp_path = Path(temp_name)
    config = get_config()
    # FR-2.8c: large PDFs are parsed in page-range slices so Docling releases
    # memory between convert() calls (it otherwise OOMs on very large books). One
    # reused converter; each slice's finished DoclingDocument is kept, transient
    # render buffers are freed. Non-PDF / small PDF → a single (legacy) convert.
    docs: list[Any] = []
    failed_pages = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(src.read_bytes())
        # FR-2.3e: resolve OCR per-document so OCR_ENABLED never needs hand-tuning
        # per file. In 'auto' mode (default) a PDF is classified from its own text
        # layer (digital vs scanned/mixed) and OCR is enabled only when needed;
        # 'on'/'off' force it. Non-PDF inputs never OCR. Detection reads the temp
        # file written above. OCR is heavy per page, so it also selects the
        # smaller OCR slice size; both feed an effective per-document config.
        pdf_kind = None
        if is_pdf and config.ocr_mode == "auto":
            from .pdf_kind import analyze

            kind_result = analyze(str(temp_path))
            pdf_kind = kind_result.kind
            logger.info("pdf kind=%s — %s", pdf_kind, kind_result.reason)
        do_ocr = config.resolve_ocr(pdf_kind) if is_pdf else False
        # OCR is heavy per page, so it uses the smaller slice; and the scanned-page
        # bitmap is the only text source, so it also renders at the sharper OCR DPI
        # (pdf_render_dpi_ocr) for legibility. Born-digital parsing keeps the 72-dpi
        # backdrop. Both fold into an effective per-document config.
        eff_batch = config.pdf_parse_batch_pages_ocr if do_ocr else config.pdf_parse_batch_pages
        eff_updates: dict[str, Any] = {"pdf_parse_batch_pages": eff_batch}
        if do_ocr:
            eff_updates["pdf_render_dpi"] = config.pdf_render_dpi_ocr
        eff_config = config.model_copy(update=eff_updates)
        # Plan slices AFTER writing bytes (page count is read from the temp file).
        page_ranges = _plan_page_ranges(temp_path, eff_config, is_pdf)
        n_slices = len(page_ranges)
        if n_slices > 1:
            logger.info("parsing PDF in %d page-slice(s) of <=%d pages (ocr=%s)",
                        n_slices, eff_batch, do_ocr)
        # FR-2.1/2.2/2.3 local Docling (DocLayNet + TableFormer + EasyOCR, GPU).
        converter = build_converter(eff_config, do_ocr=do_ocr)
        for i, page_range in enumerate(page_ranges, start=1):
            kwargs: dict[str, Any] = {"page_range": page_range} if page_range else {}
            result = converter.convert(str(temp_path), **kwargs)
            if result.status == ConversionStatus.FAILURE:
                where = f" (pages {page_range[0]}-{page_range[1]})" if page_range else ""
                return _terminal(IngestStatus.PARSE_ERROR, f"Docling reported FAILURE for {src.name}{where}")
            failed_pages += len(getattr(result, "errors", []) or [])
            docs.append(result.document)
            # Per-slice progress (FR-2.8c): without this a long OCR run looks hung,
            # since the convert loop is otherwise silent until "parse ok".
            if n_slices > 1:
                span = f" pages {page_range[0]}-{page_range[1]}" if page_range else ""
                logger.info("parsed slice %d/%d%s", i, n_slices, span)
            del result  # release this slice's transient buffers before the next
    except Exception as exc:  # noqa: BLE001 — any parse failure → terminal (FR-2.6)
        logger.error("Docling parse raised", exc_info=True)
        return _terminal(IngestStatus.PARSE_ERROR, f"parse failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)  # FR-2.5 delete temp immediately

    if not any(d.export_to_text().strip() for d in docs):  # FR-2.6 zero content
        return _terminal(IngestStatus.PARSE_ERROR, f"parsed document has no extractable text: {src.name}")

    # Title from the first slice (TITLE element is on page 1) → PDF /Title → stem.
    title = _document_title(docs[0], src)
    page_count = sum(d.num_pages() for d in docs)
    logger.info("parse ok: title=%r pages=%s slices=%d failed_pages=%d",
                title, page_count, len(docs), failed_pages)
    # FR-2.6a: a partial parse (some pages dropped under memory pressure) must NOT
    # masquerade as a clean ingest — count the dropped pages so the run can WARN.
    if failed_pages:
        logger.warning(
            "Docling parse INCOMPLETE for %s: %d/%d page(s) failed — content is missing",
            src.name, failed_pages, page_count,
        )
    # Hand the heavy Docling trees to `chunk` out-of-band (NOT via checkpointed
    # state — not msgpack-serializable; see _DOC_HANDOFF). chunk consumes the list
    # in order with continuous section/position tracking.
    _DOC_HANDOFF[state.thread_id] = docs
    return {
        "title": title,
        "page_count": page_count,
        "parse_failed_pages": failed_pages,
        "pipeline_status": IngestStatus.RUNNING,
    }


# --- Stage 3: Structural chunking (FR-3.x) ----------------------------------
# Token budget reserved for the context prefix that `_embed_text` prepends at
# embed time ("[Document: <title>] [Section: <section_path>]\n"). The per-chunk
# token ceiling is `embed_max_input_tokens` MINUS this margin, so the prefixed
# text still fits the embedding model's 8192-token hard limit (FR-6.x).
_EMBED_PREFIX_TOKEN_MARGIN = 256


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
    docs = _DOC_HANDOFF.pop(state.thread_id, None)
    if not docs:  # defensive: parse must have run in THIS process (FR-2.7)
        return _terminal(
            IngestStatus.PARSE_ERROR,
            "no parsed document available for chunking (process restarted mid-run?)",
        )

    chunks: list[dict[str, Any]] = []
    section_stack: list[tuple[int, str]] = []  # (level, heading text)
    text_buffer: list[str] = []  # accumulated paragraph text within a section

    def section_path() -> str:
        parts = [state.title] if state.title else []
        parts.extend(text for _, text in section_stack)
        return " > ".join(parts)

    # Per-chunk token ceiling: the embedding model's hard input limit, less the
    # margin for the context prefix added at embed time (FR-6.x). No chunk may
    # exceed this or the embed API rejects the whole batch with HTTP 400.
    embed_cap = max(1, config.embed_max_input_tokens - _EMBED_PREFIX_TOKEN_MARGIN)

    def _emit_chunk(text: str, content_type: str, token_count: int) -> None:
        position = len(chunks)
        chunks.append(
            {
                "id": f"{state.doc_hash}#{position}",
                "text": text,
                "content_type": content_type,
                "section_path": section_path(),
                "document_title": state.title,  # FR-3.5
                "position": position,
                "token_count": token_count,
            }
        )

    def add_chunk(text: str, content_type: str) -> None:
        text = text.strip()
        if not text:
            return
        tokens = enc.encode(text)
        if len(tokens) <= embed_cap:
            _emit_chunk(text, content_type, len(tokens))
            return
        # Oversized whole-block chunk (large table / formula / code / list run).
        # Prose is already pre-split to chunk_max_tokens, so this only triggers on
        # the "kept whole" content types — hard-split it into <=embed_cap windows
        # so the embedding API never rejects it (FR-6.x). Logged so the split is
        # visible: the block's internal structure is necessarily broken here.
        logger.warning(
            "chunk exceeds embed cap (%d > %d tokens); hard-splitting %s block into %d parts",
            len(tokens), embed_cap, content_type,
            (len(tokens) + embed_cap - 1) // embed_cap,
        )
        for window in _split_long_sentence(tokens, embed_cap, enc):
            _emit_chunk(enc.decode(window), content_type, len(window))

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

    # Consume each page-slice in order. Accumulators (chunks, section_stack,
    # text/list buffers) persist ACROSS slices so a section heading, paragraph, or
    # list that straddles a batch boundary stays continuous — we only flush at
    # section headers and once at the very end, never between slices (FR-2.8c).
    for doc in docs:
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
    # The Docling tree was popped from the handoff above, so it is already freed;
    # nothing heavy or non-serializable enters the downstream checkpoints.
    return {
        "chunks": chunks,
        "total_tokens": total_tokens,
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


def _write_doc_tx(tx, doc_id, params, tag_values) -> None:
    """Write the Document node + its iiRDS relationships in one transaction.

    Small and bounded (one doc + a handful of MERGE'd tag nodes). Runs FIRST so
    the chunk batches below have an anchor Document to MATCH (FR-7.1/7.3/7.5).
    """
    tx.run(_DOC_CYPHER, doc_id=doc_id, **params)
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


def _write_chunks_tx(tx, doc_id, chunks) -> None:
    """Write ONE batch of chunks + their HAS_CHUNK edges in a single transaction.

    Called once per batch (config.neo4j_write_batch_chunks) so no single
    transaction is large enough to fail the store-apply that wedged the DB.
    """
    tx.run(_CHUNK_CYPHER, doc_id=doc_id, chunks=chunks)


def _delete_partial_write(session, doc_id) -> None:
    """Compensating cleanup for a failed multi-batch write (FR-7.8).

    The per-document write is no longer a single transaction, so a mid-write
    failure can leave the Document plus some already-committed chunk batches.
    Delete them so a failed run still leaves ZERO partial data (all-or-nothing).
    Chunks are removed in bounded sub-transactions so the cleanup itself can never
    become an oversized transaction. iiRDS nodes are shared across documents
    (MERGE'd), so DETACH DELETE of the Document removes only this document's edges
    to them, never the shared nodes themselves.
    """
    session.run(
        "MATCH (:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
        "CALL { WITH c DETACH DELETE c } IN TRANSACTIONS OF 1000 ROWS",
        doc_id=doc_id,
    ).consume()
    session.run(
        "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
        doc_id=doc_id,
    ).consume()


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
    batch_size = max(1, config.neo4j_write_batch_chunks)
    chunks = state.chunks
    try:
        with _neo4j().driver.session(database=config.neo4j_database) as session:
            # The whole document used to be ONE transaction (FR-7.1). On very large
            # books that single transaction reached ~240k store commands and Neo4j
            # failed to apply it to the store, wedging the database. We now write
            # the Document + iiRDS edges first, then the chunks in bounded batches
            # (one transaction each) so no single transaction can grow large enough
            # to fail. Atomicity is preserved by COMPENSATION: any failure deletes
            # whatever was committed, leaving zero partial data (FR-7.8, NFR-REL-1).
            session.execute_write(_write_doc_tx, state.doc_hash, params, tags)
            for start in range(0, len(chunks), batch_size):
                session.execute_write(_write_chunks_tx, state.doc_hash, chunks[start : start + batch_size])
    except Exception as exc:  # noqa: BLE001 — FR-7.8 compensate → terminal, retryable
        logger.error("Neo4j write failed; compensating (deleting partial data)", exc_info=True)
        try:
            with _neo4j().driver.session(database=config.neo4j_database) as session:
                _delete_partial_write(session, state.doc_hash)
        except Exception:  # noqa: BLE001 — cleanup best-effort; surface the original cause
            logger.error("compensating cleanup failed for doc=%s (manual cleanup may be needed)",
                         state.doc_hash[:12], exc_info=True)
        return _terminal(IngestStatus.WRITE_ERROR, f"Neo4j write failed: {exc}")

    chunk_ids = [c["id"] for c in state.chunks]
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info("neo4j write ok: doc=%s chunks=%d batches=%d",
                state.doc_hash[:12], len(chunk_ids), n_batches)
    return {
        "neo4j_doc_id": state.doc_hash,
        "neo4j_chunk_ids": chunk_ids,
        "pipeline_status": IngestStatus.RUNNING,
    }


# --- Stage 7: Receipt (FR-8.x) ----------------------------------------------
def _safe_slug(text: str, maxlen: int = 40) -> str:
    """Filesystem-safe, truncated slug for embedding the source name in a log
    filename (keeps alnum / dot / dash / underscore)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned[:maxlen] or "doc"


def _write_ingestion_log(record: dict, thread_id: str, stamp: str) -> Path:
    """Write a fresh per-run JSON audit file (FR-8.1a).

    One file per ingestion run (not an appended cumulative log), named by
    timestamp + thread_id + source stem so runs never overwrite each other and
    the directory sorts chronologically. Returns the path written.
    """
    INGESTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(Path(record["file_name"]).stem)
    path = INGESTION_LOG_DIR / f"ingestion_log_{stamp}_{thread_id[:8]}_{slug}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@_node(RECEIPT)
def receipt(state: IngestionState) -> dict:
    config = get_config()
    meta = state.embeddings_meta or {}
    low_confidence = state.confidence is not None and state.confidence < config.tag_confidence_threshold
    now = datetime.now(timezone.utc)
    ingested_at = now.isoformat()

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
        log_path = _write_ingestion_log(record, state.thread_id, now.strftime("%Y%m%dT%H%M%SZ"))
        logger.info("ingestion log written: %s", log_path)
    except OSError:
        logger.warning("could not write ingestion log (continuing)", exc_info=True)
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
