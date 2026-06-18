"""Naive ingest + query for the embedding-only baseline (benchmark only).

Reuses the production chunker (`_pack_sentences`), the embedding context-prefix
(`_embed_text`), the tokenizer, and the OpenAI seam from `rag.ingestion.nodes`
so chunk sizing and embedding inputs are IDENTICAL to the hybrid pipeline. The
two things that are intentionally different (and that the benchmark isolates):

  * PARSE  — plain pypdfium2 text extraction, no OCR, no Docling layout/tables.
  * STORE  — Postgres + pgvector, queried by pure cosine (no BM25/graph/rerank).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from ..clients.openai_client import OpenAIClient
from ..clients.pgvector_client import PgVectorStore
from ..config import AppConfig
# Reuse the EXACT production helpers so chunking + embedding inputs match the
# hybrid pipeline (the imports are cheap — docling/torch are lazy in those seams).
from ..ingestion.nodes import (
    EMBED_USD_PER_1M_TOKENS,
    _embed_text,
    _encoder,
    _pack_sentences,
    _pdf_metadata_title,
)

logger = logging.getLogger(__name__)


def extract_pdf_text(path: Path) -> str:
    """Concatenate every page's text layer via pypdfium2 — NO OCR (FR baseline).

    This is the whole 'simple' parse: a scanned/image page contributes nothing
    (no text layer), exactly the limitation the hybrid pipeline's OCR path
    exists to remove. Born-digital PDFs extract cleanly.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        parts: list[str] = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            try:
                parts.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
        return "\n".join(parts)
    finally:
        doc.close()


def ingest_pdf(
    path: Path,
    store: PgVectorStore,
    openai: OpenAIClient,
    config: AppConfig,
) -> dict[str, Any]:
    """Extract → chunk → embed → store one PDF into the baseline pgvector table.

    Idempotent per file: the document id is the SHA-256 of the raw bytes (same
    scheme as ingestion intake), so re-ingesting the same PDF replaces its rows
    rather than duplicating them, and the id aligns with the Neo4j Document id
    for the same file (lets the benchmark match documents across both stores).
    """
    raw = path.read_bytes()
    doc_hash = hashlib.sha256(raw).hexdigest()
    title = _pdf_metadata_title(path) or path.stem
    enc = _encoder()

    t0 = time.perf_counter()
    text = extract_pdf_text(path)
    extract_ms = (time.perf_counter() - t0) * 1000.0
    if not text.strip():
        raise ValueError(
            f"no extractable text in {path.name} (likely a scanned PDF — the simple "
            "baseline has no OCR, so it cannot read it)"
        )

    # Same chunker + params as ingestion (FR-3.4). No structure is available from
    # plain text, so section_path is empty; the embedding prefix still matches.
    pieces = _pack_sentences(text, config.chunk_max_tokens, config.chunk_overlap_tokens, enc)
    chunks: list[dict[str, Any]] = []
    for position, (chunk_text, _ntok) in enumerate(pieces):
        chunks.append(
            {
                "chunk_id": f"{doc_hash}#{position}",
                "doc_hash": doc_hash,
                "document_title": title,
                "section_path": "",
                "position": position,
                "text": chunk_text,
            }
        )
    if not chunks:
        raise ValueError(f"chunking produced zero chunks for {path.name}")

    # Embed with the same model/dim/prefix as ingestion (FR-6.x), batched.
    embed_texts = [_embed_text(c, title) for c in chunks]
    tokens_used = sum(len(enc.encode(t)) for t in embed_texts)
    batch_size = config.embed_batch_size
    t0 = time.perf_counter()
    for start in range(0, len(embed_texts), batch_size):
        batch = embed_texts[start : start + batch_size]
        vectors = openai.embed(batch)
        if len(vectors) != len(batch):
            raise ValueError(f"embedding count mismatch: {len(vectors)} for {len(batch)}")
        for offset, vector in enumerate(vectors):
            chunks[start + offset]["embedding"] = vector
    embed_ms = (time.perf_counter() - t0) * 1000.0

    store.delete_document(doc_hash)
    written = store.upsert_chunks(chunks)

    cost = tokens_used / 1_000_000 * EMBED_USD_PER_1M_TOKENS
    logger.info(
        "baseline ingest ok: %s title=%r chunks=%d tokens=%d est_cost_usd=%.6f",
        path.name, title, written, tokens_used, cost,
    )
    return {
        "file_name": path.name,
        "doc_hash": doc_hash,
        "title": title,
        "chunk_count": written,
        "total_tokens": tokens_used,
        "embedding_cost_usd": round(cost, 6),
        "extract_ms": round(extract_ms, 1),
        "embed_ms": round(embed_ms, 1),
    }


def naive_query(
    query: str,
    store: PgVectorStore,
    openai: OpenAIClient,
    k: int,
) -> list[dict[str, Any]]:
    """The entire 'simple' retriever: embed the raw query, cosine top-k.

    No refine/decompose/classify, no rerank — the raw user query is embedded
    once and matched against the store, which is precisely the 'simple embedding
    search' the comparison is measured against.
    """
    vec = openai.embed([query])[0]
    return store.search(vec, k)
