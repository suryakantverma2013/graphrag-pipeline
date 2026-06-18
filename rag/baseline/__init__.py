"""The 'simple' embedding-only baseline pipeline (benchmark only).

Deliberately the naive end-to-end counterpart to the production hybrid stack:

    extract PDF text (pypdfium2, NO OCR)  ->  chunk  ->  embed  ->  pgvector
    query  ->  embed  ->  pgvector cosine top-k

No Docling layout/table parsing, no OCR, no iiRDS graph, no BM25, no
cross-encoder rerank. It shares the SAME chunker, embedding model, and
context-prefix as ingestion so the only things that differ from the hybrid
pipeline are the PARSE step (plain text vs Docling) and the RETRIEVAL step
(cosine-only vs hybrid+rerank) — which is exactly what the benchmark measures.
"""

from .pipeline import extract_pdf_text, ingest_pdf, naive_query

__all__ = ["extract_pdf_text", "ingest_pdf", "naive_query"]
