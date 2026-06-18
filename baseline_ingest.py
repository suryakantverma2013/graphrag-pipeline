#!/usr/bin/env python
"""Ingest PDFs into the SIMPLE embedding-only baseline store (benchmark only).

    python baseline_ingest.py [--reset] PATH [PATH ...]

Each PATH is a PDF file or a directory (searched for *.pdf, non-recursively;
add --recursive to descend). Text is extracted with pypdfium2 (NO OCR), chunked
and embedded exactly as the hybrid pipeline does, and written to the Postgres +
pgvector table `baseline_chunks`. Re-ingesting the same file replaces its rows.

For a FAIR head-to-head, ingest the SAME source PDFs that are in the Neo4j graph
so both stores cover the same corpus. (Documents whose PDFs aren't on disk —
e.g. the Documentum guides — simply won't be in the baseline; the benchmark
flags queries that fall outside the baseline corpus.)

    # mirror the part of the corpus you have on disk
    python baseline_ingest.py --reset samples/Java-TheCompleteReference-11Edition.pdf
    python baseline_ingest.py samples/JavaBooks samples/the-feynman-lectures-on-physics-vol1_compress.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.baseline import ingest_pdf
from rag.cli import init_runtime
from rag.clients.openai_client import OpenAIClient
from rag.clients.pgvector_client import PgVectorStore

PIPELINE = "baseline_ingest"


def _collect_pdfs(paths: list[str], recursive: bool) -> list[Path]:
    pdfs: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf") if recursive else p.glob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            print(f"skipping (not a PDF or directory): {p}", file=sys.stderr)
    # de-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in pdfs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the baseline pgvector store.")
    parser.add_argument("paths", nargs="+", help="PDF files and/or directories")
    parser.add_argument("--reset", action="store_true",
                        help="drop and recreate the baseline table before ingesting")
    parser.add_argument("--recursive", action="store_true",
                        help="descend into sub-directories when a PATH is a directory")
    args = parser.parse_args(argv)

    config = init_runtime(PIPELINE)
    pdfs = _collect_pdfs(args.paths, args.recursive)
    if not pdfs:
        print("no PDFs found in the given paths", file=sys.stderr)
        return 2

    store = PgVectorStore(config)
    store.ensure_schema(reset=args.reset)
    openai = OpenAIClient(config)

    print(f"Ingesting {len(pdfs)} PDF(s) into baseline store (reset={args.reset})\n")
    total_chunks = 0
    total_cost = 0.0
    failures = 0
    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] {pdf.name} ...", flush=True)
        try:
            stats = ingest_pdf(pdf, store, openai, config)
        except Exception as exc:  # noqa: BLE001 — keep going, report at the end
            failures += 1
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        total_chunks += stats["chunk_count"]
        total_cost += stats["embedding_cost_usd"]
        print(
            f"    ok: {stats['chunk_count']} chunks, {stats['total_tokens']} tokens, "
            f"~${stats['embedding_cost_usd']:.5f}  (extract {stats['extract_ms']:.0f}ms, "
            f"embed {stats['embed_ms']:.0f}ms)"
        )

    print(
        f"\nDone. store now holds {store.count()} chunk(s) across "
        f"{len(store.documents())} document(s). "
        f"This run: +{total_chunks} chunks, ~${total_cost:.5f} embeddings, {failures} failure(s)."
    )
    return 1 if failures and total_chunks == 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
