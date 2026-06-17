#!/usr/bin/env python
"""Ingestion entry point (FR-0.4).

    python ingest_document.py <path> [--ocr {auto,on,off}]

<path> may be a single document OR a folder. For a folder, every *.pdf under it
(recursive) is ingested in turn, continuing past errors/duplicates and printing
a batch summary at the end — the same behaviour the ingest_folder.ps1 wrapper
provided, now built in.

Each document gets a unique thread_id, runs the ingestion StateGraph under the
Postgres checkpointer (D15), and prints a completion summary (FR-9.1). If a run
suspends at human review (FR-5.1), resume it with: python review_tags.py <thread_id>.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from rag import PIPELINE_INGEST
from rag.cli import init_runtime, new_thread_id, thread_config
from rag.clients.postgres import open_checkpointer
from rag.ingestion.graph import build_ingestion_graph
from rag.ingestion.report import (
    EXIT_DUPLICATE,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_SUSPENDED,
    report_result,
)
from rag.ingestion.state import IngestionState
from rag.logging_config import set_context

logger = logging.getLogger(__name__)

# Exit code -> short status label, for the batch summary.
_STATUS = {
    EXIT_OK: "OK",
    EXIT_ERROR: "ERROR",
    EXIT_DUPLICATE: "DUPLICATE",
    EXIT_SUSPENDED: "SUSPENDED",
}


def ingest_one(config, checkpointer, file_path: str) -> tuple[str, int]:
    """Ingest a single document; return (thread_id, exit_code).

    `report_result` prints the per-document outcome (completion summary,
    duplicate, error, or suspension) and yields its distinguishable exit code.
    """
    thread_id = new_thread_id()
    set_context(thread_id=thread_id)
    logger.info("ingest start: %s", file_path)

    state = IngestionState(file_path=file_path, thread_id=thread_id)
    started = time.perf_counter()
    graph = build_ingestion_graph(config, checkpointer)
    cfg = thread_config(thread_id)
    result = graph.invoke(state, config=cfg)
    elapsed = time.perf_counter() - started
    logger.info("ingest finished: thread_id=%s", thread_id)
    # get_state (suspension check) must run while the checkpointer is open.
    return thread_id, report_result(graph, cfg, result, thread_id, elapsed)


def _ingest_folder(config, checkpointer, folder: Path) -> int:
    """Batch-ingest every *.pdf under `folder` (recursive); return an exit code.

    Continues past per-file errors/duplicates and prints a summary. The overall
    exit code is EXIT_ERROR if any document errored, else EXIT_OK.
    """
    pdfs = sorted(folder.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {folder}")
        return EXIT_OK

    n = len(pdfs)
    print(f"Found {n} PDF(s) under {folder}\n")

    rows: list[tuple[str, str, str]] = []  # (seq, filename, status)
    for idx, pdf in enumerate(pdfs, start=1):
        bar = "=" * 67
        print(f"{bar}\n[{idx}/{n}] {pdf.name}\n{bar}")
        thread_id, code = ingest_one(config, checkpointer, str(pdf))
        status = _STATUS.get(code, f"EXIT_{code}")
        rows.append((f"{idx}/{n}", pdf.name, status))
        print(f"\n-> {pdf.name}: {status}\n")

    _print_summary(rows)
    return EXIT_ERROR if any(s == "ERROR" for _, _, s in rows) else EXIT_OK


def _print_summary(rows: list[tuple[str, str, str]]) -> None:
    print("\n===================== SUMMARY =====================")
    seq_w = max((len(r[0]) for r in rows), default=3)
    file_w = max((len(r[1]) for r in rows), default=4)
    for seq, name, status in rows:
        print(f"  {seq:<{seq_w}}  {name:<{file_w}}  {status}")

    counts = {label: 0 for label in ("OK", "DUPLICATE", "ERROR", "SUSPENDED")}
    for _, _, status in rows:
        if status in counts:
            counts[status] += 1
    print(
        f"\nOK={counts['OK']}  DUPLICATE={counts['DUPLICATE']}  "
        f"ERROR={counts['ERROR']}  SUSPENDED={counts['SUSPENDED']}  "
        f"(total {len(rows)})"
    )
    if counts["SUSPENDED"]:
        print(
            "\nSUSPENDED docs need a tag review to finish ingesting:\n"
            "  python review_tags.py <thread_id>   "
            "(the thread_id is printed above each suspended run)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a document, or every *.pdf under a folder (recursive)."
    )
    parser.add_argument("path", help="path to a document or a folder of PDFs")
    parser.add_argument(
        "--ocr",
        choices=("auto", "on", "off"),
        help="override OCR for this run (sets OCR_ENABLED); unset = use .env",
    )
    args = parser.parse_args(argv)

    # Must be set before init_runtime -> get_config() (lru_cached) reads it.
    if args.ocr:
        os.environ["OCR_ENABLED"] = args.ocr

    config = init_runtime(PIPELINE_INGEST)
    target = Path(args.path)

    with open_checkpointer(config) as checkpointer:
        if target.is_dir():
            return _ingest_folder(config, checkpointer, target)
        _, code = ingest_one(config, checkpointer, args.path)
        return code


if __name__ == "__main__":
    import sys

    sys.exit(main())
