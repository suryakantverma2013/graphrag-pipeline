#!/usr/bin/env python
"""Ingestion entry point (FR-0.4).

    python ingest_document.py <path>

Generates a unique thread_id, runs the ingestion StateGraph under the Postgres
checkpointer (D15), and prints a completion summary (FR-9.1). If the run
suspends at human review (FR-5.1), resume with: python review_tags.py <thread_id>.
"""

from __future__ import annotations

import logging
import sys
import time

from rag import PIPELINE_INGEST
from rag.cli import init_runtime, new_thread_id, thread_config
from rag.clients.postgres import open_checkpointer
from rag.ingestion.graph import build_ingestion_graph
from rag.ingestion.report import report_result
from rag.ingestion.state import IngestionState
from rag.logging_config import set_context

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python ingest_document.py <path>", file=sys.stderr)
        return 2
    file_path = argv[0]

    config = init_runtime(PIPELINE_INGEST)
    thread_id = new_thread_id()
    set_context(thread_id=thread_id)
    logger.info("ingest start: %s", file_path)

    state = IngestionState(file_path=file_path, thread_id=thread_id)
    started = time.perf_counter()
    with open_checkpointer(config) as checkpointer:
        graph = build_ingestion_graph(config, checkpointer)
        cfg = thread_config(thread_id)
        result = graph.invoke(state, config=cfg)
        elapsed = time.perf_counter() - started
        logger.info("ingest finished: thread_id=%s", thread_id)
        # get_state (suspension check) must run while the checkpointer is open.
        return report_result(graph, cfg, result, thread_id, elapsed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
