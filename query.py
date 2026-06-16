#!/usr/bin/env python
"""Query entry point (FR-Q0.2 / FR-Q4.4).

    python query.py "<question>"

Generates a unique thread_id, runs the query StateGraph under the Postgres
checkpointer (D15), and prints the cited answer (FR-Q4.4). If the run suspends
at clarification (FR-Q1.5) or escalation (FR-Q3.6), resume with:
    python resume_query.py <thread_id> (D14).
"""

from __future__ import annotations

import logging
import sys
import time

from rag import PIPELINE_QUERY
from rag.cli import init_runtime, new_thread_id, thread_config
from rag.clients.postgres import open_checkpointer
from rag.logging_config import set_context
from rag.query.graph import build_query_graph
from rag.query.report import report_query_result
from rag.query.state import QueryState

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not argv[0].strip():
        print('usage: python query.py "<question>"', file=sys.stderr)
        return 2
    question = argv[0].strip()

    config = init_runtime(PIPELINE_QUERY)
    thread_id = new_thread_id()
    set_context(thread_id=thread_id)
    logger.info("query start: %r", question)

    state = QueryState(original_query=question, thread_id=thread_id)
    started = time.perf_counter()
    with open_checkpointer(config) as checkpointer:
        graph = build_query_graph(config, checkpointer)
        cfg = thread_config(thread_id)
        result = graph.invoke(state, config=cfg)
        elapsed = time.perf_counter() - started
        logger.info("query finished: thread_id=%s", thread_id)
        # get_state (suspension check) must run while the checkpointer is open.
        return report_query_result(graph, cfg, result, thread_id, elapsed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
