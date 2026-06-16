"""Shared post-run reporting for the ingestion CLIs (NFR-MAINT-7).

Both `ingest_document.py` and `review_tags.py` finish a run the same way: detect
a human-review suspension (FR-5.1), print the completion summary (FR-9.1), or
report a terminal status — each with a distinguishable exit code (FR-9.2 /
FR-0.7). Factored here so the two entry points cannot drift.
"""

from __future__ import annotations

import logging
import sys

from .state import IngestStatus

logger = logging.getLogger(__name__)

# Distinguishable exit codes (FR-9.2 / FR-0.7).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DUPLICATE = 3
EXIT_SUSPENDED = 4


def _status_of(result: dict) -> IngestStatus:
    status = result.get("pipeline_status")
    return status if isinstance(status, IngestStatus) else IngestStatus(status)


def report_result(graph, cfg: dict, result: dict, thread_id: str, elapsed: float) -> int:
    """Print the run outcome and return its exit code.

    `graph`/`cfg` are the compiled graph and its thread config, used to detect a
    pending interrupt; `result` is the value dict returned by `graph.invoke`.
    """
    # Suspended at the human-review interrupt (FR-5.1): the graph stopped before
    # that node, so a next-step is pending (FR-5.4 resume path).
    if graph.get_state(cfg).next:
        print(
            f"\nSuspended for human tag review (low confidence).\n"
            f"  Resume with: python review_tags.py {thread_id}\n"
        )
        return EXIT_SUSPENDED

    status = _status_of(result)
    if status is IngestStatus.COMPLETED:
        meta = result.get("embeddings_meta") or {}
        print(
            "\nIngestion complete:\n"
            f"  title:       {result.get('title')}\n"
            f"  chunks:      {len(result.get('chunks') or [])}\n"
            f"  embed cost:  ${meta.get('estimated_cost_usd', 0):.6f} "
            f"({meta.get('tokens_used', 0)} tokens)\n"
            f"  wall-clock:  {elapsed:.1f}s\n"
            f"  thread_id:   {thread_id}\n"
        )
        return EXIT_OK
    if status is IngestStatus.DUPLICATE:
        print(
            "\nDUPLICATE: this document is already ingested (re-ingestion not supported).\n",
            file=sys.stderr,
        )
        return EXIT_DUPLICATE

    # Terminal error (INTAKE/PARSE/EMBED/WRITE) — no partial graph data (FR-0.7).
    print(f"\n{status.value}: {result.get('error')}\n", file=sys.stderr)
    return EXIT_ERROR
