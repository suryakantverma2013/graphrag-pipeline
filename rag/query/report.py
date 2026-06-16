"""Shared post-run reporting for the query CLIs (NFR-MAINT-7).

`query.py` and `resume_query.py` finish a run the same way: print the cited
answer (FR-Q4.4), or report a clarification/escalation suspension with the resume
command, each with a distinguishable exit code (FR-Q0.7 / FR-9.2). Factored here
— mirrors `rag/ingestion/report.py` — so the two entry points cannot drift.
"""

from __future__ import annotations

import logging
import sys

from .nodes import ESCALATE, REQUEST_CLARIFICATION

logger = logging.getLogger(__name__)

# Distinguishable exit codes (FR-Q0.7 / FR-9.2), aligned with ingestion.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SUSPENDED = 4


def report_query_result(graph, cfg: dict, result: dict, thread_id: str, elapsed: float) -> int:
    """Print the run outcome and return its exit code.

    `graph`/`cfg` detect a pending interrupt (clarification or escalation);
    `result` is the value dict returned by `graph.invoke`.
    """
    pending = graph.get_state(cfg).next or ()

    # Suspended at one of the two interrupts (FR-Q1.5 / FR-Q3.6). Both resume via
    # the same thread_id CLI (D14); the message names which review is needed.
    if pending:
        if REQUEST_CLARIFICATION in pending:
            question = (result or {}).get("clarification_question")
            print(
                "\nSuspended for clarification (low-confidence refinement).\n"
                + (f"  Question: {question}\n" if question else "")
                + f"  Resume with: python resume_query.py {thread_id}\n"
            )
        elif ESCALATE in pending:
            print(
                "\nSuspended for expert review (low-confidence retrieval).\n"
                f"  Resume with: python resume_query.py {thread_id}\n"
            )
        else:  # pragma: no cover — defensive
            print(f"\nSuspended (next={pending}). Resume with: python resume_query.py {thread_id}\n")
        return EXIT_SUSPENDED

    answer = (result or {}).get("answer")
    if not answer:
        print("\nQuery failed: no answer was produced.\n", file=sys.stderr)
        return EXIT_ERROR

    citations = result.get("citations") or []
    low_conf = result.get("low_confidence_flag")
    lines = [
        "\nAnswer:",
        f"  {answer}",
        "",
        "Citations:",
    ]
    if citations:
        for c in citations:
            # section_path already starts with the document title — show it alone.
            src = c.get("section_path") or c.get("document_title") or "?"
            lines.append(f"  [{c.get('marker')}] {src}  (chunk {c.get('chunk_id')})")
    else:
        lines.append("  (none)")
    lines += [
        "",
        f"  wall-clock: {elapsed:.1f}s",
        f"  thread_id:  {thread_id}",
    ]
    if low_conf:
        lines.append("  note:       answered best-effort (clarification cap reached — low confidence)")
    print("\n".join(lines) + "\n")
    return EXIT_OK
