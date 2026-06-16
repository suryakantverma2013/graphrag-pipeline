#!/usr/bin/env python
"""Resume a query run suspended at clarification or escalation (FR-Q1.5/3.6, D14).

    python resume_query.py <thread_id>

Same static-`interrupt_before` + `thread_id` resume contract as ingest tag review
(review_tags.py): read the snapshot (`get_state`), write the human input back
(`update_state`), then continue from the interrupt with `invoke(None)`. The
pending node (`snapshot.next`) tells us WHICH interrupt:

* request_clarification (FR-Q1.5/1.6): collect the user's clarification, write
  `human_clarification` + increment `clarification_rounds`; the run loops back to
  detect_ambiguity for re-evaluation. (It is NOT the dynamic interrupt()+Command
  pattern — `update_state` on a node paused via `interrupt_before` is attributed
  to that pending node, so its body is bypassed.)
* escalate (FR-Q3.6): surface the low-confidence reranked candidates for expert
  review; on acknowledgement write `escalated=True` and the run proceeds to
  synthesis.
"""

from __future__ import annotations

import logging
import sys
import time

from rag import PIPELINE_QUERY
from rag.cli import init_runtime, thread_config
from rag.clients.postgres import open_checkpointer
from rag.logging_config import set_context
from rag.query.graph import build_query_graph
from rag.query.nodes import ESCALATE, REQUEST_CLARIFICATION
from rag.query.report import EXIT_ERROR, report_query_result

logger = logging.getLogger(__name__)


def _safe_input(prompt: str) -> str:
    """input() that treats a closed stdin (EOF) as empty (pipe-testable)."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _resume_clarification(values: dict) -> dict:
    """Show the clarification question and collect the user's answer (FR-Q1.6)."""
    question = values.get("clarification_question")
    rounds = int(values.get("clarification_rounds") or 0)
    print(f"\nClarification needed (round {rounds + 1}):")
    print(f"  original query:  {values.get('original_query')!r}")
    if values.get("refined_query"):
        print(f"  refined so far:  {values.get('refined_query')!r}")
    if values.get("refine_confidence") is not None:
        print(f"  confidence:      {values.get('refine_confidence'):.2f}")
    print(f"\n  {question or 'Please add detail to disambiguate your question.'}\n")
    answer = _safe_input("  Your clarification: ")
    # Increment the round counter here (plain int field, no reducer → replaces);
    # the loop cap (D12) is enforced by route_after_refine on the next pass.
    return {"human_clarification": answer, "clarification_rounds": rounds + 1}


def _resume_escalation(values: dict) -> dict:
    """Surface the low-confidence candidates for expert review (FR-Q3.6)."""
    confidence = values.get("confidence_signal")
    reranked = values.get("reranked") or []
    print("\nLow-confidence retrieval escalated for expert review.")
    print(f"  original query:  {values.get('original_query')!r}")
    if confidence is not None:
        print(f"  composite confidence: {confidence:.3f} (threshold below escalate trigger)")
    print(f"\n  Top reranked candidates ({len(reranked)}):")
    for hit in reranked[:5]:
        src = hit.get("section_path") or hit.get("document_title") or "?"
        preview = (hit.get("text") or "").replace("\n", " ")[:140]
        print(f"   - [{hit.get('rerank_score'):.3f}] {src}: {preview}")
    _safe_input("\n  Press Enter to proceed to synthesis (best-effort answer): ")
    return {"escalated": True}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python resume_query.py <thread_id>", file=sys.stderr)
        return 2
    thread_id = argv[0]

    config = init_runtime(PIPELINE_QUERY)
    set_context(thread_id=thread_id)
    logger.info("resume query interrupt: thread_id=%s", thread_id)

    with open_checkpointer(config) as checkpointer:
        graph = build_query_graph(config, checkpointer)
        cfg = thread_config(thread_id)
        snapshot = graph.get_state(cfg)

        if not snapshot.created_at:
            print(f"No run found for thread_id {thread_id}.", file=sys.stderr)
            return EXIT_ERROR
        pending = snapshot.next or ()
        values = snapshot.values

        if REQUEST_CLARIFICATION in pending:
            update = _resume_clarification(values)
        elif ESCALATE in pending:
            update = _resume_escalation(values)
        else:
            print(
                f"Thread {thread_id} is not awaiting clarification or escalation "
                f"(next={pending or '<none>'}). Nothing to resume.",
                file=sys.stderr,
            )
            return EXIT_ERROR

        graph.update_state(cfg, update)
        logger.info("resume: input applied; continuing run")
        started = time.perf_counter()
        result = graph.invoke(None, config=cfg)
        elapsed = time.perf_counter() - started
        return report_query_result(graph, cfg, result, thread_id, elapsed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
