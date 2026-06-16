#!/usr/bin/env python
"""Resume an ingestion run suspended at human tag review (FR-5.3/5.5, D14).

    python review_tags.py <thread_id>

Loads the suspended state from the Postgres checkpointer, shows the current
iiRDS tags, lets the operator accept or correct each field, sets
`human_reviewed=True`, and resumes the pipeline to embed → write → receipt.

The graph is compiled with a static `interrupt_before=[human_review]` (FR-5.1),
so resuming is: read the snapshot (`get_state`), write any corrections back
(`update_state`), then continue from the interrupt with `invoke(None)`. (This is
the static-interrupt resume; it is NOT the dynamic `interrupt()`+`Command(resume)`
pattern.) Same `thread_id` resume contract as query clarification/escalation (D14).
"""

from __future__ import annotations

import logging
import sys
import time

from rag import PIPELINE_INGEST
from rag.cli import init_runtime, thread_config
from rag.clients.postgres import open_checkpointer
from rag.iirds import (
    InformationType,
    LifecyclePhase,
    normalize_information_type,
    normalize_lifecycle_phase,
)
from rag.ingestion.graph import build_ingestion_graph
from rag.ingestion.nodes import HUMAN_REVIEW
from rag.ingestion.report import EXIT_ERROR, report_result
from rag.logging_config import set_context

logger = logging.getLogger(__name__)


def _safe_input(prompt: str) -> str:
    """input() that treats a closed stdin (EOF) as 'keep current' (empty)."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _edit_text(label: str, current):
    """Prompt for a free-text field. Enter keeps current; '-' clears to None."""
    raw = _safe_input(f"  {label} [{current if current else '<none>'}] (Enter=keep, '-'=clear): ")
    if raw == "":
        return current
    return None if raw == "-" else raw


def _edit_list(label: str, current: list):
    """Prompt for a comma-separated list. Enter keeps; '-' clears to []."""
    shown = ", ".join(current) if current else "<none>"
    raw = _safe_input(f"  {label} [{shown}] (comma-separated, Enter=keep, '-'=clear): ")
    if raw == "":
        return current
    if raw == "-":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _edit_enum(label: str, current, enum, normalizer):
    """Prompt for a closed-enum field; invalid input warns and keeps current."""
    allowed = [m.value for m in enum]
    raw = _safe_input(f"  {label} [{current if current else '<none>'}] {allowed} (Enter=keep, '-'=clear): ")
    if raw == "":
        return current
    if raw == "-":
        return None
    normalized = normalizer(raw)
    if normalized is None:
        print(f"    '{raw}' is not a valid {label}; keeping {current!r}", file=sys.stderr)
        return current
    return normalized


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python review_tags.py <thread_id>", file=sys.stderr)
        return 2
    thread_id = argv[0]

    config = init_runtime(PIPELINE_INGEST)
    set_context(thread_id=thread_id)
    logger.info("resume tag review: thread_id=%s", thread_id)

    with open_checkpointer(config) as checkpointer:
        graph = build_ingestion_graph(config, checkpointer)
        cfg = thread_config(thread_id)
        snapshot = graph.get_state(cfg)

        # Validate this thread is actually suspended at human review (FR-5.4).
        if not snapshot.created_at:
            print(f"No run found for thread_id {thread_id}.", file=sys.stderr)
            return EXIT_ERROR
        if HUMAN_REVIEW not in (snapshot.next or ()):
            print(
                f"Thread {thread_id} is not awaiting tag review "
                f"(next={snapshot.next or '<none>'}). Nothing to resume.",
                file=sys.stderr,
            )
            return EXIT_ERROR

        values = snapshot.values
        tags = dict(values.get("iirds_tags") or {})
        confidence = values.get("confidence")

        # FR-5.3 show current tags and collect corrections.
        print(f"\nTag review for: {values.get('title')!r}")
        print(f"Model confidence: {confidence:.2f} (threshold {config.tag_confidence_threshold})\n")
        print("Review each field — press Enter to keep the current value:\n")
        corrected = {
            "product": _edit_text("product", tags.get("product")),
            "components": _edit_list("components", tags.get("components") or []),
            "lifecycle_phase": _edit_enum(
                "lifecycle_phase", tags.get("lifecycle_phase"), LifecyclePhase, normalize_lifecycle_phase
            ),
            "information_type": _edit_enum(
                "information_type", tags.get("information_type"), InformationType, normalize_information_type
            ),
            "language": _edit_text("language", tags.get("language")),
        }

        # FR-5.5 write corrections back + mark reviewed, then resume from the
        # interrupt with invoke(None). NOTE: update_state on a graph paused via
        # interrupt_before=[human_review] attributes the write *as* that pending
        # node, so its body is treated as done and the run resumes straight into
        # embed — we therefore set human_reviewed here rather than relying on the
        # node body to set it.
        graph.update_state(cfg, {"iirds_tags": corrected, "human_reviewed": True})
        logger.info("tags corrected; resuming run")
        started = time.perf_counter()
        result = graph.invoke(None, config=cfg)
        elapsed = time.perf_counter() - started
        return report_result(graph, cfg, result, thread_id, elapsed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
