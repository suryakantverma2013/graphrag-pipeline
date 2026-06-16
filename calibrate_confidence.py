#!/usr/bin/env python
"""Confidence-threshold calibration harness for the query read-path.

    python calibrate_confidence.py [queries_file]   # default: calibration_queries.txt

WHY THIS EXISTS
---------------
`ESCALATE_CONFIDENCE_THRESHOLD` (config.py, default 0.4) and `CONFIDENCE_WEIGHTS`
(rag/query/nodes.py) were fixed as documented constants (FR-Q3.4 / NFR-MAINT-2)
WITHOUT a real corpus to calibrate against — the BGE reranker is strict, so the
right threshold can only be read off the actual composite-confidence distribution
of a representative query set over the real knowledge graph. Run this AFTER you
have ingested the real corpus.

WHAT IT DOES
------------
For each query it runs the genuine query nodes — detect_ambiguity → refine →
detect_complexity → (decompose) → prepare_retrieval → classify → the three
retrievers → merge → cross_encoder_rerank → assess_confidence — and records the
composite confidence and its three component terms (top score, avg of top-3,
rank1-rank2 gap). It STOPS before `synthesize_answer`, so there is NO synthesis
LLM call and no Postgres checkpointer is needed. The clarification interrupt is
bypassed (we want every query to reach retrieval). It still issues the small
reasoning + embedding OpenAI calls and runs the local GPU reranker, exactly as
production does, so the numbers are faithful.

If queries are TAGGED with an expected bucket it also scans candidate thresholds
and reports the one that best separates "answerable" from "should-escalate".

QUERIES FILE FORMAT
-------------------
One query per line; blank lines and lines starting with '#' are ignored. An
OPTIONAL expected-bucket label may precede the query, separated by a TAB or '|':

    answerable | What is the rated flow of the X200 pump?
    escalate   | What colour should I paint the warehouse?
    What is the impeller diameter?            # untagged is fine too

Accepted labels (case-insensitive): answerable/answer/yes/y  vs  escalate/no/n/none.
See calibration_queries.example.txt for a starter set.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

from rag import PIPELINE_QUERY
from rag.cli import init_runtime, new_thread_id
from rag.logging_config import set_context
from rag.query import nodes as qn
from rag.query.state import QueryState

DEFAULT_QUERIES_FILE = "calibration_queries.txt"
RESULTS_CSV = "calibration_results.csv"

_ANSWERABLE = {"answerable", "answer", "yes", "y", "ans", "a"}
_ESCALATE = {"escalate", "no", "n", "none", "esc", "e"}


# --- query-file parsing ------------------------------------------------------
def _parse_label(token: str) -> str:
    t = token.strip().casefold()
    if t in _ANSWERABLE:
        return "answerable"
    if t in _ESCALATE:
        return "escalate"
    return "?"


def load_queries(path: Path) -> list[tuple[str, str]]:
    """Return (label, query) pairs. label is 'answerable' | 'escalate' | '?'."""
    rows: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sep = "\t" if "\t" in line else ("|" if "|" in line else None)
        if sep:
            head, _, tail = line.partition(sep)
            label, query = _parse_label(head), tail.strip()
            if not query:  # the separator was inside the query, not a label
                label, query = "?", line
        else:
            label, query = "?", line
        rows.append((label, query))
    return rows


# --- run one query through retrieval + confidence (no synthesis) -------------
def _apply(state: QueryState, partial: dict) -> QueryState:
    """Merge a node's partial-dict result into the state (mirrors LangGraph's
    channel write; the timing reducer is irrelevant for calibration)."""
    return state.model_copy(update=partial)


def evaluate(query: str) -> dict:
    """Drive the real nodes up to assess_confidence; return the metrics dict."""
    state = QueryState(original_query=query, thread_id=new_thread_id())

    state = _apply(state, qn.detect_ambiguity(state))
    if qn.route_after_ambiguity(state) == qn.REFINE_QUERY:
        state = _apply(state, qn.refine_query(state))
    # NOTE: clarification interrupt deliberately skipped for calibration.

    state = _apply(state, qn.detect_complexity(state))
    if qn.route_after_complexity(state) == qn.DECOMPOSE_QUERY:
        state = _apply(state, qn.decompose_query(state))

    state = _apply(state, qn.prepare_retrieval(state))
    state = _apply(state, qn.classify_query(state))

    # Retrievers run in parallel in the graph; sequential here is equivalent
    # (distinct output keys, no cross-dependency).
    state = _apply(state, qn.bm25_retrieve(state))
    state = _apply(state, qn.vector_retrieve(state))
    state = _apply(state, qn.graph_traverse(state))

    state = _apply(state, qn.merge(state))
    state = _apply(state, qn.cross_encoder_rerank(state))
    state = _apply(state, qn.assess_confidence(state))

    # Recompute the three component terms from the reranked scores (assess_confidence
    # only returns the composite). Mirrors rag/query/nodes.py::assess_confidence.
    scores = sorted((h["rerank_score"] for h in state.reranked), reverse=True)
    top = scores[0] if scores else 0.0
    avg_top3 = (sum(scores[:3]) / min(3, len(scores))) if scores else 0.0
    gap = (scores[0] - scores[1]) if len(scores) > 1 else top

    return {
        "query_class": state.query_class.value if state.query_class else "?",
        "units": len(state.retrieval_units),
        "n_reranked": len(state.reranked),
        "top": round(top, 4),
        "avg_top3": round(avg_top3, 4),
        "gap": round(gap, 4),
        "composite": round(state.confidence_signal or 0.0, 4),
    }


# --- threshold recommendation ------------------------------------------------
def recommend_threshold(labeled: list[tuple[str, float]]) -> None:
    """Scan thresholds and report the one with best balanced accuracy on the
    answerable/escalate buckets (composite >= t ⇒ answer, < t ⇒ escalate)."""
    ans = [c for lab, c in labeled if lab == "answerable"]
    esc = [c for lab, c in labeled if lab == "escalate"]
    if not ans or not esc:
        print(
            "\n[threshold] need at least one 'answerable' AND one 'escalate' tagged "
            "query to recommend a threshold; got "
            f"{len(ans)} answerable / {len(esc)} escalate."
        )
        return

    print("\n[threshold] composite confidence by bucket:")
    print(f"  answerable (n={len(ans)}): min={min(ans):.3f} median={_median(ans):.3f} max={max(ans):.3f}")
    print(f"  escalate   (n={len(esc)}): min={min(esc):.3f} median={_median(esc):.3f} max={max(esc):.3f}")

    # Score every threshold on a 0.00..1.00 integer-stepped grid (avoids float
    # drift), then recommend the MIDPOINT of the best-scoring range so the chosen
    # value sits centered in the margin rather than on a bucket's edge.
    def balanced(t: float) -> float:
        sens = sum(1 for c in ans if c >= t) / len(ans)
        spec = sum(1 for c in esc if c < t) / len(esc)
        return (sens + spec) / 2

    grid = [i / 100 for i in range(101)]
    scored = [(t, balanced(t)) for t in grid]
    best_score = max(s for _, s in scored)
    best_ts = [t for t, s in scored if s == best_score]
    best_t = round((min(best_ts) + max(best_ts)) / 2, 2)  # center of the passing band

    tp = sum(1 for c in ans if c >= best_t)
    tn = sum(1 for c in esc if c < best_t)
    clean = max(esc) < min(ans)
    margin = f"{min(best_ts):.2f}-{max(best_ts):.2f}" if len(best_ts) > 1 else f"{best_ts[0]:.2f}"
    print(
        f"\n[threshold] RECOMMENDED ESCALATE_CONFIDENCE_THRESHOLD = {best_t:.2f}\n"
        f"  balanced accuracy {best_score:.0%}  "
        f"(answers {tp}/{len(ans)} answerable, escalates {tn}/{len(esc)} escalate)\n"
        f"  best over threshold band [{margin}] -> centered at {best_t:.2f}.\n"
        f"  buckets are {'cleanly separable' if clean else 'OVERLAPPING - formula re-weighting may help'}.\n"
        f"  set it in config.py / .env:  ESCALATE_CONFIDENCE_THRESHOLD={best_t:.2f}"
    )


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


# --- main --------------------------------------------------------------------
def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else Path(DEFAULT_QUERIES_FILE)
    if not path.is_file():
        print(
            f"queries file not found: {path}\n"
            f"create one (see calibration_queries.example.txt) or pass a path.",
            file=sys.stderr,
        )
        return 2

    config = init_runtime(PIPELINE_QUERY)
    set_context(pipeline=PIPELINE_QUERY)
    # Quiet the per-node INFO chatter so the table is readable; warnings still show.
    logging.getLogger("rag").setLevel(logging.WARNING)

    queries = load_queries(path)
    if not queries:
        print(f"no queries found in {path}", file=sys.stderr)
        return 2

    threshold = config.escalate_confidence_threshold
    print(f"\nCalibrating {len(queries)} query(ies) - current threshold = {threshold} (escalate if composite <)\n")
    header = f"{'label':<11} {'class':<12} {'units':>5} {'top':>7} {'avg3':>7} {'gap':>7} {'composite':>9} {'esc?':>5}  query"
    print(header)
    print("-" * len(header))

    labeled: list[tuple[str, float]] = []
    csv_rows: list[dict] = []
    for label, query in queries:
        try:
            m = evaluate(query)
        except Exception as exc:  # noqa: BLE001 — keep going on a bad query
            print(f"{label:<11} {'ERROR':<12} {'':>5} {'':>7} {'':>7} {'':>7} {'':>9} {'':>5}  {query}  ({exc})")
            continue
        would_escalate = m["composite"] < threshold
        print(
            f"{label:<11} {m['query_class']:<12} {m['units']:>5} {m['top']:>7.3f} "
            f"{m['avg_top3']:>7.3f} {m['gap']:>7.3f} {m['composite']:>9.3f} "
            f"{('YES' if would_escalate else 'no'):>5}  {query[:60]}"
        )
        if label in ("answerable", "escalate"):
            labeled.append((label, m["composite"]))
        csv_rows.append({"label": label, "query": query, "would_escalate_now": would_escalate, **m})

    # Persist the full table for offline analysis / re-thresholding.
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["label", "query", "query_class", "units", "n_reranked",
                        "top", "avg_top3", "gap", "composite", "would_escalate_now"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {len(csv_rows)} row(s) to {RESULTS_CSV}")

    recommend_threshold(labeled)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
