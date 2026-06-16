"""Typed query state (FR-Q0.3).

Threaded through the query StateGraph. The raw query is stored as `original_query`
(FR-Q0.2). Clarification/escalation interrupts persist this whole object to the
Postgres checkpointer (FR-Q0.4).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class QueryClass(str, Enum):
    """Retrieval-unit classification (FR-Q2.1); used to WEIGHT merge, not exclude (D10)."""

    EXACT_LOOKUP = "exact_lookup"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    RELATIONAL = "relational"


def _merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Additive dict-merge reducer for `stage_timings_ms` (FR-Q0.8).

    The three retrievers write `stage_timings_ms` concurrently in a single
    LangGraph super-step; without a channel reducer those same-step writes raise
    `InvalidUpdateError`. Merging (right wins on key clash — there are none, the
    stage keys are distinct) lets every node's timing accumulate. The hit-list
    channels are distinct keys per retriever and need no reducer (D16).
    Verified with a langgraph 1.2.5 / pydantic 2.13.4 fan-out spike.
    """
    return {**left, **right}


class QueryState(BaseModel):
    """Full query run state (FR-Q0.3)."""

    # Identity / control
    original_query: str
    thread_id: str

    # Pre-retrieval (§4.1)
    refined_query: str | None = None
    is_ambiguous: bool | None = None
    refine_confidence: float | None = None
    clarification_question: str | None = None    # surfaced by the interrupt (FR-Q1.5)
    human_clarification: str | None = None
    clarification_rounds: int = 0                 # capped at MAX_CLARIFICATION_ROUNDS (D12)
    is_complex: bool | None = None
    sub_queries: list[str] = Field(default_factory=list)   # <= MAX_SUBQUERIES (D12)
    retrieval_units: list[str] = Field(default_factory=list)  # uniform list (FR-Q1.11)
    query_class: QueryClass | None = None

    # Retrieval (§4.2) — each capped at PER_RETRIEVER_K, merged to RETRIEVE_TOP_K
    bm25_hits: list[dict[str, Any]] = Field(default_factory=list)
    vector_hits: list[dict[str, Any]] = Field(default_factory=list)
    graph_hits: list[dict[str, Any]] = Field(default_factory=list)
    merged_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Re-rank + confidence (§4.3) — Top-RERANK_TOP_K
    reranked: list[dict[str, Any]] = Field(default_factory=list)
    confidence_signal: float | None = None
    escalated: bool = False
    low_confidence_flag: bool = False             # set when loop caps hit (D12)

    # Synthesis (§4.4)
    answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)  # FR-Q0.7

    # Observability — additive reducer: the parallel retrievers write this in one
    # super-step, so plain replacement would conflict (FR-Q0.8 / D16).
    stage_timings_ms: Annotated[dict[str, float], _merge_timings] = Field(default_factory=dict)
