"""Query StateGraph wiring (FR-Q0.1, §4).

All three retrievers fan out in parallel from classify_query and converge at
merge (D10 / FR-Q2.2) — query_class weights the merge, it never gates which
retrievers run. request_clarification and escalate are checkpointed interrupts
(D14) compiled with `interrupt_before`; both resume via the shared thread_id
resume-CLI. The clarification loop returns to detect_ambiguity (FR-Q1.6).

Topology (rendered diagram: REQUIREMENTS.md §4.0a)::

  START
    |
    v
  detect_ambiguity <--------------------------+   clarification loop
    |  ambiguous -> refine_query              |   (<= 2 rounds, D12)
    |                 |                        |
    |                 |  refine_conf < 0.6     |
    |                 |  AND rounds < max      |
    |                 v                        |
    |          request_clarification ---------+   [interrupt_before]
    |                 |  (resume: update_state clarification_rounds+1; body bypassed)
    |  clear          |  else / cap reached (best-effort, D12)
    v                 v
  detect_complexity
    |  complex -> decompose_query (<= 5, D12) --+
    |  simple ----------------------------------+
    v                                           |
  prepare_retrieval <---------------------------+
    |
    v
  classify_query
    |  fan-out: all three retrievers always run (D10)
    +--> bm25_retrieve ----+
    +--> vector_retrieve --+--> merge  (weighted RRF D16; dedup chunk_id; top-50)
    +--> graph_traverse ---+      |
                                  v
                          cross_encoder_rerank  (BGE on GPU; top-10)
                                  |
                                  v
                          assess_confidence
                            |  conf >= 0.4 -----------------+
                            |  conf < 0.4                   |
                            v                               |
                          escalate  [interrupt_before]      |
                            |  (resume: escalated=True; body bypassed)
                            v                               v
                          synthesize_answer  (RRF sub-queries D17; citations)
                            |
                            v
                           END
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from ..config import AppConfig
from . import nodes
from .state import QueryState

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)


def build_query_graph(config: AppConfig, checkpointer: "PostgresSaver"):
    """Construct and compile the query graph (FR-Q0.1)."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(QueryState)

    for name, fn in (
        (nodes.DETECT_AMBIGUITY, nodes.detect_ambiguity),
        (nodes.REFINE_QUERY, nodes.refine_query),
        (nodes.REQUEST_CLARIFICATION, nodes.request_clarification),
        (nodes.DETECT_COMPLEXITY, nodes.detect_complexity),
        (nodes.DECOMPOSE_QUERY, nodes.decompose_query),
        (nodes.PREPARE_RETRIEVAL, nodes.prepare_retrieval),
        (nodes.CLASSIFY_QUERY, nodes.classify_query),
        (nodes.BM25_RETRIEVE, nodes.bm25_retrieve),
        (nodes.VECTOR_RETRIEVE, nodes.vector_retrieve),
        (nodes.GRAPH_TRAVERSE, nodes.graph_traverse),
        (nodes.MERGE, nodes.merge),
        (nodes.CROSS_ENCODER_RERANK, nodes.cross_encoder_rerank),
        (nodes.ASSESS_CONFIDENCE, nodes.assess_confidence),
        (nodes.ESCALATE, nodes.escalate),
        (nodes.SYNTHESIZE_ANSWER, nodes.synthesize_answer),
    ):
        builder.add_node(name, fn)

    builder.add_edge(START, nodes.DETECT_AMBIGUITY)

    # Pre-retrieval routing (§4.1)
    builder.add_conditional_edges(
        nodes.DETECT_AMBIGUITY, nodes.route_after_ambiguity,
        {nodes.REFINE_QUERY: nodes.REFINE_QUERY, nodes.DETECT_COMPLEXITY: nodes.DETECT_COMPLEXITY},
    )
    builder.add_conditional_edges(
        nodes.REFINE_QUERY,
        partial(
            nodes.route_after_refine,
            threshold=config.refine_confidence_threshold,
            max_rounds=config.max_clarification_rounds,
        ),
        {nodes.REQUEST_CLARIFICATION: nodes.REQUEST_CLARIFICATION, nodes.DETECT_COMPLEXITY: nodes.DETECT_COMPLEXITY},
    )
    # Clarification loops back for re-evaluation (FR-Q1.6)
    builder.add_edge(nodes.REQUEST_CLARIFICATION, nodes.DETECT_AMBIGUITY)
    builder.add_conditional_edges(
        nodes.DETECT_COMPLEXITY, nodes.route_after_complexity,
        {nodes.DECOMPOSE_QUERY: nodes.DECOMPOSE_QUERY, nodes.PREPARE_RETRIEVAL: nodes.PREPARE_RETRIEVAL},
    )
    builder.add_edge(nodes.DECOMPOSE_QUERY, nodes.PREPARE_RETRIEVAL)
    builder.add_edge(nodes.PREPARE_RETRIEVAL, nodes.CLASSIFY_QUERY)

    # Retrieval fan-out -> merge (D10): all three always run in parallel
    for retriever in (nodes.BM25_RETRIEVE, nodes.VECTOR_RETRIEVE, nodes.GRAPH_TRAVERSE):
        builder.add_edge(nodes.CLASSIFY_QUERY, retriever)
        builder.add_edge(retriever, nodes.MERGE)

    builder.add_edge(nodes.MERGE, nodes.CROSS_ENCODER_RERANK)
    builder.add_edge(nodes.CROSS_ENCODER_RERANK, nodes.ASSESS_CONFIDENCE)

    # Confidence routing (§4.3)
    builder.add_conditional_edges(
        nodes.ASSESS_CONFIDENCE,
        partial(nodes.route_after_confidence, threshold=config.escalate_confidence_threshold),
        {nodes.ESCALATE: nodes.ESCALATE, nodes.SYNTHESIZE_ANSWER: nodes.SYNTHESIZE_ANSWER},
    )
    # After expert review the escalated result proceeds to synthesis (FR-Q3.6)
    builder.add_edge(nodes.ESCALATE, nodes.SYNTHESIZE_ANSWER)
    builder.add_edge(nodes.SYNTHESIZE_ANSWER, END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=[nodes.REQUEST_CLARIFICATION, nodes.ESCALATE],
    )
