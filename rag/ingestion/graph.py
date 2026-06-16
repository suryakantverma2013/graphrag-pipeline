"""Ingestion StateGraph wiring (FR-0.1, §3).

Nodes: intake, parse, chunk, tag_iirds, human_review, embed, neo4j_write,
receipt + terminal error/duplicate nodes. Human review is a checkpointed
interrupt (FR-5.1) compiled with `interrupt_before` so the run suspends with
full state persisted to Postgres (D15) and resumes via review_tags.py (FR-5.4).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from ..config import AppConfig
from . import nodes
from .state import IngestionState, IngestStatus

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)

TERMINAL_DUPLICATE = "terminal_duplicate"
TERMINAL_ERROR = "terminal_error"


def _terminal_duplicate(state: IngestionState) -> dict:
    logger.info("terminal: DUPLICATE (re-ingestion not supported, FR-1.5)")
    return {}


def _terminal_error(state: IngestionState) -> dict:
    logger.error("terminal error: %s (%s)", state.pipeline_status.value, state.error)
    return {}


def build_ingestion_graph(config: AppConfig, checkpointer: "PostgresSaver"):
    """Construct and compile the ingestion graph (FR-0.1)."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(IngestionState)

    builder.add_node(nodes.INTAKE, nodes.intake)
    builder.add_node(nodes.PARSE, nodes.parse)
    builder.add_node(nodes.CHUNK, nodes.chunk)
    builder.add_node(nodes.TAG_IIRDS, nodes.tag_iirds)
    builder.add_node(nodes.HUMAN_REVIEW, nodes.human_review)
    builder.add_node(nodes.EMBED, nodes.embed)
    builder.add_node(nodes.NEO4J_WRITE, nodes.neo4j_write)
    builder.add_node(nodes.RECEIPT, nodes.receipt)
    builder.add_node(TERMINAL_DUPLICATE, _terminal_duplicate)
    builder.add_node(TERMINAL_ERROR, _terminal_error)

    builder.add_edge(START, nodes.INTAKE)
    builder.add_conditional_edges(
        nodes.INTAKE, nodes.route_after_intake,
        {nodes.PARSE: nodes.PARSE, "duplicate": TERMINAL_DUPLICATE, "error": TERMINAL_ERROR},
    )
    builder.add_conditional_edges(
        nodes.PARSE, nodes.route_after_parse,
        {nodes.CHUNK: nodes.CHUNK, "error": TERMINAL_ERROR},
    )
    builder.add_edge(nodes.CHUNK, nodes.TAG_IIRDS)
    builder.add_conditional_edges(
        nodes.TAG_IIRDS,
        partial(nodes.route_after_tag, threshold=config.tag_confidence_threshold),
        {nodes.HUMAN_REVIEW: nodes.HUMAN_REVIEW, nodes.EMBED: nodes.EMBED},
    )
    builder.add_edge(nodes.HUMAN_REVIEW, nodes.EMBED)
    builder.add_conditional_edges(
        nodes.EMBED, nodes.route_after_embed,
        {nodes.NEO4J_WRITE: nodes.NEO4J_WRITE, "error": TERMINAL_ERROR},
    )
    builder.add_conditional_edges(
        nodes.NEO4J_WRITE, nodes.route_after_write,
        {nodes.RECEIPT: nodes.RECEIPT, "error": TERMINAL_ERROR},
    )
    builder.add_edge(nodes.RECEIPT, END)
    builder.add_edge(TERMINAL_DUPLICATE, END)
    builder.add_edge(TERMINAL_ERROR, END)

    # Human review suspends here with state persisted (FR-5.1/5.2, D14).
    return builder.compile(checkpointer=checkpointer, interrupt_before=[nodes.HUMAN_REVIEW])
