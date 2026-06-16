"""LangGraph Postgres checkpointer factory (D15).

Postgres runs as a NATIVE on-prem service on the laptop (localhost:5432), not
Docker (C6 / v2.4). The checkpointer persists full run state per `thread_id` and
backs the human-in-the-loop interrupts in both pipelines (NFR-REL-5). Bootstrap
creates the checkpointer tables via `.setup()` (FR-S0.5a).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from ..config import AppConfig

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)


@contextmanager
def open_checkpointer(config: AppConfig) -> Iterator["PostgresSaver"]:
    """Yield a connected PostgresSaver bound to the configured DB (D15).

    `PostgresSaver.from_conn_string` sets autocommit + dict row factory for us.
    Caller compiles the graph with this saver inside the `with` block so the
    connection stays open for the duration of the run.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(config.checkpoint_db_uri) as saver:
        yield saver


def setup_checkpointer(config: AppConfig) -> None:
    """Idempotently create the checkpointer tables (FR-S0.5a)."""
    with open_checkpointer(config) as saver:
        saver.setup()
    logger.info("Postgres checkpointer tables ensured")
