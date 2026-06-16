"""Shared CLI helpers for the entry-point scripts.

Centralizes config + logging init, thread_id generation, and the resume pattern
used by all three human-in-the-loop interrupts (D14): tag review (ingest),
clarification and escalation (query) all resume by `thread_id`.
"""

from __future__ import annotations

import uuid

from .config import AppConfig, get_config
from .logging_config import set_context, setup_logging


def init_runtime(pipeline: str) -> AppConfig:
    """Load config, configure logging, set the pipeline log context."""
    config = get_config()
    setup_logging(config)
    set_context(pipeline=pipeline)
    return config


def new_thread_id() -> str:
    """Unique run id persisted via the checkpointer (FR-0.3 / FR-Q0.4)."""
    return uuid.uuid4().hex


def thread_config(thread_id: str) -> dict:
    """LangGraph invocation config selecting the checkpoint thread."""
    return {"configurable": {"thread_id": thread_id}}
