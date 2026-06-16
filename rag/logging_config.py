"""Shared structured logging (NFR-LOG).

Single shared configuration used by every component (NFR-LOG-1); `print()` is
not used for diagnostics. Emits structured JSON by default (NFR-LOG-2) to both
the console and a size-rotating file under `LOG_DIR` (NFR-LOG-3). Every record
carries `thread_id`, `pipeline` and `stage` for correlation (NFR-LOG-7), set via
context variables so node code does not thread them through call signatures.

Secrets and full payloads (document/chunk text, embedding vectors) must never be
passed to the logger (NFR-LOG-6 / NFR-SEC-7) — log lengths/counts/ids instead.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import AppConfig

# --- Correlation context (NFR-LOG-7) -----------------------------------------
_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("thread_id", default=None)
_pipeline: contextvars.ContextVar[str | None] = contextvars.ContextVar("pipeline", default=None)
_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar("stage", default=None)

# Third-party loggers default to WARNING regardless of app level (NFR-LOG-5).
_NOISY_LOGGERS = (
    "httpx", "openai", "neo4j", "psycopg", "transformers", "docling", "urllib3",
)

# Rotating file policy: 10 MB x 5 backups (NFR-LOG-3).
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


class _ContextFilter(logging.Filter):
    """Inject the correlation context onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.thread_id = _thread_id.get()
        record.pipeline = _pipeline.get()
        record.stage = _stage.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object (NFR-LOG-2)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread_id": getattr(record, "thread_id", None),
            "pipeline": getattr(record, "pipeline", None),
            "stage": getattr(record, "stage", None),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_TEXT_FMT = "%(asctime)s %(levelname)-7s [%(pipeline)s/%(stage)s %(thread_id)s] %(name)s: %(message)s"


def setup_logging(config: AppConfig) -> None:
    """Configure root logging from config. Idempotent (safe to call once/proc)."""
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        return

    formatter: logging.Formatter = (
        _JsonFormatter() if config.log_format.lower() == "json" else logging.Formatter(_TEXT_FMT)
    )
    context_filter = _ContextFilter()

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "rag.log", maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root.addHandler(handler)

    root.setLevel(config.log_level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    root._rag_configured = True  # type: ignore[attr-defined]


def set_context(*, thread_id: str | None = None, pipeline: str | None = None) -> None:
    """Set the run-level correlation context (thread_id, pipeline)."""
    if thread_id is not None:
        _thread_id.set(thread_id)
    if pipeline is not None:
        _pipeline.set(pipeline)


@contextmanager
def stage_context(stage: str) -> Iterator[None]:
    """Scope the `stage` field for the duration of a node (NFR-LOG-7)."""
    token = _stage.set(stage)
    try:
        yield
    finally:
        _stage.reset(token)
