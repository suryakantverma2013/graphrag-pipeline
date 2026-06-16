"""One-time environment warm-up + verification (D7, §2.4, FR-S0.1-S0.7).

Required path before any ingest/query run. Runs independently of the pipeline
CLIs (FR-S0.1). Asserts CUDA, downloads/caches local weights, smoke-tests them,
ensures Neo4j constraints/indexes and the Postgres checkpointer tables, then
prints a clear ready / NOT-ready summary and exits non-zero on any failure
(FR-S0.6). Lazy first-run download remains only as a fallback.
"""

from __future__ import annotations

import logging
import sys

from .clients.neo4j_client import Neo4jClient
from .clients.postgres import setup_checkpointer
from .clients.reranker import get_reranker
from .config import AppConfig, get_config
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


def _assert_cuda() -> str:
    """FR-S0.2: require a CUDA torch build; fail with an actionable message (RISK-D)."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the CUDA build from the wheel index in "
            "REQUIREMENTS.md §2.3: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False — a CPU-only torch build is installed. "
            "Reinstall from the CUDA wheel index (REQUIREMENTS.md §2.3)."
        )
    return torch.cuda.get_device_name(0)


def _warm_models(config: AppConfig) -> None:
    """FR-S0.3/S0.4: download + cache local weights and smoke-test on the device."""
    from .clients import docling_client

    # Reranker (BGE) — load once + smoke inference (caches weights, confirms device).
    reranker = get_reranker(config)
    reranker.load()
    score = reranker.rerank("warm-up query", ["a relevant passage"])
    logger.info("reranker smoke OK (score=%.3f)", score[0])

    # Docling (DocLayNet/TableFormer) + EasyOCR — download + smoke convert (FR-S0.3/4).
    docling_client.warm_up(config)


def run_bootstrap(config: AppConfig | None = None) -> int:
    """Execute the full bootstrap; return a process exit code (FR-S0.6)."""
    config = config or get_config()
    setup_logging(config)

    failures: list[str] = []

    try:
        device_name = _assert_cuda()
        logger.info("CUDA OK: %s", device_name)
    except RuntimeError as exc:
        failures.append(f"GPU/CUDA: {exc}")

    try:
        with Neo4jClient(config) as neo4j:
            neo4j.verify_connectivity()  # FR-S0.5
            neo4j.create_indexes()
        logger.info("Neo4j connectivity + indexes OK")
    except Exception as exc:  # noqa: BLE001 - bootstrap reports every failure
        failures.append(f"Neo4j: {exc}")

    try:
        setup_checkpointer(config)  # FR-S0.5a
        logger.info("Postgres checkpointer OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Postgres: {exc}")

    try:
        _warm_models(config)
        logger.info("Local model weights cached + smoke-tested")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Models: {exc}")

    if failures:
        logger.error("environment NOT ready")
        for f in failures:
            logger.error("  - %s", f)
        return 1

    logger.info("environment ready")
    return 0


def main() -> None:
    sys.exit(run_bootstrap())


if __name__ == "__main__":
    main()
