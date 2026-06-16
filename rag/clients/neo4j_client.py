"""Neo4j driver wrapper + idempotent index/constraint management.

Used by ingestion (atomic write, FR-3.7) and query (BM25/vector/graph retrieval,
§4.2). The required constraints/indexes are created canonically by the bootstrap
(FR-S0.5) and re-checked at write time as a safety net (FR-7.10). Bound to
localhost only (NFR-SEC-5); credentials come from config, never hard-coded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import EMBEDDING_DIMENSIONS, EMBEDDING_SIMILARITY, AppConfig

if TYPE_CHECKING:  # avoid importing the heavy driver at module import time
    from neo4j import Driver

logger = logging.getLogger(__name__)

# Index/constraint names — single source so bootstrap and retrieval agree.
DOCUMENT_ID_CONSTRAINT = "document_id_unique"
VECTOR_INDEX = "chunk_embedding"
FULLTEXT_INDEX = "chunk_fulltext"

# Idempotent DDL (FR-7.10 / FR-S0.5). Vector dims/metric come from config
# invariants so ingest and query can never drift (NFR-REL-8).
_CONSTRAINT_DDL = (
    f"CREATE CONSTRAINT {DOCUMENT_ID_CONSTRAINT} IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.id IS UNIQUE"
)
_VECTOR_DDL = (
    f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS "
    "FOR (c:Chunk) ON c.embedding "
    "OPTIONS {indexConfig: {"
    f"`vector.dimensions`: {EMBEDDING_DIMENSIONS}, "
    f"`vector.similarity_function`: '{EMBEDDING_SIMILARITY}'"
    "}}"
)
_FULLTEXT_DDL = (
    f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
    "FOR (c:Chunk) ON EACH [c.text]"  # full-text on chunk text (FR-7.6)
)


class Neo4jClient:
    """Thin wrapper over the official Bolt driver."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._driver: "Driver | None" = None

    @property
    def driver(self) -> "Driver":
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self._config.neo4j_uri,
                auth=(self._config.neo4j_user, self._config.neo4j_password),
            )
        return self._driver

    def verify_connectivity(self) -> None:
        """Raise if Neo4j is unreachable (used by bootstrap, FR-S0.5)."""
        self.driver.verify_connectivity()

    def create_indexes(self) -> None:
        """Idempotently create the unique constraint + vector + full-text indexes.

        Canonical execution of FR-7.10 (bootstrap FR-S0.5). Safe to call repeatedly.
        """
        with self.driver.session(database=self._config.neo4j_database) as session:
            for ddl in (_CONSTRAINT_DDL, _VECTOR_DDL, _FULLTEXT_DDL):
                session.run(ddl)
        logger.info("Neo4j constraints/indexes ensured")

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
