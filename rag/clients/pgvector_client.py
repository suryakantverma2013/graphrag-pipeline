"""Postgres + pgvector store for the SIMPLE baseline (benchmark only).

This is the deliberately-naive counterpart to the Neo4j hybrid retriever: plain
dense vectors in Postgres, cosine nearest-neighbour, and nothing else — no BM25,
no graph traversal, no cross-encoder rerank. It exists ONLY to measure how much
the production hybrid pipeline adds over embedding-only search (the question that
started this: "how much does hybrid improve over simple full-text / embedding
search?"). It is NOT part of the production read/write path.

It reuses the SAME Postgres instance as the LangGraph checkpointer
(`CHECKPOINT_DB_URI`, localhost:5432/langgraph) but a dedicated table, so the
baseline never touches the Neo4j graph and the two stores stay independent.

The vectors share the embedding invariants of the rest of the system
(`EMBEDDING_DIMENSIONS` = 1536, cosine) so a query embedded once can be compared
fairly against both stores (NFR-REL-8 / FR-Q0.5).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import AppConfig

logger = logging.getLogger(__name__)

# Dedicated table for the baseline embeddings (documented constant, NFR-MAINT-2).
# Lives in the checkpointer DB but is logically separate from the checkpoint
# tables and from anything in Neo4j.
BASELINE_TABLE = "baseline_chunks"
# HNSW is pgvector's graph ANN index; cosine ops match the system metric. Build
# params are pgvector defaults — fine for a benchmark corpus of this size.
_HNSW_INDEX = "baseline_chunks_embedding_hnsw"


def _vector_literal(vec: Sequence[float]) -> str:
    """Render a float vector as the pgvector text literal `[a,b,c]`.

    Used instead of the optional `pgvector` Python adapter so the baseline adds
    NO new dependency: the literal is sent as a normal string and cast `::vector`
    in SQL. 7 significant digits is well within float32 storage precision.
    """
    return "[" + ",".join(format(float(x), ".7g") for x in vec) + "]"


class PgVectorStore:
    """Thin psycopg-3 wrapper over a single `vector(dim)` table (baseline only)."""

    def __init__(self, config: AppConfig) -> None:
        self._uri = config.checkpoint_db_uri
        self._dim = config.embedding_dimensions

    # --- connection ---------------------------------------------------------
    def _connect(self):
        """Open an autocommit psycopg-3 connection (DDL + simple writes)."""
        import psycopg

        return psycopg.connect(self._uri, autocommit=True)

    # --- schema -------------------------------------------------------------
    def ensure_schema(self, *, reset: bool = False) -> None:
        """Create the pgvector extension, table, and HNSW index (idempotent).

        `reset=True` drops the table first so a benchmark run starts from a
        clean corpus. Raises a clear, actionable error if the server lacks the
        pgvector extension (so the failure isn't a cryptic SQL error).
        """
        with self._connect() as conn, conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as exc:  # noqa: BLE001 — make the cause obvious
                raise RuntimeError(
                    "pgvector extension is not available on this PostgreSQL server. "
                    "Install it (e.g. the pgvector Windows build for PG18, or "
                    "`CREATE EXTENSION vector` as a superuser once the files are present) "
                    f"then re-run. Underlying error: {exc}"
                ) from exc
            if reset:
                cur.execute(f"DROP TABLE IF EXISTS {BASELINE_TABLE}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {BASELINE_TABLE} (
                    chunk_id       text PRIMARY KEY,
                    doc_hash       text NOT NULL,
                    document_title text,
                    section_path   text,
                    position       integer,
                    text           text NOT NULL,
                    embedding      vector({self._dim}) NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {_HNSW_INDEX} "
                f"ON {BASELINE_TABLE} USING hnsw (embedding vector_cosine_ops)"
            )
        logger.info("baseline pgvector store ready (table=%s, dim=%d)", BASELINE_TABLE, self._dim)

    # --- writes -------------------------------------------------------------
    def delete_document(self, doc_hash: str) -> int:
        """Remove a document's chunks so re-ingesting the same PDF is idempotent."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {BASELINE_TABLE} WHERE doc_hash = %s", (doc_hash,))
            return cur.rowcount

    def upsert_chunks(self, rows: Sequence[dict[str, Any]], *, batch: int = 500) -> int:
        """Insert chunk rows (each with an `embedding` list). Replaces on id clash.

        Rows must carry: chunk_id, doc_hash, document_title, section_path,
        position, text, embedding (list[float] of length dim).
        """
        sql = (
            f"INSERT INTO {BASELINE_TABLE} "
            "(chunk_id, doc_hash, document_title, section_path, position, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "doc_hash=EXCLUDED.doc_hash, document_title=EXCLUDED.document_title, "
            "section_path=EXCLUDED.section_path, position=EXCLUDED.position, "
            "text=EXCLUDED.text, embedding=EXCLUDED.embedding"
        )
        written = 0
        with self._connect() as conn, conn.cursor() as cur:
            params: list[tuple] = []
            for r in rows:
                vec = r["embedding"]
                if len(vec) != self._dim:  # NFR-REL-8 fail fast on a dim mismatch
                    raise ValueError(f"embedding dim {len(vec)} != {self._dim}")
                params.append(
                    (
                        r["chunk_id"], r["doc_hash"], r.get("document_title"),
                        r.get("section_path"), r.get("position"), r["text"],
                        _vector_literal(vec),
                    )
                )
                if len(params) >= batch:
                    cur.executemany(sql, params)
                    written += len(params)
                    params.clear()
            if params:
                cur.executemany(sql, params)
                written += len(params)
        return written

    # --- reads --------------------------------------------------------------
    def search(self, query_vec: Sequence[float], k: int) -> list[dict[str, Any]]:
        """Cosine top-k nearest chunks, best-first. Score = cosine similarity.

        Pure dense retrieval — this is the whole 'simple' retriever. `<=>` is
        pgvector's cosine DISTANCE; similarity = 1 - distance, so higher is
        better and the ORDER BY (ascending distance) returns the closest first.
        """
        vec = _vector_literal(query_vec)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, doc_hash, document_title, section_path, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM {BASELINE_TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, vec, k),
            )
            cols = [d.name for d in cur.description]
            hits = [dict(zip(cols, row)) for row in cur.fetchall()]
        for rank, hit in enumerate(hits, start=1):
            hit["rank"] = rank
            hit["score"] = float(hit["score"])
        return hits

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {BASELINE_TABLE}")
            return int(cur.fetchone()[0])

    def documents(self) -> list[dict[str, Any]]:
        """(doc_hash, title, chunk_count) per ingested document — for run reports."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT doc_hash, max(document_title) AS title, count(*) AS chunks "
                f"FROM {BASELINE_TABLE} GROUP BY doc_hash ORDER BY chunks DESC"
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
