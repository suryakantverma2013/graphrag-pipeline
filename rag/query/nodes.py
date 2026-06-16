"""Query node implementations (§4).

Node contract mirrors ingestion: each node takes the typed `QueryState` and
returns a partial dict LangGraph merges. The `_node` decorator scopes the
structured-log `stage` (NFR-LOG-7), logs entry, and records wall-clock ms into
`stage_timings_ms` (FR-Q0.3) — unlike ingestion it returns a BARE `{stage: ms}`
and lets the channel reducer (`state._merge_timings`) accumulate, because the
three retrievers write that channel concurrently in one super-step (FR-Q0.8).

Reasoning nodes (ambiguity/refine/complexity/decompose/classify) call
`LLM_MODEL`; `synthesize_answer` calls `SYNTHESIS_MODEL` (D8 / FR-Q0.6). The
retrievers run fully against Neo4j (no API); the reranker runs locally on GPU
(FR-Q3.2). Heavy clients are per-process singletons (NFR-PERF-4); config comes
from the cached `get_config()` since node signatures are fixed to `(state)`.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from ..clients.neo4j_client import FULLTEXT_INDEX, Neo4jClient, VECTOR_INDEX
from ..clients.openai_client import OpenAIClient
from ..clients.reranker import Reranker, get_reranker
from ..config import get_config
from ..logging_config import stage_context
from .fusion import RRF_K, weighted_rrf, weights_for_class
from .state import QueryClass, QueryState

logger = logging.getLogger(__name__)

# Node names (FR-Q0.1) — single source shared with graph.py.
DETECT_AMBIGUITY = "detect_ambiguity"
REFINE_QUERY = "refine_query"
REQUEST_CLARIFICATION = "request_clarification"
DETECT_COMPLEXITY = "detect_complexity"
DECOMPOSE_QUERY = "decompose_query"
PREPARE_RETRIEVAL = "prepare_retrieval"
CLASSIFY_QUERY = "classify_query"
BM25_RETRIEVE = "bm25_retrieve"
VECTOR_RETRIEVE = "vector_retrieve"
GRAPH_TRAVERSE = "graph_traverse"
MERGE = "merge"
CROSS_ENCODER_RERANK = "cross_encoder_rerank"
ASSESS_CONFIDENCE = "assess_confidence"
ESCALATE = "escalate"
SYNTHESIZE_ANSWER = "synthesize_answer"

# Retriever ids — also the keys into the D16 weight matrix.
R_BM25 = "bm25"
R_VECTOR = "vector"
R_GRAPH = "graph"

# Composite-confidence term weights (FR-Q3.4 — was TBD, fixed here). The three
# terms are each in [0,1] (rerank scores are sigmoid-normalized; the rank1-rank2
# gap is a difference of two [0,1] scores), so the weighted sum is in [0,1] and
# comparable to ESCALATE_CONFIDENCE_THRESHOLD (0.4). Top-score magnitude
# dominates (is the best chunk strong?), the avg-top-3 rewards corroboration, and
# the gap rewards a decisive winner. Documented constant (NFR-MAINT-2).
CONFIDENCE_WEIGHTS = {"top": 0.5, "avg_top3": 0.3, "gap": 0.2}

# Lucene reserved characters escaped before handing a unit to the full-text index
# (FR-Q2.3); an unescaped ':' or '(' makes db.index.fulltext.queryNodes throw.
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

# Graph-entity names shorter than this are ignored as match terms (FR-Q2.5) to
# avoid spurious substring hits (e.g. a 2-char component code matching noise).
_MIN_ENTITY_NAME = 3


# --- Per-process clients (NFR-PERF-4) ---------------------------------------
_neo4j_client: Neo4jClient | None = None
_openai_client: OpenAIClient | None = None


def _neo4j() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient(get_config())
    return _neo4j_client


def _openai() -> OpenAIClient:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient(get_config())
    return _openai_client


def _reranker() -> Reranker:
    return get_reranker(get_config())  # already a process-wide singleton


def _node(stage: str) -> Callable:
    """Scope the log `stage`, log entry, and record wall-clock ms (FR-Q0.3).

    Returns a BARE `{stage: ms}` and relies on the `stage_timings_ms` reducer to
    accumulate, since the parallel retrievers write that channel in one
    super-step (FR-Q0.8) — pre-merging here would race.
    """

    def decorator(fn: Callable) -> Callable:
        def wrapper(state: QueryState) -> dict:
            with stage_context(stage):
                logger.info("stage start: %s", stage)
                start = time.perf_counter()
                result = fn(state)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if isinstance(result, dict):
                    result["stage_timings_ms"] = {stage: round(elapsed_ms, 1)}
                logger.info("stage done: %s (%.1f ms)", stage, elapsed_ms)
                return result

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


# --- LLM helpers -------------------------------------------------------------
def _structured(system: str, user: str, schema: type) -> Any:
    """One schema-validated LLM_MODEL call (FR-Q0.6); retries live in the client."""
    return _openai().complete_structured(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema,
    )


def _effective_query(state: QueryState) -> str:
    """The query to reason over: original text plus any human clarification.

    Used by the pre-refine nodes so the clarification loop (FR-Q1.6) re-evaluates
    the augmented query rather than the bare original.
    """
    base = state.original_query
    if state.human_clarification:
        return f"{base}\n\nUser clarification: {state.human_clarification}"
    return base


def _retrieval_query(state: QueryState) -> str:
    """The best single query string once refinement has run."""
    return state.refined_query or _effective_query(state)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# === Pre-retrieval layer (§4.1) =============================================
class _AmbiguitySchema(BaseModel):
    is_ambiguous: bool
    reason: str = ""


@_node(DETECT_AMBIGUITY)
def detect_ambiguity(state: QueryState) -> dict:
    """FR-Q1.1/1.2 — flag missing specifics, pronouns, or multiple readings."""
    system = (
        "You triage technical-documentation search queries. Decide if the query is "
        "AMBIGUOUS: it lacks specificity (missing model/part numbers or scope), has "
        "unresolved pronouns, or admits multiple valid interpretations. A query that "
        "is concrete and answerable as written is NOT ambiguous."
    )
    result: _AmbiguitySchema = _structured(system, _effective_query(state), _AmbiguitySchema)
    logger.info("ambiguity: is_ambiguous=%s reason=%r", result.is_ambiguous, result.reason[:120])
    return {"is_ambiguous": result.is_ambiguous}


class _RefineSchema(BaseModel):
    refined_query: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    clarification_question: str = ""


@_node(REFINE_QUERY)
def refine_query(state: QueryState) -> dict:
    """FR-Q1.3 rewrite (expand abbreviations, resolve pronouns, add scope) +
    FR-Q1.4 a `refine_confidence`. Also proposes the clarification question the
    interrupt would ask if confidence is low (FR-Q1.5), so the resume CLI can
    surface it without re-calling the model."""
    system = (
        "You refine technical-documentation search queries. Rewrite the query to be "
        "specific and self-contained: expand abbreviations, resolve pronouns, and add "
        "scope qualifiers — WITHOUT inventing facts the user did not imply. Report your "
        "confidence (0-1) that the refined query faithfully and unambiguously captures "
        "the user's intent. If confidence is low, propose ONE concise clarification "
        "question to ask the user."
    )
    result: _RefineSchema = _structured(system, _effective_query(state), _RefineSchema)
    confidence = _clamp01(result.confidence)
    logger.info("refine: confidence=%.2f refined=%r", confidence, (result.refined_query or "")[:120])
    return {
        "refined_query": result.refined_query.strip() or _effective_query(state),
        "refine_confidence": confidence,
        "clarification_question": (result.clarification_question or "").strip() or None,
    }


@_node(REQUEST_CLARIFICATION)
def request_clarification(state: QueryState) -> dict:
    """Static-interrupt safety net (FR-Q1.5/1.6, D14).

    The graph is compiled with `interrupt_before=[request_clarification]`, so it
    suspends BEFORE this node and `resume_query.py` resumes by writing
    `human_clarification` + incrementing `clarification_rounds` via `update_state`
    (which attributes the write to this pending node, bypassing this body). The
    body only runs if a run is resumed without that write — it then loops back to
    detect_ambiguity unchanged, a harmless no-op."""
    logger.info("clarification interrupt safety-net body ran (no update_state applied)")
    return {}


class _ComplexitySchema(BaseModel):
    is_complex: bool
    reason: str = ""


@_node(DETECT_COMPLEXITY)
def detect_complexity(state: QueryState) -> dict:
    """FR-Q1.7/1.8 — multi-concept / multi-doc / multi-step ⇒ complex.

    Also the join point of the clarification cap (D12 / FR-Q1.6): if we arrived
    here having exhausted MAX_CLARIFICATION_ROUNDS on a still-low-confidence
    refinement, mark the run best-effort low-confidence."""
    config = get_config()
    system = (
        "You assess whether a technical-documentation query is COMPLEX: it spans "
        "multiple distinct concepts, multiple documents/products, or requires several "
        "reasoning steps to answer. A single focused question is NOT complex."
    )
    result: _ComplexitySchema = _structured(system, _retrieval_query(state), _ComplexitySchema)
    out: dict[str, Any] = {"is_complex": result.is_complex}

    if (
        state.refine_confidence is not None
        and state.refine_confidence < config.refine_confidence_threshold
        and state.clarification_rounds >= config.max_clarification_rounds
    ):
        out["low_confidence_flag"] = True
        logger.warning(
            "clarification cap reached (rounds=%d); proceeding best-effort (low confidence)",
            state.clarification_rounds,
        )
    logger.info("complexity: is_complex=%s reason=%r", result.is_complex, result.reason[:120])
    return out


class _DecomposeSchema(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


@_node(DECOMPOSE_QUERY)
def decompose_query(state: QueryState) -> dict:
    """FR-Q1.9/1.10 — split into ≤ MAX_SUBQUERIES atomic, single-level sub-queries
    (D12). The sub-queries are retrieved independently and aggregated by RRF at
    synthesis (D17), i.e. a parallel aggregation strategy."""
    config = get_config()
    system = (
        "Decompose a complex technical-documentation query into ATOMIC sub-queries, "
        "each addressing exactly one concept and independently answerable. Do NOT "
        f"nest or chain them (single level). Produce at most {config.max_subqueries} "
        "sub-queries; fewer is better when the query is only mildly complex."
    )
    result: _DecomposeSchema = _structured(system, _retrieval_query(state), _DecomposeSchema)
    subs = [s.strip() for s in result.sub_queries if s and s.strip()][: config.max_subqueries]
    if not subs:  # defensive: fall back to the whole query as one unit
        subs = [_retrieval_query(state)]
    logger.info("decompose: %d sub-queries", len(subs))
    return {"sub_queries": subs}


@_node(PREPARE_RETRIEVAL)
def prepare_retrieval(state: QueryState) -> dict:
    """FR-Q1.11 — normalize simple + decomposed paths into uniform retrieval_units
    so every downstream retriever handles both identically."""
    if state.is_complex and state.sub_queries:
        units = list(state.sub_queries)
    else:
        units = [_retrieval_query(state)]
    logger.info("prepare_retrieval: %d unit(s)", len(units))
    return {"retrieval_units": units}


# === Retrieval layer (§4.2) =================================================
class _ClassifySchema(BaseModel):
    query_class: str


def _normalize_query_class(value: str | None) -> QueryClass:
    """Map a model label to a QueryClass; default to CONCEPTUAL if out-of-vocab."""
    if value:
        target = value.strip().casefold()
        for member in QueryClass:
            if member.value == target:
                return member
    logger.warning("unrecognized query_class %r; defaulting to conceptual", value)
    return QueryClass.CONCEPTUAL


@_node(CLASSIFY_QUERY)
def classify_query(state: QueryState) -> dict:
    """FR-Q2.1 — one class for the whole query (D16); weights the merge, never
    gates which retrievers run (D10)."""
    system = (
        "Classify a technical-documentation query into exactly one retrieval class:\n"
        "- exact_lookup: a precise term, part/model number, code, or spec value.\n"
        "- conceptual: an explanatory 'what is / why' natural-language question.\n"
        "- procedural: a 'how do I / steps to' task question.\n"
        "- relational: about connections between products, components, phases, or types.\n"
        "Answer with the single class label."
    )
    result: _ClassifySchema = _structured(system, _retrieval_query(state), _ClassifySchema)
    cls = _normalize_query_class(result.query_class)
    logger.info("classify: query_class=%s", cls.value)
    return {"query_class": cls}


def _tag_hits(rows: list[dict], unit: str, retriever: str) -> list[dict[str, Any]]:
    """Shape Neo4j rows into uniform hit dicts tagged with unit + per-unit rank
    + originating retriever (D16). Rows arrive already score-ordered."""
    hits: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        score = row.get("score")
        hits.append(
            {
                "chunk_id": row["chunk_id"],
                "text": row.get("text") or "",
                "section_path": row.get("section_path"),
                "document_id": row.get("document_id"),
                "document_title": row.get("document_title"),
                "score": float(score) if score is not None else 0.0,
                "unit": unit,
                "rank": rank,
                "retriever": retriever,
            }
        )
    return hits


_BM25_CYPHER = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.id AS chunk_id, node.text AS text, node.section_path AS section_path,
       d.id AS document_id, d.title AS document_title, score
ORDER BY score DESC
LIMIT $k
"""


@_node(BM25_RETRIEVE)
def bm25_retrieve(state: QueryState) -> dict:
    """FR-Q2.3 — Neo4j full-text (Lucene/BM25) over chunk text, per unit, ≤ K.
    A failure degrades to zero hits rather than killing the query (NFR-REL-9)."""
    config = get_config()
    hits: list[dict[str, Any]] = []
    try:
        with _neo4j().driver.session(database=config.neo4j_database) as session:
            for unit in state.retrieval_units:
                q = _LUCENE_SPECIAL.sub(r"\\\1", unit).strip()
                if not q:
                    continue
                rows = session.run(
                    _BM25_CYPHER, index=FULLTEXT_INDEX, q=q, k=config.per_retriever_k
                ).data()
                hits.extend(_tag_hits(rows, unit, R_BM25))
    except Exception:  # noqa: BLE001 — NFR-REL-9 graceful degradation
        logger.warning("bm25_retrieve failed; contributing zero hits", exc_info=True)
        return {"bm25_hits": []}
    logger.info("bm25: %d hits over %d unit(s)", len(hits), len(state.retrieval_units))
    return {"bm25_hits": hits}


_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.id AS chunk_id, node.text AS text, node.section_path AS section_path,
       d.id AS document_id, d.title AS document_title, score
ORDER BY score DESC
"""


@_node(VECTOR_RETRIEVE)
def vector_retrieve(state: QueryState) -> dict:
    """FR-Q2.4 — embed each unit (same model/dim/metric as ingest, FR-Q0.5) and
    cosine-search the Neo4j vector index, per unit, ≤ K (NFR-REL-9 on failure)."""
    config = get_config()
    units = state.retrieval_units
    hits: list[dict[str, Any]] = []
    try:
        vectors = _openai().embed(units) if units else []
        with _neo4j().driver.session(database=config.neo4j_database) as session:
            for unit, vec in zip(units, vectors):
                rows = session.run(
                    _VECTOR_CYPHER, index=VECTOR_INDEX, k=config.per_retriever_k, vec=vec
                ).data()
                hits.extend(_tag_hits(rows, unit, R_VECTOR))
    except Exception:  # noqa: BLE001 — NFR-REL-9 graceful degradation
        logger.warning("vector_retrieve failed; contributing zero hits", exc_info=True)
        return {"vector_hits": []}
    logger.info("vector: %d hits over %d unit(s)", len(hits), len(units))
    return {"vector_hits": hits}


# Traverse iiRDS relationships (FR-Q2.5): match documents whose related entity
# names (Product/Component/LifecyclePhase/InformationType) appear in the unit
# text, then return their chunks ranked by how many distinct entities matched.
_GRAPH_CYPHER = """
MATCH (d:Document)-[:RELATES_TO_PRODUCT|RELATES_TO_COMPONENT|HAS_LIFECYCLE_PHASE|HAS_INFORMATION_TYPE]->(e)
WHERE size(e.name) >= $min_len AND $text CONTAINS toLower(e.name)
MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH c, d, count(DISTINCT e) AS matched
RETURN c.id AS chunk_id, c.text AS text, c.section_path AS section_path,
       d.id AS document_id, d.title AS document_title, matched AS score
ORDER BY score DESC, c.position ASC
LIMIT $k
"""


@_node(GRAPH_TRAVERSE)
def graph_traverse(state: QueryState) -> dict:
    """FR-Q2.5 — Cypher traversal over iiRDS relationships, per unit, ≤ K.

    Least-specified retriever (per plan): we surface chunks from documents whose
    mapped entities are named in the query. Zero relational signal simply yields
    zero hits and the weighted-RRF merge degrades for free (NFR-REL-9)."""
    config = get_config()
    hits: list[dict[str, Any]] = []
    try:
        with _neo4j().driver.session(database=config.neo4j_database) as session:
            for unit in state.retrieval_units:
                rows = session.run(
                    _GRAPH_CYPHER,
                    text=unit.casefold(),
                    min_len=_MIN_ENTITY_NAME,
                    k=config.per_retriever_k,
                ).data()
                hits.extend(_tag_hits(rows, unit, R_GRAPH))
    except Exception:  # noqa: BLE001 — NFR-REL-9 graceful degradation
        logger.warning("graph_traverse failed; contributing zero hits", exc_info=True)
        return {"graph_hits": []}
    logger.info("graph: %d hits over %d unit(s)", len(hits), len(state.retrieval_units))
    return {"graph_hits": hits}


def _group_by_unit(hits: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        grouped.setdefault(hit["unit"], []).append(hit)
    return grouped


@_node(MERGE)
def merge(state: QueryState) -> dict:
    """FR-Q2.2/2.6 + D16 — per unit, weighted-RRF the three retrievers (weights
    by query_class), dedup chunk_id within unit, cap RETRIEVE_TOP_K. Hits keep
    their `unit` tag and gain a within-unit merged rank (D16 convention; no new
    state field). A retriever that returned nothing contributes nothing."""
    config = get_config()
    cls = state.query_class.value if state.query_class else None
    weights = weights_for_class(cls)

    per_unit = {
        R_BM25: _group_by_unit(state.bm25_hits),
        R_VECTOR: _group_by_unit(state.vector_hits),
        R_GRAPH: _group_by_unit(state.graph_hits),
    }

    merged: list[dict[str, Any]] = []
    for unit in state.retrieval_units:
        weighted_lists = []
        for retriever in (R_BM25, R_VECTOR, R_GRAPH):
            unit_hits = sorted(per_unit[retriever].get(unit, []), key=lambda h: h["rank"])
            weighted_lists.append((weights.get(retriever, 0.0), unit_hits))
        fused = weighted_rrf(weighted_lists, k=RRF_K)[: config.retrieve_top_k]
        for rank, hit in enumerate(fused, start=1):
            hit["rank"] = rank  # within-unit merged rank (overwrites per-retriever rank)
        merged.extend(fused)

    logger.info(
        "merge: %d candidate(s) over %d unit(s) (query_class=%s)",
        len(merged), len(state.retrieval_units), cls,
    )
    return {"merged_candidates": merged}


# === Re-ranking layer (§4.3) ================================================
@_node(CROSS_ENCODER_RERANK)
def cross_encoder_rerank(state: QueryState) -> dict:
    """FR-Q3.1/3.3 + D17 — local BGE cross-encoder, looping units, scoring
    (unit_text, chunk) jointly → per-unit Top-RERANK_TOP_K, scores in [0,1].
    Runs on GPU by default; no API call (FR-Q3.2)."""
    config = get_config()
    reranker = _reranker()
    reranked: list[dict[str, Any]] = []

    for unit, candidates in _group_by_unit(state.merged_candidates).items():
        if not candidates:
            continue
        scores = reranker.rerank(unit, [c["text"] for c in candidates])
        scored = []
        for cand, score in zip(candidates, scores):
            hit = dict(cand)
            hit["rerank_score"] = round(float(score), 6)
            scored.append(hit)
        scored.sort(key=lambda h: h["rerank_score"], reverse=True)
        top = scored[: config.rerank_top_k]
        for rank, hit in enumerate(top, start=1):
            hit["rerank_rank"] = rank  # within-unit rerank rank (D17 input)
        reranked.extend(top)

    logger.info("rerank: %d reranked chunk(s) across units", len(reranked))
    return {"reranked": reranked}


@_node(ASSESS_CONFIDENCE)
def assess_confidence(state: QueryState) -> dict:
    """FR-Q3.4 + D17 — composite from top score, avg top-3, rank1-rank2 gap,
    POOLED across units (runs before the synthesis-time RRF, so it cannot use the
    final fused order). Composite stays in [0,1] (see CONFIDENCE_WEIGHTS)."""
    scores = sorted((h["rerank_score"] for h in state.reranked), reverse=True)
    if not scores:
        logger.warning("no reranked chunks; confidence=0.0 (will escalate)")
        return {"confidence_signal": 0.0}

    top = scores[0]
    avg_top3 = sum(scores[:3]) / min(3, len(scores))
    gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    w = CONFIDENCE_WEIGHTS
    composite = _clamp01(w["top"] * top + w["avg_top3"] * avg_top3 + w["gap"] * gap)
    logger.info(
        "confidence: composite=%.3f (top=%.3f avg3=%.3f gap=%.3f)", composite, top, avg_top3, gap
    )
    return {"confidence_signal": composite}


@_node(ESCALATE)
def escalate(state: QueryState) -> dict:
    """Static-interrupt safety net (FR-Q3.6, D14).

    Compiled with `interrupt_before=[escalate]`: the graph suspends BEFORE this
    node so an expert reviews the low-confidence candidates; `resume_query.py`
    resumes by writing `escalated=True` via `update_state` (attributed to this
    pending node, bypassing this body) and the run proceeds to synthesis. This
    body only runs on a resume without that write — still marks escalated."""
    logger.info("escalate interrupt safety-net body ran")
    return {"escalated": True}


# === Synthesis layer (§4.4) =================================================
_SYNTH_SYSTEM = (
    "You answer questions about technical documentation STRICTLY from the provided "
    "context passages. Each passage is numbered [n]. Ground every claim in the "
    "passages and cite the supporting passage numbers inline like [1] or [2][3]. "
    "Do NOT use outside knowledge or invent details. If the context does not contain "
    "the answer, say so plainly rather than guessing."
)


@_node(SYNTHESIZE_ANSWER)
def synthesize_answer(state: QueryState) -> dict:
    """FR-Q4.1/4.2/4.3 — RRF the per-unit reranked lists into one context order
    (D17; no-op for a single unit), assemble the Top-RERANK_TOP_K context, and
    synthesize a grounded, cited answer with SYNTHESIS_MODEL (D8)."""
    config = get_config()

    # D17: reuse the SAME RRF helper to fuse per-unit reranked lists (uniform
    # weight — they are all post-rerank), deduping a chunk that served >1 unit.
    by_unit = _group_by_unit(state.reranked)
    weighted_lists = [
        (1.0, sorted(hits, key=lambda h: h.get("rerank_rank", 1_000_000)))
        for hits in by_unit.values()
    ]
    fused = weighted_rrf(weighted_lists, k=RRF_K)[: config.rerank_top_k]

    if not fused:
        logger.warning("no context chunks available; returning an explicit no-answer")
        return {
            "answer": (
                "I could not find supporting information in the knowledge base to "
                "answer this question."
            ),
            "citations": [],
        }

    context_blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    for marker, hit in enumerate(fused, start=1):
        # section_path already begins with the document title (ingestion prepends
        # it), so prefer it and fall back to the title alone — avoids doubling.
        source = hit.get("section_path") or hit.get("document_title") or "?"
        context_blocks.append(f"[{marker}] (source: {source})\n{hit['text']}")
        citations.append(
            {
                "marker": marker,
                "chunk_id": hit["chunk_id"],
                "document_id": hit.get("document_id"),
                "document_title": hit.get("document_title"),
                "section_path": hit.get("section_path"),
            }
        )

    question = _retrieval_query(state)
    user = "Context passages:\n\n" + "\n\n".join(context_blocks) + f"\n\nQuestion: {question}\n\nAnswer:"
    answer = _openai().chat(
        [{"role": "system", "content": _SYNTH_SYSTEM}, {"role": "user", "content": user}],
        model=config.synthesis_model,  # FR-Q4.1 / D8 — configurable, no code change
    ).strip()

    # FR-Q4.3 — report only the citations the answer actually references inline.
    # All RERANK_TOP_K chunks still feed the LLM context (above) for grounding; we
    # just drop the ones the model didn't cite, so a focused answer doesn't list
    # weakly-related chunks as "sources". Markers are kept as-is (NOT renumbered) so
    # they still line up with the [n] references in the answer text. If the model
    # emitted no markers at all (e.g. an explicit no-answer), keep the full set so a
    # substantive answer is never returned citation-less.
    referenced = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited = [c for c in citations if c["marker"] in referenced] if referenced else citations
    logger.info(
        "synthesize: answer_len=%d citations=%d/%d referenced",
        len(answer), len(cited), len(citations),
    )
    return {"answer": answer, "citations": cited}


# === Routing (real) =========================================================
def route_after_ambiguity(state: QueryState) -> str:
    """ambiguous -> refine_query; clear -> detect_complexity (FR-Q1.2)."""
    return REFINE_QUERY if state.is_ambiguous else DETECT_COMPLEXITY


def route_after_refine(state: QueryState, *, threshold: float, max_rounds: int) -> str:
    """refine_confidence < threshold -> request_clarification, unless the
    clarification cap is reached, in which case proceed best-effort (D12)."""
    conf = state.refine_confidence if state.refine_confidence is not None else 1.0
    if conf < threshold and state.clarification_rounds < max_rounds:
        return REQUEST_CLARIFICATION
    return DETECT_COMPLEXITY


def route_after_complexity(state: QueryState) -> str:
    """complex -> decompose_query; simple -> prepare_retrieval (FR-Q1.8)."""
    return DECOMPOSE_QUERY if state.is_complex else PREPARE_RETRIEVAL


def route_after_confidence(state: QueryState, *, threshold: float) -> str:
    """composite < threshold -> escalate; else synthesize_answer (FR-Q3.5)."""
    conf = state.confidence_signal if state.confidence_signal is not None else 0.0
    return ESCALATE if conf < threshold else SYNTHESIZE_ANSWER
