"""Weighted Reciprocal Rank Fusion — the single fusion primitive (D16/D17).

One helper serves BOTH fusion points by design (D17):

* **merge** (D16 / FR-Q2.6): per retrieval unit, fuse the three retrievers'
  ranked hit lists weighted by `query_class`, dedup by chunk id within the unit.
* **synthesize_answer** (D17 / FR-Q4.2): fuse the per-unit re-ranked lists
  (uniform weight) into one context order before LLM assembly — a real no-op for
  a single-unit (simple) query.

RRF is **rank-based, so it is scale-invariant** across the incompatible Lucene
(BM25), cosine, and graph-hop score scales, and a retriever that returns nothing
simply contributes nothing — graceful degradation for free (NFR-REL-9).
No magic numbers (NFR-MAINT-2): the RRF constant and the per-class×retriever
weight matrix are the documented constants below.
"""

from __future__ import annotations

from typing import Any, Sequence

# Standard RRF damping constant. Larger k flattens the contribution of top ranks;
# 60 is the canonical value from Cormack et al. (2009) and the common default.
# Distinct from PER_RETRIEVER_K/RETRIEVE_TOP_K (those are list sizes, this is the
# fusion denominator). Documented constant, not env-promoted (NFR-MAINT-2).
RRF_K = 60

# Per-`query_class` × retriever weight matrix (D10 / D16 — was TBD, fixed here).
# Weights SCALE each retriever's RRF contribution; every retriever keeps a
# non-zero weight so none is ever excluded (D10 — weight, don't gate). The
# dominant retriever for a class gets 1.0, a strong secondary 0.5, the weakest
# 0.3, reflecting FR-Q2.2's intent (BM25↑ for exact lookups, vector↑ for
# conceptual/procedural NL, graph↑ for relational). Documented tunable constant
# (NFR-MAINT-2); revisit against retrieval-quality measurement.
RETRIEVER_WEIGHTS: dict[str, dict[str, float]] = {
    "exact_lookup": {"bm25": 1.0, "vector": 0.5, "graph": 0.3},
    "conceptual":   {"bm25": 0.3, "vector": 1.0, "graph": 0.5},
    "procedural":   {"bm25": 0.5, "vector": 1.0, "graph": 0.5},
    "relational":   {"bm25": 0.3, "vector": 0.5, "graph": 1.0},
}

# Fallback when `query_class` is unknown/None: treat all retrievers equally
# (still all-run, D10). Keeps fusion well-defined without privileging a retriever.
UNIFORM_WEIGHTS: dict[str, float] = {"bm25": 1.0, "vector": 1.0, "graph": 1.0}


def weights_for_class(query_class: str | None) -> dict[str, float]:
    """Return the retriever weight vector for a class value (UNIFORM if unknown)."""
    if query_class is None:
        return UNIFORM_WEIGHTS
    return RETRIEVER_WEIGHTS.get(query_class, UNIFORM_WEIGHTS)


def weighted_rrf(
    weighted_lists: Sequence[tuple[float, Sequence[dict[str, Any]]]],
    *,
    k: int = RRF_K,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """Fuse several ranked hit lists by weighted RRF, deduped by `id_key`.

    `weighted_lists` is a sequence of `(weight, hits)` pairs where `hits` is
    already ordered best-first (rank = position). The fused score for a chunk is
    ``Σ weight / (k + rank)`` over every list it appears in (D16/D17). Returns
    NEW hit dicts (copies of the first-seen occurrence) sorted by descending
    fused score, each carrying an added ``rrf_score``. Ties break on first-seen
    order, which is stable.
    """
    scores: dict[Any, float] = {}
    first_seen: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []  # insertion order → stable tie-break

    for weight, hits in weighted_lists:
        if weight == 0.0:
            continue
        for rank, hit in enumerate(hits, start=1):
            cid = hit[id_key]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            if cid not in first_seen:
                first_seen[cid] = hit
                order.append(cid)

    ranked_ids = sorted(order, key=lambda cid: scores[cid], reverse=True)
    fused: list[dict[str, Any]] = []
    for cid in ranked_ids:
        merged = dict(first_seen[cid])
        merged["rrf_score"] = round(scores[cid], 6)
        fused.append(merged)
    return fused
