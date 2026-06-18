#!/usr/bin/env python
"""Hybrid-vs-simple retrieval benchmark (built on calibrate_confidence.py).

    python benchmark_compare.py [queries_file] [--k 10] [--limit N]

For every query it runs BOTH read paths and records a side-by-side comparison:

  * HYBRID  — the real production query nodes: detect_ambiguity -> refine ->
              detect_complexity -> (decompose) -> classify -> bm25 + vector +
              graph retrieve -> weighted-RRF merge -> BGE cross-encoder rerank ->
              assess_confidence -> synthesize_answer.  (Full Neo4j corpus.)
  * SIMPLE  — embed the raw query once, cosine top-k from the pgvector baseline
              store (no BM25, no graph, no rerank), then synthesize with the SAME
              system prompt + model.

It captures, per query: latency, estimated OpenAI cost (metered on both sides),
the retrieved DOCUMENT sets and their overlap (Jaccard + top-1 agreement), the
retrieved CONTENT overlap (word 5-gram shingle Jaccard + containment, which stays
discriminating even on a tiny corpus where document overlap saturates), and both
synthesized answers. Outputs a CSV (metrics), a JSON (full answers/hits), and a
readable Markdown report.

SCOPE / HONESTY (the chosen scoring tier): this measures latency, cost, retrieval
overlap, and shows both answers. It does NOT score answer correctness against a
labelled relevance set — without hand-labelled qrels there is no ground-truth
recall@k / nDCG here, so treat answer quality as a qualitative side-by-side, not
a number. Document overlap is computed by document id (SHA-256 of the source
file), which aligns ONLY for PDFs ingested into both stores; a query whose target
document isn't in the baseline store is flagged so a corpus gap isn't mistaken
for a retrieval loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from rag import PIPELINE_QUERY
from rag.baseline import naive_query
from rag.cli import init_runtime, new_thread_id
from rag.clients.openai_client import OpenAIClient
from rag.clients.pgvector_client import PgVectorStore
from rag.ingestion.nodes import EMBED_USD_PER_1M_TOKENS, _encoder
from rag.logging_config import set_context
from rag.query import nodes as qn
from rag.query.nodes import _SYNTH_SYSTEM
from rag.query.state import QueryState

# Reuse the calibration query-file parser so both tools read the same format.
from calibrate_confidence import DEFAULT_QUERIES_FILE, load_queries

RESULTS_CSV = "benchmark_results.csv"
RESULTS_JSON = "benchmark_results.json"
REPORT_MD = "benchmark_report.md"

# --- pricing (USD per 1M tokens) --------------------------------------------
# Documented constants, NOT tunables — VERIFY against current OpenAI pricing
# before quoting a cost figure (same policy as ingestion's EMBED rate). gpt-4o-mini
# rates as published June 2026. Token counts here are ESTIMATES: we count with
# tiktoken cl100k_base (the embedding tokenizer); gpt-4o-mini actually bills on
# o200k_base, so LLM token/cost figures are approximate (±~10%), good for a
# relative comparison, not an invoice.
GPT4O_MINI_USD_PER_1M_IN = 0.15
GPT4O_MINI_USD_PER_1M_OUT = 0.60


# ============================================================================
# OpenAI usage meter — patches the shared client so BOTH paths are measured.
# ============================================================================
class UsageMeter:
    """Tally OpenAI calls + (estimated) tokens by wrapping OpenAIClient methods.

    Both pipelines route every LLM/embedding call through OpenAIClient, so
    patching the class once captures the hybrid nodes' internal calls AND the
    simple path's calls. `reset()` zeroes the counters between queries.
    """

    def __init__(self) -> None:
        self._enc = _encoder()
        self.reset()
        self._installed = False

    def reset(self) -> None:
        self.embed_calls = self.chat_calls = self.struct_calls = 0
        self.embed_tokens = self.llm_in_tokens = self.llm_out_tokens = 0

    def _ntok(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    def _msgs_tok(self, messages: list[dict[str, str]]) -> int:
        return sum(self._ntok(m.get("content", "")) for m in messages)

    @property
    def llm_calls(self) -> int:
        return self.chat_calls + self.struct_calls

    def cost_usd(self) -> float:
        return (
            self.embed_tokens / 1e6 * EMBED_USD_PER_1M_TOKENS
            + self.llm_in_tokens / 1e6 * GPT4O_MINI_USD_PER_1M_IN
            + self.llm_out_tokens / 1e6 * GPT4O_MINI_USD_PER_1M_OUT
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "embed_calls": self.embed_calls,
            "llm_calls": self.llm_calls,
            "embed_tokens": self.embed_tokens,
            "llm_in_tokens": self.llm_in_tokens,
            "llm_out_tokens": self.llm_out_tokens,
            "est_cost_usd": round(self.cost_usd(), 6),
        }

    def install(self) -> None:
        if self._installed:
            return
        meter = self
        orig_embed = OpenAIClient.embed
        orig_chat = OpenAIClient.chat
        orig_struct = OpenAIClient.complete_structured

        def embed(self, texts):  # noqa: ANN001 — match wrapped signature
            meter.embed_calls += 1
            meter.embed_tokens += sum(meter._ntok(t) for t in texts)
            return orig_embed(self, texts)

        def chat(self, messages, *, model=None):  # noqa: ANN001
            meter.chat_calls += 1
            meter.llm_in_tokens += meter._msgs_tok(messages)
            out = orig_chat(self, messages, model=model)
            meter.llm_out_tokens += meter._ntok(out)
            return out

        def complete_structured(self, messages, schema, *, model=None):  # noqa: ANN001
            meter.struct_calls += 1
            meter.llm_in_tokens += meter._msgs_tok(messages)
            out = orig_struct(self, messages, schema, model=model)
            try:
                meter.llm_out_tokens += meter._ntok(out.model_dump_json())
            except Exception:  # noqa: BLE001 — token estimate only
                meter.llm_out_tokens += meter._ntok(str(out))
            return out

        OpenAIClient.embed = embed
        OpenAIClient.chat = chat
        OpenAIClient.complete_structured = complete_structured
        self._installed = True


# ============================================================================
# Run one query through each pipeline.
# ============================================================================
def _apply(state: QueryState, partial: dict) -> QueryState:
    return state.model_copy(update=partial)


def _docs_from_hits(hits: list[dict], id_key: str, title_key: str, limit: int) -> list[tuple[str, str]]:
    """Ordered-unique (doc_id, title) from a hit list, capped at `limit`."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for h in hits:
        doc_id = h.get(id_key)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append((doc_id, h.get(title_key) or "?"))
        if len(out) >= limit:
            break
    return out


def run_hybrid(query: str, config) -> dict[str, Any]:
    """Drive the real query nodes through synthesis (mirrors calibrate + answer)."""
    state = QueryState(original_query=query, thread_id=new_thread_id())
    t0 = time.perf_counter()

    state = _apply(state, qn.detect_ambiguity(state))
    if qn.route_after_ambiguity(state) == qn.REFINE_QUERY:
        state = _apply(state, qn.refine_query(state))
    # clarification interrupt deliberately skipped (we want every query retrieved)
    state = _apply(state, qn.detect_complexity(state))
    if qn.route_after_complexity(state) == qn.DECOMPOSE_QUERY:
        state = _apply(state, qn.decompose_query(state))
    state = _apply(state, qn.prepare_retrieval(state))
    state = _apply(state, qn.classify_query(state))
    state = _apply(state, qn.bm25_retrieve(state))
    state = _apply(state, qn.vector_retrieve(state))
    state = _apply(state, qn.graph_traverse(state))
    state = _apply(state, qn.merge(state))
    state = _apply(state, qn.cross_encoder_rerank(state))
    state = _apply(state, qn.assess_confidence(state))
    state = _apply(state, qn.synthesize_answer(state))

    latency_ms = (time.perf_counter() - t0) * 1000.0
    reranked = sorted(state.reranked, key=lambda h: h.get("rerank_score", 0.0), reverse=True)
    docs = _docs_from_hits(reranked, "document_id", "document_title", config.rerank_top_k)
    threshold = config.escalate_confidence_threshold
    conf = state.confidence_signal or 0.0
    return {
        "answer": state.answer or "",
        "docs": docs,
        "query_class": state.query_class.value if state.query_class else "?",
        "confidence": round(conf, 4),
        "would_escalate": conf < threshold,
        "n_reranked": len(state.reranked),
        "texts": [h.get("text") or "" for h in reranked[: config.rerank_top_k]],
        "latency_ms": round(latency_ms, 1),
    }


def run_simple(query: str, store: PgVectorStore, openai: OpenAIClient, config) -> dict[str, Any]:
    """Embed → pgvector cosine top-k → synthesize with the same prompt/model."""
    k = config.rerank_top_k  # same context budget as the hybrid side
    t0 = time.perf_counter()
    hits = naive_query(query, store, openai, k)

    if not hits:
        answer = (
            "I could not find supporting information in the knowledge base to answer "
            "this question."
        )
    else:
        blocks = []
        for marker, h in enumerate(hits, start=1):
            source = h.get("document_title") or "?"
            blocks.append(f"[{marker}] (source: {source})\n{h['text']}")
        user = (
            "Context passages:\n\n" + "\n\n".join(blocks)
            + f"\n\nQuestion: {query}\n\nAnswer:"
        )
        answer = openai.chat(
            [{"role": "system", "content": _SYNTH_SYSTEM}, {"role": "user", "content": user}],
            model=config.synthesis_model,
        ).strip()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    docs = _docs_from_hits(hits, "doc_hash", "document_title", k)
    return {
        "answer": answer,
        "docs": docs,
        "n_hits": len(hits),
        "texts": [h.get("text") or "" for h in hits],
        "latency_ms": round(latency_ms, 1),
    }


def _overlap(hybrid_docs: list[tuple[str, str]], simple_docs: list[tuple[str, str]]) -> dict[str, Any]:
    """Document-set agreement between the two top-k results (by document id)."""
    h_ids = {d for d, _ in hybrid_docs}
    s_ids = {d for d, _ in simple_docs}
    inter = h_ids & s_ids
    union = h_ids | s_ids
    jaccard = len(inter) / len(union) if union else 0.0
    top1_match = bool(hybrid_docs and simple_docs and hybrid_docs[0][0] == simple_docs[0][0])
    # corpus gap: the simple store didn't contain ANY doc the hybrid found relevant
    baseline_miss = bool(hybrid_docs) and not (s_ids & h_ids)
    return {
        "doc_jaccard": round(jaccard, 3),
        "top1_doc_match": top1_match,
        "shared_docs": len(inter),
        "baseline_corpus_miss": baseline_miss,
    }


# --- content-level overlap (chunk-boundary agnostic) ------------------------
# Document-set overlap saturates near 1.0 on a small corpus (almost every hit
# maps to the same handful of files), so it can't tell the two retrievers apart.
# The hybrid and simple stores ALSO chunk the same source differently, so chunk
# ids never align across them — a chunk_id Jaccard would always be 0. We instead
# compare the retrieved *text* via word n-gram shingles: this measures whether
# the two paths surfaced the same underlying passages regardless of where each
# drew its chunk boundaries.
_SHINGLE_N = 5
_WORD_RE = re.compile(r"[a-z0-9]+")


def _shingles(texts: list[str], n: int = _SHINGLE_N) -> set[tuple[str, ...]]:
    """Set of word n-gram shingles over the concatenated passages (lower-cased)."""
    sh: set[tuple[str, ...]] = set()
    for t in texts:
        toks = _WORD_RE.findall((t or "").lower())
        for i in range(len(toks) - n + 1):
            sh.add(tuple(toks[i : i + n]))
    return sh


def _content_overlap(hybrid_texts: list[str], simple_texts: list[str]) -> dict[str, Any]:
    """Retrieved-content agreement: shingle Jaccard + containment of the smaller set.

    `content_jaccard` = |H∩S| / |H∪S| — symmetric, but penalized when the two
    sides retrieve very different *amounts* of text (the simple side packs fuller
    chunks). `content_containment` = |H∩S| / min(|H|,|S|) — "how much of the
    smaller passage set is also covered by the other path", robust to that volume
    asymmetry. Read them together.
    """
    h = _shingles(hybrid_texts)
    s = _shingles(simple_texts)
    inter = len(h & s)
    union = len(h | s)
    smaller = min(len(h), len(s))
    return {
        "content_jaccard": round(inter / union, 3) if union else 0.0,
        "content_containment": round(inter / smaller, 3) if smaller else 0.0,
        "content_shingles_hybrid": len(h),
        "content_shingles_simple": len(s),
    }


# ============================================================================
# Report writers.
# ============================================================================
def _write_csv(rows: list[dict]) -> None:
    fields = [
        "query", "hybrid_class", "hybrid_conf", "hybrid_escalate",
        "doc_jaccard", "top1_doc_match", "baseline_corpus_miss",
        "content_jaccard", "content_containment",
        "hybrid_latency_ms", "simple_latency_ms",
        "hybrid_cost_usd", "simple_cost_usd",
        "hybrid_llm_calls", "simple_llm_calls",
        "hybrid_top_doc", "simple_top_doc",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _truncate(text: str, n: int = 700) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + " …[truncated]"


def _write_report(rows: list[dict], agg: dict, store_docs: list[dict]) -> None:
    L: list[str] = []
    L.append("# Hybrid vs. Simple (embedding-only) retrieval benchmark\n")
    L.append("_Hybrid = Neo4j BM25 + vector + graph, weighted-RRF, BGE rerank, full reasoning chain._  ")
    L.append("_Simple = pypdfium text → chunk → embed → **pgvector cosine top-k**, no rerank._\n")

    L.append("## Summary\n")
    L.append(f"- Queries compared: **{agg['n']}**")
    L.append(f"- Mean document-set Jaccard (top-k): **{agg['mean_jaccard']:.3f}**")
    L.append(f"- Top-1 document agreement: **{agg['top1_rate']:.0%}**")
    L.append(f"- **Mean retrieved-content overlap (5-gram shingle Jaccard): {agg['mean_content_jaccard']:.3f}** "
             f"· containment of smaller set: **{agg['mean_content_containment']:.3f}**")
    L.append(f"- Queries outside the baseline corpus (flagged): **{agg['baseline_misses']}**")
    L.append(f"- Mean latency — hybrid **{agg['mean_hybrid_latency']:.0f} ms** vs simple "
             f"**{agg['mean_simple_latency']:.0f} ms** "
             f"({agg['latency_ratio']:.1f}× )")
    L.append(f"- Est. cost/query — hybrid **${agg['mean_hybrid_cost']:.5f}** vs simple "
             f"**${agg['mean_simple_cost']:.5f}** "
             f"({agg['cost_ratio']:.1f}× )")
    L.append(f"- Mean LLM calls/query — hybrid **{agg['mean_hybrid_calls']:.1f}** vs simple "
             f"**{agg['mean_simple_calls']:.1f}**\n")

    L.append("> **Why two overlap numbers?** **Document-set Jaccard** saturates when the "
             "corpus has few documents (almost every hit maps to the same handful of files), "
             "so it can't tell the two retrievers apart. **Retrieved-content overlap** compares "
             "the actual passage *text* via word 5-gram shingles, so it still discriminates how "
             "differently the two paths retrieve even *within* the same document(s). The two "
             "stores chunk the source differently, so chunk ids can't be matched directly — "
             "shingles are chunk-boundary agnostic. *Jaccard* is symmetric; *containment* "
             "(`|H∩S| / min(|H|,|S|)`) is robust to the simple side packing fuller chunks.\n")

    L.append("> **Caveat:** answer quality below is a qualitative side-by-side. No "
             "labelled relevance set was used, so there is no recall@k / nDCG number here. "
             "Document overlap aligns by source-file hash, so it is only meaningful for "
             "PDFs ingested into BOTH stores — rows flagged `baseline_corpus_miss` are "
             "queries whose relevant document isn't in the simple store.\n")

    L.append(f"## Baseline store corpus ({len(store_docs)} document(s))\n")
    for d in store_docs:
        L.append(f"- {d.get('title') or '?'} — {d['chunks']} chunks")
    L.append("")

    L.append("## Per-query comparison\n")
    for i, r in enumerate(rows, start=1):
        L.append(f"### {i}. {r['query']}\n")
        flag = "  ⚠️ _relevant doc not in baseline store_" if r["baseline_corpus_miss"] else ""
        L.append(f"- hybrid class=`{r['hybrid_class']}` confidence={r['hybrid_conf']:.3f}"
                 f"{' (would escalate)' if r['hybrid_escalate'] else ''}")
        L.append(f"- doc Jaccard={r['doc_jaccard']:.2f}  top-1 match={r['top1_doc_match']}{flag}")
        L.append(f"- content overlap: shingle Jaccard={r['content_jaccard']:.2f}  "
                 f"containment={r['content_containment']:.2f}")
        L.append(f"- latency: hybrid {r['hybrid_latency_ms']:.0f} ms / simple {r['simple_latency_ms']:.0f} ms  ·  "
                 f"est cost: hybrid ${r['hybrid_cost_usd']:.5f} / simple ${r['simple_cost_usd']:.5f}")
        L.append(f"- hybrid docs: {', '.join(t for _, t in r['hybrid_docs']) or '(none)'}")
        L.append(f"- simple docs: {', '.join(t for _, t in r['simple_docs']) or '(none)'}")
        L.append(f"\n**Hybrid answer:** {_truncate(r['hybrid_answer'])}\n")
        L.append(f"**Simple answer:** {_truncate(r['simple_answer'])}\n")
        L.append("---\n")

    Path(REPORT_MD).write_text("\n".join(L), encoding="utf-8")


# ============================================================================
# Main.
# ============================================================================
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Benchmark hybrid vs simple retrieval.")
    parser.add_argument("queries_file", nargs="?", default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--k", type=int, default=None, help="top-k (default: RERANK_TOP_K)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    args = parser.parse_args(argv)

    path = Path(args.queries_file)
    if not path.is_file():
        print(f"queries file not found: {path}", file=sys.stderr)
        return 2

    config = init_runtime(PIPELINE_QUERY)
    set_context(pipeline=PIPELINE_QUERY)
    logging.getLogger("rag").setLevel(logging.WARNING)  # quiet per-node chatter
    if args.k:
        config = config.model_copy(update={"rerank_top_k": args.k})

    store = PgVectorStore(config)
    store.ensure_schema()
    store_docs = store.documents()
    store_count = store.count()
    if store_count == 0:
        print("baseline store is EMPTY — run baseline_ingest.py first.", file=sys.stderr)
        return 2
    openai = OpenAIClient(config)

    meter = UsageMeter()
    meter.install()

    queries = load_queries(path)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print(f"no queries found in {path}", file=sys.stderr)
        return 2

    print(f"Benchmarking {len(queries)} query(ies); baseline store = {store_count} chunks "
          f"over {len(store_docs)} docs; top-k = {config.rerank_top_k}\n")
    hdr = f"{'#':>3}  {'djac':>5}  {'cjac':>5}  {'ccon':>5}  {'t1':>3}  {'h_ms':>7}  {'s_ms':>7}  {'h_$':>8}  {'s_$':>8}  query"
    print(hdr)
    print("-" * len(hdr))

    rows: list[dict] = []
    for i, (_label, query) in enumerate(queries, start=1):
        try:
            meter.reset()
            hyb = run_hybrid(query, config)
            hyb_usage = meter.snapshot()

            meter.reset()
            sim = run_simple(query, store, openai, config)
            sim_usage = meter.snapshot()
        except Exception as exc:  # noqa: BLE001 — keep going on a bad query
            print(f"{i:>3}  ERROR: {exc}  ({query[:50]})")
            continue

        ov = _overlap(hyb["docs"], sim["docs"])
        cov = _content_overlap(hyb["texts"], sim["texts"])
        row = {
            "query": query,
            "hybrid_class": hyb["query_class"],
            "hybrid_conf": hyb["confidence"],
            "hybrid_escalate": hyb["would_escalate"],
            **ov,
            **cov,
            "hybrid_latency_ms": hyb["latency_ms"],
            "simple_latency_ms": sim["latency_ms"],
            "hybrid_cost_usd": hyb_usage["est_cost_usd"],
            "simple_cost_usd": sim_usage["est_cost_usd"],
            "hybrid_llm_calls": hyb_usage["llm_calls"],
            "simple_llm_calls": sim_usage["llm_calls"],
            "hybrid_top_doc": hyb["docs"][0][1] if hyb["docs"] else "",
            "simple_top_doc": sim["docs"][0][1] if sim["docs"] else "",
            # carried for the JSON/MD report (not the CSV)
            "hybrid_docs": hyb["docs"],
            "simple_docs": sim["docs"],
            "hybrid_answer": hyb["answer"],
            "simple_answer": sim["answer"],
        }
        rows.append(row)
        print(f"{i:>3}  {ov['doc_jaccard']:>5.2f}  {cov['content_jaccard']:>5.2f}  {cov['content_containment']:>5.2f}  "
              f"{('Y' if ov['top1_doc_match'] else '·'):>3}  "
              f"{hyb['latency_ms']:>7.0f}  {sim['latency_ms']:>7.0f}  "
              f"{hyb_usage['est_cost_usd']:>8.5f}  {sim_usage['est_cost_usd']:>8.5f}  {query[:48]}")

    if not rows:
        print("no successful comparisons", file=sys.stderr)
        return 1

    n = len(rows)
    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n
    agg = {
        "n": n,
        "mean_jaccard": mean("doc_jaccard"),
        "mean_content_jaccard": mean("content_jaccard"),
        "mean_content_containment": mean("content_containment"),
        "top1_rate": sum(1 for r in rows if r["top1_doc_match"]) / n,
        "baseline_misses": sum(1 for r in rows if r["baseline_corpus_miss"]),
        "mean_hybrid_latency": mean("hybrid_latency_ms"),
        "mean_simple_latency": mean("simple_latency_ms"),
        "mean_hybrid_cost": mean("hybrid_cost_usd"),
        "mean_simple_cost": mean("simple_cost_usd"),
        "mean_hybrid_calls": mean("hybrid_llm_calls"),
        "mean_simple_calls": mean("simple_llm_calls"),
    }
    agg["latency_ratio"] = (agg["mean_hybrid_latency"] / agg["mean_simple_latency"]
                            if agg["mean_simple_latency"] else 0.0)
    agg["cost_ratio"] = (agg["mean_hybrid_cost"] / agg["mean_simple_cost"]
                         if agg["mean_simple_cost"] else 0.0)

    _write_csv(rows)
    Path(RESULTS_JSON).write_text(
        json.dumps({"aggregate": agg, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(rows, agg, store_docs)

    print(
        f"\n— Summary over {n} query(ies) —\n"
        f"  doc Jaccard (mean):      {agg['mean_jaccard']:.3f}\n"
        f"  content overlap (mean):  jaccard {agg['mean_content_jaccard']:.3f} / "
        f"containment {agg['mean_content_containment']:.3f}\n"
        f"  top-1 doc agreement:     {agg['top1_rate']:.0%}\n"
        f"  baseline corpus misses:  {agg['baseline_misses']}\n"
        f"  latency  hybrid/simple:  {agg['mean_hybrid_latency']:.0f} / {agg['mean_simple_latency']:.0f} ms "
        f"({agg['latency_ratio']:.1f}×)\n"
        f"  est cost hybrid/simple:  ${agg['mean_hybrid_cost']:.5f} / ${agg['mean_simple_cost']:.5f} "
        f"({agg['cost_ratio']:.1f}×)\n"
        f"  LLM calls hybrid/simple: {agg['mean_hybrid_calls']:.1f} / {agg['mean_simple_calls']:.1f}\n"
        f"\nWrote {RESULTS_CSV}, {RESULTS_JSON}, {REPORT_MD}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
