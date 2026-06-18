# Hybrid vs. Simple retrieval benchmark

Measures how much the production **hybrid** pipeline (Neo4j BM25 + vector + graph,
weighted-RRF, BGE cross-encoder rerank, full query-reasoning chain) improves over
a deliberately **simple** embedding-only baseline (plain PDF text → chunk → embed
→ **pgvector** cosine top-k). It answers, on *this* corpus, the question that
started it: *"how much does hybrid search improve over simple embedding search?"*

## What's compared

| | Hybrid (production) | Simple (baseline) |
|---|---|---|
| Parse | Docling layout + tables + OCR | pypdfium2 text layer, **no OCR** |
| Chunk | sentence-pack 512/50 | **same** sentence-pack 512/50 |
| Embed | text-embedding-3-small, 1536, context prefix | **same** |
| Store | Neo4j (vector + full-text + iiRDS graph) | Postgres + **pgvector** (`baseline_chunks`) |
| Retrieve | BM25 + vector + graph → weighted-RRF → BGE rerank | **pure cosine top-k** |
| Pre-retrieval | ambiguity → refine → complexity → decompose → classify | none (raw query embedded once) |
| Synthesis | same system prompt + model | same system prompt + model |

Chunker and embedder are intentionally **shared**, so the only differences are the
**parse** step and the **retrieval** step — which is what we want to isolate.

## Prerequisites

- The hybrid corpus already ingested into Neo4j (the existing pipeline).
- Postgres with the **pgvector** extension available (confirmed: pgvector 0.8.1 on
  PostgreSQL 18; `benchmark_compare`/`baseline_ingest` run `CREATE EXTENSION IF NOT
  EXISTS vector` automatically). No new Python dependency — uses the `psycopg`
  already pulled in by the LangGraph checkpointer.

## Running it

```bash
# 1) Build the simple store from the SAME PDFs that are in Neo4j (so both stores
#    cover the same corpus). --reset starts clean.
python baseline_ingest.py --reset \
    samples/Java-TheCompleteReference-11Edition.pdf \
    samples/JavaPersistencewithHibernate.pdf
#    (directories work too: python baseline_ingest.py samples/JavaBooks)

# 2) Run the comparison over the calibration query set (or any queries file in the
#    same `<label> | <query>` format; labels are ignored here).
python benchmark_compare.py                      # all queries, top-k = RERANK_TOP_K
python benchmark_compare.py --limit 8 --k 10     # quick subset
```

Outputs (git-ignored, regenerated each run):
- `benchmark_results.csv` — per-query metrics.
- `benchmark_results.json` — full structured results incl. both answers + doc lists.
- `benchmark_report.md` — readable summary + side-by-side answers.

## What it measures (and what it does NOT)

Per query: latency, **estimated** OpenAI cost (metered on both paths), retrieved
**document-set overlap** (Jaccard + top-1 agreement, aligned by source-file hash),
and both synthesized answers side-by-side. Aggregate means + ratios at the top.

**Honest limits — read before quoting a number:**
- **No labelled relevance set.** There is no recall@k / nDCG here; answer quality
  is a *qualitative* side-by-side, not a score. Adding hand-labelled qrels is the
  next step if you want an authoritative retrieval-quality number.
- **Cost is an estimate.** Tokens are counted with `cl100k_base`; gpt-4o-mini bills
  on `o200k_base`, so LLM cost is ±~10%. Latency is exact wall-clock. Verify the
  pricing constants in `benchmark_compare.py` against current OpenAI rates.
- **Corpus parity matters.** Document overlap aligns by file hash, so it's only
  meaningful for PDFs ingested into *both* stores. The hybrid side searches the
  full Neo4j graph; the simple side searches only what `baseline_ingest.py` loaded.
  Queries whose relevant document isn't in the simple store are flagged
  `baseline_corpus_miss` so a corpus gap isn't read as a retrieval loss. (Note: the
  Documentum guides in the graph have **no source PDFs on disk**, so those queries
  can't be mirrored into the baseline.)
- **Two variables move at once** (parse *and* retrieval). This is an end-to-end
  pipeline comparison by design, not a single-variable ablation. For a clean
  retrieval-only number, re-point the baseline ingest at the *same chunk text* the
  graph holds instead of re-parsing the PDF.

## Files

- `rag/clients/pgvector_client.py` — the pgvector store (`baseline_chunks`).
- `rag/baseline/pipeline.py` — naive extract → chunk → embed → store, and query.
- `baseline_ingest.py` — CLI to populate the baseline store.
- `benchmark_compare.py` — the harness (built on `calibrate_confidence.py`).
