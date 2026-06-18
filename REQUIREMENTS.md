# Requirements Specification — LangGraph RAG Pipeline (Ingestion + Query)

**Project:** Local RAG system over technical documentation — Neo4j knowledge graph + hybrid retrieval
**Author:** (drafted with Claude Code)
**Date:** 2026-06-15
**Status:** Draft v2.19 — for review
**Sources of truth:**
- `ingestion_pipeline_diagram.html` — write path (7-stage ingestion)
- `rag_pipeline_diagram.html` — read path (query → answer)

> **Changes from v1.0:** Python **3.14 → 3.13** (resolves all GPU/Docling/dependency risks — §7) and added the complete **query / read-path** spec (§4).
> **v2.1:** pinned Python **3.13.14**; GPU reranking made a hard requirement (D5); EasyOCR locked (D6); explicit warm-up bootstrap (D7, §2.4).
> **v2.2:** resolved all remaining open questions (D8–D14): synthesis model, iiRDS enums, retriever fan-out, Top-K, loop caps, config schema (§2.5), human-in-loop UX.
> **v2.3:** checkpoint store changed **SQLite → Postgres** (D15).
> **v2.4:** Postgres deployed as a **native on-prem install on the laptop** (not Docker); Neo4j remains Dockerized.
> **v2.5:** consolidated logging into a first-class **§5.8 NFR-LOG** (configurable level/format/destination, rotation, redaction); added `LOG_*` config keys.
> **v2.6:** locked the query read-path build decisions — **D16** (per-unit retrieval loop + weighted-RRF merge) and **D17** (RRF at synthesis, shared helper); added the **§4.0a graph topology diagram**; noted the `stage_timings_ms` parallel-write reducer (FR-Q0.3). The per-`query_class` weight matrix (D10/D16) and the FR-Q3.4 confidence-formula weights remain **TBD (implementation)** — mechanism locked, constants pending (NFR-MAINT-2).
> **v2.7:** **query read-path implemented + the two TBDs fixed.** The D10/D16 per-`query_class`×retriever weight matrix and the FR-Q3.4 composite-confidence term weights are now documented constants (in `rag/query/fusion.py` and `rag/query/nodes.py` respectively; see D16/FR-Q2.6 and FR-Q3.4); added the `RRF_K`=60 fusion constant (D16/D17). Added the `clarification_question` state field (FR-Q0.3/Q1.5), produced by `refine_query` and surfaced by `resume_query.py`. All 15 node bodies, the `stage_timings_ms` reducer, the shared RRF helper, and the `query.py` / `resume_query.py` CLIs are built.
> **v2.19:** **decoupled OCR render DPI (FR-2.8d).** `PDF_RENDER_DPI` (default 72) was used for both born-digital and scanned parsing. For a born-digital PDF the page bitmap is only a layout backdrop, so 72 dpi is fine; but for a **scanned** page the bitmap is the *only* text source EasyOCR reads, and 72 dpi makes small glyphs (subscripts, math notation, diacritics) marginal. New `PDF_RENDER_DPI_OCR` (default **150**) is applied by the `parse` node **only when the OCR decision is on** (FR-2.3e), via the same effective-config fold that already selects the smaller OCR slice size; the born-digital path keeps `PDF_RENDER_DPI=72` unchanged. Higher OCR DPI costs memory/time (~dpi²), so the OCR slice size default `PDF_PARSE_BATCH_PAGES_OCR` is **lowered 25 → 12** to keep per-slice memory bounded under the sharper render (150 dpi ≈ 4.3× the pixels of 72; halving the slice nets ~2× the validated 72-dpi/25-page peak instead of ~4.3×); lower it further if a higher DPI still OOMs. Added to §2.5 config schema + `.env.example`. Defaults preserve born-digital behavior; the only change is sharper scanned-PDF OCR.
>
> **v2.18:** **configurable OCR languages + explicit encrypted-PDF rejection.** (1) `OCR_LANGUAGES` (FR-2.3g, default `fr,de,es,en` = EasyOCR's built-in Latin set) is passed to `EasyOcrOptions(lang=…)`, so non-Latin scans work (e.g. `ch_sim,en`, `ja,en`, `ar,en`, `ru,en`). Comma/space-separated; the set must be script-compatible (EasyOCR raises otherwise). (2) Intake now detects **password-protected (encrypted) PDFs** (FR-1.2a) — via pypdfium2's `FPDF_ERR_PASSWORD`, message fallback — and terminates with a clear "not supported, remove the password" message instead of a generic downstream parse error. Added to §2.5 config schema + `.env.example`.
>
> **v2.17:** **per-run application log files (NFR-LOG-3).** The on-disk app log changed from a single shared `logs/rag.log` to **one file per process invocation**, `logs/<pipeline>_<UTCstamp>_<pid>.log` (e.g. `logs/ingest_20260617T131451Z_56432.log`). Named at logging-setup time (before a `thread_id` exists) from the pipeline + UTC start time + PID, so runs never share a file; size-based rotation (10 MB × 5) still caps a single very long run. Mirrors the v2.16 per-run audit-file change. `setup_logging(config, pipeline)` gained the pipeline label; `init_runtime`/`bootstrap` pass it.
>
> **v2.16:** **per-run ingestion audit files (FR-8.1a / NFR-OBS-4).** The ingestion audit trail changed from one appended `ingestion_log.json` to **one JSON file per run** under `ingestion_logs/`, named `ingestion_log_<UTCstamp>_<thread8>_<source-stem>.json` (a single record object, not an array). Runs no longer share/rewrite a growing file — they sort chronologically and each is self-contained. The legacy `ingestion_log.json` is left in place (still git-ignored). The Neo4j `IngestionRecord` (FR-8.1b) is unchanged.
>
> **v2.15:** **automatic OCR selection per PDF (FR-2.3e/2.3f).** `OCR_ENABLED` became tri-state — `auto` (new default) | `on`/`true` | `off`/`false`. In `auto`, the `parse` node classifies each PDF from its own text layer via a shared classifier (`rag/ingestion/pdf_kind.py`): it samples up to 1000 pages and, for text-less pages, measures image coverage, labelling the file **digital** / **scanned** / **mixed** over *text-vs-scanned* pages only (blank/figure pages excluded, so a born-digital book full of diagrams isn't mis-flagged). OCR is enabled only for scanned/mixed; the OCR slice size (`PDF_PARSE_BATCH_PAGES_OCR`, default 25) is chosen automatically. This removes the per-file `OCR_ENABLED`/`PDF_PARSE_BATCH_PAGES` hand-tuning the `ingest_folder.ps1` wrapper used to do (the wrapper is now a thin batch loop). `detect_pdf_kind.py` is retained as a thin CLI over the same classifier. Added to §2.5 config schema + `.env.example`.
>
> **v2.14:** **batched Neo4j write for very large books (FR-7.1/7.8).** The per-document write was a **single transaction** over all chunks; on a large book it reached **~243,658 store commands** and Neo4j **failed to apply it to the store** (`Failed to apply transaction` → `database needs to be restarted`), wedging the DB so even reads failed until the volume was recreated. The write is now split into bounded transactions — `Document` + iiRDS edges first, then chunks in batches of the new `NEO4J_WRITE_BATCH_CHUNKS` (default 250), one transaction each. Strict single-transaction atomicity is replaced by **compensation**: any failure deletes the document + all committed chunks (itself batched via `CALL { … } IN TRANSACTIONS`), preserving the zero-partial-data guarantee (NFR-REL-1). Added to §2.5 config schema + `.env.example`.
>
> **v2.13:** **page-range batched parsing for very large PDFs (FR-2.8c).** `PDF_RENDER_DPI` was confirmed insufficient — a 610-page book OOM'd at a fixed ~page-127 depth regardless of DPI (the `std::bad_alloc` accumulation is in Docling's parse backend, not the page bitmaps). New `PDF_PARSE_BATCH_PAGES` (default 100) parses the PDF in page-range slices via repeated `convert(page_range=…)` on a reused converter, freeing memory between slices; `chunk` consumes the ordered slice list with continuous section/position tracking. Verified: the 610-page book now parses all pages (`failed_pages=0`, 7 slices, 2618 chunks vs ~1300 when half was dropped). Added to §2.5 config schema + `.env.example`.
>
> **v2.12:** **fixed checkpointer crash on PDF ingest + added partial-parse visibility.** (1) The parsed `DoclingDocument` was held in checkpointed state; the Postgres checkpointer cannot msgpack-serialize it (works for tiny markdown docs via `model_dump`, fails for large PDFs), crashing every PDF ingest at end-of-run with `TypeError: Type is not msgpack serializable: DoclingDocument`. Now passed `parse`→`chunk` via a process-local handoff, never persisted (FR-2.4). (2) Discovered that Docling `PARTIAL_SUCCESS` (pages dropped under memory pressure) was silently reported as a clean ingest — added `parse_failed_pages` tracking, a WARNING, and an `⚠ INCOMPLETE` summary line (FR-2.6a). Verified: 610-page PDF ingests to exit 0 (was crashing), and the summary now flags the ~296 pages Docling dropped.
>
> **v2.11:** **citations trimmed to referenced markers (FR-Q4.3).** The answer previously reported all Top-`RERANK_TOP_K` chunks as citations, so a focused answer listed weakly-related chunks (e.g. off-topic docs surfaced by hybrid retrieval) as "sources". `synthesize_answer` now filters returned citations to the `[n]` markers the answer actually references inline (context window unchanged — all chunks still feed the LLM for grounding); markers kept as-is, full set retained if the model emits none. Verified end-to-end (pump query → 1 citation vs. 10 before).
>
> **v2.10:** **fixed document title resolution (FR-2.7).** Because parsing runs from a private temp file (FR-2.5), Docling's `doc.name` was the `rag_ingest_*` temp stem, so any PDF without an on-page TITLE element got a junk title. New order: TITLE element → embedded PDF `/Title` (via pypdfium2, a Docling dep) → file stem. Backfilled the already-ingested *Java: The Complete Reference* document in the graph in place (hash-matched, single-property `SET d.title` — no re-ingest, citations read `d.title` live).
>
> **v2.9:** **calibrated `ESCALATE_CONFIDENCE_THRESHOLD` against the real corpus** (Java: The Complete Reference + OpenText Documentum CM manuals, ~5.3k chunks) via `calibrate_confidence.py` over 25 labeled queries. Answerable (n=17) scored 0.639–0.799; escalate (n=8) scored 0.000–0.120 — **cleanly separable**, so `CONFIDENCE_WEIGHTS` needs no re-weighting. Default lowered **0.40 → 0.38** (the midpoint of the empty 0.12–0.64 gap; 100% balanced accuracy). This **resolves the prior caveat** that BGE over-escalates — that was an artifact of the tiny synthetic corpus, not the real one. Updated FR-Q3.4, FR-Q3.5, §2.5 schema, `.env.example`.
>
> **v2.8:** added **Docling PDF memory & cost controls** to bound parse-time memory on very large PDFs (motivated by a 99 MB / ~1300-page born-digital book OOM-ing with `std::bad_alloc`): `OCR_ENABLED` (FR-2.3e), `PDF_MAX_PAGES` and `PDF_RENDER_DPI` (FR-2.8) — wired through `config.py` → `docling_client.build_converter` → `nodes.py::parse`. New config keys added to the §2.5 schema and `.env.example`. **Defaults preserve prior behavior** (OCR on, no page cap, scale 1.0); the primary large-PDF mitigation is `OCR_ENABLED=false`.

---

## 1. Overview

### 1.1 Purpose
This document specifies functional and non-functional requirements for a **local, single-user RAG system** over technical documentation. It has two pipelines, both orchestrated as **LangGraph** state machines:

- **Ingestion (write path)** — converts source documents into a tagged, vector-searchable Neo4j knowledge graph (§3).
- **Query (read path)** — answers natural-language questions over that graph using ambiguity handling, hybrid retrieval, local cross-encoder re-ranking, and LLM synthesis (§4).

### 1.2 Scope
| Pipeline | Stages |
|----------|--------|
| **Ingestion** | Intake → Docling parse → structural chunk → iiRDS tag → embed → atomic Neo4j write → receipt |
| **Query** | Ambiguity detection → (refine / clarify) → complexity detection → (decompose) → prepare → classify → hybrid retrieve (BM25 + vector + graph) → cross-encoder re-rank → confidence assessment → (escalate) → synthesize |

Both use a **Postgres checkpointer** (D15) for run state and **human-in-the-loop interrupts**.

### 1.3 Out of Scope
- Multi-user concurrency, web UI/REST service, auth/authz, multi-tenant isolation (CLI-driven, single user)
- Document deletion / graph maintenance / GDPR erasure
- Re-ingestion / in-place update of an existing document (explicitly **not supported** — FR-1.5)
- Cloud deployment, horizontal scaling, message queues
- Fine-tuning of any model
- Conversational memory across turns (each query is independent unless a clarification loop is active)

### 1.4 Confirmed Decisions (from stakeholder)
| # | Decision | Choice |
|---|----------|--------|
| D1 | LLM & embeddings | **OpenAI cloud** (`gpt-4o-mini`, `text-embedding-3-small`). Reranker runs **locally**. Only Neo4j + pipeline code + reranker run on-device. |
| D2 | Python version | **Python 3.13.14** (specific patch; clears all GPU/Docling/dependency issues from the 3.14 plan). |
| D3 | Workload | **Single-user / small volume** (one document or one query at a time). |
| D4 | Deliverable | Markdown requirements document in repo. |
| D5 | GPU acceleration | **Required** — BGE reranker runs on the NVIDIA GPU (CUDA), with CPU fallback. Docling MAY also use the GPU. See verified hardware in §6.2. |
| D6 | OCR engine | **EasyOCR** (locked). Installed via `docling[easyocr]`; runs locally on the GPU reusing the shared CUDA PyTorch (no system binaries). Chosen over RapidOCR (CPU-only default backend on Windows) and Tesseract (requires a system binary). |
| D7 | Model/env provisioning | **Explicit one-time warm-up step required.** A bootstrap command pre-downloads and verifies all local weights (Docling, BGE, EasyOCR), asserts the GPU, and creates the Neo4j indexes — before any ingest/query. Lazy first-run download retained only as fallback. Resolves Open Question #7. See §2.4. |
| D8 | Synthesis model | Default **`gpt-4o-mini`** for all reasoning **and** synthesis (faithful to the diagrams); the `synthesize_answer` model is **configurable** (`SYNTHESIS_MODEL`) and may be raised to `gpt-4o` with no code change if answer quality requires. |
| D9 | iiRDS vocabulary | `lifecycle_phase` and `information_type` are **closed enumerations** (the diagram's value sets), validated against **config-defined** lists (extensible to the full iiRDS vocabulary later). `product`/`component` remain **open free-text**, `MERGE`d in the graph. |
| D10 | Retriever fan-out | **Always run all three retrievers in parallel**, then merge; `query_class` is used to **weight** contributions at merge time, **not** to exclude any retriever. (All retrievers are local/cheap; maximizes recall, rerank handles precision.) |
| D11 | Top-K | Per-retriever **K=25** → merged/deduped **Top-50** → reranked **Top-10**. All configurable (§2.5). |
| D12 | Loop caps | Clarification loop **≤ 2 rounds**, then proceed best-effort with a low-confidence flag. Decomposition is **single-level**, **≤ 5 sub-queries**. Bounds RISK-F. |
| D13 | Config schema | All secrets + tunables via env/`.env` with an **explicit documented schema (§2.5)** and a shipped `.env.example`. No magic numbers inline. |
| D14 | Human-in-loop UX | All three interrupts (tag review, clarification, escalation) use the **same `interrupt()` + `thread_id` resume-CLI** pattern, for consistency and durable state. |
| D15 | Checkpoint store | **Postgres** via LangGraph `langgraph-checkpoint-postgres` (replacing SQLite). Runs as a **native on-prem install on the laptop** (localhost:5432) — **not** Docker. Bootstrap creates the checkpointer tables (FR-S0.5a). |
| D16 | Retrieval execution & merge fusion | Each retriever **loops over `retrieval_units`**, tagging every hit dict with its originating **`unit`** + per-unit rank; a **single `query_class`** is assigned to the whole query (not per unit, matching the typed-state field). `merge` fuses the **three retrievers per unit** via **weighted Reciprocal Rank Fusion** — `score = Σ weight[query_class][retriever] / (k + rank)` — dedup by chunk id **within unit**, cap Top-50; results stay in `merged_candidates` (`list[dict]`) **carrying the `unit` tag** — unit provenance is a dict convention, **no new state field**. Rank-based fusion is scale-invariant across the incompatible Lucene/cosine/graph score scales and degrades for free when a retriever returns nothing (NFR-REL-9). Refines D10 / FR-Q2.2 / FR-Q2.6. **The per-`query_class`×retriever weight matrix (FIXED v2.7, `rag/query/fusion.py::RETRIEVER_WEIGHTS`):** every retriever keeps a non-zero weight (weight, never gate — D10); dominant=1.0 / secondary=0.5 / weakest=0.3 — `exact_lookup`={bm25:1.0, vector:0.5, graph:0.3}, `conceptual`={bm25:0.3, vector:1.0, graph:0.5}, `procedural`={bm25:0.5, vector:1.0, graph:0.5}, `relational`={bm25:0.3, vector:0.5, graph:1.0}; unknown class → uniform 1.0. The RRF denominator `RRF_K`=60 (canonical Cormack 2009 default). Documented tunable constants (NFR-MAINT-2). |
| D17 | RRF placement & unit grouping | Unit provenance flows merge → rerank via the `unit` tag: `cross_encoder_rerank` **loops units**, scoring `(sub_query_text, chunk)` into per-unit Top-K (still tagged). `assess_confidence` runs **before** synthesis, so its composite signal **pools the rerank scores across units** (it cannot depend on the final fused order). The decomposed-query aggregation of FR-Q4.2 — RRF across the per-unit rank lists — is therefore performed at **`synthesize_answer`** (before LLM context assembly), **reusing the same RRF helper** as D16's merge. It is a real no-op for simple single-unit queries. |

### 1.5 Why Hybrid (Local vs Cloud) — Rationale
The system is **deliberately hybrid**: the expensive generative/embedding models run in the **cloud (OpenAI)**, while the document-parsing and re-ranking models run **locally on the GPU**. This is an explicit design decision, not an incidental one. Every component's execution location and the reason for it are stated below — nothing is left implied.

**Explicit component placement:**
| Component | Runs where | Runtime | Explicit reason for this placement |
|-----------|-----------|---------|-------------------------------------|
| iiRDS tagging LLM (`gpt-4o-mini`) | **Cloud** (OpenAI) | OpenAI API | Strong general reasoning needed for metadata extraction; running a comparable LLM locally would require large weights/VRAM and degrade quality. Low call volume (1/doc). |
| Query-reasoning LLM calls (ambiguity, refine, complexity, decompose, classify) | **Cloud** (OpenAI) | OpenAI API | Same reasoning-quality argument; per-query cost is bounded and acceptable (NFR-COST-2). |
| Answer synthesis LLM | **Cloud** (OpenAI) | OpenAI API | Final answer quality depends on a capable LLM; model configurable via `SYNTHESIS_MODEL`, default `gpt-4o-mini` (D8). |
| Embeddings (`text-embedding-3-small`) | **Cloud** (OpenAI) | OpenAI API | Managed, high-quality 1536-dim embeddings; identical model reused for ingestion and query (FR-Q0.5). |
| **Document parsing (Docling: DocLayNet + TableFormer)** | **Local (GPU)** | PyTorch + `transformers` | These are neural models Docling executes on-device. The ingestion diagram mandates *"no API calls."* Keeps raw document bytes off the network. |
| **OCR (EasyOCR)** | **Local (GPU)** | PyTorch (shared CUDA build) | Scanned-PDF text recognition; runs on-device so OCR image data never egresses, and avoids per-page cloud-OCR cost. Reuses the reranker's CUDA torch. |
| **Cross-encoder re-ranking (`bge-reranker-v2-m3`)** | **Local (GPU)** | PyTorch + FlagEmbedding/`sentence-transformers` | The RAG diagram mandates *"runs fully locally."* Runs on every query over ~50 candidates — local execution is free per call, low-latency on the RTX 4090, and keeps retrieved chunk text from egressing. |
| Knowledge graph (Neo4j) | **Local (Docker)** | Neo4j container | Constraint C2. |
| Orchestration / state (LangGraph) | **Local** | Python 3.13.14 | Pipeline control plane. |
| Checkpoint store (Postgres) | **Local (native on-prem)** | Postgres service on the laptop | Durable run/interrupt state; constraint C6 (D15). |

**Explicit reasons the local half is local (not cloud APIs):**
1. **Privacy / data-egress minimization** — Parsing and reranking stay on-device, so raw document bytes (Docling) and the retrieved chunk text under reranking never leave the laptop beyond what OpenAI already receives. See the egress boundary below and NFR-SEC-4.
2. **Cost** — Reranking runs per query over ~50 candidate chunks; a cloud rerank API bills per (query × document), and cloud OCR bills per page. Both are **free per call** locally.
3. **Latency / no rate limits** — The 278M-param reranker scores 50 pairs in tens of milliseconds on the RTX 4090, faster than a network round-trip and with no throttling.
4. **Retrieval quality at no extra bill** — A cross-encoder is more accurate than the bi-encoder cosine used for retrieval; local execution captures that gain without a premium API.

**Explicit reasons the cloud half is cloud (not local models):**
1. **Reasoning/embedding quality** — `gpt-4o-mini`-class reasoning and managed embeddings exceed what a laptop-hosted open model delivers without large VRAM/quality trade-offs.
2. **Operational simplicity** — No local LLM serving stack to provision, update, or keep within VRAM alongside Docling + the reranker.
3. **Bounded volume** — LLM/embedding calls per document/query are limited, so cloud cost stays acceptable (NFR-COST).

**Explicit data-egress boundary (what crosses to OpenAI):**
- **Leaves the device:** first 3000 chars of document text (tagging), all chunk text (embeddings, ingestion), query text + reasoning/synthesis context (query path).
- **Never leaves the device:** raw document bytes/files, OCR image data, the full retrieved chunk set during reranking, the Neo4j graph, embeddings vectors at rest, and all run state.

**Explicit trade-off accepted:** the local half requires a CUDA PyTorch install + first-run model-weight downloads and ongoing GPU/driver compatibility (mitigated by the pinned CUDA wheel in §2.3, RISK-D, and the startup `torch.cuda.is_available()` check). On a single-user laptop with an idle RTX 4090, this trade is favorable. Avoiding the local runtime entirely would require swapping Docling and the BGE reranker for cloud equivalents — which would contradict both diagrams, raise per-query cost, and widen the egress surface to additional vendors; therefore it is explicitly **rejected**.

### 1.6 Definitions
| Term | Meaning |
|------|---------|
| **iiRDS** | *intelligent information Request and Delivery Standard* — metadata standard used here as controlled vocabulary (lifecycle phase, information type, product, component). |
| **Chunk** | Retrievable text unit with structural/positional metadata + 1536-dim embedding. |
| **Hybrid retrieval** | Parallel BM25 (lexical) + vector (semantic) + graph (relational) retrieval, merged. |
| **Cross-encoder / reranker** | Model scoring a (query, passage) pair jointly with full cross-attention — more accurate than bi-encoder cosine. Here `BAAI/bge-reranker-v2-m3`, local. |
| **RRF** | Reciprocal Rank Fusion — rank-based merge of multiple result lists (used to aggregate sub-query results). |
| **Checkpointer** | LangGraph persistence (**Postgres**, D15) holding full state per `thread_id`; enables resume after human interrupts. |

---

## 2. Architecture & Technology Stack

### 2.1 Deployment Topology
> **Setup guides:** Neo4j (Docker) — **`NEO4J_SETUP.md`**; Postgres (native on-prem) — **`POSTGRES_SETUP.md`**.
```
┌──────────────────────────── Laptop (Windows 11, Python 3.13) ────────────────────────────┐
│                                                                                           │
│   ┌────────────────────────────────────┐  Bolt :7687  ┌──────────────────────────────┐   │
│   │  Python 3.13 process                │◄────────────►│  Neo4j (Docker container)    │   │
│   │  • LangGraph ingestion + query      │              │  graph + vector + full-text  │   │
│   │  • Docling / EasyOCR (local GPU)    │              │  index, volume-backed        │   │
│   │  • BGE reranker (local GPU)         │              └──────────────────────────────┘   │
│   │                                     │  psql :5432  ┌──────────────────────────────┐   │
│   │  • LangGraph checkpointer           │◄────────────►│  Postgres (native on-prem)   │   │
│   │                                     │              │  LangGraph checkpoints,      │   │
│   └───────────────────┬─────────────────┘              │  local data dir              │   │
│                       │ HTTPS (egress)                 └──────────────────────────────┘   │
└───────────────────────┼────────────────────────────────────────────────────────────────────┘
                        ▼
        OpenAI API  (gpt-4o-mini — tagging + query reasoning + synthesis;
                     text-embedding-3-small — chunk & query embeddings)
```

### 2.2 Component → Technology Mapping
| Concern | Technology |
|---------|------------|
| Orchestration / state / resume | LangGraph + **Postgres checkpointer** (`langgraph-checkpoint-postgres`, via `psycopg`) |
| Document parsing | Docling (DocLayNet layout, TableFormer tables) |
| **OCR (local, GPU)** | **EasyOCR** via `docling[easyocr]` (PyTorch-based, CUDA) |
| Tokenization / sizing | tiktoken |
| LLM (tagging, query reasoning, synthesis) | OpenAI `gpt-4o-mini` via `openai` SDK |
| Embeddings (chunks **and** queries) | OpenAI `text-embedding-3-small` (1536-dim, cosine) |
| **Cross-encoder reranker (local, GPU)** | `BAAI/bge-reranker-v2-m3` via **FlagEmbedding** (`FlagReranker(..., use_fp16=True, devices=["cuda:0"])`) or `sentence-transformers` CrossEncoder (`device="cuda"`) |
| Local ML runtime | PyTorch **CUDA build** (cp313) + Hugging Face `transformers`, running on the NVIDIA RTX 4090 |
| Retry / backoff | tenacity |
| Validation / typed state | pydantic v2 |
| Config / secrets | python-dotenv (`.env`) |
| Graph DB / driver | Neo4j (Docker) + official `neo4j` driver (Bolt) |
| Vector/index helpers (optional) | `neo4j-graphrag` |
| Rank fusion math | numpy (RRF) |
| Logging | Python `logging` — structured JSON, console + per-run file (NFR-LOG); separate from the JSON receipt |

### 2.3 Pinned Versions (verified compatible, June 2026)
> **Requirement:** Use the latest patch within each minor line to obtain current security fixes (stakeholder requirement). Pin exact versions in a lockfile and run a vulnerability scan (NFR-SEC-3) before each build. On Python 3.13 the full ML stack (incl. **CUDA GPU wheels**) is available, so none of the 3.14 caveats from v1.0 apply.

| Package | Minimum version | Notes (Python 3.13) |
|---------|-----------------|---------------------|
| `python` | **3.13.14** (pinned) | Confirmed decision D2. Mature, fully supported by the entire stack. |
| `torch` | latest 2.x **CUDA build** (cp313) | **Install from the PyTorch CUDA wheel index**, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu128` (any recent `cuXXX` ≤ the driver's CUDA 13.2 works). RTX 4090 = Ada / sm_89, supported by all current CUDA builds. **Do not** install the default CPU-only wheel. |
| `transformers` (HF) | latest | Reranker + Docling model backbone. |
| `docling` | latest 2.x | Full layout/table/OCR on 3.13; GPU usable. |
| `docling[easyocr]` / `easyocr` | latest (via extra) | **Locked OCR engine (D6).** PyTorch-based, runs on the shared CUDA build; first-run weight download (~100 MB). No system binaries. |
| `FlagEmbedding` | latest (or `sentence-transformers` latest) | Loads `bge-reranker-v2-m3`; `use_fp16=True` for speed. |
| `langgraph` | **1.2+** | Full 3.13 support. |
| `langgraph-checkpoint-postgres` | latest compatible w/ LangGraph 1.2 | **Postgres checkpoint backend (D15).** Provides `PostgresSaver` (+ `AsyncPostgresSaver`); `.setup()` creates tables. |
| `psycopg[binary,pool]` | **3.2+** | Postgres driver + connection pool used by the saver. |
| `langchain-core` | **1.4.7+** | Transitive; keep aligned with LangGraph 1.x. |
| `openai` | **2.41.1+** | Supports 3.9–3.14. |
| `tiktoken` | latest | 3.13 wheels available. |
| `neo4j` (driver) | **6.2+** | Match Neo4j server major series. |
| `neo4j-graphrag` | **0.6.2+** (optional) | Vector/index helpers. |
| `pydantic` | **2.13+** | v2 only. |
| `tenacity` | **9.1.4+** | Retry/backoff. |
| `python-dotenv` | **1.2.2+** | Config/secrets. |
| `numpy` | latest 2.x | RRF / score math. |
| Neo4j server (Docker) | **2026.x** matching driver 6.x | Native vector + full-text (Lucene/BM25) index. |
| Postgres server (native, on-prem) | **14+** (17/18 recommended) | Installed natively on the laptop (D15). LangGraph checkpoint store only (no pgvector needed — vectors live in Neo4j). |

> **Install order:** install the ML stack first (`torch` → `transformers`/`docling`/`FlagEmbedding`) so the heavy pins resolve, then the lighter libraries. Use a lockfile (`uv` or compiled `requirements.lock`).

### 2.4 Environment Setup & Model Warm-up (shared bootstrap)
Both pipelines depend on a **one-time bootstrap** that prepares and verifies the environment before any ingest or query run (decision D7). It is the documented, required path; lazy first-run auto-download is retained only as a fallback.

- **FR-S0.1** A single bootstrap command (e.g. `python setup.py` / `setup_models.py`) SHALL prepare and verify the runtime, and SHALL be runnable independently of the ingest/query CLIs.
- **FR-S0.2** It SHALL assert `torch.cuda.is_available() == True` and log `torch.cuda.get_device_name(0)`; if only a CPU torch build is present it SHALL fail with an actionable message (directs the user to the CUDA wheel index in §2.3). Mitigates RISK-D.
- **FR-S0.3** It SHALL download and cache all local model weights — Docling (DocLayNet, TableFormer), BGE reranker (`bge-reranker-v2-m3`), and EasyOCR (detection + recognition) — into a configured cache directory (e.g. `HF_HOME`/cache path via env), so subsequent runs require no download.
- **FR-S0.4** It SHALL run a minimal smoke inference on each local model to confirm it loads and executes on the GPU (or the configured device).
- **FR-S0.5** It SHALL verify Neo4j connectivity and **idempotently create the required constraints/indexes** (unique `Document.id`, vector index, full-text index) — this is the canonical execution of FR-7.10.
- **FR-S0.5a** It SHALL verify **Postgres** connectivity and **idempotently create the LangGraph checkpointer tables** (the saver's `.setup()`), so interrupts/resume work on first run (D15).
- **FR-S0.6** It SHALL print a clear **"environment ready"** summary, or **"environment NOT ready" + the specific reason**, and exit non-zero on any failure.
- **FR-S0.7** The ingest/query pipelines SHOULD detect missing model weights, Neo4j indexes, or Postgres checkpointer tables at start and emit a clear warning pointing to the bootstrap command (graceful behavior if bootstrap was skipped).

### 2.5 Configuration Schema (env / `.env`)
All secrets and tunables are supplied via environment variables (loaded from `.env` by python-dotenv); a `.env.example` SHALL ship with every key below. **No magic numbers inline** (NFR-MAINT-2; decisions D8/D11/D12/D13). A single typed config object (pydantic) SHALL load and validate these and be the one source consumed by both pipelines (NFR-MAINT-7, NFR-REL-8).

**Secrets / connections**
| Var | Default | Notes |
|-----|---------|-------|
| `OPENAI_API_KEY` | — *(required)* | OpenAI auth; never committed (NFR-SEC-1) |
| `OPENAI_BASE_URL` | *(unset)* | optional endpoint override |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | localhost only (NFR-SEC-5) |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | — *(required)* | changed from default (NFR-SEC-5) |
| `NEO4J_DATABASE` | `neo4j` | |
| `CHECKPOINT_DB_URI` | `postgresql://langgraph:langgraph@127.0.0.1:5432/langgraph` | LangGraph Postgres checkpointer (D15); localhost only, password not committed (NFR-SEC-1/5) |

**Models**
| Var | Default | Notes |
|-----|---------|-------|
| `LLM_MODEL` | `gpt-4o-mini` | tagging + query reasoning |
| `SYNTHESIS_MODEL` | `gpt-4o-mini` | answer synthesis; may be `gpt-4o` (D8) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | shared ingest+query (FR-Q0.5 / NFR-REL-8) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | local cross-encoder |
| `RERANKER_DEVICE` | `cuda` | fallback `cpu` (FR-Q3.2a) |
| `OCR_ENGINE` | `easyocr` | locked (D6) |
| `MODEL_CACHE_DIR` / `HF_HOME` | *(platform default)* | local weight cache (FR-S0.3) |

**Parsing / Docling** (PDF memory & cost controls; defaults preserve prior behavior)
| Var | Default | Source |
|-----|---------|--------|
| `OCR_ENABLED` | `auto` | FR-2.3e/2.3f — `auto` detects scanned/mixed PDFs per file; `on`/`true` forces OCR; `off`/`false` skips it |
| `OCR_LANGUAGES` | `fr,de,es,en` | FR-2.3g — EasyOCR language codes (comma/space-separated) used when OCR runs; must be script-compatible |
| `PDF_MAX_PAGES` | `0` | FR-2.8 — `0` = all pages; N>0 parses only the first N (PDF only) |
| `PDF_RENDER_DPI` | `72` | FR-2.8b — born-digital page rasterization DPI; `images_scale = DPI/72`; lower = less memory |
| `PDF_RENDER_DPI_OCR` | `150` | FR-2.8d — DPI used when OCR is active (the scanned bitmap is the only text source); higher = sharper OCR, more memory/time |
| `PDF_PARSE_BATCH_PAGES` | `100` | FR-2.8c — slice size for born-digital PDFs (no OCR); `0` = single convert |
| `PDF_PARSE_BATCH_PAGES_OCR` | `12` | FR-2.8c/2.3e — smaller slice size used when OCR is active (heavier per page + sharper FR-2.8d DPI) |

**Tunables** (defaults from the diagrams / decisions)
| Var | Default | Source |
|-----|---------|--------|
| `CHUNK_MAX_TOKENS` | `512` | FR-3.4 |
| `CHUNK_OVERLAP_TOKENS` | `50` | FR-3.4 |
| `EMBED_BATCH_SIZE` | `100` | FR-6.2 |
| `EMBED_MAX_INPUT_TOKENS` | `8192` | FR-3.8 — embedding model's hard per-input token limit; oversized whole-block chunks are hard-split to stay under it |
| `NEO4J_WRITE_BATCH_CHUNKS` | `250` | FR-7.1 — chunks per write transaction (bounds tx size; failure compensated by FR-7.8) |
| `TAG_CONFIDENCE_THRESHOLD` | `0.5` | FR-4.7 |
| `REFINE_CONFIDENCE_THRESHOLD` | `0.6` | FR-Q1.4 |
| `ESCALATE_CONFIDENCE_THRESHOLD` | `0.38` | FR-Q3.5 (calibrated v2.9) |
| `PER_RETRIEVER_K` | `25` | D11 / FR-Q2.6 |
| `RETRIEVE_TOP_K` | `50` | D11 / FR-Q2.6 |
| `RERANK_TOP_K` | `10` | D11 / FR-Q3.3 |
| `MAX_CLARIFICATION_ROUNDS` | `2` | D12 / FR-Q1.6 |
| `MAX_SUBQUERIES` | `5` | D12 / FR-Q1.10 |

**Logging** (NFR-LOG)
| Var | Default | Notes |
|-----|---------|-------|
| `LOG_LEVEL` | `INFO` | app log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`); NFR-LOG-4 |
| `LOG_FORMAT` | `json` | `json` or `text` (NFR-LOG-2) |
| `LOG_DIR` | `./logs` | per-run log file location, git-ignored (NFR-LOG-3) |

---

## 3. Functional Requirements — Ingestion (Write Path)

### 3.0 Orchestration (cross-cutting)
- **FR-0.1** Implemented as a LangGraph `StateGraph`: nodes `intake, parse, chunk, tag_iirds, human_review, embed, neo4j_write, receipt` plus terminal error/duplicate nodes.
- **FR-0.2** State is a typed (pydantic) object: `file_path, thread_id, doc_hash, raw_bytes|temp_path, docling_document, chunks[], iirds_tags, confidence, human_reviewed, embeddings_meta, neo4j_doc_id, neo4j_chunk_ids[], stage_timings_ms, pipeline_status, error`.
- **FR-0.3** Each run generates a unique `thread_id`, persisted via the Postgres checkpointer (D15) for tracking/resume.
- **FR-0.4** Entry point CLI: `python ingest_document.py <path>`.
- **FR-0.5** Each stage records wall-clock duration into `stage_timings_ms`.
- **FR-0.6** Conditional routing driven by typed status values.
- **FR-0.7** Any hard-terminal branch exits non-zero with an actionable message; **no partial data** remains in Neo4j.

### 3.1 Stage 1 — Intake
- **FR-1.1** Accept a single file path; verify it exists and is readable.
- **FR-1.2** Validate format ∈ {PDF, DOCX, HTML, XML, TXT, MD}; unsupported → terminate with message.
- **FR-1.2a** A **password-protected (encrypted) PDF** SHALL be detected at intake (pypdfium2 `FPDF_ERR_PASSWORD`, with a message-text fallback) and terminate with a clear, actionable message ("password-protected PDF is not supported; remove the password and re-ingest") rather than a generic downstream parse error. Other open failures are left for the parse stage (FR-2.6).
- **FR-1.3** Compute **SHA-256 of raw bytes** = canonical `Document` id.
- **FR-1.4** Query Neo4j for an existing `Document` with that hash.
- **FR-1.5** Duplicate → terminate with status **`DUPLICATE`**; **re-ingestion not supported** (remediation: ingest the updated source as a new file).
- **FR-1.6** Valid/non-duplicate → store hash in state, pass raw bytes forward.

### 3.2 Stage 2 — Docling Parse
- **FR-2.1** Parse **fully locally** (no network) via Docling.
- **FR-2.2** Use **DocLayNet** (layout) + **TableFormer** (tables).
- **FR-2.3** **OCR applied to scanned/image PDFs** (auto-detected by default, FR-2.3e), using the **EasyOCR** engine (D6).
  - **FR-2.3a** EasyOCR SHALL be installed via the `docling[easyocr]` extra (no system binaries).
  - **FR-2.3b** EasyOCR SHALL run on the **GPU (CUDA)** by default, reusing the shared CUDA PyTorch build; it MUST fall back to CPU if no CUDA device is available (consistent with FR-Q3.2a).
  - **FR-2.3c** EasyOCR model weights (detection + recognition, ~100 MB) are downloaded on first run and cached locally; subsequent runs require no re-download.
  - **FR-2.3d** The OCR engine SHALL be selected explicitly in Docling pipeline options (EasyOCR), not left to Docling's auto/default selection.
  - **FR-2.3e** OCR SHALL be controlled by the tri-state `OCR_ENABLED` (default **`auto`**; normalized to `auto` | `on`/`true` | `off`/`false`; resolves to Docling `PdfPipelineOptions.do_ocr`). `on` forces OCR for every file; `off` disables it (the primary mitigation for large born-digital PDFs that exhaust memory during parse, Docling `std::bad_alloc`); **`auto`** decides per file via FR-2.3f. The decision is made in the `parse` node per document — no per-file env tuning. Non-PDF inputs never OCR.
  - **FR-2.3f** In `auto` mode each PDF SHALL be classified from its **own text layer** by a shared classifier (`rag/ingestion/pdf_kind.py`), and OCR enabled only for `scanned`/`mixed` results. The classifier samples up to 1000 pages spread across the document and, for pages with no real text layer, measures the largest image's page coverage; a text-less page counts as **scanned** only when an image covers ≥ 80% of it (a full-page scan), so blank pages and partial figures do NOT trigger OCR. The digital/scanned/mixed label is a ratio over *text-vs-scanned* pages. On any probe failure it defaults to `digital` (the fast path) — a wrong guess yields too few chunks (visible, re-ingestable), versus a needless ~30-min OCR run. The same classifier backs the standalone `detect_pdf_kind.py` CLI.
  - **FR-2.3g** OCR languages SHALL be configurable via `OCR_LANGUAGES` (EasyOCR codes, comma/space-separated; default `fr,de,es,en` = EasyOCR's built-in Latin set), passed to `EasyOcrOptions(lang=…)`. This enables non-Latin scans (e.g. `ch_sim,en`, `ja,en`, `ar,en`). The set MUST be script-compatible (English combines with anything; mixing incompatible scripts raises at model load) — this constraint is documented, not validated in code.
- **FR-2.4** Return a `DoclingDocument` tree with typed elements: headings, paragraphs, tables, lists, warnings. **The tree is handed from `parse` to `chunk` via a process-local handoff, NOT through the checkpointed state (v2.12):** it is heavy and not msgpack-serializable, so persisting it crashes the Postgres checkpointer (`TypeError: Type is not msgpack serializable: DoclingDocument`). `parse`→`chunk` always run back-to-back in one process (the only interrupt is human_review, after chunk), so the handoff is safe.
- **FR-2.5** Write raw bytes to a temp file for parsing; **delete temp file immediately after parse** (success or failure).
- **FR-2.6** Parse failure (corrupt, unsupported encoding, zero content) → **parse-error terminal**, logging the Docling error.
- **FR-2.6a** **Partial parse visibility (v2.12).** Docling may return `PARTIAL_SUCCESS` — some pages fail (e.g. memory pressure on very large PDFs) while others parse. Since there IS extractable text, the run proceeds, but the count of failed pages SHALL be recorded (`parse_failed_pages`), logged as a WARNING, and surfaced in the completion summary (`⚠ INCOMPLETE: N/total page(s) failed`). A partial parse MUST NOT be reported as a clean ingest, since the failed pages' content is absent from the graph.
- **FR-2.7** Success → store tree, `title`, `page_count`. **Title resolution (v2.10):** an on-page Docling **TITLE element** → the embedded **PDF `/Title` metadata** (read locally via pypdfium2, a Docling dependency — no egress) → the **original file stem**. Docling's `doc.name` is deliberately *not* used: because parsing runs from the FR-2.5 private temp file, `doc.name` is always the `rag_ingest_*` temp stem and would produce meaningless titles.
- **FR-2.8** **Parse resource controls (PDF only).** Two configurable bounds limit memory/time on very large PDFs without changing default behavior:
  - **FR-2.8a** `PDF_MAX_PAGES` (default `0` = no limit). When `N > 0`, only the first `N` pages are parsed (applied as Docling `convert(..., page_range=(1, N))`); pages beyond the cap are not parsed. Applies to PDF input only.
  - **FR-2.8b** `PDF_RENDER_DPI` (default `72`, Docling's default → `images_scale = DPI/72 = 1.0`, unchanged behavior). This is the **born-digital** render DPI — the page bitmap is only a layout backdrop, so 72 dpi is sufficient. Lower DPI (e.g. `48`) shrinks per-page bitmaps to relieve memory pressure during preprocessing. **NOTE:** empirically this is NOT sufficient for very large PDFs — Docling's `std::bad_alloc` ceiling (~127 pages on a 24 GB box) is hit at a fixed page depth *independent of DPI* (the accumulation is in the parse backend, not the page bitmaps). Use FR-2.8c for those. When OCR is active, FR-2.8d's DPI applies instead.
  - **FR-2.8d** `PDF_RENDER_DPI_OCR` (default `150`). The render DPI used **when the OCR decision is on** (FR-2.3e). Unlike a born-digital page, a **scanned** page's rasterized bitmap is the *only* text source EasyOCR reads, so a low DPI (72) degrades recognition of small glyphs — subscripts, mathematical notation, diacritics. The `parse` node folds this into the effective `PDF_RENDER_DPI` only when `do_ocr` is true (via the same `model_copy` fold that selects the smaller OCR slice size, FR-2.8c/FR-2.3e); the born-digital fast path keeps FR-2.8b's 72 dpi untouched. Cost scales ~dpi², so a higher OCR DPI raises per-slice memory and parse time — if it OOMs (`std::bad_alloc`), lower `PDF_PARSE_BATCH_PAGES_OCR` (FR-2.8c) to compensate. Applies to PDF/image OCR input only. Defaults preserve born-digital behavior; the only effect is sharper scanned-PDF OCR.
  - **FR-2.8c** `PDF_PARSE_BATCH_PAGES` (default `100`; `0` = legacy single convert). **The fix for very large PDFs (v2.13).** A PDF is parsed in page-range slices of this size via repeated `convert(..., page_range=(lo, hi))` calls on the same reused converter, releasing each slice's transient memory before the next, so Docling never accumulates past its OOM ceiling. `chunk` then consumes the ordered list of slice documents with **continuous** section-stack and chunk-position tracking (buffers flush only at section headers and once at the end — never between slices), so chunking is identical to a single-shot parse. Captures the WHOLE document. Verified: a 610-page book that previously dropped ~half its pages to `std::bad_alloc` now parses all 610 pages (`failed_pages=0`) in 7 slices. Applies to PDF input only; interacts with FR-2.8a (the cap bounds the last slice).

### 3.3 Stage 3 — Structural (Layout-Aware) Chunking

Chunking is driven by the document's **structure** (Docling's typed layout items), not blind fixed-size windows: each content type is chunked by its own rule so semantically-coherent units stay intact, while prose is packed to a token budget. Continuity is preserved across parse slices (FR-2.8c).

- **FR-3.1** Tables → exactly one chunk each (structure preserved, never split).
- **FR-3.2** Lists/procedures → one chunk each (step order preserved).
- **FR-3.3** Warnings (safety-critical) → one chunk each, **never fragmented**.
- **FR-3.3a** Code/formula blocks → kept whole (one chunk each, never fragmented), like tables.
- **FR-3.4** Paragraph text → split at **512 tokens** with **50-token sentence-boundary overlap** (tiktoken-counted); an oversized single sentence is hard-split into ≤512-token windows.
- **FR-3.5** Each chunk carries: `parent_section_path, document_title, content_type, position, token_count`.
- **FR-3.6** Section headings → context metadata on chunks (build `section_path`), not standalone chunks.
- **FR-3.7** Store `ChunkList, total_tokens`, per-chunk `section_path`.
- **FR-3.8** Every chunk — including the "kept whole" types (FR-3.1/3.2/3.3/3.3a) — is hard-capped to `EMBED_MAX_INPUT_TOKENS` minus a fixed prefix margin (for the embedding context prefix, FR-6.3). Any oversized whole-block chunk is hard-split into within-limit windows, logged as a warning, so a single large table/formula/list run can never exceed the embedding model's per-input limit and fail the run (relates to FR-6.x).

### 3.4 Stage 4 — iiRDS Tagging
- **FR-4.1** **One `gpt-4o-mini` call per document** (not per chunk), sending first **3000 chars**.
- **FR-4.2** Extract: `product`, `components`, `lifecycle_phase` (Installation/Operation/Service/Repair/Disposal), `information_type` (Procedure/Concept/Warning/Specification/MaintenanceInterval/Troubleshooting), `language`.
  - **FR-4.2a** `lifecycle_phase` and `information_type` SHALL be **closed enumerations** validated against **config-defined** value sets (the lists above), rejecting/normalizing out-of-vocabulary values; the lists are extensible to the full iiRDS vocabulary. `product`/`component` are **open free-text**, `MERGE`d in the graph (D9).
- **FR-4.3** Return `confidence` ∈ 0.0–1.0.
- **FR-4.4** Retry **3× exponential backoff** on API failure (tenacity).
- **FR-4.5** All retries fail → store **empty tags and continue** (non-blocking degradation).
- **FR-4.6** Output schema-validated (pydantic); malformed = retryable.
- **FR-4.7** Routing: `confidence < 0.5` → human review; `≥ 0.5` → bypass to embed.

### 3.5 Human Review (conditional)
- **FR-5.1** LangGraph **`interrupt()`** node suspends the pipeline.
- **FR-5.2** Full state persisted to the Postgres checkpointer on suspend.
- **FR-5.3** Operator shown current tags, prompted to correct.
- **FR-5.4** Resume CLI: `python review_tags.py <thread_id>`.
- **FR-5.5** After correction, set `human_reviewed = True`, continue to embed with corrected tags.
- **FR-5.6** Suspended run survives process exit and resumes from persisted state.

### 3.6 Stage 5 — Embedding
- **FR-6.1** OpenAI **`text-embedding-3-small`**, **1536-dim**.
- **FR-6.2** Batches of **100** chunks.
- **FR-6.3** Embedded text prefixed with `"[Document: <title>] [Section: <path>]"`.
- **FR-6.4** Retry **3× exponential backoff**.
- **FR-6.5** All retries fail → **hard-exit** via embed-error terminal; **no Neo4j write**; file safe to re-submit.
- **FR-6.6** Track `tokens_used` + `estimated_cost_usd`; store in state.
- **FR-6.7** Attach embeddings to chunks.

### 3.7 Stage 6 — Neo4j Write (Atomic via batched write + compensation)
- **FR-7.1** Entire document written **atomically (all-or-nothing)** from the caller's view. A single transaction over *all* chunks can grow large enough to fail the Neo4j store-apply on very large books (observed: a ~243k-command transaction wedged the database), so the write is split into bounded transactions — the `Document` node + iiRDS edges first, then chunks in batches of **`NEO4J_WRITE_BATCH_CHUNKS` (default 250)**, one transaction each. Atomicity is preserved by **compensation** (FR-7.8), not by a single transaction.
- **FR-7.2** Create `Document {id=SHA-256, title, file_path, ingested_at, chunk_count, language}`.
- **FR-7.3** iiRDS nodes (`Product, Component, LifecyclePhase, InformationType`) via **`MERGE`** (cross-doc dedup).
- **FR-7.4** `Chunk {id, text, content_type, section_path, position, token_count, embedding[1536]}`.
- **FR-7.5** Relationships: `RELATES_TO_PRODUCT, RELATES_TO_COMPONENT, HAS_LIFECYCLE_PHASE, HAS_INFORMATION_TYPE, HAS_CHUNK`.
- **FR-7.6** Full-text (Lucene/BM25) index updated automatically on chunk text.
- **FR-7.7** Vector index populated with 1536-dim embeddings (**cosine**).
- **FR-7.8** Write failure → **compensating delete** of the document + any chunk batches already committed (no partial doc) → write-error terminal; safe to retry. The cleanup is itself batched (`CALL { … } IN TRANSACTIONS`) so it cannot become an oversized transaction. If the compensating delete *also* fails (e.g. DB down), it is logged and a re-ingest reports the document as a **DUPLICATE** until it is cleaned up manually.
- **FR-7.9** Commit → store `neo4j_doc_id, neo4j_chunk_ids[]`.
- **FR-7.10** Required constraints/indexes (unique `Document.id`, vector index, full-text index) created idempotently if absent — performed canonically by the bootstrap (FR-S0.5), and re-checked at write time as a safety net.

### 3.8 Stage 7 — Receipt
- **FR-8.1** Dual write: **(a)** write a **per-run** JSON file `ingestion_logs/ingestion_log_<UTCstamp>_<thread8>_<source-stem>.json` containing one record (`doc_hash, file_name, doc_title, ingested_at, chunk_count, total_tokens, embedding_cost_usd, stage_timings_ms, pipeline_status, low_confidence_flag`); **(b)** create `(:IngestionRecord)-[:INGESTION_OF]->(:Document)`. Each run produces its own file (no shared/appended log), so runs never overwrite each other.
- **FR-8.2** If either receipt write fails → **warning only**, still exit success (document already durable).

### 3.9 Completion
- **FR-9.1** Print: title, chunk count, embedding cost, total wall-clock, `thread_id`.
- **FR-9.2** Exit `0` on success; non-zero on terminal error, with the reason distinguishable.

### 3.10 Graph Data Model
```
(:Document {id,title,file_path,ingested_at,chunk_count,language})
   -[:HAS_CHUNK]->            (:Chunk {id,text,content_type,section_path,position,token_count,embedding[1536]})
   -[:RELATES_TO_PRODUCT]->   (:Product {name})
   -[:RELATES_TO_COMPONENT]-> (:Component {name})
   -[:HAS_LIFECYCLE_PHASE]->  (:LifecyclePhase {name})
   -[:HAS_INFORMATION_TYPE]-> (:InformationType {name})
(:IngestionRecord {ts,status,chunk_count,total_tokens,cost_usd,timings}) -[:INGESTION_OF]-> (:Document)
```

---

## 4. Functional Requirements — Query (Read Path)

### 4.0a Graph Topology (overview)

The query `StateGraph` (FR-Q0.1) wires the four layers below. Bold edges are the
parallel retriever fan-out (D10): all three retrievers always run; `query_class`
weights the merge, never gates which run. The two double-bordered nodes
(`request_clarification`, `escalate`) are checkpointed `interrupt_before` nodes —
their bodies are bypassed on resume; the dotted edges are the `thread_id` resume
path (`update_state` → `invoke(None)`, D14). Weighted Reciprocal Rank Fusion is
applied twice by design (D16 at `merge`, D17 at `synthesize_answer`) via one
shared helper.

```mermaid
flowchart TD
    START([START]) --> DA

    subgraph PRE["§4.1 Pre-retrieval — LLM_MODEL"]
        DA[detect_ambiguity<br/>FR-Q1.1/1.2]
        RQ[refine_query<br/>FR-Q1.3/1.4]
        RC{{request_clarification<br/>FR-Q1.5 interrupt_before}}
        DC[detect_complexity<br/>FR-Q1.7/1.8]
        DQ[decompose_query<br/>FR-Q1.9/1.10 · ≤5 D12]
        PR[prepare_retrieval<br/>FR-Q1.11 → retrieval_units]
    end

    DA -->|is_ambiguous| RQ
    DA -->|clear| DC
    RQ -->|refine_conf < 0.6<br/>AND rounds < 2| RC
    RQ -->|ok / cap hit best-effort| DC
    RC -. "resume: update_state<br/>clarification_rounds+1<br/>(body bypassed)" .-> DA
    DC -->|complex| DQ
    DC -->|simple| PR
    DQ --> PR

    PR --> CQ

    subgraph RET["§4.2 Retrieval — Neo4j (local, no API)"]
        CQ[classify_query<br/>FR-Q2.1 → single query_class]
        BM[bm25_retrieve<br/>FR-Q2.3 chunk_fulltext]
        VE[vector_retrieve<br/>FR-Q2.4 chunk_embedding cosine]
        GR[graph_traverse<br/>FR-Q2.5 iiRDS rels]
        MG[merge<br/>FR-Q2.2/2.6 · weighted RRF D16<br/>dedup chunk_id · top-50]
    end

    CQ ==>|fan-out D10| BM
    CQ ==>|fan-out D10| VE
    CQ ==>|fan-out D10| GR
    BM --> MG
    VE --> MG
    GR --> MG

    subgraph RR["§4.3 Re-rank + confidence"]
        CE[cross_encoder_rerank<br/>FR-Q3.1/3.3 · BGE GPU · top-10]
        AC[assess_confidence<br/>FR-Q3.4 composite signal]
        ES{{escalate<br/>FR-Q3.6 interrupt_before}}
    end

    MG --> CE --> AC
    AC -->|conf < 0.4| ES
    AC -->|conf ≥ 0.4| SY
    ES -. "resume: update_state<br/>escalated=True<br/>(body bypassed)" .-> SY

    subgraph SYN["§4.4 Synthesis — SYNTHESIS_MODEL"]
        SY[synthesize_answer<br/>FR-Q4.1/4.2/4.3<br/>RRF across sub-queries D17 + citations]
    end

    SY --> END([END])

    classDef interrupt fill:#fde,stroke:#c39,stroke-width:2px;
    class RC,ES interrupt;
```

### 4.0 Orchestration (cross-cutting)
- **FR-Q0.1** Implemented as a LangGraph `StateGraph`: nodes `detect_ambiguity, refine_query, request_clarification, detect_complexity, decompose_query, prepare_retrieval, classify_query, bm25_retrieve, vector_retrieve, graph_traverse, cross_encoder_rerank, assess_confidence, escalate, synthesize_answer` + END.
- **FR-Q0.2** Entry point CLI (e.g. `python query.py "<question>"`); the raw query is stored in state as **`original_query`**.
- **FR-Q0.3** Query state (typed/pydantic) carries at minimum: `original_query, thread_id, refined_query, is_ambiguous, refine_confidence, clarification_question, human_clarification, clarification_rounds, is_complex, sub_queries[], retrieval_units[], query_class, bm25_hits[], vector_hits[], graph_hits[], merged_candidates[], reranked[], confidence_signal, escalated, low_confidence_flag, answer, citations[], stage_timings_ms`. `clarification_question` is produced by `refine_query` when confidence is low and surfaced by the resume-CLI (FR-Q1.5).
- **FR-Q0.4** Each run has a unique `thread_id`; human-in-the-loop nodes persist full state to the Postgres checkpointer and are resumable.
- **FR-Q0.5** The query **embedding model and similarity metric MUST match ingestion** exactly: `text-embedding-3-small`, 1536-dim, cosine (FR-6.1/FR-7.7). Mismatch is a correctness defect.
- **FR-Q0.6** The LLM-reasoning nodes (ambiguity, refine, complexity, decompose, classify) use `LLM_MODEL` (default `gpt-4o-mini`, D1); `synthesize_answer` uses `SYNTHESIS_MODEL` (default `gpt-4o-mini`, D8). All use tenacity 3× backoff; transient failures retry before surfacing an error.
- **FR-Q0.7** Final answer SHALL include source attribution (originating `Document`/`Chunk` ids) for the chunks used.
- **FR-Q0.8** `stage_timings_ms` is written concurrently by the three parallel retrievers in a single LangGraph super-step and therefore carries an **additive dict-merge reducer** (LangGraph channel reducer) so the same-step writes accumulate instead of raising a conflict; the three hit-list channels (`bm25_hits`/`vector_hits`/`graph_hits`) are distinct keys and need no reducer (D16).

### 4.1 Pre-Retrieval Layer

**detect_ambiguity**
- **FR-Q1.1** Determine whether the query lacks specificity: missing model/part numbers, ambiguous pronouns, or multiple valid interpretations.
- **FR-Q1.2** Produce an `is_ambiguous` flag. Route: **ambiguous → `refine_query`**; **clear → `detect_complexity`**.

**refine_query**
- **FR-Q1.3** Apply refinement: expand abbreviations, add implicit domain context, resolve pronouns, add scope qualifiers.
- **FR-Q1.4** Produce a `refine_confidence` score. If **`refine_confidence < 0.6`** → route to **`request_clarification`**; otherwise route the refined query to **`detect_complexity`**.

**request_clarification (Human-in-the-loop)**
- **FR-Q1.5** A LangGraph `interrupt()` node triggered when `refine_confidence < 0.6`; surfaces a clarification question, persists full state, and is resumable via the shared resume-CLI by `thread_id` (D14).
- **FR-Q1.6** The user's response **loops back to `detect_ambiguity`** for re-evaluation. The clarification loop SHALL be capped at **`MAX_CLARIFICATION_ROUNDS` (default 2)**; on exceeding the cap the pipeline SHALL proceed best-effort with the most-refined query and set a low-confidence flag (D12, mitigates RISK-F).

**detect_complexity**
- **FR-Q1.7** Determine whether the query spans multiple concepts, documents, or reasoning steps.
- **FR-Q1.8** Route: **complex → `decompose_query`**; **simple → `prepare_retrieval`**.

**decompose_query**
- **FR-Q1.9** Split a complex query into **atomic sub-queries**, each addressing one concept.
- **FR-Q1.10** Support sequential, parallel, and hierarchical aggregation strategies (selected per query). Decomposition SHALL be **single-level** and produce **≤ `MAX_SUBQUERIES` (default 5)** sub-queries, to bound cost/latency (D12).

**prepare_retrieval**
- **FR-Q1.11** Normalize both the simple (single query) and decomposed (sub-query list) paths into a **uniform list structure** so downstream retrieval nodes handle both identically.

### 4.2 Retrieval Layer

**classify_query**
- **FR-Q2.1** Classify each retrieval unit as one of: `exact_lookup`, `conceptual`, `procedural`, `relational`.
- **FR-Q2.2** **All three retrievers SHALL run in parallel** for every retrieval unit; `query_class` is used to **weight** their contributions at merge time (e.g. boost BM25 for `exact_lookup`, graph for `relational`, vector for `conceptual`/`procedural`), and **SHALL NOT** be used to exclude any retriever (D10). Each retriever **loops over the `retrieval_units`** and tags every hit with its originating unit + per-unit rank; the merge fusion mechanism is fixed by **D16** (weighted RRF).

**bm25_retrieve**
- **FR-Q2.3** BM25 full-text search via the Neo4j full-text (Apache Lucene) index over chunk text. Best for exact technical terms, part numbers, model codes, specification values.

**vector_retrieve**
- **FR-Q2.4** Vector similarity via the Neo4j vector index using **cosine** similarity over the query embedding. Best for natural-language queries, synonyms, paraphrases.

**graph_traverse**
- **FR-Q2.5** Neo4j Cypher traversal over **iiRDS-mapped relationships** (`HAS_CHUNK`, `RELATES_TO_PRODUCT`, `RELATES_TO_COMPONENT`, `HAS_LIFECYCLE_PHASE`, `HAS_INFORMATION_TYPE`). Best for relational queries — components, procedures, warnings, product-lifecycle connections.

**merge**
- **FR-Q2.6** Each retriever returns up to **`PER_RETRIEVER_K` (default 25)** hits; results are merged and **deduplicated by chunk id** into a candidate set capped at **`RETRIEVE_TOP_K` (default 50)** passed to re-ranking. Values configurable (D11). The merge fusion is **weighted Reciprocal Rank Fusion** weighted by `query_class` (D16): `score = Σ weight[query_class][retriever] / (k + rank)`, with `k = RRF_K = 60`; the per-`query_class`×retriever weight matrix is the documented constant `RETRIEVER_WEIGHTS` fixed in v2.7 (see D16; NFR-MAINT-2).

### 4.3 Re-Ranking Layer

**cross_encoder_rerank**
- **FR-Q3.1** Re-rank the Top-50 candidates with the **BGE reranker (`BAAI/bge-reranker-v2-m3`)**, scoring each (query, chunk) pair jointly with full cross-attention; outputs a relevance score in **[0,1]** per pair (sigmoid-mapped).
- **FR-Q3.2** The reranker SHALL run **fully locally** (no API) and **SHALL execute on the NVIDIA GPU (`cuda`) by default**, loaded with `use_fp16=True` for throughput.
- **FR-Q3.2a** The compute device SHALL be configurable (`RERANKER_DEVICE`, default `cuda`); if no CUDA device is available at runtime the system SHALL **log a warning and fall back to CPU** rather than fail.
- **FR-Q3.2b** The reranker model SHALL be loaded **once at process start** (not per query) and kept resident on the GPU (NFR-PERF-4).
- **FR-Q3.3** Output the **Top-`RERANK_TOP_K` (default 10)** re-ranked chunks with scores into state (configurable, D11).

**assess_confidence**
- **FR-Q3.4** Derive a composite confidence signal from cross-encoder scores using: top-score magnitude, average of top-3, and the score gap between rank-1 and rank-2. For decomposed queries the signal **pools the rerank scores across units** (this node runs before the synthesis-time RRF, so it cannot depend on the final fused order — D17). The composite **weighting (FIXED v2.7, `rag/query/nodes.py::CONFIDENCE_WEIGHTS`):** `composite = 0.5·top + 0.3·avg_top3 + 0.2·gap`, where `top` is the max rerank score, `avg_top3` the mean of the top three, and `gap` the rank1−rank2 difference — each term ∈ [0,1] (rerank scores are sigmoid-normalized), so `composite` ∈ [0,1], comparable to `ESCALATE_CONFIDENCE_THRESHOLD` (0.38, calibrated v2.9). Documented constant (NFR-MAINT-2).
- **FR-Q3.5** Route: **composite confidence `< 0.38` → `escalate`**; **`≥ 0.38` → `synthesize_answer`**. (Default calibrated 2026-06-16 against the real corpus via `calibrate_confidence.py` — answerable/escalate buckets separated cleanly [0.12 vs 0.64], 0.38 is the gap midpoint; v2.9.)

**escalate (Human-in-the-loop)**
- **FR-Q3.6** LangGraph escalation node triggered when composite confidence `< 0.4`; surfaces the low-confidence result for expert review **before** any answer is returned. State persisted; resumable via the shared resume-CLI by `thread_id` (D14).

### 4.4 Synthesis Layer

**synthesize_answer**
- **FR-Q4.1** Generate the answer with the configured synthesis model (`SYNTHESIS_MODEL`, default `gpt-4o-mini`; may be `gpt-4o` — D8) over the **Top-`RERANK_TOP_K` re-ranked chunks** as context.
- **FR-Q4.2** For decomposed queries, aggregate sub-query results via **Reciprocal Rank Fusion (RRF)** before assembling the LLM context window. This RRF runs **at `synthesize_answer`**, across the per-unit rank lists, **reusing the same RRF helper** as the merge fusion (D17); it is a no-op for simple single-unit queries.
- **FR-Q4.3** Answer SHALL be grounded in retrieved chunks and include citations (FR-Q0.7); avoid unsupported claims. **Returned citations are trimmed to the passage markers the answer actually references inline (v2.11):** all Top-`RERANK_TOP_K` chunks still feed the LLM context for grounding, but a focused answer no longer reports weakly-related chunks as "sources". Markers are preserved as-is (not renumbered) so they line up with the inline `[n]` references. If the model emits no markers (e.g. an explicit no-answer), the full context set is kept so a substantive answer is never returned citation-less.

**END**
- **FR-Q4.4** Return the synthesized answer (with citations and the `thread_id`) to the user.

### 4.5 Query Routing Summary (thresholds)
| Decision point | Condition | Route |
|----------------|-----------|-------|
| `detect_ambiguity` | ambiguous | `refine_query` |
| `detect_ambiguity` | clear | `detect_complexity` |
| `refine_query` | `refine_confidence < 0.6` | `request_clarification` → loop to `detect_ambiguity` |
| `refine_query` | `refine_confidence ≥ 0.6` | `detect_complexity` |
| `detect_complexity` | complex | `decompose_query` |
| `detect_complexity` | simple | `prepare_retrieval` |
| retrieval merge | — | Top-50 candidates |
| `cross_encoder_rerank` | — | Top-10 |
| `assess_confidence` | composite `< 0.4` | `escalate` |
| `assess_confidence` | composite `≥ 0.4` | `synthesize_answer` |

---

## 5. Non-Functional Requirements

### 5.1 Performance & Resource (NFR-PERF)
- **NFR-PERF-1** Single-user, one-document/one-query at a time (D3); no concurrency requirement.
- **NFR-PERF-2** **GPU is the primary compute path:** the BGE reranker SHALL run on the NVIDIA RTX 4090 (CUDA) by default (D5/FR-Q3.2); Docling parsing SHOULD also use the GPU when beneficial. CPU fallback MUST remain functional (degraded speed) for portability/CI.
- **NFR-PERF-3** Embedding calls batched (100/call) to minimize round-trips/cost.
- **NFR-PERF-4** ML models (Docling, BGE reranker) SHALL be loaded **once per process** and cached locally after first download — not reloaded per document/query.
- **NFR-PERF-5** Query latency target (informational, single-user dev): a simple query SHOULD return within a few seconds on GPU; the dominant costs are OpenAI reasoning/synthesis round-trips and (CPU) reranking. No hard SLA.
- **NFR-PERF-6** The reranker SHALL process the Top-50 candidate set within one batch where memory permits.

### 5.2 Reliability & Data Integrity (NFR-REL)
- **NFR-REL-1** No partial state: ingestion terminal failures leave zero partial graph data (atomic write via bounded batches + compensating delete, FR-7.1/7.8).
- **NFR-REL-2** Idempotent re-run on pre-commit ingestion failure (no manual cleanup).
- **NFR-REL-3** Exactly-once per hash via SHA-256 dedup.
- **NFR-REL-4** Cross-document iiRDS dedup via `MERGE`.
- **NFR-REL-5** Resumability: human-review (ingest) and clarification/escalation (query) interrupts survive process exit and resume from persisted state.
- **NFR-REL-6** Neo4j data persisted to a **named Docker volume**; container restarts preserve the graph. Likewise the **Postgres checkpoint store** (native on-prem) persists in its own data directory so suspended/interrupted runs survive service/host restarts (D15).
- **NFR-REL-7** Transient OpenAI failures tolerated via 3× exponential backoff before any terminal/error decision (ingest and query).
- **NFR-REL-8** **Embedding consistency:** query and ingestion MUST use identical embedding model/dimension/metric (FR-Q0.5); enforced in config, not duplicated literals.
- **NFR-REL-9** Retrieval/synthesis SHALL degrade gracefully when a retriever returns zero results (e.g., empty BM25) rather than failing the whole query.

### 5.3 Security & Privacy (NFR-SEC)
- **NFR-SEC-1** OpenAI API key via env/`.env` (python-dotenv); never committed; `.env` git-ignored.
- **NFR-SEC-2** Neo4j credentials via env/config, not hard-coded.
- **NFR-SEC-3** Dependency vulnerability scan (e.g. `pip-audit`) at build; packages at latest secure version within pinned line.
- **NFR-SEC-4** **Data egress awareness:** document text (first 3000 chars at tagging) and **all chunk text** (embeddings) on the write path, and **query text + reasoning/synthesis context** on the read path, are sent to OpenAI. Documented; sensitive content needs organizational approval. (Reranking is local and does **not** egress.)
- **NFR-SEC-5** Neo4j bound to `localhost` (`127.0.0.1`) only; default credentials changed.
- **NFR-SEC-6** Temp files (raw bytes) created with restrictive permissions and deleted immediately after parse (FR-2.5).
- **NFR-SEC-7** API keys/credentials never written to logs, receipts, or the graph.

### 5.4 Observability (NFR-OBS)
- **NFR-OBS-1** Structured logging records stage entry/exit, status, and errors with `thread_id` (implemented per **§5.8 NFR-LOG**).
- **NFR-OBS-2** Per-stage timings (`stage_timings_ms`) captured for both pipelines.
- **NFR-OBS-3** Token/cost tracking per ingestion run; query-side OpenAI token usage SHOULD also be tracked per query.
- **NFR-OBS-4** The `ingestion_logs/` directory provides the ingestion audit trail — one self-contained JSON file per run (FR-8.1a).
- **NFR-OBS-5** Full run state inspectable post-hoc from the Postgres checkpointer via `thread_id` (both pipelines).
- **NFR-OBS-6** Query answers SHALL log which retrievers contributed and the final chunk ids used (traceability/debuggability).

### 5.5 Maintainability & Portability (NFR-MAINT)
- **NFR-MAINT-1** Each stage = independently testable node with typed I/O (pydantic).
- **NFR-MAINT-2** Configuration centralized & externally configurable (env/config): chunk size, overlap, embed batch size, ingest confidence threshold, refine/escalate thresholds, per-retriever/Top-50/Top-10 cutoffs, loop caps, model names, devices, Neo4j URI. **No magic numbers** inline. **Full schema: §2.5** (D13).
- **NFR-MAINT-3** Dependencies pinned in a lockfile; resolver honors the ML-stack pins (§2.3).
- **NFR-MAINT-4** Target **Python 3.13**; CI/dev runs on 3.13.
- **NFR-MAINT-5** LLM/embedding boundary wrapped behind a thin internal interface to ease future provider swaps (OpenAI default).
- **NFR-MAINT-6** Windows 11 target; use `pathlib`, avoid POSIX-only assumptions.
- **NFR-MAINT-7** Shared concerns (Neo4j connection, Postgres checkpointer, embedding client, config, logging) factored into modules reused by **both** pipelines.

### 5.6 Usability (NFR-USE)
- **NFR-USE-1** All terminal/error messages actionable (what failed + what to do).
- **NFR-USE-2** CLIs print clear per-stage progress.
- **NFR-USE-3** Human-in-the-loop CLIs (tag review, clarification, escalation) present current state clearly and accept input without editing raw state.

### 5.7 Cost (NFR-COST)
- **NFR-COST-1** Ingestion cost = one `gpt-4o-mini` call/doc + embeddings over chunk text; reported per run.
- **NFR-COST-2** **Query cost** = multiple `gpt-4o-mini` reasoning calls (ambiguity, refine, complexity, decompose, classify) + one query embedding + synthesis; reranking is **free/local**. Reasoning calls SHOULD be minimized (e.g., skip refine/decompose when not needed) to bound per-query cost.
- **NFR-COST-3** No OpenAI cost for duplicate documents (dedup precedes any call).

### 5.8 Logging (NFR-LOG)
- **NFR-LOG-1** All components SHALL log via the Python `logging` module through a **single shared configuration** (NFR-MAINT-7); `print()` SHALL NOT be used for diagnostics.
- **NFR-LOG-2** Log records SHALL be **structured JSON** by default (`LOG_FORMAT=json`), with a human-readable `text` mode for local debugging. Each record SHALL include at least: timestamp, level, logger name, message, `thread_id`, pipeline (`ingest`/`query`), and stage.
- **NFR-LOG-3** Logs SHALL be emitted to **both the console and a per-run file** under `LOG_DIR` (default `./logs`, git-ignored). Each process invocation writes its own file named `<pipeline>_<UTCstamp>_<pid>.log` so runs never share a file; within a single run, file rotation SHALL still be size-based with retained backups (e.g. 10 MB × 5) to bound disk usage.
- **NFR-LOG-4** The application log level SHALL be configurable via `LOG_LEVEL` (default `INFO`); `DEBUG` enables verbose tracing.
- **NFR-LOG-5** Third-party loggers (`httpx`, `openai`, `neo4j`, `psycopg`, `transformers`, `docling`, `urllib3`) SHALL default to `WARNING` to suppress noise, independent of the app `LOG_LEVEL`.
- **NFR-LOG-6** Logs SHALL NOT contain secrets (API keys, DB passwords, connection-string credentials) — reinforcing NFR-SEC-7 — and SHALL NOT dump full payloads (document text, chunk text, embedding vectors); such content SHALL be omitted, truncated, or summarized (lengths/counts/ids).
- **NFR-LOG-7** Every stage transition (entry/exit + status) and every error/retry SHALL be logged with its `thread_id` for correlation (consolidates NFR-OBS-1); exceptions SHALL be logged with stack traces at `ERROR`.
- **NFR-LOG-8** The on-disk application log (NFR-LOG-3) is operational/diagnostic and is **distinct** from the durable ingestion audit trail (per-run files under `ingestion_logs/`, FR-8.1 / NFR-OBS-4) and the Neo4j `IngestionRecord`; the three serve different purposes and coexist.

---

## 6. Constraints & Assumptions

### 6.1 Constraints
- **C1** Python **3.13.14** (pinned). Requires a CUDA-build PyTorch (not the CPU default).
- **C2** Neo4j runs **only** as a Docker container on the laptop.
- **C3** OpenAI is the LLM + embedding provider; **internet + valid API key required** for ingestion tagging/embedding and for query reasoning/synthesis. (Reranking works offline.)
- **C4** One document per ingest invocation; one query per query invocation (no batch).
- **C5** Windows 11 host.
- **C6** Postgres (the LangGraph checkpoint store, D15) runs as a **native on-prem service on the laptop** — localhost-bound (not Docker). Neo4j remains Dockerized (C2).

### 6.2 Assumptions & Verified Hardware
**Verified target hardware (2026-06-15):**
- **GPU:** NVIDIA GeForce RTX 4090 Laptop GPU — **16 GB VRAM**, driver 596.36, CUDA 13.2 (Ada Lovelace, sm_89). Secondary: Intel UHD iGPU (not used for compute).
- VRAM budget: BGE reranker ≈0.5 GB (fp16) + Docling models (modest) leave ample headroom within 16 GB even with desktop apps resident.

Assumptions:
- **A1** Docker Desktop installed and running (hosts the Neo4j container). **Postgres is installed natively** on the laptop (D15) and its service is running.
- **A2** NVIDIA GPU driver + a CUDA-build of PyTorch are installed (per §2.3). Disk available for Docling + BGE + **EasyOCR** weights (first-run downloads: Docling/BGE several hundred MB, EasyOCR ~100 MB).
- **A3** OpenAI account has quota for `gpt-4o-mini` + `text-embedding-3-small`.
- **A4** Source docs are supported formats (FR-1.2). Encrypted/password-protected PDFs are **detected and rejected** at intake with a clear message (FR-1.2a), not silently mis-parsed.
- **A5** Embedding model+dimension (1536) are fixed for the index lifetime; changing them requires a full re-index (and the query side must change in lockstep).
- **A6** The graph already contains ingested documents before the query pipeline is exercised (query depends on ingestion output).

---

## 7. Risks & Mitigations
> v1.0 risks RISK-1/2/3 (Docling-degraded, no-CUDA, 3.14 resolver friction) are **resolved by the move to Python 3.13** and removed.

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| **RISK-A** | OpenAI outage / rate limits at tag, embed, or query reasoning/synthesis. | Embed failure is terminal (ingest); query cannot complete. | 3× backoff; idempotent ingest re-run; tagging failure non-blocking; clear query-side error. |
| **RISK-B** | Confidential content sent to OpenAI (tagging, embeddings, query/synthesis). | Data leaves device. | Document egress (NFR-SEC-4); user gates sensitive content; reranking stays local; provider-swap seam (NFR-MAINT-5). |
| **RISK-C** | Embedding model/metric drift between ingest and query. | Silent retrieval-quality collapse. | Single shared config (NFR-REL-8 / FR-Q0.5); fail fast on dimension mismatch. |
| **RISK-D** | CUDA/driver/torch-build mismatch (wrong `cuXXX` wheel, or CPU wheel installed by accident). | Reranker silently runs on CPU or import fails. | Install torch from the CUDA index (§2.3); assert `torch.cuda.is_available()` at startup and log the device; fp16 + GPU on the RTX 4090 (NFR-PERF-2/6). |
| **RISK-E** | Multiple LLM reasoning calls per query inflate cost/latency. | Higher $/query, slower. | Skip unnecessary reasoning nodes (NFR-COST-2); cache/short-circuit clear+simple queries. |
| **RISK-F** | Decomposition/clarification loops fail to converge. | Query never returns. | Hard caps (D12, §2.5): clarification ≤2 rounds, decomposition single-level ≤5 sub-queries; on cap, fall back to best-effort with a low-confidence flag. |
| **RISK-G** | Embedding-model change later. | Index incompatibility. | Treat model+dim as immutable per index; re-index on change (A5). |
| **RISK-H** | Postgres (checkpoint store) unavailable at run start. | Pipelines can't persist state; interrupts/resume fail. | Bootstrap verifies Postgres + creates tables (FR-S0.5a); pipelines emit a clear error directing the user to start the local Postgres service; its data directory keeps prior checkpoints durable (NFR-REL-6). |

---

## 8. Acceptance Criteria

### 8.1 Ingestion
- **AC-1** Ingesting a valid PDF creates `Document` + `Chunk`s + iiRDS nodes + `IngestionRecord`, a per-run `ingestion_logs/…json` file, and a success summary (cost + timing).
- **AC-2** Re-ingesting the same file → clean `DUPLICATE` termination, no new graph data, **no** OpenAI calls.
- **AC-3** A corrupted file → parse-error termination, clear message, no graph writes.
- **AC-4** Simulated embedding failure → no Neo4j write; file re-ingestable.
- **AC-5** Low-confidence tagging suspends at human review; `python review_tags.py <thread_id>` resumes and completes with corrected tags.
- **AC-6** Simulated Neo4j write failure → zero partial data (compensating delete of the partial doc + committed chunk batches), safely retryable.
- **AC-7** Tables/lists/warnings/code each appear as single unfragmented chunks; long paragraphs split at ~512 tokens with 50-token overlap.
- **AC-7b** A document containing a whole-block chunk larger than `EMBED_MAX_INPUT_TOKENS` (e.g. a huge table or formula block) still ingests successfully: the block is hard-split into within-limit chunks (logged) and the embed call is never rejected for exceeding the input limit (FR-3.8).
- **AC-7a** A scanned/image-only PDF is parsed with text recognized via **EasyOCR running on the GPU** (no OpenAI/cloud OCR call); `RERANKER_DEVICE`/OCR CPU fallback still produces text when CUDA is forced off.
- **AC-8** Neo4j container restart preserves all ingested data (volume persistence).
- **AC-8a** A suspended (interrupted) run survives a **Postgres** service restart and resumes from persisted checkpoint state via its `thread_id` (D15).

### 8.2 Query
- **AC-9** A clear, simple query returns a cited answer synthesized from Top-10 re-ranked chunks.
- **AC-10** An ambiguous query routes through `refine_query`; a low-confidence refinement (`< 0.6`) triggers `request_clarification`, and the user's response loops back and ultimately yields an answer.
- **AC-11** A complex multi-part query is decomposed into sub-queries, retrieved independently, and aggregated via RRF in synthesis.
- **AC-12** An exact part-number lookup is satisfied via BM25; a relational query exercises `graph_traverse`; a paraphrased NL query exercises `vector_retrieve`.
- **AC-13** A low-confidence retrieval (composite `< 0.4`) routes to `escalate` (human review) instead of returning a weak answer.
- **AC-14** Reranking runs locally with **no** OpenAI call, and **on the GPU** by default — a startup/health check confirms `torch.cuda.is_available() == True` and the reranker reports device `cuda:0`; forcing `RERANKER_DEVICE=cpu` still works (fallback).
- **AC-15** Query and ingestion use the identical embedding model/dimension/metric (config-asserted).
- **AC-15a** Every query runs **all three retrievers** (fan-out verified); changing `query_class` alters ranking weights, not which retrievers execute (D10).
- **AC-15b** The clarification loop stops after `MAX_CLARIFICATION_ROUNDS` (2) and proceeds best-effort with a low-confidence flag; a complex query yields **≤ `MAX_SUBQUERIES` (5)** sub-queries (D12).
- **AC-15c** `SYNTHESIS_MODEL` is read from config — setting it to `gpt-4o` switches the synthesis model with **no code change**; routing/reasoning nodes remain on `LLM_MODEL` (D8).
- **AC-15d** Clarification and escalation interrupts resume via the same `thread_id` resume-CLI pattern as ingest tag-review (D14).

### 8.3 Stack
- **AC-16** The full stack installs and runs on **Python 3.13.14** with §2.3 versions; the CUDA torch build is installed and the RTX 4090 is used by default (CPU fallback works); `pip-audit` reports no known high/critical vulnerabilities.
- **AC-17** The bootstrap command (§2.4) on a clean machine: asserts CUDA + logs the GPU name, downloads/caches all three model sets, smoke-tests each on the GPU, creates the Neo4j constraints/indexes **and the Postgres checkpointer tables**, and prints "environment ready"; on a CPU-only torch install it instead fails fast with an actionable message. After a successful bootstrap, a subsequent ingest/query performs **no** model downloads.
- **AC-18** Logs are emitted as structured JSON to **console + a per-run file** under `LOG_DIR` (`<pipeline>_<UTCstamp>_<pid>.log`), each record carrying `thread_id`/pipeline/stage; `LOG_LEVEL=DEBUG` increases app verbosity while third-party loggers stay at `WARNING`; and **no secrets or full document/chunk/embedding payloads** appear in any log line (NFR-LOG / NFR-SEC-7).

---

## 9. Resolved Decisions (formerly Open Questions)
**All previously-open questions are now resolved (D7–D14); none remain blocking.** History retained for traceability.

1. ~~**Synthesis model**~~ → **RESOLVED (D8):** default `SYNTHESIS_MODEL=gpt-4o-mini` (faithful to the diagrams); configurable to `gpt-4o` with no code change. Routing/reasoning nodes stay on `LLM_MODEL`.
2. ~~**iiRDS controlled vocabulary**~~ → **RESOLVED (D9):** `lifecycle_phase`/`information_type` are **closed, config-defined enums** (the diagram's value sets), validated and extensible to full iiRDS; `product`/`component` open free-text, `MERGE`d.
3. ~~**Retriever fan-out policy**~~ → **RESOLVED (D10):** **always run all three retrievers in parallel**; `query_class` **weights** the merge, never excludes a retriever.
4. ~~**Top-K tuning**~~ → **RESOLVED (D11):** per-retriever **25** → merged/deduped **50** → reranked **10**; all configurable (§2.5).
5. ~~**Loop caps**~~ → **RESOLVED (D12):** clarification **≤2 rounds** then best-effort + low-confidence flag; decomposition **single-level, ≤5 sub-queries** (mitigates RISK-F).
6. ~~**`.env` schema**~~ → **RESOLVED (D13):** explicit configuration schema in **§2.5**; shipped `.env.example`.
7. ~~**Model provisioning**~~ → **RESOLVED (D7):** explicit one-time bootstrap (§2.4, FR-S0.1–S0.7); lazy download retained only as fallback.
8. ~~**Escalation/clarification UX**~~ → **RESOLVED (D14):** all three interrupts (tag review, clarification, escalation) use the **same `interrupt()` + `thread_id` resume-CLI** pattern.

---

## 10. Sources (version/compatibility verification, June 2026)
- PyTorch Python 3.13 + CUDA wheels — [pytorch/pytorch#130249](https://github.com/pytorch/pytorch/issues/130249), [PyTorch releases](https://github.com/pytorch/pytorch/releases)
- BGE reranker `bge-reranker-v2-m3` / FlagEmbedding — [BAAI/bge-reranker-v2-m3 (Hugging Face)](https://huggingface.co/BAAI/bge-reranker-v2-m3), [FlagEmbedding usage](https://huggingface.co/BAAI/bge-reranker-v2-m3#using-flagembedding)
- Docling — [Docling Installation docs](https://docling-project.github.io/docling/getting_started/installation/)
- LangGraph 1.x / Python support — [LangChain/LangGraph 1.0 milestone](https://blog.langchain.com/langchain-langgraph-1dot0/), [langchain-core on PyPI](https://pypi.org/project/langchain-core/)
- Neo4j Python driver 6.2 / vector & full-text index — [Neo4j Python Driver docs](https://neo4j.com/docs/api/python-driver/current/), [Vector indexes — Cypher Manual](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/), [neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)
- OpenAI SDK / tiktoken — [openai on PyPI](https://pypi.org/project/openai/)
- pydantic 2.x / tenacity / python-dotenv — [Pydantic releases](https://pydantic.dev/articles/pydantic-v2-12-release), [tenacity on PyPI](https://pypi.org/project/tenacity/), [python-dotenv releases](https://github.com/theskumar/python-dotenv/releases)
- Python 3.13 — [Python 3.13 What's New](https://docs.python.org/3/whatsnew/3.13.html)
