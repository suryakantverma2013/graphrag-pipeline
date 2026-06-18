"""Typed configuration object — single source of all secrets + tunables.

Implements the §2.5 configuration schema (decisions D8/D11/D12/D13). A single
pydantic-settings object loads and validates every key from the environment /
`.env` and is the ONE config consumed by both pipelines (NFR-MAINT-7, NFR-REL-8).
No magic numbers inline anywhere else (NFR-MAINT-2).
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Embedding invariants (NFR-REL-8 / FR-Q0.5) ------------------------------
# Model, dimension and metric are fixed for the index lifetime (assumption A5).
# Query and ingestion MUST agree exactly; mismatch is a correctness defect.
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_SIMILARITY = "cosine"


class AppConfig(BaseSettings):
    """Validated application configuration (§2.5).

    Field names map case-insensitively to the env vars in `.env.example`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Secrets / connections ---
    openai_api_key: str = Field(..., description="OpenAI auth; never committed (NFR-SEC-1)")
    openai_base_url: str | None = Field(default=None, description="optional endpoint override")

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(..., description="changed from default (NFR-SEC-5)")
    neo4j_database: str = "neo4j"

    checkpoint_db_uri: str = Field(
        default="postgresql://langgraph:langgraph@127.0.0.1:5432/langgraph",
        description="LangGraph Postgres checkpointer (D15); localhost only",
    )

    # --- Models ---
    llm_model: str = "gpt-4o-mini"          # tagging + query reasoning (D1)
    synthesis_model: str = "gpt-4o-mini"    # answer synthesis; may be gpt-4o (D8)
    embedding_model: str = "text-embedding-3-small"  # shared ingest+query (FR-Q0.5)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # local cross-encoder
    reranker_device: str = "cuda"           # fallback "cpu" (FR-Q3.2a)
    ocr_engine: str = "easyocr"             # locked (D6)
    model_cache_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_HOME", "MODEL_CACHE_DIR"),
        description="local weight cache (FR-S0.3)",
    )
    hf_token: str | None = Field(
        default=None,
        description="optional Hugging Face token for higher download rate limits",
    )

    # --- Parsing / Docling (PDF memory & cost controls) ---
    # OCR is needed only for scanned/image PDFs and is the heaviest, most
    # memory-hungry parse stage. Tri-state (FR-2.3/2.3e), stored normalized to
    # 'auto' | 'on' | 'off' — consume it via resolve_ocr()/ocr_mode, never as a
    # bare bool:
    #   'auto' (default) — the parse node classifies each PDF from its own text
    #                      layer (rag/ingestion/pdf_kind.py) and enables OCR only
    #                      for 'scanned'/'mixed' files; no per-file hand-tuning.
    #   'on'  (or true)  — force OCR for every file.
    #   'off' (or false) — never OCR (fast path for known born-digital PDFs).
    ocr_enabled: str = "auto"
    # OCR languages (EasyOCR codes) used when OCR runs. Comma/space-separated;
    # the default matches EasyOCR's built-in Latin set. EasyOCR requires the set
    # to be script-compatible (English combines with anything, but you cannot mix
    # e.g. Chinese + Arabic) — an incompatible set raises at model load. For
    # non-Latin scans set e.g. OCR_LANGUAGES="ch_sim,en" / "ja,en" / "ar,en" /
    # "ru,en" / "hi,en". Consume via ocr_language_list.
    ocr_languages: str = "fr,de,es,en"
    # Parse only the first N pages (0 = no limit). Bounds memory/time on very
    # large PDFs (e.g. a 1000+ page book); pages beyond the cap are not parsed.
    pdf_max_pages: int = 0
    # Page rasterization resolution. Docling's images_scale = dpi / 72; lower DPI
    # = smaller page bitmaps = less memory during preprocess. 72 = Docling default
    # (unchanged behavior); drop to e.g. 48 to relieve memory pressure. This is the
    # BORN-DIGITAL DPI: the page bitmap is only a layout backdrop, so 72 is plenty.
    pdf_render_dpi: int = 72
    # Page rasterization resolution used WHEN OCR IS ACTIVE. On a scanned page the
    # bitmap is the only text source, so EasyOCR reads it directly — 72 dpi makes
    # small glyphs (subscripts, math notation, diacritics) marginal. Rendering
    # sharper improves OCR accuracy at the cost of memory/time (it scales ~dpi^2),
    # so the OCR path is also batched smaller (pdf_parse_batch_pages_ocr). The
    # parse node folds this into pdf_render_dpi only when the OCR decision is on
    # (FR-2.3e); the born-digital fast path keeps pdf_render_dpi=72 untouched.
    pdf_render_dpi_ocr: int = 150
    # Page-range batch size for parsing (PDF only). Docling accumulates memory
    # within a single convert() call and OOMs (std::bad_alloc) on very large PDFs
    # (~127 pages on a 24 GB box, independent of DPI/OCR). When > 0, a PDF is
    # parsed in slices of this many pages, releasing memory between slices, then
    # chunked contiguously — capturing the WHOLE document. 0 = single convert
    # (legacy). 100 leaves headroom under the observed ceiling.
    pdf_parse_batch_pages: int = 100
    # Slice size used when OCR is active. OCR (rasterize + two neural models per
    # page) is far heavier per page than text extraction, AND the OCR path renders
    # at the sharper pdf_render_dpi_ocr (150 → ~4.3x the pixels of 72), so peak
    # memory per slice is much higher — hence a much smaller slice (12, vs 100
    # born-digital) to stay clear of Docling's std::bad_alloc ceiling. The parse
    # node picks this vs pdf_parse_batch_pages from the resolved OCR decision
    # (FR-2.8c / FR-2.3e). If a higher pdf_render_dpi_ocr OOMs, lower this further.
    pdf_parse_batch_pages_ocr: int = 12

    # --- Tunables (defaults from the diagrams / decisions) ---
    chunk_max_tokens: int = 512             # FR-3.4
    chunk_overlap_tokens: int = 50          # FR-3.4
    embed_batch_size: int = 100             # FR-6.2
    # Chunk-write batch size for the Neo4j write (FR-7.1). The per-document graph
    # write was originally ONE transaction; on very large books it grew to ~240k
    # store commands and Neo4j failed to apply it to the store ("Failed to apply
    # transaction"), wedging the database. Chunks are now written in batches of
    # this many per transaction so no single transaction can grow large enough to
    # fail; the Document node + iiRDS edges are written first and a failed run is
    # compensated by deleting the partial data, preserving the no-partial-data
    # guarantee (FR-7.8). Lower this if very large chunks still strain a write.
    neo4j_write_batch_chunks: int = 250
    tag_confidence_threshold: float = 0.5   # FR-4.7
    refine_confidence_threshold: float = 0.6  # FR-Q1.4
    escalate_confidence_threshold: float = 0.38  # FR-Q3.5 (calibrated 2026-06-16, real corpus)
    per_retriever_k: int = 25               # D11 / FR-Q2.6
    retrieve_top_k: int = 50                # D11 / FR-Q2.6
    rerank_top_k: int = 10                  # D11 / FR-Q3.3
    max_clarification_rounds: int = 2       # D12 / FR-Q1.6
    max_subqueries: int = 5                 # D12 / FR-Q1.10

    # --- Logging (NFR-LOG) ---
    log_level: str = "INFO"                 # NFR-LOG-4
    log_format: str = "json"                # json | text (NFR-LOG-2)
    log_dir: str = "./logs"                 # rotating-file location (NFR-LOG-3)

    @field_validator("ocr_enabled", mode="before")
    @classmethod
    def _normalize_ocr_enabled(cls, v: object) -> str:
        """Accept the documented tri-state plus bool-ish aliases, normalize to
        'auto' | 'on' | 'off'. Unrecognized values fall back to 'auto' (safe)."""
        s = str(v).strip().lower()
        if s in {"on", "true", "1", "yes", "y", "enable", "enabled"}:
            return "on"
        if s in {"off", "false", "0", "no", "n", "disable", "disabled"}:
            return "off"
        return "auto"

    @property
    def ocr_mode(self) -> str:
        """Normalized OCR intent: 'auto' | 'on' | 'off'."""
        return self.ocr_enabled

    @property
    def ocr_language_list(self) -> list[str]:
        """OCR_LANGUAGES parsed to a list of EasyOCR codes (never empty)."""
        raw = self.ocr_languages.replace(";", ",").replace(" ", ",")
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p] or ["en"]

    def resolve_ocr(self, pdf_kind: str | None) -> bool:
        """Effective do_ocr for a file (FR-2.3e). 'on'/'off' force the decision;
        'auto' enables OCR only for a 'scanned' or 'mixed' PDF kind."""
        if self.ocr_enabled == "on":
            return True
        if self.ocr_enabled == "off":
            return False
        return pdf_kind in {"scanned", "mixed"}

    @property
    def embedding_dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

    @property
    def embedding_similarity(self) -> str:
        return EMBEDDING_SIMILARITY


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the validated config once per process (NFR-MAINT-7).

    Also exports HF_TOKEN to os.environ so third-party libraries that read it
    directly (huggingface_hub, transformers, docling) pick it up — pydantic
    only loads it into this object, not the process env. `setdefault` keeps any
    value already set in the real environment.

    NOTE: We deliberately do NOT export HF_HOME. The reranker gets an explicit
    `cache_dir` (config.model_cache_dir), but Docling/HF Hub use their default
    cache (~/.cache/huggingface). Redirecting HF_HOME forced re-downloading the
    Docling weights — which already live in the default cache — over a flaky
    connection. Honoring a single project-local cache can be revisited once the
    weights are deliberately consolidated there (FR-S0.3).
    """
    config = AppConfig()  # type: ignore[call-arg]  # values come from env/.env
    if config.hf_token:
        os.environ.setdefault("HF_TOKEN", config.hf_token)
    return config
