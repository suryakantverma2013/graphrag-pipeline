"""Typed configuration object — single source of all secrets + tunables.

Implements the §2.5 configuration schema (decisions D8/D11/D12/D13). A single
pydantic-settings object loads and validates every key from the environment /
`.env` and is the ONE config consumed by both pipelines (NFR-MAINT-7, NFR-REL-8).
No magic numbers inline anywhere else (NFR-MAINT-2).
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import AliasChoices, Field
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
    # OCR is needed only for scanned/image PDFs. Born-digital PDFs (with a real
    # text layer) do NOT need it, and OCR is the heaviest, most memory-hungry
    # parse stage — disabling it lets large text PDFs parse without exhausting
    # memory (std::bad_alloc on big books). Default ON for correctness on scanned
    # input (FR-2.3); set OCR_ENABLED=false for large born-digital PDFs.
    ocr_enabled: bool = True
    # Parse only the first N pages (0 = no limit). Bounds memory/time on very
    # large PDFs (e.g. a 1000+ page book); pages beyond the cap are not parsed.
    pdf_max_pages: int = 0
    # Page rasterization resolution. Docling's images_scale = dpi / 72; lower DPI
    # = smaller page bitmaps = less memory during preprocess. 72 = Docling default
    # (unchanged behavior); drop to e.g. 48 to relieve memory pressure.
    pdf_render_dpi: int = 72

    # --- Tunables (defaults from the diagrams / decisions) ---
    chunk_max_tokens: int = 512             # FR-3.4
    chunk_overlap_tokens: int = 50          # FR-3.4
    embed_batch_size: int = 100             # FR-6.2
    tag_confidence_threshold: float = 0.5   # FR-4.7
    refine_confidence_threshold: float = 0.6  # FR-Q1.4
    escalate_confidence_threshold: float = 0.4  # FR-Q3.5
    per_retriever_k: int = 25               # D11 / FR-Q2.6
    retrieve_top_k: int = 50                # D11 / FR-Q2.6
    rerank_top_k: int = 10                  # D11 / FR-Q3.3
    max_clarification_rounds: int = 2       # D12 / FR-Q1.6
    max_subqueries: int = 5                 # D12 / FR-Q1.10

    # --- Logging (NFR-LOG) ---
    log_level: str = "INFO"                 # NFR-LOG-4
    log_format: str = "json"                # json | text (NFR-LOG-2)
    log_dir: str = "./logs"                 # rotating-file location (NFR-LOG-3)

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
