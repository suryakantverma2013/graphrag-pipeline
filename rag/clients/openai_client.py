"""Thin OpenAI wrapper for LLM reasoning + embeddings (NFR-MAINT-5).

All LLM/embedding access goes through this seam so the provider can be swapped
later (NFR-MAINT-5). Transient failures retry 3x with exponential backoff
(FR-4.4 / FR-6.4 / FR-Q0.6 / NFR-REL-7) via tenacity. The embedding model,
dimension and metric are fixed by config (NFR-REL-8 / FR-Q0.5) and shared by
ingestion and query.

NOTE (skeleton): method bodies that perform real API calls are marked TODO.
The retry/structure/signatures are in place; fill in the SDK calls next.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import AppConfig

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Shared retry policy: 3 attempts, exponential backoff (NFR-REL-7).
_retry = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)


class OpenAIClient:
    """Single seam over the OpenAI SDK for chat + embeddings."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client: "OpenAI | None" = None

    @property
    def client(self) -> "OpenAI":
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._config.openai_api_key,
                base_url=self._config.openai_base_url,
            )
        return self._client

    @_retry
    def chat(self, messages: list[dict[str, str]], *, model: str | None = None) -> str:
        """Single chat completion, returning the assistant text.

        `model` defaults to LLM_MODEL (reasoning/tagging); pass SYNTHESIS_MODEL
        for answer synthesis (D8). Used by all reasoning nodes (FR-Q0.6).
        """
        response = self.client.chat.completions.create(
            model=model or self._config.llm_model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    @_retry
    def complete_structured(self, messages: list[dict[str, str]], schema: type, *, model: str | None = None) -> Any:
        """Structured (schema-validated) completion for tagging/classification.

        Uses the SDK's native structured-output parsing (`response_format=<pydantic
        model>`); the result is validated against `schema`. A refusal or unparsable
        output raises, so tenacity retries it (malformed = retryable, FR-4.6).
        """
        response = self.client.chat.completions.parse(
            model=model or self._config.llm_model,
            messages=messages,  # type: ignore[arg-type]
            response_format=schema,
        )
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise ValueError(f"model refused structured output: {message.refusal}")
        if message.parsed is None:
            raise ValueError("structured completion returned no parsed object")
        return message.parsed

    @_retry
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts with EMBEDDING_MODEL (1536-dim, cosine).

        Callers batch to EMBED_BATCH_SIZE (FR-6.2). Identical model/dim/metric on
        ingest and query is enforced via config (NFR-REL-8 / FR-Q0.5): the 1536-dim
        is requested explicitly so the vectors always match the Neo4j vector index.
        """
        response = self.client.embeddings.create(
            model=self._config.embedding_model,
            input=list(texts),
            dimensions=self._config.embedding_dimensions,
        )
        # The API echoes inputs back in order (FR-6.2 batch integrity).
        return [item.embedding for item in response.data]
