"""Local BGE cross-encoder reranker (FR-Q3.1/3.2; runs fully locally, no API).

Loads `BAAI/bge-reranker-v2-m3` ONCE per process and keeps it resident on the
GPU (NFR-PERF-4 / FR-Q3.2b). Runs on CUDA by default with fp16; if no CUDA
device is available it logs a warning and falls back to CPU rather than failing
(FR-Q3.2a). Retrieved chunk text scored here never egresses (NFR-SEC-4).

NOTE (skeleton): model load + scoring bodies are marked TODO.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..config import AppConfig

logger = logging.getLogger(__name__)

_INSTANCE: "Reranker | None" = None


class Reranker:
    """Process-wide singleton wrapper around the BGE reranker."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = None
        self._device = self._resolve_device(config.reranker_device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Honor RERANKER_DEVICE, falling back cuda->cpu with a warning (FR-Q3.2a)."""
        if requested.lower().startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but unavailable; reranker falling back to CPU")
                    return "cpu"
            except ImportError:
                logger.warning("torch not importable; reranker falling back to CPU")
                return "cpu"
        return requested

    def load(self) -> None:
        """Load weights onto the device (once). Called by bootstrap + first use."""
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker

        # fp16 only on CUDA (FR-Q3.2); on CPU it is unsupported/slower.
        use_fp16 = self._device.startswith("cuda")
        self._model = FlagReranker(
            self._config.reranker_model,
            use_fp16=use_fp16,
            devices=self._device,
            cache_dir=self._config.model_cache_dir,
        )
        logger.info("reranker loaded: %s on %s (fp16=%s)", self._config.reranker_model, self._device, use_fp16)

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return a relevance score in [0,1] per (query, passage) pair (FR-Q3.1).

        `normalize=True` applies the sigmoid mapping the spec requires.
        """
        self.load()
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._model.compute_score(pairs, normalize=True)  # type: ignore[union-attr]
        if isinstance(scores, (int, float)):  # single pair -> scalar
            return [float(scores)]
        return [float(s) for s in scores]


def get_reranker(config: AppConfig) -> Reranker:
    """Return the process-wide reranker singleton (NFR-PERF-4)."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Reranker(config)
    return _INSTANCE
