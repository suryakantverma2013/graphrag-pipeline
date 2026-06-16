"""Docling converter factory + warm-up (FR-2.x / FR-S0.3 / FR-S0.4).

Shared so the ingestion parse node (FR-2) and the bootstrap warm-up build the
converter identically: DocLayNet layout + TableFormer tables + EasyOCR (D6,
explicitly selected per FR-2.3d), on the configured device — CUDA by default,
CPU fallback (FR-2.3b / NFR-PERF-2). Parsing runs fully locally, no network
(FR-2.1).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import AppConfig

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def _use_cuda(config: AppConfig) -> bool:
    """GPU when requested and available, else CPU with a warning (FR-2.3b)."""
    if config.reranker_device.lower().startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return True
            logger.warning("CUDA requested but unavailable; Docling/EasyOCR on CPU")
        except ImportError:
            logger.warning("torch unavailable; Docling/EasyOCR on CPU")
    return False


def build_converter(config: AppConfig) -> "DocumentConverter":
    """Build a DocumentConverter with EasyOCR + table structure on the device."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        EasyOcrOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    use_cuda = _use_cuda(config)
    accelerator = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if use_cuda else AcceleratorDevice.CPU
    )
    # D6: EasyOCR engine selected explicitly (FR-2.3d), GPU by default (FR-2.3b).
    ocr_options = EasyOcrOptions(use_gpu=use_cuda)
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,                     # FR-2.3 enable OCR for scanned/image PDFs
        ocr_options=ocr_options,
        do_table_structure=True,         # FR-2.2 TableFormer
        accelerator_options=accelerator,
    )
    # Images share the PDF pipeline in Docling; apply the same OCR options to both.
    fmt = PdfFormatOption(pipeline_options=pipeline_options)
    img = ImageFormatOption(pipeline_options=pipeline_options)
    return DocumentConverter(format_options={InputFormat.PDF: fmt, InputFormat.IMAGE: img})


def _make_smoke_image() -> str:
    """Write a tiny PNG with text for the warm-up smoke convert."""
    from PIL import Image, ImageDraw

    path = Path(tempfile.gettempdir()) / "rag_docling_smoke.png"
    image = Image.new("RGB", (320, 80), "white")
    ImageDraw.Draw(image).text((10, 30), "Docling smoke test 123", fill="black")
    image.save(path)
    return str(path)


def warm_up(config: AppConfig) -> None:
    """Download/cache Docling + EasyOCR weights and smoke-test on the device.

    FR-S0.3: pre-download DocLayNet/TableFormer + EasyOCR weights.
    FR-S0.4: convert a tiny image so the layout + OCR models actually execute.
    """
    from docling.utils.model_downloader import download_models

    # Fetch ONLY the models this pipeline uses (do_ocr + do_table_structure):
    # layout (DocLayNet) + TableFormer + EasyOCR. Skip enrichment models
    # (figure classifier, code/formula) and rapidocr — they are not used and
    # only add fragile downloads (FR-S0.3 scoped to the pipeline's needs).
    download_models(
        with_layout=True,
        with_tableformer=True,
        with_easyocr=True,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
        progress=False,
    )
    logger.info("Docling + EasyOCR weights cached")

    image_path = _make_smoke_image()
    try:
        result = build_converter(config).convert(image_path)  # FR-S0.4
        chars = len(result.document.export_to_text())
        logger.info("Docling smoke OK (extracted_chars=%d)", chars)
    finally:
        Path(image_path).unlink(missing_ok=True)
