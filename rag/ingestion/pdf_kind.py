"""Classify a PDF as 'digital', 'scanned', or 'mixed' from its text layer (FR-2.3e).

Shared by the ingestion `parse` node (OCR auto-selection when OCR_ENABLED=auto)
and the standalone `detect_pdf_kind.py` CLI, so detection is defined exactly once.

Why this is more than a text probe
----------------------------------
A naive "do a few pages have text?" check has two failure modes, both of which
surface as SILENT problems at ingest time:

  1. Hybrid books (born-digital body + a scanned appendix or inserted pages) get
     called fully digital, so the scanned pages ingest with no text and no error.
  2. Born-digital books with many FIGURE or BLANK pages (e.g. a programming book
     full of diagrams) get called scanned/mixed and trigger a needless,
     hours-long OCR run that recovers nothing useful.

So for every sampled page we look at BOTH signals: the extractable text length,
and — if a page has little/no text — how much of it a single image covers. A
text-less page counts as **scanned** only when an image covers most of it (a
full-page scan); blank pages and partial figures are excluded. The
digital/scanned/mixed decision is a ratio over *text-vs-scanned* pages only.
(Validated: a 1882-page born-digital programming book has ~16% text-less figure
pages but ZERO full-page-image pages, so it classifies digital.)

On any probe error (missing pypdfium2, unreadable/encrypted PDF) `analyze` returns
a 'digital' result whose `reason` explains why — the fast path. Rationale: a wrong
'digital' guess yields too few chunks (visible, re-ingestable), whereas a needless
OCR run can cost ~30 min.
"""
from __future__ import annotations

from dataclasses import dataclass

# A page with at least this many extractable characters is treated as having a
# real text layer. Body text pages have hundreds–thousands of chars; a scanned
# page has ~0 (maybe a stray page number). 100 cleanly separates the two while
# tolerating sparse pages (a figure with a short caption).
_TEXT_CHAR_THRESHOLD = 100

# A text-less page is counted as *scanned* only when a single image covers at
# least this fraction of the page area — i.e. a full-page scan. Partial figures
# in born-digital books top out well below this (observed: 0.4–0.8), so they are
# excluded and don't trigger OCR.
_SCANNED_IMG_COVER = 0.80

# Coverage-ratio bands over text-vs-scanned pages (blank/figure pages excluded):
#   ratio >= _DIGITAL_RATIO -> 'digital'  (allow a few scanned pages)
#   ratio <= _SCANNED_RATIO -> 'scanned'  (allow a few stray-text pages)
#   in between              -> 'mixed'    (hybrid; OCR to keep the scanned pages)
_DIGITAL_RATIO = 0.85
_SCANNED_RATIO = 0.10

# Upper bound on pages sampled. Documents at/under this are scanned in FULL (no
# blind spots); larger ones are sampled evenly. 1000 covers virtually every book
# end-to-end while bounding the worst case to a few seconds of text extraction.
_MAX_SAMPLE_PAGES = 1000


@dataclass
class PdfKindResult:
    """The classification plus the metrics that produced it (for --json / logs)."""

    kind: str                  # 'digital' | 'scanned' | 'mixed'
    page_count: int            # total pages in the PDF (0 if unknown)
    sampled_pages: int         # how many pages were actually inspected
    text_pages: int            # sampled pages with >= _TEXT_CHAR_THRESHOLD chars
    scanned_pages: int         # text-less pages dominated by a full-page image
    blank_or_figure_pages: int # text-less pages that are blank or a partial figure
    text_ratio: float          # text / (text + scanned); ignores blank/figure pages
    avg_chars: float           # mean extracted chars over sampled pages
    reason: str                # human-readable explanation


def _sample_indices(n: int, max_sample: int) -> list[int]:
    """Page indices to inspect: full coverage when small, evenly spread when large.

    Skips the cover (page 0) when there's more than one page — covers are often a
    full-page image even in born-digital PDFs and would skew the result.
    """
    if n <= 0:
        return []
    lo = 1 if n > 2 else 0  # skip cover unless the doc is tiny
    pages = list(range(lo, n))
    if len(pages) <= max_sample:
        return pages
    step = len(pages) / max_sample
    picked = sorted({lo + int(k * step) for k in range(max_sample)})
    return [i for i in picked if i < n]


def _inspect_page(doc, i: int, char_threshold: int, img_type: int) -> tuple[str, int]:
    """Classify one page. Returns (category, char_count).

    category:
      'text'    — has a real text layer (>= char_threshold chars)
      'scanned' — text-less, with an image covering >= _SCANNED_IMG_COVER of the page
      'other'   — text-less and blank, or only a partial figure (not a scan)
    A page that can't be read at all counts as 'other' with 0 chars.
    """
    try:
        page = doc[i]
    except Exception:  # noqa: BLE001 — unreadable page → neutral 'other'
        return "other", 0
    try:
        textpage = page.get_textpage()
        try:
            chars = len(textpage.get_text_range().strip())
        finally:
            textpage.close()
        if chars >= char_threshold:
            return "text", chars

        # Text-less: measure the largest single image's page coverage.
        try:
            w, h = page.get_size()
            page_area = w * h
        except Exception:  # noqa: BLE001
            page_area = 0.0
        max_cover = 0.0
        if page_area > 0:
            for obj in page.get_objects():
                if obj.type != img_type:
                    continue
                try:
                    left, bottom, right, top = obj.get_bounds()
                    max_cover = max(max_cover, abs((right - left) * (top - bottom)) / page_area)
                except Exception:  # noqa: BLE001 — skip an image we can't measure
                    continue
        return ("scanned" if max_cover >= _SCANNED_IMG_COVER else "other"), chars
    finally:
        page.close()


def _classify(text_pages: int, scanned_pages: int, other_pages: int, avg: float) -> tuple[str, str]:
    note = (f"(text={text_pages}, scanned={scanned_pages}, "
            f"blank/figure={other_pages}; avg {avg:.0f} chars/page)")
    denom = text_pages + scanned_pages
    if denom == 0:
        return "digital", f"no text and no full-page-image pages found {note}; defaulting to digital"
    ratio = text_pages / denom
    if scanned_pages == 0 or ratio >= _DIGITAL_RATIO:
        return "digital", f"{text_pages}/{denom} text-vs-scanned pages have a text layer (ratio {ratio:.2f} >= {_DIGITAL_RATIO}) {note}"
    if ratio <= _SCANNED_RATIO:
        return "scanned", f"only {text_pages}/{denom} text-vs-scanned pages have a text layer (ratio {ratio:.2f} <= {_SCANNED_RATIO}) {note} — needs OCR"
    return "mixed", (f"{text_pages}/{denom} text-vs-scanned pages have a text layer (ratio {ratio:.2f}) {note} "
                     "— part scanned; OCR recommended so the scanned pages aren't dropped")


def analyze(
    path: str,
    *,
    max_sample: int = _MAX_SAMPLE_PAGES,
    char_threshold: int = _TEXT_CHAR_THRESHOLD,
) -> PdfKindResult:
    """Inspect the PDF and classify it. Never raises — probe failures degrade to a
    'digital' result whose `reason` explains why."""
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError:
        return PdfKindResult("digital", 0, 0, 0, 0, 0, 1.0, 0.0,
                             "pypdfium2 not installed; defaulting to digital")
    try:
        doc = pdfium.PdfDocument(path)
    except Exception as exc:  # noqa: BLE001 — unreadable/encrypted → safe default
        return PdfKindResult("digital", 0, 0, 0, 0, 0, 1.0, 0.0,
                             f"could not open PDF ({exc}); defaulting to digital")

    img_type = pdfium_c.FPDF_PAGEOBJ_IMAGE
    try:
        n = len(doc)
        indices = _sample_indices(n, max_sample)
        if not indices:
            return PdfKindResult("digital", n, 0, 0, 0, 0, 1.0, 0.0,
                                 "no pages to sample; defaulting to digital")
        text_pages = scanned_pages = other_pages = 0
        total_chars = 0
        for i in indices:
            category, chars = _inspect_page(doc, i, char_threshold, img_type)
            total_chars += chars
            if category == "text":
                text_pages += 1
            elif category == "scanned":
                scanned_pages += 1
            else:
                other_pages += 1
    finally:
        doc.close()

    sampled = len(indices)
    avg = total_chars / sampled
    denom = text_pages + scanned_pages
    text_ratio = (text_pages / denom) if denom else 1.0
    kind, reason = _classify(text_pages, scanned_pages, other_pages, avg)
    return PdfKindResult(kind, n, sampled, text_pages, scanned_pages, other_pages,
                         round(text_ratio, 4), round(avg, 1), reason)


def classify(path: str) -> str:
    """Convenience wrapper: the classification word only."""
    return analyze(path).kind
