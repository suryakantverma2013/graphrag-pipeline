#!/usr/bin/env python
"""CLI wrapper around rag.ingestion.pdf_kind — classify a PDF's text layer.

    python detect_pdf_kind.py [--json] <file.pdf>

Prints exactly one word to stdout — 'digital' | 'scanned' | 'mixed' — with a
one-line explanation on stderr (`--json` prints the full metrics to stdout
instead). The classifier and its thresholds live in rag/ingestion/pdf_kind.py.

NOTE: the same classifier now runs automatically inside the ingestion `parse`
node when OCR_ENABLED=auto (the default), so you normally do NOT need to run this
by hand. It's kept for spot-checks and debugging a surprising OCR decision.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    positional = [a for a in argv if not a.startswith("-")]
    if len(positional) != 1:
        print("usage: python detect_pdf_kind.py [--json] <file.pdf>", file=sys.stderr)
        return 2

    from rag.ingestion.pdf_kind import analyze

    result = analyze(positional[0])
    # Reasoning to stderr so the stdout token stays the clean machine contract.
    print(f"{result.kind}: {result.reason}", file=sys.stderr)
    if as_json:
        print(json.dumps(asdict(result)))
    else:
        print(result.kind)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
