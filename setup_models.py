#!/usr/bin/env python
"""Bootstrap CLI (D7 / §2.4 / FR-S0.1).

One-time environment warm-up + verification. Run before any ingest/query:

    python setup_models.py
"""

from rag.bootstrap import main

if __name__ == "__main__":
    main()
