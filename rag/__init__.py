"""LangGraph RAG pipeline (ingestion + query) over a Neo4j knowledge graph.

Source of truth: REQUIREMENTS.md v2.5 (decisions D1-D15, NFR-LOG).
This package holds the shared infrastructure (config, logging, clients) and the
two LangGraph StateGraphs (ingestion write-path, query read-path).
"""

__version__ = "0.1.0"

# Pipeline identifiers used for structured logging (NFR-LOG-2) and state.
PIPELINE_INGEST = "ingest"
PIPELINE_QUERY = "query"
