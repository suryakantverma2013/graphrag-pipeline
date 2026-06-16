
# Neo4j (Docker) — Installation & Configuration Guide

**Scope:** Install and configure the Neo4j graph database used by this RAG pipeline as a local Docker container on **Windows 11**. Neo4j stores the knowledge graph (`Document`/`Chunk`/iiRDS nodes), the **vector index** (1536-dim, cosine) and the **full-text (Lucene/BM25) index**.
**Companion files:** `docker-compose.yml` (defines the service), `.env` (credentials), `REQUIREMENTS.md` (§2.1 topology, §2.3 versions, FR-7.x graph writes, FR-S0.5 index bootstrap).

> **Edition:** This guide uses **Neo4j Community Edition**, which is sufficient for the single-user scope (decision D3). Community is single-database (the default `neo4j` DB) and has no role-based auth or online backup — all acceptable here. Enterprise is **not** required.

## 1. Prerequisites

1. **Docker Desktop for Windows** installed and running, using the **WSL 2 backend** (default on Windows 11).
   - Verify: open PowerShell and run `docker version` and `docker compose version`. Both must succeed.
2. **WSL 2** enabled (Docker Desktop installs/prompts for it). Verify: `wsl --status`.
3. At least **4 GB RAM free** for the container (default config is modest; see §5 for tuning).
4. This project checked out, with a populated **`.env`** (copied from `.env.example`) — specifically `NEO4J_PASSWORD` set to a real value.

## 2. Quick start (recommended — via docker-compose)

From the project root (the folder containing `docker-compose.yml`):

---
```powershell
# 1. Ensure .env exists and NEO4J_PASSWORD is set
Copy-Item .env.example .env      # if you haven't already, then edit .env

# 2. Pull the image and start Neo4j in the background
#    (Postgres runs natively on the laptop, not via compose — see REQUIREMENTS.md C6)
docker compose up -d

# 3. Watch Neo4j become ready (Ctrl+C to stop tailing)
docker compose logs -f neo4j

# 4. Confirm it is running
docker compose ps
```
---

Neo4j is ready when the log prints `Started.` and `Bolt enabled on 0.0.0.0:7687`.

- **Bolt** (driver/app connections): `bolt://127.0.0.1:7687`
- **Browser** (web UI): http://127.0.0.1:7474 — log in with user `neo4j` and your `NEO4J_PASSWORD`.

> After the container is healthy, run the application **bootstrap** (`REQUIREMENTS.md` §2.4 / FR-S0.5) to create the constraints and indexes automatically. You can also create them manually — see §8.

The `docker-compose.yml` service (for reference):

---
```yaml
services:
  neo4j:
    image: neo4j:2026.05-community     # see §3 for tag selection
    container_name: rag-neo4j
    restart: unless-stopped
    ports:
      - "127.0.0.1:7474:7474"          # HTTP / Browser — localhost only
      - "127.0.0.1:7687:7687"          # Bolt          — localhost only
    environment:
      NEO4J_AUTH: "neo4j/${NEO4J_PASSWORD:?set NEO4J_PASSWORD in .env}"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
```
---

## 3. Choosing the image & tag

Neo4j uses **Calendar Versioning** (`YYYY.MM.PATCH`) since 2025. Always pin the **`-community`** suffix so you don't accidentally pull Enterprise.

| Tag form | Meaning |
|----------|---------|
| `neo4j:2026.05-community` | Latest patch of the 2026.05 line (recommended — tracks patches) |
| `neo4j:2026.05.0-community` | Exact pinned patch (most reproducible) |
| `neo4j:2026-community` | Latest 2026.x (looser) |
| `…-trixie` / `…-ubi10` suffix | Base image variant (Debian 13 / RedHat UBI10); optional |

**Verify a tag exists before pinning** (tags advance monthly — pick the newest stable line at setup time):

---
```powershell
docker pull neo4j:2026.05-community
```
---

If that fails, browse available tags on Docker Hub (`https://hub.docker.com/_/neo4j`) and update the `image:` line in `docker-compose.yml`. Match the Neo4j **driver 6.x** used by the app (`REQUIREMENTS.md` §2.3).

> **Vector + full-text indexes are native** to Neo4j — **no APOC or GDS plugins are required** for this pipeline.

## 4. Authentication

- The **initial** username is always `neo4j`. The initial password is set **once** at first startup via `NEO4J_AUTH=neo4j/<password>` (wired to `${NEO4J_PASSWORD}` from `.env`).
- `NEO4J_AUTH` only applies on an **empty data volume** (first init). Changing `.env` later does **not** change an existing password.
- **To change the password after first init**, either:
  - In Browser / cypher-shell: `ALTER CURRENT USER SET PASSWORD FROM '<old>' TO '<new>';`, or
  - Reset by wiping data (dev only): `docker compose down -v` then `docker compose up -d` (destroys the graph).
- To **disable** auth (NOT recommended; local trusted dev only): set `NEO4J_AUTH=none`.

Keep the password only in `.env` (git-ignored, NFR-SEC-1). Never hard-code it (NFR-SEC-2).

## 5. Configuration via environment variables

Any Neo4j config setting can be passed as an env var using this **name-mangling rule**:

> `server.memory.heap.max_size` → `NEO4J_server_memory_heap_max__size`
> (prefix `NEO4J_`, replace `.` → `_`, replace existing `_` → `__`)

Common, useful settings (add under `environment:` in the compose service):

---
```yaml
    environment:
      NEO4J_AUTH: "neo4j/${NEO4J_PASSWORD:?set NEO4J_PASSWORD in .env}"
      # --- Memory (tune to your machine; these are reasonable laptop defaults) ---
      NEO4J_server_memory_heap_initial__size: "1G"
      NEO4J_server_memory_heap_max__size: "2G"
      NEO4J_server_memory_pagecache_size: "1G"
      # --- Query guardrail (optional) ---
      NEO4J_db_transaction_timeout: "120s"
```
---

Notes:
- For a 16 GB laptop also running Docling/BGE on the GPU, **heap 2 GB + pagecache 1 GB** is plenty for this dataset size (D3). Increase pagecache if the graph grows large.
- Leaving memory settings unset lets Neo4j auto-size from available RAM — fine for getting started.

## 6. Volumes & data persistence

The compose file mounts **named volumes** so data survives `docker compose down` and container/Docker restarts (NFR-REL-6):

| Volume | Mount | Purpose |
|--------|-------|---------|
| `neo4j_data` | `/data` | Graph store, indexes, auth — **the database** |
| `neo4j_logs` | `/logs` | Server + query logs |

- Inspect: `docker volume ls` and `docker volume inspect rag-pipeline_neo4j_data`.
- **`docker compose down`** keeps volumes (data preserved).
- **`docker compose down -v`** deletes volumes (**destroys the graph** — use only to start clean).

Optional extra mounts you may add later: `- ./import:/import` (for `LOAD CSV`) and `- ./plugins:/plugins` (not needed here).

## 7. Manual `docker run` alternative (without compose)

If you prefer a one-off container instead of compose:

---
```powershell
docker run -d `
  --name rag-neo4j `
  --restart unless-stopped `
  -p 127.0.0.1:7474:7474 `
  -p 127.0.0.1:7687:7687 `
  -e NEO4J_AUTH=neo4j/JaiSaiNath@519* `
  -e NEO4J_server_memory_heap_max__size=2G `
  -e NEO4J_server_memory_pagecache_size=1G `
  -v neo4j_data:/data `
  -v neo4j_logs:/logs `
  neo4j:2026.05-community
```
---

(The backtick `` ` `` is the PowerShell line-continuation character.)

## 8. Creating the required constraints & indexes

The app **bootstrap** creates these idempotently (FR-S0.5 / FR-7.10). To do it manually, open cypher-shell:

---
```powershell
docker exec -it rag-neo4j cypher-shell -u neo4j -p JaiSaiNath@519*
```
---

Then run:

---
```cypher
// 1. Uniqueness constraint — canonical Document id = SHA-256 (FR-7.2)
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

// 2. Vector index — 1536-dim, cosine, on Chunk.embedding (FR-7.7 / FR-Q2.4)
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
} };

// 3. Full-text (Lucene/BM25) index on Chunk.text (FR-7.6 / FR-Q2.3)
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text];

// Verify
SHOW INDEXES;
SHOW CONSTRAINTS;
```
---

> The vector dimension (1536) and similarity (`cosine`) **must** match the embedding model on both ingest and query (`REQUIREMENTS.md` FR-Q0.5 / NFR-REL-8). Changing the embedding model later requires dropping and recreating this index (A5/RISK-G).

## 9. Verifying connectivity

**A. cypher-shell (inside the container):**
---
```powershell
docker exec -it rag-neo4j cypher-shell -u neo4j -p your-strong-password "RETURN 1 AS ok;"
```
---

**B. Browser:** open http://127.0.0.1:7474, log in, run `:server status`.

**C. From Python (the driver the app uses):**
---
```python
from neo4j import GraphDatabase
import os
drv = GraphDatabase.driver(os.environ["NEO4J_URI"],
                           auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
drv.verify_connectivity()
print("Neo4j OK")
drv.close()
```
---

## 10. Security hardening
- **Localhost-only ports** — the compose file binds `127.0.0.1:7474` and `127.0.0.1:7687`, so Neo4j is not reachable from the network (NFR-SEC-5). Do **not** change these to `0.0.0.0` unless you intend to expose it.
- **Change the default password** — `NEO4J_AUTH` sets a non-default password at init; never ship `neo4j/neo4j`.
- **Keep credentials in `.env`** (git-ignored) — never in source or logs (NFR-SEC-1/2/7).
- **No public exposure** — this is a single-user local DB; there is no TLS/cert setup here because traffic never leaves the loopback interface.

## 11. Backup & restore (Community)

Community Edition has no online backup; use one of these for a single-user setup:

**Option A — offline dump (clean, portable):**
---
```powershell
docker compose stop neo4j
docker exec rag-neo4j neo4j-admin database dump neo4j --to-path=/data/backups
docker compose start neo4j
# the dump file lives inside the neo4j_data volume under /data/backups
```
---
Restore with `neo4j-admin database load neo4j --from-path=/data/backups --overwrite-destination=true` (container stopped).

**Option B — volume snapshot (simplest):** stop the stack and copy the `neo4j_data` volume contents to a tar via a helper container, or rely on the named volume persisting across restarts for day-to-day use.

> Re-ingestion is not supported (FR-1.5), so a backup is the way to preserve ingested documents before risky changes.

## 12. Troubleshooting (Windows)

| Symptom | Likely cause / fix |
|---------|--------------------|
| `docker compose up` errors on `NEO4J_PASSWORD` | `.env` missing or `NEO4J_PASSWORD` unset. The `${VAR:?...}` guard fails fast on purpose — set it in `.env`. |
| Browser login fails after changing `.env` | `NEO4J_AUTH` only applies on first init. Either change password via Cypher (§4) or `docker compose down -v` to reset (destroys data). |
| Port 7687/7474 already in use | Another Neo4j/process bound the port. Find it: `netstat -ano \| findstr 7687`; stop it or remap the host port in compose (e.g. `127.0.0.1:7688:7687`). |
| Container exits / restarts repeatedly | Check `docker compose logs neo4j`. Common: too-high heap vs available RAM — lower `NEO4J_server_memory_heap_max__size`. |
| `Neo.ClientError…IndexAlreadyExists` | Harmless if you re-ran index DDL without `IF NOT EXISTS`; the bootstrap uses `IF NOT EXISTS`. |
| Vector index creation rejected | Ensure the image is a current 2026.x release; verify the exact `OPTIONS { indexConfig: { … } }` syntax in §8. |
| Slow first query after start | Page cache warming; normal. |
| WSL 2 / Docker not responding | Restart Docker Desktop; ensure WSL 2 backend is enabled (`wsl --status`). |

Useful commands:
---
```powershell
docker compose ps                 # status
docker compose logs -f neo4j      # live logs
docker compose restart neo4j      # restart
docker compose down               # stop (keep data)
docker compose down -v            # stop + DELETE data
docker exec -it rag-neo4j bash    # shell into the container
```
---

## 13. Upgrading

1. Back up first (§11).
2. Update the `image:` tag in `docker-compose.yml` to the newer `YYYY.MM-community` line.
3. `docker compose pull neo4j` then `docker compose up -d neo4j`.
4. Within a calendar-version series, data is forward-compatible; across larger jumps, consult the Neo4j upgrade guide.

## 14. References
- [Getting started with Neo4j in Docker — Operations Manual](https://neo4j.com/docs/operations-manual/current/docker/introduction/)
- [Modify the default configuration (Docker)](https://neo4j.com/docs/operations-manual/current/docker/configuration/)
- [Neo4j official Docker image — Docker Hub](https://hub.docker.com/_/neo4j)
- [Calendar Versioning & Cypher 25 announcement](https://feedback.neo4j.com/changelog/important-update-calendar-versioning-cypher-25)
- [Vector indexes — Cypher Manual](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Full-text indexes — Cypher Manual](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/)
