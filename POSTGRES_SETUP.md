# Postgres (Native, On-Prem) — Installation & Configuration Guide

**Scope:** Install and configure **PostgreSQL natively on Windows 11** to serve as the **LangGraph checkpoint store** for this RAG pipeline (decision D15). The checkpointer persists full run state per `thread_id`, enabling run tracking and resume after human-in-the-loop interrupts (tag review, clarification, escalation).
**Companion files:** `.env` (`CHECKPOINT_DB_URI`), `REQUIREMENTS.md` (§2.1 topology, §2.3 versions, C6, FR-0.3/FR-5.2/FR-Q0.4 persistence, FR-S0.5a table bootstrap). Neo4j has its own guide: `NEO4J_SETUP.md`.

> **Why native (not Docker)?** Per decision D15/C6, Postgres runs as a **native Windows service on the laptop** (localhost:5432). Neo4j remains Dockerized. Because Postgres stays on `127.0.0.1`, the on-device privacy posture is unchanged (no new egress — NFR-SEC).

> **No `pgvector` needed.** All embedding vectors live in Neo4j; Postgres holds only LangGraph checkpoint tables. A stock Postgres install is sufficient.

---

## 1. Prerequisites
- Windows 11 (the target host).
- Administrator rights to install software and run a Windows service.
- ~1 GB free disk for Postgres + the (small) checkpoint data.
- This project checked out with a populated **`.env`** (copied from `.env.example`).

---

## 2. Install PostgreSQL on Windows

Use the **EDB PostgreSQL installer** (the standard Windows distribution). Target **Postgres 17 or 18** (14+ is the minimum, per `REQUIREMENTS.md` §2.3).

### Option A — Interactive installer (recommended)
1. Download the Windows installer for PostgreSQL **18** (or 17) from the EDB downloads page.
2. Run it as Administrator. In the wizard:
   - **Installation directory:** default (`C:\Program Files\PostgreSQL\18`) is fine.
   - **Components:** keep *PostgreSQL Server* and *Command Line Tools* (which includes `psql`). **pgAdmin 4** is optional. **Stack Builder** can be skipped.
   - **Data directory:** default is fine.
   - **Superuser password:** set a strong password for the `postgres` superuser — record it; you'll need it once to create the app role.
   - **Port:** **5432** (default — must match `CHECKPOINT_DB_URI`).
   - **Locale:** default.
3. Finish. The installer registers and starts a Windows service named like **`postgresql-x64-18`**.

### Option B — winget (silent)
```powershell
winget install --id PostgreSQL.PostgreSQL.18 -e
```
(You may still need to set the superuser password / verify the service depending on the package defaults.)

### Add `psql` to PATH (so you can call it from any shell)
The client tools live in e.g. `C:\Program Files\PostgreSQL\18\bin`. Add that to your PATH, or call `psql` with its full path. Quick check:
```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" --version
```

### Confirm the service is running
```powershell
Get-Service postgresql*
# If stopped:
Start-Service postgresql-x64-18
```

---

## 3. Create the application role & database

Create a dedicated least-privilege role and database for the checkpointer (do **not** use the `postgres` superuser for the app). Connect as the superuser once:

```powershell
psql -U postgres
```

Then run (replace the password with the one you'll put in `.env`):

```sql
-- Dedicated login role for the app
CREATE ROLE langgraph WITH LOGIN PASSWORD 'your-strong-password';

-- Dedicated database owned by that role
CREATE DATABASE langgraph OWNER langgraph;

-- Connect into the new DB and ensure the role can create objects in `public`.
-- (On Postgres 15+, the database OWNER already has rights on the public schema
--  via pg_database_owner; this GRANT is an explicit safety net.)
\c langgraph
GRANT ALL ON SCHEMA public TO langgraph;

\q
```

The `langgraph` role needs **CREATE** in the database/schema because the application bootstrap calls the LangGraph saver's `.setup()` to create the checkpointer tables (FR-S0.5a). Owning the database satisfies this.

---

## 4. Wire `CHECKPOINT_DB_URI` in `.env`

The application reads a single connection URI. In `.env`, keep these consistent:

```dotenv
POSTGRES_USER=langgraph
POSTGRES_PASSWORD=your-strong-password
POSTGRES_DB=langgraph
CHECKPOINT_DB_URI=postgresql://langgraph:your-strong-password@127.0.0.1:5432/langgraph
```

- The URI **must** match the role/password/db you created in §3.
- **Special characters in the password must be URL-encoded** in the URI (e.g. `@` → `%40`, `:` → `%3A`, `/` → `%2F`, `#` → `%23`). The simplest path is to use a password without URI-reserved characters.
- Host stays `127.0.0.1` (localhost-only; NFR-SEC-5). Port `5432` matches the install.
- `.env` is git-ignored (NFR-SEC-1); never commit real credentials.

---

## 5. Verify connectivity

**A. psql via the full URI (what the app uses):**
```powershell
psql "postgresql://langgraph:your-strong-password@127.0.0.1:5432/langgraph" -c "SELECT version();"
```

**B. psql via discrete flags (will prompt for password):**
```powershell
psql -h 127.0.0.1 -p 5432 -U langgraph -d langgraph -c "SELECT 1 AS ok;"
```

**C. From Python (the driver the app uses, `psycopg`):**
```python
import os, psycopg
with psycopg.connect(os.environ["CHECKPOINT_DB_URI"]) as conn:
    print(conn.execute("SELECT 1").fetchone())   # -> (1,)
print("Postgres OK")
```

---

## 6. Checkpointer tables (created by the bootstrap)

You do **not** create these by hand — the application **bootstrap** (`REQUIREMENTS.md` §2.4 / FR-S0.5a) calls the LangGraph Postgres saver's `.setup()`, which creates them idempotently. For reference, the equivalent of what runs:

```python
from langgraph.checkpoint.postgres import PostgresSaver
import os

with PostgresSaver.from_conn_string(os.environ["CHECKPOINT_DB_URI"]) as cp:
    cp.setup()    # creates checkpoint tables if absent (safe to re-run)
```

After bootstrap, confirm the tables exist:
```powershell
psql "$env:CHECKPOINT_DB_URI" -c "\dt"
```
You should see the LangGraph checkpoint tables (e.g. `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, plus a migrations table). Inspect a run later with:
```sql
SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id;
```

---

## 7. Configuration & service management (optional)

Defaults are fine for this single-user checkpoint workload (D3). If you want to tune or lock down:

- **Config files** live in the data directory, e.g. `C:\Program Files\PostgreSQL\18\data\`:
  - `postgresql.conf` — server settings.
  - `pg_hba.conf` — client authentication rules.
- **Bind to localhost only** (defense in depth): in `postgresql.conf` set
  ```
  listen_addresses = 'localhost'
  ```
  then restart the service.
- **Service control (PowerShell):**
  ```powershell
  Get-Service postgresql-x64-18
  Restart-Service postgresql-x64-18
  Stop-Service postgresql-x64-18
  Start-Service postgresql-x64-18
  ```
  (Or use `services.msc`.) Service is set to start automatically by the installer.

---

## 8. Security hardening
- **Least privilege** — the app uses the `langgraph` role, **not** `postgres` (§3).
- **Localhost only** — keep `listen_addresses = 'localhost'`; do not expose 5432 to the network (NFR-SEC-5).
- **Strong auth** — modern installers default `pg_hba.conf` to `scram-sha-256`. Keep it; avoid `trust`.
- **Secrets in `.env` only** — never in source, logs, or the URI committed to git (NFR-SEC-1/2/7).
- **Firewall** — no inbound rule is needed since connections are loopback-only.

---

## 9. Backup & restore

The checkpoint store is small; standard logical backups suffice:

```powershell
# Backup the langgraph database to a file
pg_dump "postgresql://langgraph:your-strong-password@127.0.0.1:5432/langgraph" -Fc -f langgraph_checkpoints.dump

# Restore (into an existing empty DB)
pg_restore -h 127.0.0.1 -U langgraph -d langgraph --clean --if-exists langgraph_checkpoints.dump
```

> Checkpoints are run/resume state, not the knowledge graph. The durable corpus lives in Neo4j (`NEO4J_SETUP.md` §11). Losing checkpoints only loses in-flight/suspended runs, not ingested documents.

---

## 10. Troubleshooting (Windows)

| Symptom | Likely cause / fix |
|---------|--------------------|
| `psql: could not connect to server` / connection refused | Service not running. `Get-Service postgresql*` then `Start-Service postgresql-x64-18`. |
| `password authentication failed for user "langgraph"` | Wrong password, or `.env` URI out of sync with the role. Recheck §3/§4; reset via `ALTER ROLE langgraph WITH PASSWORD '...';` as `postgres`. |
| App connects but `.setup()` fails with `permission denied for schema public` | Role lacks CREATE on `public` (Postgres 15+). Run `GRANT ALL ON SCHEMA public TO langgraph;` in the `langgraph` DB (§3), or ensure `langgraph` owns the DB. |
| `database "langgraph" does not exist` | DB not created or name mismatch. Recreate per §3; ensure `POSTGRES_DB` / URI path agree. |
| URI auth fails but discrete-flag login works | Unencoded special character in the password inside `CHECKPOINT_DB_URI`. URL-encode it (§4) or choose a simpler password. |
| Port 5432 already in use / second Postgres | Another Postgres/process owns 5432. Find it: `netstat -ano \| findstr 5432`; stop the other instance, or install on a different port and update the URI. |
| `psql` not recognized | Client tools not on PATH. Use the full path `C:\Program Files\PostgreSQL\18\bin\psql.exe` or add `bin` to PATH (§2). |
| Service won't start after a crash | Check the Postgres log in `…\18\data\log\`; common causes are a stale `postmaster.pid` or disk-full. |

Useful one-liners:
```powershell
Get-Service postgresql*                                  # service status
psql "$env:CHECKPOINT_DB_URI" -c "\conninfo"             # who/where am I connected as
psql "$env:CHECKPOINT_DB_URI" -c "\dt"                   # list checkpoint tables
```

---

## 11. Uninstall / reset (dev only)
- **Drop just the app data** (keep the server): as `postgres`,
  ```sql
  DROP DATABASE langgraph;
  DROP ROLE langgraph;
  ```
  then recreate per §3 for a clean checkpoint store.
- **Remove Postgres entirely:** uninstall via *Apps & features* (or `winget uninstall`), which removes the service. Delete the data directory only if you also want to discard all databases.

---

## 12. References
- [PostgreSQL Downloads (Windows / EDB installer)](https://www.postgresql.org/download/windows/)
- [PostgreSQL Documentation — Server Administration](https://www.postgresql.org/docs/current/admin.html)
- [psql reference](https://www.postgresql.org/docs/current/app-psql.html)
- [Connection URIs (libpq)](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
- [LangGraph — Postgres checkpointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)
