# S3-8: API keys, rate limits and a stateless query tier

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | NFR-5; DEC-23; backlog: multi-user auth and rate limiting (API deployment hardening), request size and health-check load, statelessness |
| **Depends on** | S3-7 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-8-api-hardening`. Risky: authentication, and the job records several replicas share. |

## Goal

The API stops being localhost-first.

- **Several callers,** each with a named, scoped key the maintainer can revoke (the
  maintainer's choice).
- **Every caller is rate-limited,** and a request body is bounded before anything parses it.
- **Health probes cost almost nothing.**
- **Any replica can answer any request:** the index is in Qdrant since S3-2, and this story
  moves the ingest job records out of memory, which makes the query tier stateless (NFR-5).

Design: ARCHITECTURE.md §3.1, DEC-23. TLS is S3-9's, at the proxy.

## Scope

- **Keys** (`api/auth.py`, `api/keys_cli.py`, the entry point `rag-keys`).
  - **The file,** named by `RAG_API_KEYS_FILE`, is JSON: per key, `name`, `scopes` (some of
    `query` and `ingest`), the token's `sha256`, and `created`.
  - **Writing and reading it:** written atomically (a temporary file, then `os.replace`)
    with mode 0600; read again when it changes, checked at most every 5 s. It never holds a
    token.
  - **The command:**
    - `rag-keys add NAME --scope query[,ingest]` makes a 32-byte token with `secrets`, prints
      it once and stores its hash;
    - `rag-keys list` shows names, scopes and dates, never hashes;
    - `rag-keys revoke NAME`.
  - **The check:** hash the presented bearer token, then `hmac.compare_digest` it against
    each stored hash, on bytes, so every failure looks alike.
  - **`RAG_API_TOKEN`** still works, as a key named `token` with both scopes; a keys file and
    the token may coexist.
  - **Scopes:**
    - `query` for `POST /v1/query` and `GET /v1/documents`;
    - `ingest` for `POST /v1/ingest` and `GET /v1/ingest/{job_id}`;
    - any valid key for the docs and the schema;
    - health stays open.

    A valid key without the scope gets 403. The test that sends every route a request
    without a token grows a case for every route without its scope.
  - **The key's name** (never the token) is bound into logs and spans (S3-7's context
    variables).
  - **`rag-serve`'s rule is unchanged:** it refuses a non-loopback host unless a key or a
    token is configured.
- **Rate limits** (`api/limits.py`): token buckets of our own, in memory, per replica.
  - **Per key:** `RAG_API_RATE_LIMIT` requests a minute (default 60).
  - **Per client address,** for requests without a key (a loopback server with none
    configured).
  - **A strict per-address bucket for failed authentication,** 10 a minute, which slows
    token guessing.
  - **The health routes are exempt** (DEC-23, design review 1): Docker's and Caddy's checks
    poll them from one address, and the 5 s probe cache bounds their cost.

  The answer is 429 with `Retry-After`, and this story adds S3-7's counter for it. Behind a
  proxy, the client address comes from `X-Forwarded-For`, trusted only from the addresses in
  `RAG_API_TRUSTED_PROXIES`, passed to uvicorn's `forwarded_allow_ips` (which takes CIDR
  ranges in 0.54.0, so a compose network's range works).
- **Body size:** a pure ASGI middleware that answers 413 when `Content-Length` exceeds
  64 KiB, or when a streamed body does, before FastAPI reads or parses anything. It runs
  first, ahead of every dependency, which closes S2-7's pre-authentication gap. A test sends
  a large body with no token and gets 413, not 401 or 422.
- **Health load:** each dependency's probe (Ollama, Qdrant) is cached for 5 s and runs
  single-flight, so a flood of probes makes at most one call per dependency per 5 s. The
  breakers (S3-6) are reported beside them.
- **Durable job records** (`api/jobs.py`): SQLite, from the standard library, at
  `var/jobs.sqlite3`, in WAL mode.
  - **One job at a time across replicas:** a `BEGIN IMMEDIATE` transaction checks for a
    running job and inserts, so the 409 holds whichever replica answers.
  - **The job's thread** still runs `ingest()`, which holds the writer `flock`. Its row
    records the status, the times, the exit code, the report and the log as JSON.
  - **An abandoned job is told by a lock of its own** (DEC-23, design review 1). The replica
    takes a `flock` on `var/jobs/<id>.lock` before it inserts the row, and holds it until the
    row's final status is written; the row reads "running" only once `ingest()` holds the
    writer lock (a callback). An unfinished row whose job lock is free belongs to a process
    that died: it reads as `failed`, with `abandoned: true`. The kernel releases a `flock`
    with its process, and DEC-17/DEC-19's recovery cleans up what the run left. Never probe
    the writer lock: a writer starting at that moment would exit 2.
  - The last 100 jobs are kept.
- **A stateless query tier** (NFR-5). Check that nothing a request depends on lives only in
  one replica:
  - the served generation follows the alias (S3-2);
  - jobs live in SQLite;
  - keys live in the file;
  - what a replica caches (the generation and its manifest, probe results, rate buckets) can
    be rebuilt.

  **A test with two app instances** over one store and one job database shows:
  - a job started on one is visible, and refused with 409, on the other;
  - an ingest published by one is served by the other at its next request.
- **README:** keys (`rag-keys`), scopes, limits, the 413, 429 and 403 statuses, and the
  documented limits of NFR-5 (ingestion on one host; slots and rate limits per replica; one
  Ollama queues every replica's answers). S3-9 points `rag-serve`'s cleartext warning at its
  TLS recipe, once that exists.

## Out of scope

- TLS, the proxy and running two replicas behind it (S3-9).
- OIDC or JWT (DEC-23 rejected it).

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-8/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-8/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

export RAG_API_KEYS_FILE=$PWD/var/api-keys.json
uv run rag-keys add laptop --scope query,ingest            # the token, once
uv run rag-keys add bot --scope query
uv run rag-keys list                                       # names and scopes only
# rag-serve in a foreground terminal (tee -i), then:
#   bot's token: POST /v1/query 200; POST /v1/ingest 403
#   no token: 401; a wrong token 11 times within a minute: the 11th is 429
#   a 1 MB body without a token: 413
#   200 GETs of /v1/health in 2 s: every one answered, none 429; Ollama asked at most once
#     per 5 s (its log)
#   kill -9 rag-serve during an ingest job, start it again: the job reads failed, abandoned
uv run rag-keys revoke bot                                 # bot's token: 401 within 5 s
gh pr checks
```

## Review notes for the human

- **`auth.py`:** constant-time comparison on hashes, the same answer for every failure, no
  token in any log line, and the keys file's mode.
- **The middleware's order:** the size check must run before FastAPI parses the body.
- **The job database's transaction,** and the abandoned-job rule: the job lock is taken before
  the row exists, and a job that finishes writes its row before it releases the lock.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
