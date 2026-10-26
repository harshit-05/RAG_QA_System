# S3-8: API keys, rate limits and a stateless query tier

| | |
| --- | --- |
| **Status** | In progress. Plan approved 2026-10-26 (below, "Plan"); on Opus 5.5 at max effort (Deviation). |
| **Closes** | NFR-5; DEC-23; backlog: multi-user auth and rate limiting (API deployment hardening), request size and health-check load, statelessness |
| **Depends on** | S3-7b (S3-7 was split in its plan, 2026-10-23) |
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
    with mode 0600, under a lock, so two `rag-keys` at once lose nothing. It is read again
    when `(st_ino, st_mtime_ns, st_size)` changes, checked at most every 5 s, and never
    holds a token.
  - **Where it lives** (DEC-23, design review 2): in `var/` (`var/api-keys.json`), which
    compose mounts as a folder. A file bind-mounted on its own keeps its old inode when the
    host replaces it atomically, so a revocation would never reach the container (checked).
  - **A file that does not parse fails closed:** its keys get a 503 and health says why,
    until it parses again. `RAG_API_TOKEN` still works.
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
    variables, S3-7b's spans).
  - **"No credential" means neither a token nor a keys file** (DEC-23, design review 2).
    Only then does `rag-serve` refuse a non-loopback host, and does `check_host` refuse a
    non-loopback `Host`. Both test `settings.token is None` today (`routes.py`). So with a
    keys file alone, the Host check would refuse every request through Caddy under a public
    name, and Caddy's own health probe (`Host: app:8000`). A test sends a keys-file request
    with a public `Host`.
- **Rate limits** (`api/limits.py`): token buckets of our own, in memory, per replica.
  - **Per key:** `RAG_API_RATE_LIMIT` requests a minute (default 60).
  - **Per client address,** for requests without a key (a loopback server with none
    configured).
  - **A strict per-address bucket for failed authentication,** 10 a minute, which slows
    token guessing. It gates only requests that fail authentication: a valid key is never
    refused because someone else failed from the same address (design review 2).
  - **The health routes are exempt** (DEC-23, design review 1): Docker's and Caddy's checks
    poll them from one address, and the 5 s probe cache bounds their cost.
  - **The tables are bounded** (DEC-23, design review 2): at most 10,000 buckets each, full
    ones evicted first, then the least recently used, and an IPv6 client counted by its /64.
    Unbounded, one /64 grew them by 36 MB per 200,000 addresses.

  The answer is 429 with `Retry-After`, and this story adds S3-7's counter for it.
- **The client's address** (DEC-23, design review 2). uvicorn reads `X-Forwarded-For` by
  default and trusts it from loopback, so today any local client can name its own address,
  and get a fresh bucket with every request. `rag-serve` therefore passes
  `forwarded_allow_ips` itself: `RAG_API_TRUSTED_PROXIES`, or no one when it is unset. In the
  stack that is Caddy's one fixed address, never the network's range: a published port's
  connections arrive from the network's gateway, inside that range. A test sends a spoofed
  `X-Forwarded-For` from an untrusted peer, and the limits see its real address.
- **Body size:** a pure ASGI middleware that answers 413 when `Content-Length` exceeds
  64 KiB, or when a streamed body does, before FastAPI reads or parses anything. It runs
  first, ahead of every dependency, which closes S2-7's pre-authentication gap. A test sends
  a large body with no token and gets 413, not 401 or 422.
  - **A streamed body raises `HTTPException(413)`** from the `receive` the middleware wraps
    (DEC-23, design review 2). FastAPI turns any other exception raised while it reads the
    body into a 400, "There was an error parsing the body" (`fastapi/routing.py`). So the
    test sends the body twice: once with `Content-Length`, once chunked without it.
  - **The wrapper passes `http.disconnect` through untouched,** and the real-uvicorn
    disconnect test runs with the middleware on.
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
  - **In this order** (DEC-23, design review 2): read the row, probe the job lock, and when
    it is free, read the row again; only a row still unfinished is abandoned. Without the
    second read, a job that finished between the two steps read as abandoned in 92 of 400
    runs of a simulation, and in none with it. A verdict written back uses `UPDATE … WHERE
    status IN ('queued', 'running')`, and the start transaction applies the same rule, so a
    dead job's row never holds the 409.
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
#   a 1 MB body without a token: 413, with Content-Length and chunked without it
#   a spoofed X-Forwarded-For: the limits count the real address (design review 2)
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
  the row exists, and a job that finishes writes its row before it releases the lock. The
  row is read again after the probe (design review 2).
- **The address the limits count** (design review 2): `forwarded_allow_ips` is always passed,
  and never as a range.

## Plan (approved 2026-10-26)

Approved by the maintainer, with both open questions taken as recommended (Q1, Q2). This
section is the record to cross-check the work against; the full plan, with the files, the tests
and the recipes, is `~/.claude/plans/yup-proceed-make-the-wobbly-finch.md`. A departure from it
goes under "Deviation from plan".

**What the plan found, checking the design against the libraries as installed** (uvicorn
0.54.0, FastAPI 0.143.0, Starlette 1.7.0, opentelemetry-instrumentation-fastapi 0.66b1):

- **uvicorn trusts `X-Forwarded-For` from loopback unless told otherwise.** `Config` reads
  `FORWARDED_ALLOW_IPS`, defaulting to `127.0.0.1,::1`. Passing `forwarded_allow_ips=[]` trusts
  nobody (`_TrustedHosts([])` trusts neither loopback nor `172.18.0.10`).
- **FastAPI re-raises an `HTTPException` raised while it reads a body,** and turns any other
  exception into the 400 "There was an error parsing the body" (`fastapi/routing.py`).
- **The contrib instrumentation wraps the outermost layer** (it patches
  `build_middleware_stack`), so our middleware sits inside it and inside
  `ServerErrorMiddleware`: an unhandled exception passes through ours before it becomes a 500.
- **The hermetic suite can run two replicas:** each test has a `var/` of its own and one
  in-memory Qdrant client, which every store it builds shares.
- **No dependency is needed:** sqlite3, hmac, secrets and fcntl are the standard library, and
  SQLite 3.50.4 has `json_insert`.

| # | Decision |
| --- | --- |
| D1 | **The keys file** (`api/auth.py`): `{"version": 1, "keys": [{name, scopes, sha256, created}]}`. Names `[a-z0-9][a-z0-9_-]{0,63}`, `token` reserved; scopes a non-empty subset of `query` and `ingest`. Re-read when `(st_ino, st_mtime_ns, st_size)` moves, checked at most every 5 s. A missing, unreadable or malformed file fails closed: keyed requests get 503 with `Retry-After: 5` and fixed text, never the path. `RAG_API_TOKEN` keeps working, as the key `token` with both scopes. |
| D2 | **The check:** `Bearer <token>`, parsed as today; the sha256 of the presented bytes compared with every stored digest by `hmac.compare_digest`, on bytes; the same 401 for every failure. While the file is broken, a presentation that is not the token draws on the failed-auth bucket too, so wrong guesses cannot probe the token unthrottled. |
| D3 | **Scopes:** an async `authenticate` dependency on the `v1` and `docs` routers, then one scope per route. A valid key without it gets 403. The key's name is bound into the logs (`api_key`) and set on the server span (`rag.api_key.name`), never the token; the job thread binds it too. |
| D4 | **"No credential"** is no token and no keys file (`ServerSettings.has_credential`); only then are a non-loopback `Host` and `--host` refused. A broken keys file at start-up is a warning; **readiness answers 503 while it is broken** (Q1), with a `keys` entry. |
| D5 | **`rag-keys`** `add NAME --scope …` (the token alone on stdout, once), `list` (never hashes), `revoke NAME`; `--file` defaults to `$RAG_API_KEYS_FILE`. Each change under a `flock` on `<file>.lock`, written to a 0600 temporary file, fsynced, `os.replace`d, the folder fsynced. Exit 0 done, 1 a name taken or unknown, 2 cannot run. |
| D6 | **Rate limits** (`api/limits.py`): token buckets per key (`RAG_API_RATE_LIMIT`, 60 a minute), per address without a configured credential, and per address for failed authentication (10 a minute, drawn only after a token fails). At most 10,000 buckets a table: the first full one from the least recently used end is evicted, else the least recently used; IPv6 by its /64. Health exempt; 429 with `Retry-After`. Order: Origin 403, Host 403, authentication 401/429/503, the key's bucket 429, scope 403, then the query route's own refusals. |
| D7 | **The client's address:** `rag-serve` always passes `forwarded_allow_ips`, `RAG_API_TRUSTED_PROXIES` or `[]`, from one config builder the real-uvicorn tests share. Addresses only: a range, `*` or a name exits 2. |
| D8 | **Body size:** a pure ASGI middleware, 64 KiB: by `Content-Length` before the app runs, or `HTTPException(413)` from the wrapped `receive` as a body streams; `http.disconnect` passed through untouched. |
| D9 | **The availability SLI** (the backlog line): an unexpected 500 on `POST /v1/query` counts as `rag_refusals_total{code="500",reason="error"}`, a broken keys file as `503 keys_unreadable`; the SLI reads `code=~"5.."`. The 401, 403, 413 and 429 refusals count on neither side: the client's, or policy. Every new label set starts at 0. |
| D10 | **Health load:** the readiness probes of Qdrant's alias and of Ollama, cached for 5 s and single-flight; a question still reads the alias on every request. |
| D11 | **Durable jobs** in `var/jobs.sqlite3`, WAL: the job lock `var/jobs/<id>.lock` taken before a `BEGIN IMMEDIATE` start that checks and inserts; the row "running" from `ingest()`'s new `on_locked` callback; the final row written before the lock is released. Read the row, probe the lock without creating it, read again; only a row still unfinished is abandoned, written back `WHERE status IN ('queued', 'running')`. The last 100 kept; `GET /v1/ingest/{id}` gains `abandoned`. |
| D12 | **Docs:** the README's Access, Metrics and Alerts; DEC-23's as-built note; the records. |
| D13 | **Breaking** (`feat(api)!`): every route but health is rate-limited, loopback without a credential included; a body over 64 KiB gets 413; `X-Forwarded-For` is no longer trusted from loopback, and `FORWARDED_ALLOW_IPS` is ignored. |

**The backlog lines that touch S3-8:** the availability SLI's unseen 500s is folded in (D9).
The name-based no-secrets rule (the config schema, outside this story's scope) and the Qdrant
URL in a job's error text (which waits on S3-9's internal network) stay, led by S3-9 (Q2). The
stale `stream_options` assertion stays: this story does not run `test_fallback_openai.py`.

**Nothing hashed moves:** no edit to the config, the prompt, a model, a retriever, the corpus or
the golden questions, so `rag-eval check` stays 0 and the contexts IDENTICAL to S3-2's.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
