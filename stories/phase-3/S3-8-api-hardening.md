# S3-8: API keys, rate limits and a stateless query tier

| | |
| --- | --- |
| **Status** | Done 2026-10-26 (PR #28, awaiting its merge). Plan approved 2026-10-26 (below, "Plan"); on Opus 5.5 at max effort (Deviation). Built and committed on 2026-10-26 (`115e85a` … `2aef8d7`, then the live run's two fixes and the close-out record, `646e888`). The first review (2026-10-27, on `646e888`) found one defect, a refused read of the keys file that held every key closed, and three smaller things; the second (2026-10-27, on `bc2d892`) found no blocker and five things, among them a crash mid-ingest counted only once a client asked. The fixes are on the PR (`4a1a820`, `5caa9a7`). CI is green on `a6ed789`, the second review's record (run 40471130129: `check`, `deploy-config`, `eval-retrieval`). |
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

## Verification record (2026-10-26)

Rows 1 and 4–7 ran on the final tree, the live run's fixes included. Rows 2, 3 and 8–18 ran
in the maintainer's terminal on `2aef8d7`, logged with `tee -i` under `~/dev/rag-s3-8/`
(`retrieval.log`, `serve-1.log`, `serve-2.log`, `contexts.json`), and Ollama's side in its
journal. The client's own output (the `curl` lines, `rag-keys`) was not captured; each row
below is read from the server's access log, which records every status and the key's name.
The live run used a scratch index, `RAG_QDRANT_COLLECTION=s3_8_live`, deleted afterwards.

| # | Command | Result |
| --- | --- | --- |
| 1 | `uv run rag-eval check` | exit 0: the six parts `ok`, every floor met; "The committed run matches this checkout" |
| 2 | `RAG_LOG_FORMAT=console uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-8/contexts.json` | 0.950 (19 of 20) / 0.796 / 0.858, every floor met; 25 questions' contexts written |
| 3 | `diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-8/contexts.json` | IDENTICAL; both `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc` |
| 4 | `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q` | 1711 passed, 18 skipped, 5 xfailed; coverage 98.28% |
| 5 | `uv run ruff check && uv run ruff format --check && uv run mypy` | all checks passed; 85 files formatted; no issues in 41 source files |
| 6 | the lock: `uv lock --check`; `git diff main -- uv.lock` | resolves 194; no line differs from `main`'s: no dependency added, none moved, pypdf 6.19.0. `rag-keys` is an entry point only |
| 7 | `promtool test rules` and `check config`, CI's pinned Prometheus image | SUCCESS on both rule-test files; 41 rules. Two mutants of the SLI's selector (503 alone, every code) each failed a scenario |
| 8 | `rag-keys add laptop --scope query,ingest`, `add bot --scope query`, `list` | the server read "2 keys" from `var/api-keys.json`; no token in any log line (`grep -c 'token\|Bearer'` 0 in both logs) |
| 9 | `rag-serve` with the keys file (`serve-1.log`) | started with the file missing: a warning, "Every key is refused with 503 until it can"; read once it existed |
| 10 | no token | `GET /v1/documents` 401 |
| 11 | laptop's ingest job, building the scratch index | 202, then `Published s3_8_live--gen-20261026T170857383393Z-eb33e4f9: 3 documents, 1708 chunks` (26.9 chunks/s); every job line carries `api_key=laptop` and its `job_id` |
| 12 | bot's token: `POST /v1/query`, then `POST /v1/ingest` | 200, cut by the client after 3 s ("Answer cancelled: its client disconnected"); Ollama answered the chat 499 after 2.17 s, "client connection closed before server finished loading, aborting load". Then 403 |
| 13 | a wrong token 11 times within a minute | ten 401, then 429 (17:13:25) |
| 14 | five more, 17 s later, each with a fresh `X-Forwarded-For` | 401, 401, 429, 429, 429: the real address's bucket had earned back 2.8 tokens in 17 s, and the spoofed addresses drew on it. Believed, each would have had a fresh bucket: five 401s |
| 15 | a 1 MB body without a token, with `Content-Length`, then chunked | 413, 413 |
| 16 | 200 `GET /v1/health` at once | 200 answers, all 200, none 429, within 0.1 s (17:14:01.645 to .743 UTC); Ollama's journal: one `GET /api/tags` (22:44:01 local), the start-ups' checks apart |
| 17 | `kill -9` of rag-serve's python during a `rebuild: true` job, then a restart (`serve-2.log`) | the job was running (its log lines come from inside the writer lock). After the restart: "Serving index s3_8_live--gen-…-eb33e4f9", the published one, unchanged; `GET /v1/ingest/<job>` 200, "Ingest job 33812d46… was abandoned", once |
| 18 | `rag-keys revoke bot`, then bot's token | 401 at the next request, the file read again ("1 keys", a slip: Discovered) |
| 19 | the scratch index deleted, the test keys removed | `rag_qa`'s two generations and its alias on `…7f20891e` unchanged; `var/` holds `ingest.lock`, `jobs.sqlite3` and the dead job's lock file (Discovered) |
| 20 | `gh pr checks` (PR #28) | green on `646e888`, the close-out record (run 40370275154), on `bc2d892`, the first review's (run 40384658394), and on `a6ed789`, the second review's (run 40471130129): `check` 5m27s, `deploy-config` 23s, `eval-retrieval` 2m43s |

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

- **Two of the suite's tests polled a job through default settings** (found by the mutation
  sweep's control run). The test client's requests all come from one address, and a loopback
  server with no key now limits it to 60 a minute: a job that took over a second turned the
  poll into a 429, and the test into a `KeyError`. `conftest.UNLIMITED` is now the rate of every
  app a test polls (`test_api.py`, `test_metrics.py`, `test_tracing.py`); the limits' own
  tests set theirs. Caveat 42.
- **"1 keys."** (the live run, row 18): the keys file's read line did not agree in number, as
  S2-8's and S3-7's plural slips did not. Fixed, with a test.
- **A dead job's lock file stayed behind** (the live run, row 19): its process could not remove
  it, and the prune would only after 100 more jobs. Whoever writes the abandoned verdict now
  removes it: the lock is free, and no other job has its ID. The one left by the live run, in
  this host's `var/jobs/`, predates the fix; `rm` it, or let the prune take it.
- **The cancel reached Ollama while the model loaded** (row 12): "client connection closed
  before server finished loading, aborting load", a 499 after 2.17 s. DEC-14's cancel holds
  through the new dependencies, and a question cut that early leaves Ollama nothing to finish.
- **uvicorn's access line carries the key's name** (rows 11–18): it is logged in the request's
  own task, after the response, so it inherits the context `authenticate` bound. Every access
  line names the key that made it, a 401's none. (A key's own 429 named none until the second
  review, which binds the name before the bucket: its fix 4.)
- **Two guards close the race design review 2 measured, and each is enough alone:** the second
  read of the row, and the verdict written only over an unfinished status. Removing either
  alone changes nothing a client sees (an equivalent mutant); removing both fails the race
  test. Both are kept, as DEC-23 has them.
- **A body that is not valid JSON still gets its 422 before authentication,** now over at most
  64 KiB, and draws on no bucket: FastAPI parses before any dependency. An unauthenticated
  client can make the server parse 64 KiB as often as it likes. Bounded, and DEC-23 did not
  ask for more; the backlog has the line (STATUS.md).
- **The mutation sweeps:** 17 mutants of the keys and limits, 11 of the job records and 7 of
  the probes; every one caught but the equivalent one above. The first job sweep caught 5 of
  10: three tests were vacuous (a probe of a folder that did not exist, a status read before
  the job's thread began, a verdict never reached) and two guards had none. All five were
  fixed before the commit.

## Plan's backlog lines, as closed

- **Closed: the availability SLI sees no unexpected 500** (S3-7b's second review). An
  unexpected 500 on a question counts as `rag_refusals_total{code="500",reason="error"}`, and
  the SLI reads `code=~"5.."`. The 401, 403, 413 and 429 refusals count on neither side.
- **Moved to S3-9 (Q2):** the name-based no-secrets rule, and the Qdrant URL in a job's error
  text. S3-9's story has both lines.
- **Stays:** the stale `stream_options` assertion; this story did not run
  `test_fallback_openai.py`.

## Deviation from plan

- **The model:** Opus 5.5 at max effort, in place of Fable, which is unavailable (CLAUDE.md).
- **Two breaking commits, not one** (D13): the body limit and the trusted proxies
  (`f961a25`), and the rate limits (`bad1070`), each with its own `BREAKING-CHANGE:` trailer.
- **The hermetic chunked-body test sends 80 KiB, whole,** not 1 MB: a server that answers
  before an unread body ends may reset the connection under the client, and lose the 413 it
  sent. The live run sent the megabyte both ways (row 15).
- **An abandoned job counts as a failed run** (not in D11): once, by the replica whose write
  changed its row, so `RagIngestFailed` fires after a crash mid-ingest; and that replica
  removes its lock file (Discovered).
- **`IngestJob` records the key that started it** (an `api_key` column), so the job's thread
  can bind the name: the request's log context never reaches that thread (D3 said the thread
  binds it).
- **`KeysFileSettings`** in `settings.py`: `rag-keys` reads `RAG_API_KEYS_FILE` alone, so a
  malformed `RAG_API_PORT` cannot stop it. **`trusted_proxies` is a string** in
  `ServerSettings`, parsed by `proxy_addresses`: pydantic-settings reads a list from the
  environment as JSON.
- **The README's "Upgrading from v0.3" gains item 6,** the API's limits (not in D12).
- **The close-out adds a fix commit** before the record: the live run's two findings
  (Discovered).

## First review (2026-10-27, Opus 5.5 on `646e888`, PR #28)

Reviewed as the first reviewer, against the PR's head, in scratch copies of it (never in the
repository). The maintainer's Qdrant was not touched; the one container run was promtool, by
digest, with `--rm`. One defect in what the story claims, two gaps and one overstated comment,
all fixed on the PR.

**Reproduced as they were:** the suite (1711 passed, 18 skipped, 5 xfailed; 98.28%), ruff,
mypy, `uv lock --check` (`uv.lock` equal to `main`'s), `rag-eval check` (exit 0) and
`promtool test rules` and `check rules` (41 rules) from CI's pinned image. After the live run
(`2aef8d7`) only a log plural and the dead job's lock-file cleanup moved in `src/`, none on the
retrieval path, so the IDENTICAL contexts stand.

**Fixed:**

1. **One refused read held every key closed.** `KeyRing.refresh` stored the file's signature
   before it read the file, so an `OSError` (a full table of open files, an I/O error) left the
   signature as "seen": every later look found the file unchanged, and keyed requests got 503
   and readiness 503 until the file's mtime moved or the server restarted. That is not "until it
   parses again" (this story, the README, ARCHITECTURE). A read the system refused is now a
   `KeysFileUnreadableError`: its signature is forgotten, so the next look (5 s) reads again,
   and the warning is said once. A file that does not parse still stays closed until it
   changes. Reproduced with one injected `OSError`: no match after 1 h; after the fix, a match 6 s
   later.
2. **Nothing tested that an unchanged file is not read again** (`auth.py`'s `return` was never
   run). Removing it failed no test; it also re-parses, and re-warns, a broken file at every
   look. A test now counts the reads and the warnings over three looks.
3. **`declared_length` took `str.isdigit()`**, which accepts `\xb2` ('²'), and `int()` refuses
   it. Unreachable under h11, which answers 400 first (`httptools` is not installed here, so
   not tried). Now ASCII digits only; the unit test has the case.
4. **A comment overstated:** a question and its sources do not "fit in a fraction" of 64 KiB;
   the model's largest ASCII request is about 53 KB (82%), and names of multi-byte characters
   pass the limit. The comment says so, and a test pins the ASCII maximum under it.

**Checked, nothing to fix:**

- **Cross-process job records:** 12 processes starting at once gave 1 started and 11 refused;
  after `kill -9` of the holder its job read `failed`, `abandoned`, and the next start was
  accepted.
- **`forwarded_allow_ips`:** uvicorn's source reads `FORWARDED_ALLOW_IPS` only for `None`; an
  empty list trusts nobody.
- **The eviction scan** costs 1.1 ms a take at worst, and only above about 1,700 new addresses a
  second on the failure table (its oldest bucket must be 6 s old to be full), 10,000 on the
  others. Bounded; the "stops at once" holds below that.
- **A scoped link-local IPv6 address** (`fe80::1%eth0`) does not break `client_key`; all of
  them share one `/64` bucket.
- **Extra mutants** (the removal of the bucket cap, a deferred `BEGIN` in `start()`, the
  failure bucket drawn for valid keys, a zero recheck interval): all caught. The fifth, the
  unchanged-signature shortcut removed, survived before fix 2 and is caught now.

**After the fixes:** 1715 passed, 18 skipped, 5 xfailed; coverage 98.26%; ruff, format and mypy
clean; the lock unchanged.

## Second review (2026-10-27, Opus 5.5 on `bc2d892`, PR #28)

Reviewed as the second reviewer, against the PR's head, the first review's fix and record
included, in scratch copies of it (never in the repository). The maintainer's Qdrant was not
touched, and no model ran. No blocker. Five things were fixed on the PR: a gap in what the
alerts claim, a disclosure the docs ruled out, a race, a log gap, and a stale line in a later
story.

**Reproduced as they were:**

- The suite: 1715 passed, 18 skipped, 5 xfailed; 98.26%. In a copy that is not a git checkout,
  two hygiene tests skip.
- ruff, the format check and mypy are clean; `uv lock --check` passes, and `uv.lock` equals
  `main`'s.
- Every commit's trailers parse, the two `BREAKING-CHANGE:` ones included.

**Fixed:**

1. **A crash mid-ingest was counted only once a client asked about the job.** The probe: `kill -9`
   of a process holding a job, then a fresh replica scraped before any request. It read `failed`
   0, the row still `running`, and its lock file was left behind; only a `GET /v1/ingest/<id>` or
   a new start judged it. ARCHITECTURE and this story's Deviation said `RagIngestFailed` fires
   after a crash mid-ingest.
   - The fix: every replica judges the unfinished jobs as it starts (`JobStore.judge_unfinished`,
     called from `IngestJobs`, in the lifespan's thread), in `get()`'s own order, so a live job,
     its lock held, is left alone.
   - After it, the same probe reads `failed` 1 at the start, the row final, the lock file gone.
   - Two tests: a dead job is counted once at the start; a live one is left running.
2. **Health told anyone a key's name and its place.** With one malformed entry, `GET /v1/health`,
   open and unthrottled, said "the keys file's key 3 (ci-deploy-bot) has no valid scopes…".
   `keys_health`'s own docstring said it never tells how many keys the file holds.
   - The fix: `KeysFileError`'s message is fixed text ("the keys file holds a key that is not
     valid"). Its `detail`, which names the key, goes to the server log and to `rag-keys` alone.
   - Tests: the parse cases find the key in the detail and not in the message; health and the
     503 name neither the key nor its place, and the log does.
3. **Two probes of a dead job's lock took each other for its owner.** `held()` asked for an
   exclusive lock, so a probe that met another replica's probe read "held". With another process
   probing in a loop, 14,614 of 20,000 probes of a free lock said "held", and 42 of 300 starts
   answered 409 for a dead job. Real polls make the window microseconds wide.
   - The fix, one flag: the probe asks for a shared lock, which the owner's exclusive one still
     refuses. The same probe: 0 of 20,000, and 0 of 300.
   - Two tests, neither timed.
4. **A key's own 429 named no key** in its log lines or on its span: `authenticate` bound the
   name after the bucket check. It binds it first now, so "every access line names the key that
   made it" (Discovered) holds for a key's 429 too. A test reads the log context at the refusal,
   and the 429's server span.
5. **S3-10's story said the README asks to back the keys file up "as the secret it is".** It did
   not, and the file holds hashes only. The README's "Keys" now says to back it up with `var/`
   (a copy gives away no token; without it every key must be made again), and S3-10's line
   points there.

**Checked, nothing to fix:**

- **The second read of a job's row** sees a commit made between the two reads, even with the
  first cursor still alive: `fetchone` on a one-row result resets the statement, so the guard
  does not rely on the cursor being freed (a probe on the PR's head).
- **The scope test** walks the app's own routes, and **the spoofing tests** run the production
  `uvicorn_config`.
- **`BodyLimit`'s streamed 413** can only fire inside FastAPI's body read. After the last chunk,
  uvicorn's `receive` returns only `http.disconnect`, and the routes that take no body never call
  it.
- **The cached probes** swallow their dependency's failure (`Serving.refresh`, `llm_reachable`),
  so a dependency that is down is cached like one that is up.
- **`start()`'s probe** cannot race a live job: `BEGIN IMMEDIATE` holds that job's `finish()` off
  until the verdict is written.
- **WAL with `synchronous=NORMAL`** keeps a commit through a `kill -9`.
- **A replaced keys file always has a new inode:** the temporary file and the old one exist at
  the same time.

**After the fixes:**

- The suite: 1721 passed, 18 skipped, 5 xfailed; coverage 98.26%. ruff, format and mypy are
  clean, the lock is unchanged, and `rag-eval check` exits 0.
- Four mutants, run after an unmutated control passed, all caught by their new tests: the
  start-up judging removed, an exclusive probe, the detail back in the message, and the binding
  after the bucket.
- Only `api/` moved in `src/` since the live run (`2aef8d7`), none of it on the retrieval path,
  so the IDENTICAL contexts stand.
- An optional live check: `kill -9` of `rag-serve` mid-job, a restart, then
  `rag_ingest_runs_total{status="failed"}` scraped before any request reads 1.
