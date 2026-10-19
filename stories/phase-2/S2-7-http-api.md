# S2-7: Serve queries and ingestion over HTTP, with SSE streaming

| | |
| --- | --- |
| **Status** | Done (2026-10-12): PR #15, merged as `1ef599f`. CI was green on the PR's head, `2eff33f`, and on `main` after the merge. Verification 1–6 are run and shown below. Two reviews, no blockers: each fixed one code and one doc finding on the PR. Two commits in the middle of the PR fail their own tests (see Deviation); it was merged as is, by the maintainer's choice. |
| **Closes** | FR-6, SRS §8.1; backlog: the API must never let a request supply components, the `trust_remote_code` load guard, the ChatOllama client (API half) |
| **Depends on** | S2-1 (`stream_answer`, `QueryPipeline`), S2-6 (manifest, lock, incremental `ingest`) |
| **Plan-first** | yes |
| **Branch** | `feat/s2-7-http-api`. Risky: it adds a new network-facing surface and new main dependencies. |

## Goal

The system becomes callable over HTTP, which is the first half of the Phase 2 exit. It
offers:

- answers streamed over SSE, with sources first;
- ingestion as a background job;
- a document listing;
- health probes.

Two properties hold throughout:

- **A client that disconnects stops its generation**, as Ctrl-C does in the CLI.
- **Nothing in a request can choose or change a component.**

Design: ARCHITECTURE.md §2.1 DEC-18, interfaces and SSE protocol in §2.4.

## Scope

- **Dependencies.** `uv add fastapi uvicorn` (FastAPI ≥ 0.135 for native SSE). Record the
  resolved versions. Add `httpx` to the dev group if the tests import it directly.
- **`settings.py`: `ServerSettings`**, read from the environment only:
  - `RAG_API_TOKEN`, as `SecretStr | None`;
  - `RAG_API_MAX_CONCURRENT`, default 1;
  - the host and port defaults.

  `settings.py` stays the only reader of `RAG_*` variables.
- **`api/models.py`.** `QueryRequest` is `{question}`, 1–2,000 characters, frozen, with
  `extra="forbid"`. Plus the response and SSE payload models.
- **`api/app.py`: `create_app(config, settings)`, with a lifespan.**
  - It builds the pipeline with `build_query_pipeline(config, require_index=False)`. With
    no index, or an incompatible one, `retrieve` is `None`: the server starts and
    reports not-ready, and `/v1/health` says "no index yet" or "re-ingest with
    --rebuild". `NoIndexError` from `stream_answer` remains a backstop, mapped to an
    `error` event.
  - It records the generation it opened (`os.readlink` of the store path).
  - It logs the ASGI `spec_version` the server reports, once, at the first request.
  - It calls `aclose()` at shutdown.
- **`api/server.py`: `rag-serve --config --host --port`.** It exits 2, with the reason,
  on any of:
  - a config error;
  - a non-loopback host without a token;
  - Ollama unreachable at startup (with the hint).
- **`api/routes.py`**, as §2.4 specifies:
  - **`POST /v1/query`:**
    - **Every refusal happens before the first byte, as dependencies, in order:**
      `Origin` (403), auth (401), then the concurrency slot (503 with `Retry-After`),
      then the index (503). With FastAPI's yield-based SSE, the path function's body runs
      only after the 200 headers are sent, so a refusal from inside the generator would
      be too late.
    - then the SSE events `sources`, `token`… and `done`;
    - an exception after the stream has started becomes an `error` event, carrying the
      exception type and a fixed message for it. The full exception goes to the server
      log only.
  - **Disconnects: no producer task and no watcher** (DEC-18). `stream_answer` runs
    inside the SSE generator itself. Under uvicorn (ASGI spec 2.3), Starlette's
    `listen_for_disconnect` cancels that stream when the client drops, during a silent
    prefill too, and the cancel lands in `stream_answer`'s `finally`. A task spawned by
    the route would sit outside that cancel scope and orphan generation.
    - **Spike first:** confirm this on the resolved FastAPI and uvicorn, with a real
      uvicorn on a free port and a fake model that sleeps 10 s before its first token.
      If the drop is not seen within 1 s, add a watcher and record why here.
  - **The concurrency slot.** A dependency takes it (`locked()` then `acquire()`, with no
    await between them) and hands over a `Slot` whose `release()` is idempotent. The
    generator's `finally` releases it, and so does the dependency's teardown. The
    teardown covers a client that leaves before the generator's body ever runs. A
    leaked slot would mean a permanent 503.
    - **A freed slot is not a free CPU** (S2-4's second review). Retrieval runs in an
      executor thread that a cancel cannot stop. With the reranker on, that thread keeps
      going for up to about 2 s after its client disconnects (median 1.4 s, on torch's 10
      threads). The next request would then rerank alongside it.
    - Two reranks at once are safe (verified: 40 concurrent reranks matched sequential
      results), but they compete for the same CPU. The plan picks one of two ways to keep
      them apart:
      - hold the slot until the retrieval thread finishes;
      - serialize retrieval with a lock in the retrieval half, which also covers
        `RAG_API_MAX_CONCURRENT` above 1.

      Test it with a slow fake reranker: a request that disconnects mid-retrieval, then
      a second request, whose retrieval must not start until the first one's has ended.
  - **`POST /v1/ingest` and `GET /v1/ingest/{job_id}`:**
    - the body `{"rebuild": bool}` is required JSON, not defaulted;
    - one job at a time (409 while one runs);
    - each job runs in a worker thread;
    - an in-memory registry keeps the last 20 jobs;
    - the job keeps its ingest log lines as `log`. S2-6 already makes them, and the
      report's errors, path-free;
    - on success, only the retrieval half of the pipeline is swapped
      (`dataclasses.replace` with a fresh `build_retrieve`). `build_retrieve` builds a new
      reranker on every call (S2-4: 2.3 s and about 90 MB), and the reranker's model is not
      tied to the index. So build it once, in the lifespan, and pass it in, rather than
      reloading it on each swap or manifest-triggered reopen.
  - **Picking up an ingest run outside the server.** Each query first compares
    `os.readlink` of the store path with the generation being served, and reopens the
    retrieval half when they differ. That is one syscall, and the flip is atomic
    (DEC-17), so a half-written index is never seen.
    - The reopen runs in a worker thread, under an asyncio lock, so concurrent queries
      reload once.
    - It runs the same compatibility check as startup. An index built with a different
      embedder makes the server not-ready; it never serves incomparable vectors.
    - `/v1/documents` and `/v1/health` read the served generation's manifest.
  - **`GET /v1/documents`**, read from the manifest.
  - **`GET /v1/health`** (readiness: the index, plus Ollama's `/api/tags` within 2 s) and
    **`GET /v1/health/live`**.
  - **Auth.** When a token is set, every route except health needs
    `Authorization: Bearer <token>`. Compare with `secrets.compare_digest` on UTF-8
    **bytes**: on `str` it raises `TypeError` for non-ASCII input, which turns an attack
    into a 500.
  - **Browsers, Docs and Host.**
    - Any request carrying an `Origin` header gets 403, whatever the token setting.
      curl, the CLI and scripts send none, and a cross-site POST from a web page would
      otherwise reach a tokenless server: it arrives with `Host: 127.0.0.1`.
    - Every POST requires `content-type: application/json`.
    - With a token set, `/docs`, `/redoc` and `/openapi.json` need it (or are switched
      off).
    - Without a token, a request whose `Host` header is not loopback is refused (DNS
      rebinding).
  - **No host paths in any body.** Only corpus-relative paths, and "no index yet" rather
    than a location.
- **`schema.py`: the `trust_remote_code` guard.** A truthy `trust_remote_code` at any
  depth of a component spec is a load error that names its location (§2.3).
- **README:** an "HTTP API" section covering `rag-serve`, the token, the endpoints, and
  `curl -N` for SSE. It says that a token over a non-loopback host travels in cleartext
  until Phase 3's TLS; `rag-serve` logs the same warning at startup.
- **Tests**, with FastAPI's test client against a fake pipeline:
  - SSE framing and order (`sources` → `token`… → `done`), and the `error` event.
  - **401:** a missing header, the wrong scheme, a wrong token, an empty token, a token
    with surrounding whitespace, and a **non-ASCII** token (401, not 500). Health stays
    open.
  - **403:** any request with an `Origin` header, with and without a token.
  - **415 or 422:** a POST to `/v1/ingest` or `/v1/query` with `text/plain` or no body.
  - **422:** an empty question, an overlong one, and an extra `llm` key, so a request
    cannot pick a component.
  - **503:** no index, an incompatible index, and busy. **409:** a second ingest.
  - **The slot is free afterwards**, after each kind of ending: `done`, `error`, a
    disconnect mid-stream, and a disconnect before the first event.
  - The job lifecycle, and documents read from a fake manifest.
  - Every response body is scanned for the temporary corpus root, which must never
    appear. That includes a real ingest job over a temporary corpus holding a `chmod 000`
    file: its report and `log` are scanned too.
  - `rag-serve` refuses a non-loopback host without a token.
  - With a token, `/docs` and `/openapi.json` return 401 without it; tokenless, a
    non-loopback `Host` header is refused.
  - A server with no index starts, answers 503 on query, and serves after an ingest job.
  - A CLI-style ingest outside the server (a new generation flipped in) is served on the
    next query. One built with a different embedder makes the server not-ready.
  - **Cancellation under real uvicorn:** start uvicorn on a free port with a fake model
    that sleeps before its first token, drop the client during that sleep and again
    mid-stream. Both times the fake model's `finally` runs within 1 s, and the slot is
    freed.

## Out of scope

- Rate limiting, TLS and multi-user auth (Phase 3, backlog).
- Docker (Phase 3), SSE resume, a web UI.
- CLI changes (S2-8).

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. startup refusals
uv run rag-serve --host 0.0.0.0; echo "exit $?"        # → 2: non-loopback without a token

# 3. against scratch paths (caveat 9): the ingest endpoint writes an index
#    The server runs in a second terminal, in the foreground, with its log tee'd:
#      RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index \
#        uv run rag-serve 2>&1 | tee <scratch>/serve.log      # 127.0.0.1:8000
export RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index
curl -s localhost:8000/v1/health                        # ready: index and llm
curl -sN -X POST localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"question": "How does the YOLOv8 system detect crowds and threats?"}'
#    → event: sources, then event: token …, then event: done {"ttft_ms": …}
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/query \
     -H 'content-type: application/json' -d '{"question": "x", "llm": "components.llms.qwen2_ollama"}'   # → 422
curl -s -X POST localhost:8000/v1/ingest -H 'content-type: application/json' -d '{"rebuild": false}'
curl -s localhost:8000/v1/ingest/<job_id>               # → succeeded, with the counts; no absolute path in it
curl -s localhost:8000/v1/documents | head
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/ingest \
     -H 'Origin: https://example.com' -H 'content-type: application/json' -d '{"rebuild": false}'   # → 403
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/ingest -H 'content-type: text/plain' -d 'x'   # → 415 or 422

# 4. a disconnect stops generation (manual; caveat 21). Watch the Ollama log or `top`.
curl -sN --max-time 3  -X POST localhost:8000/v1/query -H 'content-type: application/json' -d '{"question": "…long…"}'   # drops during the prefill
curl -sN --max-time 30 -X POST localhost:8000/v1/query -H 'content-type: application/json' -d '{"question": "…long…"}'   # drops mid-stream
#    → in both cases, the server's connection to Ollama closes within ~1 s of curl
#      exiting (Ollama's log), and the runner idles: at once mid-stream, and by the end
#      of the prefill at the latest during the prefill (DEC-14). Record both timings.
#    → a query right after each drop gets a slot (not 503): nothing leaked

# 5. the token (restart the second terminal's server with RAG_API_TOKEN=s3cret --port 8001)
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8001/v1/query -H 'content-type: application/json' -d '{"question": "x"}'   # → 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8001/v1/query -H 'content-type: application/json' \
     -H 'Authorization: Bearer sécret' -d '{"question": "x"}'                                                          # → 401, not 500
curl -s -o /dev/null -w '%{http_code}\n' localhost:8001/v1/health/live                                                             # → 200

# 6. CI
gh run list --limit 2
```

## Review notes for the human

- **Attack the auth** (caveat 14): no header, the wrong scheme, an empty token, a token
  with whitespace, a non-ASCII token, and a request with an `Origin`. Check that the
  comparison is constant-time and on bytes.
- **Read the request model.** Nothing in it may reach `build_object`.
- **Check that no task is spawned for generation.** `stream_answer` must run inside the
  SSE generator. A spawned task is the shape that keeps Ollama generating after the
  client leaves.
- **Read the slot's release paths.** A leaked slot is a permanent 503.
- **Read the disconnect results in step 4, during the prefill especially.**

## Discovered

- **Resolved versions (2026-10-12):**
  - fastapi 0.143.0, starlette 1.7.0 and uvicorn 0.54.0. uvicorn is plain, not
    `[standard]`, so it serves over h11 with no uvloop or httptools.
  - httpx 0.28.1 is in the dev group; it was already present, through ollama.
  - FastAPI pulls in one new package: opentelemetry-api 1.45.1, a base dependency of
    FastAPI's.
  - torch 2.14.0+cpu and langchain-core 1.6.3 are unchanged.
- **FastAPI's native SSE runs the path generator in its own producer task**
  (`fastapi/routing.py`, `_sse_producer_cm`). The task runs in a task group that is
  entered on the request's exit stack.
  - On a disconnect, Starlette cancels its response task (uvicorn reports ASGI spec 2.3),
    and the response returns. The request stack then unwinds, and that cancels the
    producer group.
  - So the cancel does reach `stream_answer`, but not as DEC-18 put it ("Starlette
    cancels the stream that `stream_answer` runs inside"). It is still no task of ours,
    and none is spawned outside the request's lifetime.
  - **The window:** each send from the producer passes anyio's `checkpoint()` outside
    our generator. A cancel that lands there leaves the generator paused at a `yield`.
- **The spike** ran real uvicorn, the real `stream_answer` and a fake model whose
  `finally` records the time. The client dropped over a raw socket.

  | Case | Trials | Result |
  | --- | --- | --- |
  | S1: a drop during a 10 s prefill | 5 | closed in 0.6–1.4 ms, by a cancel inside the read; slot freed; `spec_version` 2.3; the generator's `finally` ran before the dependency teardown |
  | S2: a drop mid-stream, 5 ms per token, one core | 300 | 299 closed by the cancel (median 0.67 ms, max 17 ms); **1 left the model's stream open for more than 5 s** |
  | S3: the window forced (the producer blocked in its send of a token) | 5 | **5 of 5 open for more than 5 s.** Each was freed only when the cycle collector ran during a later request |
  | S3 with the teardown closing the answer stream | 5 | 5 of 5 closed in 1.1–1.75 ms (`GeneratorExit` at the paused `yield`) |

  - So the generator sits in a reference cycle once the window is hit. An idle server
    would leave Ollama generating until garbage collection happened to run.
  - **Decision:** the slot dependency holds the answer stream, and its teardown
    `aclose()`s it. That is cleanup inside the request's own structured teardown: no
    task and no watcher. No watcher was needed for the drop during the prefill, which is
    the story's own rule.
- **The suspected SSE encoding bug is not in 0.143.0.** fastapi 0.136.1 split SSE data
  with `str.splitlines()`, which would break a JSON string holding U+2028 or U+0085.
  0.143.0 splits only on `\n`, `\r\n` and `\r`. A test pins the round trip.
- **A lifespan failure under uvicorn always prints a traceback.** Starlette sends
  `lifespan.startup.failed` with `traceback.format_exc()`. So `rag-serve` checks Ollama
  before uvicorn starts.
- **anyio's worker threads are not daemon threads.** An ingest job on one would hold
  shutdown for minutes, so jobs run in a daemon thread. DEC-17 makes an abandoned run
  safe.
- **`build_query_pipeline(require_index=False)` cannot tell the API what it opened.** It
  gives no generation, no manifest, and no reason when there is none (it swallows the
  `NoIndexError`). The API opens the index through `chain.open_retrieval` instead, and
  `require_index` is removed: no production code called it.
- **FastAPI 0.143 keeps included routers whole in `app.routes`** (`_IncludedRouter`),
  rather than copying their routes in. The app's own dependencies still apply. The test
  that checks every route needs the token walks the routes as FastAPI's OpenAPI code
  does, with `fastapi.routing.iter_route_contexts`.
- **Ollama answers a disconnected chat request with 500, at once, in its log.** Its runner
  still finished the abandoned prompt evaluation: 188 s at full CPU (Verification 4). That
  is caveat 21, now seen through the API.
- **Verification 4's `--max-time 30` would still be in mistral's prefill.** Time to first
  token is about 157 s at `num_thread` 10. So the mid-stream drop killed curl after 5
  tokens, on the question just answered, whose prompt Ollama still had cached.
- **A tee'd server log loses its shutdown lines.** Ctrl-C reaches the whole foreground
  pipeline, so `tee` dies with the server. `tee -i` ignores the interrupt; use it for
  S2-9's end-to-end run.
- **`uv sync` is exact: it removed the local `eval` extra** (ragas and 24 others) that this
  `.venv` had; a plain `uv run` is inexact and keeps it. Restored with
  `uv sync --extra eval`.
- **Out of scope, for the backlog:**
  - the `eval` extra's `langchain` 1.3.2 has PYSEC-2026-2192, fixed in 1.3.9. CI never
    installs the extra, so its audit never sees it. CI's own set, with the four new
    packages, audits clean;
  - request bodies are not size-limited, and FastAPI reads and parses one **before it runs
    any dependency, the token check included** (a 20 MB body was parsed in 0.04 s and then
    refused with 401: first review). And `/v1/health` asks Ollama on every call,
    unauthenticated. Both belong with Phase 3's rate limiting. A cheap interim, if wanted:
    a 413 for a `Content-Length` over about 16 KiB, which would not cover chunked bodies;
  - a browser tab on `/` gets 404. There is no landing page, by design.
- **For S2-8:** `rag_qa.api.server.preflight` and `OLLAMA_HINT` are what `rag-query`'s
  unguarded start-up needs (its backlog line: Ollama down, the hint).
- **For S2-9:**
  - README's status line still calls the HTTP API planned;
  - §2.2 lists four `api` modules, and there are six;
  - §2.8 should fold in DEC-18's S2-7 notes.

## Verification record (2026-10-12)

1. **The gates:** `ruff check` and `mypy` were clean (30 source files), and `pytest
   --cov=rag_qa` gave **845 passed, 5 xfailed**, coverage 98.30%.
   - The `api` package is fully covered except `server.py`'s `if __name__` line.
   - The new xfail is the strict negative control for the forced window.
   - **Extras run:**
     - **Caveat 23:** the same suite under `HF_HOME=<empty dir> HF_HUB_OFFLINE=1` gave the
       same result, and the folder stayed empty.
     - **Caveat 24:** `uv run rag-eval check` exited 0, with all six fingerprint parts ok
       and every floor met. No hashed input moved.
     - **The audit:** CI's 120 packages showed no known vulnerabilities.
     - **Stability:** the timing-based tests (the uvicorn file, the lock, the refresh and
       the slot) ran 15 times, then 15 more pinned to one core: 13 passed and 1 xfailed
       every time.
2. **`uv run rag-serve --host 0.0.0.0; echo "exit $?"`** printed "Error: refusing to listen
   on 0.0.0.0 without a token…" and **exit 2**, with no traceback.
3. **Against scratch paths.** The maintainer's terminal ran the server, with
   `RAG_DATA_PATH` and `RAG_VECTOR_STORE_PATH` both set to scratch, on mistral from the
   real config. It started without an index, and its log gave the reason. The preflight
   showed in Ollama's log as the expected `/api/tags` calls.
   - **The ingest job ran before the query,** because the scratch index was empty: a
     query first would only 503.
   - `/v1/health` was **503** "no index yet", with the llm ready.
   - `POST /v1/ingest {"rebuild": false}` gave **202** with a `location`. The job
     **succeeded** in 115 s with `exit_code` 0: 561 pages, 1,708 chunks, the two parquet
     files skipped, and no path in the report or the log.
   - Then `/v1/health` was **200** ready, serving `index.gen-…`, 3 documents, and
     `/v1/documents` listed the three PDFs from the manifest.
   - **The query:**
     - `sources` came at **1.0 s**, five chunks of the YOLOv8 paper, reranked;
     - a `: ping` came every 15 s through the prefill;
     - then 85 `token` events, and `done` at 176.3 s (`ttft_ms` 157,034);
     - the answer was on topic, with no `error` event.
   - **The refusals:**
     - the `llm=` body: **422**;
     - `Origin`: **403**;
     - `text/plain`: **422**, and neither refused ingest started a job.
   - No host path appeared in any body.
4. **The drops.** Each was timed from curl leaving. The script polled `ss` for the
   connection that request opened, and read the runner's CPU from `/proc`.

   | Drop | Our connection to Ollama | Ollama's log | The runner |
   | --- | --- | --- | --- |
   | mid-stream, after 5 tokens | closed **0.03 s** after | `/api/chat` 200 after 1.04 s | 708% at 0.5 s, **idle by 1.1 s** |
   | during the prefill, at 3 s | closed **0.03 s** after | `/api/chat` **500** after 1.88 s, at once | busy (950–985%) for **188 s**, then idle: it finished the prompt evaluation |

   - A query right after each drop got **200**, a slot and not a 503. Each check left
     during retrieval, so it never reached Ollama.
   - The server logged "Answer cancelled: its client disconnected" for every drop.
5. **The token** (restarted with `RAG_API_TOKEN=s3cret --port 8001`, now serving the
   index):
   - **401** without the header;
   - **401, not 500** for `Bearer sécret`;
   - **200** for `/v1/health/live`.
   - **Extras:**
     - 401 for a wrong token and for Basic, with `www-authenticate: Bearer`;
     - `/docs` gave 401, then 200 with the token;
     - `/v1/documents` gave 200 with it;
     - a query with it got a slot;
     - `Origin` with a valid token: 403;
     - `/v1/health` stayed open (200).
   - No 500 or traceback in the server log.
6. **CI: green on PR #15.** Run 38252812472, on `5e96d1c`, took 5 min 14 s.
   - **`check`:** lint, types, test, audit and the tier-2 check all passed.
     - **841 passed, 4 skipped, 5 xfailed,** coverage 98.30%. The 4 skips are the ragas
       tests that need the `eval` extra, which CI does not install; main's last run
       skipped the same 4.
     - On the runner, `test_api_uvicorn.py` gave six passes and the forced-window control's
       expected failure.
     - The committed tier-2 run matched the checkout.
   - **`eval-retrieval`:** passed, recall 0.858 against its floor of 0.808.
   - **After the reviews,** on the PR's head `2eff33f` (run 38260306611): green, 844
     passed, 4 skipped, 5 xfailed. The uvicorn tests passed again on the runner.
   - **On `main` after the merge** (`1ef599f`, run 38262357171): green, 5 min 57 s.

## Reviews

**First review (2026-10-12): no blockers; 1 doc finding and 1 code finding, both fixed on
the PR, and 1 decision recorded.**

- **Checked and sound:**
  - the gates, re-run: ruff and mypy clean, **845 passed, 5 xfailed**, coverage 98.30%;
  - `test_api_uvicorn.py` six times pinned to one core: 6 of 6 passed, and the strict
    control failed for its reason;
  - `rag-serve --host 0.0.0.0` and an invalid token both exit 2, and the token's value is
    not echoed;
  - `rag-eval check` exits 0, and the diff touches no hashed input (caveat 24);
  - the auth, by probe: the scheme in any case, two spaces, a trailing space, a prefix, no
    scheme and a non-ASCII token all give 401 and never 500. With a valid body the order is
    `Origin` 403, bad `Host` 403 (tokenless), token 401, busy 503, an invalid model 422;
  - no task is spawned for generation: the only `create_task` is the ingest job's;
  - the commits: headers at most 70 characters, a `Refs:` trailer on each, no attribution,
    and `BREAKING-CHANGE:` on the guard's commit. The lock adds four packages, as recorded.
- **Not re-run by the reviewer:** verifications 3 and 4 against real Ollama (minutes of
  full CPU, about 5 GB of RAM). The hermetic tests agree with them.
- **Fixed on the PR: malformed JSON gets its 422 before the 403 and the 401.** FastAPI
  parses the body before any dependency. With a token set, no auth plus malformed JSON gave
  422, as did `Origin` and a bad `Host`. A valid body gives the documented order. Nothing
  runs and nothing leaks, but `routes.py`'s docstring and ARCHITECTURE.md §2.4 said a 422
  comes last. Both now describe two 422s, and `test_a_body_that_is_not_json_is_refused_
  before_the_dependencies` pins the order.
- **Decision (maintainer): the body is read before the token is checked.** It is the same
  fact seen as a limit: no size cap, and a 20 MB body is parsed before the 401. It stays in
  the Phase 3 backlog, reworded as pre-auth (Discovered). The interim 413 on
  `Content-Length` is not built.
- **Fixed on the PR (code): a reopen that failed with anything but `NoIndexError`.**
  `_try_open` caught only that. An `OSError` or a failing embedder during a reopen left
  `refresh()` raising, so `/v1/query`, `/v1/documents` and `/v1/health` all gave 500.
  - **Reproduced first:** a test with a `flaky_open` that raises `OSError` after a new
    generation is flipped in failed on the old code, with the error escaping the
    middleware.
  - **The fix:** `_try_open` logs any other exception and returns it, and `open_served`
    reports `UNAVAILABLE`, a fixed message with no path. Health is 503. No generation is
    recorded as seen, so the next request opens the index again; the same test shows the
    server serving once the failure passes.
  - **A start-up that fails this way** now starts not-ready instead of failing, which is
    the story's rule for an index it cannot use. A failing *embedder build* still stops
    the server: that is before the index opens.
  - **The cost of retrying** is one open attempt per request while the failure lasts.
    `open_retrieval` already turns a store that will not open into
    `IncompatibleIndexError`, which is not retried, so only failures unrelated to the
    index's contents repeat.
- **Nits, no change:** `RAG_API_TOKEN=a` is accepted, so a length warning on a non-loopback
  host would help. `/v1/health` is open by design, and reports the generation name, the
  document count and Ollama's reachability to whoever can connect.

**Second review (2026-10-12): no blockers; 1
code finding and 1 doc finding, both fixed on the PR.** It covered the risky half: the
auth and the browser refusals, the slot and the retrieval lock, the reopen, the ingest
jobs, and the first review's fix, which was not yet committed.

- **Checked and sound:**
  - **The gates on the working tree, the first review's fix included:**
    - ruff and mypy clean; **848 passed, 5 xfailed**, coverage 98.30%;
    - `rag-eval check` exit 0;
    - `test_api_uvicorn.py` five times on one core: 6 passed and the control xfailed
      every time.
  - **The token:**
    - an empty `RAG_API_TOKEN` counts as unset (`env_ignore_empty`). So it means loopback
      only, and an empty credential can never match;
    - the validator keeps the token to visible ASCII, so its `encode("ascii")` cannot
      raise, and the header is encoded as UTF-8, which no input makes raise.
  - **The Host check never resolves a name:** only `localhost` and loopback addresses
    pass. A rebound page's `Host` is its own domain, and is refused.
  - **`Origin`:**
    - every browser POST carries it, same-origin included;
    - a cross-site GET carries none, but cannot read the reply, and no GET changes
      anything;
    - a CORS preflight gets 405 before any dependency runs.
  - **The request models:** `QueryRequest` keeps `extra="forbid"` and `strict` (pydantic
    merges a subclass's config), so nothing a client sends reaches `build_object`.
  - **The slot:**
    - its two release paths are idempotent;
    - the teardown runs after FastAPI's producer task group has ended, since the request's
      exit stack unwinds in reverse. So it never closes a stream that is running.
  - **The locks:** the reopen lock is always taken before the retrieval lock, and a
    retrieval thread never waits on the loop, so they cannot deadlock.
  - **The ingest job:**
    - its thread writes the status last;
    - the registry never evicts the running job;
    - the job log copies `getMessage()` only, so tracebacks stay in the server log.
  - **The `trust_remote_code` guard** walks every kind, retrievers included, at any depth.
- **Finding 1 (code, fixed on the PR): concurrent requests each retried a failed reopen,
  one after another.**
  - **The cause:** the first review's fix records no generation as seen after a failed
    open, so that the next request tries again. But every request queued on the reopen
    lock behind the failing one then tried again in turn.
  - **Reproduced:** five concurrent refreshes made **five opens**, one after another (1.01 s
    for five 0.2 s opens). Each open holds the retrieval lock and reads the index.
    `/v1/health` refreshes too, without a token. The docstring's "concurrent requests
    reopen once" held only when the open succeeded.
  - **The fix:** an attempt count. A request that waited for another's attempt takes its
    result. A later request still tries again, as the first review decided.
    `test_concurrent_requests_share_one_failed_reopen` failed on the old code (5 opens) and
    passes on the new, 5 of 5 times on one core.
  - **Not done, the maintainer's call:** a back-off of a few seconds after a failed open,
    so that probes arriving one by one during an outage cost nothing. The first review
    accepted one attempt per request.
- **Finding 2 (docs, fixed on the PR): README called `/docs` "the interactive docs".**
  - A browser's "Try it out" sends a POST with `Origin`, same-origin included, so it gets
    403. With a token set, a browser cannot send the header to load the page at all.
  - That is reasoned from the Fetch standard, not tried in a browser. The row now says the
    docs are "to read".
- **Nit, no change:** "this machine" in README's Access section includes every account on
  it. A tokenless server takes questions and ingest jobs, `rebuild` included, from any
  local user. That is DEC-18's localhost-first design; a shared host should set a token.

## Deviation from plan

- **Two commits in the middle of the PR fail their own tests.**
  - `f12ef97` added the first review's tests together with the routes' fix. But the
    reopen's own fix, which two of those tests need (a reopen that raises `OSError`, and
    queued requests sharing a failed reopen), came two commits later, in `4c37911`.
  - CI ran on `badb2cd`, the head of that push. Two tests failed there (run 38260295065),
    each with the `OSError` escaping as a 500. Every commit from `4c37911` on is green.
  - The maintainer merged as is rather than fold `4c37911` into `f12ef97`. A
    `git bisect` over `main` should skip `f12ef97` and `badb2cd`
    (`git bisect skip f12ef97 badb2cd`).
- **The disconnect path differs from DEC-18's wording** (Discovered). The cancel reaches
  `stream_answer` through FastAPI's own producer task. The slot dependency holds the
  answer stream, and its teardown closes it, because the spike found a window where
  nothing else does. DEC-18 has a dated note.
- **Choices the story left to the plan** (confirmed 2026-10-12):
  - **Retrieval lock, not holding the slot.** A lock taken inside the retrieval thread
    covers `RAG_API_MAX_CONCURRENT` above 1, and frees the slot at once.
  - **A CLI-held writer lock fails the job** (exit 2, `rag-ingest`'s message) rather than
    giving a 409. A route cannot hold the `flock` for `ingest()`.
  - **The Host refusal is 403,** like the `Origin` one.
- **Changed from the story's letter:**
  - **The API does not use `build_query_pipeline(require_index=False)`.** It composes its
    pipeline from `open_retrieval`, `build_prompt` and `build_llm`, so it knows which
    generation it serves, and why when none. `require_index` is removed, and its four
    tests became API tests.
  - **Two more `api` modules,** `state.py` and `jobs.py`.
  - **`rag-serve` checks Ollama before uvicorn starts.** It builds the model, pings it,
    then closes it: a failure inside uvicorn's start-up would print a traceback.
  - **Ingest jobs run in a daemon thread.**
  - **uvicorn's graceful shutdown is 5 s,** so stopping the server cancels an answer in
    flight.
  - **Verification ran in a different order.** Step 3 ran the ingest before the query
    (empty scratch index). Step 4's mid-stream drop came after 5 tokens, not at
    `--max-time 30`.
- **Outside the Scope list, each needed by the above:**
  - `chain.live_generation` and `components.llm_reachable`;
  - `ingest.outcome()`, which `rag-ingest` now uses too (same output), and `_describe` made
    public as `describe_error`;
  - conftest isolates every `RAG_*` variable, and holds the API's fake pipeline;
  - the architecture test: `ingest` never loads `rag_qa.api`;
  - the module map in `rag_qa/__init__.py`;
  - README's layout and the `trust_remote_code` limitation line;
  - dated notes on DEC-18 and §2.4 in ARCHITECTURE.md.
