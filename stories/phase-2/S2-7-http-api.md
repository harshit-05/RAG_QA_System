# S2-7: Serve queries and ingestion over HTTP, with SSE streaming

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-6, SRS §8.1; backlog: the API must never let a request supply components, the `trust_remote_code` load guard, the ChatOllama client (API half) |
| **Depends on** | S2-1 (`stream_answer`, `QueryPipeline`), S2-6 (manifest, lock, incremental `ingest`) |
| **Model** | fable |
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
  - **`POST /v1/ingest` and `GET /v1/ingest/{job_id}`:**
    - the body `{"rebuild": bool}` is required JSON, not defaulted;
    - one job at a time (409 while one runs);
    - each job runs in a worker thread;
    - an in-memory registry keeps the last 20 jobs;
    - the job keeps its ingest log lines as `log`. S2-6 already makes them, and the
      report's errors, path-free;
    - on success, only the retrieval half of the pipeline is swapped
      (`dataclasses.replace` with a fresh `build_retrieve`).
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
