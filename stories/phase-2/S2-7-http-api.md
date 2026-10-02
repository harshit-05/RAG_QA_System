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
  - It builds the `QueryPipeline` when an index exists, and reports not-ready otherwise.
  - It calls `aclose()` at shutdown.
- **`api/server.py`: `rag-serve --config --host --port`.** It exits 2, with the reason,
  on any of:
  - a config error;
  - a non-loopback host without a token;
  - Ollama unreachable at startup (with the hint).
- **`api/routes.py`**, as §2.4 specifies:
  - **`POST /v1/query`:**
    - the checks run in order: auth, then the concurrency slot (503 with `Retry-After`),
      then 503 when there is no index;
    - then the SSE events `sources`, `token`… and `done`;
    - an exception after the stream has started becomes an `error` event.
  - **The disconnect watcher.** The route owns the generation task, and the watcher
    cancels it when the client goes away. The spike decides whether Starlette's own
    disconnect handling already does this during a silent prefill. Keep the watcher
    unless Starlette passes the same test.
  - **`POST /v1/ingest` and `GET /v1/ingest/{job_id}`:**
    - one job at a time (409 while one runs);
    - each job runs in a worker thread;
    - an in-memory registry keeps the last 20 jobs;
    - ingest log messages are captured into the job;
    - on success, only the retrieval half of the pipeline is swapped
      (`dataclasses.replace` with a fresh `build_retrieve`).
  - **`GET /v1/documents`**, read from the manifest.
  - **`GET /v1/health`** (readiness: the index, plus Ollama's `/api/tags` within 2 s) and
    **`GET /v1/health/live`**.
  - **Auth.** When a token is set, every route except health needs
    `Authorization: Bearer <token>`, compared with `secrets.compare_digest`.
  - **No host paths in any body.** Only corpus-relative paths, and "no index yet" rather
    than a location.
- **`schema.py`: the `trust_remote_code` guard.** A truthy `trust_remote_code` at any
  depth of a component spec is a load error that names its location (§2.3).
- **README:** an "HTTP API" section covering `rag-serve`, the token, the endpoints, and
  `curl -N` for SSE.
- **Tests**, with FastAPI's test client against a fake pipeline:
  - SSE framing and order (`sources` → `token`… → `done`), and the `error` event.
  - **401:** a missing header, the wrong scheme, a wrong token, an empty token. Health
    stays open.
  - **422:** an empty question, an overlong one, and an extra `llm` key, so a request
    cannot pick a component.
  - **503:** no index, and busy. **409:** a second ingest.
  - The job lifecycle, and documents read from a fake manifest.
  - Every response body is scanned for the temporary corpus root, which must never
    appear.
  - `rag-serve` refuses a non-loopback host without a token.
  - Cancellation at the task level: the watcher cancels the producer, and the fake
    model's `finally` runs.

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
export RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index
uv run rag-serve &                                      # 127.0.0.1:8000
curl -s localhost:8000/v1/health                        # ready: index and llm
curl -sN -X POST localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"question": "How does the YOLOv8 system detect crowds and threats?"}'
#    → event: sources, then event: token …, then event: done {"ttft_ms": …}
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/query \
     -H 'content-type: application/json' -d '{"question": "x", "llm": "components.llms.qwen2_ollama"}'   # → 422
curl -s -X POST localhost:8000/v1/ingest; curl -s localhost:8000/v1/ingest/<job_id>   # → succeeded, with the counts
curl -s localhost:8000/v1/documents | head

# 4. a disconnect stops generation (manual; caveat 21). Watch the Ollama log or `top`.
curl -sN --max-time 3  -X POST localhost:8000/v1/query -H 'content-type: application/json' -d '{"question": "…long…"}'   # drops during the prefill
curl -sN --max-time 30 -X POST localhost:8000/v1/query -H 'content-type: application/json' -d '{"question": "…long…"}'   # drops mid-stream
#    → in both cases, Ollama's request ends within ~1 s of curl exiting

# 5. the token
RAG_API_TOKEN=s3cret uv run rag-serve --port 8001 &
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8001/v1/query -H 'content-type: application/json' -d '{"question": "x"}'   # → 401
curl -s -o /dev/null -w '%{http_code}\n' localhost:8001/v1/health/live                                                             # → 200

# 6. CI
gh run list --limit 2
```

## Review notes for the human

- **Attack the auth** (caveat 14): no header, the wrong scheme, an empty token, a token
  with whitespace. Check that the comparison is constant-time.
- **Read the request model.** Nothing in it may reach `build_object`.
- **Read the disconnect results in step 4, during the prefill especially.** That is the
  window Starlette alone may miss.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
