# S3-7b: Traces and alerts

| | |
| --- | --- |
| **Status** | In progress. Plan approved 2026-10-25 (below, "Plan"); on Opus 5.5 at max effort (Deviation). |
| **Closes** | NFR-10 (traces); DEC-22 (traces and alerts); DEC-24's `deploy-config` job |
| **Depends on** | S3-7 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes, a short one: S3-7's plan (2026-10-23) settled the design below. The plan step checks it against the libraries as installed, and settles what is marked open. |
| **Branch** | `feat/s3-7b-traces-alerts`. Moderate risk: spans wrap the answer stream, where DEC-14's cancel guarantee lives, and the ASGI instrumentation wraps the disconnect path. |

## Goal

Every query and every ingestion run emits OpenTelemetry spans with the GenAI conventions:
retrieval, rerank and generation latency, and token counts (NFR-10's traces). Alert rules fire
on the burn rate of the SLOs that S3-7's metrics measure, and CI tests them with `promtool`.
None of it may reopen DEC-14's cancel window. Design: ARCHITECTURE.md §3.1, DEC-22, and S3-7's
plan, which split this half off (design review 1's seam).

## Scope

- **Dependencies:** main: `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` and
  `opentelemetry-instrumentation-fastapi`; versions as verified: 1.45.1, 1.45.1 and 0.66b1,
  thirteen new packages by the plan's count (first written as about 17), among them asgiref,
  wrapt and googleapis-common-protos (protobuf is locked already). `opentelemetry-semantic-conventions` 0.66b1 came with S3-7. Record what resolves; no
  locked version should move.
- **`rag_qa/observability/tracing.py`:** an OTLP/HTTP exporter, active only when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set (read in `settings.py`; the exporter appends
  `/v1/traces`). It is flushed in the API's lifespan and at the end of each command. A dead
  collector costs at most `OTEL_EXPORTER_OTLP_TIMEOUT` (10 s) at exit, with warnings. With the
  variable unset there is no instrumentation at all, so the disconnect path is untouched.
  - **Helpers:** spans with an explicit parent, ended in a `finally`, never made current
    across a `yield`.
  - **Executor threads:** copy the context into them, since `serialized`'s `run_in_executor`
    does not (LangChain's own `run_in_executor` does, on the CLI's path).
  - **Logs:** a processor adds the trace and span IDs to S3-7's log lines.
- **Spans:**
  - **The HTTP server span:** `FastAPIInstrumentor.instrument_app` with `exclude_spans`
    `["receive", "send"]` and the health routes left out (`excluded_urls`). Read again in
    0.66b1 for S3-7's plan: with `receive` excluded its ASGI middleware hands Starlette the raw
    `receive`; the server span is current only around the app call, and ends in the
    middleware's `finally` when an SSE body never completes. It also patches Starlette's
    `BackgroundTask` globally; the API uses none. If the real-uvicorn disconnect test still
    shows a difference, write a server span by hand and record why.
  - **`rag.retrieve`,** with `rag.embed_query`, `rag.vector_search` (`db.system.name` "qdrant",
    a custom value: Qdrant has no well-known one; `db.collection.name` the generation;
    `rag.mode`) and `rag.rerank` (candidates, kept, `min_score`).
  - **`chat {model}`,** one per model attempt. Its attributes are taken from the
    semantic-conventions package's own constants, never typed from memory: the operation, the
    provider, the request model, temperature and maximum tokens (`num_predict`, or
    `max_completion_tokens`), the input and output tokens of `Done`, and
    `gen_ai.response.time_to_first_chunk`. An event marks the first token, and a cancel is
    recorded as one. Our own attributes: `rag.model_ref`, `rag.attempt`, `rag.fallback`.
  - **`rag.ingest`,** with scan, prepare, embed and publish children; counts as attributes.
  - **No prompt or answer text** in any span, and **no LangChain callback handler**
    anywhere (`answering.py` says why).
- **Alerts:** `deploy/prometheus/rules.yml`, plus `rules.test.yml` for `promtool`.
  - **What they cover:** availability (5xx on `/v1/query`, and answers that end in an `error`
    event), NFR-1 (retrieval p95), NFR-2a (`sources`) and NFR-2b (TTFT), multi-window burn rates
    as in the SRE workbook; plus a breaker open for more than 5 minutes, a failed ingest, a
    scrape target down, and no target at all (`absent(up)`, the plan's D9).
  - **`deploy/prometheus/prometheus.yml`** and **`deploy/alertmanager/alertmanager.yml`:**
    a null receiver, with an email example commented.
- **CI** (needs the `workflow` scope): a `deploy-config` job runs `promtool check config`
  and `promtool test rules`. promtool runs from the Prometheus v3.15.0 image, pinned by its
  index digest, the one S3-9's compose runs (DEC-24's table, caveat 30). Prometheus publishes
  no signatures, but Docker Hub and Quay serve the same digest, and the image's `promtool`
  is byte-identical to the release tarball's (design review 2). The job also runs `amtool
  check-config` from the Alertmanager v0.34.1 image, pinned the same way (the plan's D11).
- **Tests:**
  - an in-memory span exporter shows the spans, their parents and the GenAI attributes;
  - with tracing on, DEC-14's forced-cancel tests and the real-uvicorn disconnect test pass,
    every span ends, and no "Failed to detach context" is logged.
- **README:** Observability's traces (how to point at a collector), and the alert rules in a
  sentence each.

## Settled in S3-7's plan (2026-10-23)

- **The hosted fallback's span.** When the fallback answers, it has a `chat {model}` span of its
  own (`chat openai/gpt-oss-120b` for `groq_fallback`), a sibling of the primary's attempt
  spans, each of which ended with status ERROR and `error.type`. `gen_ai.provider.name` is
  "openai", the API flavour, with `server.address` naming the host (api.groq.com), as the
  convention's note allows. `rag.fallback` is true and `rag.fallback_reason` says why:
  `breaker_open` or `connection_failed`. A breaker refusal sends no request, so it makes no
  span; the fallback's span carries the reason.
- **The provider names:** "ollama" for `langchain_ollama`, a custom value (none is well-known);
  `GenAiProviderNameValues.OPENAI` for `langchain_openai`. `components.model_identity()` (S3-7)
  maps a model reference to its provider and model name for both the metrics and the spans.
- **The semconv constants:** semconv 0.66b1 marks every GenAI constant deprecated ("moved to the
  OpenTelemetry GenAI semantic conventions repository"). Reading a constant or an enum member
  raises no warning, which the deprecation gate requires; calling an enum
  (`GenAiOperationNameValues("chat")`) raises one. Read members only.
- **A cancel** adds the event `rag.cancelled` and the attribute of the same name, and leaves the
  status unset: the client's choice, not an error. **A failure** sets status ERROR and
  `error.type` to the class name only, since an exception's message may carry text.
- **The objectives behind the burn-rate alerts:**
  - availability 99% over 30 d (bad: 503 refusals plus answers that failed after their 200;
    cancelled ones excluded);
  - NFR-1: 95% of retrievals under 0.3 s;
  - NFR-2a: 95% of `sources` under 2 s;
  - NFR-2b: 95% of first tokens under 300 s, this CPU's class (the committed tier-2 run: p95
    226.7 s, max 269.2 s); a GPU host edits it.

  Page at 14.4× the budget's burn over 1 h and 5 m, or 6× over 6 h and 30 m; ticket at 3× over
  1 d and 2 h, or 1× over 3 d and 6 h. S3-7's histograms carry the threshold buckets (0.3, 2.0,
  300.0).
- **`RagBreakerOpen`:** per breaker, when
  `max by (instance, breaker) (rag_breaker_state{state!="closed"}) == 1` for 5 m. Aggregated,
  so a breaker flipping between open and half-open keeps it pending. With a fallback configured,
  an open `llm` still fires, worded "answers come from the hosted fallback and leave the host".
  **`RagIngestFailed`:** a failed ingest job in the last hour. **`RagTargetDown`:** `up == 0` for
  5 m.

## Settled in this story's plan (2026-10-25)

What S3-7's plan left open, decided by the maintainer at this plan:

- **`rag-query`'s span per question is `rag.query`,** with `rag.retrieve` and the chat spans
  beneath it. **`rag-eval generate` gets none:** `rag-eval` never turns tracing on, and the
  shared code gets a no-op tracer unless a command did (D6).
- **`prometheus.yml` scrapes compose's app replicas,** discovered by DNS (`app`, port 9464);
  the README gives the loopback line for a run by hand (D10).

## Out of scope

- Jaeger, Prometheus and Alertmanager as running services (S3-9's `observability` profile).
  This story runs them by hand for its checks.
- Dashboards.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
RAG_LOG_FORMAT=console uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-7b/contexts.json \
  2>&1 | tee -i ~/dev/rag-s3-7b/tier1.log
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-7b/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy
# the lock: the new packages added and none moved; pypdf still 6.19.0
git diff -U0 main -- uv.lock | grep -E '^-(name|version) = '   # nothing

# traces into a throwaway Jaeger, the image S3-9 pins (DEC-24's table)
docker run --rm -d --name jaeger -p 127.0.0.1:16686:16686 -p 127.0.0.1:4318:4318 \
  jaegertracing/jaeger:2.22.0@sha256:836b967b5abcf1f7758066aa6b348619267a2f1459f72d472840570f9bd88a17
RAG_LOG_FORMAT=console OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 RAG_API_METRICS_PORT=9464 \
  uv run rag-serve 2>&1 | tee -i ~/dev/rag-s3-7b/serve.log
# one answer over HTTP, one dropped mid-stream: Jaeger shows both traces, the dropped one
# marked cancelled; an ingest's trace from a rebuild into a scratch collection, then the
# scratch generations go
RAG_LOG_FORMAT=console OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
  RAG_QDRANT_COLLECTION=s3_7b_scratch uv run rag-ingest --rebuild 2>&1 \
  | tee -i ~/dev/rag-s3-7b/ingest.log | tail -5
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_7b_scratch--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}

# alert rules and Alertmanager's file, as CI checks them (promtool and amtool from the images
# S3-9 pins; no host install)
PROMETHEUS=prom/prometheus:v3.15.0@sha256:efd719c99d83b060d9daefdcf00360461adf279f45ef5391f8d111892118753e
ALERTMANAGER=prom/alertmanager:v0.34.1@sha256:e9733bafb1bdef9b00e25a21f8f99dc26a22224bf16641ad754d1649f4c3357a
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool "$PROMETHEUS" check config /p/prometheus.yml
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool "$PROMETHEUS" test rules /p/rules.test.yml
docker run --rm -v "$PWD/deploy/alertmanager:/a:ro" --entrypoint amtool "$ALERTMANAGER" check-config /a/alertmanager.yml

docker stop jaeger
gh pr checks
```

## Review notes for the human

- **The span code in `stream_answer`:** nothing current across a `yield`, `end()` in a
  `finally`, and no callback passed to the model.
- **The disconnect test with instrumentation on** is the proof that the ASGI wrapper did not
  break DEC-18's cancel.
- **The alert rules' tests** show each alert firing and not firing, at each window.

## Plan (approved 2026-10-25)

Approved in plan mode. This section is the record to cross-check the work against; the full
plan, with the evidence for each finding, the tests, the files, the order of work and the
recipes, is `~/.claude/plans/pasted-content-id-0778-read-claude-md-toasty-kahan.md`. A
departure from it goes under "Deviation from plan".

**What the plan found, checking the design against the libraries as installed** (FastAPI
0.143.0 and opentelemetry-api 1.45.1 in the environment; the SDK, the exporter and the
instrumentation from uv's cache, read in place):

- **FastAPI 0.143 has OpenTelemetry of its own** (`fastapi/telemetry/`): server spans,
  operation spans, OTel log records and HTTP metrics. It switches on whenever a provider is
  configured, a global one included, and steps aside for contrib's middleware only when the
  stack was built after `instrument_app` ran. It wraps `receive`, and it ends a dropped
  client's server span as ERROR (`ConnectionError`), where a cancel here leaves the status
  unset. The maintainer kept the contrib instrumentation (Q1); `create_app` turns FastAPI's own
  off explicitly, which is required, not a precaution.
- **The contrib instrumentation 0.66b1 holds the settled claims:** the raw `receive` with
  `receive` excluded, a `send` wrapper that awaits nothing else, the server span current only
  around the app call and ended in a `finally`, the global `BackgroundTask` patch. Its default
  HTTP attributes are the old names (`http.method`, `http.url`).
- **Python reads `OTEL_EXPORTER_OTLP_TIMEOUT` in seconds,** where the spec says milliseconds. A
  dead collector costs at most that at exit, with warnings.
- **Thirteen new packages,** not about 17. FastAPI 0.143 already brings `opentelemetry-api`
  1.45.1, so without `--upgrade` the set resolves to 1.45.1 and 0.66b1.
- **Two pieces the design assumed are missing:** `answering.py` does not say why no LangChain
  callback handler is used, and ARCHITECTURE §3.5's spy-model test does not exist.

| # | Decision |
| --- | --- |
| D1 | **Dependencies:** `opentelemetry-sdk>=1.45.1`, `opentelemetry-exporter-otlp-proto-http>=1.45.1`, `opentelemetry-instrumentation-fastapi>=0.66b1`; `fastapi>=0.143`, a floor at the first release known to take `telemetry=`. No locked version moves; pypdf stays 6.19.0. **semconv gets no cap:** the SDK and the instrumentation require it at exactly 0.66b1, so it moves only with the whole set (the backlog line closes). |
| D2 | **`observability/tracing.py`:** on only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (`settings.TraceSettings`; a malformed value exits 2). A provider of our own, never the global one, with a batch processor and the OTLP/HTTP exporter, which reads the SDK's own variables. `service.name` is the command unless `OTEL_SERVICE_NAME` says otherwise. The SDK is imported only when tracing is on. It is flushed in the API's lifespan and shut down at each command's end. |
| D3 | **Helpers:** `span()` for synchronous code, current for its children; `start()` and `end()` for the answer stream, with an explicit parent, never current. A failure sets ERROR and `error.type` (the class name, no message). A cancel adds the event and attribute `rag.cancelled` and leaves the status unset. With tracing off they do nothing. |
| D4 | **The server span (Q1):** contrib's `instrument_app`, with `exclude_spans` `receive` and `send`, the health routes excluded, and no OTel metrics (a no-op meter provider). FastAPI's own telemetry is off in `create_app` (`tracing`, `metrics`, `logs` and `auto_configure` false). A test shows that one request makes exactly one server span. |
| D5 | **Spans:** `rag.retrieve` (`rag.embed_query`, `rag.vector_search`, `rag.rerank`), `chat {model}` per attempt, and `rag.ingest` (scan, prepare, embed, publish), with the attributes the Scope lists. No question, chunk or answer text. |
| D6 | **`rag-query` (Q2):** a `rag.query` span per question. **`rag-eval` is not traced:** the shared code gets a no-op tracer unless a command turned tracing on, and a test shows that `generate` exports nothing with the variable set. |
| D7 | **Context:** `serialized` copies it into its executor thread, and an ingest job attaches its request's. Log lines carry `trace_id` and `span_id`. |
| D8 | **No callback handler:** `answering.py` says why, and a spy model shows `stream_answer` passes the model no config. |
| D9 | **Alerts:** recording rules over seven windows for the four SLIs. Eight burn-rate alerts; `RagBreakerOpen` (a ticket for `llm` while the fallback answers); `RagIngestFailed`; `RagTargetDown`; and `RagTargetMissing` (`absent(up)`: exited replicas leave DNS discovery). Tests show each alert firing and quiet at each window. |
| D10 | **`prometheus.yml` (Q3):** DNS discovery of compose's `app` replicas, port 9464. **`alertmanager.yml`:** a null receiver, with an email one commented out. |
| D11 | **`deploy-config` (Q4):** promtool's two checks and `amtool check-config`, each image pinned by DEC-24's digest. |
| D12 | **Breaking:** a malformed `OTEL_EXPORTER_OTLP_ENDPOINT` exits 2 in `rag-serve`, `rag-query` and `rag-ingest` (`feat(observability)!`). |
| D13 | **Docs:** the README's Traces and Alerts, ARCHITECTURE's DEC-22 note; the records at close-out. |

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
