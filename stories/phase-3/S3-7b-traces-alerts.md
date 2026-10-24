# S3-7b: Traces and alerts

| | |
| --- | --- |
| **Status** | Todo |
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
  `opentelemetry-instrumentation-fastapi`; versions as verified: 1.45.1, 1.45.1 and 0.66b1, about
  17 new packages, among them asgiref, wrapt and googleapis-common-protos (protobuf is locked
  already). `opentelemetry-semantic-conventions` 0.66b1 came with S3-7. Record what resolves; no
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
    as in the SRE workbook; plus a breaker open for more than 5 minutes, a failed ingest, and a
    scrape target down.
  - **`deploy/prometheus/prometheus.yml`** and **`deploy/alertmanager/alertmanager.yml`:**
    a null receiver, with an email example commented.
- **CI** (needs the `workflow` scope): a `deploy-config` job runs `promtool check config`
  and `promtool test rules`. promtool runs from the Prometheus v3.15.0 image, pinned by its
  index digest, the one S3-9's compose runs (DEC-24's table, caveat 30). Prometheus publishes
  no signatures, but Docker Hub and Quay serve the same digest, and the image's `promtool`
  is byte-identical to the release tarball's (design review 2).
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

## Open, for this story's plan

- `rag-query`'s span per question: its name, and whether `rag-eval generate` gets one.
- `prometheus.yml`'s scrape target: the address S3-9's compose gives the app, or loopback for a
  run by hand.

## Out of scope

- Jaeger, Prometheus and Alertmanager as running services (S3-9's `observability` profile).
  This story runs them by hand for its checks.
- Dashboards.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-7b/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-7b/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

# traces into a throwaway Jaeger, the image S3-9 pins (DEC-24's table)
docker run --rm -d --name jaeger -p 127.0.0.1:16686:16686 -p 127.0.0.1:4318:4318 \
  jaegertracing/jaeger:2.22.0@sha256:836b967b5abcf1f7758066aa6b348619267a2f1459f72d472840570f9bd88a17
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 RAG_API_METRICS_PORT=9464 uv run rag-serve 2>&1 \
  | tee -i ~/dev/rag-s3-7b/serve.log
# one answer over HTTP, one dropped mid-stream: Jaeger shows both traces, the dropped one
# marked cancelled; an ingest's trace from a rebuild into a scratch collection, then the
# scratch generations go
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 RAG_QDRANT_COLLECTION=s3_7b_scratch \
  uv run rag-ingest --rebuild 2>&1 | tee -i ~/dev/rag-s3-7b/ingest.log | tail -5
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_7b_scratch--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}

# alert rules, as CI runs them (promtool from the image S3-9 pins; no host install)
PROMETHEUS=prom/prometheus:v3.15.0@sha256:efd719c99d83b060d9daefdcf00360461adf279f45ef5391f8d111892118753e
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool "$PROMETHEUS" check config /p/prometheus.yml
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool "$PROMETHEUS" test rules /p/rules.test.yml

docker stop jaeger
gh pr checks
```

## Review notes for the human

- **The span code in `stream_answer`:** nothing current across a `yield`, `end()` in a
  `finally`, and no callback passed to the model.
- **The disconnect test with instrumentation on** is the proof that the ASGI wrapper did not
  break DEC-18's cancel.
- **The alert rules' tests** show each alert firing and not firing, at each window.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
