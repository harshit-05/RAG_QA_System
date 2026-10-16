# S3-7: Structured logs, traces, metrics and alerts

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | NFR-10; NFR-1 (verified); NFR-3; DEC-22; backlog: the two plural slips in `rag-ingest`'s log, the API's ingest jobs printing pypdf's warnings |
| **Depends on** | S3-6 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-7-observability`. Moderate risk: spans wrap the answer stream, where DEC-14's cancel guarantee lives, and the ASGI instrumentation wraps the disconnect path. |

## Goal

Every query and every ingestion run emits structured logs and OpenTelemetry spans with the
GenAI conventions: retrieval, rerank and generation latency, and token counts (NFR-10).
Prometheus can scrape the numbers the SLOs are written in, and alert rules fire on their
burn rate. NFR-1 is measured at the SRS's 100k scale, and NFR-3's throughput is documented.
None of it may reopen DEC-14's cancel window. Design: ARCHITECTURE.md §3.1, DEC-22.

## Scope

**If the plan-first step splits it** (design review 1), the seam is logs, metrics and the two
benchmarks against traces and alerts, the half that touches DEC-14's cancel guarantee.

- **Dependencies:**
  - main: `structlog`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`,
    `opentelemetry-instrumentation-fastapi` and `prometheus-client`;
  - versions as verified: 26.1.0, 1.45.1, 1.45.1, 0.66b1 and 0.26.0. Record what resolves;
    no locked version should move.
- **`rag_qa/observability/`**, which loads no model stack (an architecture test):
  - **`logs.py`:** `configure_logging()`.
    - structlog's `ProcessorFormatter` over the standard library's handlers, so ours and
      uvicorn's records are formatted alike: JSON when not a terminal or with
      `RAG_LOG_FORMAT=json`, console lines otherwise.
    - The request ID, the trace and span IDs and the API key's name (S3-8) are bound
      through context variables.
    - `rag-ingest`'s progress stays as readable as today.
    - **pypdf's logger is set to ERROR for the process** (backlog). The API's jobs stop
      printing its warnings.
  - **`tracing.py`:** an OTLP/HTTP exporter, active only when
    `OTEL_EXPORTER_OTLP_ENDPOINT` is set. It is flushed in the API's lifespan and at the end
    of each command.
    - **Helpers:** spans with an explicit parent, ended in a `finally`, never made current
      across a `yield`.
    - **Executor threads:** copy the context into them, since `serialized`'s
      `run_in_executor` does not.
  - **`metrics.py`:** the histograms, counters and gauges DEC-22 lists, on their own
    listener, `RAG_API_METRICS_PORT` (off unless set). Names follow the GenAI conventions
    where one exists (`gen_ai_client_token_usage`). The 429 counter comes with S3-8, which
    makes the 429s. Ingest metrics come from the API's jobs: a `rag-ingest` run has no
    listener, and reports in its log.
- **Spans:**
  - **The HTTP server span:** `FastAPIInstrumentor` with the `send` and `receive` spans
    excluded and the health routes left out. 0.66b1 has `exclude_spans`, and with `receive`
    excluded its ASGI middleware hands Starlette the raw `receive`, so the disconnect path
    is untouched (checked in design review 1). If the real-uvicorn disconnect test still
    shows a difference, write a server span by hand and record why.
  - **`rag.retrieve`,** with `rag.embed_query`, `rag.vector_search` (`db.system.name`
    qdrant, the collection, the mode) and `rag.rerank` (candidates, kept, `min_score`);
  - **`chat {model}`.** Its attributes are taken from the semantic-conventions package's own
    constants, never typed from memory: the operation, the provider, the request model,
    temperature and maximum tokens. Input and output tokens come from the last chunk's
    `usage_metadata`. An event marks the first token, and a cancel is recorded as one.
  - **`rag.ingest`,** with scan, prepare, embed and publish children; counts as attributes.
  - **No prompt or answer text** in any span, and **no LangChain callback handler**
    anywhere (`answering.py` says why).
- **`Done` gains `input_tokens` and `output_tokens`,** from `usage_metadata`. The SSE `done`
  event and tier 2's answers file record them; both changes are additive.
- **Logs.** The two plural slips ("Embedding 1 chunks", "1 documents, 1 chunks") are fixed
  (backlog). `rag-ingest` reports pages per second, chunks embedded per second and points
  written per second (NFR-3).
- **Alerts:** `deploy/prometheus/rules.yml`, plus `rules.test.yml` for `promtool`.
  - **What they cover:** availability (5xx on `/v1/query`), NFR-1 (retrieval p95),
    NFR-2a (`sources`) and NFR-2b (TTFT), multi-window burn rates as in the SRE workbook;
    plus a breaker open for more than 5 minutes, a failed ingest, and a scrape target down.
  - **`deploy/prometheus/prometheus.yml`** and **`deploy/alertmanager/alertmanager.yml`:**
    a null receiver, with an email example commented.
- **CI** (needs the `workflow` scope): a `deploy-config` job runs `promtool check config`
  and `promtool test rules`. promtool runs from the Prometheus v3.15.0 image, pinned by its
  index digest, the one S3-9's compose runs (DEC-24's table, caveat 30). Prometheus publishes
  no signatures, but Docker Hub and Quay serve the same digest, and the image's `promtool`
  is byte-identical to the release tarball's (design review 2).
- **NFR-1.** `scripts/bench_retrieval.py` fills a scratch collection on the local server
  with 100k synthetic unit vectors, plus sparse vectors, and measures the query embedding
  plus an exact search, dense and hybrid, over 200 real-looking questions, at k=20: the
  reranker's candidates, stricter than NFR-1's 10 (design review 1). It reports p50 and p95
  and deletes its collection. Record the numbers in the README's new "Capacity planning"
  section, beside NFR-3's table.
- **NFR-2a** (DEC-27, design review 1). The same script times the real retrieval half, the
  rerank included, over the 25 golden questions with the model idle, and reports the p95 of
  what the `sources` event waits for. The target is under 2 s; record it beside NFR-1's.
- **Tests:**
  - an in-memory span exporter shows the spans, their parents and the GenAI attributes;
  - with tracing on, DEC-14's forced-cancel tests and the real-uvicorn disconnect test pass,
    every span ends, and no "Failed to detach context" is logged;
  - the metrics move as answers, refusals and ingests happen.
- **README:** Observability (how to point at a collector, what each metric means), Capacity
  planning, and the alert rules in a sentence each.

## Out of scope

- Jaeger, Prometheus and Alertmanager as running services (S3-9's `observability` profile).
  This story runs them by hand for its checks.
- Dashboards.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-7/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-7/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

# traces into a throwaway Jaeger (the image S3-9 pins), metrics on their own port
docker run --rm -d --name jaeger -p 127.0.0.1:16686:16686 -p 127.0.0.1:4318:4318 jaegertracing/jaeger:2.22.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 RAG_API_METRICS_PORT=9464 uv run rag-serve 2>&1 | tee -i /tmp/s3-7-serve.log
# one answer over HTTP, one dropped mid-stream: Jaeger shows both traces, the dropped one
# marked cancelled; curl -s 127.0.0.1:9464/metrics | grep rag_ shows ttft, sources, tokens
# JSON lines, and throughput in the summary: a rebuild into a scratch collection, since an
# up-to-date ingest embeds nothing; then the scratch generations go
RAG_LOG_FORMAT=json RAG_QDRANT_COLLECTION=s3_7_scratch uv run rag-ingest --rebuild | tail -3
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_7_scratch--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}

# alert rules, as CI runs them (promtool from the image S3-9 pins; no host install)
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool prom/prometheus:v3.15.0 check config /p/prometheus.yml
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool prom/prometheus:v3.15.0 test rules /p/rules.test.yml

uv run python scripts/bench_retrieval.py    # p95 < 300 ms at 100k, k=20, dense and hybrid;
                                            # NFR-2a: the golden questions' p95 < 2 s
docker stop jaeger
gh pr checks
```

## Review notes for the human

- **The span code in `stream_answer`:** nothing current across a `yield`, `end()` in a
  `finally`, and no callback passed to the model.
- **The disconnect test with instrumentation on** is the proof that the ASGI wrapper did not
  break DEC-18's cancel.
- **The benchmark measures what NFR-1 says**, the embedding plus the search, without the
  rerank, and says so in its output.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
