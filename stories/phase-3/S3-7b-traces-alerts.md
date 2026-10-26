# S3-7b: Traces and alerts

| | |
| --- | --- |
| **Status** | Done 2026-10-26 (PR #26, merged 2026-10-26 as `e6e1f26`; CI green on `main` after it, run 40238581649). Plan approved 2026-10-25 (below, "Plan"); on Opus 5.5 at max effort (Deviation). Built and committed on 2026-10-25 (`845133b` … `0815ac1`). The first review (2026-10-25, on `0815ac1`) found one false claim (a span's exception text), six gaps in what the tests would catch and two smaller things; the second (2026-10-26, on `0a11270`) found an alert that paged a healthy idle server, and a check that gated no merge, for the maintainer. The fixes are on the PR (`cc11d55` … `98ddc6e`). CI is green on `98ddc6e` (run 40235082842: `check`, `eval-retrieval`, `deploy-config`) and on the close-out record, `492b66f` (run 40237334537). The maintainer added `deploy-config` to `main`'s ruleset after the merge (2026-10-26). An answer abandoned before its first token was invisible to every alert: fixed after the merge, in PR #27, merged 2026-10-26 as `12f4768` ("After the merge"). |
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
  - **What they cover:** availability (the 503 refusals of `/v1/query`, and answers that end
    in an `error` event; no metric counts an unexpected 500, the second review), NFR-1
    (retrieval p95), NFR-2a (`sources`) and NFR-2b (TTFT), multi-window burn rates as in the
    SRE workbook; plus a breaker open for more than 5 minutes, a failed ingest, a
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
docker run --rm -v "$PWD/deploy/prometheus:/p:ro" --entrypoint promtool "$PROMETHEUS" test rules /p/rules.test.yml /p/rules.causes.test.yml
docker run --rm -v "$PWD/deploy/alertmanager:/a:ro" --entrypoint amtool "$ALERTMANAGER" check-config /a/alertmanager.yml

docker stop jaeger
gh pr checks
```

## Verification record (2026-10-25 and 2026-10-26)

Rows 1 and 4–7, 15–17 and the audit were run again on 2026-10-26 on `98ddc6e`, the PR's head
after both reviews; their first runs, on 2026-10-25 before the commits, gave the same results
with the suite then at 1578 tests (98.46%). Rows 2, 9–13 and 18 ran in the maintainer's
terminal on 2026-10-25, on the tree committed as `845133b` … `0815ac1`, each logged with
`tee -i` under `~/dev/rag-s3-7b/`, where Jaeger's two exports and their span trees are kept
(`jaeger-rag-serve.json`, `jaeger-rag-ingest.json`, `traces.txt`).

| # | Command | Result |
| --- | --- | --- |
| 1 | `uv run rag-eval check` | exit 0: the six parts `ok`, every floor met; "The committed run matches this checkout" |
| 2 | `RAG_LOG_FORMAT=console uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-7b/contexts.json` | 0.950 (19 of 20) / 0.796 / 0.858, every floor met; 25 questions' contexts written |
| 3 | `diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-7b/contexts.json` | IDENTICAL; both `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc`. The first review ran it again on `0815ac1`; since then only `scrub.py`, the export path of `tracing.py` and `deploy/` changed, and `rag-eval` never traces |
| 4 | `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q` | 1591 passed, 18 skipped, 5 xfailed; coverage 98.47%; `HF_HOME` stayed empty |
| 5 | `uv run ruff check && uv run ruff format --check && uv run mypy` | all checks passed; 77 files formatted; no issues in 38 source files |
| 6 | `RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q` | 11 passed (the fallback module's skip is the 1 skipped); the live collections unchanged: `rag_qa`'s two, the alias on `…7f20891e` |
| 7 | the lock against `main`: `git diff -U0 main -- uv.lock \| grep -E '^-(name\|version) = '` | nothing: 13 packages added, none removed or moved. Added: `opentelemetry-sdk`, `-exporter-otlp-proto-http`, `-exporter-otlp-proto-common` and `-proto` 1.45.1; `-instrumentation`, `-instrumentation-asgi`, `-instrumentation-fastapi`, `-util-http`, `-exporter-http-transport` and `-exporter-otlp-common` 0.66b1; asgiref 3.13.0, googleapis-common-protos 1.76.0, wrapt 2.5.1. pypdf 6.19.0; `uv lock --check` resolves 194. CI's Audit command, run locally on 2026-10-26: "No known vulnerabilities found, 11 ignored" |
| 8 | `uv sync --extra fallback`, the fallback and tracing tests, then `uv sync` (2026-10-25) | the new test of a real `ChatOpenAI`'s chat span passed (`openai`, `openai/gpt-oss-120b`, temperature 0.0, max tokens 1024, `api.groq.com`:443, `rag.fallback` true, `connection_failed`). One older test failed, on `main`'s tree too: a stale assertion (Discovered). The environment back to CI's |
| 9 | Jaeger 2.22.0 by digest (the story's `docker run`) | up; its query API answers at `/api/v3/`, not `/api/services` (Discovered) |
| 10 | `RAG_LOG_FORMAT=console OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 RAG_API_METRICS_PORT=9464 uv run rag-serve 2>&1 \| tee -i …/serve.log`, one answer, then one dropped after three tokens | `Sending traces to http://127.0.0.1:4318 over OTLP/HTTP, as service rag-serve.`; each request's lines carry `request_id`, `trace_id` and `span_id`. The answer: 164.6 s, first token at 158.3 s, 1,515 input and 33 output tokens. The drop: "Answer cancelled: its client disconnected". A clean shutdown, no exporter warning; `grep -c 'Failed to detach'` 0 in all three logs |
| 11 | `curl -s 127.0.0.1:9464/metrics \| grep -E '^(rag_answers_total\|rag_ttft_seconds_count)'` (before Ctrl-C) | `rag_answers_total` answered 1, cancelled 1, failed 0; `rag_ttft_seconds_count` 2 |
| 12 | Jaeger's `/api/v3/traces?query.service_name=rag-serve…` | services `rag-serve`, `rag-ingest`, `jaeger`. Each answer one trace of six spans: `POST /v1/query` (SERVER, the contrib scope, status unset) → `rag.retrieve` (`rag.chunks` 5) → `rag.embed_query`, `rag.vector_search` (CLIENT; `qdrant`, the generation, `dense`) and `rag.rerank` (20 → 5); and `chat mistral` (CLIENT; `ollama`, `mistral`, temperature 0, max tokens 512, `time_to_first_chunk` 157.3 s, usage 1515/33, attempt 1, not the fallback; event `rag.first_token`). The dropped answer's chat span: `rag.cancelled`, its event at 165.1 s after the first token's at 164.3 s, no usage; its server span unset. No question or answer text anywhere; no `server.address` on the Ollama span (Discovered). The span's first token counts from the attempt's request, and the `done` event's from the request's arrival (S3-7's second review), so the event's is about a second later: the retrieval |
| 13 | `RAG_LOG_FORMAT=console OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 RAG_QDRANT_COLLECTION=s3_7b_scratch uv run rag-ingest --rebuild …`, its trace, then the scratch delete | `rag.ingest` (104.1 s; a full rebuild; 3 files, 561 pages, 1,708 chunks and points, 3 added) with `scan` 24 ms, `prepare` 32.0 s, `embed` 64.4 s, `publish` 0.6 s: each the summary's throughput line ("17.5 pages/s loaded, 26.5 chunks/s embedded, 2,850 points/s written"). The delete: `{"result":true}`; the scratch alias went with its collection; `rag_qa`'s two generations and alias unchanged; `var/` holds `ingest.lock` only |
| 14 | the six commits' split, in a throwaway index, and `main`'s code over the new lock (a scratch tree) | the six cover all 37 paths, `.claude/` left out; `main`'s suite over the new lock: 1490 passed, 20 skipped (the hygiene test outside a checkout), 5 xfailed; ruff and mypy clean on it |
| 15 | `promtool check config` (the pinned image) | SUCCESS: 1 rule file, 41 rules |
| 16 | `promtool test rules /p/rules.test.yml /p/rules.causes.test.yml` | SUCCESS, both files, 24 s |
| 17 | `amtool check-config` (the pinned image) | SUCCESS: global config, route, 0 inhibit rules, 1 receiver, 0 templates |
| 18 | `docker stop jaeger` | stopped by the maintainer after the traces were read |
| 19 | `gh pr checks` | green on `0815ac1`, on `0a11270` (run 40216686158) and on `98ddc6e` (run 40235082842): `check` 4m5s, `deploy-config` 27s, `eval-retrieval` 2m32s |

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

The first item is the first review's; the rest are the implementation's (2026-10-25).

- **The contrib server span records an unhandled exception's message and stacktrace,** which
  the story's own rule forbids ("`error.type` to the class name only", "no prompt or answer
  text in any span"). Found by a probe: a route raising `RuntimeError(f"…{question}")` put the
  question, and a stacktrace with absolute host paths, in the span's `exception` event.
  - **Where it comes from:** `FastAPIInstrumentor`'s wrapper (`instrumentation/fastapi/__init__.py`
    :361-364) records the exception and writes `"Type: message"` into the status description,
    and the ASGI middleware records it again (`instrumentation/asgi/__init__.py:673`). No
    switch turns either off, and an instrumentation added later may do the same.
  - **Why it matters:** a trace goes to a collector that is not this host (`OTEL_EXPORTER_OTLP_ENDPOINT`
    is any URL, authenticated with `OTEL_EXPORTER_OTLP_HEADERS`); the log stays on it, and keeps
    the full traceback. DEC-18 keeps host paths off HTTP, and S3-6 found that pydantic's error
    text repeats config values. "No path today" is not protection against the exception nobody
    planned for.
  - **The decision (maintainer, 2026-10-25):** scrub at the exporter, not narrow the claim.
    `rag_qa/observability/scrub.py`'s `ScrubbingExporter` wraps the OTLP exporter in
    `configure()`: each `exception` event keeps `exception.type` and loses the rest, and the
    status keeps its code and loses its description. It covers every span. Not
    `SpanLimits(max_event_attributes=0)`, which also drops `exception.type`, and any attribute
    an event gains later.
  - **A correction found while building it:** the status description is also written (by the
    FastAPI wrapper), but in the probe the 500 response's own status replaced it, so the event
    was the leak. The scrub covers the description for the cases where nothing replaces it.
  - **Tests:** a negative control (the unwrapped exporter shows the message), the same route
    through a provider that builds the wrapper, the real `configure()` path into a fake
    collector (no byte of the message, the stacktrace or a host path in the protobuf), and unit
    tests that the rest of a span, its drop counts included, is untouched. Five mutants, each
    caught.

- **FastAPI 0.143 has OpenTelemetry of its own** (the plan's first finding, D4; ARCHITECTURE's
  DEC-22 note). Off in `create_app` by explicit flags. A FastAPI release that changes its
  defaults, or how it detects the contrib middleware, fails the test of one server span per
  request with a global provider set.
- **Python reads `OTEL_EXPORTER_OTLP_TIMEOUT` in seconds** (exporter 1.45.1, `float(raw)`,
  default 10), where the specification says milliseconds: a value written for the
  specification, 10000, means about 2.8 hours. The README says seconds.
- **Two pieces the design cited were missing:** `answering.py`'s reason for no LangChain
  callback handler, and ARCHITECTURE §3.5's spy-model test. Both are in (D8).
- **The server span's HTTP attributes are the old names,** `http.method`, `http.url`,
  `net.peer.ip` (seen in Jaeger): the contrib default unless
  `OTEL_SEMCONV_STABILITY_OPT_IN=http`. The README says so.
- **langsmith 0.12, through langchain-core, imports the SDK's `TracerProvider`** in every
  command once the SDK is installed, as a type (about 20 ms; it configures nothing). So the
  architecture tests hold out the exporter, the instrumentation and a global provider, and
  check that `tracing.py` imports no SDK at module level (Deviation).
- **pytest's `caplog.text` in a fixture's teardown holds the teardown's records only.** The
  `traced` fixture's first "Failed to detach context" check read it there and saw nothing
  of the test itself: a mutant that makes the chat span current across the stream's yields
  passed it. It reads `caplog.get_records()` for each phase now, and that mutant fails the
  forced window's tests.
- **Ollama's chat spans carry no `server.address` or `server.port`.** `mistral_ollama` names
  no `base_url`, so the ollama client takes `OLLAMA_HOST` or its default, kept in a private
  attribute, and `chat_attributes` reads only a URL the entry names. A `base_url` in
  `config.yaml` would move the fingerprint, so it stays so. The fallback's span has
  `api.groq.com`:443 (a test).
- **Jaeger 2.22 serves its query API at `/api/v3/`** (`/api/v3/services`;
  `/api/v3/traces?query.service_name=…&query.start_time_min=…&query.start_time_max=…`, in
  OTLP JSON). The plan's `/api/services` answers 404. S3-9's checks read it there.
- **Two of the thirteen new packages were days old when locked:** asgiref 3.13.0 (uploaded
  2026-10-23) and googleapis-common-protos 1.76.0 (2026-10-22), both transitive. DEC-24's age
  rule is for the pinned images and binaries; nothing applies one to the lock. The audit
  finds nothing.
- **A stale assertion in `tests/test_fallback_openai.py`:**
  `test_the_fallback_streams_the_answer_and_sends_what_the_entry_says` asserts that no
  `stream_options` is sent, and `groq_fallback`'s `stream_usage: true` (S3-7) sends one. It
  fails with the `fallback` extra installed, on `main` too; CI installs no extra and skips
  the file. Out of scope: a backlog line.

## First review (2026-10-25, on `0815ac1`)

Reviewed as the first reviewer, against the PR's head, in scratch copies of it (never in the
repository). The live Qdrant was read, tier 1 only; the running Jaeger was not touched. No
blocker; DEC-14's cancel guarantee holds with tracing on. One false claim, six gaps in what the
tests would catch, two smaller things, all fixed on the PR.

**Fixed:**

1. **The server span's exception text** (Discovered): scrubbed at the exporter.
2. **The alert-rule tests missed 6 of 19 mutations of `rules.yml`.** All 13 others (every
   threshold, label and `le`) were caught. The survivors: a fast-burn pair's short window
   (5 m → 30 m on NFR-1), a slow-burn pair's short window (6 h → 2 h on first token), the 5 m
   and 1 h recording rules (the SLI test read the 30 m one only), `for: 1m` for `5m` on
   `RagBreakerOpen` and `RagTargetDown` (a 5-minute evaluation interval cannot tell them
   apart), and `up == 0` without `{job="rag"}`. Now: the three latency fast-burn scenarios
   check quiet five minutes after the burn (the 30 m and 1 h windows still burn, the 5 m one
   is clean); the four slow-burn scenarios check that the 3 d and 6 h pair alone fires at 27 h;
   `rules.causes.test.yml` evaluates every minute (quiet at 4 minutes, firing at 5); an
   `up{job="other"}` series; and a pytest holds the 28 recording rules to one expression per
   SLI across the seven windows. All 19 mutations are now caught (promtool or the pytest).
3. **Nothing tested the API lifespan's span flush,** the only thing that sends the last spans
   on a SIGTERM. A test serves a request into a batch processor aimed at the fake collector,
   exits the lifespan without `shutdown()`, and finds the spans there; without the flush it
   fails.
4. **`tracing.py` said the SDK "is never imported" with tracing off.** langsmith imports its
   `TracerProvider` as a type whenever the SDK is installed (ARCHITECTURE.md and a test
   comment already said so). The docstring now says what our code imports.
5. **S3-9's note:** with a dead collector, `rag-serve`'s shutdown waits up to the exporter's 10 s
   after `GRACEFUL_SHUTDOWN_S` (5 s); measured 6.3 s against a closed port and 10.0 s against
   one that never answers. A `docker stop`'s default 10 s can SIGKILL it and lose the flush.
   S3-9's "Signals" item now says to set `stop_grace_period` and check it.
6. **Nits:** the README says `http.url` includes the query string; `pyproject.toml`'s comment
   says `telemetry=` exists from FastAPI 0.142.0 (0.143.0 is what the story checked).

**Held:** the gates on `0815ac1` (1578 passed, coverage 98.46%; `rag-eval check` exit 0; tier 1
IDENTICAL to S3-2's, `5a36e2f9…`; CI green); the lock (13 packages added, none moved, pypdf
6.19.0); the image pins (digests equal DEC-24's and Quay's; releases 19 to 38 days old); 16 of
18 mutations of the tracing code caught, the other two an equivalent mutant (the chat span's
explicit parent equals the current context) and item 3; a normal answer's spans hold no
question, chunk, token or header; the README's claims about `OTEL_EXPORTER_OTLP_TIMEOUT`
(seconds), `OTEL_SEMCONV_STABILITY_OPT_IN=http` and the burn-rate percentages.

**After the fixes:** 1591 passed, 18 skipped, 5 xfailed, coverage 98.47%; ruff, mypy and
`uv lock --check` clean (the lock unchanged); `rag-eval check` exit 0; `promtool check config`,
both `promtool test rules` files and `amtool check-config` pass from the pinned images; the new
tracing and rule tests passed ten repeats in a row.

**For the close-out:** the Verification record, Discovered and Deviation are still the
implementer's to write.

## Second review (2026-10-26, Opus 5.5, on `0a11270`)

**1 alert defect, fixed on the PR with four scenarios; 1 gating gap, for the maintainer.
Nothing blocks DEC-14's cancel path.** Run on the PR's head after the maintainer's push
(local and origin in sync), with the first review's record treated as claims to check. The
maintainer's Qdrant was stopped and left so: the server tests ran on a throwaway Qdrant from
compose's digest (`review-s3-7b-qdrant`, port 16333), the probes and mutations in scratch
copies of `HEAD`, promtool and amtool from the pinned images. No model was run.

- **Reproduced on `0a11270`:** ruff, format, mypy and `uv lock --check` clean; **1591 passed,
  18 skipped, 5 xfailed**, coverage **98.47%**, the HF cache empty; the 11 `qdrant_server`
  tests; `rag-eval check` exit 0; `promtool check config` (41 rules), both rule-test files and
  `amtool check-config`; CI green (run 40216686158: `check`, `eval-retrieval`,
  `deploy-config`). The lock: 13 packages added, none moved, pypdf 6.19.0. Tier 1 was not
  re-run: since the first review's IDENTICAL run on `0815ac1`, only `scrub.py` and the export
  path of `tracing.py` changed, and `rag-eval` never traces.
- **DEC-14 with tracing on:** no span is current across a `yield`; each chat span ends in a
  `finally` of its own, which a second cancel inside the stream's close cannot skip; every
  real-uvicorn drop runs again with the contrib middleware and an SDK provider, then requires
  every span ended and no context failed to detach; `rag-query`'s Ctrl-C marks its question
  and its attempt cancelled.
- **Six guards, each caught by mutation after a clean control:** the scrubbing exporter, the
  executor thread's copy of the request's context, FastAPI's telemetry flags, the chat span's
  `end()`, the ingest job's attach, and `receive` left out of the spans.
- **The claims, against the installed source:** FastAPI 0.143's explicit `telemetry=` values
  override `FASTAPI_OTEL_AUTO_CONFIGURE`; the rules' `le` strings ("0.3", "2.0", "300.0") are
  prometheus_client's own exposition (S3-7's real scrape); every burn-rate threshold and the
  README's percentages (2%, 5%, 10%, 10%); a malformed endpoint exits 2 in all three
  commands without printing it.

Finding, fixed:

1. **`RagBreakerOpen` paged a healthy idle server after a model outage, until the next
   question.** A breaker is open for 30 s, then half-open until a question probes it, and a
   half-open breaker with no probe out refuses nothing. Readiness probes Qdrant's breaker
   (`refresh()` reads the alias through it) but never the model's or the fallback's
   (`llm_reachable()` touches no breaker), and the rule counted half-open as not closed. On
   the real app, with the llm breaker opened and Ollama answering again: an hour idle,
   `/v1/health` 200 ready, `refusal_s()` None, the scrape `rag_breaker_state{breaker="llm",
   state="half-open"} 1`; the next question was let through, and closed it. promtool, with
   that shape, fired the page at 10, 30 and 59 minutes, with a description ("refused with
   503") false in that state. The tests held a breaker `open` for 9 minutes and more, which
   the real one never is. Both rules now also need the breaker open within the last 4
   minutes (`max_over_time(…{state="open"}[4m])`), under the 5 of `for`, so the opening that
   began the outage never pages alone. `rules.causes.test.yml` gains four scenarios in the
   breaker's real shape: idle after the outage, quiet at 8, 30 and 60 minutes, with and
   without a fallback; probes failing at a question every 2 minutes, page (or ticket) at 8
   minutes and not at 7. Each rule put back fails its own idle scenario. The cost: an outage
   whose questions come more than 4 minutes apart does not page through this alert; the
   availability alerts still do, on the failed answers (the 6 h and 30 m pair). README's row
   and ARCHITECTURE's note say so.

For the maintainer:

2. **`deploy-config` gates no merge yet.** Ruleset 25561360 requires `check` and
   `eval-retrieval` only, and S3-9 was to add `deploy-config` with `container`: until then, a
   broken rule or a failing promtool test merges. The job runs on every pull request, so
   requiring it is safe. The second review hands over a PUT, built from the ruleset as it is,
   that adds it after S3-7b's merge; STATUS and S3-9 now say so (a PUT replaces the whole
   list, so S3-9 builds its own from the ruleset as it then is).

**After the fix:** 1591 passed, 18 skipped, 5 xfailed, coverage 98.47%; the rules' pytest;
`promtool check config` and both rule-test files from the pinned image; ruff clean;
`rag-eval check` exit 0.

**Nits left:**

- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` alone turns nothing on, since tracing keys on the
  generic variable; set beside it, the SDK sends the spans there while the start-up line
  names the generic one. No doc says so.
- NFR-2b's SLI has no answer cancelled before its first token: a client that gives up after
  300 s leaves no TTFT and counts in neither SLI, so slower answers make it look better.
- The Scope says availability counts "5xx on `/v1/query`"; the rules count the 503 refusals
  and the failed answers, and an unexpected 500 is seen by none, there being no HTTP status
  metric.

## Deviation from plan

- **The model:** Opus 5.5 at max effort, in place of Fable, which is unavailable (CLAUDE.md).
- **The architecture tests guard the exporter and the instrumentation, not the SDK** (D2: "the
  SDK is imported only when tracing is on"). langsmith imports it in every command
  (Discovered); our code still imports it only in `configure()`, and a test says so.
- **No `end()` helper** (D3): the answer stream ends each chat span with the span's own `end()`,
  in a `finally` of its own.
- **The answer's and the ingest's span tests are in `test_tracing.py`;** the plan's file list
  named `test_api.py` and `test_ingest.py`, which did not change.
- **The alert-rule tests run at a 5-minute evaluation interval** (D9 named none). At 1 minute,
  `rules.test.yml` took 3 min 56 s; at 5, with 1-minute samples only where a 5-minute window is
  read and the slow scenarios compressed to about a day, 22 s. The first review found what a
  5-minute interval cannot tell apart (`for: 1m` against `5m`) and added
  `rules.causes.test.yml`, at 1 minute, and a pytest over the 28 recording rules.
- **A scrubbing exporter, `observability/scrub.py`,** which the plan did not have: the contrib
  spans record an exception's message (the first review; the maintainer chose to scrub at the
  exporter).
- **`RagBreakerOpen` also needs the breaker opened within the last 4 minutes** (the second
  review), where S3-7's plan settled `state!="closed"` for 5 minutes: a half-open breaker on
  an idle server paged.
- **`deploy-config` joins `main`'s ruleset at this story's merge,** not at S3-9 (DEC-24): the
  maintainer, at the second review.

## After the merge: answers abandoned before their first token (2026-10-26)

The second review's nit that NFR-2b's SLI leaves out answers cancelled before their first
token is a blind spot, not only a bias. Three exclusions combine: the availability SLI drops
cancelled answers; `rag_ttft_seconds` is observed only at a first token; and a cancel calls
`tried.abandoned()` ("nothing learned about the service"), so the breaker never trips. Only a
client that waits out Ollama's 600 s read timeout produces a failure. So a hung model whose
users give up first is a whole outage no alert sees: a promtool probe of `rules.yml`, two hours
of questions all cancelled before a first token, kept `RagAvailabilityFastBurn`, `…SlowBurn`,
`RagFirstTokenLatencyFastBurn`, `…SlowBurn`, `RagBreakerOpen` and `RagTargetDown` quiet. A
retrieval that hangs is the same for the `sources` SLI.

**The maintainer's decision (2026-10-26):** fix it before S3-8, on
`fix/s3-7b-abandoned-answers`, the `sources` stage included. A histogram of how long an answer
had waited when its client left before its first token; each SLI counts those past its
objective as slow, and leaves younger ones out of both sides.

**As built** (PR #27, `d685104` … `5b72815`; on Opus 5.5 at max effort, as S3-7b ran):

- **`rag_abandoned_seconds{waiting_for}`** (`metrics.py`): `sources`, then `first_token` once
  the sources event went out. It has the first-token histogram's buckets, which hold both
  objectives, and both label sets are there at zero from the start: an absent series would
  empty the SLI's sum and silence its alert.
- **Recorded in `Slot.release()`,** the one place every ending passes once, a drop between
  events included. `Slot.begin(received_at)` sets the outcome, the stage and the arrival time
  together, so no age is taken from an unset time; the route moves the stage at `sources` and
  at the first token. Synchronous, with no new `await`: DEC-14's window is untouched. Answers
  refused before their stream began stay out, as they do of `rag_answers_total`.
- **The SLIs** (`rules.yml`, 14 recording rules): slow = (events over the objective + answers
  abandoned past it) / (all events + answers abandoned past it). NFR-2a reads the `sources`
  stage at 2 s, NFR-2b both stages at 300 s. The alerts themselves are unchanged.

| # | Check | Result |
| --- | --- | --- |
| 1 | `uv run rag-eval check` | exit 0: the committed run matches this checkout |
| 2 | `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q` | 1594 passed, 18 skipped, 5 xfailed; coverage 98.47%; `HF_HOME` stayed empty |
| 3 | `uv run ruff check && uv run ruff format --check && uv run mypy` | all checks passed; 77 files formatted; no issues in 38 source files |
| 4 | the lock | `uv.lock` and `pyproject.toml` unchanged; `uv lock --check` resolves 194 |
| 5 | `promtool check config`, `promtool test rules` on both files, `amtool check-config` (the pinned images) | SUCCESS, 41 rules; SUCCESS, both files; SUCCESS |
| 6 | the real drops (`test_api_uvicorn.py`, tracing off and on) | a drop during the prefill counts once under `first_token`, and one during retrieval once under `sources`; a drop mid-stream, and the forced window's (its blocked event is the first token, already observed), count none |
| 7 | promtool's new scenarios | a hung model whose clients leave after 5 minutes: quiet at 15 minutes, the first-token page at 35, which the merged rules never gave; a hung retrieval: the sources page at 35 minutes, and no first-token page; clients that leave sooner: the SLI at 0, both alerts quiet. The arithmetic: sources 0.5 and first token 0.625, values a wrong stage, objective or side would change |
| 8 | mutants, in scratch copies | 4 of the code (no record at release, no stage at `sources`, no clear at the first token, failed answers recorded too) and 6 of the rules (the `sources` stage dropped, its wait read at 300 s, the first token's at 2 s, the `first_token` stage alone, no abandoned answers below the line, the rule as merged), each caught |
| 9 | the `fix(api)` commit alone: `main` plus its five files, in a scratch tree | 1592 passed, 20 skipped (the hygiene test outside a checkout), 5 xfailed; ruff and mypy clean; promtool green on the rules as merged |
| 10 | the live check: a question dropped during the prefill on the real API | not run: optional, and the maintainer's call |
| 11 | `gh pr checks 27` | green on `5b72815` (run 40280659911): `check` 5m10s, `deploy-config` 30s, `eval-retrieval` 3m7s |

**Deviation from the fix's plan:** `Slot.begin()`, which the plan did not name, and the
route's own first-token flag gave way to `slot.waiting_for`. The rule tests' existing latency
scenarios carry the new series at zero, which their SLIs now need.

The same probe found one more thing, for the backlog: on a quiet host (a question every 20
minutes), one failed answer pages `RagAvailabilityFastBurn`, since with so few events one
failure is 14.4 times the budget.

**Merged** (2026-10-26, PR #27, merge commit `12f4768`). CI was green on the PR's last head,
`d3ed26a` (run 40283612574), and is green on `main` after the merge (run 40284703179:
`check`, `deploy-config`, `eval-retrieval`). No Dependency Graph run: the PR changed no
dependency. The live check on the real API (row 10) was optional, and was not run.
