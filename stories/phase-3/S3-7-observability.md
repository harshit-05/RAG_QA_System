# S3-7: Structured logs, metrics and the capacity benchmarks

| | |
| --- | --- |
| **Status** | In progress (plan approved 2026-10-23): on `feat/s3-7-observability`. The plan splits the story at design review 1's seam: the traces and the alerts are S3-7b. On Opus 5.5 at max effort (Deviation). |
| **Closes** | NFR-10 (logs and metrics); NFR-1 (verified); NFR-2 (a), measured (DEC-27); NFR-3; DEC-22 (logs and metrics); backlog: the two plural slips in `rag-ingest`'s log (S3-4's "Published …: 1 documents" is the second of them), the API's ingest jobs printing pypdf's warnings, and the traceback an open breaker logs mid-stream (S3-6) |
| **Depends on** | S3-6 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-7-observability`. Moderate risk, less than before the split: the metrics and the token counts sit in the query route and `stream_answer`, where DEC-14's cancel guarantee lives, though nothing new is awaited there; and `rag-ingest`'s progress moves to stderr. |

## Goal

Every query and every ingestion run emits structured logs, JSON when read by a machine, and
the API exposes the numbers the SLOs are written in for Prometheus to scrape: retrieval,
rerank and generation latency, and token counts (NFR-10's logs and metrics). NFR-1 is measured
at the SRS's 100k scale, NFR-2's (a) over the golden questions, and NFR-3's throughput is
documented. None of it may reopen DEC-14's cancel window. Design: ARCHITECTURE.md §3.1,
DEC-22; the plan below.

**Split off to S3-7b** (this story's plan): the OpenTelemetry spans and the FastAPI
instrumentation, the alert rules and their tests, and the `deploy-config` CI job. That half
touches DEC-14's cancel guarantee; this one adds nothing that awaits.

## Scope

- **Dependencies:** main: `structlog`, `prometheus-client` and
  `opentelemetry-semantic-conventions`, whose constants `metrics.py` imports; versions as
  verified: 26.1.0, 0.26.0 and 0.66b1. They bring nothing else: semconv needs
  `opentelemetry-api` 1.45.1, locked through FastAPI. Record what resolves; no locked version
  should move.
- **`rag_qa/observability/`**, which loads no model stack (an architecture test):
  - **`logs.py`:** `configure_logging()`, the first call of every command.
    - structlog's `ProcessorFormatter` over the standard library's handlers, so ours and
      uvicorn's records are formatted alike: JSON when stderr is not a terminal or with
      `RAG_LOG_FORMAT=json`, console lines otherwise, or with `RAG_LOG_FORMAT=console`.
    - The request ID is bound through context variables, and a job's ID in its thread; the
      trace and span IDs come with S3-7b, the API key's name with S3-8.
    - Every log goes to stderr. `rag-ingest`'s progress stays as readable as today in a
      terminal; its summary stays on stdout.
    - **pypdf's logger is set to ERROR for the process** (backlog). The API's jobs stop
      printing its warnings.
  - **`metrics.py`:** the histograms, counters and gauges DEC-22 lists, on their own listener,
    `RAG_API_METRICS_PORT` (off unless set). Names follow the GenAI conventions where one exists
    (`gen_ai_client_token_usage`). The 429 counter comes with S3-8, which makes the 429s. Ingest
    metrics come from the API's jobs: a `rag-ingest` run has no listener, and reports in its log.
- **`Done` gains `input_tokens` and `output_tokens`,** from `usage_metadata`. The SSE `done`
  event and tier 2's answers file record them; both changes are additive (the plan's D5 says
  how the answers file stays so).
- **Logs.** The two plural slips ("Embedding 1 chunks", "1 documents, 1 chunks") are fixed
  (backlog). `rag-ingest` reports pages per second, chunks embedded per second and points
  written per second (NFR-3). An open breaker's refusal after the stream began is one warning
  line, without a traceback (backlog). One line per answer, from the API and `rag-query`.
- **NFR-1.** `scripts/bench_retrieval.py` fills a scratch collection on the local server
  with 100k synthetic unit vectors, plus sparse vectors, and measures the query embedding
  plus an exact search, dense and hybrid, over 200 real-looking questions, at k=20: the
  reranker's candidates, stricter than NFR-1's 10 (design review 1). It reports p50 and p95
  and deletes its collection. Record the numbers in the README's new "Capacity planning"
  section, beside NFR-3's table.
- **NFR-2a** (DEC-27, design review 1). The same script times the real retrieval half, the
  rerank included, over the 25 golden questions with the model idle, and reports the p95 of
  what the `sources` event waits for. The target is under 2 s; record it beside NFR-1's.
- **Tests:** the metrics move as answers, refusals and ingests happen; the log formats, the
  bound fields and the levels; the token counts through `stream_answer`, the SSE event and
  tier 2's file, whose committed identity is pinned; the benchmark on a small scale.
- **README:** Observability (the logs, what each metric means; traces and alerts in S3-7b) and
  Capacity planning.

## Out of scope

- The spans, the alert rules and the `deploy-config` job: S3-7b.
- Jaeger, Prometheus and Alertmanager as running services (S3-9's `observability` profile).
- Dashboards.
- The retrieval deadline that is not a hard bound: S3-9 (the plan's reasons).

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-7/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-7/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server
# the lock: three packages added and none moved; pypdf still 6.19.0
git diff -U0 uv.lock | grep -E '^-(name|version) = '      # nothing

# metrics on their own port; logs readable under tee
RAG_LOG_FORMAT=console RAG_API_METRICS_PORT=9464 uv run rag-serve 2>&1 | tee -i ~/dev/rag-s3-7/serve.log
# one answer over HTTP (done carries the token counts) and a 503 while it streams; then
curl -s 127.0.0.1:9464/metrics | grep -E '^(rag_|gen_ai_)'  # sources, ttft, tokens, refusals, breakers
# JSON lines, and throughput in the summary: a rebuild into a scratch collection, since an
# up-to-date ingest embeds nothing; then the scratch generations go
RAG_LOG_FORMAT=json RAG_QDRANT_COLLECTION=s3_7_scratch uv run rag-ingest --rebuild 2>&1 \
  | tee -i ~/dev/rag-s3-7/ingest-json.log | tail -5
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_7_scratch--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}

uv run python scripts/bench_retrieval.py 2>&1 | tee -i ~/dev/rag-s3-7/bench.log
                                            # p95 < 300 ms at 100k, k=20, dense and hybrid;
                                            # NFR-2a: the golden questions' p95 < 2 s
gh pr checks
```

## Review notes for the human

- **The query route and `stream_answer`:** the metrics are synchronous observations and the
  token counts a read of each chunk; nothing new is awaited, and the cancel paths are as they
  were (the forced-cancel and real-uvicorn tests still pass).
- **The answers file's new fields** are left out while unset, so the committed run hashes as
  before; a test pins it.
- **The benchmark measures what NFR-1 says**, the embedding plus the search, without the
  rerank, and says so in its output.

## Plan (approved 2026-10-23)

Approved in plan mode. This section is the record to cross-check the work against; the full
plan, with the evidence for each finding, the files, the order of work and the verification
recipes, is `~/.claude/plans/pasted-content-id-7028-read-claude-md-joyful-anchor.md`. A
departure from it goes under "Deviation from plan".

**What the plan settled for the maintainer:**

- **Token counts in tier 2's answers file are not additive as first written.** The six
  fingerprint parts never see the file's shape, but `rag-eval check` compares the committed
  scores' `answers` (the sha256 of the answers file's parsed content, `a11e3ddd…`) with the
  file as the new code loads it. Fields defaulting to `None` would hash as `21601f51…` and
  demand a re-score; required fields would not load the committed file at all. Fields left out
  of the dump while unset keep `a11e3ddd…` (measured on the committed file, read-only). No edit
  in this story needs a tier-2 re-run.
- **The split,** at design review 1's seam. Only the traces touch DEC-14 and DEC-18; the lock
  changes split cleanly (3 packages here, about 17 there); each half has one model-loading
  check of its own; the alerts need this half's metric names, and S3-8 binds the key's name
  into spans, so it follows S3-7b.
- **The backlog lines:** the open-breaker traceback is taken (D8). "Published …: 1 documents"
  is taken, and is the same log line as the story's second slip. The retrieval deadline goes
  to S3-9: qdrant-client 1.19.1's `get_aliases`, `get_collection(s)` and `info` take no
  per-call timeout, so a hard bound is a resilience change of its own, and S3-9's compose can
  reproduce the hang (`docker compose pause qdrant`).
- **The hosted fallback answering** counts as an answer and in `rag_fallback_answers_total`,
  and its token counts carry provider "openai" and its model, when the provider reports usage.
  langchain-openai 1.6.2 turns `stream_usage` off for a custom `base_url` or HTTP client, so
  `groq_fallback` gets `stream_usage: true` only after one real call shows Groq accepts it
  (not hashed). Its chat span is S3-7b's.
- **The breakers** are `rag_breaker_state{breaker, state}`, read at scrape time from the
  objects `/v1/health` reports; S3-7b's `RagBreakerOpen` fires on one not closed for 5 min.
- **Found while planning:** semconv 0.66b1 marks every GenAI constant deprecated ("moved to the
  OpenTelemetry GenAI semantic conventions repository", which publishes no Python package); a
  constant or an enum member raises no warning, but calling an enum does, so the code reads
  members only. ChatOllama reports its usage on its final, empty-text chunk. uvicorn with
  `log_config=None` sets no level.

| # | Decision |
| --- | --- |
| D1 | **Dependencies:** `structlog>=26.1.0`, `prometheus-client>=0.26.0`, `opentelemetry-semantic-conventions>=0.66b1` (declared, since `metrics.py` imports its constants). Re-locked without `--upgrade`; no locked version moves, pypdf stays 6.19.0 (caveat 37). |
| D2 | **Logs:** `configure_logging()` is every command's first call: one root handler on stderr (looked up at each record), structlog's `ProcessorFormatter` (context variables, level, logger, ISO UTC time, `extra=` fields). JSON with `RAG_LOG_FORMAT=json`, or when it is unset and stderr is not a terminal; console otherwise; another value exits 2. rag-serve's console lines carry time, level and logger; the CLIs' are the message alone, today's progress. Root at WARNING, `rag_qa` at INFO, `pypdf` at ERROR for the process; uvicorn gets `log_config=None, log_level="info"`. Every log goes to stderr, so `rag-ingest`'s progress moves off stdout (its banner and summary stay there): the commit carries `!`. In JSON mode the embedding progress bar is off. |
| D3 | **Bound context:** `request_id`, bound by an `async` app-level dependency that runs first, so it binds in the request's task and the SSE producer inherits it; `job_id` in the ingest job's thread. |
| D4 | **Metrics:** a registry of our own (with the process collector, `_created` series off); the listener on `RAG_API_METRICS_PORT`, on the API's own host, started by `rag-serve` before uvicorn and stopped after; the API's port, or one in use, exits 2. The histograms `rag_retrieval_seconds`, `rag_rerank_seconds`, `rag_sources_seconds`, `rag_ttft_seconds`, `rag_answer_seconds` and `gen_ai_client_token_usage`; the counters `rag_answers_total{outcome}`, `rag_fallback_answers_total`, `rag_refusals_total{code, reason}`, `rag_ingest_runs_total{status}` and `rag_ingest_chunks_total`; the gauges `rag_breaker_state{breaker, state}` and `rag_index_documents`, read at scrape time. The token histogram's names come from semconv's constants, its buckets and required labels from the convention. |
| D5 | **Token counts:** `Done.input_tokens` and `output_tokens`, from the last chunk carrying `usage_metadata`, read before the empty-text skip, reset per attempt; `None` when the provider reports none. The SSE `done` event gains both keys; tier 2's `AnswerItem` gains both, left out of the dump while unset (`exclude_if`), and a test pins the committed identity. |
| D6 | **One log line per answer** from the API and `rag-query`: the model, the times and the token counts, as text and as fields. No question, prompt or answer text in any log line. |
| D7 | **Ingest log and NFR-3:** the two plurals through `_counted`; `IngestReport` gains `load_s`, `embed_s`, `write_s` and `points`; the summary gains a throughput line; the "Published …" line carries the counts, seconds and rates as fields. The API's job report is unchanged. |
| D8 | **An open breaker's refusal after the stream began** (a `BreakerOpenError`, or a store error it caused) logs one WARNING line naming the service and the wait, without a traceback; the `error` event is unchanged. |
| D9 | **`scripts/bench_retrieval.py`:** a scratch collection (`bench_s3_7`, never the alias, refused if it exists, deleted in a `finally`, `--cleanup` for a killed run) built like a generation from 100k seeded unit vectors and BM25 vectors of texts resampled from the live index; timed only once Qdrant's background indexing is done. 200 questions through `QdrantRetriever` at k=20, dense and hybrid; then `open_retrieval` over the 25 golden questions, rerank included. p50, p95 (nearest rank), max; it warns when Ollama holds a model; exit 0, 1 for a missed target, 2 when it cannot run. |
| D10 | **Architecture:** `rag_qa.observability` loads no LangChain, torch, sentence-transformers, transformers or qdrant_client. |
| D11 | **Docs:** README's Observability and Capacity planning, the `done` example, the variables; ARCHITECTURE.md's DEC-22 as-built note, §3.4, the gap table and §3.5 for the split. |
| D12 | **Commits**, on `feat/s3-7-observability`: this record (with S3-6's merge and S3-7b's story); the dependencies; the feature (`feat(observability)!`); the benchmark; the docs; the close-out record. |

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
