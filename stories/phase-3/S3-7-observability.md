# S3-7: Structured logs, metrics and the capacity benchmarks

| | |
| --- | --- |
| **Status** | Done 2026-10-25 (PR #25, awaiting its merge). Plan approved 2026-10-23; it split the story at design review 1's seam, so the traces and the alerts are S3-7b. On Opus 5.5 at max effort (Deviation). Built and committed on 2026-10-24 (`a052877` … `df6fd57`). The first review (2026-10-24) found one privacy defect, two flaky tests and six smaller things; the second (2026-10-25, on `c9647b5`) found one metrics defect, two doc claims and one test gap; both were fixed on the PR (`ce42616` … `bc5dcbe`). CI is green on `bc5dcbe` (run 40128517639). NFR-1 held; NFR-2a missed, and the maintainer gave its levers to S3-11's stage 2 at the close-out. |
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

## Verification record (2026-10-23 to 2026-10-25)

Rows 1–7 were run again on 2026-10-25 on `bc5dcbe`, the PR's head after both reviews; their
first runs, on 2026-10-23, gave the same results with the suite then at 1481 tests. Rows 9–13
ran in the maintainer's terminal on 2026-10-23, each logged with `tee -i` to
`~/dev/rag-s3-7/`; rows 2, 9 and 13 were run twice.

| # | Command | Result |
| --- | --- | --- |
| 1 | `uv run rag-eval check` | exit 0: the six parts `ok`, every floor met. `groq_fallback`'s `stream_usage: true` moved nothing (its spec is not hashed), and neither did the answers file's token fields |
| 2 | `uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-7/contexts.json`, then `contexts-2.json` | 0.950 (19 of 20) / 0.796 / 0.858 both times, every floor met. The first log ends in a `KeyboardInterrupt` traceback from pydantic's schema building, its head overwritten by a complete run's output, the mark of a second writer on the file; the watched second run exited cleanly (Discovered) |
| 3 | `diff ~/dev/rag-s3-2/contexts-qdrant.json` against each | IDENTICAL, both; `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc` |
| 4 | `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q` | 1492 passed, 18 skipped, 5 xfailed; coverage 98.36%; `HF_HOME` stayed empty |
| 5 | `uv run ruff check && uv run ruff format --check && uv run mypy` | all checks passed; 73 files formatted; no issues in 36 source files |
| 6 | `RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q` | 11 passed, each on a collection of its own (`test_fallback_openai.py`'s module skip is the 1 skipped); the live collections unchanged |
| 7 | the lock against `main`: `git diff -U0 main -- uv.lock \| grep -E '^[-+](name\|version) = '` | added structlog 26.1.0, prometheus-client 0.26.0 and opentelemetry-semantic-conventions 0.66b1; moved filelock 4.0.0 → 4.2.4 and setuptools 84.0.0 → 84.2.1 for their advisories (Deviation); pypdf 6.19.0; `uv lock --check` resolves 181. CI's Audit command, run locally: "No known vulnerabilities found, 11 ignored" |
| 8 | `uv sync --extra fallback`, the answering and fallback tests, then `uv sync` | 80 passed; the environment back to CI's. One real Groq call with `stream_usage: true` reported `input_tokens` 78, `output_tokens` 18 (8 of them reasoning), so the config line went in |
| 9 | `RAG_LOG_FORMAT=console RAG_API_METRICS_PORT=9464 uv run rag-serve 2>&1 \| tee -i …/serve.log`, then `serve-2.log` | "Serving the metrics on 127.0.0.1, port 9464", then uvicorn's lines through our formatter, each request's with its `request_id`. First run: one answer (183.5 s, first token at 156.4 s) and a 503 while it streamed. Second: four questions: answered in 184.8 s (first token 155.7 s); a client that left after 17 s ("Answer cancelled: its client disconnected"); answered in 257.9 s (first token 228.6 s, behind the dropped prompt, Discovered); and a 503. Each answer: "Answered by components.llms.mistral_ollama …, 1,460 input and 128 output tokens", with the same values as fields. `done`: `{"ttft_ms": 228604.2, "total_ms": 257934.1, "model": "components.llms.mistral_ollama", "input_tokens": 1460, "output_tokens": 128}` |
| 10 | `curl -s 127.0.0.1:9464/metrics \| grep -E '^(rag_\|gen_ai_)' > …/metrics.txt` (second run) | `rag_answers_total` answered 2, cancelled 1, failed 0; `rag_refusals_total{code="503",reason="busy"} 1`; `gen_ai_client_token_usage` input count 2, sum 2920, output sum 256; `rag_sources_seconds` count 3; `rag_ttft_seconds` count 2; `rag_breaker_state` `qdrant` and `llm` closed; `rag_index_documents` 3 |
| 11 | `RAG_LOG_FORMAT=json RAG_QDRANT_COLLECTION=s3_7_scratch uv run rag-ingest --rebuild 2>&1 \| tee -i …/ingest-json.log \| tail -5`, then the scratch generations deleted | JSON lines; the summary's "Throughput: 16.2 pages/s loaded, 24.2 chunks/s embedded, 2,695 points/s written."; the "Published" line's fields: 561 pages, 1,708 chunks and points, `load_s` 34.692, `embed_s` 70.672, `write_s` 0.634. Afterwards only `rag_qa`'s two generations remain, the alias on `…7f20891e` (as in `collections-before.txt`, and again on 2026-10-25) |
| 12 | `uv run python scripts/bench_retrieval.py --points 10000 2>&1 \| tee -i …/bench-10k.log` | the dry run found the two-runs race: a second run, started 4 s after the first, deleted the first's collection mid-timing (`GenerationGoneError`). Fixed with a test (Discovered) |
| 13 | `uv run python scripts/bench_retrieval.py 2>&1 \| tee -i …/bench.log`, then `bench-2.log` | NFR-1, 100,000 points, 200 questions, k=20: dense p50 23.9 / p95 28.2 / max 42.4 ms, hybrid 28.6 / 35.1 / 48.4 ms; then dense 26.6 / 29.7 / 34.3 ms, hybrid 31.5 / 38.0 / 46.3 ms: held. NFR-2a, 25 golden questions, the rerank included: p50 1.45 / p95 2.09 / max 2.11 s, then 1.46 / 2.02 / 2.09 s: MISSED, exit 1. Filling took 53 s, then 51 s, and `bench_s3_7` was deleted each time |
| 14 | `gh pr checks` | red on `a12d32f` (run 40073376741): Audit alone, on seven advisories published on 2026-10-23 (three pypdf, three filelock, one setuptools). Green on `df6fd57` (run 40080480605), on `c9647b5` (run 40116138049) and on `bc5dcbe` (run 40128517639): `check` and `eval-retrieval` |

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

## First review (2026-10-24, on `df6fd57`)

Reviewed as the first reviewer. Everything below was reproduced against the PR's head in a
scratch copy of `HEAD` (never in the repository); the live Qdrant was only read (tier 1). No
blocker for the design or the cancel path; one privacy defect, two flaky tests and six
smaller findings, all fixed on the PR.

**Fixed (tests came with 1, 2, 3 and 7):**

1. **`rag-serve`'s console log printed the question and the retrieved chunks after a failed
   answer.** `ConsoleRenderer` formats `exc_info` with rich and `show_locals=True` when rich
   is installed (it is, in the non-dev export), and the console path skipped
   `format_exc_info`. A failing answer through the real route leaked in `console` with
   `detailed=True` only (`rag-serve` in a terminal, or `RAG_LOG_FORMAT=console`: the
   verification's own command). Now `exception_formatter=structlog.dev.plain_traceback`; a
   test drives a failed answer through all four format/`detailed` combinations and greps
   stderr for the question and the chunk text.
2. **`test_one_document_of_one_chunk_is_logged_in_the_singular` was flaky:** `write_s > 0` on
   a field logged as `round(x, 3)`; a 0.4 ms in-memory write is `0.0`. It failed in 4 of 6
   warm runs. Now `>= 0`.
3. **`test_an_ingest_job_moves_the_runs_and_the_chunks` was flaky:** it read
   `rag_index_documents` as the job's status flipped, before the server's refresh. With the
   refresh delayed by 50 ms it failed every time (`0.0 == 5`). It now waits for the gauge.
4. **README's ingestion table** said the embed phase made "dense and keyword vectors". The
   shipped config sets no sparse encoder: dense only.
5. **The pypdf exception is eleven ignores,** not eight, in `ci.yml` since `df6fd57`. STATUS
   caveat 37, the DEC-24 row, the backlog line, S3-9 (four places and its audit command) and
   S3-11b now say eleven, and name 6.21.2 as the bump's floor. (S3-6's review fixed the same
   drift for eight.) The three new reasons match their advisories (GHSA-gmh7-3g9r-mjh5,
   GHSA-jr96-gvgh-j3r4, GHSA-6c7f-9265-g8fw, published 2026-10-23, fixed in 6.21.2).
6. **`pydantic>=2.11` was too low:** `Field(exclude_if=…)` exists from 2.12 (checked at
   v2.11.10 and v2.12.0). On 2.11.10 it is accepted with a deprecation warning and dumps
   `null`, which would change the committed answers' identity. Now `>=2.12`; `uv lock` moves
   the specifier line and no version (2.13.5).
7. **Four guards survived mutation** across the whole suite: the `async` of `bind_request`
   (a sync one loses `request_id`), the `job_id` binding, the rerank leaking into
   `rag_retrieval_seconds`, and the progress bar in JSON mode. Four tests now fail with
   each defect put back. Of 15 guards broken in all, the other 11 were already caught.
8. **JSON lines are not the only stderr lines:** the model loader's `Loading weights` bar
   still prints between them (`HF_HUB_DISABLE_PROGRESS_BARS=1` silences it). Documented in
   the README and ARCHITECTURE; not set in code, since the variable is read when
   huggingface_hub is imported.

**Held:** `rag-eval check` exit 0; tier 1 re-run on `HEAD`, contexts IDENTICAL to S3-2's
(`5a36e2f9…`); the lock (three packages added, filelock and setuptools bumped alone, pypdf
6.19.0); the README's 13 metric names equal the registry's, every label set starts at zero;
commit headers and trailers; CI green. After the fixes: 1490 passed, 18 skipped, 5 xfailed,
coverage 98.36%; eight warm repeats of the files that flaked all green.

**Open for the maintainer (not defects):**

- **NFR-2a is missed twice** (p95 2.09 s, then 2.02 s, against under 2 s), and the benchmark
  exits 1. ARCHITECTURE had said STATUS's backlog sends the rerank levers to S3-11; it did
  not, and S3-11's levers do not include them. The text now calls it an open decision. Options:
  accept "measured, not met" and add a backlog line; or add reranker candidates / a smaller
  cross-encoder to S3-11's levers (both move hashed inputs, so tier 2 once, in S3-11b).
  **Decided at the close-out (2026-10-25):** the second; S3-11's stage 2 owns them.
- **Verification 87** (`git diff -U0 uv.lock | grep …` "nothing") shows the two advisory bumps
  (filelock 4.0.0 → 4.2.4, setuptools 84.0.0 → 84.2.1, `1cc5c54`); record under Deviation.
- **D7 said the API's job report is unchanged;** it gained `load_s`, `embed_s`, `write_s`
  and `points` (additive; its mirror test requires them). Record under Deviation.

**Nits left:** `bench_retrieval.py` exits 1 on a mid-run Qdrant error (the docs reserve 1 for
a missed target) and leaks its scratch collection if the payload index fails right after
creation; `opentelemetry-semantic-conventions>=0.66b1` has no cap while its constants are
deprecated; `/metrics` carries no token (documented), so S3-9's compose must keep it internal.

## Second review (2026-10-25, Opus 5.5, on `c9647b5`)

**1 defect in the metrics, 2 doc claims and 1 test gap, all fixed on the PR; nothing blocks
the cancel path.** Run on the PR's head, with the first review's record treated as claims to
check. The live Qdrant was only read: the alias still names `…7f20891e`, and its two
generations are the only collections. Probes and mutations ran in a scratch copy of `HEAD`;
the library claims were checked against the installed source (langchain-openai 1.6.2 in an
isolated environment); the maintainer's logs of 2026-10-23 in `~/dev/rag-s3-7/` were read
as evidence. No model was run.

- **Reproduced on `c9647b5`:** ruff, format and mypy clean; **1490 passed, 18 skipped, 5
  xfailed**, coverage **98.36%**, the HF cache empty; the 11 `qdrant_server` tests;
  `rag-eval check` exit 0; tier 1 0.950 / 0.796 / 0.858, the contexts digest `5a36e2f9…`,
  IDENTICAL; CI green on the head (run 40116138049); CI's audit command over the locked
  default export, today: "No known vulnerabilities found, 11 ignored".
- **DEC-14's cancel path:** what the story adds to the route and `stream_answer` is
  synchronous, and every ending of an answer reaches `Slot.release()` once, its outcome set
  before.
- **The first review's privacy fix holds:** with `plain_traceback` removed, the
  console/detailed case fails and the other three pass (control clean). structlog 26.1 shows
  locals by default in its dictionary form (`SHOW_LOCALS = True`) and in rich's console form.
- **The claims, against the installed source:** langchain-openai 1.6.2 streams usage only
  for OpenAI's own base URL and client; ChatOllama reports usage only on its `done` chunk,
  which has no text; uvicorn with `log_config=None` only sets levels; calling a semconv enum
  warns, reading a member does not; `observability` imports no LangChain, torch, numpy or
  qdrant_client; a bad `RAG_LOG_FORMAT` exits 2 in all four commands without printing it.
- **The lock:** three packages added, filelock and setuptools bumped for their advisories,
  pypdf 6.19.0; each new floor equals its locked version.
- **The real run's logs** (2026-10-23): `request_id` on each request's lines; the `done`
  events' counts (1,460 and 128) equal to the token metric's; the outcomes and the `busy`
  refusal counted; the README's capacity tables equal to `bench.log` and `bench-2.log`. The
  10k run's `GenerationGoneError` was the two-runs race that `create()`'s 409 now prevents.
- **Records and commits:** the eleven ignores consistent everywhere; the ten commits'
  headers and trailers, the `!` and `BREAKING-CHANGE:`, no body line over 72, no attribution.

Findings, all fixed:

1. **The answer's latency histograms missed the time spent in the route's dependencies.**
   FastAPI 0.143 starts the SSE producer, and so the route's body, only once
   `solve_dependencies` returns. `rag_sources_seconds`, `rag_ttft_seconds` and
   `rag_answer_seconds` therefore started after `served_pipeline`'s `await
   serving.refresh()`: the alias read before each answer (about 3 ms here), a reopen after an
   ingest, and Qdrant's retries of the read, each attempt up to `timeout_s: 10`. With
   `refresh()` delayed by 1.5 s, the client waited 1.529 s and the three recorded 0.006,
   0.009 and 0.010 s; their help and the README say "from a question". Now `bind_request`,
   the first dependency, notes `request.state.received_at`, and the route times from it. A
   new test delays the refresh by 0.3 s and requires all three to count it; it fails with the
   old start.
2. **The README's `done` example** carried token counts (1,302 and 87) that no run produced,
   under "From a run on this host's CPU": the run is S2-7's, older than the counts, and
   S3-7's real runs asked another question (1,460 and 128). The text now says the two are
   illustrative.
3. **ARCHITECTURE said the token histogram's buckets come from
   opentelemetry-semantic-conventions 0.66b1.** The package's
   `create_gen_ai_client_token_usage` carries the name, description and unit only; the
   buckets are the convention's text (v1.37.0's `gen-ai-metrics.md`), which they equal.
   Reworded, as `metrics.py`'s docstring already had it.
4. **No test said a question refused after taking its slot counts no answer.** The no-index
   503 and a malformed body's 422 are refused by dependencies after `take_slot`. With
   `Slot.release`'s `if self.outcome is not None` removed, they would add a
   `rag_answers_total{outcome="None"}` series, and the 165 tests of `test_metrics.py` and
   `test_api.py` still passed. A new test covers both refusals and every series of the
   counter; it fails with the guard removed.

**After the fixes:** 1492 passed, 18 skipped, 5 xfailed, coverage 98.36%; the 11 server
tests; ruff, format and mypy clean; `rag-eval check` exit 0. Four mutations beyond the first
review's 15: two caught as the tests stood (the usage read after the empty-text skip; the
answers file's token fields dumped as `null`), one caught only by finding 4's test, and one
survives (below). Both new tests fail with their defect put back, after clean controls.

**Nits left:**

- "The last chunk that reports usage wins" is untested (a first-wins mutant survives), and is
  not LangChain's own merge, which sums the chunks' usage (`add_usage`). The two agree for
  Ollama and OpenAI, which report once; a provider that reports deltas would lose its input
  count.
- The `BREAKING-CHANGE:` trailer names `rag-ingest`'s progress only. Every command's log is
  now JSON when stderr is not a terminal, `rag-serve`'s under a service manager included:
  worth a line in the PR body or the merge message. **Taken at the close-out:** both carry
  it (Deviation).
- The first review's three nits stand.

## Discovered

- **semconv 0.66b1 marks every GenAI constant deprecated,** "moved to the OpenTelemetry
  GenAI semantic conventions repository", which publishes no Python package yet; the strings
  are unchanged. Reading a constant or an enum member raises no warning, but calling an enum
  (`GenAiOperationNameValues("chat")`) raises a `DeprecationWarning`, which the deprecation
  gate fails on. So `observability/genai.py` reads members only. The package has no upper
  bound (S3-7b's backlog line).
- **ChatOllama reports its usage on its final chunk, which has no text,** and
  `stream_answer` skipped empty chunks before reading anything. The usage is now read first.
- **langchain-openai 1.6.2 turns `stream_usage` off for a custom `base_url` or HTTP
  client,** and the fallback has both. `groq_fallback` sets `stream_usage: true` after one
  real call showed Groq answers it (row 8).
- **The answers file's identity is a trap for any new field** (caveat 40). A field
  defaulting to `None` changed the committed answers' identity from `a11e3ddd…` to
  `21601f51…`, so `rag-eval check` would have asked for a T4 re-score. `exclude_if` keeps
  it, from pydantic 2.12 (the first review), and a test pins the identity.
- **structlog shows each frame's local variables by default,** in `dict_tracebacks`
  (`show_locals=True`) and, through rich, in the console renderer. Either would have put a
  question or the retrieved chunks into the log. Tracebacks are text in JSON
  (`format_exc_info`) and plain in the console (the first review).
- **A client dropped between two events leaves the route's generator paused,** and only
  the garbage collector closes it, so an outcome counted in its `finally` was missed (the
  forced-window test). The outcome is counted once, in `Slot.release()`.
- **FastAPI runs a route's body only after its dependencies,** so an answer's times
  started late, after the alias read and its retries; they now start at the first
  dependency (the second review). That dependency is `async`, since a sync one runs in a
  thread whose context, and so the request ID, is lost.
- **uvicorn's default `dictConfig` gives the `uvicorn` logger a handler of its own and
  stops it propagating;** with `log_config=None` it configures nothing and sets no level,
  so `rag-serve` passes `log_level="info"`.
- **`serialized`'s `run_in_executor` copies no context variables;** LangChain's own does,
  on the CLI's path. Nothing in S3-7 needs it; S3-7b's spans do.
- **`rag-eval check` printed the corpus walk's INFO lines** among its tables, so
  `rag-eval` logs warnings only.
- **pytest's own handler formats a record before ours,** which leaves `message` and
  `asctime` on it. They are dropped, so they never become JSON fields.
- **Under `2>&1 | tee -i`, stderr is a pipe,** so the log is JSON unless
  `RAG_LOG_FORMAT=console` (caveat 39). The model loader's `Loading weights` bar still
  prints between JSON lines (S3-9's backlog line).
- **The benchmark could delete another run's collection.** Two runs started 4 s apart both
  passed the check for an existing collection, during the embedder's load. The second's
  create got a 409, and its cleanup then deleted the first's collection mid-timing (row 12).
  `create()` now reports a lost race, the run exits 2, and it deletes only a collection it
  created; a test reproduces the race.
- **NFR-2a is missed by the rerank:** a p95 of 2.09 s, then 2.02 s. The search takes about
  30 ms; `ms-marco-MiniLM-L-6-v2` scoring 20 candidates is the rest. Its levers move hashed
  inputs (S3-11's stage 2, the backlog line).
- **A cancelled question delays the next one's first token.** In row 9's second run, the
  answer after a dropped question reached its first token at 228.6 s, not about 156 s:
  Ollama most likely read the dropped prompt first, as README's known limitation says. It is
  inside the 300 s objective S3-7b's alert will use.
- **The first tier-1 log's `KeyboardInterrupt`** came from pydantic's schema building, not
  our exit path. Its traceback begins mid-line where a complete run's output ends, the mark
  of a second writer on the same file. The watched second run exited cleanly, with the same
  scores and contexts.
- **Seven advisories were published on 2026-10-23,** after the local audit and before the
  push: three in pypdf (the exception, now eleven), three in filelock and one in setuptools
  (bumped alone).

## Deviation from plan

- **The model:** Opus 5.5 at max effort, in place of Fable, which is unavailable (CLAUDE.md).
- **`model_identity` is in `observability/genai.py`,** not `components.py` (D4). It reads
  the model's class and attributes without importing LangChain, so `observability` still
  loads no model stack, and S3-7b's spans can share it.
- **The API's job report gained `load_s`, `embed_s`, `write_s` and `points`;** D7 said it
  was unchanged. Its model must mirror `IngestReport`, as a test requires. Additive.
- **`rag-eval` logs at WARNING,** where D2 set `rag_qa` at INFO for every command
  (Discovered).
- **Tracebacks are text, not `dict_tracebacks`** (D2), and plain in the console (the first
  review): both forms structlog offered show local variables (Discovered).
- **An answer's outcome is counted in `Slot.release()`,** not in the query route (D4),
  and its times start at the first dependency, not the route's body (the second review).
- **The lock moved two versions D1 did not plan:** filelock 4.0.0 → 4.2.4 and setuptools
  84.0.0 → 84.2.1, each bumped alone for advisories published during the PR (`1cc5c54`).
  pydantic's floor rose to 2.12, its locked version unchanged (`47709e9`). So
  Verification's lock line prints those two bumps, not nothing.
- **The `BREAKING-CHANGE:` trailer of `f11f694` names `rag-ingest`'s progress only.** Every
  command's log is now JSON when stderr is not a terminal, `rag-serve`'s under a service
  manager included, and an unknown `RAG_LOG_FORMAT` exits 2. The commit is on the pushed
  branch, so the PR body and the merge commit's message say it in full.
