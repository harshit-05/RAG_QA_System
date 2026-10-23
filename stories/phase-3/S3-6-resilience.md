# S3-6: Timeouts, retries, circuit breakers and the hosted fallback

| | |
| --- | --- |
| **Status** | Reviewed (2026-10-23): PR #24 on `feat/s3-6-resilience`. `97cb2b6` S3-5's merge and this plan; `81af84d` the dependencies; `2269a2c` the feature (breaking: an `api_key` in a component); `626358d` the docs; the `ci(audit)` commit extending the pypdf exception, and this record. Verification 1–14 run and shown below, all as expected; the one real check ran against Groq on 2026-10-23. CI went red on `626358d` (run 39916798730) in Audit alone, on three pypdf advisories published on 2026-10-22: the maintainer extended caveat 37's exception (Discovered). On Opus 5.5 at max effort (Deviation). The first review (2026-10-23) found no blocker and fixed two things on the PR ("First review"); its commits are `fb66a39` and `a0b77bd`, and CI is green on `a0b77bd` (run 39932776608, both jobs). The second review (2026-10-23) found one defect, a readiness poll during Qdrant's half-open probe unserving the index, and fixed it on the PR with three tests ("Second review"). Once CI is green on its commits, merge it with a merge commit. |
| **Closes** | NFR-6; DEC-9; DEC-21; SRS §2.4 (a hosted-provider fallback); SRS §11 resilience; backlog: the hosted-LLM fallback, NFR-6, the back-off after a failed index reopen |
| **Depends on** | S3-2 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-6-resilience`. Moderate risk: it wraps the answer stream, whose cancellation DEC-14 proved, and it adds a path that sends retrieved text off the host. |

## Goal

Every network call gets a timeout, a retry where a retry is safe, and a circuit breaker, so
a dead dependency costs a fast, explicit refusal rather than a hang (NFR-6, DEC-9). And
Ollama gets an opt-in hosted fallback, the "LLM gateway (Ollama primary, hosted fallback)"
of SRS §6.2. It answers only when Ollama cannot, and it is checked once, for real, with a
key the maintainer supplies (DEC-21). Design: ARCHITECTURE.md §3.1, DEC-21.

## Scope

**At the start:** ask the maintainer which OpenAI-compatible provider to configure, and
confirm the key is in their shell under the provider's own variable. Never write it to a
file in the repository, and never print it.

**If the plan-first step splits it** (design review 1), the seam is resilience (timeouts,
retries and breakers) against the fallback, which waits on the provider's key.

- **`rag_qa/resilience.py`.**
  - `CircuitBreaker(name, failures=3, reset_after_s=30, clock=time.monotonic)`: closed,
    then open after that many consecutive failures, then half-open, letting one probe
    through after `reset_after_s`. It is thread-safe (retrieval runs in worker threads) and
    tested on a fake clock.
  - `BreakerOpenError`, which names the dependency and how long until the probe.
  - Retry policies on tenacity (`uv add tenacity`, 9.1.4): jittered exponential back-off,
    with the attempts and the cap as constants.
- **Timeouts.**
  - **Ollama:** `client_kwargs` on each llm entry, 5 s to connect and 600 s to read; the
    read must outlast a silent prefill (caveat 3). YAML cannot spell an `httpx.Timeout`, so
    the plan decides how a mapping becomes one (likely in `components.build_llm`). These
    keys are runtime keys, never hashed (`fingerprint.RUNTIME_KEYS`); a test confirms.
  - **Qdrant:** `timeout_s`, from S3-2.
- **Retries.**
  - **Qdrant:** every call in `vectorstore`, on connection errors, timeouts and 5xx, with
    what a repeat counts as done stated per call (DEC-21, design review 1): a create that
    finds our uniquely named collection, or a delete that finds nothing, has succeeded; a
    snapshot is never retried.
  - **A bounded budget** (DEC-21, design review 2). Retrieval holds the retrieval lock
    throughout (`serialized`), so a hung Qdrant would hold it for a timeout per attempt. The
    breaker therefore wraps each attempt, and an open breaker ends the loop. One deadline
    covers a retrieval's attempts, 15 s by default. A 404 for a generation is never
    retried: the generation is gone, and the reader resolves the alias once more (DEC-19).
  - **Ollama:** only before the first token, and only on a connection error, 2 attempts. A
    retry after a token would duplicate text.
  - The retry loop runs in the consuming task: no new task near the stream (DEC-14). It is
    written by hand in `stream_answer`: tenacity retries a call, and the model's stream is an
    async generator.
- **Breakers**, one per dependency: Ollama, Qdrant, and the fallback.
  - **API:** a breaker that is open is a dependency that answers 503 with `Retry-After`,
    before the first byte (DEC-18's rule), and `/v1/health` reports each breaker's state.
    Both name the service, never its address (DEC-18).
  - **Start-up checks** (`rag-query`, `rag-ingest`, `rag-eval`, `rag-serve`) exit 2 naming
    the service.
  - **With a fallback configured, Ollama down is not a refusal** (DEC-21, design review 1).
    An open Ollama breaker routes the answer to the fallback, and the 503 comes only when the
    fallback's breaker is open too. The start-up checks warn about Ollama instead of exiting
    2, and the primary is built with `validate_model_on_init` off (a runtime key, so nothing
    hashed moves). Without this, `preflight` would end `rag-query` before the real check
    below could start.
  - **The reopen back-off (backlog):** while Qdrant's breaker is open, `Serving.refresh()`
    returns at once.
- **The hosted fallback.**
  - **Schema:** `pipeline.query.fallback_llm`, an optional reference to an llms entry
    different from `pipeline.query.llm`.
  - **Config:** a `components.llms` entry on `langchain_openai.ChatOpenAI`, with `model`,
    `base_url`, `temperature: 0` and a token limit. It is referenced by nothing in the
    committed `config.yaml`; the real check uses a scratch overlay or config.
  - **`max_retries: 0` and a `timeout` on it** (DEC-21, design review 2). The OpenAI SDK
    otherwise retries twice on its own, with a 600 s timeout (`openai/_constants.py`), which
    would hide failures from the fallback's breaker for minutes.
  - **The extra:** `uv add --optional fallback "langchain-openai>=1"`; 1.6.2 is already
    locked through ragas, so nothing new should resolve (check). Building a fallback without
    the extra is a load error: "install it with `uv sync --extra fallback`".
  - **The allowlist** gains `langchain_openai.` (ADR-020), with tests.
  - **No secrets in config.** A key named `api_key`, or ending in `_api_key`, anywhere in
    `components` is refused at load, as `trust_remote_code` is. The key comes from the
    environment only.
  - **When it answers:** while Ollama's breaker is open, or after Ollama failed with a
    connection error before its first token, once the retries are spent.
  - **How it streams:** `stream_answer` streams the fallback model itself, in the consuming
    task, with the same `finally`. Never `.with_fallbacks()`.
  - **`Done.model` is new:** the model that answered. Over HTTP the `done` event carries it,
    the server log names a fallback answer, and the CLI prints the fallback's name under
    the answer.
  - **`QueryPipeline` gains `fallback_llm`;** `aclose()` closes both models' clients.
  - **Tier 2:** `rag-eval generate` builds no fallback, and its spec is not hashed (tests).
  - **README:** what the fallback is for, that it is off by default, and that retrieved
    chunks leave the host when it answers.
- **Tests, hermetic**, with fake models:
  - a primary that refuses the connection before its first token, then a fallback that
    answers;
  - a primary that fails after a token: an `error` event, and no fallback;
  - a breaker opening, refusing with 503 before the first byte, then recovering after a
    successful probe;
  - a cancel during the fallback's stream, forced at the worst moment, as DEC-14's tests do:
    the stream is closed before the consumer returns;
  - Qdrant's retries, with a fake transport;
  - a hung Qdrant, a fake transport that never answers: the breaker opens within one
    retrieval's deadline, and the retrieval lock is released (design review 2).
- **A config test:** if a test imports every `_target_` in `config.yaml`, give an entry whose
  extra CI does not install a way through. CI does not install `fallback`.

## Out of scope

- Caching the health probes and rate limits (S3-8).
- Metrics and spans for the breakers (S3-7 adds them).

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0: runtime keys only
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-6/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-6/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

# Ollama down: a fast, explicit refusal (no fallback configured).
# The maintainer runs the systemctl lines (sudo).
sudo systemctl stop ollama
uv run rag-query; echo "exit $?"                          # 2, naming Ollama, within seconds
# over HTTP: 503 with Retry-After before the first byte, once the breaker is open

# the one real check (maintainer's key in the shell; a scratch overlay naming the fallback)
uv sync --extra fallback
uv run rag-query --config <scratch config with fallback_llm>   # starts, warns; "(answered by …)"
# Ctrl-C during the fallback's answer: the prompt returns at once, nothing printed after
# rag-serve on the same config: the answer streams from the fallback (`done`'s model), no 503
sudo systemctl start ollama && uv sync
gh pr checks
```

## Verification record (2026-10-22/23)

Run on the working tree on `main`'s `1815e00`, before the branch's commits, except 15. Steps
10–14 ran in the maintainer's terminal (sudo, the key), each logged with `tee` to
`~/dev/rag-s3-6/`. The scratch config is `config.local.yaml` (gitignored): `config.yaml` with
`fallback_llm: components.llms.groq_fallback` uncommented, one line.

| # | Command | Result |
| --- | --- | --- |
| 1 | `uv run rag-eval check` | exit 0: all six parts `ok`, every floor met, with the Ollama timeouts, the fallback entry and `deadline_s` in `config.yaml` |
| 2 | `uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-6/contexts.json` | 0.950 (19 of 20) / 0.796 / 0.858, every floor met |
| 3 | `diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-6/contexts.json` | IDENTICAL; both `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc` |
| 4 | `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q` | 1391 passed, 18 skipped (the 15 server tests, the fallback module, 2 import tests that need the extra), 5 xfailed; coverage 98.30%; `HF_HOME` stayed empty. Three tests were added after it (preflight's two fallback failures, a missing `langchain_openai`): `tests/test_components.py` 53 passed |
| 5 | the suite again under `uv sync --extra fallback` | 1399 passed, 15 skipped, 5 xfailed: the 6 real-`ChatOpenAI` tests and the 2 import tests run; then `uv sync` |
| 6 | `uv run ruff check && uv run ruff format --check && uv run mypy` | all checks passed; 65 files formatted; no issues in 32 source files, with and without the extra |
| 7 | `RAG_TEST_QDRANT_URL=… uv run pytest -m qdrant_server -q`, on a throwaway v1.19.2 (port 6390) | 11 passed; the probes of what a repeat counts as done ran there too (Discovered); the container is removed, and the live alias was never touched |
| 8 | `uv lock`, then `git diff uv.lock` | the project's own metadata only: 178 packages, every version as before, pypdf 6.19.0 |
| 9 | `rag-query --config config.local.yaml` without the key; `echo exit \| rag-query` | exit 2 before any model: "the hosted fallback components.llms.groq_fallback needs its API key in RAG_FALLBACK_API_KEY, which is not set."; a plain session starts and exits 0 |
| 10 | `sudo systemctl stop ollama`; `uv run rag-query; echo "exit $?"` | exit 2 within seconds: "cannot build the language model components.llms.mistral_ollama: ConnectionError: Failed to connect to Ollama…", and the hint |
| 11 | `rag-serve` started with Ollama up, then Ollama stopped; three queries, health | 200 with `error` "the language model is unreachable" (both attempts); 200 with `error` "Ollama is not answering" (its first attempt opened the breaker, which refused the second); `503`, `retry-after: 30`, no stream. Health 503: the `llm` breaker `open`, `retry_after_s` 30 |
| 12 | Ollama started, `sleep 31`, a query, health | the probe answered: first token at 150.6 s, done at 168.8 s, `model` `components.llms.mistral_ollama`; health `ready`, every breaker `closed` |
| 13 | the real check, `rag-query --config config.local.yaml`, Ollama stopped, the key set | the warning (Ollama did not answer; questions and chunks go to `https://api.groq.com/openai/v1`), the banner, then answers from Groq each followed by "(answered by the hosted fallback, components.llms.groq_fallback)": GLIDER's described correctly, with no reasoning text, and the refusal sentence for a question the corpus cannot answer. Ctrl-C mid-answer: "(answer interrupted; …)" and nothing after it |
| 14 | `rag-serve --config config.local.yaml`, Ollama stopped; four queries, health, a drop | four `200`s, each `done` naming `components.llms.groq_fallback`, first token in 1.6–2.0 s, and no `503` once Ollama's breaker opened; the server logged "Answered by the hosted fallback …" for each. Health 200: "the hosted fallback answers in its place", `llm` open (23 s left), `fallback` closed. The `curl --max-time 2` drop: Groq finished the answer first, so it logged an answer, not a cancel (Discovered). No key-shaped string in any log |
| 15 | `gh pr checks` | red on `626358d` (run 39916798730): `check` failed in Audit alone, on three new pypdf advisories, and `eval-retrieval` was skipped behind it. Lint, Format, Types and Test passed: 1394 passed, coverage 98.44% |

The first attempt at 13 and 14 stopped at start-up, as it should: the key was not set in
those terminals, and both commands exited 2 naming the variable. In the second, Ollama was
started again 8 s after `rag-serve` (its journal), so the HTTP queries went to mistral; the
third kept it stopped. Row 13's answers come from the second attempt and its Ctrl-C from the
third; row 14 is the third's.

## Review notes for the human

- **The stream path:** no new task, no `.with_fallbacks()`, and the same `finally` closing
  whichever model streamed. Then read the forced-cancel tests.
- **Retries only before the first token,** and only on connection errors.
- **The fallback's routing:** with one configured, Ollama down starts every command and the
  fallback answers; with none, the refusals stand.
- **The secret rule:** no key name, and no value from a provider's variable, in any log
  line, error or event.

## Plan (approved 2026-10-22)

Approved in plan mode. This section is the record to cross-check the work against; the full
plan, with the design by module, the tests and the verification recipe, is
`~/.claude/plans/pasted-content-id-6f69-read-claude-md-sorted-kettle.md`. A departure from it
goes under "Deviation from plan". The provider is Groq, `openai/gpt-oss-120b` (human
prerequisites in STATUS.md), so the story is not split.

| # | Decision |
| --- | --- |
| D1 | **`RAG_FALLBACK_API_KEY`, the maintainer's proposal, is accepted**, amending DEC-21's "the provider's own variable". `settings.FallbackSettings` reads it, as `StoreSettings` reads `RAG_QDRANT_API_KEY`, and our code always passes it to `ChatOpenAI` as `api_key`. Without one, `ChatOpenAI` reads `OPENAI_API_KEY`, or a LangSmith gateway key when the gateway is on (langchain-openai 1.6.2, langchain-core 1.6.3's `_gateway.py`), never `GROQ_API_KEY`. A fallback configured without the variable exits 2 at start-up, naming it and never a value. `.env` stays gitignored and is loaded by the shell (`set -a; . ./.env; set +a`); a committed `.env.example` holds names only, which a hygiene test checks, and S3-9 extends. |
| D2 | **Breakers belong to their clients:** `QdrantStore.breaker`, and the pipeline's `llm_breaker` and `fallback_breaker`. No process-wide registry; an ingest job's own store has its own. |
| D3 | **`Done.model`** is the config reference of the model that answered, e.g. `components.llms.groq_fallback`: unambiguous, since the fallback must differ from `llm`, and how tier 2 and the CLI already name models. |
| D4 | **Numbers, as constants in `resilience.py`:** a breaker opens after 3 consecutive failures and probes after 30 s; Qdrant gets 3 attempts, Ollama 2 (before its first token, on connection errors only), the fallback 1 (`max_retries: 0`); the back-off is full jitter, `U(0, min(4, 0.5·2^(n−1)))` s, one function that tenacity's Qdrant loop and `stream_answer`'s hand-written loop share. |
| D5 | **What a breaker counts:** a failure is a connection error, a timeout, a 429, a 5xx or Ollama's in-stream error; any other answer, a 4xx included, is a success; a cancel or an unrelated error is no verdict, and frees a half-open probe. For a model, the first token decides, and nothing after it is judged. `resilience.classify` matches by class name and module across the MRO (httpx, httpx2, which openai 3.3.0 is built on, openai, ollama's `status_code`, the builtins) and imports none of them. |
| D6 | **The retrieval deadline is a store setting,** `deadline_s: 15` in `config.yaml` (the store's spec is hashed nowhere), set by the retrieval halves through a `ContextVar` inside the worker thread, the sooner of two nested ones winning. The Qdrant loop stops when the next sleep would pass it, and `search()` passes the time left as `query_points(timeout=…)`, which qdrant-client 1.19.1 also uses as the request's own timeout. |
| D7 | **A timeout is a mapping** of `connect`, `read`, `write` and `pool` seconds, all four, checked at load. Building the model turns it into its client library's `Timeout`: `httpx.Timeout` in ChatOllama's `*client_kwargs`, `openai.Timeout` (httpx2's) in ChatOpenAI's `timeout`. |
| D8 | **The fallback entry,** once `pipeline.query.fallback_llm` names it, must be another llms entry than `llm`, on `langchain_openai.ChatOpenAI`, with a `base_url` (where the chunks go), `max_retries: 0` and a `timeout`; checked at load. |
| D9 | **The fallback gets an HTTP client of its own** (`openai.DefaultAsyncHttpxClient`, imported lazily). langchain-openai 1.6.2 caches its default async client per base URL and timeout (`lru_cache`), shared by every instance, so closing one would close it for all. `aclose_llm` learns `ChatOpenAI.root_async_client`. |
| D10 | **`rag-eval generate`** runs the start-up check, so a stopped Ollama exits 2 naming it rather than with a traceback, and never builds the fallback. |
| D11 | **Refusing `api_key` and `*_api_key` keys in `components` is a config contract change:** the feat commit carries `!` and a `BREAKING-CHANGE:` trailer, as S2-4's allowlist cut did. |
| D12 | **Dependencies:** `tenacity>=9.1.4` (locked already); `httpx>=0.28.1` becomes a runtime dependency, since `src` imports it, and its dev-group line goes; the `fallback` extra is `langchain-openai>=1,<2`, the bound the other LangChain packages carry. Re-locked without `--upgrade`, so pypdf stays at 6.19.0 (caveat 37). |
| D13 | **S3-3's backlog line**, a read timeout with Qdrant up, is covered by the retries. When every attempt times out, the error says that Qdrant did not answer in time, not only "Is Qdrant running?". |

**Commits**, on `feat/s3-6-resilience`: this record (with S3-5's merge); the dependencies;
the feature (`feat(resilience)!`); the docs; the close-out record.

## Discovered

- **CI went red on `626358d` in Audit alone** (run 39916798730). Three pypdf 6.19.0
  advisories were published on 2026-10-22 at 16:49 UTC, after `main`'s last green run, so
  `main`'s lock fails them too:
  - GHSA-6vvc-x485-93v7 (CVE-2026-113789), form fields: unreachable, since `get_fields` is
    called only by itself and `get_form_text_fields`;
  - GHSA-cgf9-327m-v77c (CVE-2026-113791), content streams: reached by `extract_text`
    (`_page.py`, `_extract_text`);
  - GHSA-f54x-cmq8-qg57 (CVE-2026-113790), arrays and dictionaries: reached by any read.

  All three are fixed in 6.21.0, which S3-11b's bump to 6.21.1 or later covers. Caveat 37
  made this the maintainer's decision; on 2026-10-23 they extended the exception rather than
  bump pypdf here, which would have cost a tier-2 re-run outside S3-11b's. pip-audit prints
  the CVE IDs, and the GHSA ignores match them through their aliases (checked locally: "No
  known vulnerabilities found, 8 ignored").
- **langchain-openai 1.6.2 shares one async HTTP client between instances,** cached per
  base URL and timeout (`_client_utils._cached_async_httpx_client`, an `lru_cache`).
  Closing one model's would close every other's, so the fallback gets its own (D9;
  caveat 38).
- **openai 3.3.0 is built on httpx2,** pydantic's fork of httpx: its `Timeout` and its
  transport errors are httpx2's. `resilience.classify` reads both by module name. httpx2 is
  in the plain environment already, through langsmith.
- **The read-back after a failed alias switch must not wait on the breaker.** The switch's
  own three failed attempts open it, and the read that decides "published or nothing" was
  then refused. S3-2's `test_a_switch_the_server_applied_but_whose_answer_was_lost_is_published`
  caught it once the retries landed; `_switched_to` now retries without the breaker.
- **pydantic's `ValidationError` repeats its input in its text, a value included**
  (`input_value=…`). `load_config` raises `ConfigError` `from None` and renders the messages
  only, so a key in a config never reaches a terminal; the schema test reads the rendered
  messages.
- **qdrant-client 1.19.1 uses a request's `timeout` as that HTTP request's own timeout too**
  (`http/api_client.py`), which D6 relies on to cut a hung search at the deadline.
- **What a repeat counts as done, on a throwaway Qdrant 1.19.2:** deleting a missing
  collection answers `False`; creating an existing one, 409; a repeated
  `create_payload_index` and a repeated alias switch both succeed with nothing changed.
- **The `fallback` extra's four packages** (jiter 0.14.0, langchain-openai 1.6.2, openai
  3.3.0, tiktoken 0.14.0) are all in the `eval` extra already, so S3-9's `eval` Audit step
  covers them.
- **langchain-ollama 1.1.0's `validate_model` catches httpx's `ConnectError`,** but ollama's
  client has wrapped it in the builtin `ConnectionError` by then, which escapes: a stopped
  Ollama at build is "ConnectionError: Failed to connect to Ollama…", not the `ValueError` its
  docstring promises. Harmless: `preflight` reports either.
- **An open breaker mid-stream logs a full traceback** ("An answer failed after its stream
  began", the real check's `2-serve.log`). It is an expected refusal, so one warning line
  would do: a backlog line for S3-7, whose structured logs take that log over.
- **S3-9: the image installs no extra.** A stack that wants the fallback needs the `fallback`
  extra in the image and `RAG_FALLBACK_API_KEY` in `.env`, which compose's `env_file` passes.
  A backlog line for S3-9.
- **The real check's drop could not land mid-stream:** Groq answered within `curl
  --max-time 2`. The two real-uvicorn drops through the fallback, during its prefill and
  mid-stream, are the proof that its stream closes.
- **A dropped answer is a cut-off chunked body:** a fake Ollama speaking HTTP/1.0 ends a body
  by closing the connection, which the client reads as a clean end. The test's fake speaks
  HTTP/1.1 with chunked transfer.

## First review (2026-10-23)

Run by the first reviewer on the PR's head, `d6671a7`, with a throwaway Qdrant (v1.19.2 by
the compose file's digest), a local fake OpenAI-compatible server with failure modes, and a
dead port for a stopped Ollama. The maintainer's Ollama, their Qdrant (read-only) and the
Groq key were not used. Reproduced:

- ruff, format and mypy clean; **1394 passed, 18 skipped, 5 xfailed**, coverage **98.44%**;
  **1402 passed** with `--extra fallback`; the 11 server tests pass; `rag-eval check` exit 0;
  the contexts file identical to S3-2's (`5a36e2f9…`); `uv.lock` moves only the project's own
  metadata.
- **The CI exception, extended:** the three GHSA IDs were published 2026-10-22 16:49 UTC,
  after `main`'s last green run (07:07 UTC); Audit without the ignores finds exactly 8 pypdf
  advisories and with them "No known vulnerabilities found, 8 ignored"; `get_fields` is called
  only by itself and `get_form_text_fields`; the expiry stays 2027-01-20.
- **Through the real httpx, ollama and openai stacks** (`rag-query` and `rag-serve` on the
  live index, read-only), Ollama on a dead port:
  - the fallback answers and is named; the server received `RAG_FALLBACK_API_KEY`'s value,
    never `OPENAI_API_KEY`'s (D1), and the chunks;
  - a 401 names the variable and no value, and is not retried; four questions against a 500
    reach the server 3 times, the 4th refused by the breaker ("29 s"); a hang times out after
    one request; a stream cut mid-answer is not retried or routed; no fallback is configured:
    exit 2 in 2 s;
  - over HTTP: `done` carries `components.llms.groq_fallback`, health reports `qdrant`, `llm`
    and `fallback`, the log names the fallback answer; a client dropped mid-fallback leaves 1
    socket to the provider while streaming and 0 within 0.4 s, "Answer cancelled", and the
    slot free for the next question;
  - no key value in any CLI output, server log or response (0 hits, unfiltered); a refused
    `api_key` does not echo its value.
- **The tier-2 fingerprint:** the Ollama timeouts, `deadline_s`, `timeout_s`, `fallback_llm`
  and edits to the fallback's entry move no part; `num_predict` moves `query`, so the probe
  can fail.

Findings, none a blocker:

1. **A second question during a half-open probe was let in** (fixed). `Serving.unavailable()`
   refused only an `OPEN` breaker, so with the probe out (`retry_after_s` 0.0; yours took
   168 s, and the read timeout allows 600) a second question took a slot, ran a retrieval and
   a rerank, streamed `sources` and ended in `error: Ollama is not answering`, with HTTP 200.
   `CircuitBreaker.refusal_s()` now says how long an attempt would be refused: the wait while
   open, and `PROBE_WAIT_S` (5 s) while half-open with the probe out. `unavailable()` uses it,
   so that question gets the 503 with `Retry-After` before any retrieval. The first question
   after the wait still goes through: it is the probe.
2. **A dropped fallback stream was blamed on Ollama** (fixed). The fallback's stream raises
   its HTTP client's own errors (httpx2's), which `raised_by_fallback` could not tell from
   Ollama's: `rag-query` printed "the connection to Ollama failed… Is Ollama running?" and the
   API "the language model is unreachable". `stream_answer` now marks an error raised by the
   fallback's stream (`resilience.mark_fallback`), and `raised_by_fallback` reads the mark.
   Re-run on the real stack: "the hosted fallback failed (RemoteProtocolError: …)".
3. **The retrieval deadline is not a hard bound** (backlog). Measured against a socket that
   accepts and never answers: a search honours `deadline_s: 15` (15.4 s), but the alias read
   inside the same deadline takes 20.2 s, since only `search` trims its request's timeout;
   the start-up check takes 31 s before exit 2.
4. **The no-secrets rule is name-based** (backlog). `default_headers` with `Authorization` or
   `x-api-key`, a hyphenated `api-key`, `token`, and a plain `http://` remote `base_url` all
   load.

**The fixes, and their tests** (9 added): `test_it_refuses_while_open_and_while_its_probe_is_out`,
`test_an_abandoned_probe_stops_the_refusal`, and two on the mark (`test_resilience.py`); two on
which errors are marked (`test_answering.py`); `test_a_dropped_fallback_stream_is_the_fallback_s_not_ollama_s`
(`test_cli.py`); `test_a_second_question_during_a_slow_probe_is_refused_before_the_first_byte`
and `test_a_dropped_fallback_stream_is_reported_as_the_fallback_s` (`test_api.py`). Each fix was
broken in turn (the `OPEN`-only check back, the mark not made, the mark not read): the new
tests failed each time, and the files were restored by checksum. After: ruff, format and mypy
clean; **1403 passed, 18 skipped, 5 xfailed**, coverage **98.45%**; **1411 passed** with the
extra.

Not re-run: the 15 hermetic cancel tests of DEC-14 one by one (they pass in the suite), and
the Groq check itself, which is the maintainer's key.

## Second review (2026-10-23, Opus 5.5)

**1 defect, fixed on the PR with three tests; S3-9's plan corrected.** Run on the PR's head,
`a0b77bd`, with both earlier records treated as claims to check. The live Qdrant was only
read: the alias still names `…7f20891e`. The server tests ran on a throwaway Qdrant v1.19.2
from the compose file's digest (`review-s3-6-qdrant`, port 16333, removed afterwards), and
the reproductions in a scratch copy of `HEAD`. The maintainer's Ollama and the Groq key were
not used.

- **Reproduced on `a0b77bd`:** ruff, format, mypy and `uv lock --check` clean, with and
  without the extra; **1403 passed, 18 skipped, 5 xfailed**, coverage **98.45%**, the HF
  cache empty; **1411 passed** with `--extra fallback`; the 11 `qdrant_server` tests;
  `rag-eval check` exit 0; tier 1 0.950 / 0.796 / 0.858, the contexts digest `5a36e2f9…`,
  IDENTICAL; CI green on the head (run 39932776608).
- **The breaker under contention:** 300 races of 64 threads on a half-open breaker, each
  granted exactly one probe; `abandoned()` freed exactly one successor; only the first
  outcome counted.
- **The Ollama timeouts are real:** the `ChatOllama` that `build_llm` makes from
  `config.yaml` has `Timeout(connect=5, read=600, write=30, pool=30)` on its async client
  and its sync one.
- **Every attempt is ended, on every path,** so no probe can leak and leave a breaker
  refusing for good: `_call` records a transient error as a failure, any other exception as
  an answer, and a `BaseException` as abandoned; `stream_answer` judges at the first token,
  by `_judge` on an exception, and abandons on a cancel or a `GeneratorExit`. Qdrant's
  wrapped transport errors are classified by their `.source`; the deadline is set inside the
  worker thread, where the `ContextVar` holds.
- **Secrets:** an HTTP client is sent fixed text per verdict, so no provider's body
  (OpenAI's 401 repeats a masked key) leaves the server; the key is a `SecretStr` our code
  passes.
- **The audit exception, today:** CI's Audit command finds exactly the eight pypdf
  advisories and clears them with `ci.yml`'s eight ignores. The three new ones match their
  reasons (GHSA-6vvc form fields, GHSA-cgf9 content streams, GHSA-f54x arrays and
  dictionaries; medium, fixed in 6.21.0), and `get_fields` is reached only through itself
  (`_build_field`, `_check_kids`) and `get_form_text_fields`, neither of which the loader
  calls.
- **Commits:** the eight on the PR have parseable trailers, the `!` and a
  `BREAKING-CHANGE:` trailer on the feat commit, no body line over 72 and no attribution,
  nor does the PR body.

Findings, both fixed:

1. **A readiness poll during Qdrant's half-open probe unserved a healthy index.**
   `Serving.refresh()` drops the served index when its alias read fails, and a read the
   breaker refused, half-open with another request's probe out, failed the same way.
   Readiness, `/v1/documents` and every question refresh, and a supervisor polls readiness
   while the clients told `Retry-After: 30` all come back at once. Reproduced with the
   suite's fixtures in a scratch copy of `HEAD`, three runs of three: a poll during a slow
   probe got 503 ("the index could not be opened just now"), the log said "Not serving an
   index: Qdrant is not answering … Is Qdrant running?" while Qdrant answered, and the
   probe's question paid a reopen (7 alias reads, where the fix needs 1). It healed on the
   next request. The fix, in `api/state.py`: a refused read changes nothing, before the
   reopen lock and under it. Under it, the first read's failure still decides when that
   read failed against Qdrant: its three attempts are what opened the breaker. A first fix
   that returned on every refusal hid a real outage from the log; the second test caught it.
   The three tests in `tests/test_api.py`:
   - `test_a_readiness_poll_during_qdrant_s_probe_keeps_the_index_served`: readiness 200
     during the probe, the probe's answer `done`, one alias read, no warning;
   - `test_qdrant_down_on_a_refresh_is_named_by_its_own_failure_not_the_breaker_s`: Qdrant
     down unserves the index, and the warning is "cannot use Qdrant", where `HEAD` logged
     the breaker's "Qdrant is not answering";
   - `test_a_read_again_refused_under_the_reopen_lock_changes_nothing`: a refusal under the
     lock opens and drops nothing.

   Against four versions of `state.py` in the scratch copy: `HEAD` fails all three; a return
   on every refusal fails the second; "the first read always decides" fails the third; the
   fix passes all three.
2. **S3-9's plan still carried five of the eight pypdf ignores** in two places, its `eval`
   Audit step's text and its verification command, after this story's change to its prose.
   Both now name the eight; the IDs equal `ci.yml`'s.

Considered, no change: a non-probe attempt that outlives an open period can close the
breaker before a stale probe's failure opens it again on one failure. It needs an attempt
longer than 30 s overlapping a probe, so Ollama with `max_concurrent` above 1 only, and it
costs one extra open period.

After the fixes: ruff, format and mypy clean; **1406 passed, 18 skipped, 5 xfailed**,
coverage **98.45%**; `rag-eval check` exit 0. Not re-run with `--extra fallback`: the fix is
in the API's refresh, which the fallback does not touch.

## Deviation from plan

- **Model:** fable-routed, run on Opus 5.5 at max effort, since Fable is unavailable
  (CLAUDE.md, "Model routing").
- **`langchain-openai>=1,<2`,** where the story says `>=1`: the bound the other LangChain
  packages carry (D12, as approved).
- **Additions the plan left implicit:**
  - `QueryPipeline` also carries `llm_ref` and `fallback_ref`, what `Done.model` names;
  - `chain.assemble`, the one composition the CLI and the API share;
  - `components.Preflight(problem, warning)`, in place of a reason or `None`;
  - `rag-serve` logs at start-up that a fallback can answer, when Ollama is up.
- **The S3-3 backlog line's debug-logging suggestion was not taken:** the timeout never
  reproduced, the retries absorb one, and an exhausted one now says "did not answer in time"
  (D13).
- **The second review changed `api/state.py`** (its finding 1, with three tests) and S3-9's
  plan (finding 2).
- **A CI change outside the story's scope** (maintainer, 2026-10-23): the Audit step's pypdf
  exception grew by three advisories (Discovered), in a `ci(audit)` commit of its own, as
  S3-5's `ba00edc` did for the first five.
