# S3-6: Timeouts, retries, circuit breakers and the hosted fallback

| | |
| --- | --- |
| **Status** | In progress (2026-10-22): the plan below was approved in plan mode on 2026-10-22, on `feat/s3-6-resilience`. |
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
