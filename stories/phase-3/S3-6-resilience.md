# S3-6: Timeouts, retries, circuit breakers and the hosted fallback

| | |
| --- | --- |
| **Status** | Todo |
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
  - **Qdrant:** every call in `vectorstore`, on connection errors, timeouts and 5xx. All of
    them are idempotent (DEC-19).
  - **Ollama:** only before the first token, and only on a connection error, 2 attempts. A
    retry after a token would duplicate text.
  - The retry loop runs in the consuming task: no new task near the stream (DEC-14).
- **Breakers**, one per dependency: Ollama, Qdrant, and the fallback.
  - **API:** a breaker that is open is a dependency that answers 503 with `Retry-After`,
    before the first byte (DEC-18's rule), and `/v1/health` reports each breaker's state.
  - **Start-up checks** (`rag-query`, `rag-ingest`, `rag-eval`, `rag-serve`) exit 2 naming
    the service.
  - **The reopen back-off (backlog):** while Qdrant's breaker is open, `Serving.refresh()`
    returns at once.
- **The hosted fallback.**
  - **Schema:** `pipeline.query.fallback_llm`, an optional reference to an llms entry
    different from `pipeline.query.llm`.
  - **Config:** a `components.llms` entry on `langchain_openai.ChatOpenAI`, with `model`,
    `base_url`, `temperature: 0` and a token limit. It is referenced by nothing in the
    committed `config.yaml`; the real check uses a scratch overlay or config.
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
  - Qdrant's retries, with a fake transport.
- **A config test:** if a test imports every `_target_` in `config.yaml`, give an entry whose
  extra CI does not install a way through. CI does not install `fallback`.

## Out of scope

- Caching the health probes and rate limits (S3-8).
- Metrics and spans for the breakers (S3-7 adds them).

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0: runtime keys only
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

# Ollama down: a fast, explicit refusal (no fallback configured).
# The maintainer runs the systemctl lines (sudo).
sudo systemctl stop ollama
uv run rag-query; echo "exit $?"                          # 2, naming Ollama, within seconds
# over HTTP: 503 with Retry-After before the first byte, once the breaker is open

# the one real check (maintainer's key in the shell; a scratch overlay naming the fallback)
uv sync --extra fallback
uv run rag-query --config <scratch config with fallback_llm>   # answers; "(answered by …)"
# Ctrl-C during the fallback's answer: the prompt returns at once, nothing printed after
sudo systemctl start ollama && uv sync
gh pr checks
```

## Review notes for the human

- **The stream path:** no new task, no `.with_fallbacks()`, and the same `finally` closing
  whichever model streamed. Then read the forced-cancel tests.
- **Retries only before the first token,** and only on connection errors.
- **The secret rule:** no key name, and no value from a provider's variable, in any log
  line, error or event.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
