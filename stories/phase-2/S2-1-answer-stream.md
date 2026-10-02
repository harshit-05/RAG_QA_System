# S2-1: Stream answers as events; make Ctrl-C cancel generation

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-5 (structured citation output); backlog: Ctrl-C does not cancel generation, ChatOllama client left open (CLI half), `py.typed` + `types-PyYAML` |
| **Depends on** | — |
| **Model** | fable |
| **Plan-first** | yes |
| **Branch** | `feat/s2-1-answer-stream`. Risky: it replaces how every answer reaches the terminal, and the cancellation path is subtle (DEC-14's verified trap). |

## Goal

After this story, every answer flows through one async event stream,
`rag_qa.answering.stream_answer`:

- the CLI renders it from this story on;
- the API (S2-7) and the evaluation harness (S2-5) will consume it later.

Ctrl-C during an answer stops the generation in Ollama. Today the session waits for the
whole response first: 43 s with phi3 in the S1-8 manual test, minutes with mistral.

Design: ARCHITECTURE.md §2.1 DEC-14, interfaces §2.4. The story also opens Phase 2's
version line (`0.3.0.dev0`) and takes the two typing items the backlog parked on the
first packaging change.

## Scope

- **`chain.py`.**
  - `QueryPipeline`: a frozen dataclass with `retrieve`, `answer`, `llm`, `corpus_root`
    and `async aclose()`.
  - `build_retrieve(config, embeddings)` and `build_query_pipeline(config)`.
  - `retrieve` is the retriever. The existing, disabled reranker path moves inside it
    unchanged, and S2-4 replaces it.
  - `answer` is `prompt | llm | StrOutputParser()`.
  - `build_rag_chain(config)` keeps its signature and the §0.4 contract, composed from
    the same parts. Its docstring says it is not for cancellable streaming (DEC-14).
- **`answering.py`, new.**
  - Types: `SourceRef`, `Sources`, `Token`, `Done`, `AnswerEvent`.
  - `stream_answer(pipeline, question)`:
    - awaits `pipeline.retrieve.ainvoke(question)` and yields `Sources`;
    - streams `pipeline.answer.astream(...)` **directly**, and calls `aclose()` on it in
      a `finally`;
    - yields `Done(ttft_ms, total_ms)`.
  - `SourceRef.source` is made relative to `corpus_root`, so no absolute host path
    leaves the stream, even before S2-6 stores `source` relative.
- **`components.aclose_llm(llm)`.** Closes ChatOllama's async and sync clients, through
  the private `_async_client` / `_client`, via `getattr` (there is no public close). It
  is a no-op for models without them.
- **`cli.py`.**
  - One `asyncio.Runner` per session. Each answer is
    `runner.run(render(stream_answer(...)))`, with tokens as they arrive and the numbered
    sources after, as today.
  - Ctrl-C during an answer: the Runner cancels the task, the stream closes,
    "(answer interrupted…)" is printed, and the prompt returns.
  - Ctrl-C at the prompt, end of input, `exit` and `quit` behave as today.
  - On exit, `runner.run(pipeline.aclose())`.
  - Exit codes (0/2) and staying alive after a failed answer (ISS-06) are unchanged.
- **Version:** `0.2.0` → `0.3.0.dev0`, in both `pyproject.toml` and
  `src/rag_qa/__init__.py`.
- **Typing.** Add an empty `src/rag_qa/py.typed`. Run `uv add --dev types-PyYAML`, drop
  `yaml` from mypy's no-stub overrides, and fix what mypy then reports.
- **Tests.**
  - `tests/test_answering.py`: event order and content against a fake pipeline built on
    a fake chat model. `SourceRef` numbering matches the prompt's `[n]`, and the timings
    are populated.
  - **Cancellation:** a fake `BaseChatModel` whose `_astream` sleeps (a simulated
    prefill), then yields slowly and records in a `finally` that it was closed.
    Cancelling the consumer must find the stream already closed when the consumer
    returns. Test it both mid-stream and during the prefill.
  - **Negative control:** the same assertion against the `RunnablePassthrough.assign`
    shape must fail. That shape is the bug.
  - `tests/test_cli.py`: the REPL over a fake pipeline. A SIGINT raised inside the answer
    (`signal.raise_signal` from the fake model, under the real `Runner`) returns to the
    prompt, and the next question is answered.
  - `aclose_llm` closes a real `ChatOllama`'s httpx clients. No server is needed, since
    the clients are created at construction.
  - The existing `build_rag_chain` tests pass unchanged.

## Out of scope

- The reranker itself (S2-4), the HTTP API and its disconnect watcher (S2-7), `readline`
  and the other CLI polish (S2-8), and storing `source` relative in the index (S2-6).
- Any change to the prompt or to `format_docs`' output. The same retrieval must give the
  same prompt text, because the tier-2 fingerprint will hash the prompt.

## Verification

```bash
# 1. the gates, as CI runs them
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. the trap and the fix, hermetically
uv run pytest tests/test_answering.py -v -k cancel
#    → the direct shape closes before the consumer returns, mid-stream and in prefill
#    → the negative control (the assign shape) fails the same assertion

# 3. real Ollama, real terminal (manual; caveat 21). Check `ollama ps` and `free -h`
#    first (caveat 8). In a second terminal:
journalctl -u ollama -f        # or watch the ollama runner's CPU in `top`
uv run rag-query
#    ask a long question; Ctrl-C after a few tokens
#      → the prompt is back within ~1 s, and Ollama's request ends at that moment
#    ask again; Ctrl-C during "Searching…", i.e. during the prefill → same
#    ask a third question: it starts answering without waiting for an earlier one
#    Ctrl-C at the prompt → exit 0, with no traceback and no "Exception ignored"

# 4. packaging
grep '^version' pyproject.toml; grep __version__ src/rag_qa/__init__.py   # → 0.3.0.dev0
uv run python -c "import importlib.resources as r; print(r.files('rag_qa').joinpath('py.typed').is_file())"   # → True
grep -n '"yaml"' pyproject.toml          # → gone from the mypy overrides

# 5. CI
gh run list --limit 1                    # → green on the branch
```

## Review notes for the human

- **Read the negative-control test first.** If it ever passes, the suite no longer tells
  the bug from the fix.
- **Then read `stream_answer`'s `finally` and the CLI's `Runner` handling.** The whole
  story is the claim that the HTTP stream to Ollama is closed before `runner.run()`
  returns.
- **Check that `build_rag_chain` and `stream_answer` build their prompts from the same
  parts.** Two compositions that drift apart would make the API and the CLI answer
  differently.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
