# S2-1: Stream answers as events; make Ctrl-C cancel generation

| | |
| --- | --- |
| **Status** | Done 2026-10-03 (PR #7). CI green on the branch at `330c54d`, push and PR runs (Verification 5). Reviewed twice; follow-ups below |
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
    and `async aclose()`. `retrieve` may be `None`, for the API's start before the first
    ingest (S2-7).
  - `build_retrieve(config, embeddings)` and
    `build_query_pipeline(config, *, require_index=True)`. With `require_index=False` and
    no usable index, `retrieve` is `None` and the LLM half is still built. With an index,
    both halves are built either way (§2.4).
  - `NoIndexError`, raised by `stream_answer` when `retrieve` is `None`.
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
    leaves the stream, even before S2-6 stores `source` relative. When a chunk's source is
    not under `corpus_root` (an index built on another machine), it falls back to the
    bare file name rather than raising.
- **`components.aclose_llm(llm)`.** Closes ChatOllama's async and sync clients, through
  the private `_async_client` / `_client`, via `getattr` (there is no public close). It
  is a no-op for models without them.
- **`cli.py`.**
  - One `asyncio.Runner` per session. Each answer is
    `runner.run(render(stream_answer(...)))`, with tokens as they arrive and the numbered
    sources after, as today.
  - Ctrl-C during an answer: the Runner cancels the task, the stream closes,
    "(answer interrupted…)" is printed, and the prompt returns. Catch both
    `KeyboardInterrupt` and `CancelledError` from `runner.run()` (DEC-14).
  - A second Ctrl-C while that answer is still unwinding quits the session. The task is
    then not done, so call `runner.close()`, which cancels and finalizes it, rather than
    leaving it pending for the next `run()`.
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
    shape must fail. That shape is the bug. Write it as `xfail(strict=True)`, so that a
    langchain-core fix turns the suite red instead of leaving a silent pass.
  - A chunk whose source is outside `corpus_root` yields a file-name `SourceRef.source`.
  - `tests/test_cli.py`: the REPL over a fake pipeline. A SIGINT raised inside the answer
    (`signal.raise_signal` from the fake model, under the real `Runner`) returns to the
    prompt, and the next question is answered. The test first asserts
    `signal.getsignal(signal.SIGINT) is signal.default_int_handler`, since the Runner
    installs its handler only then. A plugin's handler then fails the test with a clear
    message instead of aborting the pytest session.
  - Two SIGINTs inside one answer end the session with exit 0, and no task is left
    pending on the loop.
  - `aclose_llm` closes a real `ChatOllama`'s httpx clients. No server is needed, since
    the clients are created at construction.
  - The existing `build_rag_chain` tests pass unchanged.

## Out of scope

- The reranker itself (S2-4), the HTTP API and its disconnect handling (S2-7), `readline`
  and the other CLI polish (S2-8), and storing `source` relative in the index (S2-6).
- Any change to the prompt or to `format_docs`' output. The same retrieval must give the
  same prompt text: the tier-2 fingerprint will hash the template and a rendered probe
  of it (DEC-15), and the committed baseline is measured on that text.

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
#      → the prompt is back within ~1 s, the log shows the request ending at that
#        moment, and the runner's CPU drops to idle at once
#    ask again; Ctrl-C during "Searching…", i.e. during the prefill
#      → the prompt is back within ~1 s and the request ends in the log. The runner may
#        finish the prompt evaluation in progress before idling (DEC-14, reported for
#        Ollama, not yet seen here): record how long it stays busy. Not a failure,
#        provided it idles by the end of the prefill and no tokens are generated
#    ask a third question: it starts answering without waiting for an earlier one
#    Ctrl-C twice, quickly, mid-answer → the session exits 0, no traceback
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

- **Measured against real Ollama** (mistral, 2026-10-02; caveats 3 and 21). Driven through a
  pseudo-terminal with real Ctrl-C bytes, under `uv run rag-query` as written above.
  - **First token after 254 s** for a k=5 prompt. Ollama ran mistral on 2 threads of this
    i7-1255U (200% CPU), with about 6 GB free. Caveat 3's 5–15 s does not hold here today.
  - **Ctrl-C mid-answer.** The prompt is back at once, and the `/api/chat` line ends in the
    same second as the Ctrl-C. The runner idles 0.5–1.5 s later, once the token in flight
    is done.
  - **Ctrl-C in the prefill.** The prompt is back at once. The request ends at the Ctrl-C
    (logged as a 500 after 2.98 s). The runner then **finishes the abandoned prompt
    evaluation and generates nothing**: caveat 21, now seen here.
    - One abandoned prefill went idle about 4 min after it began.
    - The same question asked next answered in 0.3 s, from the prompt cache that prefill
      left.
    - A question asked during those minutes queues in Ollama behind it (more than 5 min in
      the first pass). Our side is finished at the Ctrl-C; the wait is Ollama's.
  - **One Ctrl-C under `uv run` returns to the prompt.** `uv` does not deliver an extra
    SIGINT.
  - **Two Ctrl-Cs at 100, 20, 5 and 1 ms apart** all went "back to the prompt, then quit",
    with exit 0 and no traceback. Against real Ollama the unwind ends in under 1 ms, so only
    the hermetic test's slow unwind reaches the quit-mid-unwind path.
- **Closing `stream_answer` at a `yield` does not close the model stream before `aclose()`
  returns.** This is input for S2-7.
  - langchain-core 1.6.3's `RunnableSequence._atransform` and `BaseChatModel.astream` iterate
    their inner generators with `async for` and never close them.
  - The loop's async-generator finalizer closes them, one level per iteration: about 20,
    driven by reference counting, so it needs no garbage collection. No token is generated
    in between.
  - Cancellation (Ctrl-C, and Starlette's disconnect under DEC-18) unwinds every frame first,
    so it is unaffected. S2-7 must keep a disconnect a cancel, as DEC-18 already has it.
  - `test_a_consumer_that_stops_early_closes_the_stream_while_the_loop_runs` pins this.
- **Telling the second Ctrl-C from the first needs the answer's task**, which `Runner.run()`
  keeps to itself. `_Answer.run()` records `asyncio.current_task()`, and the session quits
  when that task is pending, or finished with `KeyboardInterrupt`: the second SIGINT can land
  in the task's own step, and reading the exception also stops a "Task exception was never
  retrieved" log.
- **Empty text chunks are not `Token`s.** ChatOllama's last chunk is empty, so `ttft_ms` times
  the first visible token.
- **`SourceRef.source` also falls back to the file name for a relative source that climbs out
  with `..`**, so nothing outside the corpus can be named.
- **`citation()`'s page logic is now `chain.page_label()`**, shared with `SourceRef.page`. Its
  output is unchanged, as the existing tests show.
- **The deprecation gate also runs `stream_answer`** (maintainer, 2026-10-02), since every
  front end answers through it now.

## Deviation from plan

- **Plan-first was skipped at the start.** Code was written before a plan. The maintainer had
  the plan written over the work already in the tree (approved 2026-10-02), and nothing was
  rewritten.
- **The second-Ctrl-C quit also closes the model's clients.** The scope says to call
  `runner.close()`. That closes the loop, and ChatOllama's async client can only be closed on
  its own loop.
  - `cli.close_session()` cancels the leftover answer before the loop runs again, waits for
    it, runs `pipeline.aclose()`, then calls `runner.close()`.
  - Without the cancel, an `aclose` that takes a few loop iterations (httpx's does) lets the
    half-unwound answer resume with the `KeyboardInterrupt` and raise it out of the REPL.
    A mutant showed it, and the two-Ctrl-C test now fails that mutant.
- **Model:** this session ran on Opus 5.5; the story names Fable.
- **Verification 5 (CI)** can go green only after the maintainer pushes the branch.
- **First-review follow-up (2026-10-02):**
  - a failed `aclose()` in `stream_answer`'s `finally` no longer replaces the stream's
    own error (`suppress(Exception)`; a cancel still passes), with a test;
  - the no-index check moved into `build_query_pipeline` (`NoIndexError`), so the CLI
    has one source for it;
  - a Ctrl-C while the session closes exits quietly;
  - the README names the prefill wait;
  - the claim that a cancel during retrieval returns at once was measured.
- **Second review (2026-10-03).** The core claim holds: the gates are green; the
  cancellation and CLI tests passed five runs in a row; the negative control fails for
  its intended reason (`assert False`, model still streaming); a Ctrl-C during a slow
  retrieval returned the prompt in under 1 s with a process-directed SIGINT; the lock is
  current. Follow-ups:
  - **`close_session` caught only Ctrl-C.** A probe showed an error from `aclose()`
    escaping `repl()` as a traceback at exit. It now catches `Exception` too, best
    effort, and says so in one line.
    `test_an_error_while_the_clients_close_still_exits_cleanly` fails without the fix.
  - **A stale docstring.** `answering.py` said the API's "disconnect watcher" stops
    generation, but DEC-18 has had no watcher since the second design review. It now
    names Starlette's own cancel, and warns against a separate task.
  - **The negative control's `xfail` gained `raises=AssertionError`**, so an unrelated
    error in the test can no longer count as the expected failure.
  - **The 254 s has a cause.** Ollama's journal shows `NumThreads:2` on every mistral
    load (2 performance cores out of 12 threads), and a 2.3 GB swap peak. Caveats 3
    and 16, S2-5b, ARCHITECTURE.md (DEC-15) and the README now say so. A new backlog
    line has S2-5b set `num_thread` first: it is hashed into the tier-2 fingerprint, so
    it must precede the baseline. Caveat 3 also no longer states the figure twice.
  - Not changed: a cancel that is not a Ctrl-C (a library's own) is reported as "answer
    interrupted". Cosmetic, and `Runner.run()` can raise `CancelledError` after a real
    Ctrl-C as well, so the two cannot be told apart cleanly.
  - For S2-7: `NoIndexError` from `build_query_pipeline` carries the absolute index
    path. That is fine on the CLI's stderr, but it must never reach an HTTP body
    (DEC-18).
