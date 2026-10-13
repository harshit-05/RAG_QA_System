# S2-8: Polish the command-line tools

| | |
| --- | --- |
| **Status** | In review (2026-10-13): implemented on `fix/s2-8-cli-polish`, not yet committed. Verification 1–5 are run and shown below, 3 by the maintainer's hand. Verification 6 waits for the PR. First review done: no blockers, two code findings fixed, one commit-marking finding recorded. |
| **Closes** | backlog: "CLI polish" (found in S1-4, plus two items from the v0.2 manual test) |
| **Depends on** | S2-1 (rewrites `cli.py`), S2-6 (rewrites `ingest.py`) |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `fix/s2-8-cli-polish`. Low risk: messages, exit codes and one stdlib import. It is on a branch so that CI runs before the merge. |

## Goal

Clear the batch of small command-line defects. None of them was worth a story alone; each
one makes the tools read as unfinished:

- tracebacks where a sentence belongs;
- arrow keys arriving as text;
- noise above the banner;
- zero-count lines on every run.

Each item is listed under the "S2-8 — CLI polish" line in the STATUS.md backlog, with the
reason behind it.

## Scope

- **`rag-query`.**
  - **Line editing:** `import readline` in `cli.py` gives editing and history. Today an
    up-arrow reaches the model as the question `^[[A`.
  - **Startup guard:** Ollama being down (`validate_model_on_init`), and Ctrl-C while the
    models load, each give one clean message and exit 2, instead of a traceback.
  - **Connection errors name Ollama**, with the hint "Is Ollama running?
    `systemctl status ollama`".
  - **No empty header:** the `Answer:` header prints only once the first token or the
    error arrives.
- **`rag-ingest`.**
  - Summary lines with a zero count are left out. The failure line says "item(s)", since
    folders count too.
  - An unreadable corpus root reports "Corpus directory not readable" and exits 2, to
    match "not found".
  - The `pypdf` logger is set to ERROR, so a corrupt PDF's own warnings stop printing
    above the banner, before the clean error.
- **`scripts/fetch_dataset.py`.**
  - Its corpus guard also checks `paths.data` and `RAG_DATA_PATH`, not only the repo's
    `corpus/`.
  - It checks the `PAR1` header before writing.
  - It declares `requests` (`uv add requests`) instead of relying on a transitive copy.
- **Tests** for each item that can be tested hermetically:
  - the startup guard, with a fake LLM factory that raises;
  - the summary formatting;
  - the root-unreadable exit code;
  - the `fetch_dataset` guard and the header check, with the network faked.

## Out of scope

- Anything not on the backlog list. New findings go under Discovered.
- The API's error messages (S2-7).

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. Ollama unreachable, without touching the real service: a scratch config whose LLM
#    points at a closed port
sed 's/model: "mistral"/model: "mistral"\n      base_url: "http:\/\/127.0.0.1:9"/' config.yaml > <scratch>/no-ollama.yaml
RAG_DATA_PATH=$PWD/corpus RAG_VECTOR_STORE_PATH=$PWD/vectorstore/db_faiss uv run rag-query --config <scratch>/no-ollama.yaml; echo "exit $?"
#    → 2, one line naming Ollama, no traceback

# 3. manual, in a real terminal: arrow keys and history at the prompt; Ctrl-C while loading → exit 2, no traceback

# 4. ingest messages, on scratch paths (caveat 9)
RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-ingest 2>&1 | grep -c ' 0 '   # → 0
mkdir <scratch>/locked && chmod 000 <scratch>/locked
RAG_DATA_PATH=<scratch>/locked RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-ingest; echo "exit $?"      # → 2, "not readable"
chmod 700 <scratch>/locked

# 5. a corrupt PDF: no pypdf noise above the banner, then the clean error (exit 1)

# 6. CI
gh run list --limit 1
```

## Review notes for the human

- **Each change is small; check each matches its backlog line.**
- **Check that the startup guard catches exactly the startup failures.** It must not
  widen into swallowing errors in the REPL loop, which ISS-06 already handles on purpose.

## Discovered

- **libedit drops a marked sequence that ends the prompt.** uv's Python builds `readline`
  on libedit ("EditLine wrapper"), whose `prompt_print` gives up on a `\001…\002` sequence
  at the very end of a prompt (its source says `XXX: We lose`). With the reset last, the
  typed question and "Searching for an answer..." stayed blue. The pty pre-check of
  Verification 3 showed it. Fixed here: the prompt ends on its space, after the reset, and
  a test pins that.
- **ChatOllama raises a plain `ConnectionError` at build time** when Ollama is down
  (`validate_model_on_init`), not a pydantic error. So `preflight` reports
  "cannot build the language model …: ConnectionError: Failed to connect to Ollama…". Its
  stream raises httpx's `ConnectError` ("All connection attempts failed") unwrapped. The
  REPL's Ollama naming matches httpx's errors by class name rather than importing httpx,
  which reaches us only through ollama.
- **Two log lines keep the plural slip "1 pages/sections" had:** "Embedding 1 chunks..." and
  "Published …: 1 documents, 1 chunks." They are not on the backlog list, so they were left
  as they are (backlog line).
- **The API's ingest jobs still print pypdf's warnings** to `rag-serve`'s stderr. The
  ERROR level is set for `rag-ingest`'s run only, like its log handler, and the API is out
  of scope (backlog line).
- **For S2-9:** ARCHITECTURE.md §2.2 has a dated note for `cli → components`. Its layout
  line for `components.py` still lists only `build_reranker, aclose_llm`; since S2-7 and
  S2-8 it also holds `llm_reachable`, `preflight` and `OLLAMA_HINT`.

## Verification record (2026-10-13)

1. **The gates:** `ruff check` and `mypy` were clean (30 source files). `pytest
   --cov=rag_qa` gave **882 passed, 5 xfailed**, coverage 98.34%, on the final tree. After
   the first review's two fixes it is **887 passed, 5 xfailed**, coverage 98.31%.
   - `cli.py`'s uncovered lines are all older: the empty-sources return, the
     timing-dependent unwind branches, and `__main__`.
   - **Caveat 23:** the suite under `HF_HOME=<empty dir> HF_HUB_OFFLINE=1` gave the same
     882 passed, and the folder stayed empty.
   - **Caveat 24:** `uv run rag-eval check` exited 0, with all six fingerprint parts ok
     and every floor met. No hashed input moved.
   - **The lock:** `uv add requests` added two lines to `uv.lock` (requests 2.34.2, already
     locked as a transitive dependency) and moved no version. `uv lock --check` passed.
   - **Stability:** `tests/test_cli.py`, with its new real-SIGINT test, passed 15 of 15
     runs pinned to one core.
2. **Ollama unreachable.** The scratch config from the story's `sed`, with
   `base_url: "http://127.0.0.1:9"` on `mistral_ollama`, against the real index (read
   only). It printed, with **exit 2** and no traceback:

   ```text
   Error: cannot build the language model components.llms.mistral_ollama: ConnectionError: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download
   Is Ollama running? Try: systemctl status ollama
   ```

   Piped, stderr printed above the banner, since stdout is block-buffered off a terminal.
3. **Passed by hand** in the maintainer's terminal (2026-10-13): arrow keys and history at
   the prompt, and Ctrl-C while loading, exit 2 with no traceback. It ran on the tree with
   the first review's fixes.

   **Pre-checked before that in a pseudo-terminal.** A driver ran `rag-query` under
   `pty.fork()` with `TERM=xterm-256color`, so `input()` read through readline.
   - **History and editing,** on a scratch index with the fake embedder and a fake LLM, so
     no question loaded mistral. A question was answered. Then Up redrew it at column 11,
     right after the 10 visible characters of "Question: ", so the zero-width markers
     work, and Enter asked it again. Then `eit`, Left, Left, `x` became `exit`: "Exiting...",
     exit 0.
   - **Ctrl-C while loading,** with the real config: `^C` 0.3 s and 1.5 s after "Loading
     vector store and models..." each printed "Interrupted while loading; the session
     did not start." and **exit 2**, no traceback. `ollama ps` showed no model loaded.
4. **Ingest messages,** with the real config on scratch paths (caveat 9). The corpus was a
   text file, a markdown file and a PNG.
   - **The first run** (full rebuild, exit 0) printed "1 page/section" per file, then
     "Loaded 2 document pages/sections, split into 2 chunks.", "Rebuilt the index in full:
     there is no index yet.", "Documents: 2 added." and "Skipped 1 file(s) with no
     configured loader.". `grep -c ' 0 '` gave **0**.
   - **The second run** (nothing changed, exit 0) printed "Documents: 2 unchanged." and the
     skipped line. `grep -c ' 0 '` gave **0**.
   - **A `chmod 000` corpus root:** "Error: Corpus directory not readable: …/locked
     (Permission denied; set by paths.data or RAG_DATA_PATH)", **exit 2**.
5. **Two corrupt PDFs** added to that corpus (`not a pdf at all`, and `%PDF-1.4` with
   nothing after it). **Exit 1.** No pypdf line was printed, only "Error loading junk.pdf:
   PdfStreamError: …" for each, the summary ("Failed 2 item(s).") and the error line.
   - **The control:** the same run with the `pypdf` setting disabled in-process printed
     "invalid pdf header: b'not a'" and "EOF marker not found" again.
6. **CI:** `gh run list --limit 1` shows `main`'s push run for `313671a`, a success. This
   story's run comes with its PR, since CI runs on pull requests, not on branch pushes.

## Review (2026-10-13)

**First review (Opus 5.5): no blockers; 2 code findings fixed, 1 commit finding recorded,
3 nits left.** It reviewed the uncommitted tree against `HEAD`.

- **Checked and sound:**
  - the gates, re-run: ruff and mypy clean, **882 passed, 5 xfailed**, 98.34%; the same
    suite under an empty offline HF cache; `rag-eval check` exit 0; `uv lock --check`;
  - Verification 2, 4 and 5, run again on scratch paths with the same results, and the
    index-before-Ollama order with both missing;
  - line editing, in a pseudo-terminal with the real `question_prompt()` under libedit:
    Up recalled the previous question, and the colour reset came before the typed text;
  - an API ingest job over an unreadable corpus root: `exit_code` 2, and no host path in
    its log (the story's tests used a stub exception);
  - the startup guard wraps `start()` only, so it does not reach the REPL loop's own
    handling (ISS-06); the new Ctrl-C test passed 8 of 8 runs on one core;
  - the `preflight` move is a clean extraction, and the architecture test pins the rule.
- **Not re-run by the reviewer:** Verification 3 by hand, Verification 6 (it needs the PR),
  and a real Ollama chat error (`describe_failure` was given constructed errors).
- **Fixed on the branch, test first (both failed on the old code):**
  - **An `OSError` from the index check got the Hugging Face hint.** One broad catch
    covered `check_index` and the model build, so a `PermissionError` on the manifest read
    "cannot load the models" with "a model may be missing from the Hugging Face cache".
    `check_index` is now caught on its own: "cannot read the index: …", with no hint.
  - **`ReadTimeout` and `PoolTimeout` got "Is Ollama running?".** `_connection_failed`
    matched all of httpx's `TransportError`. It now matches `ConnectError`,
    `ConnectTimeout`, `ReadError`, `WriteError` and `RemoteProtocolError`, plus the builtin
    `ConnectionError`. A slow model prints its error without the hint.
- **Recorded: the exit code of an unreadable corpus root is a contract change.** It was 1
  and is 2, for `rag-ingest` and for an API ingest job's `exit_code`. CLAUDE.md marks that
  with `!` and a `BREAKING-CHANGE:` trailer. The same commit drops the zero-count summary
  lines and words the last log line "N item(s)", which an API job returns. Mark the
  `rag-ingest` commit, and the board's Now section says so.
- **Nits, no change:**
  - `requests>=2.34.2` is a main dependency for a standalone helper, with its lower bound
    at the locked version. A dev-group entry would keep it out of the Phase 3 image; it is
    installed transitively already, and the story asked for it declared;
  - the `PAR1` check covers the header only. Parquet files end in `PAR1` too, so the tail
    would catch a truncated download.

## Deviation from plan

Ran on Opus 5.5. Scope calls made during implementation, for review:

- **"item(s)" in both failure lines.** The summary's "Failed N item(s)." and the error line
  "N item(s) could not be read" both count `report.failed`, folders included. The error
  line is also an API ingest job's last log line, so one assertion in `test_api.py` moved
  with it.
- **The unreadable root is a new `CorpusUnreadableError`.** It is added to the
  could-not-start lists of `rag-ingest` and of the API's ingest jobs, so both report exit 2
  for it, as they do for a missing root. Catching `PermissionError` there instead would
  have made a failure to write the index read as "could not start".
- **The zero-count rule also covers the "Updating the index: …" log line**, which
  Verification 4's grep reads.
- **`rag-query`'s start order:** the index first, from its files, then `preflight`, then
  the models. So a missing index never waits on Ollama, and Ollama down never waits on the
  embedder. After `preflight`, a model that will not load is one broad catch at start-up,
  as `preflight`'s own is, with a Hugging Face cache hint for an `OSError`. The REPL's own
  catch is unchanged.
- **`api/server.py` imports only `preflight`** from `components`. `OLLAMA_HINT` is used
  inside `preflight` alone, and an unused import fails ruff. The tests import the hint from
  `components`.
- **Added beyond the listed tests:**
  - an architecture test that `rag_qa.cli` loads no `rag_qa.api` module, and does load
    `readline`;
  - the Ctrl-C test with a real SIGINT during `preflight`'s ping.
