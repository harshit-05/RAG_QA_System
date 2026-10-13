# S2-8: Polish the command-line tools

| | |
| --- | --- |
| **Status** | Done (2026-10-13): PR #16, merged as `5961de8`. CI was green on the PR's head, `573e733` (run 38343630988), and on `main` after the merge (run 38344128852). Verification 1–6 are run and shown below, 3 by the maintainer's hand. Two reviews, no blockers: the first fixed two code findings and marked the breaking commit; the second fixed one code finding (`a5f75b9`). |
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
6. **CI is green on PR #16** (2026-10-13). `gh run list --limit 1` shows run 38341250960,
   a `pull_request` run on `b3bec07`, the PR's head, in 5 min 29 s. Both jobs passed:
   `check` and `eval-retrieval`.
   - **The commits:** five, one per area. The state after each of the first three was
     checked before committing, in a scratch copy of `HEAD`: ruff and mypy clean, and 834,
     858 and 866 passed. `test_repo_hygiene.py` was left out there, since that copy is not
     a git repository. The fourth state is the full tree.
   - **The PR's title** first read "fixpolish the command-line tools". The recipe put
     `fix!: …` in double quotes, and zsh expanded `!:` as history. It was fixed with
     `gh pr edit` in single quotes. The commits used quoted heredocs and were intact.
   - **After the reviews:** CI was green on `4c73f70` (run 38341913718) and on `573e733`,
     the PR's last head, with the second review's fix (run 38343630988). It was green on
     `main` after the merge, `5961de8` (run 38344128852).

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

**Second review (2026-10-13, Opus 5.5): no blockers; 1 code finding, fixed on the branch,
test first.** It covered the contract change, the start-up guard, the prompt under
readline in a real pseudo-terminal, and the dataset guard.

- **Checked and sound:**
  - **The gates, with the fix:** ruff and mypy clean; **888 passed, 5 xfailed**, coverage
    98.37%; `rag-eval check` exit 0.
  - **The prompt under readline (libedit), in a pseudo-terminal:** the real `repl()`,
    with a stub retriever and a fake model that waits 5 s before its first token. Four
    endings each left the terminal's ECHO, ICANON and ISIG on, with exit 0 and no
    traceback:
    - Ctrl-C at an empty prompt;
    - Ctrl-C after typing;
    - Ctrl-D;
    - Ctrl-C during the prefill. That one cancelled the answer and came back to a working
      prompt, where Up and then `exit` worked.

    Neither the suite (no terminal) nor Verification 3 (arrows, history, Ctrl-C while
    loading) had covered these.
  - **The contract change:** `86788b5` carries `!` and a `BREAKING-CHANGE:` trailer that
    git parses. `rag-ingest` and an API job both map `CorpusUnreadableError` to exit 2.
    `_check_corpus` also turns a `PermissionError` from `is_dir()` itself into it.
  - **The dataset guard compares resolved paths on both sides.** `paths.data` is resolved
    at load, symlinks included, and the `RAG_DATA_PATH` fallback too. A corpus reached
    through a symlink cannot slip past it.
  - **`preflight` moved verbatim,** and the architecture test pins the import rule.
  - **The PR's title** is fixed: `fix!: polish the command-line tools`.
- **Finding (code, fixed): `rag-eval` printed a traceback for an unreadable corpus root.**
  - **The cause:** S2-8 made the walk raise `CorpusUnreadableError` for an unreadable
    root. Tier 2's `corpus_scan` caught only `FileNotFoundError`, so the new error escaped
    `rag-eval check` and `generate` to their crash handler.
  - **Measured:** on `main`, from a `git archive` export, that corpus gave one line, "Error:
    rag-ingest cannot read part of the corpus either… ./: PermissionError", and exit 2. On
    the branch it gave a full traceback, still exit 2. So CI's reading did not change, but
    this is the defect S2-8 exists to remove.
  - **The fix:** `corpus_scan` catches both, as `FingerprintError`.
    `test_an_unreadable_corpus_root_has_no_fingerprint_and_no_traceback` failed on the
    branch and passes. The real `rag-eval check` now prints "Error: Corpus directory not
    readable: … (Permission denied; …)", exit 2, with no traceback.
  - The corpus part's hash is unchanged, and code is not hashed, so no tier-2 re-run is
    needed (caveat 24).
- **Not in this PR:** the untracked `.claude/` folder (Claude Code's settings). If
  `settings.json` is ever committed, `settings.local.json` belongs in `.gitignore`.

## Deviation from plan

Ran on Opus 5.5, where the story says opus-fast.

**One miss, found in the second review.** `CorpusUnreadableError` was added to the
could-not-start lists of `rag-ingest` and of the API's jobs, but not to the third caller of
the corpus walk, tier 2's `corpus_scan`. So `rag-eval` printed a traceback for an
unreadable root, still with exit 2. It was fixed on the PR (`a5f75b9`). The lesson: a new
exception from a shared function means checking every caller of that function, not only
the ones the story names.

Scope calls made during implementation, for review:

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
