# S1-4: Explicit error handling and a real CLI surface

| | |
| --- | --- |
| **Status** | In review (2026-10-01) — all six verification steps passed (step 3 live, by the maintainer); second review's fixes applied (unreadable folders fail the run; exit-1 contract documented), one decision open (partial-index policy, see Discovered); branch `s1-4-error-handling-cli`, commit + CI + PR pending |
| **Closes** | ISS-05, ISS-06, ISS-18, NFR-7 |
| **Depends on** | S1-3 (ARCHITECTURE.md §1.1 DEC-9) |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

Failures stop being silent or fatal in the wrong direction. A corrupt document no
longer drops out of the corpus unnoticed — the run continues, reports it, and
exits non-zero (NFR-7). One Ollama hiccup no longer kills the REPL mid-session.
And both entry points grow `--help` and `--config`, so `rag-ingest --help` stops
launching a real ingestion run.

## Scope

- **`ingest.py` (ISS-05, NFR-7)**: replace the bare `except Exception` with a
  narrow catch that records `(path, error)` in `IngestReport.failed` and
  continues. `main()` prints the summary and exits non-zero if **any** document
  failed — today it only exits non-zero when *nothing* loaded. Keep the existing
  "skipped, no loader configured" path distinct from "failed".
- **`cli.py` (ISS-06)**: wrap the chain call in try/except — print the error,
  keep the REPL alive. `KeyboardInterrupt` during a stream returns to the prompt;
  a second one exits. `EOFError` exits cleanly (piped stdin).
- **`cli.py` / `ingest.py` argparse**: `--config PATH` (precedence: flag, then
  `$RAG_CONFIG`, then `./config.yaml`, matching `resolve_config_path`) and
  `--help` that exits without doing any work. `rag-ingest --help` must not
  ingest.
- Remove the `# noqa: BLE001  # S1-4` suppressions S1-2 added in `ingest.py`
  and `scripts/fetch_dataset.py` — the fixes below make them unnecessary, and
  `grep -rn "noqa" src scripts` should come back empty of S1-4 tags.
- **`scripts/fetch_dataset.py` (ISS-18)**: `/blob/` → `/resolve/` URL form (the
  `/blob/` URL downloads HTML, not parquet), add `timeout=`, and write outside
  `corpus/` — anything in the corpus directory gets embedded (DEC-4), and a
  parquet file has no loader anyway.
- Tests: `tests/test_ingest.py` — a deliberately corrupt fixture yields a failed
  entry, a non-zero exit, and does **not** abort the other documents;
  `tests/test_cli.py` — `answer()` survives a chain that raises, `--help` exits
  0 without ingesting.

## Out of scope

- Timeout, retry-with-backoff, circuit-breaking (**DEC-9**: NFR-6 is Phase 3,
  with the serving decision). This story adds no retries.
- The "Sources on a refusal is misleading" fix — Phase 2, it needs the eval
  harness to tune a relevance threshold.
- The `ChatOllama` unclosed-socket `ResourceWarning` — Phase 2, the API owns the
  client lifecycle.

## Verification

```bash
# 1. --help does nothing but print
stat -c %Y vectorstore/db_faiss/index.faiss            # before
uv run rag-ingest --help && uv run rag-query --help
stat -c %Y vectorstore/db_faiss/index.faiss            # after → identical

# 2. a corrupt document is reported, does not abort the run, and exits non-zero
cp corpus/*.pdf <scratch>/corpus/ && head -c 200 /dev/urandom > <scratch>/corpus/broken.pdf
#    Set BOTH paths (S1-3 habit): with RAG_DATA_PATH alone the scratch build
#    overwrites the real vectorstore/db_faiss.
RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-ingest; echo "exit=$?"   # exit=1, others still ingested

# 3. the REPL survives a failing chain. Ollama must be UP at startup:
#    validate_model_on_init makes build_rag_chain fail before the REPL exists,
#    which is outside the try/except and proves nothing. Start the REPL, then
#    in another terminal `systemctl stop ollama`, then ask two questions:
uv run rag-query      # Q1 → error printed, prompt returns; Q2 → same; `exit` works
#    (tests/test_cli.py covers the same path hermetically with a raising fake chain)

# 4. --config is honoured over $RAG_CONFIG
uv run rag-ingest --config <scratch>/config.yaml 2>&1 | head -3

# 5. fetch_dataset writes outside corpus/ and times out
uv run python scripts/fetch_dataset.py --help 2>&1 | head -3
git status --short corpus/    # → empty

uv run pytest tests/test_ingest.py tests/test_cli.py -q
```

### Results (2026-10-01)

**Risk call: branch** (`s1-4-error-handling-cli`). This changes `rag-ingest`'s
exit-code contract: a partial failure used to exit 0 and now exits 1. CI jobs,
scripts and Phase 2's job-status endpoint will rely on that contract. The REPL's
interrupt handling is also the kind of code that gets reworked.

**Decisions taken with the maintainer before coding** (2026-09-30):

- **Catch policy: split, not narrow.** A probe found 9 kinds of bad file raising
  6 unrelated exception families. Building a loader (a config problem) aborts
  the run as a `ConfigError`. Reading one document is caught broadly: recorded
  with its exception type, the run continues, and it fails at the end.
- **Two S1-3 leftovers are in scope:** ignored files are counted, and symlinks
  are never followed (they are listed as skipped).

| Check | Result |
| --- | --- |
| 1. `--help` does no work | pass: both print usage with `--config` and the exit-status table; `rag-ingest --help` exits 0; index mtime identical before and after |
| 2. corrupt PDF among the real three | pass: **exit 1**; `broken.pdf: PdfStreamError: Stream has ended unexpectedly` recorded; the other 561 pages / 1,708 chunks still indexed; scratch index saved; **real index untouched** (both paths set). 122 s |
| 3. REPL survives Ollama stopping mid-session | pass, **run live by the maintainer** (2026-10-01): REPL started, `sudo systemctl stop ollama` in a second terminal, two questions each printed `Error: ConnectError: [Errno 111] Connection refused` and returned to the prompt, `exit` → `Exiting...`, **exit status 0** (re-captured with Ollama back up). The live error is httpx's `ConnectError` from under the Ollama client, not a builtin `ConnectionError`, so the hermetic test now raises that exact error |
| 4. `--config` beats `$RAG_CONFIG` | pass: `RAG_CONFIG=/nonexistent` plus `--config <scratch>` exits 0, index built beside the scratch config. Without the flag: one line, `Config file not found: /nonexistent/config.yaml`, **exit 2**, no traceback |
| 5. `fetch_dataset.py` | `--help` exits 0 in the default env (before: crashed importing `datasets`, which is only in the `eval` extra); `--out corpus/sub` is **refused, exit 2**, nothing created; `git status corpus/` empty. Real download to scratch: **6,341,254 bytes** (the exact advertised size), starts and ends with `PAR1`, so a genuine parquet file where `/blob/` gave a 128 KB HTML page. Exit 0, preview skipped with the `eval` extra hint. 10 m on this network (the timeout is per operation, not total) |
| 6. `pytest tests/test_ingest.py tests/test_cli.py` | pass: **24 passed**. Full suite **129 passed**, `HF_HUB_OFFLINE=1` |
| piped empty stdin, real chain | `printf '' \| rag-query` → `Exiting...`, **exit 0, no traceback**. Every transcript since S0-6 used to end in an `EOFError` traceback |
| `grep -rn noqa src scripts` | no S1-4 tags; two `BLE001`s remain at the deliberate broad catches (ingest's per-document read, the REPL boundary), each with its reason in the comment above |
| ruff (whole project) | All checks passed |

## Review notes for the human

The exit-code change is the one with teeth: confirm a *partial* failure exits
non-zero, because that is the difference between NFR-7 met and a corpus quietly
missing a document. In the REPL, check the except clause does not swallow
`KeyboardInterrupt`/`SystemExit` into "continue" — an un-quittable REPL is a
worse bug than the one being fixed. `scripts/fetch_dataset.py` is not covered by
CI and stays a script: keep the change minimal.

## Discovered

- **What bad files actually raise** (probe, 2026-09-30): the same 6 families
  come up across random bytes, an empty file, a truncated or cut PDF, a non-zip
  and a zip-without-document `.docx`, broken XML, Latin-1 text and an unreadable
  file:
  - pypdf `PdfReadError` / `PdfStreamError` / `EmptyFileError`;
  - `zipfile.BadZipFile`;
  - `KeyError`;
  - `xml.etree.ElementTree.ParseError` (a `SyntaxError`);
  - `UnicodeDecodeError`;
  - `PermissionError`.

  This is the evidence behind the split catch policy, and a ready-made case list
  for S1-5's regression suite.
- **ISS-18 was worse than logged.**
  - `/blob/` served a 128 KB HTML page, which the script saved under the
    parquet name; `/resolve/` redirects to the real 6.3 MB parquet (both checked
    with HEAD requests).
  - The script wrote into the *current directory*, not `corpus/` as such; it
    landed in the corpus only because it was once run from there.
  - It could not start at all in the default environment: `datasets` is only in
    the `eval` extra and was imported at the top, so even `--help` crashed.
- **`requests` is an undeclared dependency.** The script imports it; it arrives
  only transitively. It's a script, not package code, so this is backlog:
  declare it in the `eval` extra, or switch to `huggingface_hub.hf_hub_download`,
  which is already installed and handles `/resolve/`, caching and resume.
- **A missing index used to surface as FAISS's raw
  `RuntimeError: … could not open … for reading`.** `rag-query` now checks first
  and says "Run rag-ingest first". The check lives in
  `vectorstore.store_exists`, because what counts as an index is FAISS-specific
  (ADR-013 seam), and the Phase 3 swap rewrites it with the other two functions.
- **Ollama being down at `rag-query` *startup* is still a traceback.**
  `validate_model_on_init` fails inside `build_rag_chain`, before the REPL's
  boundary exists. ISS-06 covers the session, not startup. A startup message is
  a small follow-up → backlog.
- **`requests`' `timeout=` is per network operation, not total.** A slow but
  progressing download can take longer than 60 s, which the real verification
  download showed on this network; only a connection that stalls for 60 s
  fails. That is the intended meaning, and the docstring says "stalled".
- **The corpus still holds two old parquet files** (`corpus/0000.parquet`,
  `corpus/train.parquet`): gitignored relics of the old script, reported as
  skipped on every run. Left alone (data on disk); the maintainer can delete them
  or move them to `downloads/`.
- **S1-4's own verification step 2 had S1-3's index-overwriting bug** (only
  `RAG_DATA_PATH` set). Fixed here before running it.
- **From the live step-3 run, two small usability points** → backlog:
  - `Error: ConnectError: [Errno 111] Connection refused` doesn't say *what*
    refused. A one-line hint when the error is a connection failure ("Is
    Ollama running? `systemctl status ollama`") would make it self-explaining.
  - An empty `Answer:` header prints before the error, because the header goes
    out before the stream starts. Cosmetic.

### Second review (2026-10-01)

- **An unreadable subfolder dropped out of the corpus silently.** Probed with a
  `chmod 000` folder holding a `.txt`: `discover_files` returned only the other
  files, with nothing failed, skipped or counted, and `rag-ingest` exited **0**
  (the new end-to-end test showed `assert 0 == 1` before the fix). Python 3.12's
  `rglob` skips a folder it cannot open without a word. An unreadable *file* was
  already handled (`load()` raises `PermissionError`, the run fails). **Fixed in
  the review follow-up:** discovery walks with `Path.walk(on_error=…)`, and
  `CorpusListing.unreadable` holds each folder with its error. `load_documents`
  records it in `report.failed` as `locked/: PermissionError: Permission
  denied`, so the run exits 1 and the rest is still indexed. An unreadable folder
  inside an ignored path (`.git`, a checkpoint folder) is ignored like the rest.
  Same files, same order on the real corpus. Tests first: 3 new, plus the
  `--help` assertion below; 4 failed before, **132 passed** after, `ruff` clean.
  The `chmod` tests skip when run as root, where they would prove nothing.
- **Exit 1 also meant "crashed".** With a scratch config whose embedder cannot
  load offline, `rag-ingest` printed an `OSError` traceback, exited **1** and
  wrote no index, while `--help` said 1 meant a document could not be read or
  nothing was indexed. 1 is Python's own status for an unhandled exception.
  **Fixed as a documentation change**, the review's recommendation: the module
  docstring and the `--help` epilog now say 1 also covers an unexpected error
  (shown with its traceback), so 1 alone does not say whether an index was
  saved. A test asserts the epilog says so. No distinct code was added; the
  traceback is the useful output for an unexpected error.
- **Decision for the maintainer: a partial failure replaces a good index.**
  `ingest()` saves whenever any document loaded, so one transient read error (a
  `PermissionError`, a file still being copied) replaces a complete index with
  one missing that document. It is loud (exit 1, "rebuilt without them"), and
  deliberate. The alternative is to keep the previous index whenever anything
  failed: safer for a scheduled re-ingest, but one bad file then blocks every
  update. Both are defensible. The Phase 2 job endpoint inherits whichever is
  chosen, so it should be recorded as a decision (a DEC entry or a line in
  ARCHITECTURE §1), not left implicit in a message. **Not changed in code.**
  **Resolved 2026-10-01 → DEC-13: replace, exit 1** (maintainer's call). Phase 1
  ingests are run by hand and watched, so a loud partial rebuild is re-run, not
  missed. Phase 2's hash manifest dissolves the dilemma: a document that fails
  keeps its previously indexed chunks.
- **Third review (2026-10-01), checked and agreed.**
  - The unreadable-folder tests fail against `b9c4e4e` for the right reasons
    (4 failures, incl. the `--help` text) and pass after (132).
  - The fixture restores permissions even when a test fails, and the tests skip
    only as root (GitHub's runner is not root, so they run in CI).
  - The real `corpus/` lists the same 5 files in the same order.
  - New edge case: an unreadable corpus **root** is reported as `./:
    PermissionError`, with exit 1 and nothing indexed. It is loud and correct,
    but `./` is an odd name, and "not readable" would match "not found" better
    with exit 2 ("cannot start") → backlog.

## Deviation from plan

- **Split catch, not a narrow tuple** (maintainer's call, 2026-09-30).
  Constructing a loader is config: a failure aborts as `ConfigError`, exit 2.
  Reading a document is caught broadly, typed, recorded, and fails the run. The
  story's "noqa comes back empty of S1-4 tags" is met, but two `BLE001`
  suppressions remain on purpose, each at a boundary where catching everything
  is the requirement (NFR-7, ISS-06), with the reason written above it.
- **Three exit codes, not just "non-zero"**: 0 ok, 1 run failed (a document
  unreadable, or nothing indexed), 2 cannot start (config, missing corpus or
  index, usage — the code argparse already uses). Shown in `--help`.
- **Two S1-3 leftovers taken in** (maintainer-approved): ignored files are
  counted in the summary, and symlinks are never followed and are listed as
  skipped. `discover_files` now returns a `CorpusListing`; `IngestReport` gained
  `symlinks` and `ignored`.
- **Additions:**
  - the `rag-query` missing-index check (with `vectorstore.store_exists`);
  - `repl()` factored out of `main()` so it can be tested with a fake chain. It
    looks `input` up at call time: bound as a default argument, a patched
    `input` would be ignored, and a test of `main()` would block on real stdin.
- **`fetch_dataset.py` changed more than "minimal".** `--help` had to work
  without downloading or crashing, which meant a `main()` with `argparse` and a
  lazy `datasets` import. It also gained a `corpus/` guard, and `downloads/`
  was added to `.gitignore`.
