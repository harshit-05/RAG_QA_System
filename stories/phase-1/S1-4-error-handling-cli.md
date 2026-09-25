# S1-4: Explicit error handling and a real CLI surface

| | |
| --- | --- |
| **Status** | Todo |
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
RAG_DATA_PATH=<scratch>/corpus uv run rag-ingest; echo "exit=$?"   # exit=1, others still ingested

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

## Review notes for the human

The exit-code change is the one with teeth: confirm a *partial* failure exits
non-zero, because that is the difference between NFR-7 met and a corpus quietly
missing a document. In the REPL, check the except clause does not swallow
`KeyboardInterrupt`/`SystemExit` into "continue" — an un-quittable REPL is a
worse bug than the one being fixed. `scripts/fetch_dataset.py` is not covered by
CI and stays a script: keep the change minimal.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
