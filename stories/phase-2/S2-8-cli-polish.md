# S2-8: Polish the command-line tools

| | |
| --- | --- |
| **Status** | Todo |
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
