# S0-3: Collapse v1/v2/temp into a single package

| | |
| --- | --- |
| **Status** | Done (2026-09-19) — commit pending, maintainer commits manually |
| **Closes** | ISS-10, ISS-12 |
| **Depends on** | S0-2 |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

One source tree: `src/rag_qa/` (ARCHITECTURE.md §0.2), built from v2 as
the basis, with v1 retired. Scripts become entry points runnable from repo
root — the "must cd into v2/" trap is gone — and the module boundaries match
the architecture so Phase 1 tests are written against the right seams.

## Scope

- Move v2 modules into `src/rag_qa/` (the `__init__.py` already exists from
  S0-2), splitting along ARCHITECTURE.md §0.2:
  - `pipeline_builder.py` lines 11–41 (`import_from_string`, `build_object`,
    `get_component_from_path` → renamed `resolve_ref`) → `registry.py`.
    Pure Python, **no LangChain imports**.
  - `pipeline_builder.py` `build_rag_chain` → `chain.py`.
  - The FAISS calls (`FAISS.from_documents` + `save_local` from
    `file_processor.py`, `FAISS.load_local` from `pipeline_builder.py`) →
    `vectorstore.py` as `create_store(chunks, embeddings, cfg)` and
    `open_store(embeddings, cfg)`. Same calls, same args, just relocated.
  - The YAML `open`/`safe_load` (duplicated in both modules) → `config.py`
    `load_config(path) -> dict`. Path resolution and env overrides are S0-4.
  - `file_processor.py` → `ingest.py`; `main2.py` → `cli.py`;
    `evaluate.py` → `evaluate.py` with its module-level body wrapped in
    `def main():` so importing it no longer runs an evaluation.
  - `config.yaml` moves to repo root.
- Dependency direction after the move: `cli → chain → vectorstore/registry/config`;
  `ingest → vectorstore/registry/config`. **`ingest.py` must not import `chain.py`.**

- Delete on the way in: the ~45-line unreachable string literal in
  `pipeline_builder.py` (lines 72–117), commented-out alternative code
  blocks, `check_config.py` (its job is replaced by validation in Phase 1;
  note this in Discovered if it feels premature).

- Delete `v1/` entirely (git history preserves it) including
  `requirements.txt` if still present.

- Wire `[project.scripts]` in pyproject: `rag-ingest`, `rag-query`
  (thin wrappers calling the moved modules' mains).

- Fix intra-package imports (`from rag_qa.registry import build_object`, etc.).
- No behavior changes beyond imports/paths/module split — the code still
  targets old LangChain APIs and `import rag_qa.chain` is **expected** to
  fail with `ModuleNotFoundError: langchain` until S0-5. `rag_qa.registry`
  and `rag_qa.config` must import cleanly (they have no LangChain imports).

## Out of scope

- Config content fixes (S0-4). API changes (S0-5). Type hints, error
  handling, REPL polish (Phase 1).

## Verification

```bash
ls v1 v2 2>&1                            # both → No such file or directory
ls src/rag_qa/                           # __init__ registry config vectorstore ingest chain cli evaluate
uv run python -c "import rag_qa.registry, rag_qa.config; print('pure modules ok')"   # must pass
uv run python -c "import rag_qa.chain" 2>&1 | tail -1   # expected: ModuleNotFoundError for langchain.* (record it; S0-5 fixes)
grep -n "import" src/rag_qa/ingest.py | grep -c "chain"  # → 0 (ingest does not import chain)
grep -n "langchain" src/rag_qa/registry.py | wc -l       # → 0
grep -rn '"""' src/rag_qa/chain.py | wc -l               # no orphaned mega-string (spot-check)
uv run rag-query --help 2>&1 | head -3   # entry point exists (may fail deeper until S0-5)
git log --oneline -1                     # single commit for the move
```

## Review notes for the human

This is a `git mv`-heavy diff — check the moves are moves (rename detection)
so history follows the files; `pipeline_builder.py` → `chain.py` should be
detected as a rename with `registry.py` split out. Confirm nothing from v1
was silently needed (the only v1-unique logic was the LOADER_MAPPING pattern,
superseded by config-driven loaders). Confirm `ingest.py` no longer imports
`chain.py`.

## Discovered

- **The split is load-bearing already.** `rag-ingest` now resolves its entry
  point and runs all the way to `FileNotFoundError: /home/harshit/RAG_System/docs`,
  i.e. it fails only on ISS-02, while `rag-query` still stops at
  `ModuleNotFoundError: langchain`. That asymmetry is the proof the dependency
  direction is correct: ingestion no longer drags the query stack in with it.
- **ruff finds exactly one issue**, and it is already catalogued: `BLE001`
  blind `except Exception` at `ingest.py:38` — that is ISS-05, Phase 1 error
  handling. Confirmed with `--isolated`, so it comes from ruff's own defaults,
  not an inherited config. No ruff config is committed yet.
- **Editor vs linter disagreement.** The IDE reports `E501` at 79 characters
  while ruff's default is 88, so the same file looks clean or dirty depending
  on which tool you ask. The Phase 1 CI story should commit an explicit
  `[tool.ruff]` block (line-length + rule selection) so local, editor and CI
  agree on one answer (NFR-9).
- `evaluate.py`'s ragas/datasets/pandas imports were moved inside `main()`
  so the module is importable without the optional `eval` extra installed.
- `get_component_from_path` renamed to `resolve_ref` per ARCHITECTURE.md §0.2.
- `build_rag_chain` still takes a `config_path`, not a loaded config dict.
  S0-5 changes the signature (its verification block already assumes the new
  one: `build_rag_chain(load_config('config.yaml'))`).

### Multi-angle review pass (2026-09-24)

Eight review angles run over the S0-3 diff; findings verified against the code
before recording. Nothing was fixed here — all of it is either another story's
scope or pre-existing, per CLAUDE.md's no-detour rule.

| Finding | Disposition |
| --- | --- |
| `chain.py:31` mutates the loaded config: `resolve_ref` returns a live reference and a built retriever is written into it, so config stops being inert on the reranker path | Carried over from v2 unchanged. **S0-5 scope** already forbids it; structural guarantee in Backlog |
| `config.yaml:48` still defines `llmS` while `:95` references `components.llms` — `rag-query` will `KeyError` the first time it runs | ISS-01, **S0-4 scope**. Confirmed still present |
| Embedder resolution duplicated between `chain.py:23` and `ingest.py:65-66` | New → Backlog (Phase 1). They must agree or index and query vectors diverge |
| Loaders bypass the `build_object` funnel that Phase 1's import allowlist attaches to | Already in Backlog |
| `vectorstore.py` functions take the whole config to read one key | New → Backlog |
| `README.md` still documents `v1/`/`v2/`, `pip install`, and image ingestion | S0-6 owns the rewrite; flagged in Backlog as actively wrong meanwhile |
| Git history: `e781b17` fuses the S0-1 fixup with the S0-3 moves | Deliberate, message amended to match; root cause and rule recorded in STATUS.md "Now" |
| STATUS.md pending-commits section stale | Fixed in this close-out |

## Deviation from plan

None in scope. The story's `grep -c '"""' chain.py` spot-check is a weak test
(it counts docstring lines, and a healthy file scores 3); replaced in practice
by a direct grep for the dead block's marker comments, which finds nothing.
