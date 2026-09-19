# S0-3: Collapse v1/v2/temp into a single package

| | |
| --- | --- |
| **Status** | Todo |
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

—

## Deviation from plan

—
