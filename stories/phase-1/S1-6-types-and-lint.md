# S1-6: Type annotations, ruff and mypy configuration

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | ISS-19, NFR-9 |
| **Depends on** | S1-5 |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

Every public function in `rag_qa` carries type hints and the tree passes
`ruff check` and `mypy` with zero errors, both blocking in CI (NFR-9). Typed
config from S1-1 is what makes this cheap: `config.pipeline.query.llm` has a type
where `config["pipeline"]["query"]["llm"]` had `Any`.

## Scope

- Annotate every function in `src/rag_qa/` — parameters and returns. LangChain
  return types come from `langchain_core` (`Embeddings`, `BaseChatModel`,
  `Runnable`, `VectorStore`, `Document`), not from `langchain_community`.
- **`[tool.ruff]`** in `pyproject.toml`: `line-length = 100` (the existing code's
  actual shape), and an explicit lint rule set rather than the default — at
  minimum `E`, `F`, `I` (import order), `B`, `UP`. Record any per-file ignores
  with the reason inline.
- **`[tool.mypy]`**: `disallow_untyped_defs = true` scoped to `src/rag_qa`,
  `ignore_missing_imports = true` for third-party packages without stubs.
  `evaluate.py` may be excluded — its imports live in the optional `eval` extra.
- Add both to CI, ordered cheapest-first: `ruff check` → `mypy` → `pytest`.
- Fix what they find; if a fix is behavioural rather than cosmetic, it is a
  Discovered note, not a silent change.

## Out of scope

- `ruff format` / `ruff format --check` (**DEC-11**): the tree is not formatted,
  and a whole-tree reformat is exactly the diff that gets rubber-stamped in a
  review-the-diff workflow. Backlog.
- `tests/` under `disallow_untyped_defs` — lint them, don't force annotations on
  fixtures.
- Strict mode (`--strict`), `Any` elimination, generics on the registry.
  `build_object` is `Any`-in/`Any`-out by nature (ADR-010).

## Verification

```bash
uv run ruff check src tests          # → All checks passed
uv run mypy src/rag_qa               # → Success: no issues found in N source files
uv run pytest -q                     # unchanged, still green
uv run rag-ingest && echo "What is this corpus about?" | uv run rag-query   # behaviour unchanged
gh run list --limit 1                # CI green with both gates blocking
```

## Review notes for the human

Annotation passes are where behaviour changes sneak in disguised as type fixes —
scan the diff for anything that is not purely a signature or an import. Check
the ruff ignore list: each entry should carry a reason, and `BLE001` should now
be *gone* rather than ignored, since S1-4 replaced the bare
`except Exception` it flagged.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
