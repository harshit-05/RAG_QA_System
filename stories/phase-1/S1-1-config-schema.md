# S1-1: Validate the config with a frozen Pydantic model

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-8, ISS-01, ISS-11, NFR-11 (re-verified) |
| **Depends on** | — (first Phase 1 story; ARCHITECTURE.md §1.1 DEC-6, §1.4) |
| **Model** | fable |
| **Plan-first** | yes |

## Goal

`load_config()` stops returning a raw dict and returns a frozen `RagConfig`
instead, validated in full before any component is built. A typo in a component
key, an unresolvable pipeline reference, an absolute path, a missing field or an
empty file each fail at load with a message naming the file, the location and
what to do — instead of a `KeyError`, a `pathlib` `TypeError` or an
`AttributeError` on `NoneType` somewhere downstream. This is FR-8, and it closes
ISS-01 and ISS-11 structurally: with typed component kinds, `llmS` cannot be
written and `vector_stores` cannot come back.

## Scope

- **`schema.py`** (new): `ComponentSpec`, `Components`, `Pipeline` (with
  `Ingestion`, `Query`, `Prompt`), `Paths`, `RagConfig` — shapes exactly as in
  ARCHITECTURE.md §1.4.
  - `ComponentSpec`: `model_config = ConfigDict(extra="allow", frozen=True,
    populate_by_name=True)` and `target: str = Field(alias="_target_")`.
    **A field literally named `_target_` is silently dropped** (Pydantic treats
    leading underscores as private attributes — verified 2026-09-25:
    `model_dump()` returns `{}`). `spec()` returns `model_dump(by_alias=True)`
    so `build_object` still sees a `_target_` key.
  - `RetrieverSpec`: **retrievers are not buildable components.**
    `components.retrievers.vector_search` has no `_target_` — it is
    `search_kwargs: {k: 5}`, passed straight to `VectorStore.as_retriever()`.
    Typing it as `ComponentSpec` makes the real config fail to load (verified
    2026-09-25: `_target_` "Field required"). `RetrieverSpec` is
    `search_kwargs: dict[str, Any] = {}` plus `search_type: str | None`,
    `extra="forbid"`, and `kwargs()` returns a fresh dict for `as_retriever`.
  - `Components`: typed kinds, `extra="forbid"`; `retrievers:
    dict[str, RetrieverSpec]`, every other kind `dict[str, ComponentSpec]`.
    `rerankers` defaults to `{}`.
  - `RagConfig`: frozen; a `model_validator(mode="after")` resolves every dotted
    reference in `pipeline`; `component(ref)` returns the `ComponentSpec` (or
    `RetrieverSpec`).
  - **What "frozen" guarantees, precisely.** `frozen=True` blocks attribute
    assignment only; dicts inside the model (`components.llms`, a leaf's
    `model_kwargs`) stay mutable (verified). The structural guarantee is that
    call sites never touch those: they go through `component(ref).spec()` /
    `.kwargs()`, which return fresh copies, so a caller mutating what it got
    back cannot reach the config. Test exactly that, not a deeper claim.
- **`settings.py`** (new): pydantic-settings layer for `RAG_CONFIG`,
  `RAG_DATA_PATH`, `RAG_VECTOR_STORE_PATH`. **The S0-4 anchoring semantics are
  an acceptance criterion, not an implementation detail**: a relative path in the
  config file anchors to the config file's directory; a relative path in an env
  var anchors to cwd.
- **`config.py`**: `load_config(path) -> RagConfig`; catch `ValidationError` and
  re-raise `ConfigError` rendering the config path plus one `loc`-path + message
  line per error. Never surface a raw Pydantic dump.
  - Unresolvable references name the nearest key with
    `difflib.get_close_matches`, so the ISS-01 case reads
    `... 'components' has no key 'llms' (did you mean 'llmS'?)`.
- **Call sites** move to attribute access: `chain.py` and `ingest.py` (`cli.py`
  and `evaluate.py` only pass the config through). `registry.resolve_ref` is
  **deleted**; `RagConfig.component()` replaces it.
- **`vectorstore.py`**: narrow both functions to take the path itself —
  `create_store(chunks, embeddings, path)` / `open_store(embeddings, path)` —
  so the Phase 3 store swap has a smaller contract (ADR-013's noted weakness).
- **Absolute paths (NFR-11)** are rejected only for values **written in the
  config file**, checked on the raw value *before* resolution. Env overrides
  (`RAG_DATA_PATH=/abs/...`) stay allowed — later stories' verification relies
  on them — and the stored `paths.*` are absolute after resolution by design.
- **Version**: `pyproject.toml` and `src/rag_qa/__init__.py` → `0.2.0.dev0`
  (CLAUDE.md: `.dev0` between tags; both files hardcode the version).
- Tests for this story: `tests/test_schema.py`, `tests/test_config.py` —
  valid config loads; frozen model raises on assignment; a mutated `spec()`
  result does not change the config; the five malformed configs in the Goal
  fail with `ConfigError`; both path-anchoring rules hold; an absolute env
  override is accepted.

## Out of scope

- The `_target_` allowlist (S1-2) — the schema string-checks nothing about
  prefixes yet.
- The loader/extension reshape (S1-3): keep `components.loaders.*.extensions`
  working as it is today, so `ingest.py` is untouched apart from attribute
  access. `ComponentSpec`'s `extra="allow"` carries `extensions` unchanged.
- Error handling and exit codes (S1-4). Annotations and mypy (S1-6) — annotate
  what you write here, don't sweep the tree.

## Verification

```bash
# 1. the real config still loads, and is frozen
uv run python -c "
from rag_qa.config import load_config
c = load_config()
print(type(c).__name__, c.pipeline.query.llm, c.paths.data)
print(c.component('components.llms.mistral_ollama').spec())
try: c.paths.data = '/tmp'
except Exception as e: print('frozen:', type(e).__name__)
"

# 2. every malformed config fails with an actionable ConfigError (not a traceback
#    from pathlib/yaml). Show the message text for each.
uv run pytest tests/test_config.py tests/test_schema.py -q

# 3. path anchoring unchanged from S0-4
cd /tmp && RAG_CONFIG=<repo>/config.yaml uv run --project <repo> python -c "
from rag_qa.config import load_config; print(load_config().paths.data)"   # → <repo>/corpus
RAG_DATA_PATH=./mydocs uv run python -c "
from rag_qa.config import load_config; print(load_config().paths.data)"   # → <cwd>/mydocs

# 4. end to end still works on the real corpus (index rebuilt first — the
#    pre-flight stale-index trap)
uv run rag-ingest && echo "What is this corpus about?" | uv run rag-query
```

## Review notes for the human

The `_target_` alias is the one thing that silently breaks everything if it
regresses: check `ComponentSpec.spec()` round-trips a `_target_` key, and that
`build_object` is still handed that key, not `target`. Second, read the
`ConfigError` messages as a stranger would — FR-8 says "specific and
actionable", so "1 validation error for RagConfig" is a fail. Third, check the
retriever path: `chain.py` must pass `component(...).kwargs()` to
`as_retriever`, not a `ComponentSpec` dump with a `_target_` in it.
(`evaluate.py` needs no change — it only calls `build_rag_chain(load_config())`,
whose signature is unchanged — but `grep -n config src/rag_qa/evaluate.py`
should confirm no dict indexing crept in.)

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
