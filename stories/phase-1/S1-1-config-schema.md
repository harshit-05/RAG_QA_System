# S1-1: Validate the config with a frozen Pydantic model

| | |
| --- | --- |
| **Status** | Done (2026-09-26) — `801ddea` |
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

### Results (2026-09-26)

| Check | Result |
| --- | --- |
| 1. real config loads, typed, frozen | pass: `RagConfig`, `paths.data` absolute under the repo, `spec()` carries `_target_`, `retriever().kwargs()` is `{'search_kwargs': {'k': 5}}`, assignment → `ValidationError` |
| 2. `pytest tests/test_config.py tests/test_schema.py` | pass: **42 passed in 0.26 s**, hermetic (no LangChain import, no network) |
| 3a. `RAG_CONFIG` from a foreign cwd | pass: `→ <repo>/corpus` |
| 3b. `RAG_DATA_PATH=./mydocs` from a foreign cwd | pass: `→ <cwd>/mydocs` |
| 4a. `rag-ingest`, real config | pass: 561 pages → **1,708 chunks**, 0 failed, 1 m 54 s — identical to the S0-6 baseline |
| 4b. `rag-query`, real config, **mistral** | pass: answered with 5 numbered sources, 7 m wall time on CPU |
| 4c. `rag-query` via a scratch config, **phi3** | pass: same 5 sources (retrieval is LLM-independent); run first while free RAM was 4.7 GB, below mistral's ~6 GB |
| `grep 'config\[\|resolve_ref' src/rag_qa/` | only `loader_config["_target_"]` in `ingest.py` (the loader's own spec dict, replaced in S1-3) and a docstring mention |
| ruff, S1-1 files + tests | All checks passed |
| ruff, whole tree | 4 errors, the pre-existing set S1-2 plans to suppress (none added) |
| ADR-007 grep / ADR-009 no LangChain in `schema`, `settings`, `config`, `registry` | 0 / clean |

The `ConfigError` messages, as a stranger reads them (the malformed cases from
`tests/test_config.py`; each is headed `Invalid config: <path>`):

```text
components: unknown key 'llmS' (did you mean 'llms'?). Expected one of: loaders, splitters, embedders, llms, retrievers, rerankers
pipeline.query.llm: 'components.llms' has no entry 'mistral_olama' (did you mean 'mistral_ollama'?)
pipeline.query.retriever: 'components.llms.mistral_ollama' points into components.llms, but this slot needs an entry of components.retrievers
pipeline.query.llm: 'mistral' is not a component reference; expected 'components.llms.<name>'
paths: 'data' is an absolute path ('/home/someone-else/corpus') in the config file. Use a path relative to the config file's directory (NFR-11), or set RAG_DATA_PATH for a machine-specific location
paths: 'vector_store' is an absolute path ('~/indexes/faiss') in the config file. Use a path relative to the config file's directory (NFR-11), or set RAG_VECTOR_STORE_PATH for a machine-specific location
paths: 'data' is empty. Give a directory relative to the config file, or set RAG_DATA_PATH
components: unknown key 'vector_stores'. Expected one of: loaders, splitters, embedders, llms, retrievers, rerankers
unknown key 'vector_store_path'. Expected one of: components, pipeline, paths
pipeline.query.prompt: required key is missing
components.llms.mistral_ollama._target_: required key is missing
components.retrievers.vector_search: unknown key 'serch_kwargs' (did you mean 'search_kwargs'?). Expected one of: search_type, search_kwargs
```

File-level failures:

```text
Config file <scratch>/empty.yaml is empty. It needs 'components' and 'pipeline' sections (and optionally 'paths'); the repo's config.yaml shows the shape.
Config file <scratch>/list.yaml must be a mapping with 'components' and 'pipeline' sections, got a list.
Config file not found: <scratch>/missing.yaml. Pass a path, or set $RAG_CONFIG, or run from the repo root.
```

Invalid YAML carries yaml's own `line N, column M` marks. The literal ISS-01 bug
(`sed 's/^  llms:/  llmS:/'` on the real `config.yaml`) produces exactly one
line: the first one above.

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

- **Ruff 0.16's default rule set is broad: 788 rules**, not the old
  `E4/E7/E9/F`. There is no ruff config anywhere (repo, parents, `~/.config`),
  so `BLE001`, `TRY004` and `PLR1722` all come from the defaults. CI will match
  because 0.16.8 is locked. **This matters for S1-6**: its scope says to pick an
  explicit rule set "rather than the default — at minimum `E`, `F`, `I`, `B`,
  `UP`". That set is now *narrower* than the default, so adopting it would
  quietly weaken the gate. S1-6 should start from the default and
  `extend-select`/`ignore` from there.
- **`TRY004` false positive in Pydantic validators**, suppressed with the reason
  inline in `schema.py`. Pydantic converts only `ValueError`/`AssertionError`
  into a `ValidationError`; the `TypeError` ruff suggests would escape as a raw
  traceback and bypass `ConfigError`.
- **Behaviour change:** `~/…` in the config file is now rejected as absolute
  (it expands to one user's home). The old loader accepted it. That is correct
  under NFR-11, and env overrides still take `~`.
- **FR-8 follow-up candidate:** the prompt template is not checked for the
  `{context}` and `{question}` placeholders. A `human` prompt missing
  `{context}` would answer with no retrieved context at all — silently. It is a
  one-validator change, but outside this story's scope → Backlog.
- **The piped-stdin `EOFError` is still there.** `echo … | rag-query` ends in a
  traceback once stdin closes. It is pre-existing and not a regression; S1-4
  already owns it.
- **Ollama keeps the last model resident for ~5 min** (`keep_alive`). phi3 held
  3.7 GB after its run, so `free -h` showed 3.3 GB available rather than 7.0 GB.
  Run `ollama ps` / `ollama stop <model>` before pre-flight caveat 8's memory
  check → S1-8 doc hygiene (STATUS pre-flight caveats).
- **`gemma2:9b` is now pulled on this host.** CLAUDE.md's environment facts list
  only phi3, codellama and mistral → S1-8 doc hygiene.
- **Answer quality (Phase 2 evidence, not an S1-1 issue):** mistral's corpus
  summary numbered its contexts inconsistently — it called p. 477 the "fifth
  document" when it is `[3]` — and presented arXiv bibliography entries found
  inside chunk `[1]` as if they were corpus documents. Retrieval was identical
  across phi3 and mistral, so this is the known faithfulness gap from S0-6.

## Deviation from plan

- **Two typed accessors.** `RagConfig.component()` returns only buildable
  `ComponentSpec`s, and a new `RagConfig.retriever()` returns `RetrieverSpec`s.
  The story had one `component()` returning either type; a union return would
  have to be narrowed in `chain.py` and would be flagged by S1-6's mypy gate.
  The review note's intent is unchanged: `as_retriever` receives `kwargs()`,
  never a `_target_` dump.
- **Where ISS-01 is caught.** With typed component kinds, `llmS` fails one level
  earlier than the Scope text assumed: as an unknown key under `components`,
  with "did you mean 'llms'". That fires before reference resolution runs.
  Reference-level did-you-mean still covers entry-name typos.
- **Addition: slot-kind check.** A pipeline reference must point into the kind
  its slot needs (`query.llm` → `components.llms`, and so on). Otherwise a
  retriever pointed at an LLM would fail later, inside `build_object`, with a
  confusing error.
