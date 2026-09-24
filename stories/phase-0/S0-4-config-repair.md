# S0-4: Config repair — keys, paths, corpus rename, current _target_ paths

| | |
| --- | --- |
| **Status** | Done (2026-09-25) — commit pending, maintainer commits manually |
| **Closes** | ISS-01, ISS-02, ISS-11, DEC-4 (rename half) |
| **Depends on** | S0-3 |
| **Model** | fable |
| **Plan-first** | no |

## Goal

`config.yaml` becomes internally consistent, portable, and truthful: every
referenced key exists, every `_target_` names a class that exists in the
locked dependency versions, and no path assumes a particular home directory.

## Scope

- Rename `llmS:` → `llms:`; `reranker:` → `rerankers:` (matching the
  pipeline references and the plural convention of sibling sections).

- **Rename the corpus directory `docs/` → `corpus/` (DEC-4).** `git mv docs
  corpus` (three PDFs tracked, two parquet files ignored but on disk; all
  move together). Update `.gitignore`'s `docs/*.parquet` → `corpus/*.parquet`
  and the comment on `scripts/fetch_dataset.py`'s download target. Do **not**
  touch the CLAUDE.md hard rule or move any project doc here — that is S0-7.
  After this story `docs/` does not exist; S0-7 recreates it for documentation.

- Replace absolute paths with a `paths:` block (ARCHITECTURE.md §0.3):
  `paths: {data: corpus, vector_store: vectorstore/db_faiss}`, resolved
  against the config file's own directory in `config.py` (`load_config`
  returns them absolute), overridable via `RAG_DATA_PATH` /
  `RAG_VECTOR_STORE_PATH`. `RAG_CONFIG` selects the config file itself
  (default: `config.yaml` at repo root). Update `ingest.py`, `chain.py`,
  `vectorstore.py` call sites from `config["data_path"]` /
  `config["vector_store_path"]` to the new keys.

- Delete the dead `components.vector_stores` block (ISS-11) and the
  `pipeline.ingestion.vector_store` reference to it — single source of
  truth is `paths.vector_store`.

- Update every `_target_` to its import location in the locked LangChain
  version (verified 2026-09-18, re-verify with the story's script):
  - splitter → `langchain_text_splitters.RecursiveCharacterTextSplitter`
  - LLM → `langchain_ollama.ChatOllama`
  - embedder unchanged → `langchain_huggingface.HuggingFaceEmbeddings`
  - reranker (commented) → `langchain_classic.retrievers.ContextualCompressionRetriever`
    with `base_compressor: {_target_: langchain_classic.retrievers.document_compressors.CrossEncoderReranker,
    model: {_target_: langchain_community.cross_encoders.HuggingFaceCrossEncoder,
    model_name: cross-encoder/ms-marco-MiniLM-L-6-v2}, top_n: 3}`.

- Fix the reranker component per SRS ISS-03 _shape_ as above but leave it
  commented-out/disabled — its first live run is a Phase 2 story, which may
  replace the classic wrapper with a hand-rolled Runnable.

- DEC-2 settings (ARCHITECTURE.md §0.1): add `components.llms.mistral_ollama`
  (`model: mistral`) and keep `qwen2_ollama`; both get `temperature: 0`,
  `num_ctx: 4096`, `num_predict: 512`, `validate_model_on_init: true`.
  `pipeline.query.llm` → `components.llms.mistral_ollama`. Retriever
  `search_kwargs.k` 10 → 5.

- Loaders: drop `.doc` from the docx entry (docx2txt cannot read legacy
  `.doc`). Leave the `extensions` key as is (its removal is a Phase 1
  backlog item).

- Delete remaining commented-out config carcasses (duplicate `#pipeline:`
  block etc.).

- Host has **no NVIDIA GPU**: the selected embedder must stay
  `minilm_cpu`; delete the `device: cuda` component entries
  (`minilm_gpu`, `multilingual_mpnet`) or clearly comment them as
  non-functional-on-this-host examples.

## Out of scope

- Schema validation via Pydantic (Phase 1 — FR-8).
- Enabling the reranker at runtime (Phase 2 — FR-4).

## Verification

```bash
uv run python - <<'EOF'
import yaml
from importlib import import_module
cfg = yaml.safe_load(open("config.yaml"))
# every pipeline reference resolves
def walk(d, path=""):
    if isinstance(d, dict):
        if "_target_" in d:
            mod, cls = d["_target_"].rsplit(".", 1)
            getattr(import_module(mod), cls)   # raises if wrong
            print("ok:", d["_target_"])
        for v in d.values(): walk(v)
walk(cfg["components"])
# key references used by the query pipeline exist
q = cfg["pipeline"]["query"]
for ref in [q["llm"], q["retriever"], cfg["pipeline"]["ingestion"]["embedder"], cfg["pipeline"]["ingestion"]["splitter"]]:
    node = cfg
    for k in ref.split("."): node = node[k]
    print("resolves:", ref)
assert q["llm"].endswith("mistral_ollama"), q["llm"]
llm = cfg["components"]["llms"]["mistral_ollama"]
assert (llm["temperature"], llm["num_ctx"], llm["validate_model_on_init"]) == (0, 4096, True), llm
assert cfg["components"]["retrievers"]["vector_search"]["search_kwargs"]["k"] == 5
assert set(cfg["paths"]) == {"data", "vector_store"}
EOF
uv run python -c "from rag_qa.config import load_config; c = load_config('config.yaml'); print(c['paths'])"   # absolute paths under this checkout
RAG_DATA_PATH=/tmp/x uv run python -c "from rag_qa.config import load_config; print(load_config('config.yaml')['paths']['data'])"   # → /tmp/x
grep -c "/home/" config.yaml              # → 0
grep -c "cuda" config.yaml                # → 0
grep -c '".doc"' config.yaml              # → 0
git ls-files corpus/                      # → the three .pdf files (rename detected)
ls docs 2>&1                              # → No such file or directory (S0-7 recreates it)
git check-ignore -v corpus/0000.parquet   # → matched by corpus/*.parquet
grep -rn "docs/" config.yaml .gitignore | wc -l   # → 0
```

## Review notes for the human

The `_target_` list IS the LangChain 0.2→1.x migration surface for config.
Check each printed `ok:` line — that's the proof each class was actually
imported from the locked versions, not assumed.

## Verification results (2026-09-25)

All 10 `_target_`s import cleanly against the locked versions — this is the
proof for ISS-03, whose `CrossEncoderRerank` never existed under that name:

```text
ok: langchain_community.document_loaders.{PyPDFLoader,Docx2txtLoader,TextLoader}
ok: langchain_text_splitters.RecursiveCharacterTextSplitter
ok: langchain_huggingface.HuggingFaceEmbeddings
ok: langchain_ollama.ChatOllama                       (mistral + qwen2 entries)
ok: langchain_classic.retrievers.ContextualCompressionRetriever
ok: langchain_classic.retrievers.document_compressors.CrossEncoderReranker
ok: langchain_community.cross_encoders.HuggingFaceCrossEncoder
```

Paths resolve absolute against the config file's directory, honour
`RAG_DATA_PATH` / `RAG_VECTOR_STORE_PATH`, and — the actual ISS-02 test —
resolve identically when run from `/tmp` via `RAG_CONFIG`. A missing config
now raises a `FileNotFoundError` naming the path instead of a `KeyError`
cascade. `grep` checks for `/home/`, `cuda`, `".doc"`, `docs/`,
`vector_stores` and `^#pipeline:` all return 0.

## Discovered

- **The reranker is defined but not wired.** Leaving the component in
  `components.rerankers` (rather than commenting the whole block out) is what
  lets the verification script import-check its class names; it stays inert
  because `pipeline.query.reranker` is the line that is commented. Disabled
  means "unreferenced by the pipeline", not "absent from the library".
- **`langchain-classic` is used by that block but is only a transitive
  dependency** (it arrives via `langchain-community`). Nothing imports it at
  runtime while the reranker is off, so DEC-1 rule 1 holds today. If Phase 2
  keeps this shape rather than the hand-rolled Runnable, it must become an
  explicit dependency in `pyproject.toml` — added to Backlog.
- **The story's own `grep -c "cuda"` check is prose-sensitive.** A comment
  explaining _why_ the GPU embedders were deleted tripped it. Reworded the
  comment so the check stays a valid signal rather than weakening the check.
- `rag-ingest` is now genuinely runnable: its imports resolve, the corpus path
  points at a real directory with 3 PDFs, and the old `FileNotFoundError` is
  gone. Deliberately **not** executed here — it downloads the ~90 MB embedder
  and does minutes of CPU embedding, which is S0-6's job (and `ingest.main()`
  takes no arguments, so even `rag-ingest --help` would start a real run).
- `rag-query` still fails at `ModuleNotFoundError: langchain`; that is S0-5.
- ruff remains at exactly one finding, `BLE001` at `ingest.py:38` (ISS-05).

### Review pass (2026-09-25) — one real bug, fixed before commit

**`$RAG_CONFIG` was dead through both entry points.** `load_config` had the
precedence right (explicit arg → `$RAG_CONFIG` → default), but `ingest.main`
and `build_rag_chain` both defaulted to the literal `"config.yaml"`, which
counts as an explicit argument and silently won. So the story's original claim
that paths "resolve identically from /tmp via RAG_CONFIG" held only for direct
`load_config()` calls, not for `rag-ingest` or `rag-query`. Both defaults are
now `config_path=None`, letting `load_config` own the precedence, and
`chain.py` carries a comment saying why the default must stay `None`.

The S0-3 review had predicted this exact failure from the duplicated default.
Lesson: a default value that duplicates a lower layer's fallback is not a
convenience, it is an override that shadows the layer below.

**Verification added**, since the old checks could not have caught it: a run
through `ingest.main` (not `load_config`) from a different working directory
with `$RAG_CONFIG` set, stubbing `load_documents` to stop before any real work.

**Environment-variable relative paths now anchor to the caller's cwd**, while
config-file relative paths keep anchoring to the config file's directory. That
is the conventional split: `RAG_DATA_PATH=./mydocs` from `/tmp` resolves to
`/tmp/mydocs`, which is what a shell user typing it expects, whereas
`paths.data: corpus` in the file resolves next to the file so a clone runs
anywhere. Both behaviours are verified and documented in `_resolve_path`.

**Scope item that did not apply:** the story asked to update the comment on
`scripts/fetch_dataset.py`'s download target, but that script has no `docs/`
reference — it writes `train.parquet` into whatever directory it is run from.
Nothing to update; its real problems are ISS-18 in Phase 1. The `.gitignore`
comment that claimed those parquet files were "fetched by
scripts/fetch_dataset.py" was inaccurate and was reworded.

**Deferred to Phase 1 (FR-8), now in Backlog:** malformed configs still fail
with raw internal errors rather than actionable ones — `paths: {data: }` raises
a `pathlib` `TypeError` and an empty YAML file raises `AttributeError` on
`NoneType`. Config _validation_ is Phase 1's job; S0-4 only had to make a
_missing_ file clear, which it does.

## Deviation from plan

The corpus rename used a plain `mv`, not `git mv`, so that Claude stages
nothing (see the collision rule in STATUS.md "Now"). Rename detection is
computed from content at commit time, so `git log --follow` is unaffected.
