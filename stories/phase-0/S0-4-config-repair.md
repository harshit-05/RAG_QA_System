# S0-4: Config repair — keys, paths, corpus rename, current _target_ paths

| | |
| --- | --- |
| **Status** | Todo |
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

## Discovered

—

## Deviation from plan

—
