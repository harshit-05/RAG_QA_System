# S1-3: Own loaders, extension mapping in the pipeline, recursive corpus walk

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | ISS-13, FR-2 (recursive discovery), DEC-5 step 1 of 3 |
| **Depends on** | S1-1 (ARCHITECTURE.md §1.1 DEC-8, §1.3) |
| **Model** | fable |
| **Plan-first** | no |

## Goal

The three `langchain_community` document loaders are replaced by ~40 lines of our
own on `pypdf` and `docx2txt` — both already locked, and both already what the
community loaders wrap. At the same time the extension → loader mapping moves out
of the component entries into `pipeline.ingestion.loaders`, leaving exactly one
instantiation path for every component in the system, and corpus discovery
becomes recursive. After this story `langchain_community` is imported only by
`vectorstore.py` (FAISS) and named only by the disabled reranker config.

## Scope

- **`loaders.py`** (new): `PdfLoader`, `DocxLoader`, `TextLoader`. Each takes
  `file_path` and exposes `load() -> list[Document]` (`Document` from
  `langchain_core.documents` — core-only, DEC-1 rule 1).
  - **Metadata contract**, which citations depend on: `source` (the file path),
    `page` (0-indexed, PDF only), `page_label` (the printed page number, PDF
    only), `loader` (class name). `chain.citation()` must keep working
    unchanged.
- **`config.yaml`**: `components.loaders.*` lose `extensions` and point at
  `rag_qa.loaders.*`; `pipeline.ingestion.loaders` maps `".pdf" | ".docx" |
  ".txt" | ".md"` to component references. `.doc` stays absent (docx2txt cannot
  read the legacy binary format).
- **`schema.py`**: `Ingestion.loaders: dict[str, str]`, references validated like
  every other one.
- **`components.py`** (new): `build_embedder`, `build_splitter`, `build_llm`,
  `build_loader(config, path)`. One instantiation path —
  `build_object({**spec, "file_path": str(path)})` — allowlist-covered by S1-2.
  `build_embedder` always reads `pipeline.ingestion.embedder`: the invariant that
  index and queries share an embedder becomes one function instead of the same
  two lines duplicated in `chain.py` and `ingest.py`.
- **`ingest.py`**: `Path.rglob` instead of `os.listdir` (ISS-13); loader lookup
  by extension through `build_loader`.
- **Baseline check (the risk of this story).** *Before* changing anything,
  record over the real three-PDF corpus: chunk count, and every `citation()`
  string for a fixed query. Reproduce both after. A citation regression here is
  silent and would only surface as degraded answers.
- Tests: `tests/test_loaders.py` over `tests/fixtures/` (a tiny PDF, DOCX, TXT,
  MD) asserting text extraction and the metadata contract; a subdirectory
  fixture proving recursion; `tests/test_components.py` for the single
  instantiation path.

## Out of scope

- Content hash and ingestion timestamp in metadata (SRS §7.3) — Phase 2, where
  the manifest needs them.
- Dropping `langchain-community` from `pyproject.toml`: FAISS still needs it
  until Phase 3 (DEC-5).
- Per-loader splitter strategies (SRS §7.2) — not storied yet.
- Ingestion error policy and exit codes: S1-4.

## Verification

```bash
# 1. baseline BEFORE the change (record in this file, under Results)
uv run rag-ingest 2>&1 | tail -3        # chunk count
uv run python -c "
from rag_qa.chain import build_rag_chain; from rag_qa.config import load_config
r = build_rag_chain(load_config()).invoke({'question': 'What is this corpus about?'})
from rag_qa.chain import citation; print([citation(d) for d in r['context']])"

# 2. ... and the same two commands AFTER. Chunk count and citations must match.

# 3. recursion works and the loader map is honoured
mkdir -p <scratch>/corpus/sub && cp corpus/*.pdf <scratch>/corpus/sub/
RAG_DATA_PATH=<scratch>/corpus uv run rag-ingest 2>&1 | tail -5   # finds the subdir

# 4. unit tests, no network
uv run pytest tests/test_loaders.py tests/test_components.py -q

# 5. community imports are down to FAISS only
#    (import lines only — registry.py's allowlist names the prefix as a string)
grep -rnE "^\s*(from|import) langchain_community" src/rag_qa/ | grep -v vectorstore.py | wc -l   # → 0
```

## Review notes for the human

The before/after citation list is the whole review — if those strings differ,
`page_label` handling regressed and every answer's sources are subtly wrong.
Second, check `build_embedder` is the *only* place an embedder is constructed
(`grep -rn "embedder" src/rag_qa/`), since a drift between ingest and query is a
silent wrong-answer bug rather than a crash. Third, `pypdf` page text can differ
from `PyPDFLoader`'s in whitespace; a chunk-count match is the bar, not
byte-identical text.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
