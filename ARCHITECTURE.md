# RAG_QA_System — Architecture

Per-phase design, written in the WORKFLOW.md Step 1 pass and confirmed once by the
maintainer. Implementation sessions follow this file; they do not re-litigate it.
Requirements live in `SRS.md`; the board lives in `stories/STATUS.md`.

---

## Phase 0 — Make it run, make it honest

_Confirmed 2026-09-18. Scope: SRS §12 Phase 0. Exit: fresh clone ingests and answers a
query end to end; tag `v0.1`._

### 0.1 Decisions

**DEC-1 — LangChain 1.x, migrated now.** Verified fresh resolve on Python 3.12
(2026-09-18): `langchain-core 1.6.3`, `langchain-text-splitters 1.1.2`,
`langchain-community 0.4.2`, `langchain-huggingface 1.2.2`, `langchain-ollama 1.1.0`,
`langchain-classic 1.0.8` (transitive via community), `faiss-cpu 1.15.1`,
`sentence-transformers 6.0.1`, `torch 2.14.0+cpu`, `ragas 0.4.3` (coexists in one lock).

Two rules:

1. **Import surface.** Application code imports only from `langchain_core`,
   `langchain_text_splitters`, `langchain_community` (FAISS, loaders),
   `langchain_huggingface`, `langchain_ollama`. The `langchain` meta-package is not a
   dependency. `langchain_classic` is installed transitively but is **not imported** in
   Phase 0–1; the query chain is hand-composed LCEL (§0.4), not `create_retrieval_chain`.
   Phase 2 may import `langchain_classic.retrievers.document_compressors.CrossEncoderReranker`
   if a hand-rolled reranker Runnable proves larger than the wrapper.

2. **Explicit PyTorch index.** The CPU wheel index at `download.pytorch.org/whl/cpu`
   hosts stale `langchain-community` releases. Declared as a general index it makes uv
   resolve the whole stack to LangChain 0.3.x silently. It must be scoped to torch:

   ```toml
   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-cpu" }
   ```

   Verified: this locks core 1.6.3, `torch 2.14.0+cpu`, zero `nvidia-*` wheels.

Rejected: pinning 0.2.x (unmaintained, pydantic-v1 shims, ragas 0.4 needs core ≥ 1 so
the eval extra could not share the lock) and 0.3.x (same import rewrite later, no gain).

**DEC-2 — `mistral` for the Phase 0 proof; `phi3` as light fallback.** `mistral` and
`qwen2:7b` are the same weight class (~4.4 GB Q4, ~5 GB resident), so the reason is zero
download and known-good on this host, not RAM. `qwen2_ollama` stays as an unused config
entry; DEC-2 is revisited when the Phase 2 eval harness exists. LLM settings for every
Ollama entry: `temperature: 0`, `num_ctx: 4096`, `num_predict: 512`,
`validate_model_on_init: true`. Retriever `k: 5` (10 × 1000-char chunks overflows a
2048 default window silently and doubles CPU prefill). S0-6 checks `free -h` and uses
`phi3` if under ~6 GB free.

**DEC-4 — Corpus directory renamed to `corpus/`; `docs/` becomes project
documentation.** The old layout needed a standing rule to stay safe, and the text
loader claims `.md`, so any project doc left in the corpus directory gets chunked and
embedded into the index. S0-4 does the rename with the rest of the path rewrite; S0-7
moves `SRS.md`, `WORKFLOW.md` and this file into `docs/` and inverts the CLAUDE.md rule.
`CLAUDE.md` and `README.md` stay at root: the former auto-loads from the working
directory upward, the latter is what GitHub renders. `stories/` stays at root as
working state, not documentation.

**DEC-3 (lean, decided in the Phase 3 pass) — Qdrant over pgvector.** Native hybrid
dense + sparse retrieval, one container, no Postgres to operate. pgvector wins only if a
Postgres already exists in the deployment. Nothing store-specific lives outside
`vectorstore.py` in Phase 0–2.

**Implicit decisions reversed in Phase 0:** completion `llms.Ollama` → `ChatOllama`;
flat `PromptTemplate` → `ChatPromptTemplate` with `system`/`human` parts (OWASP LLM01
separation); `RetrievalQA` → LCEL Runnable with the contract in §0.4; config loaded from
cwd → `RAG_CONFIG` env var with paths resolved relative to the config file; `.doc` and
`device: cuda` entries removed.

### 0.2 Layout at Phase 0 exit

```text
repo root
├── pyproject.toml  uv.lock  .python-version  .gitignore
├── config.yaml                 single config, repaired (§0.3)
├── README.md  CLAUDE.md  stories/          root: GitHub landing + session bootstrap + board
├── docs/                       project documentation: SRS.md, WORKFLOW.md, ARCHITECTURE.md (S0-7)
├── corpus/                     the RAG corpus: the three PDFs (renamed from docs/ in S0-4, DEC-4)
├── scripts/fetch_dataset.py    moved out of the corpus dir in S0-1; fixed in Phase 1 (ISS-18)
├── src/rag_qa/
│   ├── __init__.py             __version__
│   ├── registry.py             import_from_string, build_object, resolve_ref — pure, no LangChain imports
│   ├── config.py               load_config(path) -> dict; path resolution; env overrides
│   ├── vectorstore.py          create_store(chunks, embeddings, cfg) / open_store(embeddings, cfg) -> VectorStore
│   ├── ingest.py               load_documents(cfg), ingest(cfg) -> IngestReport, main()
│   ├── chain.py                build_rag_chain(cfg) -> Runnable
│   ├── cli.py                  REPL, main()
│   └── evaluate.py             main() — made functional in Phase 2
├── tests/                      Phase 1
└── vectorstore/                gitignored; produced by rag-ingest
```

Entry points (`[project.scripts]`): `rag-ingest = rag_qa.ingest:main`,
`rag-query = rag_qa.cli:main`.

Dependency direction: `cli` → `chain` → `vectorstore`, `registry`, `config`;
`ingest` → `vectorstore`, `registry`, `config`. **`ingest` never imports `chain`.**

### 0.3 Config shape

```yaml
components:            # the library: every entry with _target_ is built by registry.build_object
  loaders:    {pdf: ..., docx: ..., txt: ...}          # extensions key stays until Phase 1
  splitters:  {english_recursive: {_target_: langchain_text_splitters.RecursiveCharacterTextSplitter, ...}}
  embedders:  {minilm_cpu: {_target_: langchain_huggingface.HuggingFaceEmbeddings, model_kwargs: {device: cpu}}}
  llms:       {mistral_ollama: {_target_: langchain_ollama.ChatOllama, model: mistral, temperature: 0, num_ctx: 4096, num_predict: 512, validate_model_on_init: true},
               qwen2_ollama:  {... model: "qwen2:7b" ...}}
  retrievers: {vector_search: {search_kwargs: {k: 5}}}     # kwargs for VectorStore.as_retriever
  rerankers:  {cross_encoder: ...}                          # corrected shape, commented until Phase 2
pipeline:              # the assembly: dotted references into components
  ingestion: {splitter: components.splitters.english_recursive, embedder: components.embedders.minilm_cpu}
  query:     {llm: components.llms.mistral_ollama, retriever: components.retrievers.vector_search,
              prompt: {system: "...", human: "Context:\n{context}\n\nQuestion: {question}"}}
paths:
  data: corpus                 # renamed from docs/ per DEC-4
  vector_store: vectorstore/db_faiss
```

Rules: no absolute paths; `paths.*` resolve relative to the config file's directory;
env overrides `RAG_CONFIG` (which file), `RAG_DATA_PATH`, `RAG_VECTOR_STORE_PATH`. The
query-time embedder is always `pipeline.ingestion.embedder` (index and query must share).
Schema validation and a `_target_` allowlist are Phase 1 and land in `registry.py` /
`config.py` only.

### 0.4 Query chain contract

```python
retriever = store.as_retriever(**retriever_cfg)          # store = vectorstore.open_store(...)
prompt = ChatPromptTemplate.from_messages([("system", cfg_system), ("human", cfg_human)])
to_prompt = RunnableLambda(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
chain = (
    RunnablePassthrough.assign(context=itemgetter("question") | retriever)
    | RunnablePassthrough.assign(answer=to_prompt | prompt | llm | StrOutputParser())
)
# chain.invoke({"question": q}) -> {"question": str, "context": list[Document], "answer": str}
```

`format_docs` numbers chunks and prefixes each with `source` and `page` metadata so the
model can cite. The CLI prints `answer`, then one line per context document. Phase 2
FastAPI calls `chain.astream` on the same object; Phase 2 eval feeds `answer` and
`context` to RAGAs. `build_rag_chain` must not mutate the config it is given.

### 0.5 Vector store seam

`vectorstore.py` holds the only FAISS calls in the codebase:

- `create_store(chunks, embeddings, cfg) -> VectorStore` — `FAISS.from_documents` + `save_local`.
- `open_store(embeddings, cfg) -> VectorStore` — `FAISS.load_local(..., allow_dangerous_deserialization=True)`.

Invariant (ISS-16), documented in the module: the index directory is only ever produced
by `rag-ingest` on this host; never open an index from an untrusted source. Callers use
only `.as_retriever()` / `.add_documents()`. Phase 3 replaces the two bodies and adds a
`_target_`-built store component; nothing else changes.

### 0.6 Deliberately not built in Phase 0

VectorStore ABC or plugin registry, async code paths, Pydantic schema, retries and
circuit breakers, ingestion manifest, tests directory, FastAPI, hosted-LLM fallback.
Each has its phase; the seams above are enough to keep them from forcing a rewrite.

### 0.7 Phase mapping

| Target-stack row | Phase 0 | Later |
| --- | --- | --- |
| LangChain 1.x + LCEL | full migration, hand-composed chain | — |
| `ChatOllama` + hosted fallback | `ChatOllama` only | Phase 1: `groq_llama3` entry behind optional extra + `.with_fallbacks()` |
| Pydantic config + allowlisted `_target_` | keys/paths fixed; `registry.py` isolated | Phase 1, in `registry.py` / `config.py` |
| pytest / ruff / mypy | declared in dev group | Phase 1: tests, CI, coverage gate |
| FastAPI + SSE | Runnable contract | Phase 2: `api.py` over `chain.astream` |
| Reranker | config block corrected, disabled | Phase 2: hand-rolled Runnable or classic `CrossEncoderReranker` |
| RAGAs gate | `evaluate.py` importable, `eval` extra locked | Phase 2: golden set, local Ollama judge, GPU offload for sweeps |
| Qdrant / pgvector, hybrid | `vectorstore.py` seam | Phase 3 |
| OTel / structlog, Docker, CI scans | — | Phase 3 |

### 0.8 Host constraints that shape Phase 0

CPU-only, 15 GB RAM shared with desktop apps. Expect 15–30 s to first token and 1–2 min
per full answer with a 7B model; that is not a defect. NFR-2 (<2 s first token) is not
achievable on this host and is carried to the Phase 3 pass as an SLO revision or a GPU
serving decision.
