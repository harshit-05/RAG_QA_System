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

**DEC-5 — Staged exit from `langchain-community` (added 2026-09-25, S0-5).** The package
was sunset on 2026-05-22 (official issue #674): frozen, unmaintained, and it emits a
`DeprecationWarning` on import. This amends DEC-1 rule 1, whose import surface included it
for FAISS and the loaders. It stays through Phase 0, since nothing is broken and the lock
plus the `langchain-core<2` bound hold it steady. The exit rides work already planned:
loaders become our own ~40 lines on `pypdf`/`docx2txt` in Phase 1, the cross-encoder calls
`sentence-transformers` directly in Phase 2, and FAISS leaves with the Phase 3 move to
`langchain-qdrant`. After Phase 3, `langchain-community` and its transitive
`langchain-classic` are removed from `pyproject.toml`. Consequence for testing: a
command-line `-W error::DeprecationWarning` gate is unusable here, because
`langchain_core` overrides it on import. Deprecation checks record warnings in-process,
allow only the sunset notice, and carry a negative control.

_S3-3 (2026-10-20): step 3 is done. `langchain-community` and `faiss-cpu` left
`pyproject.toml` (`langchain-classic` was never in it), and a plain `uv sync` installs
neither LangChain package; ragas still brings both into the `eval` extra. The deprecation
gate now allows no warning, the sunset notice included, and its negative control is
`ChatOllama.dict()`, deprecated in langchain-core 1.4.2 and removed in 2.0, which the `<2`
bound keeps out._

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

**As built in S0-5 (verified 2026-09-25):**

- Signature is `build_rag_chain(config)`, taking a loaded config dict, not a path. It is
  silent (no progress prints) so the Phase 2 API can call it; front ends print progress.
- **Stream order is `question` → the whole `context` in one chunk → `answer` token by
  token.** So a streaming client can send citations before the first answer token. The
  Phase 2 SSE endpoint should emit a `sources` event from the `context` chunk, then
  `token` events, then `done`.
- Citations render as `file.pdf, p. <page_label>`, falling back to `page + 1`, with no
  page for docx/txt. The CLI numbers sources `[n]` exactly as the prompt does.
- The prompt is split into `system` (instructions) and `human` (retrieved context plus
  question), which is the structural separation OWASP LLM01 asks for.

### 0.5 Vector store seam

`vectorstore.py` holds the only FAISS calls in the codebase:

- `create_store(chunks, embeddings, cfg) -> VectorStore` — `FAISS.from_documents` + `save_local`.
- `open_store(embeddings, cfg) -> VectorStore` — `FAISS.load_local(..., allow_dangerous_deserialization=True)`.

Invariant (ISS-16), documented in the module: the index directory is only ever produced
by `rag-ingest` on this host; never open an index from an untrusted source. Callers use
only `.as_retriever()` / `.add_documents()`. Phase 3 replaces the two bodies and adds a
`_target_`-built store component; nothing else changes.

_S3-3 (2026-10-20): the invariant is moot, because the pickle is gone. Since S3-2 the index
is a Qdrant collection, and S3-3 removed `faiss-cpu` and `langchain-community` from the
dependencies, so no index file is deserialized and `allow_dangerous_deserialization`
appears nowhere in `src/` (ISS-16, §9)._

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

---

## Phase 1 — Make it trustworthy

_Confirmed 2026-09-25. Scope: SRS §12 Phase 1. Exit: CI is green and would have caught
every Phase-0 bug; tag `v0.2`, version 0.2.0._

_Delivered 2026-10-02 (S1-1 … S1-8, released as `v0.2`). §1.1–§1.7 are kept as
confirmed; §1.8 records where the delivery differs from them._

Phase 0 made the system run. Everything it deliberately left standing is what Phase 1
closes: the config is a raw dict validated by nothing (a one-character typo, `llmS`, was
a runtime `KeyError` that cost real debugging time), `registry.import_from_string` will
import and instantiate any dotted path the config names (ISS-04) — and `ingest.py` calls
it _directly_, bypassing the `build_object` funnel where ADR-015 promised the fix would
land — there are zero tests (ISS-07), no CI, no type hints (ISS-19), a bare
`except Exception` that drops a corrupt file from the corpus silently (ISS-05), and no
error handling at all around the REPL's chain call (ISS-06).

The exit criterion is taken literally. §1.5's regression suite encodes each Phase-0 bug
as a test case, so "CI would have caught every Phase-0 bug" is an executable claim
rather than an assertion.

### 1.1 Decisions

**DEC-6 — The config is a frozen Pydantic model; `_target_` is reached through an
alias.** `load_config()` returns a frozen `RagConfig`, not a dict; call sites move to
attribute access. This closes FR-8 and turns the "`build_rag_chain` must not mutate its
config" rule from a convention into a structural guarantee, **scoped precisely**:
`frozen=True` raises on attribute assignment, but dicts inside the model (a kind's
mapping, a leaf's `model_kwargs`) stay mutable — verified. The guarantee therefore rests
on the access path: call sites only ever receive fresh copies via `component(ref).spec()`
/ `.kwargs()`, so nothing they do reaches the config. Component _kinds_ are typed with
`extra="forbid"`, so `llmS` fails at load (ISS-01 caught by the schema itself) and the
dead `vector_stores` block (ISS-11) cannot return. Component _leaves_ keep
`extra="allow"`: their kwargs are open by design (ADR-010). **Retrievers are not
components:** `components.retrievers.*` carry no `_target_` — they are kwargs for
`VectorStore.as_retriever()` — so they get their own `RetrieverSpec`; typing them as
`ComponentSpec` makes the real config fail to load (verified).

> **Verified trap (2026-09-25).** Pydantic treats a field literally named `_target_` as a
> private attribute: `class M(BaseModel): _target_: str` accepts the input and
> `model_dump()` returns `{}` — the target is silently dropped, which would break every
> `build_object` call with no error. The schema must use
> `target: str = Field(alias="_target_")` with `populate_by_name=True`, and dump with
> `model_dump(by_alias=True)` so `build_object` still sees a `_target_` key.

**DEC-7 — The `_target_` allowlist lives at the import funnel, hardcoded.** Allowed
module prefixes: `langchain_core.`, `langchain_community.`, `langchain_huggingface.`,
`langchain_ollama.`, `langchain_text_splitters.`, `langchain_classic.`, `rag_qa.`.
Enforced inside `import_from_string` rather than `build_object`, because that is the one
choke point both call sites share — it covers `ingest.py`'s direct call, the concrete
gap ADR-015 named. The list is a module constant: not config-overridable, no env escape
hatch, since an allowlist the config can edit is not an allowlist. Config load
_string-checks_ every component's prefix — nested `_target_`s included — without
importing anything; imports still happen only when a component is built. A separate
`check_imports(config, refs)` imports named targets and raises `ConfigError`; load never
calls it, tests do (it is how the ISS-03 regression, a misspelled class under an allowed
prefix, is caught). `langchain_classic.` is allowed as a **config**
prefix so the disabled reranker entry validates; ADR-007 constrains **source** imports
under `src/rag_qa/` and is untouched. Closes ISS-04.

_S2-4 (2026-10-04): `langchain_community.` and `langchain_classic.` left the list, and so
did the disabled reranker entry. Our own reranker replaced it (DEC-16), so no `_target_`
names either package, and a v0.2 config that still has the old entry is refused with a
message saying how to migrate. The allowed prefixes are now `langchain_core.`,
`langchain_huggingface.`, `langchain_ollama.`, `langchain_text_splitters.` and `rag_qa.`
(§2.8)._

**DEC-8 — Our own loaders; the extension mapping moves into the pipeline section.**
`rag_qa/loaders.py` implements `PdfLoader`, `DocxLoader` and `TextLoader` on `pypdf` and
`docx2txt`, both already locked and both already what the community loaders wrap.
`components.loaders.*` lose their `extensions` key and `pipeline.ingestion.loaders` maps
extension → component reference, leaving exactly one instantiation path —
`build_object({**spec, "file_path": str(path)})` — which is allowlist-covered by DEC-7.
Corpus discovery becomes `Path.rglob` (ISS-13, FR-2).

This is step 1 of 3 of the DEC-5 exit: afterwards `langchain_community` is imported only
by `vectorstore.py` (FAISS) and named only by the disabled reranker config.
**The risk is citations**, which read loader metadata: the story records the `v0.1`
baseline — chunk count plus every `citation()` string over the real three-PDF corpus —
_before_ the change and must reproduce it after. Metadata contract: `source`, `page`
(0-indexed), `page_label`, `loader`. Content hash and ingestion timestamp (SRS §7.3) are
Phase 2, where the manifest needs them.

**DEC-9 — NFR-6 splits; Phase 1 does error handling, not resilience.** ISS-05 (collect
per-file failures, print a summary, exit non-zero — NFR-7) and ISS-06 (REPL try/except)
are Phase 1. Timeout, retry-with-backoff and circuit-breaking (SRS §11 Resilience) need
the service shape to be meaningful and stay with the Phase 3 operations work. Recorded
so that NFR-6 is deferred deliberately rather than by omission.

**DEC-10 — The device axis stays as explicit config entries.** This supersedes the
backlog's plan to express device as `${RAG_EMBED_DEVICE:-cpu}` and let pydantic-settings
interpolate it: **pydantic-settings has no `${VAR:-default}` expansion for arbitrary
YAML values**, so that item rested on a false premise. With the CUDA torch variant
landing (DEC-12), one documented sync command plus repointing `pipeline.ingestion.embedder` at
`minilm_cuda` is already a one-line switch, and the four embedder entries restored in
`ebcfc2e` stay exactly as they are (FR-1 axes).

**DEC-11 — CI is GitHub Actions, the suite is hermetic, and the gates block from the
first workflow commit.** No Ollama, no model download, no network anywhere in the suite:
`langchain_core.embeddings.DeterministicFakeEmbedding` for embeddings (verified present
in the locked core 1.6.3) and a fake chat model for chain shape. Gate order, cheapest
first: `ruff check` → `mypy` (`disallow_untyped_defs` on `src/rag_qa`) →
`pytest --cov=rag_qa --cov-fail-under=80` → `pip-audit`. `evaluate.py` is omitted from
coverage: it is Phase 2 work and imports the optional `eval` extra. Each gate is added
by the story that makes it passable, and blocks from that moment. **No
`ruff format --check` in Phase 1** — the existing code is not ruff-formatted, and a
whole-tree reformat is exactly the kind of diff that gets rubber-stamped in a
review-the-diff workflow (ADR-002's reasoning) → backlog.

**DEC-12 — Mutually exclusive CPU / CUDA torch variants, and a plain `uv sync` stays
CPU.** `torch` moves behind two uv-conflicting variants, each routed to its own
`explicit = true` index, so one documented command makes the `_cuda` embedder entries
genuinely installable for the Colab/Kaggle re-embedding path. **Mechanism is chosen by
S1-7's spike**, because of one trap: extras have no default, so with torch only inside
`cpu`/`cuda` extras a plain `uv sync` would pull PyPI's CUDA torch through
`sentence-transformers`. Preferred: dependency groups with
`default-groups = ["dev", "cpu"]`; fallback: extras, with every install path (CI,
README, fresh clone) naming `--extra cpu`. This touches the load-bearing DEC-1 rule 2
setup, so the story re-runs S0-2's smoke test on the **installed environment** (the
lockfile legitimately holds both variants): a plain sync must still yield
`langchain-core` 1.x, `torch 2.14.0+cpu` and zero `nvidia-*` packages. On
this host the CUDA path is verifiable only as a resolve, never as a run (CLAUDE.md GPU
policy); the story states that as its limit rather than implying more.

**DEC-13 (added 2026-10-01, S1-4) — A partial ingestion failure still replaces the
index, and the run exits 1.** When some documents (or folders) cannot be read,
`rag-ingest` indexes the rest, saves over the previous index, and exits 1, saying
"rebuilt without them". The alternative was to keep the previous index whenever
anything failed. It is safer for unattended re-ingests, but one persistently bad
file would then block every update. Phase 1 ingests are run by hand and watched, so
a loud partial rebuild gets re-run, not missed. Phase 2's hash manifest dissolves
the choice: a document that fails keeps its previously indexed chunks, so the
`/v1/ingest` job reports the failure without regressing the index. Exit codes: 0
all indexed; 1 a document or folder unreadable, nothing indexed, or an unexpected
error (traceback); 2 could not start.

_Superseded in part by DEC-17 (S2-6, 2026-10-05)._ In an update, a document or folder that
cannot be read keeps its previous chunks and manifest entry, and exit 1 reads "The index
was updated; the failed documents kept their previous chunks". A full rebuild still leaves
failed documents out and says "rebuilt without them" (maintainer, 2026-10-05). Its old
chunks are unusable, or were asked away with `--rebuild`. Exit 2 also covers another
ingestion holding the lock.

### 1.2 Layout at Phase 1 exit

```text
src/rag_qa/
├── schema.py      NEW  RagConfig, Components, ComponentSpec, Pipeline, Paths
├── settings.py    NEW  pydantic-settings env layer: RAG_CONFIG, RAG_DATA_PATH, RAG_VECTOR_STORE_PATH
├── config.py           load_config(path) -> RagConfig; actionable ConfigError
├── registry.py         import_from_string (+ ALLOWED_PREFIXES), build_object
├── loaders.py     NEW  PdfLoader / DocxLoader / TextLoader on pypdf + docx2txt
├── components.py  NEW  build_embedder / build_splitter / build_llm / build_loader
├── vectorstore.py      create_store(chunks, embeddings, path) / open_store(embeddings, path)
├── ingest.py           recursive walk, failure collection, non-zero exit
├── chain.py            build_rag_chain(config: RagConfig) -> Runnable
├── cli.py              argparse (--config/--help), REPL try/except
└── evaluate.py         untouched (Phase 2)
tests/                  conftest.py, one module per source module, regressions, fixtures/
.github/workflows/ci.yml
```

Dependency direction extends ADR-009: `cli → chain → {components, vectorstore, config}`;
`ingest → {components, vectorstore, config}`; `components → {registry, schema}`.
**`ingest` still never imports `chain`.**

`components.py` is where the duplicated embedder resolution collapses. Today `chain.py`
and `ingest.py` each build the embedder from the same two lines, and if they ever drift
the index and the queries use different vectors — a silent wrong-answer bug, not a
crash. After Phase 1 that invariant is one function, `build_embedder(config)`, which
always reads `pipeline.ingestion.embedder`, rather than a convention two modules happen
to share.

### 1.3 Config shape changes

```yaml
components:
  loaders:
    pdf:  {_target_: rag_qa.loaders.PdfLoader}     # `extensions` key gone (DEC-8)
    docx: {_target_: rag_qa.loaders.DocxLoader}
    txt:  {_target_: rag_qa.loaders.TextLoader}
pipeline:
  ingestion:
    loaders:                                        # extension → component reference
      ".pdf":  components.loaders.pdf
      ".docx": components.loaders.docx
      ".txt":  components.loaders.txt
      ".md":   components.loaders.txt
```

Everything else in §0.3 stands: paths relative to the config file's directory, the three
`RAG_*` env overrides, and the rule that the query-time embedder is always
`pipeline.ingestion.embedder`.

### 1.4 Config schema and error contract

```python
# schema.py
class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)
    target: str = Field(alias="_target_")     # a bare _target_ field is silently dropped
    def spec(self) -> dict: ...               # model_dump(by_alias=True)

class RetrieverSpec(BaseModel):               # no _target_: kwargs for as_retriever()
    model_config = ConfigDict(extra="forbid", frozen=True)
    search_type: str | None = None
    search_kwargs: dict[str, Any] = {}
    def kwargs(self) -> dict: ...             # fresh dict, never a live reference

class Components(BaseModel):                  # extra="forbid" ⇒ `llmS` fails at load
    loaders: dict[str, ComponentSpec]
    splitters: dict[str, ComponentSpec]
    embedders: dict[str, ComponentSpec]
    llms: dict[str, ComponentSpec]
    retrievers: dict[str, RetrieverSpec]
    rerankers: dict[str, ComponentSpec] = {}

class RagConfig(BaseModel):                   # frozen
    components: Components
    pipeline: Pipeline
    paths: Paths

    @model_validator(mode="after")            # every dotted reference resolves
    def _check_references(self) -> "RagConfig": ...
    def component(self, ref: str) -> ComponentSpec: ...
```

`registry.resolve_ref` is deleted and `RagConfig.component()` replaces it at both call
sites — reference navigation belongs to the thing that knows the config's shape.

Two error-quality rules, which together are what FR-8 means by "a specific, actionable
error before building any component":

1. An unresolvable reference names the nearest candidate via
   `difflib.get_close_matches`. The ISS-01 message reads:
   `pipeline.query.llm → 'components.llms.mistral_ollama': 'components' has no key
   'llms' (did you mean 'llmS'?)`.
2. `config.py` catches `ValidationError` and re-raises `ConfigError` rendering the config
   file's path plus one line per error as `loc`-path + message — never a raw Pydantic
   dump.

`settings.py` preserves the S0-4 anchoring semantics exactly, and this is an acceptance
criterion, not an implementation detail: a relative path **in the config file** anchors
to the config file's directory (NFR-11); a relative path **in an environment variable**
anchors to the caller's working directory, because that is what a shell user typing
`RAG_DATA_PATH=./mydocs` means.

### 1.5 Testing and CI

Hermetic and fast. `registry`, `schema` and `config` import no LangChain at all
(ADR-009), so their tests run in milliseconds without the ML stack; `tests/fixtures/`
holds a tiny PDF, DOCX, TXT and MD, and everything that would otherwise reach the
network is faked.

The centerpiece is `tests/test_config_regressions.py` — one parametrized case per
Phase-0 bug, each asserting a clear `ConfigError` before anything is built (at load for
all but ISS-03, which only an import can catch):

| Case | Caught by |
| --- | --- |
| `llmS` component key (ISS-01) | `Components` `extra="forbid"` + reference resolution |
| absolute path in `paths` (ISS-02) | `Paths` validator (NFR-11) |
| `_target_` outside the allowlist (ISS-04) | prefix check at load |
| `CrossEncoderRerank` — nonexistent attribute (ISS-03) | `check_imports` (load never imports — DEC-7) |
| dead `vector_stores` block (ISS-11) | `extra="forbid"` |
| `paths: {data: }` → `TypeError` | typed `Paths` field |
| empty YAML → `AttributeError` | `RagConfig` required fields |

Per SRS §10 the suite import-checks every **referenced** `_target_` against the real
`config.yaml` and prefix-checks the unreferenced ones — so the disabled reranker entry
is validated without importing `langchain_classic`, and ADR-007 holds.

The S0-5 deprecation gate becomes a pytest that records warnings in-process and keeps
its negative control. Never a command-line `-W error` gate here: `langchain_core` calls
`warnings.filterwarnings("default", ...)` on import and overrides it, which is what made
the original gate structurally unable to fail (S0-5, Discovered).

CI (`.github/workflows/ci.yml`) runs on push and pull request: checkout →
`astral-sh/setup-uv` with caching → `uv sync --locked` → the DEC-11 gate order. The
`--locked` flag is itself a check: it fails if `uv.lock` has drifted from
`pyproject.toml`.

### 1.6 Deliberately not built in Phase 1

FastAPI and SSE, reranker activation, incremental ingestion with a hash manifest, the
RAGAs gate and `eval_dataset.jsonl` (all Phase 2). Qdrant, hybrid retrieval, OTel,
Docker and Trivy (Phase 3). Retry and circuit-breaking (DEC-9). `ruff format` (DEC-11).
The hosted-LLM fallback (`groq_llama3` + `.with_fallbacks()`) is **re-deferred from
Phase 1 to Phase 3**, where the serving decision lives: it widens the DEC-1 import
surface and needs an API key, for something Phase 1 could only exercise against a fake
failing LLM.

### 1.7 Phase mapping delta

| Target-stack row | Phase 1 | Later |
| --- | --- | --- |
| Pydantic config + allowlisted `_target_` | done: `schema.py` + allowlist at the import funnel | — |
| pytest / ruff / mypy | done: suite, CI, coverage gate, annotations | Phase 2 adds the RAGAs gate |
| `ChatOllama` + hosted fallback | `ChatOllama` only | Phase 3: fallback with the serving decision |
| Reranker | config validated, still disabled | Phase 2 |
| Own loaders (DEC-5 exit) | step 1 of 3: loaders | Phase 2 cross-encoder, Phase 3 FAISS |
| CI vulnerability scanning | `pip-audit` | Phase 3: Trivy + container build |

### 1.8 As delivered (2026-10-02)

The design held: no decision in §1.1 was reversed. These are the places where the
code differs from the text above, each with the story that records why.

- **Two typed accessors, not one** (S1-1). `RagConfig.component()` returns only
  buildable `ComponentSpec`s, and `RagConfig.retriever()` returns `RetrieverSpec`s. A
  union return would have needed narrowing at every call site under the mypy gate.
- **ISS-01 is caught one level earlier than §1.4's sample message** (S1-1). With
  typed kinds, `llmS` fails as an unknown key under `components` ("did you mean
  'llms'"), before reference resolution runs. Reference-level did-you-mean still
  covers typos in entry names. Pipeline slots are also kind-checked: `query.llm` must
  point into `components.llms`.
- **The allowlist has three checks, not one** (S1-2). Beyond the prefix check, the
  resolved object must be _defined_ under an allowed prefix (its `__module__`), and
  it must be a class. Prefixes alone let re-exported names and helper functions
  through, such as `importlib.import_module` via `rag_qa.registry`, and
  `langchain_core`'s `guard_import`. Both were verified, so the allowlist alone did not
  stop arbitrary code. Detail in the `registry.py` docstring.
- **Trust boundary, stated (S1-2 review, written down in S1-8).** The allowlist limits
  which classes a config can build, not their kwargs: `trust_remote_code: true`
  passed through `model_kwargs` runs a model repository's code. The config is
  trusted input like code. Never load one from an untrusted source, and **the Phase 2
  API must never let a request supply or override components.**
- **Test fixtures are generated, not stored** (S1-3). `tests/fixtures/` does not exist:
  `conftest.py` writes the PDF and DOCX at test time, because binaries are never
  committed. The parity tests against the replaced community loaders, one of which
  reads a real corpus PDF, leave with `langchain-community` in Phase 3.
- **Ingestion walks with stated rules** (S1-3, S1-4). Hidden paths and lock files are
  ignored and counted; symlinks are never followed and are listed as skipped; a
  loader that cannot be _built_ is a config error (exit 2), while a document that
  cannot be _read_ is recorded and fails the run (exit 1, DEC-13).
- **`rag-query` refuses to start without an index** (S1-4), exiting 2 instead of
  failing inside FAISS.
- **CI carries more than §1.5's gate order:**
  - `HF_HUB_OFFLINE=1` as a tripwire against model downloads (S1-2);
  - uv-managed Python pinned with `UV_PYTHON_PREFERENCE: only-managed` (S1-5);
  - a step that fails unless the installed torch is the `+cpu` build (S1-7).

  The coverage floor and omissions live in `pyproject.toml`, so a local
  `pytest --cov` enforces what CI does (S1-5).
- **`pip-audit` reads a freeze of the installed environment** (S1-8), not `uv.lock`,
  which records both torch variants. A plain `pip-audit` silently skips torch, because
  `2.14.0+cpu` does not exist on PyPI. The local label is stripped so torch is
  audited as its public release.
- **mypy and ruff were tightened** (S1-6). `warn_unreachable` is ISS-12's only guard,
  and `warn_unused_ignores` is on. Ruff starts from its 0.16 defaults plus `E501` at
  100 columns, rather than an explicit narrower list.
- **Torch variants are dependency groups** (S1-7, DEC-12 option (a)), on the `cu130`
  index, the only CUDA index that carries torch 2.14.

---

## Phase 2 — Make it a service

_Confirmed 2026-10-02. Scope: SRS §12 Phase 2. Exit: the system is callable over HTTP
and has an enforced quality bar; tag `v0.3`, version 0.3.0._

_Delivered 2026-10-13 (S2-1 … S2-9, released as `v0.3`). §2.1–§2.7 are kept as
confirmed, with the dated notes the stories added; §2.8 records where the delivery
differs from them._

_This pass ran on Opus 5.5, although ADR-018 routes architecture passes to Fable. The
maintainer chose to accept it rather than re-run it (2026-10-02)._

_Reviewed twice before any code (2026-10-02). The second review changed four decisions:
the fingerprint's parts and where it is computed (DEC-15), a versioned judge Modelfile
(DEC-15), generation directories with a symlink flip instead of a two-rename swap
(DEC-17), and cancelling the SSE stream itself instead of a separate producer task
(DEC-18). STATUS.md, "Design reviews", lists every finding._

Phase 1 made the command-line tool trustworthy. Phase 2 makes the system something other
programs can call, and gives it a quality bar that a change can fail. Today, four things
stand in the way:

- **The only front end is a blocking REPL, and Ctrl-C does not stop generation.** The
  v0.2 manual test measured 43 s of Ollama work after an interrupt (S1-8).
- **The reranker is still a disabled config entry** (FR-4, ISS-03), written against the
  maintenance-mode `langchain_classic`.
- **Every ingest re-embeds the whole corpus** (ISS-14): 1,708 chunks, about two minutes
  here. A document that fails to load drops out of the index (DEC-13).
- **Answer quality is unmeasured.** S0-6 found the 7B model misstating retrieved facts,
  and nothing would notice if a prompt or config change made that worse (FR-7, ISS-15).

Both halves of the exit criterion are made executable, as Phase 1's were:

- **"Callable over HTTP"**: the API's tests, plus a manual end-to-end run against real
  Ollama.
- **"An enforced quality bar"**: two CI gates (DEC-15).

**Where this pass reads the SRS more narrowly or more widely than it is written.** The SRS
is not edited here (WORKFLOW: only an explicit amend session changes it). These are the
differences, for that session to fold into a revision note:

- **§12 and §10 say RAGAs is wired into CI as a gate.** DEC-15 runs retrieval in CI and
  generation offline, then checks the committed result in CI. The maintainer chose this
  because a CPU judge takes hours.
- **§7.4 names `question`, `ground_truth` and `expected_sources`.** The golden set adds
  `id`, `answerable`, `must_not_contain` and `notes`, a superset.
- **§8.1 says the final SSE event carries the cited chunks.** The stream sends `sources`
  first, so a client can show citations before the first token (the §0.4 order).
- **§8.1 lists five endpoints.** `GET /v1/health/live` is added for process liveness.

### 2.1 Decisions

**DEC-14 — One answer-event stream serves every front end. Stopping an answer means
cancelling the task that consumes it.** `rag_qa/answering.py` turns a question into typed
events. The CLI renders them, the API encodes them as SSE, and the evaluation harness
records them:

```python
async def stream_answer(pipeline: QueryPipeline, question: str) -> AsyncIterator[AnswerEvent]
# yields Sources(sources), then Token(text) for each chunk, then Done(ttft_ms, total_ms)
```

There is one cancellation mechanism: the front end cancels the asyncio task that consumes
the stream.

- **CLI:** the session runs on one `asyncio.Runner`, and the Runner's own SIGINT handling
  does the cancelling.
- **API:** Starlette cancels the response stream when the client disconnects, and
  `stream_answer` runs inside that stream (DEC-18).

**Why the REPL blocks today.** `RunnableAssign._transform` (langchain-core 1.6.3,
`runnables/passthrough.py:565`) runs the answer branch in a `get_executor_for_config`
thread. Closing the generator waits for that executor to exit, which happens only when
Ollama finishes.

> **Verified trap (2026-10-02): going async is not enough on its own.**
> `RunnableParallel._atransform` (`runnables/base.py`, from line 4319) waits on its
> per-step tasks with `asyncio.wait` and never cancels them. Under
> `RunnablePassthrough.assign`, which is today's chain shape, a cancelled consumer
> therefore returns while the generation task runs on, unowned, until its next chunk
> arrives.
>
> The trial used a slow fake stream, cancelled at 0.5 s with the event loop kept running:
>
> | Shape | Consumer returned | Stream closed |
> | --- | --- | --- |
> | under `assign`, cancelled during a 2 s simulated prefill | 0.5 s | 2.0 s |
> | streamed directly | 0.5 s | 0.5 s, before the consumer returned |
>
> On CPU, that gap is 5–15 s of Ollama work. In a CLI, the loop is paused between
> questions, so the orphaned task would not stop at all. **`stream_answer` therefore never
> streams the `assign` chain.** It awaits retrieval, yields `Sources`, renders the prompt,
> then streams the model itself (the second trap, below) and calls `aclose()` on it in a
> `finally`.

> **Second verified trap (2026-10-04): streaming `prompt | llm | StrOutputParser()`
> directly is not enough either.** S2-1 shipped that shape, and its mid-stream cancel test
> turned out flaky in CI.
>
> - **The cause.** A Runnable sequence runs each chunk in its own task:
>   `coro_with_context` always calls `create_task` (`runnables/utils.py:155`). The parser
>   also hands each chunk to `run_in_executor` (`output_parsers/transform.py:47`).
> - **What goes wrong.** A cancel that arrives while a chunk is in flight ends that chunk's
>   task, not the model's read. The model's generator is left paused at a `yield`, and only
>   the loop's async-generator finalizer closes it, 7 loop iterations later.
> - **What a user would see.** In the CLI the loop is idle at the prompt, so the stream to
>   Ollama stayed open, and generation ran on, until the next question.
>
> | Shape | Mid-stream cancel, one core | Cancel forced while a chunk is processed |
> | --- | --- | --- |
> | `prompt \| llm \| parser`, streamed directly | 66 of 300 left the stream open | open on return, every time |
> | the model itself, `llm.astream(messages)` | 0 of 300 | closed before return, every time |
>
> **`stream_answer` therefore renders the prompt (`prompt.format_messages`) and streams
> `llm.astream(messages)` in its own task.** The model's stream is then only ever paused
> in its own read, so a cancel always lands there. This holds with no callback handlers
> (none are configured). A handler whose `on_llm_new_token` suspends, such as a tracer
> that hops threads, would reopen the window until the loop's finalizer closes the stream.

Other facts checked in this pass:

- **The cancel reaches the HTTP request.**
  - `BaseChatModel.astream` spawns no task; only `stream_events` does. Streamed itself,
    the model runs in the consuming task, so a cancel reaches `ChatOllama._astream`.
  - _Corrected 2026-10-04:_ this pass first reasoned that a sequence's chunk tasks,
    created by `_atransform_stream_with_config` and awaited directly, pass a cancel on
    just as well. They do, except when it arrives while a chunk is in flight: the second
    trap above.
  - That method streams through `ollama.AsyncClient`, which wraps the request in
    `async with httpx…stream(...)` (`ollama/_client.py:777`). The HTTP response is
    therefore closed.
  - That Ollama then abandons the request is the one link only the real daemon can
    show. S2-1's manual check covers it.
  - **Ollama may finish a prompt evaluation already in progress before it notices.** A
    third-party measurement (AbstractCore, 2026-09-23) reports exactly that, while
    generation stops at once. Not yet reproduced here. The acceptance criterion is
    therefore split: **our** HTTP connection closes within about 1 s of the cancel, and
    Ollama's runner goes idle no later than the end of the prefill (at most the 5–15 s
    of caveat 3). S2-1 records what it measures.
- **`asyncio.Runner` handles both Ctrl-Cs** (Python 3.12, `runners.py:105–157`).
  - The first SIGINT during `run()` cancels the running task, and `run()` raises
    `KeyboardInterrupt` once that task has unwound.
  - A second SIGINT raises at once. That is also the fix for the v0.2 finding that a
    second Ctrl-C was swallowed.
  - That second raise happens inside the loop, while the cancelled task may still be
    unwinding. The task then stays pending, and would resume during the next `run()`.
    So the CLI treats a `KeyboardInterrupt` from `run()` whose task is not done as
    "quit", and calls `runner.close()`, which cancels and finalizes it.
  - `run()` re-raises `CancelledError`, not `KeyboardInterrupt`, when the task's cancel
    count does not return to zero. The CLI catches both.
  - Between `run()` calls, the default handler is back, so `input()` behaves as before.
  - The loop lives for the whole session, so ChatOllama's async connection pool stays on
    one loop.

Consequences:

- **`chain.py` gains `QueryPipeline`, which holds the parts** (§2.4):
  - `retrieve`: the retriever, then the reranker when one is configured;
  - `prompt`: the chat prompt, which `stream_answer` renders (`chain.build_prompt`);
  - the `llm` itself, which `stream_answer` streams, and whose clients `aclose()` closes.
- **`build_rag_chain` stays, composed from the same parts:** the same `build_prompt`
  and `llm`, as `prompt | llm | StrOutputParser()`. The §0.4 `invoke` contract and
  ADR-008 therefore stand. Its docstring says it is not for cancellable streaming, and
  `test_chain` checks that both paths give the model the same prompt.
- **`components.aclose_llm(llm)` closes ChatOllama's HTTP clients.**
  - ChatOllama has no public close. `ollama.AsyncClient.close()` exists, but is reachable
    only through the private `_async_client`.
  - So the helper is narrow, uses `getattr`, and is tested.
  - This closes the backlog's `ResourceWarning: unclosed socket`.
- **`Done.ttft_ms` is measured on every answer**: in the CLI, the API and evaluation
  alike. It is the data the Phase 3 NFR-2 decision needs (ADR-016).
- **Retrieval cannot be cancelled while it runs.** The FAISS search, the query embedding
  and the cross-encoder run in executor threads. A cancel returns the awaiting task at
  once (measured in the S2-1 review) and the thread finishes on its own.
  - The search and the embedding take milliseconds.
  - Since S2-4, the rerank takes a median of 1.4 s on this CPU (up to 2 s), on torch's
    10 threads. So a cancelled answer can leave its thread busy for a second or two.
    Running two reranks at once is safe: 40 concurrent reranks from 8 threads matched
    sequential results exactly (S2-4's second review). The cost is CPU, not correctness.
  - Still not worth cancelling the thread. In the CLI the user is typing the next
    question meanwhile.
  - In the API it does matter: a disconnected request frees its slot while its rerank
    runs on. So S2-7 must keep that rerank from overlapping the next request's: two
    reranks would compete for the same 10 threads. Its plan picks how. Either hold the
    slot until the retrieval thread finishes, or serialize retrieval with a lock in the
    retrieval half, which also covers `RAG_API_MAX_CONCURRENT` above 1.

Rejected:

- **Keeping `chain.astream` and closing the HTTP client from outside on cancel.** It
  reaches into private state on every answer, and it still leaves the orphaned task.
- **A thread per answer with a stop flag.** Python threads cannot be cancelled, which is
  the shape of the current bug.
- **Keeping `prompt | llm | StrOutputParser()` and letting the loop drain after a
  cancel** (2026-10-04). The stream would close only once the loop ran again, so the CLI
  would need a drain step after every interrupt. Streaming the model itself removes the
  per-chunk tasks instead.

**DEC-15 — The quality bar is two gates. Retrieval is recomputed in CI; generation is run
offline and checked in CI.** With a local judge, RAGAs takes hours on this host. A sweep
of about 25 questions × 4 metrics with `gemma2:9b` is estimated at 3–4 h, plus about
40 min to generate the answers. S2-1 later measured 254 s to first token, which would put
generation nearer 2 h. That ran on Ollama's default of 2 threads, with swapped weights
(STATUS.md caveat 3), so S2-5b sets `num_thread` and re-measures before it commits to a
figure. That cannot run on every push. A hosted judge would put a paid, networked
dependency into CI, and the maintainer rejected it (2026-10-02). The bar is therefore
split by what can be computed where.

**Tier 1: retrieval, recomputed on every pull request and every push to `main`.**

- **What it measures.** `rag-eval retrieval` runs every answerable golden question
  through `pipeline.retrieve`, which is exactly what the prompt would receive. It scores
  the result against the question's `expected_sources`: a file and its printed pages.
  - **Hit rate**: an expected page, or one of its `also_pages`, appears anywhere in the
    context.
  - **MRR**: the reciprocal rank of the first such page.
  - **Recall**: the share of expected `pages` the context covers. `also_pages` are not in
    its denominator.

  `also_pages` are pages that answer the whole question on their own, by repeating the
  fact (S2-2 review). Without them, retrieving the repeat would count as a miss, and a
  reranker that prefers the cleaner repeat would look worse than it is (S2-4).
- **Deterministic, with no LLM.**
- **The CI job, `eval-retrieval`**, ingests the real corpus into a temporary index
  (`RAG_VECTOR_STORE_PATH`). It then computes the metrics under `HF_HUB_OFFLINE=1`.
- **Network.** The job needs the embedder and cross-encoder weights, about 180 MB. They
  are cached by `actions/cache` and downloaded on a miss, which makes this the one CI job
  with network access. The unit-test job stays hermetic, exactly as DEC-11 says.
  - **The weights are fetched before the offline step.** `rag-ingest` loads only the
    embedder, so a separate warm-up step loads the cross-encoder once S2-4 adds it.
    Otherwise the offline step would fail on a cache miss. The warm-up builds the reranker
    from the config, through `components.build_reranker`, rather than downloading a model
    by name. It therefore caches exactly the files the offline step will open.
  - **The cache key is the model names, not `config.yaml`.** Any prompt edit would
    otherwise drop 180 MB of weights.
  - **The baseline is also confirmed on the runner.** Embeddings can differ in the last
    bits across CPUs, so the first CI run must reproduce the locally measured numbers
    within tolerance (0.05 is one question on about 20).

**Tier 2: generation, run offline, committed and checked in CI.**

- **`rag-eval generate`** consumes `stream_answer` (DEC-14) for every golden question. It
  writes `eval/runs/answers-latest.json`: the answer, contexts, sources and ttft for each.
  The generator is `pipeline.query.llm`, which is mistral.
- **`rag-eval score`** runs RAGAs over that file: faithfulness, answer relevancy, context
  precision and context recall.
  - **The judge is `gemma2:9b`**, reached through Ollama's OpenAI-compatible endpoint
    (`/v1`). It is a different model family from the generator, so it does not grade
    its own phrasing, and it is already pulled.
  - **The judge's settings are a committed Modelfile, `eval/judge.Modelfile`**:
    `FROM gemma2:9b`, plus `num_ctx 8192`, `temperature 0` and a fixed `seed`. Running
    `ollama create rag-judge -f eval/judge.Modelfile` makes the model the config names.
    It reuses gemma2's weights, so nothing is downloaded. This is needed because:
    - the OpenAI endpoint cannot set the context size per request (Ollama docs,
      "OpenAI compatibility");
    - Ollama defaults to a 4k context below 24 GiB of VRAM, which covers this CPU host
      and a Colab T4;
    - RAGAs' prompts (five contexts plus few-shot JSON) come close to 4k, and Ollama
      truncates silently. That is caveat 7 again, on the judge side, and it would
      produce confident, wrong scores.

    The Modelfile's text is part of the `judge` hash. **`score` refuses to run when the
    served model's `num_ctx` (from `/api/show`) differs from the Modelfile.** It also
    records the largest `usage.prompt_tokens` it saw, and fails the run if any prompt
    came within 256 tokens of `num_ctx`. A truncated judgement is then an error, not a
    number.
  - **Unanswerable questions get a decline rate instead** (FR-5). It is
    deterministic, and an answer counts as a decline only if both hold:
    - it contains the refusal's core phrase, `evaluation.decline_marker` (for example
      "could not find the answer"), compared case-insensitively;
    - it contains none of the record's `must_not_contain` strings, also compared
      case-insensitively: the prior-knowledge leak that S0-6 checked for ("Canberra").

    `rag-eval score` also refuses a golden set whose unanswerable `ground_truth` lacks the
    marker, which the pure loader cannot know.

    It is not an exact match on the full refusal sentence, because S0-6 showed mistral
    paraphrasing that sentence while keeping its core.
  - The output is `eval/runs/generation-latest.json`.
- **The two steps exchange only the answers file.** Scoring can therefore move to
  Colab/Kaggle (STATUS.md, GPU offload notes). There it needs a clone at the same commit
  and `uv sync --extra eval`, because `score` checks the answers file against that
  checkout. It does not need the index, the embedder or the generator. S2-5b records the
  recipe.
- **`rag-eval check` runs in the main CI job.** It needs neither ragas nor Ollama. It
  fails in two cases:
  - an aggregate is under its floor in `eval/thresholds.yaml`;
  - the committed run is **stale**: its fingerprint no longer matches the repository.

  A stale run fails rather than warns (maintainer, 2026-10-02). That is what makes the
  bar enforced.

**The fingerprint** (shape in §2.4) hashes everything that changes answers and that CI
can recompute from a checkout. It has six parts, so that `check` can name the re-run that
fixes a stale result:

| Part | What it hashes | When it moves, re-run |
| --- | --- | --- |
| `query` | the llm, retriever and reranker specs (content, not ref names), `reranker_candidates`, the prompt template, and a **rendered-prompt probe** | generate and score |
| `ingestion` | the embedder identity (DEC-17), the splitter spec, the loader map | generate and score |
| `corpus` | the sha256 of every file the corpus walk would load | generate and score |
| `questions` | each golden record's `id`, `question`, `answerable` and `must_not_contain` | generate and score |
| `references` | each golden record's `ground_truth` | score only |
| `judge` | the judge model name, the metric set, the ragas and instructor versions and `eval/judge.Modelfile`'s text | score only |

- **The rendered-prompt probe** is the chat prompt rendered over two fixed fake
  documents with `chain.format_docs`. A change to `format_docs`, to `citation()` or to
  how the prompt is assembled then moves `query`, not only an edit to the template.
- **Left out on purpose:**
  - paths, the judge's `base_url` and device keys, which are machine-specific;
  - the golden set's `notes` and `expected_sources`, which tier 2 never reads (tier 1
    recomputes its own numbers on every pull request and every push to `main`).
- **Where it is computed.** `evaluation/fingerprint.py` computes it, and `generate`,
  `score` and `check` all import it.
  - It may import `langchain_core` and `rag_qa.chain`, which the probe needs. Both are
    installed in CI's `check` job.
  - It must never import ragas, openai or the embedder's stack. `check` therefore stays
    a CPU-only, model-free step.
  - The ingestion identities come from `manifest.py` (S2-6), unchanged.

**Model digests are recorded, not hashed.** CI has no Ollama, so it cannot know a digest,
and a hashed digest would make every CI run stale.

- The answers file records the generator's Ollama `digest`, from `/api/tags`, when
  `generate` runs. The scores file records the judge's digest, and its base model's, when
  `score` runs.
- Locally, where Ollama is present, `rag-eval check --with-ollama` also compares the
  recorded digests with the live ones. A re-pulled `mistral` is then caught, but only on
  a machine that can see it. That is stated as a limit, not hidden.

**Thresholds** are the first measured baseline minus a tolerance. The story that measures
each tier sets them: S2-3 for tier 1, S2-5b for tier 2. They are only ever raised
deliberately.

- **Tier 1 is deterministic.** Its tolerance is 0.05, which is one question in about 20.
- **Tier 2's tolerance comes from measured noise, not from a guess.** S2-5b scores the
  same answers file twice. Each floor is then the baseline minus the larger of 0.05 and
  twice the spread observed between the two scorings. A 9B judge is not deterministic
  even at temperature 0, and a floor inside its noise would fail at random on a re-run.
  A metric whose two scorings differ by more than 0.1 is recorded but not gated.

**Honest limit: CI checks the committed tier-2 result. It does not recompute it.** A solo
maintainer vouches for a run by committing it. The answers file committed beside the scores
lets a reviewer see what was judged.

The fingerprint covers configuration, data and the rendered prompt. It does **not** cover:

- **code:** a change to a loader, the splitter, the reranker or `answering.py` that alters
  answers leaves it unchanged. Tier 1 recomputes retrieval on every pull request, which
  catches the retrieval half. For the generation half, the maintainer re-runs tier 2 after
  such a change; the check cannot force it.
- **model weights behind a tag:** these are recorded and checked locally only (above).

**`rag-eval score` copies the generation parts from the answers file**, and refuses to
score an answers file whose `query`, `ingestion`, `corpus` or `questions` parts do not
match the checkout it runs in. That check is cheap, and it stops a Colab run from scoring
stale answers. `references` and `judge` are computed at score time. An edit to a
`ground_truth` therefore costs a re-score (about 3–4 h on CPU, less on Colab), not a
re-generate. A typo fixed in `notes` costs nothing.

**Judge risk.** Judges of 7–9B parameters sometimes misparse RAGAs' structured prompts.
S2-5b therefore checks three answers by hand against the judge's verdicts before its
numbers become thresholds.

**RAGAs facts (verified 2026-10-02):**

- ragas 0.4.3 is the current release, and it resolves with core 1.6.3, in the `eval`
  extra only.
- It brings in `langchain`, `langchain-openai`, `langgraph`, `instructor`, `openai`,
  `datasets` 5 and `pandas` 3.
- Its 0.4 API is `llm_factory(model, client=AsyncOpenAI(...))` plus
  `ragas.metrics.collections`. The LangChain wrapper classes are deprecated.
- Ollama's OpenAI-compatible endpoint has no way to set `num_ctx` per request, and the
  default context is 4k below 24 GiB of VRAM (Ollama docs, checked 2026-10-02). Hence
  the judge Modelfile.
- S2-5's spike settles the exact judge wiring for Ollama: the instructor mode, and the
  embedder that answer relevancy needs.

_S2-5 (2026-10-06/07), on Opus 5.5 in place of Fable. The spike and the build settled
these; the story file has the recipe and the measurements._

- **ragas 0.4.3 resolves with the stack but does not import.** `ragas/llms/base.py`
  imports `langchain_community.chat_models.vertexai`, which langchain-community 0.4.2
  (the sunset release, our pin) removed (vibrantlabsai/ragas#2741, open, unreleased).
  ragas uses the class only in an `isinstance()` list on its legacy LangChain path, which
  the judge's `InstructorLLM` never takes. **Maintainer's call: a narrow import shim** in
  `ragas_scoring.py`. Only when that one module is missing, a stand-in class nothing is
  an instance of is registered. A test fails once a ragas release no longer needs it.
- **The judge wiring.** `llm_factory("rag-judge", client=AsyncOpenAI(...))` patches the
  client with instructor's JSON mode, which Ollama turns into grammar-constrained JSON;
  gemma2 has no tool support, so the default TOOLS mode would fail. There were 0 parse
  failures on the spike's 11 calls. Answer relevancy embeds with MiniLM on CPU, through
  ragas's own `HuggingFaceEmbeddings` (maintainer).
- **The Modelfile is what runs.** ragas sends `temperature 0.01, top_p 0.1, max_tokens
  1024` with every call, and a request overrides the Modelfile. So `score` sends the
  Modelfile's `temperature`, `seed` and `num_predict` (2048, set from the spike), and the
  hashed text is what is used.
- **Failures are classified by cause.** instructor wraps a connection error or a timeout
  in the same `InstructorRetryException` as a parse failure that exhausted its retries.
  Parse failures, truncated outputs and NaN scores are counted per metric; anything else
  aborts the run, so a dead Ollama cannot pass as parse failures. The context guard also
  fails a call whose prompt and answer together fill `num_ctx`, where Ollama would shift
  the context silently. Ollama's `prompt_tokens` counts the whole prompt even on a KV-cache
  hit (verified in 0.22.1's runners), so the 256-token check holds.
- **What the fingerprint table did not say:**
  - `manifest.CHUNKING_VERSION` (S2-6's third review) is in `ingestion`. A bump changes
    the chunks, so the answers: re-run generate and score.
  - `decline_marker` is in `judge`. It defines the decline rate, which `score` computes:
    re-run score.
  - The ragas version is `fingerprint.RAGAS_VERSION`, since CI has no ragas to ask. A
    test holds it to `uv.lock`, and `score` refuses an installed ragas that differs.
    instructor's is pinned the same way (the second reviewer, below).
  - `judge` also hashes the answer-relevancy embedder, the instructor mode and the metric
    settings. Answer relevancy asks one question, not three: under greedy decoding the
    spike's three were identical, so the mean is the same at a third of the calls.
- **`generate` refuses an index that lags the corpus or config** (maintainer). Otherwise
  the answers would come from old chunks under a fingerprint that claims the new ones. So
  a moved `corpus` or `ingestion` part says "re-run rag-ingest, then generate and score".
- **The two reviews added three rules:**
  - The scores file records the sha256 of the answers file it judged, and `check`
    compares it. A re-generate with unchanged inputs (a re-pulled model, unhashed code)
    moves no part, so without it `check` would pass scores of answers no longer
    committed.
  - `thresholds.yaml`'s `generation:` section has a required `max_unscored`: the largest
    share of answerable items a gated metric may leave unscored (parse, truncated or no
    number). A mean over the few items a misparsing judge did score would otherwise pass.
  - `score` checks that `rag-judge` is served with **every** PARAMETER of its Modelfile,
    and built on its FROM. An edited Modelfile that was never re-created would otherwise be
    hashed while the old judge ran.
- **The maintainer's review (2026-10-07) narrowed two hashes before any baseline exists:**
  - Runtime-only llm keys no longer move `query`: `validate_model_on_init`,
    `keep_alive` and the HTTP clients' settings, beside `base_url` and `device`.
    `num_thread` stays hashed, since a thread count can change an answer.
  - `judge` hashes the Modelfile without its comment and blank lines, so a comment edit
    costs no re-score. A SYSTEM or TEMPLATE text is kept whole.
  - Accepted, as DEC-17 accepted it for the retrieval embedder: answer relevancy's
    embedder is pinned by name, not by revision.
- **The second reviewer (2026-10-07) closed two gaps before any baseline exists:**
  - **instructor's version is pinned and hashed into `judge`, as ragas's is**
    (`fingerprint.INSTRUCTOR_VERSION`). In JSON mode instructor adds its own system
    message to every judge call, and writes the re-ask after a parse failure. So a lock
    bump changed the judge's prompts without moving `judge`. A test holds the pin to
    `uv.lock`, and `score` refuses an installed instructor that differs.
  - **The served judge is checked both ways.** A created model serves its base's
    parameters, SYSTEM and TEMPLATE, with the Modelfile's own in their place, as Ollama's
    `/api/show` shows. So `score` also refuses a parameter that neither gives it, an
    inherited one that changed, and a SYSTEM or TEMPLATE that is not the Modelfile's (or
    the base's, when the Modelfile sets none). Each is a Modelfile edit never re-created.
    When Ollama cannot show the base (removed after `ollama create`), only what the
    Modelfile sets is checked.
- **Cost, for S2-5b:** one answerable item took 39 min to score on this CPU (2 Ollama
  threads), so a full local scoring is about 13 h, not 3–4 h. Colab/Kaggle is the default.

_S2-5b (2026-10-11), on Opus 5.5. The baseline settled these; the story file has the
measurements._

- **Runs resume** (`evaluation/checkpoint.py`).
  - Each finished item is appended to `<out stem>.partial.jsonl`, `fsync`'d.
  - A re-run skips the items it holds, but only for the same identity: the generation
    parts and the generator's digest for `generate`; the answers sha, `references`,
    `judge`, the digests and the versions for `score`; and `--limit` for both.
  - The run file is still written once, atomically, and the checkpoint is deleted after.
- **`num_thread: 10`** is on every Ollama llm, in `query`. The judge's Modelfile has none,
  since it scores on a GPU.
- **The judge repeated exactly.**
  - Two scorings of the same answers file on one Colab T4, with Ollama 0.22.1, agreed to
    the last digit. The Modelfile's `seed` and temperature 0 made the run deterministic on
    one setup, which this decision had not expected.
  - So the measured spread was 0, and each floor is the baseline minus the 0.05 minimum.
    On `decline_rate`, whose 5 items are 0.2 each, that allows no change; it is strict on
    purpose (first review).
  - Cross-hardware spread is unmeasured, so re-scores run on the same setup.
- **Ollama digests are portable across machines for the same Ollama version.** The same
  Modelfile on the same `gemma2:9b` blobs gave `d85f8193c353` on Colab and here, so
  `check --with-ollama` holds for a Colab-scored run. The recipe pins `OLLAMA_VERSION`.
- **The judge is lenient, not noisy.** The hand check and the two reviews found that it:
  - scored two wrong answers as faithful: a made-up expansion, and numbers moved between
    a table's rows (S0-6's misattributed scores);
  - credited four references that the contexts held only in part.

  The floors stand as regression gates for this judge; the absolute numbers run high.
  `context_recall` stays gated: the judge's errors all leaned generous, so a fall it
  reports is likely real.

Rejected:

- **RAGAs in CI with a hosted judge**: cost, a secret, network access and a non-local
  dependency (maintainer).
- **Offline only**: nothing would be recomputed per push. Retrieval regressions are the
  cheapest kind to catch, and they would wait for a manual run.
- **A warn-only freshness check**: a bar that does not fail is not enforced.

**DEC-16 — The reranker is our own cross-encoder, on `sentence-transformers`** (DEC-5
step 2). `rag_qa/rerankers.py` defines `CrossEncoderReranker`.

- **What it is.** A `langchain_core` `BaseDocumentCompressor` around
  `sentence_transformers.CrossEncoder` (6.1.0: `predict`, `rank`).
  - Settings: `model_name`, `top_n`, `device`, and an optional `min_score`.
  - It writes `rerank_score` into the metadata of each chunk it keeps, and
    `SourceRef.score` surfaces it.
  - It is about 40 lines. The alternative is three classes from two maintenance-mode
    packages, all for one model call: `langchain_classic`'s
    `ContextualCompressionRetriever` and `CrossEncoderReranker`, and
    `langchain_community`'s `HuggingFaceCrossEncoder`.
- **It is built through `build_object` like every component**
  (`_target_: rag_qa.rerankers.CrossEncoderReranker`), so the allowlist covers it.
  - `components.build_reranker(config)` returns it, or `None`.
  - `QueryPipeline.retrieve` applies it after the retriever, and the stream order is
    unchanged.
- **Candidates.** `pipeline.query.reranker_candidates` (20) is how many chunks the
  retriever fetches when a reranker is on; `build_retrieve` overrides the retriever's `k`
  with it, and the reranker cuts them to `top_n` 5.
  - It is one switch: with the reranker off, the retriever keeps its own k=5, so 20 chunks
    can never reach a 4,096-token prompt silently (caveat 7).
  - Schema rule: `reranker_candidates` is required when `reranker` is set, forbidden when
    it is not, and at least the reranker's `top_n`.
- **Entries are named by both axes, like the embedders**: `ms_marco_minilm_cpu` and
  `ms_marco_minilm_cuda`.
- **`ALLOWED_PREFIXES` loses `langchain_classic.` and `langchain_community.`.** After this
  story no `_target_` names either, and an allowlist should allow only what is used
  (ADR-020). Three follow-ons:
  - CLAUDE.md's line about the reranker entry keeping `langchain_classic` named is
    rewritten.
  - The backlog line about declaring `langchain-classic` explicitly closes, as not
    needed.
  - The ISS-03 regression case keeps its meaning (a misspelled class under an allowed
    prefix, caught by `check_imports`), but moves to a `rag_qa.` path. The old path now
    fails the prefix check first.
- **Whether it is on by default is decided by numbers.** S2-4 records three:
  - the tier-1 metrics without the reranker;
  - the same metrics with it;
  - the measured CPU latency of reranking 20 candidates.

  The result also goes into the config comment.
- **`min_score` is the knob for a backlog line**: "Sources lists what was retrieved, not
  what was used". It defaults to off, and is tuned with the eval harness later.
- **ISS-03's other half dissolves.** That was a bare model string where an object was
  expected. The reranker takes `model_name: str` by design.

**DEC-17 — Ingestion is incremental, tracked by a content-hash manifest, and published by
an atomic flip between generations.** `<vector_store>/manifest.json` sits inside each
generation, beside its index, and records what is in it. `rag_qa/manifest.py` owns
hashing, the diff and atomic writes: a temp file, `fsync`, then `os.replace`. It is pure
Python, with no LangChain.

- **What gets re-embedded.** Each document is keyed by its corpus-relative path. Its
  entry carries the sha256 of the file's bytes and its loader's identity. A run compares
  the corpus walk against the manifest and sorts documents into four sets:

  | Set | What the run does |
  | --- | --- |
  | added | load and embed |
  | changed: the sha256 **or the loader identity** differs | load and embed; delete the old chunks |
  | removed | delete the chunks (FAISS `delete(ids)`, verified) |
  | unchanged | nothing |

  The loader identity is part of "changed". Without it, an edit to the extension map
  (say `.md` moving to another loader) or to a loader's spec would leave that loader's
  old chunks in the index forever.
- **Index updates are all or nothing, because FAISS's are not** (verified in
  langchain-community 0.4.2):
  - `add_embeddings` puts the vectors into the index (`faiss.py:312`) _before_ the
    docstore rejects a duplicate ID (`in_memory.py:26`), so a failed add leaves vectors
    with no document;
  - `delete` raises if any ID is missing (`faiss.py:930`).

  A run therefore works in two phases:
  1. **Prepare.** Load, split and embed every added and changed document. A document
     that fails here is recorded and keeps its old chunks; nothing has touched the
     store yet.
  2. **Apply.** Load the live generation into memory, apply every delete and add, and
     save the result as a new generation. Any exception aborts the run with nothing
     published, because the live generation on disk was never written to.

  If the manifest and the index disagree (a `delete` reports missing IDs, or the index
  cannot be opened), the run logs why and does a full rebuild instead of failing.

- **When it is a full rebuild.** Any one of these triggers it:
  - there is no manifest, which is true of every v0.2 index;
  - the manifest version is unknown;
  - the embedder identity or the splitter spec changed;
  - `rag-ingest --rebuild` asks for it.

  _S3-4 (2026-10-21): a splitter change no longer rebuilds in full._ Each document records
  the splitter that chunked it, its extension's own (`pipeline.ingestion.splitters`) or the
  default, and `diff` counts a document whose splitter changed as changed, as it does a
  changed loader. So a changed splitter re-embeds the documents it chunks, and the rest
  keep theirs. The manifest's top-level splitter names the default and decides nothing
  (§3.1, manifest version 2).

  _S2-6's third review (2026-10-06): the code that makes chunks is versioned too._ The
  identities see the config only. A change to `rag_qa.loaders` or to the metadata
  `ingest` adds, or a pypdf, docx2txt or text-splitters upgrade that changes their output,
  would otherwise leave every unchanged file's old chunks in place for good. So
  `manifest.CHUNKING_VERSION` is recorded as the manifest's `chunking`, and a different
  one rebuilds in full. **Bump it whenever the same file and config would make other
  chunks.** A test pins the sample corpus's chunks to it, and says to bump when they
  change. A manifest without the key reads as version 1, so the indexes S2-6 built stay
  as they are. The query side does not check it: old chunks still answer.
- **The embedder identity is its spec minus the keys that do not change the vectors**:
  `device`, `model_kwargs.device`, `show_progress`, `cache_folder` and `multi_process`.
  - So `minilm_cuda` on Colab and `minilm_cpu` here count as the same embedder, and the
    DEC-12 offload path keeps working.
  - A change to `model_name` or `encode_kwargs` is a different embedder.
  - The dimension is recorded too, as a backstop.
  - _S2-6 (2026-10-05, maintainer): `multi_process` stays **in** the identity._
    langchain-huggingface 1.2.2's multi-process path calls `encode_multi_process(texts,
    pool)` without `encode_kwargs`. So `normalize_embeddings`, a prompt and the precision
    are dropped, and switching it can change the vectors. The exclusions are `device`,
    `model_kwargs.device`, `show_progress` and `cache_folder`. A `model_kwargs` left
    empty is dropped as well, since `{}` is what leaving it out means (S2-6's first
    review).
  - _S2-6's third review (2026-10-06): a default spelled out is the same embedder._ For
    `HuggingFaceEmbeddings`, `encode_kwargs: {}`, `query_encode_kwargs: {}` and
    `multi_process: false` are dropped. That holds only for classes listed in
    `manifest.EMBEDDER_DEFAULTS`, whose defaults a test checks against the installed
    library. A blanket rule would be unsafe: in `HuggingFaceBgeEmbeddings`,
    `encode_kwargs: {}` turns normalisation off. No existing identity changed.
  - _S2-9's review (2026-10-14): the BGE example is wrong._ In the installed
    langchain-community 0.4.2, that class's `encode_kwargs` already defaults to `{}`, and
    it never normalises by itself. The rule stands for its own reason: a default counts
    only once a test checks it for that class, and an unlisted class keeps every key
    (ADR-026).
- **Chunk IDs are `uuid5(namespace, f"{relative_path}:{sha256}:{index}")`.**
  - They are deterministic, so an unchanged document keeps its IDs.
  - The path is part of the key because two files with identical bytes (a copy in another
    folder) would otherwise share IDs. LangChain's FAISS raises on duplicate IDs
    (`faiss.py:308`), and deleting one document's chunks would delete the other's.
    A rename therefore re-embeds the document, which is the safe side.
  - UUIDs are what Qdrant accepts as point IDs (DEC-3).
- **A failed document keeps its previous chunks and its manifest entry.** The run still
  exits 1. This is the way out of DEC-13 that DEC-13 anticipated.
- **Generations and a symlink flip.** FAISS writes two files, and not atomically. So
  `paths.vector_store` (`vectorstore/db_faiss`) becomes a **symlink** to a generation
  directory beside it, `db_faiss.gen-<UTC timestamp>-<8 hex>`. A run:
  1. writes the new generation (index and manifest) into a fresh directory;
  2. `fsync`s each file in it, then the directory itself;
  3. creates a temporary symlink to it, and `os.replace`s it over `db_faiss`. That is one
     atomic `rename(2)`, so a reader sees the old generation or the new one, and never
     neither;
  4. `fsync`s the parent directory;
  5. deletes generations older than the previous one.

  This is the release-directory pattern deploy tools use. It replaces a two-rename swap,
  which left a moment with no index at the path, and needed a repair rule for each
  leftover state. Here recovery has one rule, run under the lock at start:
  - delete every generation that is neither the symlink's target nor the newest older
    one;
  - delete stray temporary symlinks.

  An unreferenced _newer_ generation is a crash before the flip, so it is incomplete by
  definition and goes too.

  _S2-6 (2026-10-05): the stamp carries microseconds_, as in
  `db_faiss.gen-20261005T120000.123456Z-1a2b3c4d`. Recovery orders generations by name,
  and a second cannot order runs made within one. A new stamp also always sorts after the
  live generation's, even if the clock has stepped back (S2-6's first review).

  _S2-6's second review (2026-10-05) tightened what counts as ours:_
  - **A real folder is a v0.2 index only if it holds exactly `index.faiss` and
    `index.pkl`.** Before, any folder at the path was renamed aside, which with a mis-set
    path could be the corpus.
  - **Anything else at the path makes `rag-ingest` exit 2:** a file, another folder, or a
    link to anything but one of the store's generations.
  - **The link is read however it is written**, absolute, `./` or with a trailing slash,
    so a hand-made rollback never reads as "nothing is live" to recovery.
  - **Every update opens the live generation**, one that changes nothing included. A
    corrupt index is rebuilt rather than reported up to date. This needs no model: writing
    a generation never embeds.
  - **An update probes the embedder's dimension before preparing anything**, and a
    changed one rebuilds in full.

  _S2-6's third review (2026-10-06):_ a generation folder named as the index path, or a
  copy of one, is in the way too, and is named as such. The query side checks for anything
  in the way first, and says to point the path at the index, not to re-run `rag-ingest`,
  which would refuse it.

  - **Migration from v0.2.** If `db_faiss` is a real directory, the run is a full
    rebuild anyway (no manifest). It builds a generation, renames the old directory to
    `db_faiss.v02-<timestamp>`, and puts the symlink in its place. This is the only
    two-step moment, it happens once, and `rag-query` refuses that old index regardless.
  - **The generation id is the target's name.** The manifest records it, and the API
    detects a new index by `os.readlink`, with no mtime involved (DEC-18).
  - The corpus rule that symlinks are never followed applies to the corpus walk only.
    This symlink is ours, inside `vectorstore/`, which is gitignored.
- **One writer at a time.** An advisory `fcntl.flock` (flock(2), not `fcntl.lockf`) on
  `vectorstore/db_faiss.lock` is held for the whole run, outside the generations.
  - It is `flock` because its locks belong to the open file. So the API's ingest thread
    conflicts with the same process's other opens, and closing an unrelated descriptor
    never drops it, which POSIX `lockf` locks would.
  - A second `rag-ingest`, or an API job (DEC-18), exits 2 with "another ingestion is
    running".
  - The lock is Linux-only, as the deployment target is (SRS §2.4).
- **The query side refuses a mismatched index.** If there is no manifest, or the embedder
  identity differs, `rag-query` exits 2 with "re-run rag-ingest", instead of silently
  comparing vectors that are not comparable. This is the check the backlog asked for, and
  the first time the query path reads the manifest.
- **Chunk metadata reaches SRS §7.3.**
  - `source` becomes corpus-relative. It used to be the absolute host path, which the API
    must not leak, and which tied the index to one machine.
  - New: `source_sha256` and `ingested_at`, beside the existing `page`, `page_label`,
    `total_pages` and `loader`.
  - The loaders keep their contract; `ingest` rewrites `source` after loading.
  - `citation()` reads only the file name, so prompts and citations do not change.
- **Progress goes to a logger.** Ingest progress moves from `print` to the
  `rag_qa.ingest` logger. `rag-ingest` attaches a plain handler, so its output reads the
  same, while the API job and Phase 3's structlog consume the same messages.
- **The corpus walk prunes ignored folders** instead of descending into `.git` (a backlog
  item from S1-3 and S1-4).
- **Breaking:** every existing index must be rebuilt once. The commit carries `!` and a
  `BREAKING-CHANGE:` trailer.

Rejected:

- **Change detection by mtime**: copies and checkouts reset mtimes, and hashing the
  26 MB corpus costs well under a second.
- **A database for the manifest**: one JSON file, read whole and written atomically, is
  enough until Qdrant holds the metadata in Phase 3.
- **Keeping absolute `source` paths**: they tie the index to one machine and leak over
  HTTP.
- **A two-rename directory swap** (`live → .previous`, `.staging → live`), the first
  draft of this decision. Between the renames the path holds no index, and each crash
  point needs its own repair rule. The symlink flip has neither problem.
- **Mutating the store per document, to isolate failures**: FAISS's add is not atomic
  (above), so one failed add would corrupt the generation that gets saved.

**DEC-18 — The HTTP API is FastAPI with native SSE, shaped by what a CPU host can serve.**

- **Package.** `rag_qa/api/` holds an `app.py` factory with the lifespan, `routes.py`,
  `models.py` and `server.py`. The entry point is `rag-serve`.
- **Dependencies.** `fastapi>=0.135` and `uvicorn` are main dependencies, so a plain
  `uv sync` can serve.
  - Version 0.135 is where native SSE arrived: `fastapi.sse.EventSourceResponse` and
    `ServerSentEvent`, with keep-alive pings and anti-buffering headers built in.
  - Verified: FastAPI 0.142.2, Starlette 1.7.0 and uvicorn 0.54.0 resolve with the stack.
  - `sse-starlette` is not needed.
- **Endpoints.** The five from SRS §8.1, under `/v1`, plus `GET /v1/health/live`. Shapes
  are in §2.4.
- **The request never supplies components** (the constraint in §1.8).
  - `QueryRequest` is `{question}` only, 1–2,000 characters, with `extra="forbid"`.
  - Nothing in a request reaches `build_object`. A test posts `llm=` and expects 422.
  - The companion check lands with it: config load rejects a truthy `trust_remote_code`
    anywhere in `components`, nested `model_kwargs` included.
- **Auth** (maintainer, 2026-10-02):
  - By default the server binds `127.0.0.1` and needs no token.
  - With `RAG_API_TOKEN` set, every endpoint except the health probes needs
    `Authorization: Bearer <token>`, compared in constant time. Both sides are compared
    as UTF-8 bytes, because `secrets.compare_digest` raises `TypeError` on a non-ASCII
    `str`, which would turn a hostile header into a 500 instead of a 401.
  - `rag-serve` refuses a non-loopback `--host` unless a token is set, and exits 2.
  - With a token set, the interactive docs and `/openapi.json` are served only with it
    (or switched off); otherwise they would describe the surface to anyone.
  - Without a token, requests whose `Host` header is not loopback are refused, which
    stops DNS rebinding from a web page reaching the local server.
  - **Requests from a browser page are refused.** A cross-site "simple" POST still
    arrives with `Host: 127.0.0.1`, so the Host check does not stop it.
    - Any request with an `Origin` header is refused with 403, whatever the token
      setting. curl, the CLI and scripts send none, and the API serves no browser UI in
      Phase 2.
    - Every POST requires a JSON body with `content-type: application/json`, including
      `/v1/ingest` (`{"rebuild": false}` is required, not defaulted). A JSON
      content-type cannot be sent cross-site without a CORS preflight, which the server
      never answers.
  - A non-loopback host with a token sends the token in cleartext until Phase 3's TLS.
    `rag-serve` logs a warning at startup, and the README says so.
  - Rate limiting, TLS and multi-user auth belong to the Phase 3 deployment.
- **Every refusal happens before the first byte.** With FastAPI's yield-based SSE, the
  path function's body runs only once streaming has begun, after the 200 headers are
  sent. So auth, the busy check and the no-index check are **dependencies**, run in that
  order, and the generator holds only the answer.
- **Concurrency.** A CPU host generates one answer at a time. `RAG_API_MAX_CONCURRENT`
  (default 1) is a semaphore. When it is full, the request gets 503 with `Retry-After`
  rather than waiting silently in a queue.
  - A dependency takes the slot (`locked()` then `acquire()`, with no await between
    them) and hands the route a `Slot` whose `release()` is idempotent.
  - The SSE generator releases it in its `finally`. The dependency's teardown releases
    it too, which covers a client that leaves before the generator ever starts (the
    generator's `finally` cannot run if its body was never entered).
  - A leaked slot would answer 503 forever, so a test asserts that the slot is free
    after each kind of ending: done, error, disconnect, and disconnect before the first
    event.
- **A disconnect cancels generation** (DEC-14), through Starlette and the stream itself,
  **with no separate producer task.**
  - FastAPI's `EventSourceResponse` is a Starlette `StreamingResponse`; the SSE encoding
    lives in FastAPI's routing layer.
  - **uvicorn reports ASGI `spec_version` "2.3"** for HTTP (`httptools_impl.py`;
    release 0.32.1, "Drop ASGI spec version to 2.3"). On 2.3, `StreamingResponse` runs
    `listen_for_disconnect` beside the stream, and cancels it as soon as the client
    drops, during a silent prefill too. The "only on the next send" behaviour belongs to
    spec ≥ 2.4, which uvicorn does not claim.
  - So `stream_answer` runs **inside** the SSE generator. Starlette's cancel lands in
    `stream_answer`'s `finally`, which closes the stream to Ollama, exactly as Ctrl-C
    does in the CLI. A task spawned by the route would be outside Starlette's cancel
    scope, and would orphan generation: DEC-14's trap, one layer up.
  - `rag-serve` runs uvicorn only. If a later server reports spec ≥ 2.4, a disconnect
    watcher becomes necessary, and the startup log says which spec version is in use.
  - S2-7 proves it under real uvicorn, not only with the test client, which does not
    model a mid-stream disconnect.
- **Ingest jobs.**
  - `POST /v1/ingest` starts one job at a time. It returns 409 while a job runs, and the
    DEC-17 lock also stops a concurrent CLI run.
  - The job runs `ingest()` in a worker thread. Jobs live in an in-memory registry, which
    keeps the last 20.
  - On success, the server reopens the store and swaps the **retrieval half** of its
    `QueryPipeline`. In-flight answers keep the old one, and the LLM and its clients are
    not rebuilt.
  - **An ingest run outside the server** (the CLI) is picked up too.
    - Each query first compares `os.readlink(vector_store)`, the current generation, with
      the generation the loaded index came from. It is one syscall. Since the flip is
      atomic and the previous generation survives one more flip (DEC-17), there is no
      moment at which the reload can find a half-written index.
    - A reload runs the same compatibility check as startup. If the new generation was
      built with a different embedder (a CLI `--rebuild` after a config change), the
      server goes not-ready and keeps refusing queries until it is restarted with a
      matching config. Serving incomparable vectors is the bug DEC-17 exists to prevent.
    - The reopen runs in a worker thread, under an asyncio lock, so concurrent queries
      reload once.
    - `/v1/documents` and `/v1/health` read the manifest of the generation being served,
      so they never disagree with it.
- **Lifespan.**
  - It loads the config and builds the pipeline once.
  - The server can start without an index. It reports not-ready until a job builds one.
  - An incompatible index at startup counts as no index: the server starts, reports
    not-ready, and `/v1/health` says "re-ingest with --rebuild".
  - At shutdown it calls `QueryPipeline.aclose()`.
  - If Ollama is unreachable at startup (`validate_model_on_init`), `rag-serve` exits 2
    with an Ollama hint, not a traceback.
- **No host paths over HTTP.** Responses carry only corpus-relative paths. Errors say "no
  index yet", not where the index would be. A test asserts that the corpus root never
  appears in a response body. Three sources carry paths today and are covered:
  - `IngestReport.failed` holds `str(error)`, and an `OSError`'s text includes the
    absolute file path. Errors are stored as `type: message` with the corpus root and the
    store path made relative.
  - Ingest log lines name `data_path` and the store path (`ingest.py:155`). They become
    relative too, and the job keeps them as its `log`.
  - The SSE `error` event carries the exception type and a fixed message per type
    ("the language model is unreachable", …). The full exception goes to the server log
    only, since FAISS and httpx messages carry paths and URLs.

  The test for this ingests a temporary corpus holding a real `chmod 000` file through
  the real job, then scans the job's body and log.
- **Server settings come from the environment only**, as `RAG_API_*`, read by
  `rag_qa.settings`. That module stays the only reader of `RAG_*` variables, and
  `config.yaml` stays about the pipeline.

Rejected:

- **`sse-starlette`**: native SSE does the same since FastAPI 0.135.
- **WebSockets**: the token stream is one-way, and SSE is what SRS §8.1 names and what
  `curl` can read.
- **Queueing excess requests**: on CPU, a queued answer would wait minutes with no
  feedback.
- **Building the chain per request**: that is seconds of model loading per question.
- **A producer task plus a disconnect watcher**, the first draft of this decision. Under
  uvicorn, Starlette already watches for the disconnect. A task spawned outside its
  cancel scope is the one thing that could keep generating after the client leaves.
- **Detecting a CLI ingest by the manifest's `mtime_ns`**: it races a swap in progress,
  and coarse timestamps can hide one. The generation symlink cannot.

_S2-7 (2026-10-12), as built on fastapi 0.143.0, starlette 1.7.0 and uvicorn 0.54.0. The
story file has the spike's numbers._

- **The disconnect goes through FastAPI's own producer task.** Native SSE does not hand
  the route's generator to Starlette. It iterates the generator in a producer task, in a
  task group entered on the request's exit stack.
  - **How the cancel arrives.** On a disconnect, Starlette cancels its response task (spec
    2.3), and the response returns. The request stack unwinds, and that cancels the
    producer group: the cancel lands in `stream_answer`, as above.
  - **What stays true.** No task of ours, and none outside the request's lifetime. A drop
    during the prefill closed the model's stream in about 1 ms.
  - **The window.** Each event the producer sends passes a checkpoint outside the route's
    generator. A cancel landing there leaves the generator paused at a `yield`, in a
    reference cycle, with the stream to Ollama open until garbage collection runs. That
    happened in 1 of 300 mid-stream drops, and every time when forced.
  - **So the slot holds the answer stream, and its teardown closes it.** That teardown is
    the request's own; still no watcher.
  - **The review note changes.** "Check that no task is spawned for generation" now reads:
    no task _of ours_.
- **The API opens the index itself.** `build_query_pipeline(require_index=False)` could
  not say which generation it opened, give that generation's manifest, or say why there
  was none. So the API:
  - builds the LLM half from `build_llm` and `build_prompt`;
  - opens the retrieval half with `chain.open_retrieval`, which returns the generation and
    its manifest;
  - and `require_index` is gone.

  A reload compares `chain.live_generation` (the link, however it is written) with the
  generation served. It retries once when a flip lands mid-open.
- **One retrieval at a time, by a lock, not by holding the slot.** Every retrieval half
  served takes a `threading.Lock` inside its worker thread.
  - An orphaned rerank therefore delays the next retrieval rather than overlapping it, at
    any `RAG_API_MAX_CONCURRENT`.
  - The slot is freed at once, so a disconnect never turns into a 503.
  - A waiter whose client left skips its work.
  - Reopens take the same lock.
- **An open that fails for another reason is not ready, and is retried** (S2-7's first
  review). Only `NoIndexError` (and `IncompatibleIndexError`) says something about the
  index. A disk error or a failing embedder during a reopen used to escape `refresh()` and
  answer 500 on query, documents and health alike. Now the server is not ready with a
  fixed message, the exception goes to the log, and no generation is recorded as seen, so
  the next request opens it again. Requests that queued behind the failed attempt take its
  result instead of each trying again (S2-7's second review).
- **The refusals are dependencies, as above.**
  - `Origin`, then `Host` (tokenless only), apply to every route; the token to all but the
    two health probes; the slot and the index to the query.
  - **Two 422s, in two places** (S2-7's first review). FastAPI parses the body before it
    runs any dependency, so a body that is not valid JSON gets its 422 ahead of the 403 and
    the 401. A body that parses but breaks the request model is validated after them, so
    that 422 comes last. Nothing runs in either case, and a test pins the order.
  - **So a body is read before the token is checked.** Nothing limits its size yet (Phase 3,
     backlog), which a token does not protect against.
  - The docs and the schema are re-added as routes, so they sit behind the token.
  - A test sends every route a request without the token, so a route added unprotected
    turns CI red.
  - The Host check is 403, parsed by hand: Starlette's TrustedHost breaks on `[::1]`.
- **`rag-serve` checks Ollama before uvicorn starts.** Starlette hands uvicorn a failed
  lifespan's formatted traceback, which uvicorn prints, so "exit 2 with a hint, not a
  traceback" needs the check before uvicorn starts. uvicorn runs with
  `timeout_graceful_shutdown` 5 s, so stopping cancels an answer in flight.
- **Ingest jobs run in a daemon thread.** anyio's worker threads are joined at exit, and
  would hold a stopping server for the whole run. An abandoned run is safe (DEC-17). A
  `rag-ingest` holding the lock fails the job at once with exit 2, not a 409: a route
  cannot hold the `flock` on `ingest()`'s behalf.
- **The SSE encoding was checked.** fastapi 0.136.1 split SSE data with
  `str.splitlines()`, which would break JSON holding U+2028; 0.143.0 does not, and a test
  pins it.

### 2.2 Layout at Phase 2 exit

```text
src/rag_qa/
├── answering.py   NEW  AnswerEvent, SourceRef, stream_answer (DEC-14)
├── rerankers.py   NEW  CrossEncoderReranker (DEC-16)
├── manifest.py    NEW  pure: file hashing, embedder identity, diff, atomic JSON (DEC-17)
├── evaluation/    NEW  dataset.py         golden-set schema and loading (pure)
│                       retrieval.py       tier-1 metrics
│                       generation.py      rag-eval generate (consumes stream_answer)
│                       ragas_scoring.py   the only ragas importer; coverage-omitted
│                       fingerprint.py     the six fingerprint parts; langchain_core only
│                       gate.py            freshness + floors: rag-eval check (no models)
│                       checkpoint.py      resumable runs: <stem>.partial.jsonl (S2-5b)
│                       cli.py             rag-eval {retrieval, generate, score, check}
├── api/           NEW  app.py      create_app, the lifespan
│                       routes.py   the endpoints, and the refusals as dependencies
│                       models.py   request and response bodies
│                       state.py    Serving: the served generation, the slots, the
│                                   retrieval lock (S2-7)
│                       jobs.py     single-flight ingest jobs, in a daemon thread (S2-7)
│                       server.py   rag-serve: settings, preflight, uvicorn
├── chain.py            + QueryPipeline, build_retrieve, build_query_pipeline, build_prompt,
│                         open_retrieval, live_generation, check_index,
│                         IncompatibleIndexError; build_rag_chain kept for invoke (§0.4)
├── components.py       + build_reranker, aclose_llm, llm_reachable, preflight, OLLAMA_HINT
│                         (the start-up check both rag-serve and rag-query run, S2-8)
├── vectorstore.py      + incremental apply, generations + symlink flip, manifest-aware open
├── ingest.py           incremental, --rebuild, logging, lock, pruned walk
├── cli.py              one asyncio.Runner per session, over stream_answer
├── schema.py           + prompt placeholders, trust_remote_code guard, `evaluation`
├── settings.py         + ServerSettings (RAG_API_*)
├── registry.py         ALLOWED_PREFIXES without langchain_classic. / langchain_community.
├── py.typed       NEW
└── evaluate.py         DELETED (replaced by evaluation/)
eval/                   NEW, at the repo root, beside config.yaml
├── eval_dataset.jsonl  the golden set (SRS §7.4)
├── thresholds.yaml     tier-1 and tier-2 floors
├── judge.Modelfile     the judge's settings: gemma2:9b, num_ctx 8192, temperature 0, seed
└── runs/               answers-latest.json, generation-latest.json (committed, small text)
```

Entry points: `rag-ingest` and `rag-query` as before, plus
`rag-serve = rag_qa.api.server:main` and `rag-eval = rag_qa.evaluation.cli:main`.

Dependency direction extends ADR-009:

- `cli → {answering, chain, components, config}`, and never `api`. `components` came in
  with S2-8 (maintainer): `rag-query` runs the start-up check `rag-serve` runs, so
  `preflight` and `OLLAMA_HINT` moved from `api/server.py` into `components`.
  `tests/test_architecture.py` checks that the CLI loads no `api` module.
- `api → {answering, chain, ingest, manifest, components, config, settings}`.
- `evaluation → {answering, chain, components, config, manifest}`.
- `answering → chain`, for `QueryPipeline`, `citation` and `format_docs`.
- `ingest → {components, manifest, vectorstore, config}`. It still never imports `chain`,
  `cli` or `api`.
- `manifest` and `evaluation.dataset` import no LangChain. `tests/test_architecture.py`
  adds them to its config-layer list.
- `evaluation.fingerprint` and `evaluation.gate` may import `langchain_core` and
  `rag_qa.chain` (for the rendered-prompt probe), and nothing heavier.
  `tests/test_architecture.py` asserts that importing `evaluation.gate` loads none of
  ragas, openai, torch, sentence-transformers or langchain_huggingface. That is what lets
  `check` run in CI's model-free job.

### 2.3 Config shape changes

```yaml
components:
  retrievers:
    vector_search:     {search_kwargs: {k: 5}}
  rerankers:                                            # rewritten (DEC-16)
    ms_marco_minilm_cpu:
      _target_: rag_qa.rerankers.CrossEncoderReranker
      model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      top_n: 5
      device: cpu
    ms_marco_minilm_cuda: {...same, device: cuda}
pipeline:
  query:
    retriever: components.retrievers.vector_search
    reranker: components.rerankers.ms_marco_minilm_cpu     # on or off by S2-4's numbers
    reranker_candidates: 20                                # only with a reranker; overrides k
evaluation:                                                # NEW, optional; only rag-eval reads it (S2-5)
  decline_marker: "could not find the answer"              # core of the system prompt's refusal
  judge:
    model: "rag-judge"                                     # ollama create rag-judge -f eval/judge.Modelfile
    modelfile: "eval/judge.Modelfile"                      # hashed into `judge`; num_ctx checked at score time
    base_url: "http://localhost:11434/v1"                  # Ollama's OpenAI-compatible endpoint
```

_S2-5 (2026-10-06, maintainer): `modelfile` names a file in the eval folder (`eval/`
beside the config file, or `--eval-dir`), so the config holds `judge.Modelfile`, not
`eval/judge.Modelfile`. The eval files then travel together, and S2-5b's stale check runs
a scratch config against the repository's eval folder as written. The judge also gained
`embedding_model` and `timeout_s` (per call; runtime only, never hashed)._

`evaluation` and its `judge` are closed (`_Strict`) models.

- S2-5 adds whatever its spike shows the judge needs, for example the embedder that
  answer relevancy uses.
- `decline_marker` must appear in `pipeline.query.prompt.system`, checked at load. Edit
  the refusal sentence without updating the marker, and the config fails to load; the
  alternative is a decline rate that quietly drops to zero.

New validation in `schema.py`:

- **`Prompt`** (S2-5; the FR-8 follow-up found in S1-1):
  - `human` must contain `{context}` and `{question}`.
  - `system` must not contain `{context}`: retrieved text in the system turn would undo
    the OWASP LLM01 separation.
  - No other `{placeholder}` may appear anywhere, because each would be a required
    variable that fails on every question.
  - Parsing uses `string.Formatter`, which treats braces exactly as an f-string
    `ChatPromptTemplate` does, so `schema.py` stays free of LangChain.
- **`trust_remote_code`** (S2-7): a truthy `trust_remote_code` key in a component spec, at
  any depth, is a load error that names its location.

`paths` is unchanged. The manifest and the lock live beside `paths.vector_store`, and the
eval files under `eval/` beside the config file.

### 2.4 Interfaces

**Query pipeline and answer events** (DEC-14):

```python
@dataclass(frozen=True)
class QueryPipeline:
    retrieve: Runnable[str, list[Document]] | None   # retriever, then reranker; awaited, never streamed.
                                                     # None only while the API has no index (S2-7)
    prompt: ChatPromptTemplate                # rendered by stream_answer (chain.build_prompt)
    llm: BaseChatModel                        # streamed itself, never via a sequence (DEC-14);
                                              # kept too so aclose() can close its HTTP clients
    corpus_root: Path                         # SourceRef.source is relative to it
    async def aclose(self) -> None: ...

def build_retrieve(config: RagConfig, embeddings: Embeddings, *, reranker=None) -> Runnable[str, list[Document]]
def open_retrieval(config, embeddings, *, reranker=None) -> Retrieval  # S2-7: + generation, manifest
def build_query_pipeline(config: RagConfig) -> QueryPipeline
    # CLI and eval: no usable index raises NoIndexError or IncompatibleIndexError.
    # S2-7: the API composes its own pipeline (open_retrieval, build_prompt, build_llm),
    # starts without an index, and `dataclasses.replace`s the retrieval half after an
    # ingest; `require_index=False`, designed for it, was removed unused
class NoIndexError(Exception)   # stream_answer raises it when pipeline.retrieve is None
def build_rag_chain(config: RagConfig) -> Runnable    # §0.4 contract, same parts; invoke only

@dataclass(frozen=True)
class SourceRef:
    n: int               # the [n] the prompt used
    citation: str        # "2412.14140v2.pdf, p. 7"
    source: str          # corpus-relative path; the bare file name when the chunk's source is
                         # not under corpus_root (an index built on another machine)
    page: str | None     # printed page label, as the citation shows it
    score: float | None  # rerank_score when reranked

# AnswerEvent = Sources(sources: list[SourceRef]) | Token(text: str)
#             | Done(ttft_ms: float | None, total_ms: float)
```

`build_retrieve` builds the retrieval half on its own. Tier 1 runs it without an LLM or
Ollama, and the API swaps it after an ingest. `ttft_ms` is `None` when the answer yields no
token. A cancelled stream ends with `CancelledError`, not with `Done`. Errors propagate to
the front end, which renders them.

**SSE protocol** (`POST /v1/query`, `text/event-stream`):

| Event | Data (JSON) | When |
| --- | --- | --- |
| `sources` | `{"sources": [SourceRef, ...]}` | once, before the first token |
| `token` | `{"text": "..."}` | for each chunk |
| `done` | `{"ttft_ms": ..., "total_ms": ...}` | last event of a complete answer |
| `error` | `{"type": "...", "message": "..."}` | instead of `done`, when the answer fails after the stream began |

- **Before the stream begins, failures are HTTP statuses:**
  - 401: the token is missing or wrong;
  - 422: the body is invalid;
  - 503: there is no index yet, or the server is busy (with `Retry-After`).
- **A stream that ends with neither `done` nor `error` was cut off.**
- **There is no `Last-Event-ID` resume.** A dropped connection cancels the answer
  (DEC-14), and the client asks again.

**HTTP endpoints:**

| Method and path | Auth | Returns |
| --- | --- | --- |
| `POST /v1/query` | token, if set | the SSE stream above |
| `POST /v1/ingest` `{"rebuild": false}` | token, if set | 202 `{job_id, status}`; 409 while a job runs |
| `GET /v1/ingest/{job_id}` | token, if set | `{job_id, status, started_at, finished_at, exit_code, report, log}`; 404 if unknown. `log` is the job's ingest log lines, with paths made relative |
| `GET /v1/documents` | token, if set | `{documents: [{source, sha256, chunks, loader, ingested_at}]}`, from the manifest |
| `GET /v1/health` | none | readiness: 200 or 503, with `{status, index, llm}` |
| `GET /v1/health/live` | none | 200 while the process runs |

- **Readiness** means the index is present and compatible, and Ollama answers
  `/api/tags` within 2 s.
- **Job `status`** is one of `queued`, `running`, `succeeded` or `failed`.
- **`exit_code`** is what `rag-ingest` would have returned.
- **`report`** is the `IngestReport` as JSON. DEC-17 adds the added, changed, removed and
  unchanged counts.

**Manifest** (`<vector_store>/manifest.json`, DEC-17):

```json
{
  "version": 1,
  "generation": "db_faiss.gen-20261002T120000Z-1a2b3c4d",
  "chunking": 1,
  "embedder": {"ref": "components.embedders.minilm_cpu", "identity": "<sha256>", "dimension": 384},
  "splitter": {"ref": "components.splitters.english_recursive", "identity": "<sha256>"},
  "documents": {
    "2412.14140v2.pdf": {
      "sha256": "<sha256>", "loader": "components.loaders.pdf", "loader_identity": "<sha256>",
      "chunk_ids": ["<uuid5>", "..."], "ingested_at": "2026-10-02T12:00:00Z"
    }
  }
}
```

**Golden set** (`eval/eval_dataset.jsonl`, SRS §7.4, one object per line):

```json
{"id": "glider-purpose", "question": "...", "ground_truth": "...", "answerable": true,
 "expected_sources": [{"source": "2412.14140v2.pdf", "pages": ["7", "8"]}], "notes": "S0-6 failure case"}
```

- **`pages`** are printed page labels: what `citation()` shows and what a chunk's
  `page_label` holds. They are omitted for formats without pages.
- **`also_pages`** (optional, per source) are pages that repeat the whole answer. Tier 1
  counts them towards hit rate and MRR, but not towards recall. A page that repeats only
  part of the answer stays out, and is mentioned in `notes` only. Each one must be named
  in `notes`, which a test checks.
- **`source`** is corpus-relative. Tier 1 makes a chunk's `source` relative to
  `paths.data` before matching, so matching works both before and after DEC-17 makes
  `source` relative in the index.
- **`answerable: false` records** carry no `expected_sources`. Their `ground_truth` is
  the refusal sentence, and they count only towards the decline rate.
- **`must_not_contain`** is optional, and allowed only on unanswerable records. It lists
  strings whose appearance in the answer would show the model answering from prior
  knowledge, for example `["Canberra"]`.
- **`evaluation/dataset.py` validates the file:**
  - ids are unique;
  - an answerable record has a non-empty ground truth and sources;
  - an unanswerable record has no sources;
  - `source` and page labels are written exactly as they will be matched: normalised,
    with no surrounding spaces, and none repeated;
  - a page is in `pages` or `also_pages`, never both.

  Tests check that every `source` exists in the corpus, that every page label
  (`also_pages` included) exists in that file, and that every passage `notes` quotes is
  on the page it names.

**Floors and runs:**

```yaml
# eval/thresholds.yaml: floors, set from the first baseline minus a tolerance, raised deliberately
retrieval:  {hit_rate: ..., mrr: ..., recall: ...}                            # tier 1 (S2-3)
generation: {faithfulness: ..., answer_relevancy: ..., context_precision: ...,
             context_recall: ..., decline_rate: ...}                          # tier 2 (S2-5b)
```

`generation-latest.json` records:

- the `fingerprint`, as hashes of `query`, `ingestion`, `corpus`, `questions`,
  `references` and `judge` (DEC-15);
- the generator and judge models, with their recorded Ollama digests (not hashed), and
  the ragas, instructor and openai versions (the first two also hashed into `judge`);
- the judge's `num_ctx` and the largest `prompt_tokens` seen;
- the parse failures, counted per metric;
- timestamps;
- the aggregates, and the scores for each item.

`answers-latest.json` carries `query`, `ingestion`, `corpus` and `questions`, plus the
generator's digest. `check` requires the two files to agree on those four parts.

### 2.5 Testing and CI

The unit suite stays hermetic (DEC-11). Each story adds tests:

- **Cancellation (S2-1).** A fake chat model whose `_astream` is slow and records, in a
  `finally`, when it is closed.
  - Asserted: cancelling the consumer closes the stream before the consumer returns:
    mid-stream, during the simulated prefill, and with the cancel forced while a chunk
    is being processed (deterministic, the second trap). The CLI test raises a real
    SIGINT at that moment and checks that the stream is closed when the prompt returns.
  - Negative controls: the same assertion fails against the `RunnablePassthrough.assign`
    shape, and the forced cancel fails against `prompt | llm | StrOutputParser()`. Both
    are `xfail(strict=True)`, so if langchain-core ever fixes the behaviour, the suite
    goes red and says the trap is gone, not silently green.
  - Plus the manual check against real Ollama, which only the daemon can show: our
    connection closes within about 1 s, and the runner idles by the end of the prefill
    at the latest (DEC-14).
- **Reranker (S2-4).** A stub scoring model stands in for the download. Ordering,
  `top_n`, `min_score` and `rerank_score` are checked without network.
- **Manifest and ingest (S2-6).** Pure manifest tests, plus incremental runs with
  `DeterministicFakeEmbedding` covering:
  - an add, a change (by content, and by loader identity) and a delete;
  - a failure that keeps the old chunks;
  - each rebuild trigger, including a manifest that disagrees with its index;
  - a crash before the symlink flip, and recovery at the next start;
  - the v0.2 migration from a real directory to the symlink;
  - the lock.
- **API (S2-7).** FastAPI's test client against a fake pipeline covers:
  - SSE framing and event order;
  - 401, 403, 409, 422 and 503, and the slot freed after every kind of ending;
  - components refused in a request;
  - no host path in any response body, an ingest job over a real unreadable file
    included.

  One test runs real uvicorn on a free port and drops the client during a fake prefill.
  The fake model's `finally` must run, and the slot must be free.
- **Evaluation (S2-2, S2-3, S2-5):**
  - dataset validation;
  - metric arithmetic on hand-built contexts;
  - fingerprint sensitivity: each input moves its own part and only that one, and
    `notes` and `expected_sources` move nothing;
  - floor failures.

CI:

- **The `check` job keeps the DEC-11 gate order.** From S2-5b it gains a final step,
  `uv run rag-eval check`, which checks tier 2's floors and freshness. It needs no models
  and no Ollama.
- **The `eval-retrieval` job (S2-3)** runs after `check`:
  1. `uv sync --locked`;
  2. restore the HF cache;
  3. ingest the corpus into `$RUNNER_TEMP`;
  4. run `rag-eval retrieval` under `HF_HUB_OFFLINE=1` against `eval/thresholds.yaml`.

  Expect a few minutes: the 1,708 chunks take about 2 minutes to embed here.
- **Coverage** omits `evaluation/ragas_scoring.py` instead of `evaluate.py`, because it
  imports the `eval` extra, which CI does not install. mypy's no-stub overrides follow the
  same module.
- **The architecture test** grows in two ways:
  - `manifest` and `evaluation.dataset` join the modules that must load no LangChain;
  - importing `evaluation.gate` must load none of ragas, openai, torch,
    sentence-transformers or langchain_huggingface;
  - `api` joins the modules that `ingest` must never load.

### 2.6 Deliberately not built in Phase 2

- **Hybrid retrieval (BM25 + RRF) and metadata filtering**, the SHOULDs of FR-3. They
  belong to Phase 3, where Qdrant does both natively.
- **Rate limiting, TLS and multi-user auth.** These belong to the Phase 3 deployment.
  Phase 2 is localhost-first, with an optional token.
- **A durable job queue and a stateless query tier** (NFR-5). The ingest job registry
  lives in memory and the index lives in process. Both move in Phase 3, with Qdrant and
  the container.
- **Retry, backoff and circuit breaking** (NFR-6, DEC-9): Phase 3.
- **OTel and structlog** (NFR-10): Phase 3. Phase 2 only moves ingest progress to stdlib
  logging and measures ttft.
- **The hosted-LLM fallback**: Phase 3, with the NFR-2 decision.
- **SSE resume** (`Last-Event-ID`): a dropped stream is a cancelled answer.
- **The DEC-2 revisit** (mistral vs phi3 vs qwen2). It needs the tier-2 harness, so it
  comes after S2-5b, as a Colab/Kaggle sweep. It stays on the backlog.
- **Per-loader splitter strategies and `ruff format`**: still on the backlog, unchanged.

### 2.7 Phase mapping delta

| Target-stack row | Phase 2 | Later |
| --- | --- | --- |
| FastAPI + SSE | `rag_qa/api`, native SSE, a disconnect cancels generation | Phase 3: container, rate limiting, TLS |
| Reranker | own cross-encoder (DEC-16), on or off by tier-1 numbers | — |
| Own components (DEC-5 exit) | step 2 of 3: the cross-encoder; the allowlist shrinks | Phase 3: FAISS → Qdrant, and `langchain-community` leaves our direct dependencies (ragas still pulls it into the `eval` extra) |
| Incremental ingestion | manifest, generations with a symlink flip, lock, §7.3 metadata | Phase 3: Qdrant upsert, snapshots and backup (SRS §11) |
| RAGAs gate | two tiers: retrieval in CI; generation committed and checked | Phase 3: the DEC-2 revisit feeds the serving decision |
| `ChatOllama` + hosted fallback | `ChatOllama`, with its clients closed | Phase 3 |
| NFR-2 (<2 s to first token) | measured: `ttft_ms` on every answer and in every tier-2 run | Phase 3 decision (ADR-016) |
| OTel / structlog | stdlib logging in ingest | Phase 3 |

### 2.8 As delivered (2026-10-13)

No decision in §2.1 was reversed. Two mechanisms were reshaped by what the code met,
DEC-14's stream and DEC-18's disconnect, and the rest are refinements. The dated notes in
§2.1 carry the detail. This list is the index to them, with the story that records each
change.

- **DEC-14, the answer stream.**
  - **`stream_answer` streams `llm.astream(messages)` itself**, after rendering the
    prompt, not `prompt | llm | StrOutputParser()` (the second trap, fixed 2026-10-04,
    PR #10). A sequence runs each chunk in its own task, so a cancel could leave the
    model's stream open until the loop ran again. Strict `xfail` controls pin both traps.
  - **The second Ctrl-C closes the model's clients before the loop closes** (S2-1).
    ChatOllama's async client closes only on its own loop, so `cli.close_session` cancels
    the leftover answer, runs `pipeline.aclose()`, then `runner.close()`.
  - **`rag-query` starts in a fixed order** (S2-1, S2-8): the index, checked from its
    files (`NoIndexError` comes from `build_query_pipeline`, one source); then Ollama
    (`preflight`); then the models. Each failure is one message and exit 2.
- **DEC-15, the quality bar.**
  - **The golden set gained `also_pages`** (S2-2's second review), pages that repeat the
    whole answer. They count towards hit rate and MRR but not recall. Validation is
    stricter than §2.4 first listed: normalised sources, strict types, and a committed
    test that every quote in the notes is on its page.
  - **Both tiers' floor checks live in `evaluation/gate.py`** (S2-3), so the model-free
    job can run them. The floors themselves are in `eval/thresholds.yaml`, rounded down
    to 3 decimals.
  - **CI runs on pull requests, pushes to `main` and `v*` tags only** (S2-3's reviews): a
    story branch is tested by its PR. `eval-retrieval` has a 20-minute limit, saves the
    HF cache right after ingest, and runs offline on an exact cache hit. Any error while
    retrieving exits 2, so a crash never reads as a missed floor.
  - **S2-5's notes:** ragas 0.4.3 needs a narrow import shim; the judge runs in
    instructor's JSON mode and sends its Modelfile's settings with every call; it is
    checked against the whole Modelfile, both ways; instructor is pinned and hashed into
    `judge`, as ragas is; failures are classified by cause; the scores record the sha256
    of the answers they judged; `max_unscored` caps unscored items; `generate` refuses an
    index that lags the corpus; and `modelfile` is relative to the eval folder.
  - **S2-5b's notes:** runs resume from a checkpoint (`evaluation/checkpoint.py`);
    `num_thread: 10` is set on every Ollama llm; two scorings agreed exactly, so each
    floor is the baseline minus 0.05; Ollama digests carry across machines for one Ollama
    version; and the judge is lenient, which is recorded, not fixed.
- **DEC-16, the reranker** (S2-4). Tier 1 switched it on: hit rate 0.800 to 0.950.
  - Load never imports a class, so the schema reads `top_n` from the entry:
    `reranker_candidates` must be at least that, and at least 1.
  - The reranker refuses unknown keys, a `top_n` below 1, and a model with more than one
    score per pair, at construction.
  - A v0.2 config's refusal names the migration.
  - CI warms the cross-encoder up before saving the cache, whose key names both models.
- **DEC-17, incremental ingestion** (S2-6's notes):
  - `CHUNKING_VERSION` versions the code that makes chunks;
  - `multi_process` stays in the embedder identity (maintainer), and a default spelled
    out is the same embedder;
  - generation stamps carry microseconds;
  - only a folder holding exactly `index.faiss` and `index.pkl` is migrated as v0.2, and
    anything else at the index path exits 2;
  - every update opens the live generation and probes the embedder's dimension;
  - a full rebuild leaves failed documents out (maintainer).
- **DEC-18, the API** (S2-7's notes, unless named):
  - **The disconnect goes through FastAPI's own SSE producer task**, not straight from
    Starlette into our generator. The slot's teardown closes a stream that a drop between
    events left paused, the window S2-7's spike found.
  - **The API opens the index itself** (`chain.open_retrieval`, `live_generation`), and
    `require_index` is gone.
  - **One retrieval at a time, by a lock, not by the slot**, so a disconnect never becomes
    a 503 and an orphaned rerank never overlaps the next.
  - **A failed reopen is "not ready" and is retried**; requests queued behind it share
    the attempt.
  - **Two 422s:** a body that is not JSON is refused before the auth checks.
  - **`rag-serve` checks Ollama before uvicorn starts.** S2-8 moved that check into
    `components`, for `rag-query`.
  - **Ingest jobs run in a daemon thread**, and the `api` package has six modules, not
    §2.2's original four.
  - **Ctrl-C ends `rag-serve` with exit 0, as documented** (S2-9). uvicorn re-raises the
    SIGINT it captured after its graceful shutdown, asyncio's runner turns it into
    `KeyboardInterrupt`, and it escaped `main()` as a traceback with status 130. S2-7's
    log was tee'd without `-i`, so the end-to-end run was the first to see it. `main()`
    catches it, and a test sends a real SIGINT to a child process. A second Ctrl-C
    during the graceful wait still exits 0, but it does not shorten the wait: uvicorn
    then skips the lifespan's shutdown and logs the cancelled tasks (S2-9's review).
- **Dependencies:** fastapi 0.143.0, starlette 1.7.0 and uvicorn 0.54.0 (plain, over h11),
  with opentelemetry-api through FastAPI (S2-7); `requests`, declared rather than
  transitive (S2-8).
- **The exit criterion, as run** (S2-9). A fresh clone synced CPU-only torch, ingested
  the corpus (1,708 chunks), re-ran ingest with nothing to embed, and answered over HTTP
  and through `rag-query`. A disconnect during the prefill, a disconnect mid-stream and a
  Ctrl-C each closed our connection to Ollama within 0.03 s. Tier 1 and tier 2 are green
  in CI. The story file has the table.

---

## Phase 3 — Make it production-grade

_Confirmed 2026-10-15. Scope: SRS §12 Phase 3. Exit: the gaps in SRS §9 and §11 are closed;
tag `v1.0`, version 1.0.0. Phase 3 completes the SRS, so FR-3, NFR-1 to NFR-6, NFR-10, §2.4,
§7.1, §7.2 and §8.3 close in it too, or are amended in S3-12's amend session._

_This pass ran on Opus 5.5 at max effort, although ADR-018 routes architecture passes to
Fable, which is unavailable (CLAUDE.md, "Model routing"). The maintainer took the five
decisions that were theirs on 2026-10-14. Two design reviews run before any code, against
the pull request that adds this section, as Phase 2's did._

_Design review 1 (2026-10-15, breadth) found four blockers, fixed below. Qdrant's server breaks
no RRF ties, so our code makes the fusion's cut (DEC-20). A restore holds the writer lock
throughout (DEC-25). With a fallback configured, an open Ollama breaker routes to it rather
than answering 503 (DEC-21). The `eval` extra has four advisories, not one, so the `langchain`
bump moves into S3-9 (DEC-24). It also made the gap-closure table one row per requirement,
corrected and extended the SRS differences, and split S3-11. STATUS.md, "Design reviews",
lists every finding._

_Design review 2 (2026-10-16, depth) attacked the store, keys and limits, cancellation, the
sweep and the supply chain, against a real Qdrant v1.19.2, a real Caddy 2.11.7 and the
installed libraries. It found no blocker. Its 18 findings and 9 nits are fixed below, each
marked "design review 2". The store's write protocol, parity with FAISS, the snapshot round
trip, and cancellation under tracing and through the proxy all held. What changed:_

- _our code makes every cut and fuses hybrid results itself, since the server decides a tie
  by the collection's layout (DEC-19, DEC-20);_
- _a lost live collection no longer costs the previous one, and restores and snapshots leave
  nothing behind on the server (DEC-19, DEC-25);_
- _the host and the containers share one uid, and the client's address is trusted from Caddy
  alone, with bounded rate tables (DEC-23, DEC-24);_
- _every pin is verified once, by provenance pinned to its workflow and tag (DEC-24);_
- _the sweep's rules are margins fixed before its first run, and Colab restores this host's
  index rather than embedding again (DEC-26)._

_STATUS.md, "Design reviews", lists every finding with its evidence._

Phase 2 made the system a service. Phase 3 makes it one that can be deployed, watched and
recovered. Five things stand in the way:

- **The index is a pickle inside the API's process.** FAISS saves with pickle (ISS-16), and
  each replica would need its own copy (NFR-5).
- **Retrieval is dense only** (FR-3). A name, an acronym or a number that the embedder blurs
  is missed: tier 1's one miss, `glider-slm`, is of that kind.
- **Nothing is observable** beyond `ttft_ms` and log lines (NFR-10), and no call to Ollama
  has a timeout or a circuit breaker (NFR-6).
- **The API is localhost-first:** one token, no TLS, no rate limit, no body limit, and jobs
  held in memory (DEC-18).
- **There is no container, no image scan and no backup** (§11).

**The maintainer's decisions (2026-10-14, and four more from design review 1 on 2026-10-15).**
Each was asked with a recommendation first.

| Decision | Answer |
| --- | --- |
| DEC-3, the store | **Qdrant**, with three additions, each folded in below: backup and restore are tested against a real server, because local mode has no snapshots (DEC-25); the source of the sparse vectors is decided explicitly, and they enter the real index only in the batched re-run (DEC-20); the quickstart change is stated, a Qdrant container rather than local mode with `path=`, whose lock admits one process (DEC-19). |
| NFR-2 | **Revise the SLO** (DEC-27). |
| Hosted fallback | **An opt-in seam, plus one real check** with a provider key the maintainer supplies, in an environment variable only (DEC-21). |
| Auth | **Named, scoped API keys** (DEC-23). |
| `ruff format` | **Format the code once and gate it**, with Markdown excluded (DEC-28). |
| The `langchain` bump (review 1) | **In S3-9**, with the audit policy it serves, not S3-11 (DEC-24). |
| `max_file_mb` (review 1) | **50 MB** in `config.yaml`: a declared limit that skips no corpus file (S3-4). |
| The SRS amend session (review 1) | **S3-12 runs it**, before the release (WORKFLOW's explicit session). |
| S3-11 (review 1) | **Split** into the sweep (S3-11) and the commit (S3-11b), as S2-5 and S2-5b were. |

**Tier 2 is re-run once in this phase** (DEC-26). Every story before S3-11b moves no hashed
input and changes no answer, and shows how it knows: `rag-eval check`, and from S3-3 on the
contexts digest against S3-2's. S3-11 measures every lever that does, and S3-11b commits them.

**The exit, made executable: the gap-closure table.** S3-12 runs every row from a fresh
clone, on the compose stack. The story in the third column produces the evidence first, in its
own Verification; a row closed in an earlier phase names the test that still holds it. Design
review 1 made the table one row per requirement, and added §9's error handling, §11's hygiene,
NFR-4 and the whole of §2.4.

| SRS row | Gap at v0.3 | Closed by | Evidence at the exit |
| --- | --- | --- | --- |
| §9 LLM01, prompt injection | isolation is prompt wording only | S3-11b: retrieved chunks are delimited as data, and the system turn says so | a unit test that a chunk cannot close its delimiter; tier 2's floors hold |
| §9 LLM05, supply chain | the image is unscanned; the `eval` extra is not audited, and has four advisories | S3-9: Trivy over the image, plus secret and misconfiguration scans; the `eval` extra audited after the `langchain` bump; a weekly run that also scans every pinned third-party image; the audit policy; every pin verified once by DEC-24's recipe (design review 2) | CI green on the release tag, the `container` job and the `eval` audit included; each pin's check recorded beside it |
| §9 LLM07/08, excessive agency | none (the allowlist, S1-2) | the allowlist grows only by `langchain_openai.` (S3-6) | `tests/test_registry.py` |
| §9 deserialization | the FAISS pickle (ISS-16) | S3-2 and S3-3: no pickle anywhere; a restore verifies its checksum | `allow_dangerous_deserialization` appears nowhere in `src/` |
| §9 secrets | none in the repository | S3-9's secret scan; API keys stored as hashes; an `api_key` in a component spec refused (S3-6) | CI; a schema test |
| §9 errors handled explicitly | none (ISS-05, ISS-06: S1-4) | Phase 1; S3-6 adds NFR-6's explicit refusals | `tests/test_ingest.py` (a bad document fails the run), `tests/test_cli.py` (the REPL survives a failing chain) |
| §11 CI | no container build or scan; a red stage does not block a merge | S3-9, plus a branch ruleset the maintainer applies | the CI run; the ruleset |
| §11 containerization | none | S3-9: the image, compose, and the probes behind `/v1/health` | `docker compose up`; both probes 200 |
| §11 config management | no environment overlays | S3-9: `RAG_CONFIG_OVERLAY` | the stack runs on `config.docker.yaml` |
| §11 resilience | none | S3-6: timeouts, retries, breakers; S3-2: an interrupted ingest never publishes half an index | tests; a `kill -9` mid-ingest drill |
| §11 backup and DR | none | S3-10 | a backup and restore drill; RPO and RTO in the README |
| §11 repository hygiene | none (ISS-09: S0-1; history not rewritten, ADR-002) | Phase 0 | `tests/test_repo_hygiene.py` |
| FR-3, hybrid retrieval and filters | dense only | S3-5: BM25 sparse vectors fused by RRF, and a `sources` filter, both measured by tier 1; S3-11b switches hybrid on if tier 1 earns it | S3-5's tier-1 table; a filtered query that cites one file, and a 422 for an unknown one |
| NFR-1, retrieval under 300 ms at p95 | never measured | S3-7: `scripts/bench_retrieval.py` on 100k synthetic chunks, at k=20 | the p95, dense and hybrid, in README's "Capacity planning" |
| NFR-2, time to first token | 2 s is out of reach on CPU (ADR-016) | revised by DEC-27; S3-7 measures (a), and missed it; S3-11's stage 2 takes its levers; S3-11 and S3-11b report (b) | (a) the `sources` p95 over the golden questions, model idle; (b) the TTFT p95 here and on a T4 |
| NFR-3, ingestion throughput | undocumented | S3-7: `rag-ingest` reports it | a `--rebuild` into a scratch collection; README's table |
| NFR-4, incremental upsert | embedding is incremental (S2-6), but FAISS rewrites the index whole | S3-2: unchanged points copied, the changed documents' chunks upserted | a second `rag-ingest` embeds nothing; a one-file change embeds that file only |
| NFR-5, a stateless query tier | jobs in memory, the index in process | S3-8: job records in SQLite, the alias resolved per request; S3-9: two replicas behind Caddy | S3-8's two-instance test; `--scale app=2` |
| NFR-6, timeouts, retries, breakers | none | S3-6 | Ollama stopped: exit 2 within seconds, or a 503 with `Retry-After` before the first byte; the fallback answering when one is configured |
| NFR-10, logs and GenAI spans | log lines and `ttft_ms` only | S3-7 (logs, metrics), S3-7b (traces, alerts) | JSON log lines and the scrape (S3-7); an answer's and an ingest's traces in Jaeger, and `promtool test rules` (S3-7b) |
| §7.1, a declared maximum file size | none | S3-4: `max_file_mb: 50` in `config.yaml` | a scratch file over the limit skipped and reported |
| §7.2, chunking per document type | one splitter for every type | S3-4: per-extension splitters | a scratch `.md` file chunked by the markdown splitter, and re-embedded alone when that splitter changes |
| §8.3, environment overlays | none | S3-9: `RAG_CONFIG_OVERLAY` | the overlay tests; the stack on `config.docker.yaml` |
| §2.4, the operating environment | no containers, no fallback | S3-9 (Linux containers), S3-6 (the hosted fallback), S3-8 (the documented path to several replicas) | `docker compose up`; S3-6's real check; README's NFR-5 limits |

**Where this pass reads the SRS differently.** The SRS is not edited here (WORKFLOW: only an
explicit amend session changes it), and S3-12 runs that session (maintainer, design review
1). These are Phase 3's own differences, for it to fold in beside Phase 2's (the Phase 2
intro above):

1. **NFR-2** is split in two, as DEC-27 states.
2. **§6.2:** OpenTelemetry carries traces only. Metrics come from `prometheus-client`, which
   Prometheus scrapes.
3. **FR-3:** metadata filtering is a `sources` filter on the document path, not a filter on
   any metadata field.
4. **NFR-1** is measured at k=20, the reranker's candidates (the SRS says 10), as the query
   embedding plus the search, without the rerank that FR-4 adds. It is verified on 100k
   synthetic chunks.
5. **NFR-5 and §6.2:** the query tier is stateless and scales out. Ingestion stays on one
   host, as `rag-ingest` or an API job, with no separate worker, and the concurrency slots
   and rate limits are per replica.
6. **NFR-6** covers network calls: Ollama, Qdrant and the fallback. The embedder and the
   reranker run in process and have no network to retry.
7. **§8.3:** overlays are explicit (`RAG_CONFIG_OVERLAY`); `config.local.yaml` is never
   loaded on its own. Secrets come from the environment only, never from a config file.
8. **§11 backup:** snapshots go to a directory. Copying them off the host is the operator's
   documented step.
9. **§9 LLM01:** the corpus stays trusted, since nothing uploads into it. Chunks are
   delimited as data.
10. **§2.4:** the hosted fallback is opt-in, off by default, and never used by tier 2.
11. **§8.1:** `POST /v1/query` takes an optional `sources`; the `done` event gains `model`
    and token counts (additive); new statuses 413, 429, and 403 for a missing scope; an
    ingest job can read `abandoned`.
12. **§11 "blocking merge on any red stage"** is a branch ruleset the maintainer applies.
    The repository is public, so the free plan has rulesets.
13. **§2.6** names `qwen2:7b`. The model becomes whatever S3-11 decides.
14. **NFR-4:** an update writes a new generation. Unchanged points are copied, not embedded
    again, and the changed documents' chunks are upserted into it; the live collection is
    never updated in place, so no reader sees half an update.
15. **NFR-10:** spans are exported only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; the
    structured logs are always on.
16. **§11 repository hygiene:** the tracked binaries were untracked in S0-1, not removed from
    history (ADR-002).

_Design review 1 (2026-10-15):_ the first draft also listed three things the SRS does not
prescribe: no LangGraph (§6.2 says "LCEL / LangGraph"), Jaeger (one of §6.3's three trace
backends), and how BM25 is stored and fused (FR-3 asks only for both). Items 14 to 16 were
missing. Earlier phases left differences outside both phase lists too: ADR-008's
hand-composed chain and DEC-14's stream in place of §6.3's `create_retrieval_chain`, §6.3's
allowlist prefixes (`langchain_community.` is refused since S2-4), and §10's coverage targets,
which name files that no longer exist. S3-12's amend session sweeps the ADRs and §1.8 for
them.

**Versions verified for this pass (2026-10-14).** Python packages were resolved with uv in a
scratch copy of the project; the real `uv.lock` was not touched. Adding them moved no locked
version. Design review 1 (2026-10-15) resolved each again in a scratch copy, and found every
release and image tag below published.

| Component | Version | Notes |
| --- | --- | --- |
| `qdrant-client` | 1.19.1 | brings grpcio 1.84.0, protobuf 7.36.2, portalocker 3.2.0, h2 |
| Qdrant server | v1.19.2 (2026-10-05) | collection metadata since 1.16, parameterised RRF since 1.16, weighted RRF since 1.17 |
| `structlog` | 26.1.0 | |
| `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` | 1.45.1 | the same release as the `opentelemetry-api` FastAPI already brings |
| `opentelemetry-instrumentation-fastapi` | 0.66b1 | with the SDK and the exporter, 17 new packages, among them asgiref, wrapt, protobuf and googleapis-common-protos |
| `prometheus-client` | 0.26.0 | |
| `tenacity` | 9.1.4 | already locked, through langchain-core |
| `langchain-openai` (the `fallback` extra) | 1.6.2 | already locked through ragas, so nothing new resolves |
| `langchain` (the `eval` extra) | 1.3.2 → 1.4.3 | the bump resolves, in S3-9 (DEC-24). It also moves langgraph to 1.2.15, langgraph-checkpoint to 4.3.0, langgraph-sdk to 0.4.7, and websockets, which the main environment shares through langsmith, down from 17.1 to 16.1.1. The main environment still audits clean (design review 1). ragas 0.4.3 is still the newest release |
| Jaeger, Prometheus, Alertmanager, Caddy | 2.22.0, v3.15.0, v0.34.1, 2.11.7 | from their release lists |
| Trivy | v0.75.0 | run as a binary pinned by sha256 (DEC-24); v0.76.0 appeared on 2026-10-14, too new to pin. Design review 2 verified v0.75.0 by DEC-24's recipe: provenance signed by `reusable-release.yaml` at the tag, built from `591e979` on `main`, sha256 `c6e65abd…` |

### 3.1 Decisions

**DEC-3 — Resolved: Qdrant** (maintainer, 2026-10-14). The design is DEC-19.

**DEC-19 — The store is Qdrant. A generation is a collection, published by an atomic alias
switch, and the manifest travels in the collection's metadata.**

- **Our own module on `qdrant-client`, not `langchain-qdrant`.** This amends DEC-5 step 3's
  mechanism ("FAISS → `langchain-qdrant`"); its goal, leaving `langchain-community`, stands.
  - Ingest embeds every chunk in one call, in walk order, and that is what reproduces v0.2's
    vectors exactly (S2-6). `langchain-qdrant`'s `add_texts` embeds again, in batches of 64,
    which can move the last bits of a vector and break the parity proof below.
  - The generation protocol, the manifest and the hybrid query would be ours either way.
- **ADR-024's pattern carries over unchanged, with one recovery rule.**
  - **Names.** The configured `collection` is the name of an **alias**. Collection names must
    match `[a-z0-9_]{1,48}`, so the separator `--gen-` is unambiguous. The alias points at a
    generation, `<collection>--gen-<UTC stamp with microseconds>-<8 hex>`.
  - **Writing a generation**, all or nothing as in DEC-17:
    1. create the collection: a dense vector `VectorParams(size, COSINE)`, a payload index on
       `metadata.source`, and the manifest in the collection's `metadata`;
    2. copy the live generation's points, without the deleted ones (a scroll with vectors
       and payload);
    3. upsert the new points with `wait=True`, then check the point count;
    4. mark it complete: `complete: true`, beside the manifest (design review 2, below);
    5. switch the alias in one `update_collection_aliases` call that deletes and creates it.
       Qdrant documents that "multiple alias actions are performed atomically". Design review
       2 checked it on the server: the same call works on the first publish, with no alias
       to delete, and a batch that fails leaves the alias as it was;
    6. recover, to prune.

    Every step can be repeated: chunk IDs are uuid5 and generation names are unique, so a
    retry never doubles a point. A create or a delete repeated after a lost response is not
    a no-op, though, so DEC-21 states what a retry of each call counts as done (design
    review 1). Design review 2 killed a prototype of this protocol after each step, and at 8
    random moments, against the real server. The alias always kept the old generation, with
    its full count, and one recovery left exactly the live generation and the previous one.
  - **Recovery**, under the writer lock: delete every `<collection>--gen-*` collection that
    is neither the alias's target nor the newest older one. A crash before the switch leaves
    a newer orphan, which goes.
  - **With no alias, recovery deletes only incomplete generations** (design review 2).
    Qdrant deletes an alias together with its target (checked), so losing the live
    collection, by hand or in S3-10's drill, leaves no alias at all. "Keep the target and the
    newest older one" would then delete every generation, the previous one included, which is
    by then the last usable copy. FAISS's dangling link kept the target's name, so its
    previous generation survived (`_live_name` in v0.3's `vectorstore.py`).
    - A generation is complete once its metadata says `complete: true` and its manifest
      names the collection itself; with no alias, only the others go.
    - `rag-query`'s "no index" message then names the newest complete generation, and the
      README line that points the alias at it. The next publish prunes as usual.
  - **Something in the way**, a real collection named like the alias, or an alias pointing
    outside the pattern, is refused with exit 2 and never renamed (S2-6's second review).
    It is told from the two listings, `get_collections()`, which omits aliases, and
    `get_aliases()`. Never from `collection_exists` or `get_collection`, which resolve an
    alias to its target (checked, design review 2). The server refuses the collision too: a
    collection named like an alias (400), and an alias named like a collection (409).
  - **Readers bind to the generation they resolved**, not to the alias, so a switch between
    the check and the query cannot mix two generations. The previous generation survives one
    more switch, for readers still on it.
  - **Readers resolve the alias again for each question** (design review 1): the API per
    request, as its refresh does; `rag-query` before each question; `rag-eval` before each
    item. A FAISS reader held its index in memory, so deleting a generation never broke it.
    A Qdrant reader's collection lives on the server and is deleted two switches later, so a
    REPL session or a long `rag-eval generate` that outlived two ingests would fail. A search
    whose collection has gone resolves the alias once more. The server answers that search
    with a 404, mid-search too and never a 500, so the 404 is the signal, and it is never
    retried (design review 2). In the same review, 6 readers through the alias got answers
    from one generation in each of 2,973 queries while it switched 500 times.
  - **Cost:** the copy is linear in the index, about 1–2 s for 1,708 chunks. A clone from a
    snapshot is the path for a much larger corpus, not built in this phase.
- **Manifest version 2, kept in the collection's metadata.** Qdrant has had collection
  metadata since 1.16; the client and its local mode were checked in 1.19.1.
  - **Under one key, written whole** (design review 2). A metadata update merges top-level
    keys, a `null` deletes one, and a nested object is replaced whole (checked on the
    server). So the manifest sits under a key of its own, beside `complete`, and every write
    sends all of it. No stale field can then survive inside it, after a restore included.
  - **Its size.** A request is limited to 32 MiB, about 100,000 documents at 300 bytes each.
    And `GET /collections/{name}` returns the whole manifest: 9.5 MB came back for a 10 MB
    one. So hot paths never call `get_collection`: the refresh compares `/aliases`, counts
    come from `count`, and the manifest is read once per new generation.
  - **It holds** the identities (the embedder and its dimension, the splitter, the chunking
    version, the sparse encoder once one is set, and the store's own
    `{distance: cosine, exact: true}`), and, per document, `sha256`, `loader`,
    `loader_identity`, `splitter` (S3-4), `chunks` and `ingested_at`.
  - **S3-2 writes the whole shape** (design review 1): `sparse: null` until S3-11b sets an
    encoder, and each document's `splitter` as the default splitter's identity until S3-4
    adds per-extension splitters. The records are strict (`extra="forbid"`), so a field added
    later would make every version-2 manifest unreadable.
  - **A changed splitter re-embeds only the documents that use it** (S3-4), where DEC-17
    rebuilt in full. The per-document identities decide; the top-level splitter record names
    the default and decides nothing.
  - **The `chunk_ids` lists become a count.** A chunk's ID is derived as
    `chunk_id(name, sha256, i)` (DEC-17), so the list was redundant, and dropping it keeps
    the metadata near 200 bytes per document.
  - **Its readers keep their interface:** `check_index`, `generate`'s staleness check and
    `/v1/documents`.
- **Search is always exact,** `SearchParams(exact=True)`, in each hybrid branch too. Tier 1
  stays deterministic across machines, as FAISS's flat index was. One exact search over 100k
  vectors of 384 dimensions took 28 ms on the review server (design review 2), far below
  NFR-1's 300 ms, which S3-7 measures.
- **Our code makes every cut** (design review 2). Where several points tie at a limit, the
  server keeps a set that depends on how the collection was built. With 60 points tied and
  a limit of 10, three insertion orders kept three different sets, dense and sparse alike,
  and each set changed after a server restart, while every repeat within one process agreed.
  So `vectorstore` asks for more than it needs, extends the limit while the point at the cut
  ties with the next, orders by score, then point ID, and cuts. The real corpus has no such
  tie today: it has no duplicate chunk, and no tie straddles a cut of 20 or 40 for any golden
  question.
- **The distance is cosine.** MiniLM's pipeline ends in a `Normalize` module (checked in the
  cached model), so its vectors have unit length, and cosine ranks exactly as FAISS's L2 did.
- **State that is not the index.** `paths.state` (default `var/`) replaces
  `paths.vector_store`. It holds the writer lock, a `flock` on `var/ingest.lock` as in
  DEC-17, and S3-8's job database and job locks (`var/jobs/`). There is one writer per host.
  _(S3-2's second review, 2026-10-20: one writer per `var/`, strictly. The lock is a file
  and the index a collection, so writers that share an alias exclude each other only when
  they share `var/`. A second checkout on one host has its own, and on a throwaway server
  its recovery deleted the other run's new generation, and once the alias with it. The
  rule stands; caveat 36 says how to keep it.)_
  Compose bind-mounts `var/`, so the CLI and the containers share the locks. A `flock` holds
  across the host and a container on that mount, and so does SQLite's WAL, written from six
  containers and the host at once (both checked). Both sides need one uid (DEC-24, design
  review 2).
- **Breaking.** The story's commit carries `!` and a `BREAKING-CHANGE:` trailer.
  - `paths.vector_store` and `RAG_VECTOR_STORE_PATH` are **refused at load**, with the
    migration in the message. They are never ignored: caveat 9's habit of setting
    `RAG_VECTOR_STORE_PATH` for a scratch ingest would otherwise send that ingest into the
    real index.
  - A scratch index is now `RAG_QDRANT_COLLECTION=<name>`.
  - Every index is ingested again, once. v0.3's `vectorstore/` folder stays on disk;
    `rag-query` mentions it when it finds one, and the upgrade guide says to delete it.
- **Parity is proven, so the store move costs no tier-2 re-run.**
  - **The tool:** `rag-eval retrieval --contexts-out FILE` (new in S3-2) writes, for all 25
    golden questions, the sha256 of the rendered context (`format_docs` over the reranked
    chunks) and the citations, in order.
  - **The check:** S3-2 runs it on FAISS before changing the store, then on Qdrant. Identical
    files, and identical tier 1 question by question, mean every prompt is byte-identical,
    so the committed tier-2 run still describes the system. It runs twice on Qdrant, and the
    two files must match too: an exact search must not vary between runs (the cut rule
    above). Design review 2 ran it on a prototype: the live FAISS generation's 1,708 vectors
    and payloads, written to the review server in walk, ID and shuffled order, gave FAISS's
    top 20 in FAISS's order for all 25 questions, and the same rendered contexts. Every
    payload came back equal, with `page` still an int and `page_label` a str.
  - **Every later story repeats it** (design review 1). The fingerprint does not hash code,
    and S3-3 to S3-10 change code on the retrieval path, so each diffs a new contexts file
    against S3-2's Qdrant one: about 45 s.
  - **If a near-tie differs,** the story records which question and why, and S3-11b's planned
    re-run covers it. No re-run is added.
- **The quickstart changes** (the maintainer's addition).
  - "Install Ollama and go" becomes "start a Qdrant container too". `docker compose up -d
    qdrant` starts one service, on 127.0.0.1:6333, with a named volume and
    `restart: unless-stopped`, before the first `rag-ingest`.
  - The README's requirements gain Docker, or Qdrant's static binary on a host without it.
  - **Local mode with `path=` is not offered.** `qdrant-client` takes an exclusive,
    non-blocking `portalocker` lock on the folder (checked in `qdrant_local.py`), and
    `rag-serve` and `rag-ingest` run side by side. `":memory:"` is for tests only.
  - Every command checks Qdrant before anything else: one message, exit 2, and the hint
    `Is Qdrant running? docker compose up -d qdrant`.
- **ISS-16 closes in S3-3**, with FAISS.

Rejected:

- **pgvector:** no native hybrid search (BM25 needs ParadeDB's `pg_search` or `tsvector`
  ranking, and RRF written by hand), no in-process mode for the hermetic suite, and no
  Postgres in the deployment, which was DEC-3's condition for it.
- **`langchain-qdrant`:** above.
- **Updating the live collection in place:** a reader could see half an update, since
  Qdrant has no transaction across operations.
- **Local mode with `path=` for real use:** the one-process lock.
- **HNSW by default:** approximate, so tier 1 would vary, for no latency the corpus needs.

**DEC-20 — Hybrid retrieval. Our own BM25 encoder writes the sparse vectors, our code fuses
the dense and sparse candidates with RRF, and the cross-encoder ranks what survives.**

- **Where the sparse vectors come from** (the maintainer asked for this to be explicit):
  `rag_qa/sparse.py`, a `Bm25Encoder` of about 80 lines with no new dependency.
  - **Tokens:** lowercase Unicode word tokens, minus a fixed English stopword list, with no
    stemming in v1.0. Each token becomes a uint32 index through blake2b, which is stable
    across processes, unlike `hash()`.
  - **Weights:** a document's are `tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))`, and a
    query's are 1 per distinct token. Qdrant applies the IDF itself
    (`SparseVectorParams(modifier=IDF)`, supported in local mode too, checked). That is BM25
    exactly, as Qdrant documents it.
  - **Settings:** `k1`, `b` and `avg_len` are config. `avg_len` is measured once from the
    corpus in S3-5 and then fixed: a corpus statistic that moved with every ingest would
    rewrite every sparse vector.
  - **Versioning:** `SPARSE_VERSION` versions the tokenizer, as `CHUNKING_VERSION` versions
    the chunks.
- **Rejected sources:**
  - **fastembed's `Qdrant/bm25`:** a dependency tree with onnxruntime, and a download from
    Hugging Face on first use, which would be a third model in CI's cache key and in the
    image. Its tokenizer also moves with fastembed's releases.
  - **Qdrant's own BM25.** The open-source server has had it since v1.15.2. But the client
    sends text to the server only with `cloud_inference=True`, which local mode refuses
    (checked), so the hermetic suite could not run it. And its tokenizer changed in v1.15.3
    and v1.19.0: a dependency on the server's version that the fingerprint cannot see.
- **The query: two searches, fused by our code** (design review 2).
  - Dense (`using="dense"`, exact, the filter) and sparse (`using="bm25"`, the filter) run as
    two `query_points` calls, each cut to `P` by DEC-19's cut rule.
  - `vectorstore` fuses the two lists by RRF with `rrf_k`: each point scores the sum of
    `1 / (rrf_k + rank)` over the lists it is in, with one-based ranks. It orders the fused
    points by score, ties by point ID, and cuts them to `reranker_candidates`.
  - The cross-encoder then takes them down to `top_n` 5.
  - **Why not Qdrant's prefetch and fusion.** Design review 1 found that the server sorts
    fused points by score alone, in hash-map order ("Does not break ties"), so it had Qdrant
    return the whole union and let our code make the last cut. But each prefetch still cut
    its own list on the server. Design review 2 found the server decides a tie at such a cut
    by the collection's layout: three insertion orders kept three different sets, and a
    restart changed each. So no cut is left to the server. Two searches cost one more round
    trip, and the hermetic suite now tests the fusion that runs, where local mode
    reimplemented the server's.
  - **Ties are the rule in RRF:** a point at position i of one list scores the same
    whichever list it is in. Our order breaks them by point ID.
  - **`rrf_k` is always explicit,** 60 by default and tuned by tier 1. Qdrant's own default
    was 2 over zero-based ranks (design review 1, caveat 31). It no longer applies, since the
    fusion is ours. `P` defaults to 40 per branch, so fusion has more than 20 candidates to
    choose from.
  - Weights per list are a parameter of our function, unused unless tier 1 shows a need.
- **The retriever spec.** `RetrieverSpec` gains `mode` (absent means dense), `prefetch` and
  `rrf_k`. Unset keys stay out of `kwargs()`, as they do today, so `vector_search`'s dump,
  and with it tier 2's `query` part, do not move. `search_type` narrows to `similarity`: our
  retriever has none of LangChain's other search types, and the real config sets none.
- **Metadata filtering** (FR-3's second SHOULD) is a `sources` filter by corpus-relative
  path.
  - **Over HTTP:** `POST /v1/query {"question": ..., "sources": [...]}`, at most 50 entries,
    each one in the served manifest, else 422.
  - **In the CLI:** `rag-query --source PATH`, repeatable.
  - **In Qdrant:** a `MatchAny` on the indexed `metadata.source`, in each branch's search.
  - It is data, never a component (DEC-18). Tiers 1 and 2 pass none.
- **Tier 1 decides; the batched re-run switches.**
  - **S3-5** compares dense with hybrid retrieval, both reranked, with `rrf_k` and prefetch
    variants, on a scratch config and a scratch collection, and records the table and a
    recommendation. The real config keeps `vector_search` and **sets no sparse encoder**,
    so neither `query` nor `ingestion` moves.
  - **S3-11b** sets `pipeline.ingestion.sparse_encoder` and switches the retriever together.
    Both parts move, the index rebuilds (its vector configuration changes), and the one
    re-run covers it. `ingestion` leaves an absent key out of its hash, so v0.3's hash holds
    until then.

_As built in S3-5 (2026-10-21):_ the tokenizer also joins a soft hyphen (U+00AD) with the
whitespace after it, since the CADE proceedings mark 823 line-break hyphenations with one, and
the stopwords are NLTK's English list less its 26 forms with an apostrophe (153). `avg_len` is
100.76. A hybrid retriever must state `prefetch` and `rrf_k`, with no default in code, so the
dump tier 2 hashes says everything it does; an explicit `mode: dense` dumps as no mode. A query
with no tokens runs the dense search alone. Tier 1 recommends hybrid on with `rrf_k` 60 and
`prefetch` 40, and `min_score` −4 (S3-5's story, "Tier 1"); S3-11 decides.

**DEC-21 — Network calls get timeouts, retries where a retry is safe, and circuit breakers;
Ollama gets an opt-in hosted fallback.** This resolves DEC-9, and NFR-6.

- **Scope: network calls only,** to Ollama, Qdrant and the fallback. The embedder and the
  reranker run in this process; there is nothing to time out or retry.
- **Timeouts.**
  - Ollama, through the llm spec's `client_kwargs`, which are runtime keys and not hashed:
    5 s to connect, 600 s to read. The read must outlast a silent prefill, which took 250 s
    on 2 threads (caveat 3).
  - Qdrant: 10 s.
- **Retries,** with tenacity 9.1.4, jittered and exponential.
  - **Qdrant:** every read and write, with what a retry counts as done stated per call
    (design review 1). A create that finds the collection already there under our unique
    name has succeeded, and so has a delete that finds nothing. A snapshot is never retried:
    a repeat would leave a second one on the server, so S3-10 removes any it made.
  - **Ollama:** only before the first token, only on a connection error, 2 attempts. A retry
    after a token would duplicate text. The loop is written by hand in `stream_answer`:
    tenacity retries a call, and the model's stream is an async generator.
  - **A bounded budget inside the retrieval lock** (design review 2, from the code).
    `serialized` holds the retrieval lock for a whole retrieval (`api/state.py`), and a hung
    Qdrant costs a timeout per attempt. So the breaker wraps each attempt, not the retried
    call, and an open breaker ends the retry loop. One deadline covers a retrieval's
    attempts, 15 s by default. A 404 for a generation is never retried: the generation is
    gone, and the reader resolves the alias once more (DEC-19).
  - tenacity's async retry awaits in the caller's task and creates none (checked, design
    review 2).
- **Circuit breakers,** in `rag_qa/resilience.py`: about 60 lines of our own, since
  pybreaker is synchronous only.
  - One per dependency. It opens after N consecutive failures, and lets one probe through
    after T seconds (half-open). It is thread-safe and tested on a fake clock.
  - **While it is open,** the API answers 503 with `Retry-After`, as a dependency before the
    first byte (DEC-18's rule), and a start-up check exits 2 naming the service. With a
    fallback configured, Ollama's breaker is the exception (below).
  - It absorbs the backlog's **back-off after a failed reopen:** an open Qdrant breaker
    short-circuits `Serving.refresh()`.
- **The LLM gateway and the hosted fallback** (maintainer: an opt-in seam, plus one real
  check).
  - **Config:** `pipeline.query.fallback_llm` names an llms entry built on
    `langchain_openai.ChatOpenAI` with a `base_url`. Any OpenAI-compatible provider works;
    the maintainer names one when S3-6 starts. The entry sets `max_retries: 0` and a
    `timeout`. Otherwise the OpenAI SDK retries twice on its own, with a 600 s timeout
    (`openai/_constants.py`, design review 2), which would hide failures from the
    fallback's breaker for minutes.
  - **Dependency:** a `fallback` extra, `langchain-openai>=1`. Version 1.6.2 is locked
    already, through ragas, so nothing new resolves. The allowlist gains
    `langchain_openai.`: it is used, so it is allowed (ADR-020).
  - **The key comes from the environment only,** in the provider's own variable. An
    `api_key` key in any component spec is refused at load, as `trust_remote_code` is.
  - **When it answers:** only while Ollama's breaker is open, or after Ollama failed with a
    connection error before its first token, once the retries are spent.
  - **With a fallback configured, Ollama down is not a refusal** (design review 1). The first
    draft also had the API answer 503 while Ollama's breaker was open, and `rag-query` exit 2
    when Ollama did not answer at start-up (`preflight`): the very case the fallback is for,
    so the one real check below could not run. Now an open Ollama breaker routes the answer
    to the fallback, and the 503 comes only when the fallback's breaker is open too. The
    start-up checks warn about Ollama instead of exiting 2, and the primary is built without
    `validate_model_on_init`, a runtime key, so nothing hashed moves. Without a fallback,
    every rule above stands.
  - **How it streams:** `stream_answer` streams the fallback model itself, in the consuming
    task. Never `.with_fallbacks()`, whose wrapper would put back the Runnable machinery
    DEC-14 removed from the stream. Checked in design review 2: langchain-openai 1.6.2 reads
    the stream inside `async with`, so a cancel closes it. Its per-chunk timeout (120 s)
    awaits through `asyncio.wait_for`, which on Python 3.12 runs in the caller's task.
  - **Visibility:** the `done` event's `model` names the model that answered, and so does
    the server log. The README says that retrieved chunks leave the host only then.
  - **Tier 2:** `rag-eval generate` builds no fallback, and the fallback's spec is not
    hashed. Tier 2 measures the model it names.
  - **The one real check** (S3-6): with the maintainer's key set in their shell and Ollama
    stopped, `rag-query` answers from the fallback, and a dropped stream still closes its
    connection.

Rejected: `.with_fallbacks()`; retrying after the first token; pybreaker; retries around
models that run in process.

_As built in S3-6 (2026-10-22), from its approved plan (the story's "Plan", D1–D13):_

- **The key is `RAG_FALLBACK_API_KEY`** (D1, the maintainer's proposal), which amends "the
  provider's own variable" above. `settings.FallbackSettings` reads it, and our code passes
  it to `ChatOpenAI`. Without an explicit `api_key`, langchain-openai 1.6.2 reads
  `OPENAI_API_KEY`, or a LangSmith gateway key, never `GROQ_API_KEY`, and would send it to
  the entry's `base_url`. A fallback set without the variable exits 2. A gitignored `.env`,
  loaded by the shell, may hold it, and `.env.example` names it.
- **Breakers belong to their clients** (D2): `QdrantStore.breaker`, and the pipeline's
  `llm_breaker` and `fallback_breaker`. A breaker opens after 3 consecutive failed attempts
  and probes after 30 s, one probe at a time. While the probe is out the API refuses other
  questions as it does while open, with `Retry-After: 5`, before any retrieval
  (`CircuitBreaker.refusal_s`; S3-6's first review). A failure is a connection error, a
  timeout, a 429 or a 5xx; any other answer counts as a success, and a cancel as no verdict.
  An error raised by the fallback's stream is marked as its own, since its HTTP client's
  errors look like Ollama's (`resilience.mark_fallback`).
  `resilience.classify` matches errors by class name and module, for httpx, httpx2 (openai
  3.3.0 is built on it), openai, ollama and the builtins, without importing them (D5).
- **Qdrant:** each request is tried up to 3 times through tenacity, with a full-jitter
  back-off of `U(0, min(4, 0.5·2^(n−1)))` s. What a repeat counts as done was checked on
  1.19.2: a create that finds our collection listed; a delete answering `False`; a repeat of
  the alias switch, which is a no-op. The read-back after a failed switch is retried but
  never refused by the breaker, since the switch's own failures may just have opened it
  (found in S3-6). The retrieval deadline is the store's `deadline_s: 15` (D6), and a search
  inside it asks for what is left as its own timeout, which qdrant-client 1.19.1 also
  bounds the HTTP request by.
- **Ollama's timeouts** are `client_kwargs.timeout: {connect: 5, read: 600, write: 30,
  pool: 30}`. A mapping of the four phases becomes httpx's `Timeout`, or openai's for a
  `langchain_openai` model, and the schema checks it at load (D7).
- **The fallback entry** is `components.llms.groq_fallback`, Groq's `openai/gpt-oss-120b`,
  with `reasoning_effort: low`, `extra_body: {include_reasoning: false}`,
  `max_completion_tokens: 1024`, `max_retries: 0` and `timeout: {connect: 5, read: 60,
  write: 30, pool: 30}`. Once `fallback_llm` names an entry, the load requires another
  entry than `llm`, on `ChatOpenAI`, with a `base_url`, `max_retries: 0` and a `timeout`
  (D8).
- **It gets an HTTP client of its own** (D9): langchain-openai caches one default async
  client per base URL and timeout, shared by every instance, so closing one would close all.
- **`Done.model` is the config reference of the model that answered** (D3).
  `rag-eval generate` runs the start-up check, so a stopped Ollama exits 2 rather than
  with a traceback, and it builds no fallback (D10).
- **A component key named `api_key` or `*_api_key` is refused at load** (D11), which is a
  contract change. The S3-3 read timeout with Qdrant up is absorbed by the retries, and an
  exhausted one now says "did not answer in time" (D13).

**DEC-22 — Observability: structured logs through structlog, OpenTelemetry spans we create
ourselves, Prometheus metrics, and burn-rate alerts** (NFR-10).

- **Logs.** structlog 26.1.0 formats every stdlib log record, ours and uvicorn's, through
  `ProcessorFormatter`, so existing `logging.getLogger(__name__)` calls stay.
  - JSON when the output is not a terminal, or with `RAG_LOG_FORMAT=json`; console lines
    otherwise. `rag-ingest`'s progress stays readable.
  - The request ID, the trace ID and the API key's name are bound per request.
  - pypdf's logger is set to ERROR for the whole process, and the remaining plural slips are
    fixed: two backlog lines that waited for this move.
- **Traces.** opentelemetry-sdk 1.45.1, the same release as the `opentelemetry-api` FastAPI
  already brings, exported over OTLP/HTTP. Off unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
  - **The spans:**
    - the HTTP server span, from `opentelemetry-instrumentation-fastapi` 0.66b1, with its
      `send` and `receive` spans excluded, so SSE events are not spans, and with the health
      routes left out;
    - `rag.retrieve`, with children `rag.embed_query`, `rag.vector_search`
      (`db.system.name` qdrant) and `rag.rerank`;
    - `chat {model}`, with the GenAI attributes taken from the semantic-conventions
      package's own constants: the operation, the provider, the request model, temperature
      and maximum tokens, and the input and output tokens from the last chunk's
      `usage_metadata`. An event marks the first token;
    - `rag.ingest`, with children for the scan, prepare, embed and publish steps.

    No prompt or answer text goes into a span.
  - **No LangChain callback handlers,** neither OpenLLMetry nor Langfuse's.
    `answering.py` records why: a handler that suspends in `on_llm_new_token` reopens
    DEC-14's cancel window.
  - **Spans that survive a cancel.** No span is made current across a `yield` in
    `stream_answer`: a context attached in one task and detached in another fails. Each
    span takes an explicit parent and ends in a `finally`, marked as cancelled when it was.
    The context is copied into executor threads by hand, because `Serving`'s `serialized`
    wrapper calls `run_in_executor`, which does not copy context variables. The real-uvicorn
    disconnect tests run with tracing on.
  - **Checked in design review 2:**
    - a span made current across a `yield`, then closed from another task, logged "Failed
      to detach context"; one with an explicit parent logged nothing;
    - `run_in_executor` did not copy context, while `copy_context().run` did;
    - with `exclude_spans`, the ASGI middleware hands Starlette the raw `receive`
      (`opentelemetry-instrumentation-asgi` 0.66b1, `_get_otel_receive`);
    - a client dropped during 30 s of silence cancelled the stream within milliseconds, with
      the instrumentation off, excluded or full, and every span ended.
- **Metrics.** prometheus-client 0.26.0, served on a port of its own
  (`RAG_API_METRICS_PORT`, off unless set).
  - Histograms: `rag_retrieval_seconds` (NFR-1: the query embedding plus the search),
    `rag_rerank_seconds`, `rag_sources_seconds` (NFR-2a), `rag_ttft_seconds` (NFR-2b),
    `rag_answer_seconds`, and `gen_ai_client_token_usage`.
  - Counters and gauges: answers by outcome, 503s and 429s (S3-8 adds the 429s), breaker
    states, ingest runs and chunks, and the documents served. Ingest metrics come from the
    API's jobs: a `rag-ingest` run has no listener, and reports in its log.
  - Not the OpenTelemetry metrics SDK: its Prometheus exporter is still a 0.x beta, and
    Prometheus scraping is what the SRS names.
- **Alerts.** `deploy/prometheus/rules.yml` holds multi-window burn-rate alerts, as in the
  SRE workbook, on availability, NFR-1, NFR-2a and NFR-2b, a breaker open for more than
  5 minutes, a failed ingest and a scrape target down. CI tests them with
  `promtool test rules`. Alertmanager v0.34.1 ships with a null receiver and an email
  example.
- **The trace backend is Jaeger** v2.22.0: one container, OTLP in, in-memory storage.
  Rejected: Langfuse, which needs Postgres, ClickHouse, Redis and object storage on a
  15 GB host; Phoenix, another Python service.
- **NFR-1.** `scripts/bench_retrieval.py` fills a scratch collection on the real server with
  100k synthetic unit vectors, plus sparse ones, and measures the query embedding plus an
  exact search, dense and hybrid, over 200 queries, at k=20: the reranker's candidates,
  stricter than NFR-1's 10 (design review 1). The target is under 300 ms at p95; the README
  records it under "Capacity planning". Synthetic vectors are honest here: a search's cost
  does not depend on what the vectors mean.
- **NFR-2a.** The same script times the real retrieval half, the rerank included, over the 25
  golden questions with the model idle, and reports the p95 that DEC-27 (a) sets a target for
  (design review 1).
- **NFR-3.** `rag-ingest` reports pages per second, chunks embedded per second and points
  written per second, and the README keeps a table.

_As built in S3-7 (2026-10-23), from its approved plan (the story's "Plan", D1–D12):_

- **Split in two** (the maintainer's approval of the plan). S3-7 built the logs, the metrics,
  the token counts, the benchmark and NFR-3's report. The traces, the FastAPI
  instrumentation, the alert rules, Jaeger and the `deploy-config` job are S3-7b's, on the
  design above. Only the spans touch DEC-14's cancel guarantee; this half adds no `await` to
  the answer path.
- **Logs** (`observability/logs.py`):
  - **One root handler, on stderr,** looked up at each record. `RAG_LOG_FORMAT` is `json` or
    `console`, and another value exits 2. Every log line goes to stderr: `rag-ingest`'s
    progress moved there from stdout, a contract change. Results stay on stdout.
  - **A traceback is plain text in every format** (`format_exc_info`, and `plain_traceback`
    for the console). structlog's dictionary form and its console renderer, which uses rich
    when it is installed, record each frame's local variables by default, which could log a
    question or a key. The console path did, until S3-7's first review.
  - **Levels:** the root at WARNING, `rag_qa` at INFO, and pypdf at ERROR; `rag-eval` logs
    warnings only, which keeps the corpus scan's lines out of its tables. `rag-serve` gives
    uvicorn `log_config=None, log_level="info"`.
  - **Bound fields:** `request_id`, from an `async` first dependency (FastAPI runs a sync one
    in a thread, whose context is lost), and `job_id`, in the ingest job's thread. The trace
    ID comes in S3-7b, the key's name in S3-8.
  - **One INFO line per complete answer,** from the route and from `rag-query`: the model,
    the times and the token counts as fields, never a question's or an answer's text.
- **Metrics** (`observability/metrics.py`), on a registry of their own with the `_created`
  series off, and every label set started at 0:
  - **the names in the list above,** plus `rag_answers_total{outcome}`,
    `rag_fallback_answers_total`, `rag_refusals_total{code, reason}`,
    `rag_ingest_runs_total{status}`, `rag_ingest_chunks_total`,
    `rag_breaker_state{breaker, state}` and `rag_index_documents`; the last two are read at
    scrape time;
  - **the listener** is on the API's own host. The API's port, or one in use, exits 2;
  - **an answer's outcome is counted once, when its slot is released.** A client dropped
    between two events leaves the route's generator paused, and only the garbage collector
    closes it, so a count in its `finally` was missed (the forced-window test);
  - **an answer's times run from its first dependency,** `bind_request`, which notes when the
    request arrived. FastAPI starts the route's body only once every dependency has resolved,
    so timed from there, the alias read before each answer and Qdrant's retries of it went
    unseen (second review: 1.5 s waited, 0.006 s recorded);
  - **the token histogram** takes its name and labels from
    opentelemetry-semantic-conventions 0.66b1, and its buckets from the convention's text
    (semantic conventions 1.37.0), which the package does not carry. Its GenAI constants are
    marked deprecated, moved to the GenAI conventions' own repository, which has no Python
    package yet. Reading a member raises no warning; calling an enum does, so the code reads
    members only.
    `ollama` is a custom provider value, which the convention allows. `model_identity` lives
    in `observability/genai.py` and maps a model by its class's package, so the package
    still loads no model stack.
- **Token counts.**
  - **Read from the last chunk carrying `usage_metadata`.** ChatOllama puts it on its final,
    empty chunk, which `stream_answer` used to skip.
  - **langchain-openai 1.6.2 turns `stream_usage` off for a custom `base_url`,** so
    `groq_fallback` sets `stream_usage: true`; one real call to Groq returned the counts.
    The fallback's spec is not hashed.
  - **Tier 2's `AnswerItem` gains both fields, left out of the dump while unset**
    (`exclude_if`). So the committed answers keep their identity, and `rag-eval check` its
    exit 0, with no re-run. A plain `None` default would have changed it, so a test pins the
    committed file's identity to the committed scores.
- **Ingest.** `IngestReport` gains `load_s`, `embed_s`, `write_s` and `points`. The summary
  ends with the rates, and the "Published" line carries them as fields. The API's job report
  carries them too, as its mirror test requires. In JSON mode the embedder shows no progress
  bar; the model loader's own `Loading weights` bar still prints between JSON lines unless
  `HF_HUB_DISABLE_PROGRESS_BARS=1` (first review).
- **Measured** (README, "Capacity planning"; this laptop, 2026-10-23):
  - **NFR-1 held:** at 100k points and k=20, a p95 of 29.7 ms dense and 38.0 ms hybrid.
  - **NFR-2a missed, narrowly:** a p95 of 2.09 s, then 2.02 s on a second run. The rerank is
    nearly all of it, and its levers (fewer candidates, a smaller cross-encoder) move hashed
    inputs. The maintainer gave them to S3-11's stage 2 at S3-7's close-out (2026-10-25),
    measured by tier 1 and this benchmark; S3-11b times what ships; and the amend session
    (S3-12) sets DEC-27 (a) from that number only if it still misses.
  - **NFR-3:** 16.2 pages/s loaded, 24.2 chunks/s embedded, 2,695 points/s written.
- **The benchmark deletes only a collection it created** (found in its own run). Two runs
  started seconds apart both passed the check for an existing collection. The second's
  create then failed with a 409, and its cleanup deleted the first's collection mid-timing.

_As built in S3-7b (2026-10-25), from its approved plan (the story's "Plan", D1–D13):_

- **Tracing** (`observability/tracing.py`) is on only when a command calls `configure()` and
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set: `rag-serve`, `rag-query` and `rag-ingest`, never
  `rag-eval`. A provider of our own, never the global one, with a batch processor and the
  OTLP/HTTP exporter, which reads the SDK's own variables; our code imports them only then. The
  shared code, retrieval and the answer stream, gets no-op spans otherwise, whatever the
  environment holds. Flushed in the API's lifespan (a SIGTERM ends `rag-serve` before `main`'s
  `finally`), shut down at each command's end. A malformed endpoint exits 2.
- **Scrubbed at the exporter** (`observability/scrub.py`, S3-7b's first review). Our own spans
  record a failure as its class name (`error.type`), but the contrib server span records an
  unhandled exception itself, message and stacktrace, in two places (its FastAPI wrapper and its
  ASGI middleware), and its FastAPI wrapper writes `"Type: message"` into the status
  description. It has no switch to stop. Traces leave this host and logs do not, and DEC-18
  keeps host paths off HTTP, so `ScrubbingExporter` wraps the OTLP one: each `exception` event
  keeps `exception.type` alone, and the status keeps its code and loses its description. It
  covers every span, an instrumentation added later included. `SpanLimits(max_event_attributes=0)`
  would also drop `exception.type`, and any attribute an event gains. In the probe the 500
  response's status replaced the description; the scrub covers the cases it does not.
- **The spans**, as designed above: `rag.retrieve` in `open_retrieval`'s half, its children in
  `QdrantRetriever` and `_bound`; one `chat {model}` per attempt in `stream_answer`, started
  after the breaker allows it, with an explicit parent and never current, ended in a nested
  `finally`; `rag.query` per `rag-query` question; `rag.ingest` and its four steps over what
  `load_s`, `embed_s` and `write_s` time. `serialized` runs its executor thread in a copy of the
  request's context; an ingest job attaches its request's in its thread, so its run is in the
  `POST /v1/ingest` trace. Log lines carry `trace_id` and `span_id` under a span.
- **FastAPI 0.143 has OpenTelemetry of its own** (found in S3-7b's plan): server spans,
  operation spans, OTel logs and metrics, switched on whenever a provider is configured,
  a global one included. It ends a dropped client's server span as an error
  (`ConnectionError`), where a cancel here leaves the status unset, and it wraps `receive`. The
  maintainer kept the contrib instrumentation; `create_app` passes `telemetry` with `tracing`,
  `metrics`, `logs` and `auto_configure` false, and a test shows one server span per request
  with a global provider set, against a control app that makes its own.
- **Python reads `OTEL_EXPORTER_OTLP_TIMEOUT` in seconds,** not the specification's
  milliseconds. A dead collector costs at most that at exit, with the exporter's warnings.
- **langsmith 0.12, through langchain-core, imports the SDK's `TracerProvider`** whenever the
  SDK is installed, as a type: every command loads `opentelemetry.sdk.trace` (about 20 ms) and
  configures nothing. The architecture test holds the exporter and the instrumentation out
  instead, and no command sets a global provider.
- **The cancel guarantee, with tracing on:** DEC-14's forced-cancel tests and every real-uvicorn
  drop pass with the instrumentation wrapping the app, every span that starts ends, and no
  context fails to detach. A mutant that makes the chat span current across the stream's
  yields fails the forced window's tests with "Failed to detach context".
- **The alerts** (`deploy/prometheus/rules.yml`): recording rules for the four SLIs over seven
  windows, eight burn-rate alerts, `RagBreakerOpen` (a ticket for the model's while the hosted
  fallback answers; only while the breaker keeps opening again, since an idle server leaves a
  model breaker half-open after its outage, with the model back, and readiness never probes
  it: second review), `RagIngestFailed`, `RagTargetDown`, and `RagTargetMissing`: compose's
  replicas are found by DNS, so an exited one leaves the targets and its `up` disappears. The
  tests run with a 5-minute evaluation interval, about 20 s; at one minute they took four.
- **The rules' tests** are two files: `rules.test.yml` evaluates every 5 minutes, and
  `rules.causes.test.yml` every minute, since at 5 minutes `for: 1m` and `for: 5m` behave the
  same. A pytest holds the 28 recording rules to one expression per SLI across the seven
  windows.
- **The `deploy-config` job** runs `promtool check config`, `promtool test rules` and `amtool
  check-config`, from the images DEC-24 pins.

_Fixed after S3-7b's merge (2026-10-26, `fix/s3-7b-abandoned-answers`):_

- **An answer abandoned before its first token was invisible to every alert.** The
  availability SLI leaves cancelled answers out, `rag_ttft_seconds` is observed only at a
  first token, and a cancel teaches the breaker nothing (`tried.abandoned()`), so a hung model
  whose clients leave before Ollama's 600 s read timeout moved no SLI: a promtool probe of the
  merged rules kept every alert quiet through two hours of it.
- **The fix:** `rag_abandoned_seconds{waiting_for}`, how long such an answer had waited, and
  for what (`sources`, or `first_token` after them), observed in `Slot.release()`, the one
  place every ending passes once, with no new `await`. The `sources` and first-token SLIs
  count those past their objective as slow, since their event would have come later still,
  and leave younger ones out of both sides: the event is unknown, and a client that leaves
  at 30 s has not missed a 300 s objective. The first-token SLI reads both stages. Answers
  refused before their stream began stay out, as in `rag_answers_total`.

**DEC-23 — The API beyond localhost: named, scoped keys, rate limits, a body-size limit,
durable job records and a stateless query tier** (NFR-5; DEC-18's Phase 3 half).

- **Keys** (maintainer).
  - **The file,** named by `RAG_API_KEYS_FILE`, is JSON, one entry per key: `name`,
    `scopes` (some of `query` and `ingest`), the token's `sha256`, and `created`. It is
    written atomically, read again within 5 s of a change, and never holds a token. A
    32-byte random token needs no slow hash.
  - **Where it lives, and how a change is seen** (design review 2).
    - It lives in `var/`, which compose mounts as a folder. A file bind-mounted on its own
      keeps its old inode when the host replaces it atomically, so a revocation would never
      reach the container (checked).
    - A change is seen when `(st_ino, st_mtime_ns, st_size)` moves, and `rag-keys` takes a
      lock around its read and rewrite, so two at once lose nothing.
    - A file that does not parse fails closed: its keys are refused with a 503, and health
      says why, until it parses again. `RAG_API_TOKEN` still works.
  - **The command:** `rag-keys add NAME --scope query,ingest` prints a new token once;
    `rag-keys list` and `rag-keys revoke NAME` complete it.
  - **The check** hashes the presented token and compares it with each stored hash through
    `hmac.compare_digest`, on bytes. A wrong token meets every hash, so wrong tokens all
    take the same time; an early exit on a match tells only that key's own holder where it
    sits. 10,000 keys take 0.36 ms (design review 2).
  - `RAG_API_TOKEN` keeps working, as one key named `token` with both scopes.
  - **"No credential" means neither a token nor a keys file** (design review 2). DEC-18's
    Host check and `rag-serve`'s refusal to listen beyond loopback both test
    `settings.token is None` today (`api/routes.py`). With a keys file and no token, the
    Host check would refuse every request through Caddy under a public name, and Caddy's own
    health probe, whose `Host` is `app:8000`, while S3-9's `https://localhost` check passed.
    Both rules move to "no key and no token".
  - **Scopes:** `query` for `POST /v1/query` and `GET /v1/documents`; `ingest` for the two
    ingest routes. The docs and the schema need any valid key; health stays open; metrics
    have their own port. A valid key without the scope gets 403.
  - Logs and spans carry the key's name, never the token.
- **Rate limits:** token buckets of our own, in memory, per replica.
  - Per key: `RAG_API_RATE_LIMIT`, 60 a minute by default.
  - Per client IP, for requests without a key (a loopback server with none configured), and
    a strict per-IP bucket for failed authentication, 10 a minute, which slows token
    guessing. That bucket gates only requests that fail authentication: a valid key is
    never refused because someone else failed from the same address (design review 2).
  - **The health routes are exempt** (design review 1). Docker's `HEALTHCHECK` and Caddy's
    active checks poll them from one address, and a 429 there would mark a replica unhealthy.
    The 5 s single-flight probe cache already bounds their cost.
  - **The tables are bounded** (design review 2): at most 10,000 buckets each, full ones
    evicted first, then the least recently used, and an IPv6 client counted by its /64. A
    bucket costs 180 bytes, and one /64 holds 2^64 addresses. Unbounded, the tables grew by
    36 MB per 200,000 addresses (measured).
  - The answer is 429 with `Retry-After`.
- **The client's address** (design review 2).
  - uvicorn 0.54.0 reads `X-Forwarded-For` by default, and trusts it from `127.0.0.1` and
    `::1` (its `config.py`). So today any local client can name its own address: `curl -H
    'X-Forwarded-For: 1.2.3.4'` was seen as `1.2.3.4`, a fresh bucket per request.
  - So `rag-serve` passes `forwarded_allow_ips` itself: `RAG_API_TRUSTED_PROXIES`, or no one
    when it is unset.
  - In the stack that is Caddy's own address, which compose fixes, never the network's range.
    A published port's connections arrive from the network's gateway, inside that range
    (`peer=172.18.0.1`, checked), and every container sits in it too. So the app's port is
    never published.
  - Caddy itself replaces a client's `X-Forwarded-For` with the real address (checked).
- **Body size.** A pure ASGI middleware answers 413 to a body over 64 KiB, by its
  `Content-Length` or as it streams, before FastAPI parses anything. That closes the
  pre-authentication gap S2-7's first review found.
  - **As it streams, it raises `HTTPException(413)`** from the `receive` it wraps (design
    review 2). FastAPI turns any other exception raised while it reads a body into a 400,
    "There was an error parsing the body" (`fastapi/routing.py`). Checked with a chunked
    body that had no `Content-Length`: our own exception gave 400, `HTTPException` gave 413.
  - It passes `http.disconnect` through untouched, and the real-uvicorn disconnect test runs
    with it on: it wraps `receive`, as the instrumentation can.
- **Health.** Each dependency's probe is cached for 5 s and runs single-flight, so a flood of
  probes costs at most one call per dependency per 5 s. `/v1/health` reports the breakers.
- **Durable job records** in SQLite (the standard library), at `var/jobs.sqlite3`, in WAL
  mode.
  - **One job at a time,** across replicas: a `BEGIN IMMEDIATE` transaction checks and
    inserts, so the 409 holds whichever replica answers.
  - **An abandoned job is told by a lock of its own** (design review 1). The replica takes a
    `flock` on `var/jobs/<id>.lock` before it inserts the job's row, and holds it until the
    row's final status is written; the row reads "running" only once `ingest()` holds the
    writer lock. The kernel releases a `flock` with its process, so an unfinished row whose
    job lock is free belongs to a process that died. It reads as `failed`, with
    `abandoned: true`, and ADR-024's recovery handles what the run left.
  - **In this order** (design review 2): read the row, probe the job lock, and, when it is
    free, read the row again; only a row still unfinished is abandoned. Without the second
    read, a job that finished between the two steps read as abandoned in 92 of 400 runs of a
    simulation, and in none with it. A verdict written back uses `UPDATE … WHERE status IN
    ('queued', 'running')`, so it never overwrites a final status. The start transaction
    applies the same rule, so a dead job's row never holds the 409.
  - **Not the writer lock.** The first draft probed the ingest `flock` itself. A row is
    written before its thread takes that lock, so a live job could read as abandoned, and a
    probe that took the lock to test it would make a writer starting at that moment exit 2.
  - The last 100 jobs are kept, and any replica answers for any of them.
- **A stateless query tier** (NFR-5). The index lives in Qdrant, resolved through the alias
  on each request; job records live in SQLite and keys in a file. What a replica caches
  (the served generation and its manifest, probe results, rate buckets) can be rebuilt from
  those. Documented limits: ingestion runs on one host; the slots and the rate limits are
  per replica; and one Ollama queues every replica's answers, so scaling the query tier
  pays off when the model backend scales.
- **TLS** terminates at Caddy (DEC-24). `rag-serve` stays plain HTTP, and its cleartext
  warning now points to the compose recipe.

Rejected: OIDC or JWT (an identity provider to run); slowapi (an unmaintained wrapper around
what is 30 lines here); Redis or Postgres for job records (a service for single-flight rows);
a separate ingest worker (on one host there is no CPU to isolate it on).

_As built in S3-8 (2026-10-26), from its approved plan (the story's "Plan", D1–D13):_

- **The keys** (`api/auth.py`, `api/keys_cli.py`, `rag-keys`). The file is
  `{"version": 1, "keys": [...]}`; a name is `[a-z0-9][a-z0-9_-]{0,63}`, and `token` is
  reserved for `RAG_API_TOKEN`. `KeyRing` looks at the file at most every 5 s and reads it
  again when its signature moved; a file that cannot be used holds no key, its problem is fixed
  text (never the path), keys get 503 with `Retry-After: 5`, and readiness is 503 with a `keys`
  entry (the maintainer's choice at the plan). A wrong token presented while the file is broken
  draws on the failure bucket too, or guesses at `RAG_API_TOKEN` would meet unthrottled 503s.
  `rag-keys` writes under a `flock` on `<file>.lock`, through a 0600 temporary file and
  `os.replace`, the folder fsynced; it loads no web framework and no model stack.
- **The dependency order on a route:** Origin 403; Host 403 (no credential only);
  `authenticate` (401, 429 after 10 failures a minute from the address, 503 for a broken
  file, then the key's own 429); the route's `scope` (403); the query route's own refusals.
  FastAPI still parses JSON before any dependency, so a malformed body's 422 comes first, now
  over at most 64 KiB, and draws on no bucket.
- **The limits** (`api/limits.py`): `Buckets` holds `[tokens, when]` in an `OrderedDict` in
  recency order; a full table scans from its least recently used end for a full bucket, which
  under a flood of new addresses is the first it looks at. They are used on the event loop
  alone, so they take no lock. The 429's `Retry-After` is the wait for one token.
- **The client's address:** `server.uvicorn_config` always passes `forwarded_allow_ips`, so
  uvicorn's `FORWARDED_ALLOW_IPS` default never applies; `RAG_API_TRUSTED_PROXIES` takes IP
  addresses only.
- **The body limit and the 500s:** two pure ASGI middlewares, `BodyLimit` outermost, then
  `CountServerErrors`, inside Starlette's error middleware and the contrib instrumentation. An
  unexpected 500 on a question counts as `rag_refusals_total{code="500",reason="error"}`. The
  availability SLI reads `code=~"5.."`: the 503s (a broken keys file's included) and the 500s
  are bad, and the 401, 403, 413 and 429 refusals, the client's or the limits' policy, count on
  neither side. A pytest ties the rules' selector to `metrics.REFUSAL_CODES`.
- **The job records** (`api/jobs.py`): one table, `synchronous=NORMAL` (a commit survives the
  process, which is the abandoned case; a power cut may lose the last few, never the
  database), a connection per call. A row reads `running` from `ingest()`'s `on_locked`
  callback. Each guard against the race design review 2 measured, the second read and the
  conditional verdict, is enough alone; both are kept, and a test removes them together. An
  abandoned job counts once in `rag_ingest_runs_total{status="failed"}`, by the replica whose
  write changed its row, so `RagIngestFailed` fires after a crash mid-ingest too. The last 100
  jobs are kept.
- **The health probes:** `state.Probe`, a shared task behind `asyncio.shield`, cached for 5 s;
  an exception is not cached. Readiness reads the alias and the model's server through one
  each; a question never does.
- **NFR-5's test** (`tests/test_api_replicas.py`): two apps over one in-memory Qdrant and one
  `var/`, and a third started afterwards that rebuilds the same view from the shared state.

**DEC-24 — One image, one compose file, and scanning in CI** (SRS §11; §8.3's overlays).

- **The image,** built in stages.
  - **Base:** `debian:trixie-slim`, with uv 0.11.8 and the uv-managed CPython 3.12.13 that
    development and CI use, every image pinned by digest.
  - **Install:** `uv sync --locked --no-dev --no-install-project`, then the project with
    `--no-editable`. Checked: `--no-dev` selects `torch 2.14.0+cpu` and no `nvidia-*` wheel,
    while `--no-default-groups` would select PyPI's CUDA torch. The build fails unless the
    installed torch is `+cpu`.
  - **Models:** a stage builds the embedder and the reranker from `config.yaml`, as CI's
    warm-up step does, and the runtime runs with `HF_HUB_OFFLINE=1`.
  - **Runtime:** any non-root uid, and a `HEALTHCHECK` on `/v1/health/live`. The image runs
    under whatever uid compose gives it: the models readable by all, nothing written outside
    `var/`.
  - **One uid on both sides** (design review 2). Compose runs `app` and `backup` as
    `user: "${UID}:${GID}"`, both set in `.env`, since neither bash nor zsh exports them.
    Under any other uid (all three checked):
    - the host CLI could not open a lock file a container had created (`PermissionError`);
    - nor write a WAL database a container had created ("attempt to write a readonly
      database");
    - and the container could not read the host's 0600 keys file.

    `_writer_lock` turns such a `PermissionError` into exit 2, naming the file.
- **Signals.** Compose sets `init: true`. With an init process as PID 1, uvicorn's re-raised
  SIGTERM ends `rag-serve` with status 143, the conventional one, which stays: the backlog
  line about mapping it to 0 closes. Without an init, PID 1 ignores the re-raised signal
  and exits 0. S3-9 checks both with `docker stop`.
- **`compose.yaml`**, every image pinned by digest. S3-2 creates it with `qdrant` alone.
  - **`qdrant`:** v1.19.2, on 127.0.0.1, with an API key from `.env`. It is the
    `-unprivileged` image, from S3-2 on: the plain one runs as root (`user=0:0`), and a
    volume that starts unprivileged never holds root's files (design review 2).
  - **`ollama`:** 0.22.1, the version on this host, so ADR-025's digests stay portable, plus
    a one-shot service that pulls the configured model: a second copy, about 4.1 GB for
    mistral, since the container keeps its own models (design review 1).
  - **`app`:** bind-mounts `./var`, `./corpus` (read-only) and the overlay.
  - **`caddy`:** 2.11.7, with `tls internal` for a local name and ACME for a public one. It
    balances the `app` replicas, health-checked on **liveness**, so a stack with no index
    yet still accepts the ingest that builds it. SSE is flushed at once, and the body limit
    applies at the edge too. It has a fixed address on the stack's network, the one address
    the app trusts for `X-Forwarded-For` (DEC-23), and the app publishes no port at all
    (design review 2).
  - **Profiles:** `observability` (Jaeger 2.22.0, Prometheus v3.15.0, Alertmanager v0.34.1)
    and `backup` (DEC-25).
  - **Memory:** about 7.5 GB with mistral loaded. The README says to unload the host's own
    Ollama model first (`ollama stop`), and the same holds the other way: a host `rag-query`
    talks to the host's Ollama, so the stack's model is stopped first.
  - **Qdrant's key in a host shell** (design review 1). Compose reads `.env`, while
    `settings.py` reads only the environment, so a host CLI or a `curl` against the stack's
    Qdrant loads it first: `set -a; . ./.env; set +a` (caveat 32).
- **A disconnect through the proxy.** Caddy cancels its upstream request when the client
  leaves. S3-9 shows against real Ollama that our connection closes within about 1 s of a
  client dropping through Caddy: DEC-14's criterion, one hop further out.
  - **Design review 2's spike** put Caddy 2.11.7 in front of a stand-in for our SSE route
    that went silent for 30 s after `sources`, as a prefill does. Over HTTP/1.1, TLS
    HTTP/1.1 and HTTP/2, the upstream stream was cancelled within milliseconds of the client
    leaving.
  - Caddy's 64 KB limit answered 413 to a chunked body too.
- **Overlays** (§8.3).
  - `RAG_CONFIG_OVERLAY` names a YAML file deep-merged over the base config before it is
    validated: mappings merge, lists and scalars replace.
  - Paths resolve against the base config's folder: one rule.
  - `config.docker.yaml` (committed) sets the addresses inside the compose network: the
    llms' `base_url` (a runtime key) and the store's `url`.
  - Nothing is loaded without being named. An overlay nobody named could change a hashed
    input without saying so.
- **CI.** Every change to `ci.yml` needs gh's `workflow` scope (caveat 13).
  - **The `container` job:** buildx (`docker/setup-buildx-action` v4.4.1 and
    `docker/build-push-action` v7.4.0, by SHA, with GitHub's cache), no push; a smoke run;
    then Trivy v0.75.0 over the image (HIGH and CRITICAL, `--ignore-unfixed`, failing the
    job) and over the repository (`fs --scanners secret,misconfig`).
  - **Trivy runs as its release binary, checked against a sha256 written in `ci.yml`,**
    never through `aquasecurity/trivy-action`. That action's tags were hijacked in
    March 2026 (CVE-2026-33634), when Trivy v0.69.4 and Docker Hub images were malicious
    too. A hash we hold beats a tag, or a checksum file fetched from beside the binary.
  - **Every pin is verified once, and how is recorded beside it** (design review 2). A
    signature alone would not have caught that incident. The attacker made v0.69.4 by
    "Tagging this commit as `v0.69.4`, triggering the release pipeline" (GHSA-69fq-xp46-6x23),
    so it was signed as every release is. What tells it apart is where its commit came from,
    and when. So a Trivy pin needs all four of these, and S3-9 has the commands:
    1. the tarball's sha256 equals GitHub's own asset digest, and the release's checksums
       file;
    2. `gh attestation verify` passes, pinned to the signer workflow
       (`.github/workflows/reusable-release.yaml`), and to the tag's ref and commit;
    3. that commit is an ancestor of `main`;
    4. the release is at least 14 days old, and no advisory names it.

    Design review 2 ran all four for v0.75.0. The provenance was signed by that workflow at
    `refs/tags/v0.75.0`, from `591e979`, which `main` contains; the sha256 is
    `c6e65abddb348e25f10549df887045629cf28cc72453cd1c63acb717316b3f3f`; and a wrong commit
    failed the check. v0.76.0 is too new.
  - **promtool runs from the Prometheus image,** pinned by its index digest, the image
    compose runs, rather than as a second tarball. Prometheus publishes no signatures. But
    Docker Hub and Quay serve the same digest, and the image's `promtool` is byte-identical
    to the release tarball's (both checked, design review 2).
  - **Images are pinned by index digest,** each with the date and its check. The check is a
    second registry serving the same digest, an attestation, or Docker's own provenance. An
    image on Docker Hub alone gets the age rule. As observed on 2026-10-16:

    | Image | Index digest | Checked by |
    | --- | --- | --- |
    | `qdrant/qdrant:v1.19.2-unprivileged` | `sha256:efb96a9425a90d2d5a1a0a474156280df1892dcdf1af3e8515bd2589b1bfd88b` | Docker Hub only: the age rule |
    | `ollama/ollama:0.22.1` | `sha256:3ca37ec2b9cb6341b62554074205c616778fe98abcf9e4fc50361b79a07407ae` | Docker Hub only: the age rule |
    | `caddy:2.11.7` | `sha256:685ae3d7c34d9b5f8caeddc5d14051db7bbd88ff1f5a2fa0c16bd8ca328fb7af` | Docker's SLSA provenance |
    | `debian:trixie-slim` | `sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f` | a Docker Official Image, as Caddy is, so its provenance is checked when it is pinned; a moving tag, bumped deliberately |
    | `ghcr.io/astral-sh/uv:0.11.8` | `sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347` | the same on Docker Hub; GitHub attestations, signed from `main` |
    | `prom/prometheus:v3.15.0` | `sha256:efd719c99d83b060d9daefdcf00360461adf279f45ef5391f8d111892118753e` | the same on Quay |
    | `prom/alertmanager:v0.34.1` | `sha256:e9733bafb1bdef9b00e25a21f8f99dc26a22224bf16641ad754d1649f4c3357a` | the same on Quay |
    | `jaegertracing/jaeger:2.22.0` | `sha256:836b967b5abcf1f7758066aa6b348619267a2f1459f72d472840570f9bd88a17` | the same on Quay |

    A later pin of the same immutable tag that shows another digest is investigated, not
    accepted.
  - **The `deploy-config` job:** `promtool test rules`, `promtool check config` and
    `docker compose config -q`, with promtool from the Prometheus image pinned above. S3-7b
    built it with promtool's two and `amtool check-config`, from the Alertmanager image pinned
    above (the maintainer, at its plan); S3-9 adds the compose check.
  - **The audit policy** (a backlog line, decided here):
    - `pip-audit --ignore-vuln ID` only for an advisory with no fixed release, with a
      reason and a date beside it in `ci.yml`, and a line in STATUS.md;
      _One exception, taken by the maintainer on 2026-10-22 (S3-5; STATUS.md caveat 37):_
      five pypdf 6.19.0 advisories with fixed releases are ignored, because those releases
      extract the corpus differently and change 7 of the 25 golden contexts. The bump rides
      S3-11b's one re-run with `CHUNKING_VERSION` 2. The step fails on its own after
      2027-01-20, 90 days out, as Trivy's ignores expire. _Extended by the maintainer on
      2026-10-23 (S3-6's CI) to three more pypdf advisories published on 2026-10-22, for
      the same reasons and to the same end: eight in all;_
    - Trivy's ignores in `.trivyignore.yaml`, each with a statement and an `expired_at` at
      most 90 days out;
    - a weekly scheduled run of the whole workflow on `main`, so an advisory surfaces
      without a pull request. It also runs `trivy image` over every pinned third-party
      image, under the same policy: a fixed advisory there means a pin bump (design
      review 2);
    - the `eval` extra audited from `uv export --no-dev --extra eval`, with no install, and
      torch's `+cpu` label stripped as the main Audit step does (caveat 12: pip-audit skips
      torch otherwise).
  - **The `eval` extra's advisories** (design review 1). The first draft named one,
    `langchain` 1.3.2's PYSEC-2026-2192, and kept it as an ignore entry until a bump in
    S3-11. The policy above does not allow that, since it has a fixed release. There are
    four, all published before this pass:

    | Package | Advisory | Fixed in |
    | --- | --- | --- |
    | `langchain` 1.3.2 | PYSEC-2026-2192 | 1.3.9 |
    | `langgraph-sdk` 0.3.15 | CVE-2026-104873 | 0.4.4 |
    | `diskcache` 5.6.3 | PYSEC-2026-2447, unsafe unpickling | none |
    | `ragas` 0.4.3 | PYSEC-2026-3046, SSRF in the multimodal faithfulness metric | none |

    - **S3-9 bumps `langchain` to 1.4.3** (maintainer, moved from S3-11). That takes
      `langgraph-sdk` to 0.4.7, so the first two clear. It moves no tier-2 part, since `judge`
      pins ragas and instructor, not langchain, and it moves five packages in all (the
      versions table).
    - **The other two are the policy's first ignore entries.** diskcache's unpickling reaches
      only ragas's own cache folder on a scoring machine, and the judge uses no multimodal
      metric.
  - **A merge waits for every stage:** a branch ruleset on `main` requires `check`,
    `eval-retrieval`, `deploy-config` and `container`. The maintainer applies it with a
    `gh api` recipe. The repository is public, so the free plan has rulesets (checked in
    design review 1).

**DEC-25 — A backup is a snapshot of an immutable generation; a restore is a new
generation and an alias switch** (SRS §11, backup and DR).

- **`rag-backup create`** runs under the writer lock, so recovery cannot delete the
  collection mid-snapshot. A published generation is never written again, so nothing else
  needs to pause.
  1. `create_snapshot` on the live generation;
  2. download it to `paths.backups/<collection>/`, beside a sidecar file: the whole manifest
     (a few hundred bytes a document), Qdrant's sha256, the server's version and the time;
  3. verify the download against that checksum;
  4. delete the server's copy;
  5. keep the last 7.
- **`rag-backup restore FILE`,** under the writer lock from the upload to the end, as
  `create` and `rag-ingest` hold it (design review 1). The first draft took the lock only to
  switch, so an ingest starting mid-restore would delete the uploaded generation as a crash's
  orphan, or publish over the restore.
  1. verify the checksum;
  2. refuse a snapshot the config cannot use, from its sidecar. Its whole manifest must load
     under the current model, and `check_index`'s rule must pass: the embedder's identity,
     and the sparse one with a hybrid retriever (design review 2; the first draft checked
     only the embedder);
  3. take the writer lock;
  4. check that the new name is free, then upload the snapshot under it, through Qdrant's
     `snapshots/upload` endpoint, which `qdrant-client` exposes only through its generated
     HTTP API (`client.http.snapshots_api.recover_from_uploaded_snapshot`, checked).
     - **An upload to a name that exists replaces that collection's data in place.** Given
       the alias's name, it overwrote the live generation, with no switch (checked, design
       review 2). So restore refuses a name taken by a collection or an alias, and the
       runbook never shows a manual upload.
     - The checksum goes with the upload, so the server verifies it too; a wrong one is
       refused with a 400 and leaves nothing;
  5. delete the copy the upload left in the server's `snapshots/` folder, then write the
     manifest whole, with `generation` corrected;
  6. switch the alias, then recover.
- **`rag-backup list`** shows what is on disk.
- **The manifest in a snapshot.** A collection's `config.json` holds its metadata (Qdrant's
  `lib/collection/src/config.rs`), so a snapshot should carry the manifest (design review 1).
  Design review 2 ran the round trip against the server: the restored collection carried the
  source's metadata, its payload index on `metadata.source` and every point, and the
  download's sha256 matched the server's checksum. S3-10's first server test still asserts
  it, and the sidecar's copy stays the fallback.
- **Snapshot files outlive their collections** (design review 2, checked on the server).
  - An upload leaves a full copy under the new collection's `snapshots/` folder, and a crash
    between `create_snapshot` and its deletion leaves one under the live generation's.
  - Deleting a collection deletes neither. They can then no longer be listed or downloaded
    (404), so nothing could clean them up through the API.
  - So restore deletes the upload's copy at once (step 5), `create` deletes its server copy
    in a `finally`, and `recover` deletes a generation's snapshots before the generation.
  - Never delete `snapshots/tmp/`: uploads land there first, and without it every upload
    fails with a 500.
  - Snapshot names have one-second resolution, so two creates in one second share a name;
    the writer lock keeps them apart.
- **A stale backup raises no alert:** the backup service has no metrics listener. `rag-backup
  list` shows the newest, and the README says to check it after any change to the schedule.
- **Schedule:** the compose `backup` profile runs `rag-backup create --every 24h`. It mounts
  `./var` as well as `./backups`, under the app's uid: without `./var` it would not share the
  writer lock (design review 2). Each run takes the lock for itself, and a run that finds an
  ingest holding it tries again 5 minutes later. A cron or systemd-timer recipe covers a host
  without compose.
- **Object storage:** snapshots land in a directory. Copying it off the host (rclone,
  `aws s3 sync`) is the operator's documented step.
- **RPO and RTO.** The corpus, with git, is the source of truth, so the index can always be
  rebuilt. Restoring a snapshot takes about a minute at this size; rebuilding from the corpus
  takes the ingest time, about 2 minutes here, and grows as NFR-3's table says. The index's
  RPO is the backup interval; the corpus's own backup is the operator's.
- **Tested against a real server** (the maintainer's addition). Local mode has no snapshots.
  - In CI, a `qdrant_server` test runs in `eval-retrieval`'s service container: create,
    download, checksum, restore, switch, query, on a small collection.
  - S3-10 and S3-12 restore the real index on the stack, and show tier 1 and the contexts
    digest unchanged afterwards.

**DEC-26 — Tier 2 is re-run once in this phase. S3-11 measures every lever that moves a
hashed input, and S3-11b commits them.**

- **Before S3-11b,** no story moves a hashed input or changes an answer. Each shows
  `rag-eval check` exiting 0, and from S3-2 on the contexts digest, because those stories
  change code that the fingerprint does not hash (design review 1 extended the digest from
  S3-2 to every story).
- **Two stories, as S2-5 and S2-5b** (maintainer, design review 1). S3-11 runs stages 1–4,
  from its branch, and merges only what moves no hashed input: the hand labels, the tables,
  and the code the runs need. S3-11b commits the chosen configuration together with its run,
  so `main` is never stale in between.
- **S3-11's levers:**
  - the judge, decided from the maintainer's hand labels;
  - the prompt: the refusals that paraphrase past the decline marker, `glider-name`'s
    refusal, and LLM01's delimiting of chunks as data;
  - the hybrid switch, with the sparse encoder (S3-5's table);
  - `min_score`, from S3-5's tier-1 table of cutoffs;
  - the model (the DEC-2 revisit): mistral and phi3, plus qwen2:7b if pulled, plus a
    current model if the maintainer wants one;
  - `rag-eval check` recomputing the aggregates from the items;
  - a context digest for each question in the answers file, so a later retrieval change can
    be checked here without a re-run, and the Qdrant server's version beside the generator's
    digest, recorded and not hashed, as DEC-15 records digests: IDF and RRF run in the server
    (design review 1).

  The `eval` extra's `langchain` bump left this list for S3-9 (maintainer, design review 1):
  it moves no hashed input, and the audit policy allows no ignore for a fixed advisory.
- **Staged, so the grid stays small.** The estimates are given up front, as CLAUDE.md asks.
  0. **The hardware** (design review 2). v0.3's exact configuration is generated on the T4,
     and its 25 answers are diffed with the committed ones, with no scoring (about 15
     minutes). It measures how far a T4 result carries to this CPU, and the record says so
     beside every choice.
  1. **The judge.** gemma2:9b's agreement with the hand labels costs no run. If it passed a
     wrong answer as faithful, or its mean error exceeds 0.15, the v0.3 answers are scored
     again by gemma3:12b at 4 bits, and the better judge stays.
     - **That scoring is certain** (design review 2): S2-5b found two wrong answers that
       gemma2:9b scored as faithful. It is one T4 scoring, about 45–50 minutes.
     - **gemma3:12b fits a T4.** Its weights are 8.2 GB (Ollama's `gemma3:12b`, q4_K_M;
       Ollama has served Gemma 3 since v0.6.0). At 8k its KV cache is about 0.8 GiB with
       its 1,024-token sliding window (5 of every 6 layers), or 3 GiB without, so it needs
       about 9.5 to 12 GB of the T4's 15. Ollama runs with `OLLAMA_NUM_PARALLEL=1`, since
       each extra slot adds a cache, and `ollama ps` confirms the size first.
     - **"Better," fixed in advance:** fewer `no`-labelled answers scored faithful (0.5 or
       more), then the lower mean absolute error. A tie keeps gemma2:9b. If both pass a
       wrong answer, the better one stays and its leniency is recorded.
  2. **The retrieval levers** come fixed from tier 1; they are not swept in tier 2.
  3. **The prompt:** v0.3's against the revised one, with mistral, generated and scored on a
     Colab T4. **Keep the revision** if the decline rate improves, no answerable question
     newly declines, and no metric falls by more than 0.05 against the v0.3 prompt's own T4
     run, scored by the same judge (design review 2). The floors are not the bar here: they
     were set on CPU answers, scored by gemma2:9b.
  4. **The model:** the chosen prompt with each candidate. **mistral stays** unless a
     candidate leads by at least 0.10 in faithfulness (2 of 20 items) or 0.2 in the decline
     rate (1 of 5), with no metric behind by more than 0.05. A tie goes to the time to first
     token on this CPU (design review 2).
  5. **The commit, S3-11b:** the chosen configuration, generated here (about 70 minutes for
     mistral) and scored on the T4 (35–50 minutes, by the judge). Each floor is the baseline
     minus 0.05, and tier 1 is re-baselined if hybrid retrieval is on.

  **The rules are fixed before the first run** (design review 2). The golden set can show
  only large differences:

  - 4 of the 20 faithfulness items are below 1 (0.0, 0.67, 0.75 and 0.75);
  - context recall is 1.0 on all 20;
  - a decline rate of 3 in 5 has a 95% interval of 0.23 to 0.88;
  - a sign test needs 6 changed items, all one way, for p = 0.03.

  So each rule is a margin with a default, written in S3-11 and committed before stage 0
  runs. None is a significance claim drawn after the results.

  **In all, about 5–9 hours of T4 time across 3–4 sessions,** and about 70 minutes here
  (design review 2; the first draft said 4–6 hours). That is 5–7 scorings at 35–50 minutes,
  generations of about 15 minutes each, and 15–25 minutes of setup per session. The
  generation time is an estimate: it has not been timed on a T4, so stage 0's run revises it.
  Every candidate is generated on the T4, since a GPU's answers can differ from a CPU's in
  the last bits; only the committed run is generated here.
- **Retrieval on Colab is this host's** (design review 2).
  - The notebook restores this host's index from a snapshot (S3-10's `rag-backup restore`,
    into Qdrant's static binary there), rather than ingesting again with `minilm_cuda`. With
    hybrid retrieval on, the hybrid index is built here and restored there the same way.
  - Queries are embedded with `minilm_cpu`, and each question's context digest is compared
    with this host's before anything is generated.
  - The T4 runs Ollama only. That keeps GPU embedding drift out of the contexts, and the CUDA
    torch install out of the notebook: S1-7's cu130 wheels need a recent NVIDIA driver,
    which no Colab run has shown yet (the backlog's CUDA 12 line).
- **One check outside the labels** (design review 2). The judge is chosen on the 20 answers
  it is then measured against. So after S3-11b, the maintainer also labels the committed
  answers that the judge scored below 1, or that the two judges disagree on (about 5).
  Their agreement is recorded beside stage 1's, with no re-run.
- **The hand labels** are committed as `eval/hand_labels.jsonl`: one line per answerable
  `id`, with `faithful` (`yes`, `partly` or `no`), `unsupported` (the claims the cited
  pages do not support) and `notes`.
- **Upgrading Ollama** past 0.22.1 is out of scope, unless a candidate model needs it. It
  would then ride the same re-run, since the digests change.

**DEC-27 — NFR-2 is revised rather than bought** (maintainer). This resolves ADR-016.

- **(a)** The first answer event, `sources`, arrives in under 2 s at p95 with the model
  server idle. It was about 0.8 s in S2-9. `rag_sources_seconds` watches it.
  - **Measured as a p95** (design review 1): one sample is not one, and the rerank alone
    takes a median of 1.4 s, up to 2 s (§2.1). S3-7 times the 25 golden questions with the
    model idle, and S3-11b times them again on what ships. _As measured in S3-7:_ a p95 of
    2.09 s, then 2.02 s, a miss. S3-11's stage 2 owns the rerank's levers (DEC-22's as-built
    note).
- **(b)** The time to first token is measured on every answer and reported per class of
  hardware: on this CPU, the p95 from the committed tier-2 run (S3-11b; about 155–170 s
  today); on a T4, the p95 measured in S3-11.
- **Serving from a GPU later needs no code:** the llm's `base_url` points at a GPU host. It
  is a runtime key, so nothing hashed moves.

**DEC-28 — The code is formatted once, and formatting is gated** (maintainer). That is S3-1.

- **One commit runs `ruff format`** over the 30 Python files it changes (11 in `src`, 19 in
  `tests`, about 860 changed lines, measured with `ruff format --check` on 2026-10-14), plus
  the formatter's Markdown exclusion. Nothing else is in it.
- **Markdown is excluded,** `[tool.ruff.format] exclude = ["*.md"]`. ruff 0.16 also
  reformats the Python blocks inside Markdown, which would rewrite examples in two docs and
  two closed story files.
- **Then** `ruff format --check` runs after Lint in CI's `check` job, and
  `.git-blame-ignore-revs` names the reformat commit, which a merge commit preserves.
- No hashed input moves: formatting changes no behaviour, and `config.yaml` is not Python.

**Earlier decisions, settled here:**

- **DEC-2** is revisited in S3-11 (DEC-26).
- **DEC-5 step 3** is S3-3, by the mechanism DEC-19 amends. Afterwards, application code
  imports `langchain_core`, `langchain_text_splitters`, `langchain_huggingface` and
  `langchain_ollama`; `langchain_openai` is reached only through the fallback's
  `_target_`.
- **DEC-9** is resolved by DEC-21.

### 3.2 Layout at Phase 3 exit

```text
src/rag_qa/
├── vectorstore.py   REWRITTEN  QdrantStore; generations as collections, the alias switch,
│                               recovery; the manifest in collection metadata; QdrantRetriever,
│                               which makes every cut and fuses the hybrid branches itself
│                               (S3-2, S3-5); snapshot and restore (S3-10). The only module
│                               that imports qdrant_client.
├── sparse.py        NEW  Bm25Encoder, SPARSE_VERSION; the standard library only (S3-5)
├── resilience.py    NEW  CircuitBreaker, retry policies (S3-6)
├── observability/   NEW  logs.py (structlog), metrics.py (Prometheus), genai.py (S3-7);
│                         tracing.py (OpenTelemetry) (S3-7b)
├── backup.py        NEW  rag-backup (S3-10)
├── manifest.py      + version 2: chunk counts, the per-document splitter, the sparse record
├── chain.py         + check_index and open_retrieval over the store, the fallback llm, filters
├── answering.py     + spans, token usage, the fallback stream, sources, Done.model
├── ingest.py        + the Qdrant store, per-extension splitters, a maximum file size, sparse
│                      vectors
├── api/
│   ├── auth.py      NEW  keys and scopes (S3-8)
│   ├── limits.py    NEW  token buckets, the body-size middleware (S3-8)
│   ├── keys_cli.py  NEW  rag-keys (S3-8)
│   ├── jobs.py      + SQLite records (S3-8)
│   └── state.py     + the alias refresh, breakers, the probe cache
├── settings.py      + RAG_QDRANT_{URL,COLLECTION,API_KEY}, RAG_STATE_PATH, RAG_BACKUP_PATH,
│                      RAG_CONFIG_OVERLAY, RAG_LOG_FORMAT,
│                      RAG_API_{KEYS_FILE,RATE_LIMIT,TRUSTED_PROXIES,METRICS_PORT}
├── config.py        + the overlay merge (S3-9)
└── schema.py        + vector_stores, sparse_encoders, pipeline.vector_store, splitters,
                       max_file_mb, fallback_llm, paths.state and backups, retriever modes
deploy/              NEW  Caddyfile; prometheus/{prometheus.yml, rules.yml, rules.test.yml, rules.causes.test.yml};
                          alertmanager/alertmanager.yml
Dockerfile, .dockerignore, config.docker.yaml   NEW (S3-9)
compose.yaml         NEW (S3-2, with qdrant only; the rest in S3-9)
.git-blame-ignore-revs (S3-1); scripts/bench_retrieval.py (S3-7); eval/hand_labels.jsonl (S3-11)
var/, backups/       gitignored
```

Entry points: as before, plus `rag-keys` (S3-8) and `rag-backup` (S3-10).

Dependency direction extends ADR-009:

- only `vectorstore` imports `qdrant_client`;
- `sparse` and `resilience` load no LangChain;
- `observability` loads no model stack, so any module may import it;
- `backup → {vectorstore, manifest, config, settings}`, never `chain` or `api`;
- `ingest` still never imports `chain`, `cli` or `api`.

### 3.3 Config shape changes

```yaml
components:
  splitters:
    markdown: {...}                                  # S3-4; used by no extension by default
  sparse_encoders:                                   # S3-5
    bm25: {_target_: rag_qa.sparse.Bm25Encoder, k1: 1.2, b: 0.75, avg_len: <measured in S3-5>}
  llms:
    <provider>_fallback:                             # S3-6; opt-in
      {_target_: langchain_openai.ChatOpenAI, model: ..., base_url: ...}   # key from the env
  retrievers:
    vector_search: {search_kwargs: {k: 5}}           # unchanged: its dump must not move
    hybrid: {mode: hybrid, search_kwargs: {k: 5}, prefetch: 40, rrf_k: 60}   # S3-5
  vector_stores:                                     # S3-2: the _target_-built store §0.5 planned
    qdrant:
      _target_: rag_qa.vectorstore.QdrantStore
      url: "http://localhost:6333"                   # RAG_QDRANT_URL overrides
      collection: rag_qa                             # the alias; RAG_QDRANT_COLLECTION overrides
      timeout_s: 10
pipeline:
  ingestion:
    splitters: {}            # S3-4: per-extension overrides of `splitter`
    max_file_mb: 50          # S3-4: SRS §7.1's limit; the largest corpus file is 25.4 MB
    # sparse_encoder: components.sparse_encoders.bm25    # set by S3-11b, with the hybrid switch
  query:
    fallback_llm: null       # S3-6
  vector_store: components.vector_stores.qdrant      # S3-2: shared by ingestion and queries
paths:
  data: corpus
  state: var                 # S3-2; paths.vector_store is refused, with the migration
  backups: backups           # S3-10
```

Every new key is optional, and is either absent from the real config, left out of the hashes
while unset, or never hashed: `max_file_mb` reaches `corpus` only through the walk, so a limit
that skips nothing moves nothing. A unit test holds the real config's six fingerprint parts
unchanged from S3-1 to S3-11.

_As built in S3-4 (2026-10-21):_ `ingestion` hashes the `splitters` entries whose spec differs
from the default's, and only when there is one. An entry with the default's spec makes the
default's chunks and re-embeds nothing, so it moves nothing either (ADR-026's rule: what
determines the chunks). The real config's `ingestion` part is pinned to the committed run's
by a unit test, which a change to the part's shape fails, and the six parts by
`rag-eval check`.

### 3.4 Interfaces

**The store** (`vectorstore.py`, the seam):

```python
class QdrantStore:  # the _target_; its api key comes from RAG_QDRANT_API_KEY; tests inject a client
    def __init__(self, url: str, collection: str, timeout_s: float = 10.0) -> None: ...

def live_index(store) -> tuple[str, Manifest] | str       # the generation behind the alias, or why none
def in_the_way(store) -> str | None
def open_generation(store, generation: str) -> OpenGeneration   # its point IDs, checked against the manifest
def write_generation(store, manifest, *, base, delete, add: Embedded) -> Published  # build, then switch
def recover(store) -> list[str]
def retriever(store, generation: str, embeddings, spec, *, sparse=None, k: int) -> BaseRetriever
def snapshot(store, generation: str, dest: Path) -> BackupInfo                       # S3-10
def restore(store, backup: Path, *, expect_embedder: str) -> Published             # S3-10

@dataclass(frozen=True)
class Embedded:  # as in Phase 2, plus the sparse vectors
    ids: list[str]; texts: list[str]; metadatas: list[dict[str, Any]]
    vectors: list[list[float]]; sparse: list[SparseVector] | None
```

**The query side:**

- `QueryPipeline` gains `fallback_llm: BaseChatModel | None`.
- `stream_answer(pipeline, question, *, sources=())`.
- `retrieve` takes a `RetrievalQuery(question, sources=())`. Tier 1 passes no sources; S3-5's
  plan settles the exact type. _As built in S3-5:_ a frozen dataclass in `chain.py`, and the
  retrieval half is `Runnable[RetrievalQuery, list[Document]]`, with no `str` form; tier 1
  passes `RetrievalQuery(item.question)`.
- `Done(ttft_ms, total_ms, model, input_tokens, output_tokens)`. _As built in S3-6:_ `model`
  is the config reference of the model that answered; S3-7 adds the token counts. _As built
  in S3-7:_ both default to `None`, which a provider reporting no usage leaves them; the
  `done` event and tier 2's answers file carry them.

**HTTP changes** (to §2.4's table):

| Route | Change |
| --- | --- |
| `POST /v1/query` | an optional `sources`; 413, 429, and 403 for a missing scope; 503 while a breaker is open; `done` gains `model` and the token counts |
| `POST /v1/ingest`, `GET /v1/ingest/{job_id}` | the `ingest` scope; the 409 holds across replicas; a job can be `abandoned` |
| `GET /v1/documents` | the `query` scope; `chunks` comes from manifest version 2 |
| `GET /v1/health` | adds `store` and `breakers`; probes cached for 5 s |
| `GET /metrics` | new, on its own listener (`RAG_API_METRICS_PORT`) |

**Manifest version 2** (in the generation collection's metadata):

```json
{"version": 2, "generation": "rag_qa--gen-20261020T120000123456Z-1a2b3c4d", "chunking": 1,
 "embedder": {"ref": "components.embedders.minilm_cpu", "identity": "<sha256>", "dimension": 384},
 "splitter": {"ref": "components.splitters.english_recursive", "identity": "<sha256>"},
 "sparse": null, "store": {"distance": "cosine", "exact": true},
 "documents": {"2412.14140v2.pdf": {"sha256": "<sha256>", "loader": "components.loaders.pdf",
   "loader_identity": "<sha256>", "splitter": "<sha256>", "chunks": 412,
   "ingested_at": "2026-10-20T12:00:00Z"}}}
```

**Environment variables** (new or changed; `settings.py` stays their only reader):

| Variable | Story | Meaning |
| --- | --- | --- |
| `RAG_QDRANT_URL`, `RAG_QDRANT_COLLECTION`, `RAG_QDRANT_API_KEY` | S3-2 | the store's address, the alias (a scratch index is another name), its key |
| `RAG_FALLBACK_API_KEY` | S3-6 | the hosted fallback's key, whatever its provider (DEC-21's as-built note) |
| `RAG_STATE_PATH` | S3-2 | replaces `RAG_VECTOR_STORE_PATH`, which is refused while set |
| `RAG_LOG_FORMAT`; `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME` | S3-7; S3-7b | logs, `json` or `console`, unset following whether stderr is a terminal; traces (the standard variables) |
| `RAG_API_METRICS_PORT` | S3-7 | the metrics listener, off unless set, on the API's own host; the API's port, or one in use, exits 2 |
| `RAG_API_KEYS_FILE`, `RAG_API_RATE_LIMIT`, `RAG_API_TRUSTED_PROXIES` | S3-8 | keys (in `var/`), limits, the proxy's own address, never a range (design review 2) |
| `RAG_CONFIG_OVERLAY` | S3-9 | the overlay file |
| `RAG_BACKUP_PATH` | S3-10 | overrides `paths.backups` |

### 3.5 Testing and CI

- **The unit suite stays hermetic** (DEC-11). Qdrant runs in process, as
  `QdrantClient(":memory:")`, through a fixture. Local mode in 1.19.1 supports aliases,
  collection metadata, sparse vectors with IDF, and RRF with a constant and weights
  (checked). The fusion is ours since design review 2, so the hermetic suite runs the code
  that fuses. No Docker, no network.
- **The real server, where local mode is not enough.** Local mode is a reimplementation in
  Python, and it has no snapshots. Tests that need the server carry a `qdrant_server` marker:
  - the generation switch and recovery, with no alias too;
  - the metadata round trip;
  - the cut rule and the hybrid query;
  - a snapshot's create, download and restore: the manifest survives it, a taken name is
    refused, and no snapshot file is left behind.

  A tie test compares two collections built in different insertion orders, since within one
  process the server always agreed with itself (design review 2). The tests run in
  `eval-retrieval`, against its service container (`qdrant/qdrant:v1.19.2`), and are
  skipped elsewhere unless `RAG_TEST_QDRANT_URL` is set.
- **Parity and invariance:**
  - the contexts digest before and after the store move (S3-2), twice on Qdrant, and again in
    every story from S3-3 to S3-10 against S3-2's file (design review 1);
  - hybrid tier 1 run twice in S3-5, the second after a server restart, with identical
    contexts (DEC-19's cut rule);
  - `rag-eval check` exiting 0 on every story from S3-1 to S3-11;
  - unit tests that each new optional key, unset, leaves the real config's six fingerprint
    parts unchanged.
- **Cancellation, proven again at each new layer:**
  - with tracing on: the forced-cancel tests and the real-uvicorn drop, with an in-memory
    span exporter showing every span ended and no context-detach error (S3-7b);
  - through the fallback's stream (S3-6);
  - with the body-limit middleware on (S3-8), since it wraps `receive` too (design review 2);
  - through the proxy, against real Ollama, by hand (S3-9).
- **The architecture test grows:**
  - only `vectorstore` imports `qdrant_client`;
  - no `langchain_community`, `faiss` or `langchain_classic` anywhere in `src` (S3-3);
  - `sparse` and `resilience` load no LangChain;
  - `observability` loads no model stack: no LangChain, torch, sentence-transformers,
    transformers or qdrant_client (S3-7);
  - `stream_answer` passes the model no callbacks and no config (a spy model).
- **CI at the exit:** `check` (with the format gate), `eval-retrieval` (with the Qdrant
  service and the server tests), `deploy-config`, `container`, and the weekly schedule,
  which also scans every pinned third-party image (design review 2).
  Coverage stays at 80% or more.

### 3.6 Deliberately not built in Phase 3

- **Storage and scale:** ingestion on several hosts, a distributed job queue, a Qdrant
  cluster; a clone from a snapshot for a much larger corpus; HNSW and quantization tuning;
  stemming in BM25, until tier 1 shows misses it would fix.
- **Access:** OIDC or JWT, per-document access control, multi-tenancy (out of the SRS's
  scope).
- **Observability:** prompt or answer text in spans; Grafana dashboards; Langfuse.
- **Serving:** a GPU host (the `base_url` seam is enough); Kubernetes; autoscaling.
- **Ollama** past 0.22.1, unless S3-11 needs it.
- **Interfaces:** SSE resume, WebSockets, a web UI.

### 3.7 Phase mapping delta

| Target-stack row | Phase 3 | After v1.0 |
| --- | --- | --- |
| Vector store | Qdrant: generations behind an alias, exact search, the manifest in metadata | a cluster; a clone from a snapshot |
| Hybrid retrieval (FR-3) | our own BM25, fused by our own RRF, then the cross-encoder; a source filter | stemming; weighted RRF if tier 1 asks |
| Own components (DEC-5) | step 3 done: `langchain-community` and FAISS leave (ragas keeps them in the `eval` extra) | the ragas import shim goes when ragas allows |
| Resilience (NFR-6) | timeouts, safe retries, breakers, the opt-in fallback | — |
| Observability (NFR-10) | structlog, our own spans, Prometheus, burn-rate alerts, Jaeger | dashboards |
| FastAPI + SSE (DEC-18) | keys and scopes, limits, durable jobs, TLS at Caddy | OIDC |
| Containers and scanning (§11) | the image, compose, Trivy pinned by hash, the audit policy | Kubernetes |
| Backup (§11) | snapshots, a restore drill, RPO and RTO stated | off-host copies automated |
| RAGAs gate (DEC-15) | one batched re-run; a context digest per question | — |
| NFR-2 | revised; measured on this CPU and on a T4 | a GPU host, by `base_url` |
