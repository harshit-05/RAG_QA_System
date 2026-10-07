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
| `judge` | the judge model name, the metric set, the ragas version and `eval/judge.Modelfile`'s text | score only |

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
- **Cost, for S2-5b:** one answerable item took 39 min to score on this CPU (2 Ollama
  threads), so a full local scoring is about 13 h, not 3–4 h. Colab/Kaggle is the default.

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
│                       cli.py             rag-eval {retrieval, generate, score, check}
├── api/           NEW  app.py · routes.py · models.py · server.py (rag-serve)
├── chain.py            + QueryPipeline, build_retrieve, build_query_pipeline;
│                         build_rag_chain kept for invoke (§0.4 contract)
├── components.py       + build_reranker, aclose_llm
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

- `cli → {answering, chain, config}`.
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

def build_retrieve(config: RagConfig, embeddings: Embeddings) -> Runnable[str, list[Document]]
def build_query_pipeline(config: RagConfig, *, require_index: bool = True) -> QueryPipeline
    # require_index=True (CLI, eval): no usable index raises NoIndexError or
    # IncompatibleIndexError. require_index=False (API): with no usable index, retrieve is
    # None and the LLM half is still built; with one, both halves are built. The API then
    # `dataclasses.replace`s the retrieval half in after an ingest
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
  the ragas version;
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

§2.8, "As delivered", is written by S2-9.
