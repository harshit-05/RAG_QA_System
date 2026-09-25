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
