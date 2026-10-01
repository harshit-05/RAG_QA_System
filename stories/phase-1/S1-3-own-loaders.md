# S1-3: Own loaders, extension mapping in the pipeline, recursive corpus walk

| | |
| --- | --- |
| **Status** | Done (2026-09-30) — PR #1, rebase-merged into `main` as `3b0e77f` + review follow-up `ddcd98f`; CI green on the branch, the PR and `main` |
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
#    Set BOTH paths: with RAG_DATA_PATH alone, rag-ingest saves the scratch build
#    over the real vectorstore/db_faiss (found while verifying, 2026-09-30).
RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-ingest 2>&1 | tail -5   # finds the subdir

# 4. unit tests, no network
uv run pytest tests/test_loaders.py tests/test_components.py -q

# 5. community imports are down to FAISS only
#    (import lines only — registry.py's allowlist names the prefix as a string)
grep -rnE "^\s*(from|import) langchain_community" src/rag_qa/ | grep -v vectorstore.py | wc -l   # → 0
```

### Results (2026-09-30)

**Risk call: branch.** This is the first story that can silently degrade answers
without failing anything. It replaces every loader, changes the config shape and
refactors `chain.py` / `ingest.py`. So it goes on `s1-3-own-loaders`, then a PR
into `main`.

**Steps 1–2: before/after on the whole index, not one query.** The FAISS
docstore holds every chunk, so the snapshot recorded all 1,708: citation string,
SHA-1 of the text and metadata keys. Baseline taken on `f68eee3` with the
`langchain_community` loaders; "after" re-ingested with ours.

| Check | Before (`PyPDFLoader`) | After (`PdfLoader`) |
| --- | --- | --- |
| pages / chunks | 561 / 1,708 | 561 / 1,708 |
| chunks whose **citation** differs (index order) | — | **0 / 1,708** |
| chunks whose **text** differs (SHA-1, index order) | — | **0 / 1,708**: byte-identical, stronger than the story's chunk-count bar |
| fixed query's 5 citations | p. 9, 340, 477, 233, 81 | identical |
| metadata key sets | 3 different sets, 11–14 keys by PDF producer | **1 set, 5 keys**, all 1,708 chunks |

Metadata: kept `source`, `page`, `page_label`, `total_pages`; added `loader`;
dropped the PDF document-info fields `author`, `creationdate`, `creator`,
`keywords`, `moddate`, `producer`, `ptex.fullbanner`, `rgid`, `subject`,
`title`, `trapped`.

The snapshot queried the retriever directly: retrieval is LLM-independent, so
this is seconds instead of a 7-minute mistral run. The story's literal step-2
command then ran end to end twice, and the rewritten `chain.py` behaves
identically in both:

- **phi3** through a scratch config (5.8 GB free, below mistral's ~6 GB): keys
  `['answer', 'context', 'question']`, the same 5 citations, and an answer
  word-for-word identical to S1-1's phi3 run.
- **mistral on the real, unmodified config**, once RAM was freed (9.0 GB): the
  same 5 citations, and an answer word-for-word identical to the S1-1 and S1-2
  mistral runs (7 m 21 s).

| Check | Result |
| --- | --- |
| 3. recursion on real files (3 PDFs only under `sub/`) | pass: all three found as `sub/…`, 561 pages → 1,708 chunks. Index pointed at scratch too; the repo index was untouched (mtime checked) |
| 4. `pytest tests/test_loaders.py tests/test_components.py` | pass: **27 passed**. Full suite **104 passed**, `HF_HUB_OFFLINE=1` |
| 5. `langchain_community` imports outside `vectorstore.py` | **0**; the only one left is `vectorstore.py:18` (FAISS) |
| review: embedder construction | only `components.build_embedder`, called from `ingest.py:97` and `chain.py:69` |
| review: direct `import_from_string` callers | `registry.build_object` and `config.check_imports` only; the old loader call site is gone |
| ADR-009, at runtime | importing `rag_qa.ingest` does not load `rag_qa.chain` |
| parity oracle tests | ours == `PyPDFLoader` on the fixture PDF and the real 13-page PDF (text + `source`/`page`/`page_label`/`total_pages`); == `Docx2txtLoader`, == old `TextLoader` |
| upgrading an old config | the pre-S1-3 `config.yaml` from `main` fails with *"'extensions' moved out of the loader entries into pipeline.ingestion.loaders (S1-3)…"* plus *"pipeline.ingestion.loaders: required key is missing"* |
| ruff (whole project) | All checks passed |

## Review notes for the human

The before/after citation list is the whole review — if those strings differ,
`page_label` handling regressed and every answer's sources are subtly wrong.
Second, check `build_embedder` is the *only* place an embedder is constructed
(`grep -rn "embedder" src/rag_qa/`), since a drift between ingest and query is a
silent wrong-answer bug rather than a crash. Third, `pypdf` page text can differ
from `PyPDFLoader`'s in whitespace; a chunk-count match is the bar, not
byte-identical text.

## Discovered

- **Verification step 3 as written would overwrite the real index.** It sets
  `RAG_DATA_PATH` to the scratch corpus but not `RAG_VECTOR_STORE_PATH`, so
  `rag-ingest` saves the scratch build over `vectorstore/db_faiss`. It ran with
  both overrides set. Worth fixing in the story template's habits: any
  verification that ingests a scratch corpus sets both paths.
- **PDF metadata was producer-dependent.** `PyPDFLoader` copied each PDF's own
  document-info dict onto every page, so chunks carried 3 different key sets
  (11–14 keys), including `ptex.fullbanner` and `rgid`. After: one uniform
  5-key set. If Phase 2's `GET /v1/documents` wants a title or author, read them
  from the PDF there, once per document, not per chunk.
- **The ADR-009 grep in the habit list gives a false positive:** `^(from|import)
  .*chain` matches `langchain_core`. The runtime check (`import rag_qa.ingest`
  leaves `rag_qa.chain` unloaded) is the reliable form → S1-5 regression suite.
- **The `langchain-community` sunset warning now comes from one line only,**
  `vectorstore.py:18` (FAISS). That is the DEC-5 exit made visible; Phase 3's
  Qdrant move removes the last one.
- **File names in ingestion output are now relative paths** (`sub/report.docx`),
  both printed and in `IngestReport.skipped` / `.failed`, so same-named files in
  different subdirectories stay distinguishable. S1-4's tests should expect that
  form.

### Second review (2026-09-30)

- **The recursive walk ingested hidden folders and Office lock files.** Probed on
  `e92332b` with the real config over a working-folder layout: a
  `.ipynb_checkpoints/notes-checkpoint.md` copy was **indexed** next to
  `notes.md` (a duplicate competing for the `k: 5` slots); `.git/` was walked;
  Word's `~$report.docx` lock file ends in `.docx`, so it landed in
  `report.failed` ("File is not a zip file"). After S1-4 makes failures exit
  non-zero, an open Word document would have broken ingestion. The flat
  `os.listdir` never entered subfolders, so this arrived with ISS-13 itself.
  **Fixed in the review follow-up:** `discover_files` leaves out any path with a
  part starting with `.` (LibreOffice's `.~lock.*#` included) and names starting
  with `~$`. They are not reported as skipped, since they were never documents.
  Tests first: `sample_corpus` gained a hidden folder, a hidden file and a `~$`
  file, and three tests failed against `e92332b` (the exact discovery list, the
  loaded-documents map, and a new named test). After: **105 passed**, `ruff`
  clean, the real `corpus/` discovers the same five files as before.
- **Open: symlinks are handled inconsistently.** A symlinked *file* is followed,
  even when it points outside the corpus (probe: `link.txt` → `../outside/`
  was indexed). A symlinked *directory* is not walked: Python 3.12's `rglob`
  does not follow directory symlinks. Not a security issue while the corpus is
  trusted content (SRS §2.6), but the rule should be one deliberate choice, "skip
  symlinks" or "follow both", stated in `discover_files`' docstring. Candidate
  for S1-4, which already owns ingestion behaviour, or the Phase 2 upload path,
  where the corpus stops being hand-curated.
- **Resolved: the branch.** At review time the work was uncommitted on `main`
  while this file said `s1-3-own-loaders`; it has since been committed there as
  `e92332b` and pushed. The follow-up lands on the same branch and PR.

## Deviation from plan

- **Fixtures are generated at test time, not stored in `tests/fixtures/`.**
  CLAUDE.md forbids committing binaries, and a PDF and a DOCX are binaries. So
  `conftest.py` writes them, as a hand-built 3-page PDF labelled i, ii, 1 and a
  3-part DOCX, plus a nested corpus with an uppercase suffix and an unmapped
  type. Hand-built because pypdf only adds objects through a private API.
- **`total_pages` kept, beyond the story's four-key contract.** `PyPDFLoader`
  emitted it, it is always meaningful, and dropping it would be a regression for
  nothing.
- **Additions beyond scope, all small and in service of it:**
  - a migration error for a pre-S1-3 config (`extensions` left in a loader
    entry), which would otherwise load and then fail at build time as an
    unexplained `TypeError`;
  - validation that extension keys are lowercase with their dot (`"PDF"` or
    `".Md"` could never match a file);
  - `discover_files` names a missing corpus directory, where `rglob` would
    silently yield nothing;
  - `TextLoader` pins UTF-8 (the old one used the locale's encoding), with an
    `encoding` kwarg;
  - one reference walk (`RagConfig._slots`) shared by validation and
    `references()`, so the loader map can't be checked but not listed, or the
    reverse.
- **Parity tests against the loaders being replaced** (`PyPDFLoader`,
  `Docx2txtLoader`, `TextLoader`) as oracles, including one real corpus PDF.
  They go when `langchain-community` leaves in Phase 3.
- Before/after compared all 1,708 chunks from the docstore rather than one
  query's five, and retrieval ran without the LLM. The literal step-2 command
  was still run once, after, end to end.
