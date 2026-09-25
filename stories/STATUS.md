# Board

> Updated at every story close-out. This file + the story files ARE the
> project memory across sessions.

## Now

**Phase 0 is complete.** The exit criterion was met literally: a fresh clone
ran `uv sync`, ingested the corpus and answered through `rag-query`. What is
left is manual: commit S0-6, tag `v0.1`, and push.

```bash
git add -A
git commit -m "S0-6: end-to-end proof on the real corpus, README, release 0.1.0 (Phase 0 exit)"
git tag -a v0.1 -m "v0.1: Phase 0 complete; runs end to end"
git push origin main
git push origin v0.1
```

S0-7 (`55344bd`) is committed but was not yet pushed; the push above sends it
too. **Next session:** the Phase 1 architecture pass, WORKFLOW.md Step 1, in
plan mode.

**Known quality gap carried into Phase 1+:** retrieval is reliable, but the
7B model sometimes misstates retrieved facts (S0-6 fact-check: misattributed
scores, an invented acronym expansion). Faithfulness is unmeasured until the
Phase 2 evaluation harness; the S0-6 story records the failing question and
its verified ground truth.

**Commit history so far** (S0-1 → S0-4; ISS-09 verified fully closed, `git
ls-files` shows no parquet, FAISS index, cache, media or `.save` file):

| Commit | Contents |
| --- | --- |
| `ee86e1f` | pre-v0.1 baseline: workflow kit + Phase 0 arch pass |
| `f702dc8` | S0-1 hygiene (incomplete — missed four files) |
| `e781b17` | S0-1 fixup **fused with** the S0-3 file moves |
| `0987c60` | S0-2 uv project |
| `2bc7658` | S0-3 module split + entry points |
| `e16c07d` | S0-4 config repair + corpus rename |
| `ebcfc2e` | S0-4 follow-up: GPU and multilingual embedders restored as config axes |
| `6630222` | S0-5 LangChain 1.x LCEL chain, streaming CLI with sources |
| `55344bd` | S0-7 project docs into `docs/`, corpus rule inverted |

`e781b17` deliberately carries two stories' worth of change: the S0-3 `git mv`
operations were staged when the fixup was committed, so the content merged, and
the message was amended to describe both rather than rewrite already-public
history. Read it as "S0-1 fixup + first half of S0-3". **Rule that came out of
it: Claude must not run staging git commands (`git mv`, `git rm`, `git add`)
while the maintainer commits manually — three collisions came from exactly
that.** S0-4's corpus rename therefore used a plain `mv`; rename detection is
computed from content at commit time, so history is unaffected.

The Phase 0 architecture pass is done: `ARCHITECTURE.md` Phase 0 is confirmed and
DEC-1/DEC-2/DEC-4 are resolved below.

## Release mapping (per SRS §12, rev 1.1)

| Phase exit | Git tag | Package version |
| --- | --- | --- |
| Phase 0 | `v0.1` | 0.1.0 — **exit met 2026-09-25**; tag pending maintainer |
| Phase 1 | `v0.2` | 0.2.0 |
| Phase 2 | `v0.3` | 0.3.0 |
| Phase 3 (SRS complete) | `v1.0` | 1.0.0 |

The final story of each phase performs the tag + version bump at close-out.

## Human prerequisites (do these anytime, no session needed)

- [x] `gh auth login` — done 2026-09-24 as `harshit-05` (HTTPS, keyring;
      scopes: repo, read:org, gist). Needed to push and for Phase 1 CI/PR work.

- [x] Resolve DEC-1 / DEC-2 — done in the Phase 0 arch pass, 2026-09-18.

- [ ] Nothing else: uv ✓, Python 3.12 (uv-managed) ✓, Ollama daemon ✓,
      Docker 29.6.2 ✓ (Phase 3), disk 311 GB free ✓. **No MCP servers are
      required for any phase** — built-in tools cover the whole workflow.

## Pre-flight caveats (read before the first story session)

1. **Three commits pending (maintainer commits manually).** S0-1 and S0-2 ran
   before the baseline commit, so the index holds all three.

   **`git reset` silently undoes S0-1.** Unstaging restores the index to HEAD,
   where the binaries are tracked again; `.gitignore` does **not** untrack an
   already-tracked file, so a plain `git add -A && git commit` after a reset
   keeps ~10 MB of index/parquet/media in the tree and ISS-09 quietly fails.
   (This already happened once on 2026-09-18 and was re-applied.) If the index
   ever looks wrong, check with
   `git ls-files | grep -E '__pycache__|vectorstore/|\.webm|\.save|\.parquet'` —
   it must print nothing. The recipe below is safe to re-run from any state:

   ```bash
   # 1 — baseline: pre-existing v2 edits + workflow kit + Phase 0 arch pass
   git reset
   git add v2/config.yaml v2/file_processor.py v2/pipeline_builder.py v2/check_config.py \
           CLAUDE.md SRS.md WORKFLOW.md ARCHITECTURE.md stories/
   git commit -m "pre-v0.1 baseline: workflow kit, Phase 0 architecture, uncommitted v2 edits"

   # 2 — S0-1: the rm --cached IS the story; files stay on disk
   git rm -r -q --cached __pycache__ v1/__pycache__ vectorstore/db_faiss \
          "Screencast from 30-07-25 04_25_00 PM IST.webm" v1/docx_processor.py.save \
          docs/0000.parquet docs/train.parquet
   git add .gitignore scripts/fetch_dataset.py
   git add -u                     # stages the on-disk deletions + the rename
   git commit -m "S0-1: repo hygiene — gitignore, purge tracked artifacts (ISS-09)"

   # 3 — S0-2
   git add pyproject.toml uv.lock .python-version src/rag_qa/__init__.py
   git commit -m "S0-2: uv project — pyproject, pinned 3.12, lockfile (ISS-08)"
   ```

2. **Stale FAISS index trap (S0-5/S0-6).** `vectorstore/db_faiss/index.pkl`
   was pickled under the old LangChain — after the 1.x migration it will
   likely fail to unpickle. Any chain-construction check must re-ingest
   first; never debug an unpickle error there, just rebuild the index.
3. **CPU latency expectations.** 7B on CPU: expect ~5–15 s to first token
   and ~1–2 min full answers; the current small corpus ingests in minutes.
   Fine for Phase 0 proof. Consequence: **NFR-2 (<2 s first-token p95) is
   not achievable CPU-only** — by v1.0 either revise the SLO or plan GPU
   serving. Flagged in GPU offload notes below.
4. **First run downloads models.** Ingestion pulls the MiniLM embedder
   (~90 MB) from HuggingFace on first use — needs network once.
5. **`docs/` parquet files have no loader** — S0-1 untracks them and moves
   `docs/dataset.py` to `scripts/`; after S0-1 the corpus dir holds only the
   three PDFs.
6. **PyTorch index trap (S0-2).** `download.pytorch.org/whl/cpu` hosts stale
   `langchain-community` releases. Declared as a general index it makes uv
   silently resolve the whole stack to LangChain 0.3.x. It must be
   `explicit = true` and bound to torch via `[tool.uv.sources]` — see
   ARCHITECTURE.md §0.1. Check `langchain-core` is 1.x in `uv.lock`.
7. **Ollama context window.** Default `num_ctx` is 2048; the old config's
   k=10 × 1000-char chunks overflowed it silently. Config now sets
   `num_ctx: 4096`, `k: 5`. If answers look ungrounded, check these first.
8. **Memory at S0-6.** This host showed only 4.3 GB free with desktop apps
   open (2026-09-18). Run `free -h` before the proof; use `phi3` if under
   ~6 GB free, and record which model produced the transcript.

## GPU offload notes (nothing *requires* GPU; Colab/Kaggle available)

- **Phase 0–1: no GPU work at all.**
- **Phase 2 — RAGAs eval sweeps**: with a local CPU judge, a full metric
  sweep is an hours-scale run. Best Colab/Kaggle candidate: run the eval
  notebook (judge model on their GPU) against exported answers/contexts.
- **Bulk re-embedding** only if the corpus grows to thousands of docs.
- **Serving SLO (NFR-2)** — Colab/Kaggle are batch sandboxes with session
  limits, not hosting; if the <2 s SLO must hold at v1.0, that's a real
  GPU box or a hosted-LLM fallback, decided in the Phase 3 arch pass.

## Phase 0 — Make it run, make it honest

Exit criterion (SRS §12): a fresh clone runs ingestion and answers a query
end to end. Exit ⇒ tag `v0.1`.

| Story | Title | Closes | Depends | Status |
| --- | --- | --- | --- | --- |
| [S0-1](phase-0/S0-1-repo-hygiene.md) | Repo hygiene: gitignore + purge artifacts | ISS-09 | — | Done 2026-09-18 |
| [S0-2](phase-0/S0-2-uv-init.md) | uv project: pyproject, pinned 3.12, lockfile | ISS-08 | S0-1, DEC-1 | Done 2026-09-18 |
| [S0-3](phase-0/S0-3-collapse-trees.md) | Collapse v1/v2/temp into one package | ISS-10, ISS-12 | S0-2 | Done 2026-09-19 |
| [S0-4](phase-0/S0-4-config-repair.md) | Config repair: keys, paths, corpus rename | ISS-01, ISS-02, ISS-11, DEC-4 | S0-3 | Done 2026-09-25 |
| [S0-5](phase-0/S0-5-langchain-migration.md) | Migrate code to resolved LangChain version | ISS-17 | S0-4, DEC-1 | Done 2026-09-25 |
| [S0-7](phase-0/S0-7-docs-layout.md) | Project docs into `docs/`; invert the CLAUDE.md corpus rule | DEC-4 | S0-4 | Done 2026-09-25 |
| [S0-6](phase-0/S0-6-end-to-end-proof.md) | End-to-end proof: ingest + answered query | — (exit) | S0-5, S0-7, DEC-2 | Done 2026-09-25 (commit + tag pending) |

Execution order follows the **Depends** column, not the story number: S0-7 was
added after the initial sharding (DEC-4) and runs between S0-5 and S0-6, so the
`v0.1` tag ships the final layout.

## Decisions log

| ID | Decision | Status | Notes |
| --- | --- | --- | --- |
| DEC-1 | LangChain: migrate to 1.x vs pin legacy 0.2.x | **Resolved 2026-09-18: migrate to 1.x** | Verified resolve: core 1.6.3, community 0.4.2, ollama 1.1.0, huggingface 1.2.2, text-splitters 1.1.2, classic 1.0.8 (transitive only). Rules: app code imports only `langchain_core` / `_text_splitters` / `_community` / `_huggingface` / `_ollama`; chain is hand-composed LCEL, no `langchain_classic` imports in Phase 0–1; PyTorch index `explicit = true`. Full rationale: ARCHITECTURE.md §0.1. |
| DEC-2 | Query LLM: pull `qwen2:7b` vs repoint to already-pulled `mistral` | **Resolved 2026-09-18: `mistral`, `phi3` fallback** | Same weight class as qwen2:7b, so the reason is zero download + known-good here, not RAM. `qwen2_ollama` entry stays in config unused. All Ollama entries: `temperature 0`, `num_ctx 4096`, `num_predict 512`, `validate_model_on_init true`; retriever `k 5`. Revisit with the Phase 2 eval harness. |
| DEC-4 | Repo layout: `docs/` currently holds the RAG corpus, colliding with the universal convention that `docs/` is project documentation | **Resolved 2026-09-18: corpus → `corpus/`, `docs/` becomes project documentation** | Maintainer decision. The old name needed a CLAUDE.md hard rule to stay safe, and the text loader claims `.md`, so a project doc dropped in there gets embedded into the index. Renamed in S0-4 (which rewrites every path anyway); `SRS.md` / `WORKFLOW.md` / `ARCHITECTURE.md` move into `docs/` in S0-7, which also inverts the CLAUDE.md rule. `CLAUDE.md` and `README.md` stay at root (auto-load; GitHub renders README from root only). `stories/` stays at root as working state. |
| DEC-5 | `langchain-community` was sunset 2026-05-22 (issue #674): frozen, unmaintained, warns on import. We use it for FAISS, the loaders, and the disabled cross-encoder | **Resolved 2026-09-25: staged exit, keep through Phase 0** | Nothing is broken; the lock pins it and `langchain-core<2` bounds drift. Exit rides planned work: loaders → ~40 lines of own code on `pypdf`/`docx2txt` (Phase 1, with the loader-selection item); cross-encoder → `sentence-transformers` directly (Phase 2 reranker); FAISS → `langchain-qdrant` (Phase 3). After Phase 3 both `langchain-community` and its transitive `langchain-classic` leave `pyproject.toml`. No official standalone FAISS/loader package exists; the unofficial `langchain-faiss` 0.1.1 is rejected on supply-chain grounds. |
| DEC-3 | Vector store for Phase 3: Qdrant vs pgvector | **Lean: Qdrant** (decide in Phase 3 arch pass) | Native hybrid dense+sparse, one container, no Postgres to run. pgvector only if a Postgres already exists in the deployment. Store-specific code is confined to `vectorstore.py` so either works. |

## Backlog (discovered, not yet storied)

- `scripts/fetch_dataset.py` (moved from `docs/dataset.py` in S0-1) uses a
  HuggingFace `/blob/` URL that downloads HTML, no timeout, and writes into
  the corpus dir (ISS-18) — Phase 1 story.
- Loader selection via `pipeline.ingestion.loaders` keyed by extension; drop
  the `extensions` key from `_target_` dicts so there is one instantiation
  path (`build_object({**cfg, "file_path": path})`). Needed before the
  Pydantic schema. Phase 1.
- `build_rag_chain` must not mutate the config it is given (falls out of a
  frozen Pydantic model). Phase 1. **Confirmed live** at `src/rag_qa/chain.py:31`:
  `resolve_ref` hands back a live reference into the config dict and the next
  line writes a constructed retriever object into it, so "inert config" stops
  being inert as soon as the reranker path runs. Harmless while every call
  re-reads the YAML; it bites the moment a config is reused (Phase 2 API, eval
  loops). **S0-5 already owns the immediate fix** (its scope forbids mutation);
  this entry covers the structural guarantee.
- Embedder resolution is duplicated: `chain.py:23` and `ingest.py:65-66` both
  do `build_object(resolve_ref(cfg, cfg["pipeline"]["ingestion"]["embedder"]))`.
  The two must agree or the index and queries use different vectors, so it wants
  one shared helper rather than a convention. Phase 1.
- `vectorstore.create_store`/`open_store` take the whole config to read one key
  (`vector_store_path`). Narrow to the path itself so the Phase 3 store swap has
  a smaller contract. Phase 1 or Phase 3, whichever touches it first.
- `README.md` still documents the deleted `v1/`/`v2/` layout, `pip install -r
  v1/requirements.txt`, a tracked screencast and image ingestion that never
  existed — actively wrong on a public repo as of S0-3. S0-6 owns the rewrite;
  if S0-6 slips, this is worth a standalone fix (ISS-20).
- Error handling: ISS-05 (collect ingestion failures, non-zero exit) and
  ISS-06 (REPL try/except around invoke). Phase 1.
- GPU embedder path is declarable but not installable: the `_cuda` embedder
  entries need a CUDA torch build, while the lockfile pins CPU-only torch
  (DEC-1 rule 2). Add an optional `cuda` extra with a CUDA torch index so
  `uv sync --extra cuda` makes those entries real, for the Colab/Kaggle
  re-embedding path. Phase 1.
- Device selection would be cleaner as one setting than as duplicated embedder
  entries per device (`minilm_cpu` / `minilm_cuda` differ by one field).
  `pydantic-settings` arrives in Phase 1 and handles env-var interpolation
  properly, e.g. `device: ${RAG_EMBED_DEVICE:-cpu}`; do it there rather than
  hand-rolling interpolation in `load_config` now. Phase 1.
- Malformed configs fail with raw internal errors instead of actionable ones:
  `paths: {data: }` raises a `pathlib` `TypeError`, an empty YAML file raises
  `AttributeError` on `NoneType`. The Pydantic schema (FR-8) is the right place
  to fix this, not ad-hoc guards in `load_config`. Phase 1.
- Hosted-LLM fallback: `groq_llama3` config entry behind an optional extra +
  `.with_fallbacks()` in `chain.py`. Phase 1.
- `langchain-classic` is referenced by the (disabled) reranker component but is
  only a transitive dependency via `langchain-community`. If Phase 2 keeps the
  `ContextualCompressionRetriever` + `CrossEncoderReranker` shape instead of a
  hand-rolled Runnable, declare it explicitly in `pyproject.toml`. Phase 2.
- RAGAs judge must be pointed at local Ollama explicitly (default is OpenAI);
  a full metric sweep on CPU is hours — Colab/Kaggle offload candidate. Phase 2.
- Ingestion manifest records embedder model name + dimension; refuse to open
  an index built with a different embedder. Phase 2.
- NFR-2 (<2 s first token) is unachievable CPU-only — revise the SLO or plan
  GPU serving in the Phase 3 arch pass.

- Turn the S0-5 deprecation gate into a pytest (record warnings in-process,
  allow only the DEC-5 sunset notice, keep the negative control). Never use a
  command-line `-W error` gate here: `langchain_core` overrides it on import.
  Phase 1.
- `rag-ingest` and `rag-query` take no arguments, so `rag-ingest --help` starts
  a real ingestion run. Add `argparse` with `--help` and `--config`. Phase 1.
- CLI "Sources" lists what was retrieved, not what the answer used, so a
  refusal still shows a source. Fix with a relevance threshold or citation
  parsing, tuned by the eval harness. Phase 2.
- `ChatOllama` leaves its HTTP client open (`ResourceWarning: unclosed socket`
  at exit). Harmless in the CLI; the Phase 2 API must own and close the
  chain's client across its lifecycle. Phase 2.
- DEC-5 exit tasks, one per phase: own loaders (Phase 1), cross-encoder via
  `sentence-transformers` (Phase 2), drop `langchain-community` and
  `langchain-classic` from `pyproject.toml` after the Qdrant move (Phase 3).

- Phase 1+ stories: shard after Phase 0 exit via WORKFLOW.md Step 2.

## Done

(nothing yet)
