# Board

> Updated at every story close-out. This file + the story files ARE the
> project memory across sessions.

## Now

**Now: [S1-8](phase-1/S1-8-phase-1-exit.md) is verified and awaiting the
maintainer** (2026-10-02): review, PR on `chore/s1-8-phase-1-exit`, green CI,
merge, then tag `v0.2` on the merge commit. Everything else in its Verification
section ran and passed; see the story. **After the tag: the Phase 2 architecture
pass** (WORKFLOW Step 1, fable, plan mode), then shard Phase 2. Inputs it must
take: the re-deferred Phase 2 backlog lines below, the "API never supplies
components" constraint, and the Phase 3 Docker rule from S1-7 (install with
`uv sync` or `uv export`, never `pip install .`). Phase 2's first story bumps
the version to `0.3.0.dev0`. Done so far: S1-1 `b22a37a`; S1-2 `1f1d6ff` + follow-up `f68eee3`; S1-3 `3b0e77f`
+ follow-up `ddcd98f` (PR #1); S1-4 `eff50ea` + follow-up `a6f21b3` + DEC-13
`20c4a6a` (PR #2); S1-5 `e3927a4` … `4efa53d` (PR #3, 8 commits); S1-6
`ff61088` … `be545a4` (PR #4, 9 commits); S1-7 `6a50792` … `234f437` (PR #5,
7 commits). **CI exists**: every story closes on a green run, and from S1-8 it
gates ruff → mypy → pytest + coverage → pip-audit.

**Commit and branch convention (maintainer, 2026-10-01):** Conventional Commits +
Conventional Branch, as set out in CLAUDE.md's hard rules. Story and SRS IDs go in a `Refs:` trailer,
not the header. History up to S1-4 predates it and is left as is.

**Branch rule (maintainer, 2026-09-30):** risky stories go on a feature branch
and a PR, not straight to `main`. The call is made and stated at the start of each
story.

**The Phase 1 architecture pass is done** (2026-09-25): `ARCHITECTURE.md` has a
confirmed Phase 1 section (§1.1–§1.7) and DEC-6 … DEC-12 are resolved below.
Implementation sessions follow it and do not re-litigate it. Every Phase 1
backlog item was resolved or re-confirmed in that pass — each line in the
Backlog now names its owning story or its re-deferral.

**Phase 0 is complete and released as `v0.1`** (2026-09-25): tag on
`ae15a23`, pushed to GitHub along with `main`. The exit criterion was met
literally: a fresh clone ran `uv sync`, ingested the corpus and answered
through `rag-query`.

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
| `ae15a23` | S0-6 end-to-end proof, README, 0.1.0 — tagged `v0.1` |

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
| Phase 0 | `v0.1` | 0.1.0 — **released 2026-09-25** (`ae15a23`) |
| Phase 1 | `v0.2` | 0.2.0 |
| Phase 2 | `v0.3` | 0.3.0 |
| Phase 3 (SRS complete) | `v1.0` | 1.0.0 |

The final story of each phase performs the tag + version bump at close-out.

## Human prerequisites (do these anytime, no session needed)

- [x] `gh auth login` — done 2026-09-24 as `harshit-05` (HTTPS, keyring;
      scopes: repo, read:org, gist). Needed to push and for Phase 1 CI/PR work.

- [x] **`workflow` scope** — granted 2026-09-29, lost when `gh` was logged in
      again (by 2026-10-02 the token had `admin:public_key gist read:org repo`
      and S1-7's push was refused), and restored on 2026-10-02 with
      `gh auth refresh -h github.com -s workflow`. Git pushes to github.com
      through `gh auth git-credential`. Every push that changes
      `.github/workflows/*.yml` needs this scope, so check with `gh auth status`
      after any re-login. **Note:** the token now lives in
      `~/.config/gh/hosts.yml` (plain text), not the system keyring as before.
      Re-login with keyring storage if that was not intended.

- [x] Resolve DEC-1 / DEC-2 — done in the Phase 0 arch pass, 2026-09-18.

- [ ] Nothing else: uv ✓, Python 3.12 (uv-managed) ✓, Ollama daemon ✓,
      Docker 29.6.2 ✓ (Phase 3), disk 311 GB free ✓. **No MCP servers are
      required for any phase** — built-in tools cover the whole workflow.

## Pre-flight caveats (read before a story session)

Numbered because stories cite them by number; resolved ones stay as one line so
those references still land. Items 9 onward were learned in Phase 1; each one
cost a session something.

1. **Resolved (2026-09-18), historical.** The three pending commits (baseline,
   S0-1, S0-2) landed as `ee86e1f`, `f702dc8` + `e781b17` and `0987c60`. The
   lesson still holds: `git reset` does not untrack anything, and `.gitignore`
   never untracks an already-tracked file. `tests/test_repo_hygiene.py` now reads
   `git ls-files` on every CI run and fails if an artifact is tracked (ISS-09).
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
5. **Resolved (S0-1, S0-4), historical.** The parquet files were untracked,
   `docs/dataset.py` became `scripts/fetch_dataset.py`, and `corpus/` holds only
   the three PDFs.
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
   ~6 GB free, and record which model produced the transcript. Run `ollama ps`
   first: Ollama keeps the last model resident for ~5 min, so `free -h`
   under-reports, and `ollama stop <model>` gives that memory back (S1-1).

9. **Scratch ingests set both paths.** Any command that ingests a scratch corpus
   sets `RAG_DATA_PATH` **and** `RAG_VECTOR_STORE_PATH`. With only the first,
   the scratch build overwrites the real index (S1-3, S1-4).
10. **Check a guard the way CI runs it.** A plain `uv run` re-syncs the
    environment first, so it can repair the very state a check is meant to
    catch and then pass. Guards on the installed environment use
    `uv run --no-sync` (S1-7).
11. **Never gate warnings with `-W error` on the command line.**
    `langchain_core` resets the warning filters on import, so the gate cannot
    fail. Record warnings in-process, as `tests/test_deprecations.py` does (S0-5,
    S1-5).
12. **A plain `pip-audit` skips torch.** It looks up `2.14.0+cpu` on PyPI, finds
    no such release and only warns. CI's Audit step strips the local label, and a
    negative control showed that an old torch then fails the gate (S1-8).
13. **Pushing a workflow change needs `gh`'s `workflow` scope.** A re-login drops
    it, and the push is refused. Run `gh auth status` after any re-login (S1-7).
14. **Attack a security check; don't only confirm it accepts the good input.**
    The prefix-only allowlist passed every positive test and was bypassed twice
    (S1-2, ADR-020).

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
| [S0-6](phase-0/S0-6-end-to-end-proof.md) | End-to-end proof: ingest + answered query | — (exit) | S0-5, S0-7, DEC-2 | Done 2026-09-25 |

Execution order follows the **Depends** column, not the story number: S0-7 was
added after the initial sharding (DEC-4) and runs between S0-5 and S0-6, so the
`v0.1` tag ships the final layout.

## Phase 1 — Make it trustworthy

Exit criterion (SRS §12): CI is green and would have caught every Phase-0 bug.
The second half is literal — `tests/test_config_regressions.py` (S1-5) encodes
each Phase-0 bug as a case. Exit ⇒ tag `v0.2`, version 0.2.0.
Design: ARCHITECTURE.md §1.1–§1.7.

| Story | Title | Closes | Depends | Status |
| --- | --- | --- | --- | --- |
| [S1-1](phase-1/S1-1-config-schema.md) | Validate the config with a frozen Pydantic model | FR-8, ISS-01, ISS-11, NFR-11 | — | Done 2026-09-26 |
| [S1-2](phase-1/S1-2-target-allowlist.md) | Allowlist `_target_` imports; stand up CI | ISS-04 | S1-1 | Done 2026-09-29; review follow-up 2026-09-30 pending commit |
| [S1-3](phase-1/S1-3-own-loaders.md) | Own loaders, extension map, recursive walk | ISS-13, FR-2, DEC-5 step 1 | S1-1 | Done 2026-09-30 (PR #1) |
| [S1-4](phase-1/S1-4-error-handling-cli.md) | Explicit error handling and a real CLI surface | ISS-05, ISS-06, ISS-18, NFR-7 | S1-3 | Done 2026-10-01 (PR #2) |
| [S1-5](phase-1/S1-5-regression-suite.md) | Phase-0 regression suite and coverage gate | ISS-07, NFR-8 | S1-2, S1-4 | Done 2026-10-01 (PR #3) |
| [S1-6](phase-1/S1-6-types-and-lint.md) | Type annotations, ruff and mypy configuration | ISS-19, NFR-9, ISS-12 | S1-5 | Done 2026-10-02 (PR #4) |
| [S1-7](phase-1/S1-7-cuda-extra.md) | GPU embedder path installable (CPU/CUDA torch variants) | FR-1 (GPU axis) | S1-1 | Done 2026-10-02 (PR #5) |
| [S1-8](phase-1/S1-8-phase-1-exit.md) | Phase 1 exit: audit gate, doc hygiene, 0.2.0 | — (exit) | S1-6, S1-7 | Verified 2026-10-02; awaiting PR, CI and tag |

CI lands in S1-2, not at the end, so every later story closes on green rather
than the whole phase arriving unverified at once. Each gate is added by the
story that makes it passable (ruff + pytest in S1-2, coverage in S1-5, mypy in
S1-6, pip-audit in S1-8) and blocks from that moment (DEC-11). S1-2 lands the ruff
gate on a tree that already has 4 default-rule errors, so it suppresses them with
`noqa` tags naming S1-4, and S1-4 removes them. S1-7 depends only on S1-1 and can
be run whenever convenient before S1-8.

## Decisions log

| ID | Decision | Status | Notes |
| --- | --- | --- | --- |
| DEC-1 | LangChain: migrate to 1.x vs pin legacy 0.2.x | **Resolved 2026-09-18: migrate to 1.x** | Verified resolve: core 1.6.3, community 0.4.2, ollama 1.1.0, huggingface 1.2.2, text-splitters 1.1.2, classic 1.0.8 (transitive only). Rules: app code imports only `langchain_core` / `_text_splitters` / `_community` / `_huggingface` / `_ollama`; chain is hand-composed LCEL, no `langchain_classic` imports in Phase 0–1; PyTorch index `explicit = true`. Full rationale: ARCHITECTURE.md §0.1. |
| DEC-2 | Query LLM: pull `qwen2:7b` vs repoint to already-pulled `mistral` | **Resolved 2026-09-18: `mistral`, `phi3` fallback** | Same weight class as qwen2:7b, so the reason is zero download + known-good here, not RAM. `qwen2_ollama` entry stays in config unused. All Ollama entries: `temperature 0`, `num_ctx 4096`, `num_predict 512`, `validate_model_on_init true`; retriever `k 5`. Revisit with the Phase 2 eval harness. |
| DEC-4 | Repo layout: `docs/` currently holds the RAG corpus, colliding with the universal convention that `docs/` is project documentation | **Resolved 2026-09-18: corpus → `corpus/`, `docs/` becomes project documentation** | Maintainer decision. The old name needed a CLAUDE.md hard rule to stay safe, and the text loader claims `.md`, so a project doc dropped in there gets embedded into the index. Renamed in S0-4 (which rewrites every path anyway); `SRS.md` / `WORKFLOW.md` / `ARCHITECTURE.md` move into `docs/` in S0-7, which also inverts the CLAUDE.md rule. `CLAUDE.md` and `README.md` stay at root (auto-load; GitHub renders README from root only). `stories/` stays at root as working state. |
| DEC-5 | `langchain-community` was sunset 2026-05-22 (issue #674): frozen, unmaintained, warns on import. We use it for FAISS, the loaders, and the disabled cross-encoder | **Resolved 2026-09-25: staged exit, keep through Phase 0** | Nothing is broken; the lock pins it and `langchain-core<2` bounds drift. Exit rides planned work: loaders → ~40 lines of own code on `pypdf`/`docx2txt` (Phase 1, with the loader-selection item); cross-encoder → `sentence-transformers` directly (Phase 2 reranker); FAISS → `langchain-qdrant` (Phase 3). After Phase 3 both `langchain-community` and its transitive `langchain-classic` leave `pyproject.toml`. No official standalone FAISS/loader package exists; the unofficial `langchain-faiss` 0.1.1 is rejected on supply-chain grounds. |
| DEC-3 | Vector store for Phase 3: Qdrant vs pgvector | **Lean: Qdrant** (decide in Phase 3 arch pass) | Native hybrid dense+sparse, one container, no Postgres to run. pgvector only if a Postgres already exists in the deployment. Store-specific code is confined to `vectorstore.py` so either works. |
| DEC-6 | Config validation: how far Pydantic reaches into the call sites | **Resolved 2026-09-25: frozen `RagConfig`, attribute access everywhere** | `load_config()` returns a frozen model, not a dict; typed component kinds (`extra="forbid"`) make ISS-01 and ISS-11 unwritable; component leaves stay `extra="allow"` (ADR-010); retrievers get their own `RetrieverSpec` (no `_target_` — typing them as components fails the real config, verified). Non-mutation becomes structural via the access path: call sites only get fresh copies from `component(ref).spec()`/`.kwargs()` — `frozen=True` alone does not freeze nested dicts (verified). **Verified trap:** a field literally named `_target_` is silently dropped by Pydantic (leading underscore ⇒ private attribute; `model_dump()` returned `{}`) — the schema must use `target: str = Field(alias="_target_")` and dump `by_alias=True`. S1-1. ARCHITECTURE.md §1.1, §1.4. |
| DEC-7 | Where the `_target_` allowlist lives, and whether it is configurable | **Resolved 2026-09-25: hardcoded, inside `import_from_string`** | Enforcing at the import funnel rather than in `build_object` covers `ingest.py`'s direct call — the gap ADR-015 flagged. Not config-overridable and no env escape hatch: an allowlist the config can edit is not an allowlist. `langchain_classic.` is allowed as a *config* prefix (disabled reranker entry); ADR-007 governs *source* imports and is unaffected. Load string-checks prefixes (nested too) and never imports; a separate `check_imports` helper, used by tests, catches misspelled classes (ISS-03). S1-2. Closes ISS-04. |
| DEC-8 | DEC-5 step 1: replace the `langchain-community` loaders now or later | **Resolved 2026-09-25: now, in Phase 1** | Own `PdfLoader`/`DocxLoader`/`TextLoader` on `pypdf`/`docx2txt`; `extensions` leaves the component entries for `pipeline.ingestion.loaders`, giving one instantiation path; discovery becomes recursive (ISS-13). Afterwards `langchain_community` is imported only by `vectorstore.py`. Risk is citation metadata, so the story records the `v0.1` chunk-count + citation baseline before changing anything and must reproduce it. S1-3. |
| DEC-9 | NFR-6 (timeout / retry / circuit-breaking): Phase 1 or later | **Resolved 2026-09-25: Phase 1 does error handling only** | ISS-05 (collect failures, exit non-zero — NFR-7) and ISS-06 (REPL try/except) are S1-4. Timeout, retry-with-backoff and circuit-breaking (SRS §11 Resilience) need the service shape and move to Phase 3 with the serving decision. Deferred deliberately, not by omission. |
| DEC-10 | Device selection: collapse the `_cpu`/`_cuda` embedder entries into one env-driven setting | **Resolved 2026-09-25: no — the entries stay explicit** | The backlog item assumed pydantic-settings interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside YAML values. **It does not** — there is no `${VAR:-default}` expansion for arbitrary values. With the CUDA torch variant (DEC-12), one sync command plus repointing `pipeline.ingestion.embedder` is already a one-line switch, so the four entries restored in `ebcfc2e` stay as FR-1 axes. |
| DEC-11 | CI: host, hermeticity and how hard the gates bite | **Resolved 2026-09-25: GitHub Actions, hermetic suite, blocking gates** | No Ollama, no model download, no network in tests: `DeterministicFakeEmbedding` (verified present in core 1.6.3) + a fake chat model. Gate order: `ruff check` → `mypy` (`disallow_untyped_defs` on `src/rag_qa`) → `pytest --cov-fail-under=80` (omitting `evaluate.py`, Phase 2 + `eval` extra) → `pip-audit`. Each gate blocks from the story that adds it. **No `ruff format --check` in Phase 1** — the tree is unformatted and a whole-tree reformat is the diff that gets rubber-stamped → backlog. |
| DEC-12 | Making the declared GPU embedder entries actually installable | **Resolved 2026-09-25: uv-conflicting CPU/CUDA torch variants; plain `uv sync` stays CPU** | Each variant routed to its own index, `explicit = true` on both. **Mechanism chosen by S1-7's spike**: extras have no default, so torch-only-in-extras would make a plain sync pull PyPI CUDA torch via `sentence-transformers`. Preferred: dependency groups + `default-groups = ["dev", "cpu"]`; fallback: extras with `--extra cpu` on every install path. S1-7 re-runs the S0-2 smoke test on the **installed env** (the lock legitimately holds both variants): core 1.x, `torch 2.14.0+cpu`, zero `nvidia-*`. CUDA is verifiable here as a **resolve only** — this host is CPU-only. |
| DEC-13 | Partial ingestion failure: replace the index with what loaded, or keep the previous one | **Resolved 2026-10-01: replace, exit 1** | Maintainer's call, raised by S1-4's third review. Some documents or folders unreadable → index the rest, save over the old index, exit 1 ("rebuilt without them"). Keeping the old index is safer when unattended, but one persistently bad file would block every update. Phase 1 ingests are hand-run and watched. Phase 2's hash manifest dissolves it: a failed document keeps its previously indexed chunks. Exit codes: 0 all indexed · 1 document/folder unreadable, nothing indexed, or unexpected error (traceback) · 2 cannot start. ARCHITECTURE.md §1.1. |

## Backlog

Reconciled at the Phase 1 exit (S1-8, 2026-10-02). Every Phase 1 line is either
**closed** by the story named, or **re-deferred** with a phase and a reason. The
detail behind each closed line is in its story file.

### Closed in Phase 1

- **S1-1:** the config stops being mutable (`build_rag_chain` gets fresh copies
  only); `create_store` / `open_store` take a path, not the whole config;
  malformed configs (`paths: {data: }`, an empty YAML) fail with a `ConfigError`
  instead of a `TypeError` / `AttributeError`.
- **S1-2:** the `_target_` allowlist (ISS-04). The trust-boundary *doc lines*
  from its second review were written in S1-8: `registry.py` docstring,
  ARCHITECTURE.md §1.8, and README "Known limitations".
- **S1-3:** loaders selected by extension through `pipeline.ingestion.loaders`
  (DEC-8); embedder resolution in one place (`components.build_embedder`).
- **S1-4:** ISS-05 / ISS-06 error handling and exit codes (DEC-9, DEC-13);
  `argparse` with `--help` / `--config`; `fetch_dataset.py` (ISS-18); ignored
  files counted and symlinks never followed (two S1-3 leftovers).
- **S1-5:** the deprecation gate as a pytest; CI on uv-managed Python;
  two-dot extension keys rejected; ADR-009 as a runtime test.
- **S1-6:** ruff from its defaults plus `E501`; `warn_unreachable` (ISS-12's
  guard); the private-field test made behavioural.
- **S1-7:** the GPU embedder path installable (DEC-12).
- **S1-8:** the S1-1 doc items (Ollama `keep_alive` vs `free -h`, now in
  caveat 8 and CLAUDE.md; `gemma2:9b` in CLAUDE.md's environment facts). The
  "scratch ingests set both paths" habit is now pre-flight caveat 9.

### Re-deferred at the Phase 1 exit

- **Backlog — audit-gate policy (S1-8 review).** CI's Audit step does a live
  advisory lookup and blocks, so a new advisory with no fixed release turns every
  PR red, docs-only ones included. Decide an `--ignore-vuln` policy (ID + reason in
  `ci.yml`) or a scheduled audit beside a PR gate for new dependencies. Not needed
  until it first happens.
- **Phase 2 — prompt placeholder check (FR-8 follow-up, found in S1-1).** A
  `human` prompt missing `{context}` silently answers without retrieval. One
  validator on `Prompt`. S1-5 left it open, as the board allowed. It belongs with
  the Phase 2 evaluation harness, which is what would catch an ungrounded
  answer anyway. Ride the first Phase 2 story that touches `schema.py`.
- **Phase 2 — the API must never let a request supply or override components
  (S1-2 second review).** A constraint for the Phase 2 architecture pass, with
  the optional load-time guard that rejects `trust_remote_code` anywhere in the
  config. Re-deferred because it only matters once the config is reachable from
  outside the host.
- **Phase 2 — CLI polish (found in S1-4).** No single item is worth a story, so
  batch them as one story when Phase 2 is sharded, or let them ride the first
  story that touches `cli.py` / `ingest.py`:
  - `rag-query` startup is unguarded: Ollama being down
    (`validate_model_on_init`) and Ctrl-C while models load both print a
    traceback. One startup try/except with a clean message.
  - The connection error doesn't name Ollama (`ConnectError: [Errno 111]`). Add
    the hint "Is Ollama running? `systemctl status ollama`", and stop the empty
    `Answer:` header printing before the error.
  - Zero-count summary lines ("Skipped 0 symlink(s)") print on every run; print
    them only when nonzero.
  - An unreadable corpus *root* reports `./: PermissionError`, exit 1. "Corpus
    directory not readable" with exit 2 would match "not found". "Failed N
    file(s)" also counts folders now; say "item(s)".
  - `fetch_dataset.py`: its corpus guard checks only the repo's `corpus/`, not
    `paths.data` / `RAG_DATA_PATH`; nothing checks the `PAR1` header; and
    `requests` is undeclared (transitive only).
- **Phase 2 — the walk descends into ignored trees** (found in S1-3 and S1-4).
  It walks all of `.git` before discarding it. Performance only. Phase 2's
  incremental ingestion rewrites discovery around the manifest, so prune there
  (`Path.walk` is top-down, and pruning `folders[:]` in place fixes it).
- **Phase 2 — type the YAML boundary; add `py.typed` (found in S1-6).** Add
  `types-PyYAML` as a dev dependency so `pyyaml` stops being `Any`. An empty
  `src/rag_qa/py.typed` makes a misspelled first-party import read as "cannot
  find module". Both are small and optional; ride the first story that touches
  `config.py` or packaging.
- **Phase 3 — the parity tests and the corpus PDF they read** (found in S1-3).
  They are accepted until `langchain-community` leaves. Removing that PDF before
  then fails them loudly, which is right.
- **On the first real GPU run — CUDA 12 fallback (found in S1-7).** `cu130`
  wheels need a recent NVIDIA driver, and this host cannot check that. If
  Colab/Kaggle refuse, add a `cuda12` variant on `cu128`, which caps torch at 2.11.
  This is triggered by an event, not by a phase.
- **Accepted, no action — torch parity between the variants (found in S1-7).**
  CUDA resolves `2.14.1+cu130`, CPU stays at `2.14.0+cpu`.
  `uv lock --upgrade-package torch` aligns them if it ever matters.

### Re-deferred or dropped in the Phase 1 architecture pass (2026-09-25)

- **Dropped — device selection as one env-driven setting.** The item assumed
  `pydantic-settings` interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside
  YAML values; it does not (no `${VAR:-default}` expansion for arbitrary
  values). The CUDA torch variant (DEC-12) makes the switch one line anyway. See **DEC-10**.
- **Phase 3 — hosted-LLM fallback** (`groq_llama3` entry behind an optional
  extra + `.with_fallbacks()` in `chain.py`). Moved out of Phase 1: it widens
  the DEC-1 import surface and needs an API key, for something Phase 1 could
  only exercise against a fake failing LLM. It belongs with the Phase 3 serving
  decision (NFR-2), which is the problem it actually solves.
- **Phase 3 — NFR-6** timeout / retry-with-backoff / circuit-breaking, per
  **DEC-9**.
- **Backlog — `ruff format`.** Not gated in Phase 1: the tree is unformatted and
  a whole-tree reformat is the kind of diff that gets rubber-stamped (DEC-11).
- **Not yet storied — per-loader splitter strategies** (SRS §7.2): chunking
  configurable per document type. The `_target_` mechanism already supports it;
  it needs more splitter entries and a per-loader default, not new architecture.

### Later phases

- Chunk metadata: content hash of the source file + ingestion timestamp
  (SRS §7.3). Phase 2, with the manifest that needs them.
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

- CLI "Sources" lists what was retrieved, not what the answer used, so a
  refusal still shows a source. Fix with a relevance threshold or citation
  parsing, tuned by the eval harness. Phase 2.
- `ChatOllama` leaves its HTTP client open (`ResourceWarning: unclosed socket`
  at exit). Harmless in the CLI; the Phase 2 API must own and close the
  chain's client across its lifecycle. Phase 2.
- DEC-5 exit tasks, one per phase: own loaders (**Phase 1 → S1-3**),
  cross-encoder via `sentence-transformers` (Phase 2), drop
  `langchain-community` and `langchain-classic` from `pyproject.toml` after the
  Qdrant move (Phase 3).
- Phase 2+ stories: shard at each phase exit via WORKFLOW.md Step 2.

## Done

**Phase 0 — shipped as `v0.1` (`ae15a23`, 2026-09-25).** S0-1 repo hygiene ·
S0-2 uv project · S0-3 single package · S0-4 config repair + corpus rename ·
S0-5 LangChain 1.x LCEL chain with streaming · S0-7 docs layout · S0-6
end-to-end proof. Per-story detail in `stories/phase-0/`; the Phase 0 table
above is the index.

**Phase 1 — version 0.2.0, tag `v0.2` pending the S1-8 merge (2026-10-02).**
S1-1 frozen Pydantic config · S1-2 `_target_` allowlist + CI · S1-3 own loaders,
recursive walk · S1-4 error handling, exit codes, real CLI · S1-5 Phase-0
regression suite + coverage gate · S1-6 types and lint · S1-7 CPU/CUDA torch
variants · S1-8 audit gate, doc hygiene, release. As delivered:
ARCHITECTURE.md §1.8; per-story detail in `stories/phase-1/`.

**Architecture passes.** Phase 0 confirmed 2026-09-18 (DEC-1/2/4, later DEC-5);
Phase 1 confirmed 2026-09-25 (DEC-6 … DEC-12), sharded into S1-1 … S1-8.
