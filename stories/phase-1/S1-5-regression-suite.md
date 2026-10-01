# S1-5: The Phase-0 regression suite and the coverage gate

| | |
| --- | --- |
| **Status** | In review (2026-10-01) — steps 1–3 passed locally; second review's fixes applied (ISS-09 guard, ISS→guard table, subpackage-aware AST check), pending commit; step 4 (CI) after the push; branch `chore/s1-5-regression-suite` |
| **Closes** | ISS-07, NFR-8 |
| **Depends on** | S1-2, S1-4 (S1-2 owns `ci.yml`, the allowlist and `check_imports`; ARCHITECTURE.md §1.5) |
| **Model** | fable |
| **Plan-first** | no |

## Goal

This is the story that makes the Phase 1 exit criterion — "CI would have caught
every Phase-0 bug" — an executable claim instead of an assertion. Each Phase-0
bug becomes a test case asserting a clear `ConfigError` before anything is
built (at load, or — for ISS-03 only — from `check_imports`), the S0-5 deprecation
gate becomes a pytest, and coverage is gated at 80% so the suite cannot quietly
rot.

## Scope

- **`tests/test_config_regressions.py`**: one parametrized case per historical
  bug, each asserting `ConfigError` with a message that names the problem:

  | Case | Expected to be caught by |
  | --- | --- |
  | `llmS` component key (ISS-01) | `Components` `extra="forbid"` + reference resolution |
  | absolute path in `paths` (ISS-02) | `Paths` validator (NFR-11) |
  | `_target_` outside the allowlist (ISS-04) | prefix check at load |
  | `CrossEncoderRerank`, nonexistent attribute (ISS-03) | `check_imports` on a fixture config that references the reranker |
  | dead `vector_stores` block (ISS-11) | `extra="forbid"` |
  | `paths: {data: }` → `TypeError` | typed `Paths` field |
  | empty YAML → `AttributeError` | `RagConfig` required fields |

  ISS-03 is the one case `load_config` alone cannot catch, **by design**: DEC-7
  says loading only string-checks prefixes and never imports, and a misspelled
  class under an allowed prefix passes a string check. So that case loads the
  fixture, then asserts `check_imports` raises `ConfigError` naming the target.
  The test name says so, so nobody later "fixes" it by importing at load.

- **Real-config resolution test** (SRS §10): load the repo's actual
  `config.yaml`, **import-check every referenced `_target_`** (via
  `check_imports`, nested targets included) and
  **prefix-check the unreferenced ones**. The split is deliberate: the disabled
  reranker entry gets validated without importing `langchain_classic`, so
  ADR-007 holds.
- **`tests/test_deprecations.py`**: the S0-5 gate as a pytest — record warnings
  in-process over imports *and* full chain construction, allow only the DEC-5
  `langchain-community` sunset notice, fail on any `LangChainDeprecationWarning`.
  Keep the **negative control** (legacy `langchain_community.llms.Ollama` must
  trip it); a check with no known-bad case run against it is an assumption, not
  a check.
  - Never a command-line `-W error` gate: `langchain_core` calls
    `warnings.filterwarnings("default", ...)` on import and overrides it. That
    is what made the original gate structurally unable to fail (S0-5,
    Discovered).
- **`pytest-cov`** added to the dev group; `--cov=rag_qa --cov-fail-under=80`
  added to the CI workflow, with `evaluate.py` omitted (Phase 2, needs the
  `eval` extra).
- Fill coverage gaps left by earlier stories to clear 80%.
- `tests/conftest.py`: shared fixtures — `DeterministicFakeEmbedding`
  (`langchain_core.embeddings`, verified present in core 1.6.3), a fake chat
  model, a config factory writing scratch YAML. **No network, no Ollama, no
  model download anywhere in the suite.**

## Out of scope

- The RAGAs quality gate and `eval_dataset.jsonl` (Phase 2, FR-7). Coverage here
  is about code paths, not answer quality.
- mypy (S1-6) and pip-audit (S1-8).
- Raising the gate above 80% — NFR-8 says 80%; a higher number invites
  coverage-chasing tests.

## Verification

```bash
# 1. every Phase-0 bug is caught, and the message is readable
uv run pytest tests/test_config_regressions.py -v

# 2. the deprecation gate passes AND its negative control still bites
uv run pytest tests/test_deprecations.py -v

# 3. the whole suite, hermetic: no network and no Ollama, and it must still pass.
#    An unprivileged network namespace cuts both without touching the system
#    service (Ollama runs as a *system* unit, so `systemctl --user` cannot stop it).
#    -c maps you as yourself; -r (root) would correctly skip the chmod-000 tests.
unshare -cn sh -c 'HF_HUB_OFFLINE=1 .venv/bin/python -m pytest --cov=rag_qa -q -rs'
uv run pytest --cov=rag_qa --cov-report=term-missing --cov-fail-under=80

# 4. CI green on the pushed branch
gh run list --limit 1
```

### Results (2026-10-01)

**Risk call: branch** (`chore/s1-5-regression-suite`, the first branch under the
new convention). This story turns on a *blocking* coverage gate and changes CI's
Python. A broken gate on `main` would block every later push, so it proves
itself on a PR first.

**Coverage before writing anything:** already 91% (96% with `evaluate.py`
omitted), so "fill gaps to clear 80%" was not the work. The real gap was
`chain.py` at **62%**: `build_rag_chain` had never run in the suite.

**The bugs reproduced are the literal v2 shapes, from git.** The original
`v2/config.yaml` (`ee86e1f`) was read before writing the cases. It held more than
Appendix A listed. ISS-03's reranker block was wrong three ways:

- a `reranker:` kind key, where the pipeline expected `rerankers`;
- a `langchain.` meta-package target;
- the `CrossEncoderRerank` typo.

ISS-02 and ISS-11 also lived in top-level `data_path` / `vector_store_path` keys
and in an ingestion `vector_store:` reference. Each shape is its own case.

| Check | Result |
| --- | --- |
| 1. `pytest tests/test_config_regressions.py -v` | pass: **12 passed**, names prefixed with the issue ID (`test_iss01_…` … `test_phase0_…`). Each message names the file, the location and the fix, e.g. `components: unknown key 'llmS' (did you mean 'llms'?)`, `paths: 'data' is an absolute path ('/home/harshit/RAG_System/docs') in the config file…`, `components.rerankers.cross_encoder.base_compressor._target_: Could not import '…CrossEncoderRerank'` |
| 1b. **mutation test of the suite** (scratch copy, one protection disabled at a time) | unknown keys allowed → the 5 key-typo cases fail; allowlist off → the 2 allowlist cases fail; absolute paths accepted → the ISS-02 path case fails; `check_imports` neutered → the ISS-03 class-name case fails. No mutation tripped an unrelated case: each protection is pinned by exactly the tests that depend on it |
| 2. `pytest tests/test_deprecations.py -v` | pass: **2 passed**. In a fresh interpreter, ingest + build + invoke + stream records exactly one warning, the DEC-5 sunset notice (no third-party noise, checked first with a prototype). **Negative control:** legacy `Ollama(model="x")` → `LangChainDeprecationWarning`, flagged |
| 3. whole suite with **no network and no Ollama** | pass: inside `unshare -cn` (Ollama and internet both unreachable), **169 passed, 0 skipped**, coverage 97.90%. Under `unshare -rn` (mapped to root), the 3 chmod-000 tests correctly *skip*: root can list any folder |
| 3b. coverage gate | `Required test coverage of 80.0% reached. Total coverage: 97.90%`; `chain.py` 62% → **93%** (2 lines left: the reranker branch, disabled until Phase 2) |
| architecture tests (board item) | **17 passed**; negative controls in a scratch copy: `ingest` importing `chain`, `langchain_core` in `config.py`, and a `langchain_classic` import in `cli.py` are each caught by exactly their test |
| ruff (whole project) | All checks passed |
| 4. CI | after the push |

## Review notes for the human

Read the regression test names against the ISS list in `docs/SRS.md` Appendix A
— the point of this file is that a future reader can see each historical bug is
pinned. Then check the suite genuinely runs with Ollama stopped and the network
down; a test that quietly downloads MiniLM will pass locally and hang in CI.
Last, look at what the coverage gate excludes: only `evaluate.py` should be
omitted.

## Discovered

- **The v2 reranker's bare model string still isn't caught before build.**
  ISS-03's block also passed `model: "cross-encoder/…"` (a string) where
  `CrossEncoderReranker` needs an object. That is a leaf kwarg, and leaves are
  open (ADR-010), so neither loading nor `check_imports` sees it; it fails when
  the component is built. Acceptable while the reranker is disabled. Phase 2's
  reranker story should build it once in a test.
- **Verification step 3 as written could not run here.** `systemctl --user stop
  ollama` addresses a user unit; Ollama runs as a system unit (`sudo` needed).
  An unprivileged network namespace (`unshare -cn`) cuts the internet *and*
  Ollama without touching any service. Note `-c`, not `-r`: mapped to root,
  the chmod-000 tests rightly skip themselves.
- **In-process warning checks are blind inside pytest.** Import-time warnings
  fire once per process, and earlier tests have already imported LangChain.
  The deprecation gate and the architecture checks therefore run in fresh
  subprocesses. It is the S0-5 lesson again: a check that cannot fail is an
  assumption.

### Second review (2026-10-01)

Verified independently: the coverage gate blocks with CI's exact command
(`pytest --cov=rag_qa`, threshold from config only, raised to 99.9% in a scratch
coverage config: `FAIL Required test coverage…`, pytest **exit 1**), and the
whole suite passes inside `unshare -cn` (169/169, 97.90%).

- **ISS-09 had no guard.** The exit criterion is *every* Phase-0 bug. Nothing in
  the suite or CI read what git tracks, so a `git add -f vectorstore/…` would have
  passed. `.gitignore` only keeps *untracked* files out, which is how ISS-09
  happened. **Fixed in the review follow-up:** `tests/test_repo_hygiene.py` runs
  `git ls-files` against one pattern (bytecode, the FAISS index, media, `.save`
  backups, parquet; the corpus PDFs stay allowed on purpose). It is a pytest,
  not a shell step in `ci.yml`, so it runs locally too and has controls:
  - a negative control that force-adds an index in a scratch git repo the test
    creates;
  - the pattern checked against ISS-09's own files and their lookalikes
    (`vectorstore.py`, corpus PDFs).

  Replayed read-only over `ee86e1f`, the tree from before S0-1's cleanup, it
  flags 9 files, including the screencast, `__pycache__` and the parquet.
- **No issue-by-issue map existed**, so "CI would have caught every Phase-0 bug"
  was still a claim. **Added:** a table in `test_config_regressions.py`'s
  docstring, mapping each ISS to the test or CI step that catches it, or saying
  why CI cannot and who owns it. Building it found one gap: **ISS-12** (dead code
  after `return`) is caught by nothing yet. Ruff 0.16 has no unreachable-code
  rule (checked), while mypy's `warn_unreachable` flags exactly that shape
  (checked) → S1-6, on the board.
- **The legacy-import AST check read only top-level files** (`glob("*.py")`), so
  Phase 2's likely `rag_qa/api/` subpackage would never have been scanned.
  **Fixed:** `rglob`, with a test over a scratch package that failed before the
  change (`assert set() == {'api/routes.py'}`).
- **Note, not changed:** the deprecation gate fails on *any* third-party
  `DeprecationWarning` except DEC-5's notice, as the scope asked. A torch or
  transformers upgrade can turn CI red for a non-LangChain reason. The fix then
  is a named, reasoned allowance beside `SUNSET_NOTICE`, not a looser predicate.

After the follow-up: **181 passed** (12 new), coverage 97.90%, `ruff` clean;
the new tests also pass inside `unshare -cn`.

## Deviation from plan

- **No "fill coverage gaps to 80%" work was needed** (91% before, 96% with
  `evaluate.py` omitted). The effort went to `tests/test_chain.py`, which pins the
  S0-5 chain contract hermetically:
  - `invoke` returns `question` / `context` / `answer`;
  - `stream` sends the context before the first answer token, as a separate
    chunk;
  - building does not alter the config;
  - `format_docs` numbers chunks as the CLI does.

  It took `chain.py` from 62% to 93% because the contract is worth pinning,
  not for the number.
- **The coverage threshold lives in `pyproject.toml`** (`[tool.coverage]`:
  `fail_under = 80`, `evaluate.py` omitted), and CI runs `pytest --cov=rag_qa`.
  A local `pytest --cov` then enforces exactly what CI does, instead of the
  flag existing only in the workflow.
- **Three board items rode along**, as the board had queued them for S1-5:
  - CI pinned to uv-managed Python (`UV_PYTHON_PREFERENCE: only-managed`),
    with the `cache-python` comment fixed;
  - two-dot extension keys (`.tar.gz`) rejected, with a regression case;
  - the ADR-009 check made a runtime test (`tests/test_architecture.py`),
    plus an AST version of ADR-007's legacy-import grep.
- **Shared fakes in `conftest.py`:** `use_fake_embedder` / `use_fakes` (the
  `DeterministicFakeEmbedding` and `FakeListChatModel` component entries) and a
  `fake_rag` fixture (the real ingest over `sample_corpus` into a scratch
  index). `test_ingest.py` now uses the shared helper instead of its own copy.
- The prompt-placeholder validator (an S1-1 board item, "if the maintainer
  agrees") was **not** included; it is still open.
