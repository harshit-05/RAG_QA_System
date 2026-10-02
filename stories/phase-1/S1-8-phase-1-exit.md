# S1-8: Phase 1 exit — audit gate, doc hygiene, release 0.2.0

| | |
| --- | --- |
| **Status** | Reviewed twice 2026-10-02, both follow-ups applied; CI was green before them; awaiting the follow-ups' CI run, merge and tag |
| **Closes** | — (phase exit; SRS §12 Phase 1) |
| **Depends on** | S1-6, S1-7 |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

Close Phase 1: add the dependency-audit gate, bring the documentation back in
line with what the code actually is, and cut the release. Exit criterion
(SRS §12): **CI is green and would have caught every Phase-0 bug** — the first
half is the workflow, the second half is `tests/test_config_regressions.py` from
S1-5. Exit ⇒ version 0.2.0, tag `v0.2`.

## Scope

- **`pip-audit`** added to CI as a blocking step (OWASP LLM05, SRS §11). If it
  flags something unfixable within this story, record the advisory ID and the
  reason in Discovered rather than silencing the gate quietly.
- **Doc hygiene** — all of the following are currently misleading:
  - `CLAUDE.md`: says `gh` is **NOT** authenticated (it has been since
    2026-09-24, as `harshit-05`); omits `docs/ADR.md` from the read-list; the
    GPU policy needs the clause that the CUDA torch variant (S1-7) exists for
    Colab/Kaggle, not for this host; environment facts are dated 2026-08-01 and
    should carry the re-verification date.
  - `stories/STATUS.md`: pre-flight caveats 1 and 5 describe a pre-S0-1 state
    that no longer exists; add the Phase 1 stories to `## Done` (Phase 0 was
    added there in the Phase 1 architecture pass) and the Phase 1 caveats
    learned on the way.
  - `README.md`: the config shape changed in S1-3 (`pipeline.ingestion.loaders`),
    `--config`/`--help` exist (S1-4), tests and CI exist, and the GPU sync command (S1-7)
    needs its line.
  - `docs/ARCHITECTURE.md`: mark Phase 1 confirmed-and-delivered; note any place
    the delivery differed from §1.1–§1.7.
  - `docs/ADR.md`: add ADR entries for the decisions that generalize — the
    Pydantic `_target_` alias trap, the allowlist at the import funnel, and
    the own-loaders step of the DEC-5 exit. Index them, do not renumber.
- **Release**: version `0.2.0.dev0` → `0.2.0` in **both** `pyproject.toml` and
  `src/rag_qa/__init__.py` (each hardcodes it); commit; `git tag -a v0.2`;
  push both. Per the release mapping in STATUS.md, the final story of a phase
  does this.
- Reconcile the STATUS.md backlog: every Phase 1 line either closed by a story or
  re-deferred with a phase and a reason.

## Out of scope

- Trivy, the container build and the Dockerfile (Phase 3).
- Any Phase 2 work — the FastAPI shape, the reranker, the manifest, RAGAs.
- Rewriting Phase 0 sections of ARCHITECTURE.md. Phase 0 is history; annotate,
  don't revise.

## Verification

```bash
# 1. the full gate, locally and in CI
uv run ruff check src tests && uv run mypy src/rag_qa
uv run pytest --cov=rag_qa --cov-fail-under=80 -q
uv run pip-audit 2>&1 | tail -5
gh run list --limit 1                 # → success on main

# 2. the exit criterion, stated literally
uv run pytest tests/test_config_regressions.py -v    # every Phase-0 bug, caught

# 3. a fresh clone still runs end to end (the v0.1 criterion must not regress)
#    (use the exact sync command README documents — plain `uv sync` unless S1-7
#    chose option (b); and confirm the fresh env has no nvidia-* packages)
git clone . <scratch>/fresh && cd <scratch>/fresh && uv sync
uv pip list | grep -icE 'nvidia'      # → 0
uv run rag-ingest && echo "What is this corpus about?" | uv run rag-query

# 4. docs tell the truth
grep -n "gh auth\|ADR.md\|cuda" CLAUDE.md
grep -rno "\](\([^)]*\.md\)[^)]*)" docs/ stories/ CLAUDE.md README.md   # no dangling links

# 5. release
grep '^version' pyproject.toml        # → 0.2.0
grep __version__ src/rag_qa/__init__.py   # → "0.2.0"
git tag -l | grep v0.2
```

## Review notes for the human

Verification step 3 is the one that matters most: Phase 1 touched the config
shape, the loaders and the dependency extras, so the *Phase 0* exit criterion is
the thing most likely to have silently regressed. Then read `CLAUDE.md` as a
fresh session would — it is auto-loaded into every future session, so a stale
line there is worse than a stale line anywhere else in the repo (ADR-012's
reasoning: a rule you have to remember is a rule that gets forgotten).

## Discovered

- **A plain `pip-audit` never audits torch.** It looks `2.14.0+cpu` up on PyPI,
  finds no such release, and only lists it under "skipped", with exit 0. The
  verification's literal `uv run pip-audit` therefore passes while saying nothing
  about the package with the most CVEs in the tree. CI's Audit step writes
  `uv pip freeze --exclude-editable`, with torch's `+cpu` label stripped, to a
  file, checks that the file is non-empty, and runs
  `pip-audit --no-deps --disable-pip -r` on it (the pipe into `/dev/stdin` was
  replaced at the first review; see Deviation). The CPU wheel is the same
  source release as PyPI's 2.14.0, so the same advisories apply. Negative
  controls: `requests==2.19.0` fails it (3 PYSEC advisories), and so does
  `torch==2.5.1` (4). No advisory is open against the real environment.
- **Two ADR index anchors were broken since they were written** (ADR-010 and
  ADR-015): GitHub keeps the underscores of `_target_` in heading slugs. Fixed,
  and all 18 `.md` links in docs/, stories/, CLAUDE.md and README.md were checked
  with a GitHub-style slugger.
- **`src/rag_qa/__init__.py`'s module list** was missing `loaders` and
  `components` (S1-3). Fixed with the version bump.
- **Two rows of the issue map** in `tests/test_config_regressions.py` still
  pointed forward ("pip-audit → S1-8", "mypy → S1-6"). Both now name the CI
  step that guards them.
- **The exit criterion, read literally against that map:** of the 21 Appendix A
  issues, 14 have a CI guard: 01, 02, 03 (partly), 04, 05, 06, 07, 08, 09, 11,
  12, 13, 17 and 19. ISS-21's REPL half has one too. The rest cannot be CI
  checks:
  - ISS-10 is structural;
  - ISS-14 and ISS-15 are missing features (Phase 2);
  - ISS-16 is a documented invariant;
  - ISS-20 is the README.

  Two are real gaps, both stated in the map rather than hidden:
  - ISS-03's bare model string passes load, because leaf kwargs are open. It is
    owned by Phase 2's reranker story.
  - ISS-18 is a standalone script that CI does not run. It was fixed and checked
    by hand in S1-4.

  "Would have caught every Phase-0 bug" holds for every bug that code or config
  could reintroduce.

## Deviation from plan

- **Fresh-clone step:** `git clone .` would clone the committed `main`, not this
  story's uncommitted tree. The scratch "clone" was therefore built from
  `git ls-files` with working-tree contents. That is the same file set, since the
  story adds no untracked files. Results:
  - `uv sync`, 0 `nvidia-*` packages, torch `2.14.0+cpu`, version 0.2.0;
  - `rag-ingest`: 561 pages, 1,708 chunks (the `v0.1` count), 0 failed, exit 0,
    2:01;
  - the query, with the shipped `mistral` config unmodified (7.6 GB available
    after the maintainer freed RAM): exit 0, 7:06 including model load.
- **First-review follow-up (2026-10-02):** the Audit step could pass while auditing
  nothing. GitHub's default shell has no `pipefail`, and `pip-audit` on an empty
  input prints "No known vulnerabilities found" and exits 0 (reproduced). The step
  now sets `pipefail`, writes the freeze to a file and asserts it is non-empty.
  Controls: real environment exit 0; a failing `uv pip freeze` exit 1. The label
  rewrite is narrowed to `torch==…+cpu`, so another local label (`requests==2.19.0+local`)
  stays visible. README's command matches, and CLAUDE.md's `langchain_classic`
  line now says the reranker entry and allowlist still name it until Phase 2.
  Second review (2026-10-02): README's local audit line now carries CI's guards
  (`pipefail` and a non-empty check, in a subshell), plus `--python .venv`, because
  without a `.venv` a bare `uv pip freeze` froze uv's managed 3.12 and passed. The
  tag instruction now says to tag `main` after the rebase-merge. The Phase 3 Docker
  rule moved from "Now" into the backlog. ADR-021 says about 60 lines (58 in
  `loaders.py`).
  Not adopted: a scheduled audit or `--ignore-vuln` policy for advisories with no
  fix (a live lookup can redden unrelated PRs) → backlog.
- **The Audit step is not a plain `pip-audit`** (see Discovered). The story's
  literal command was run as well, and it reports torch as skipped.
- **Additions:** the trust-boundary doc lines owed from S1-2's review
  (`registry.py` docstring, ARCHITECTURE.md §1.8, README); pre-flight caveats
  9–14 (Phase 1 lessons); ADR-015 marked superseded by ADR-020.
- **Not run here:** `gh run list` and the tag. Both follow your push and merge.
