# S1-8: Phase 1 exit — audit gate, doc hygiene, release 0.2.0

| | |
| --- | --- |
| **Status** | Todo |
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
