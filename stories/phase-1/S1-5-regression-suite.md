# S1-5: The Phase-0 regression suite and the coverage gate

| | |
| --- | --- |
| **Status** | Todo |
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

# 3. the whole suite, hermetic: unplug the network and it must still pass
uv run pytest -q                       # with Ollama stopped: systemctl --user stop ollama
uv run pytest --cov=rag_qa --cov-report=term-missing --cov-fail-under=80

# 4. CI green on the pushed branch
gh run list --limit 1
```

## Review notes for the human

Read the regression test names against the ISS list in `docs/SRS.md` Appendix A
— the point of this file is that a future reader can see each historical bug is
pinned. Then check the suite genuinely runs with Ollama stopped and the network
down; a test that quietly downloads MiniLM will pass locally and hang in CI.
Last, look at what the coverage gate excludes: only `evaluate.py` should be
omitted.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
