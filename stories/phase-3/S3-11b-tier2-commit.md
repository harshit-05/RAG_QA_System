# S3-11b: Commit the tier-2 re-run: v1.0's configuration and floors

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | DEC-26 (the one re-run); SRS §9 LLM01 (chunks delimited as data); NFR-2 (its numbers, DEC-27); backlog: the tier-2 follow-ups (fixed), `min_score` (set) |
| **Depends on** | S3-11 (its choices) |
| **Model** | opus (S3-11 made the decisions; this story commits them and runs the reference generation) |
| **Plan-first** | no |
| **Branch** | `feat/s3-11b-tier2-commit`. Risky in cost: about 70 min of generation here and a T4 scoring, and it sets v1.0's floors. |

## Goal

The choices S3-11 measured become v1.0's configuration, and tier 2 is re-run once for all of
them. The run is generated here, on the reference host, and scored on the T4 (DEC-26). The
committed run, with new floors, describes the system that ships as v1.0, and gives NFR-2's
revised numbers.

The maintainer split it from S3-11 in design review 1, as S2-5b was from S2-5. The config
and the run land together, so `main` is never stale. Design: ARCHITECTURE.md §3.1, DEC-26,
DEC-27.

## Scope

**Before any run, the estimate** (CLAUDE.md): about 70 min of generation here, then 35–50 min
of scoring on the T4, by the judge S3-11 chose.

- **The choices, from S3-11's records:**
  - **the judge:** `evaluation.judge` points at the chosen Modelfile, and
    ARCHITECTURE.md §2.1's DEC-15 gets a dated note if it changed;
  - **the prompt:** the revised refusal instruction, and LLM01's delimiting in `format_docs`.
    A unit test shows that a chunk containing the delimiter cannot close it, and a system
    sentence says the documents are data, not instructions;
  - **retrieval:**
    - with hybrid on, `pipeline.ingestion.sparse_encoder` and `pipeline.query.retriever` are
      switched together, then `rag-ingest` runs (it rebuilds: the vector configuration
      changes);
    - tier 1's floors are re-baselined in `eval/thresholds.yaml` (the baseline minus 0.05),
      on CI's numbers;
    - `min_score` goes on the reranker entry;
  - **the chunks** (caveat 37, added 2026-10-22): pypdf to 6.21.2 or later, with
    `CHUNKING_VERSION` 2, so the run's index is rebuilt from them, and S3-5's soft-hyphen
    question decided in the same change (STATUS.md backlog). Then remove the date check and
    the eleven pypdf `--ignore-vuln` (five from S3-5, three each from S3-6 and S3-7) from both Audit steps
    in `ci.yml`, the main one and the
    `eval` extra's that S3-9 adds: the exception ends here (DEC-24);
  - **the model:** the chosen one as `pipeline.query.llm`. The others stay as entries (keep
    the alternatives as config axes).
- **The run.**
  - **Generate here** (caveats 8 and 17: `free -h`, `ollama ps`, generator and judge never
    together).
  - **Score on the T4,** with Ollama pinned to 0.22.1.
  - **Set the floors** to the baseline minus 0.05, with `max_unscored` kept.
- **One check outside the labels** (DEC-26, design review 2). The judge was chosen on the
  20 answers it is measured against. So the maintainer labels the committed answers that it
  scored below 1, or that the two judges disagree on (about 5), and their agreement is
  recorded beside S3-11's stage 1. No re-run follows from it.
- **NFR-2's numbers** (DEC-27):
  - (b): the p95 of `ttft_ms` here, from this run, and on the T4, from S3-11's runs;
  - (a): the `sources` p95 again, with S3-7's script, on what ships: the bumped chunks and
    S3-11's retrieval and rerank choices. S3-7 measured 2.09 s, then 2.02 s, a miss; if it
    still misses, S3-12's amend session sets the target from this number.
- **Docs:** README's evaluation numbers and known limitations.

## Out of scope

- Measuring or choosing anything: S3-11's tables decide.
- Ollama past 0.22.1, unless the chosen model needs it.

## Verification

```bash
# the committed run: fresh and above its floors, in CI's way
uv run rag-eval check; echo "exit $?"                     # 0, against the new run and floors
uv run rag-eval check --with-ollama --with-index; echo "exit $?"   # 0: digests and contexts match here
uv run rag-eval retrieval                                 # the new tier-1 baseline
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy
gh pr checks                                              # check, eval-retrieval, container green
```

## Review notes for the human

- **The config diff matches S3-11's choices,** lever by lever.
- **LLM01's delimiter:** read `format_docs` and its test for a chunk that could close the
  delimiter.
- **The floors:** each is the baseline minus 0.05, and `decline_rate`'s is strict on purpose
  (S2-5b).

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
