# S3-11: The tier-2 sweep: one re-run for every lever

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | DEC-2 (the revisit); DEC-26; SRS §9 LLM01 (chunks delimited as data); NFR-2 (its numbers, DEC-27); backlog: the tier-2 follow-ups (refusal paraphrases, generation misses, the lenient judge, answer relevancy's hedging), `min_score`, the `eval` extra's `langchain` advisory (the bump), `rag-eval check` recomputing the aggregates |
| **Depends on** | S3-5 (the tier-1 tables), S3-10; the maintainer's hand labels |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-11-tier2-sweep`. Risky in cost rather than code: hours of GPU and CPU time, and decisions that set v1.0's quality floors. |

## Goal

Every change in Phase 3 that moves a hashed input lands here, and tier 2 is re-run once for
all of them (DEC-26). The levers:

- the judge;
- the prompt (the refusal paraphrases, `glider-name`'s refusal, and LLM01's delimiting);
- hybrid retrieval with its sparse encoder;
- `min_score`;
- the model (the DEC-2 revisit);
- the `eval` extra's `langchain` bump.

The committed run, with new floors, describes the system that ships as v1.0. It also gives
NFR-2's revised numbers: TTFT on this CPU and on a T4. Design: ARCHITECTURE.md §3.1,
DEC-26, DEC-27.

**If the runs outgrow one session,** split it as S2-5/S2-5b was: the sweep (stages 1–4),
then the commit (stage 5).

## Scope

**Before any run, the estimates** (CLAUDE.md: surface them first). About 4–6 h of T4 time
across sessions, plus about 70 min of generation here.

- **Stage 1, the judge** (no generation). The maintainer's `eval/hand_labels.jsonl` gives
  one line per answerable `id`: `faithful` (`yes`, `partly` or `no`), `unsupported` and
  `notes`. Commit it.
  - **Measure:** compute gemma2:9b's agreement from the committed v0.3 scores: mean absolute
    error against yes = 1, partly = 0.5, no = 0, and every answer labelled `no` that it
    scored faithful.
  - **If it fails** (any false "faithful", or an error over 0.15): score the v0.3 answers on
    the T4 with a stronger judge, `gemma3:12b` at 4 bits (about 45 min, estimated), through
    a new `eval/judge.Modelfile`. Keep whichever judge agrees better.
  - **Record both tables here.** A judge change moves `judge` (DEC-15), which the final
    scoring pays for anyway.
- **Stage 2, retrieval, fixed from tier 1** (S3-5's tables; no tier-2 sweep):
  - hybrid on or off;
  - `rrf_k` and the prefetch limit;
  - `min_score`.

  If hybrid is on, set `pipeline.ingestion.sparse_encoder` and switch
  `pipeline.query.retriever` together, then `rag-ingest` (it rebuilds: the vector
  configuration changes). Re-baseline tier 1's floors in `eval/thresholds.yaml` (the
  baseline minus 0.05) on CI's numbers.
- **Stage 3, the prompt**, with mistral, generated and scored on a Colab T4: v0.3's prompt
  against the revised one.
  - **The revision:**
    - a refusal instruction that keeps the decline marker's phrase, for the two refusals
      that paraphrased past it;
    - a look at `glider-name`'s refusal, whose title chunk was in context;
    - LLM01: each chunk delimited as data in `format_docs`, with a closing delimiter that a
      chunk cannot forge, and a system sentence that the documents are data, not
      instructions.
  - **The rule:** keep the revision if the decline rate improves and no floor falls. A unit
    test shows that a chunk containing the delimiter cannot close it.
- **Stage 4, the model** (DEC-2), with the chosen prompt and retrieval, on the T4.
  - **Candidates:** mistral, phi3, `qwen2:7b` if the maintainer pulled it, and a current
    model if they named one. Its Ollama support on 0.22.1 is checked first; a newer Ollama
    would ride this re-run (backlog line).
  - **The choice:** faithfulness and the decline rate first. Then each finalist's time to
    first token on this CPU, from one timed question here.
  - **The config:** the chosen model becomes `pipeline.query.llm`. The others stay as entries
    (keep the alternatives as config axes).
- **Colab generation** (the notebook from S2-5b, extended): Qdrant's static binary started
  in the background, the repository at this branch's commit, `rag-ingest` with
  `minilm_cuda` (the same identity), and `rag-eval generate --out` with Ollama 0.22.1 on the
  T4. Every candidate is generated there, so they share hardware.
- **Stage 5, the commit.**
  - **Generate here** (CPU, the reference host, about 70 min for mistral; caveats 8 and 17:
    `free -h`, `ollama ps`, generator and judge never together). **Score on the T4** (about
    35 min), with Ollama pinned to 0.22.1.
  - **Set the floors** to the baseline minus 0.05, with `max_unscored` kept.
  - **NFR-2's numbers** (DEC-27): p95 TTFT here from this run's `ttft_ms`, and on the T4
    from stage 4's runs.
- **Code that rides the re-run.**
  - **`rag-eval check` recomputes the aggregates** from the items, and refuses a file whose
    aggregates disagree (backlog; S2-5b's second review). A test edits an item and expects
    the refusal.
  - **The answers file records each question's context digest** (S3-2's
    `--contexts-out`), so a later retrieval change can be checked here without a re-run.
    `rag-eval check --with-index` compares it locally, as `--with-ollama` compares digests.
  - **The `langchain` bump:** `uv lock --upgrade-package langchain`, to 1.4.3 (checked), and
    nothing else. Look at the lock diff. Then remove S3-9's ignore entry for
    PYSEC-2026-2192, so the `eval` audit passes on its own.
- **Records:** every stage's table goes here, with what it cost. ARCHITECTURE.md §2.1's
  DEC-15 gets a dated note for any judge change, and README's evaluation numbers and known
  limitations are updated.

## Out of scope

- A hosted judge (DEC-15 rejected it).
- Weighted RRF or stemming, unless S3-5's table made the case, recorded there.
- Ollama past 0.22.1, unless a chosen candidate needs it.

## Verification

```bash
# the committed run: fresh and above its floors, in CI's way
uv run rag-eval check; echo "exit $?"                     # 0, against the new run and floors
uv run rag-eval check --with-ollama --with-index; echo "exit $?"   # 0: digests and contexts match here
uv run rag-eval retrieval                                 # the new tier-1 baseline
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy
git diff main -- uv.lock | grep '^[-+]name\|^[-+]version'   # only the langchain bump's packages
# a hand-edited item in a scratch copy of generation-latest.json: check refuses it
gh pr checks                                              # check, eval-retrieval, container green
```

The stage tables and the costs are pasted below, run by run.

## Review notes for the human

- **The decisions are the deliverable:** the judge's agreement table, the prompt comparison
  and the model table. Each needs its numbers here, not a summary.
- **LLM01's delimiter:** read `format_docs` and its test for a chunk that could close the
  delimiter.
- **The floors:** each is the baseline minus 0.05, and `decline_rate`'s is strict on purpose
  (S2-5b).

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
