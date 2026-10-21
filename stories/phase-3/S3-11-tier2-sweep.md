# S3-11: The tier-2 sweep: measure every lever on a T4

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | DEC-2 (the revisit: the choice); DEC-26 (the sweep, stages 1–4); backlog: the lenient judge (measured), the tier-2 follow-ups (measured), `rag-eval check` recomputing the aggregates |
| **Depends on** | S3-5 (the tier-1 tables), S3-10; the maintainer's hand labels |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-11-tier2-sweep`. Risky in cost rather than code: hours of GPU time, and the decisions that set v1.0's quality floors. |

## Goal

Every change in Phase 3 that moves a hashed input is measured here, and chosen (DEC-26). The
levers:

- the judge;
- the prompt (the refusal paraphrases, `glider-name`'s refusal, and LLM01's delimiting);
- hybrid retrieval with its sparse encoder;
- `min_score`;
- the model (the DEC-2 revisit).

S3-11b commits the choices with the one re-run, and the committed run describes the system
that ships as v1.0.

**Split from the commit** (maintainer, design review 1), as S2-5 and S2-5b were. The sweep
spans several sessions of T4 time, and this story's pull request moves no hashed input, so
`main` stays fresh while it runs. Its runs are made from its branch. A prompt or
`format_docs` revision under test lives there, and lands only in S3-11b. Design:
ARCHITECTURE.md §3.1, DEC-26, DEC-27.

## Scope

**Before any run, the estimates** (CLAUDE.md: surface them first). About 5–9 h of T4 time
across 3–4 sessions, this story's and S3-11b's together (DEC-26, design review 2; the first
draft said 4–6 h). That is 5–7 scorings at 35–50 min, generations of about 15 min each, and
15–25 min of setup per session. Generation has not been timed on a T4, so stage 0's run
revises the estimate. S3-11b adds about 70 min of generation here.

**The rules are committed before stage 0 runs** (DEC-26, design review 2). Each stage's rule
below is a margin with a default, fixed in this file in a commit before the first run, never
read off the results. The golden set can show only large differences:

- 4 of the 20 faithfulness items are below 1 (0.0, 0.67, 0.75 and 0.75);
- context recall is 1.0 on all 20;
- a decline rate of 3 in 5 has a 95% interval of 0.23 to 0.88;
- a sign test needs 6 changed items, all one way, for p = 0.03.

**The stages, in order:**

- **Stage 0, the hardware** (no scoring; design review 2). Generate v0.3's exact
  configuration on the T4, and diff its 25 answers with the committed ones (about 15 min).
  It measures how far a T4 result carries to this CPU, and the record says so beside every
  choice below.
- **Stage 1, the judge** (no generation). The maintainer's `eval/hand_labels.jsonl` gives
  one line per answerable `id`: `faithful` (`yes`, `partly` or `no`), `unsupported` and
  `notes`. Commit it; tier 2 never reads it, so nothing hashed moves.
  - **Measure:** compute gemma2:9b's agreement from the committed v0.3 scores: mean absolute
    error against yes = 1, partly = 0.5, no = 0, and every answer labelled `no` that it
    scored faithful.
  - **If it fails** (any false "faithful", or an error over 0.15): score the v0.3 answers on
    the T4 with a stronger judge, `gemma3:12b` at 4 bits, through a candidate Modelfile,
    `eval/judge.<name>.Modelfile`. It is hashed only once the config names it, which S3-11b
    does if it wins.
  - **It will fail** (design review 2): S2-5b found two wrong answers it scored as faithful.
    So this scoring is certain, about 45–50 min.
  - **gemma3:12b fits a T4.** Its weights are 8.2 GB, and at 8k context its KV cache is
    about 0.8 GiB with its sliding window, or 3 GiB without, so it takes about 9.5–12 GB of
    the T4's 15. Run Ollama with `OLLAMA_NUM_PARALLEL=1`, and check the size with `ollama ps`
    before scoring.
  - **"Better," fixed in advance:** fewer `no`-labelled answers scored faithful (0.5 or
    more), then the lower mean absolute error. A tie keeps gemma2:9b. If both pass a wrong
    answer, keep the better one and record its leniency.
  - **Record both tables here.**
- **Stage 2, retrieval, fixed from tier 1** (S3-5's tables; no tier-2 sweep):
  - hybrid on or off;
  - `rrf_k` and the prefetch limit;
  - `min_score`.

  The choice is recorded here. The candidate runs use it through a scratch config, with a
  hybrid ingest on the T4 when hybrid is on; S3-11b sets it in `config.yaml`.
  - **On the chunks S3-11b ships** (caveat 37, added 2026-10-22). S3-5 measured with pypdf
    6.19.0, but S3-11b bumps it with `CHUNKING_VERSION` 2, which changed 7 of the 25 golden
    contexts. So S3-11 first repeats S3-5's dense-against-hybrid and `min_score` tables
    with the bump, on its branch (about 15 minutes of tier 1, with S3-5's
    `~/dev/rag-s3-5/run.sh`), and every candidate run uses those chunks.
- **Stage 3, the prompt**, with mistral, generated and scored on a Colab T4: v0.3's prompt
  against the revised one.
  - **The revision:**
    - a refusal instruction that keeps the decline marker's phrase, for the two refusals
      that paraphrased past it;
    - a look at `glider-name`'s refusal, whose title chunk was in context;
    - LLM01: each chunk delimited as data in `format_docs`, with a closing delimiter that a
      chunk cannot forge, and a system sentence that the documents are data, not
      instructions.
  - **The rule** (design review 2): keep the revision if all three hold:
    - the decline rate improves;
    - no answerable question newly declines;
    - no metric falls by more than 0.05 against the v0.3 prompt's own T4 run, scored by the
      same judge. The floors are not the bar here: they were set on CPU answers, scored by
      gemma2:9b.
- **Stage 4, the model** (DEC-2), with the chosen prompt and retrieval, on the T4.
  - **Candidates:** mistral, phi3, `qwen2:7b` if the maintainer pulled it, and a current
    model if they named one. Its Ollama support on 0.22.1 is checked first; a newer Ollama
    would ride the re-run (backlog line).
  - **The choice** (design review 2): mistral stays unless a candidate leads by at least
    0.10 in faithfulness (2 of 20 items) or 0.2 in the decline rate (1 of 5), with no metric
    behind by more than 0.05. A tie goes to the time to first token on this CPU, from one
    timed question here.
  - The chosen model becomes `pipeline.query.llm` in S3-11b; the others stay as entries
    (keep the alternatives as config axes).
- **Colab generation** (the notebook from S2-5b, extended; DEC-26, design review 2):
  - Qdrant's static binary, started in the background;
  - the repository at this branch's commit, synced with a plain `uv sync`: CPU torch, since
    the T4 runs Ollama only;
  - this host's index, restored from a snapshot made here (`rag-backup create` here,
    `rag-backup restore` there), not ingested again with `minilm_cuda`. With hybrid on, the
    hybrid index is built here too;
  - queries embedded with `minilm_cpu`, and each question's context digest compared with
    this host's contexts file before anything is generated;
  - `rag-eval generate --out`, with Ollama 0.22.1 on the T4.

  Every candidate is generated there, so they share hardware. No CUDA torch is installed:
  S1-7's cu130 wheels need a recent NVIDIA driver, which no Colab run has shown yet (the
  backlog's CUDA 12 line), and GPU embeddings could move a near-tie in the contexts.
- **Code that the runs need, merged here.** It moves no hashed input.
  - **`rag-eval check` recomputes the aggregates** from the items, and refuses a file whose
    aggregates disagree (backlog; S2-5b's second review). A test edits an item and expects
    the refusal. The committed v0.3 run passes: S2-5b recomputed it exactly.
  - **The answers file records each question's context digest** (S3-2's `--contexts-out`),
    and the Qdrant server's version beside the generator's digest, recorded and not hashed:
    IDF and RRF run in the server (design review 1). `rag-eval check --with-index` compares
    the digests locally, as `--with-ollama` compares model digests. v0.3's answers file,
    which has neither, still loads.
- **Records:**
  - every stage's table goes here, with what it cost;
  - the choices for S3-11b: the judge, the prompt revision, the retrieval settings, and the
    model;
  - the T4's time to first token (DEC-27 b);
  - stage 0's count of answers that differ, and the context digests' comparison.

## Out of scope

- **Committing the choices** (S3-11b): the config, the prompt and `format_docs`, the judge's
  Modelfile, the run and the floors.
- **The `eval` extra's `langchain` bump:** S3-9 makes it (design review 1).
- **A hosted judge:** DEC-15 rejected it.
- **Weighted RRF or stemming,** unless S3-5's table made the case, recorded there.
- **Ollama past 0.22.1,** unless a chosen candidate needs it.

## Verification

```bash
# nothing hashed moved: the choices stay records until S3-11b
uv run rag-eval check; echo "exit $?"                     # 0, against v0.3's run
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy
# a hand-edited item in a scratch copy of generation-latest.json: check refuses it
gh pr checks                                              # check and eval-retrieval green
```

The stage tables and the costs are pasted below, run by run.

## Review notes for the human

- **The decisions are the deliverable:** the judge's agreement table, the prompt comparison
  and the model table. Each needs its numbers here, not a summary.
- **Nothing hashed lands on `main` here:** the revisions under test stay on the branch, for
  S3-11b.
- **The rules came first** (design review 2): the commit that fixed them precedes the first
  run's results.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
