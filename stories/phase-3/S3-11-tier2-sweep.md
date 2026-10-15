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

**Before any run, the estimates** (CLAUDE.md: surface them first). About 4–6 h of T4 time
across sessions, this story's and S3-11b's together; S3-11b adds about 70 min of generation
here.

- **Stage 1, the judge** (no generation). The maintainer's `eval/hand_labels.jsonl` gives
  one line per answerable `id`: `faithful` (`yes`, `partly` or `no`), `unsupported` and
  `notes`. Commit it; tier 2 never reads it, so nothing hashed moves.
  - **Measure:** compute gemma2:9b's agreement from the committed v0.3 scores: mean absolute
    error against yes = 1, partly = 0.5, no = 0, and every answer labelled `no` that it
    scored faithful.
  - **If it fails** (any false "faithful", or an error over 0.15): score the v0.3 answers on
    the T4 with a stronger judge, `gemma3:12b` at 4 bits (about 45 min, estimated), through
    a candidate Modelfile, `eval/judge.<name>.Modelfile`. It is hashed only once the config
    names it, which S3-11b does if it wins. Keep whichever judge agrees better.
  - **Record both tables here.**
- **Stage 2, retrieval, fixed from tier 1** (S3-5's tables; no tier-2 sweep):
  - hybrid on or off;
  - `rrf_k` and the prefetch limit;
  - `min_score`.

  The choice is recorded here. The candidate runs use it through a scratch config, with a
  hybrid ingest on the T4 when hybrid is on; S3-11b sets it in `config.yaml`.
- **Stage 3, the prompt**, with mistral, generated and scored on a Colab T4: v0.3's prompt
  against the revised one.
  - **The revision:**
    - a refusal instruction that keeps the decline marker's phrase, for the two refusals
      that paraphrased past it;
    - a look at `glider-name`'s refusal, whose title chunk was in context;
    - LLM01: each chunk delimited as data in `format_docs`, with a closing delimiter that a
      chunk cannot forge, and a system sentence that the documents are data, not
      instructions.
  - **The rule:** keep the revision if the decline rate improves and no floor falls.
- **Stage 4, the model** (DEC-2), with the chosen prompt and retrieval, on the T4.
  - **Candidates:** mistral, phi3, `qwen2:7b` if the maintainer pulled it, and a current
    model if they named one. Its Ollama support on 0.22.1 is checked first; a newer Ollama
    would ride the re-run (backlog line).
  - **The choice:** faithfulness and the decline rate first. Then each finalist's time to
    first token on this CPU, from one timed question here.
  - The chosen model becomes `pipeline.query.llm` in S3-11b; the others stay as entries
    (keep the alternatives as config axes).
- **Colab generation** (the notebook from S2-5b, extended):
  - Qdrant's static binary, started in the background;
  - the repository at this branch's commit;
  - `rag-ingest` with `minilm_cuda` (the same identity);
  - `rag-eval generate --out`, with Ollama 0.22.1 on the T4.

  Every candidate is generated there, so they share hardware.
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
  - the T4's time to first token (DEC-27 b).

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

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
