# S2-5b: Take the tier-2 baseline and turn the gate on

| | |
| --- | --- |
| **Status** | In progress (2026-10-07) |
| **Closes** | FR-7 (tier 2 enforced) |
| **Depends on** | S2-5 (the harness and the gate) |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `chore/s2-5b-tier2-baseline`. Risky: it adds the CI step that can turn every PR red, and the numbers it commits become the floors. |

## Goal

The hours-scale half of tier 2, split from S2-5. It produces the committed baseline, sets
the floors, and makes `rag-eval check` a CI step. After it, the "enforced quality bar"
half of the Phase 2 exit holds. Design: ARCHITECTURE.md §2.1 DEC-15.

## Scope

- **Set `num_thread` first**, unless an earlier story already did (STATUS.md backlog,
  "S2-5b, first step"). It is part of the hashed llm spec, so it must precede the
  baseline.
- **Surface the estimate**, and let the maintainer choose local or Colab/Kaggle
  (CLAUDE.md GPU policy, caveats 16–18).
- **The baseline run.** Long runs go in the foreground, with a `tee`'d log and the
  `tail -f` command given first.
  1. Run `rag-eval generate` here with mistral. Check `ollama ps` and `free -h` first
     (caveat 8). The estimate is about 40 min if answers take 1–2 min. S2-1 measured
     254 s to first token on 2 threads with swapped weights (about 1.8 h for 25
     questions), so time one question at the chosen `num_thread` and estimate from that.
  2. Run `ollama stop mistral`.
  3. Run `rag-eval score`, **twice**, into two output files. Locally that is about 3–4 h
     each, which is why Colab/Kaggle is the default here. The second scoring is what
     measures the judge's noise (step 5).
     - **Colab/Kaggle recipe (record it here as run):**
       - clone at the same commit, since `score` refuses an answers file that does not
         match the checkout;
       - install uv, then `uv sync --extra eval`. The default groups bring CPU torch,
         which scoring never uses; never sync the CUDA group just for this;
       - install Ollama, `ollama pull gemma2:9b`, and check that its digest matches the
         one recorded here, with `ollama list`;
       - `ollama create rag-judge -f eval/judge.Modelfile`;
       - copy `eval/runs/answers-latest.json` in, and run `score` twice;
       - copy both score files back.
     - Locally, `rag-judge` holds about 8 GB with its 8k context (S2-5's spike measures
       the real figure). Run it only with mistral stopped and the desktop light.
  4. Check three scored answers by hand against the judge's verdicts.
  5. **Set the floors from the measured noise.** For each metric, the floor is the
     baseline (the first scoring) minus the larger of 0.05 and twice the difference
     between the two scorings. Record the baseline, both scorings and the floor here.
     If one metric's two scorings differ by more than 0.1, that metric is too noisy to
     gate: record it, and leave it out of `thresholds.yaml` rather than set a floor
     inside the noise.
  6. Commit `eval/runs/answers-latest.json` and the first scoring as
     `eval/runs/generation-latest.json`. The second scoring's aggregates go in this file
     only.
- **CI:** `uv run rag-eval check` becomes the last step of the `check` job.

## Inputs from S2-5's review (2026-10-07)

- **Confirm `check --with-ollama` across machines before relying on it.** A Colab
  baseline records the digest of Colab's `ollama create rag-judge`. It is not known
  whether the same Modelfile on the same `gemma2:9b` blobs gives the same digest here.
  Run `ollama create` on both, compare `ollama list`, and record the answer. If they
  differ, `--with-ollama` cannot check a Colab-scored run from this host. Say so in the
  README instead.
- **Make the runs resumable first** (STATUS.md backlog): one transport error on item 19
  would lose a 13–18 h scoring.
- **Land `num_thread` before the baseline**, as the Scope says. It stays in the `query`
  hash on purpose, while the other runtime-only llm keys do not (S2-5's review).

## Out of scope

- Changing the prompt, `min_score` or the model to improve a number: follow-ups, each
  measured by this harness.
- The DEC-2 revisit (backlog).

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. the gate on the committed baseline
uv run rag-eval check; echo "exit $?"            # → 0

# 3. stale detection, without touching the real config: a one-word prompt change
cp config.yaml <scratch>/stale.yaml
sed -i 's/helpful and precise/helpful, precise/' <scratch>/stale.yaml
RAG_DATA_PATH=$PWD/corpus uv run rag-eval check --config <scratch>/stale.yaml --eval-dir eval; echo "exit $?"
#    → 1, and the message says "query changed: re-run generate and score"
#    and a ground-truth-only edit in a scratch copy of the golden set
#    → 1, "references changed: re-run score" (no re-generate)

# 3b. locally, with Ollama up: the recorded digests still match
uv run rag-eval check --with-ollama; echo "exit $?"   # → 0

# 4. what was committed
uv run python -c "import json; r = json.load(open('eval/runs/generation-latest.json')); print(r['judge'], r['aggregate'])"

# 5. CI
gh run list --limit 2                            # check (including rag-eval check) and eval-retrieval green
```

## Review notes for the human

- **Before trusting any floor, read the three hand-checked answers.** A judge that
  misparses quietly produces confident numbers.
- **Compare the two scorings.** The floors are only as trustworthy as the gap between
  them, and a metric dropped as too noisy should be named here.
- **The two run files are the maintainer's attestation.** Their fingerprints must match
  the tree being merged.

## Discovered

### Resumable runs (2026-10-07)

- `generate` and `score` keep each finished item in `<out stem>.partial.jsonl` beside the
  file they write (`evaluation/checkpoint.py`), and the same command resumes from it.
- A checkpoint resumes only the run that wrote it. It must match on:
  - the generation parts, the generator and its digest (for `generate`);
  - the answers sha, `references`, `judge`, the judge's and base's digests and the
    versions (for `score`);
  - `--limit`.

  Anything else is refused with "delete it to start over".
- A `/code-review` pass found 9 issues, and 7 were fixed:
  - a torn last line is now cut from the file, not only skipped in memory: the next
    append ran on from it, so a second interruption found a bad line;
  - `--limit` joined the identity: a smoke run into the same `--out` had resumed from a
    full run's checkpoint and then deleted it;
  - an Ollama that cannot report the generator's digest yet says "run again", not
    "delete";
  - Ctrl-C and a failed final write now say where the finished items are;
  - the resume hint is worded so it holds even after a refusal;
  - `generate` walks the items once, in order.
- The score loop got its first end-to-end tests, with ragas faked.

### `num_thread` (2026-10-11)

Mistral, golden question 2, the model loaded, 8.7 GB available and no swap.
`journalctl` confirmed `NumThreads:2`, `6` and `10`.

| num_thread | time to first token | total | question 1 (with the reload) |
| --- | --- | --- | --- |
| 2 (Ollama's default here) | 250.9 s | 260.5 s | 298.3 s |
| 6 | 186.6 s | 193.7 s | 223.9 s |
| **10** | **167.6 s** | **175.4 s** | 190.4 s |

- Both answers were byte-identical at all three counts.
- 10, the physical core count, is set on all three Ollama llms.
- `query` moved, as it should.
- S2-1's 254 s figure at 2 threads (with swapped weights) is reproduced at 2 threads
  with no swap. So the thread count was the main cost, not swap.

### The digest question, this host's half (2026-10-11)

`ollama create rag-judge -f eval/judge.Modelfile` again, 10 hours after the first, gave
the same ID, `d85f8193c353`. So `create` is deterministic on one host, with no timestamp
in the manifest. Colab's half is in the scoring record below.

### Generation (2026-10-11)

- **The run:** 25 answers in **69 min**, at `num_thread` 10. That is a median of 159 s per
  answer and at most 274 s, with a median ttft of 148 s. The estimate was 75–80 min.
- **The answers file:**
  - its fingerprint matches the checkout;
  - mistral's digest is `6577803aa9a0`;
  - every prompt got 5 chunks;
  - no answer is empty.
- **Decline rate 3/5 = 0.6.** Two unanswerable answers are honest refusals that
  paraphrase past the marker "could not find the answer":
  - `unanswerable-yolo-dataset`: "I could not find information about which specific
    public dataset…";
  - `unanswerable-cade8-venue`: "I could not find information about the location of…".

  **The maintainer kept the marker (2026-10-11).** The prompt asks for that exact
  sentence, so a paraphrase is a prompt-following miss. Redefining the metric after
  seeing the data would move the goalposts. A measured follow-up (tighten the prompt, or
  revisit the rule) goes to the backlog.
- **A false refusal: `glider-name`** ("What does GLIDER stand for?").
  - Chunk 3 is the paper's title page: "GLIDER: Grading LLM Interactions and Decisions
    using Explainable Ranking".
  - Mistral still answered that the context does not mention it.
  - Retrieval worked, and generation missed. The baseline's RAGAs numbers carry it.
  - Fixing it is a prompt or model change, which is out of scope here.

## Deviation from plan

(Filled at close-out.)
