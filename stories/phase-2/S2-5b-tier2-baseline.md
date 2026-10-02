# S2-5b: Take the tier-2 baseline and turn the gate on

| | |
| --- | --- |
| **Status** | Todo |
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
