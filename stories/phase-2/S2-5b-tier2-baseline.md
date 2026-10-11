# S2-5b: Take the tier-2 baseline and turn the gate on

| | |
| --- | --- |
| **Status** | In review (2026-10-11): PR #14, every verification shown below; CI green on `da4b96b`. First review: no blockers, 4 doc findings, fixed on the PR. Next: CI on the review commit, then merge |
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
- The other two needed no change:
  - one asked for a branch, which the story's first step creates;
  - one flagged the score checkpoint's `started_at`, which a resume never reads. It stays,
    so that both kinds of checkpoint share one header.
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

### The digest question (2026-10-11)

`ollama create rag-judge -f eval/judge.Modelfile` again, 10 hours after the first, gave
the same ID, `d85f8193c353`. So `create` is deterministic on one host, with no timestamp
in the manifest.

**Across machines, too (2026-10-11).** On Colab, the setup matched this host:

- Ollama 0.22.1 from `install.sh` with `OLLAMA_VERSION=0.22.1`;
- `ollama pull gemma2:9b` gave `ff02c3702f32`, the same as here;
- `ollama create` from the same Modelfile, at `c268f54`, gave **`d85f8193c353`**, the same
  as here.

So `check --with-ollama` can check a Colab-scored run from this host, and the README needs
no caveat. It holds for the same Ollama version: another version may write the manifest
differently, which is why the recipe pins it.

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

### Scoring on a Colab T4 (2026-10-11): the recipe as run

The maintainer chose Colab. It runs from a notebook, and the checkpoint and outputs sit on
Google Drive, so a disconnect resumes the run. The cells:

1. `nvidia-smi` showed a Tesla T4 with 15360 MiB. Drive was mounted, with
   `OUT=/content/drive/MyDrive/rag-s2-5b`.
2. **Ollama, pinned:**
   - `apt-get install zstd`, then
     `curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.22.1 sh`;
   - `ollama serve` was started through `subprocess.Popen`, since there is no systemd;
   - the installer warns "Unable to detect NVIDIA/AMD GPU" (Colab has no `lspci`).
     That is harmless: `ollama.log` showed `inference compute … library=CUDA`, the T4.
3. **The judge's base:** `ollama pull gemma2:9b` gave `ff02c3702f32`, the same as this
   host.
4. **The repo:**
   - `git clone -b chore/s2-5b-tier2-baseline …`, at `c268f54`, the commit with the
     answers file;
   - uv 0.11.8's installer, which put uv in `/usr/local/bin`;
   - `uv sync --extra eval`. That gives CPU torch, which scoring never uses; the CUDA
     group was never synced.
5. **The judge:** `ollama create rag-judge -f eval/judge.Modelfile` gave `d85f8193c353`, the
   same as this host.
6. **Smoke run:** `uv run --no-sync rag-eval score --limit 2 --out $OUT/smoke.json`.
   - It took 6.6 min, including the MiniLM download and the judge's first load.
   - `ollama ps` showed rag-judge at 8.8 GB, 100% GPU, context 8192.
   - The writes to Drive worked, the checkpoint and the atomic replace alike.
   - Its two items matched the full scorings exactly.
7. **The two scorings:** `… score --out $OUT/scoring-1.json`, then the same into
   `scoring-2.json`, each `tee`'d to a log on Drive.
   - About **35 min each**: scoring 2 finished 34.8 min after scoring 1. That is about
     1.7 min per answerable item, against 39–54 min measured on this CPU at 2 threads.
   - No disconnect happened, so the resume path never ran.

**An HF token is not needed.** The one Hugging Face download is MiniLM (about 90 MB,
public). The warning about unauthenticated requests is harmless.

### The two scorings (2026-10-11)

**Identical.** Apart from `scored_at`, `scoring-1.json` and `scoring-2.json` match exactly:
every item's four scores, the aggregates, and 0 parse, 0 truncated, 0 unscored and 0
retried in both.

- The judge's largest prompt was 3,620 tokens and its largest answer 513, of 8,192.
- Both are fresh against this checkout, and both cover all 25 questions.

| metric | scoring 1 (committed) | scoring 2 | spread | floor |
| --- | --- | --- | --- | --- |
| faithfulness | 0.9083 | 0.9083 | 0 | 0.858 |
| answer_relevancy | 0.7186 | 0.7186 | 0 | 0.668 |
| context_precision | 0.9203 | 0.9203 | 0 | 0.870 |
| context_recall | 1.0 | 1.0 | 0 | 0.950 |
| decline_rate | 0.6 (3 of 5) | 0.6 | 0 | 0.550 |
| `max_unscored` | 0 of 20 unscored | 0 | | 0.1 |

- **The floors follow DEC-15's rule.** Each is the baseline minus the larger of 0.05 and
  twice the spread. With no spread, that is the baseline minus 0.05, rounded down to 3
  decimals.
- **No metric is left ungated**: none differed by more than 0.1.
- **`max_unscored` is 0.1**, the larger of 0.1 and the worst unscored share either
  scoring left (0).
- **What zero spread means.** DEC-15 expected a 9B judge to be non-deterministic even at
  temperature 0. On one T4 with Ollama 0.22.1, the Modelfile's `seed` and greedy decoding
  made it exactly repeatable. So the two scorings measure repeatability on that one setup.
  They do not measure a re-score on other hardware (this CPU, another GPU), which remains
  unmeasured. The 0.05 minimum is therefore the whole tolerance: on the four RAGAs
  metrics, one item of 20 moving by 1.0 is exactly 0.05.
- **`decline_rate`'s floor allows no change** (found in review). Its 5 items are 0.2
  each, so one more refusal that misses the marker gives 0.4, below 0.55. It stays
  strict: a paraphrased refusal is a prompt-following miss, and generation here is
  deterministic.

### Hand check (caveat 18, 2026-10-11)

Five items were read against their contexts and references: the best, the worst, tier 1's
one miss, and both answer-relevancy zeros. F, AR, CP and CR below are faithfulness, answer
relevancy, context precision and context recall.

| item | why | F / AR / CP / CR | verdict |
| --- | --- | --- | --- |
| `yolo-objectives` | best | 1 / 1 / 1 / 1 | **Agree.** The three objectives are in context 2, and the answer states them. |
| `glider-name` | worst | 0 / 0 / 1 / 1 | **Agree.** The title chunk is in context, so the refusal is unfaithful and does not answer. Retrieval was perfect. |
| `glider-slm` | tier 1's miss | **1** / 0.75 / 1 / **1** | **The judge is wrong twice.** The answer, "SLM stands for Summarization as Judge Model", is made up: no context expands SLM. F should be about 0. "Small Language Model" is in no context (page 2 was not retrieved), so CR should be about 0.5. |
| `lusk-overbeek-itp` | AR 0 | 0.75 / **0** / 0.95 / **1** | **CR is wrong; AR is the metric's design.** The reference's "written in Pascal … approximately fifty sites" is on p. 43, but in a chunk not retrieved, so CR should be about 0.5–0.6. AR 0 is ragas's rule for a "noncommittal" answer: one hedge zeroes an answer that is two-thirds right. |
| `glider-data-filtering` | AR 0 | 0.67 / 0 / 0.75 / 1 | **Agree.** The answer says the number is not in the context, then gives it from context 4. The judge marked that claim unfaithful. |

**What it means:**

- **The judge is lenient, not noisy.** It never misparsed, it repeated exactly, and it was
  right wherever the evidence was plain.
- **It missed one fabrication** (faithfulness), and twice credited a reference the contexts
  only half held (context recall). The review found a third such credit, `glider-purpose`
  (Review, below).
- **So the baseline is somewhat high.** Context recall's 1.000 on all 20 items is partly a
  ceiling.
- **As a regression gate it still holds.** The same judge scores a later run the same way.
  Context recall is the weakest of the four as a signal, but tier 1 gates retrieval
  directly.
- **Answer relevancy is all-or-nothing on hedging.** One hedged answer moves the mean by
  about 0.04.
- **Tier 1 is page-level.** It calls `lusk-overbeek-itp` a hit, although the chunk with
  half the reference was not retrieved.

### The judge, assessed (2026-10-11)

**As machinery, it beat expectations:**

| Expected (DEC-15, caveat 18) | Measured |
| --- | --- |
| Two scorings differ: "a 9B judge is not deterministic even at temperature 0" | Identical to the last digit, on one T4 with Ollama 0.22.1 |
| Some misparses of RAGAs' structured prompts | 0 parse failures and 0 retries, over 360 judge calls. Each answerable item takes 9 calls in ragas 0.4.3: 2 for faithfulness (statements, then verdicts), 1 for answer relevancy at `strictness` 1, 1 per context for context precision (5), and 1 for context recall. That is 9 × 20 items × 2 scorings. Ollama's grammar-constrained JSON mode removes format errors, so the risk moves from format to content. |
| Prompts near the context limit | Largest prompt 3,620 tokens, largest answer 513, of 8,192 |
| About 1 h per scoring on a T4 | About 35 min |

**As an evaluator, it is lenient.**

- **What it gets right:** the blatant cases, namely the false refusal, the
  self-contradiction and the clean answer.
- **What it misses:**
  - a fabrication that shares words with the context: "Summarization **as Judge Model**"
    against the context's "SLM-**as-judge model**". Word overlap is the likely cause, a
    known weakness of small judges;
  - context recall: it saturates, at 1.0 on all 20 items, including three found only
    partly supported by hand: two in the hand check, and `glider-purpose` in the review.
- **On the hand-checked verdicts** (5 items × 4 metrics), 3 of 20 were clearly wrong,
  about 15%, all too generous: `glider-slm`'s faithfulness and context recall, and
  `lusk-overbeek-itp`'s context recall.
  - The sample was chosen for anomalies, so the rate on a random item is likely lower.
  - Five items cannot pin it down.

**What it means for reading the gate:**

- **The bias is systematic, not random.** Every error leaned the same way, too generous. A
  floor compares this judge with itself, so the bias costs less than noise would.
- **The real cost is sensitivity.** A change that brought in `glider-slm`-style
  fabrications might not move faithfulness at all. A lenient judge is a quieter alarm.
- **How to read each floor:**
  - faithfulness and context precision: coarse alarms that catch large regressions;
  - context recall: close to signal-free near the top; tier 1 gates retrieval directly;
  - answer relevancy: volatile on hedging.
- **The absolute numbers are not quality claims.** "Faithfulness 0.91" overstates it; they
  are relative baselines for this judge.
- **`context_recall` stays gated** (Review, below). A `null` floor would cost nothing,
  since floors are not hashed, but DEC-15's rule removes a metric only for noise, and this
  judge showed none. The stronger reason is that its errors all leaned generous, so a fall
  it reports is likely real.

**Follow-ups** (STATUS.md backlog, "After S2-5b"):

- **Measure the judge first.** Hand-label faithfulness for the 20 answers, about one to two
  hours of reading, and compute the judge's agreement. That turns "lenient" into a number.
- **Then spot-check with a stronger judge.** A 14B model at 4-bit should fit on a T4 (not
  measured: check its size and its context's memory first). The DEC-2 revisit is the
  natural place. A new judge moves `judge` and means a re-score.

## Verification record (2026-10-11)

1. **The gates:** `ruff check` and `mypy` were clean, and `pytest --cov=rag_qa` gave **677
   passed, 4 xfailed**, coverage 97.91%.
   - The same suite under `HF_HOME=<empty dir> HF_HUB_OFFLINE=1` gave the same result, and
     the folder stayed empty (caveat 23).
2. **`uv run rag-eval check`: exit 0.** All six parts were ok, and every floor was met.
   - It also passed on a `git archive` export of the tree with the new files laid over it,
     at another path. So it does not depend on this checkout's path or its untracked
     files.
   - The corpus scan reads exactly the three tracked PDFs; the ignored `.parquet` files are
     not loaded.
3. **Stale detection:**
   - A one-word prompt edit in a scratch copy of the config gave **exit 1**: "query changed:
     re-run generate and score". Only `query` was CHANGED.
   - A ground-truth-only edit (`glider-name`) in a scratch copy of the golden set gave
     **exit 1**: "references changed: re-run score". Only `references` was CHANGED.
   - 3b. **`check --with-ollama`: exit 0.** The Colab-scored run's judge (`d85f8193c353`),
     its base (`ff02c3702f32`) and the generator (`6577803aa9a0`) all match this host's
     live Ollama.
4. **The committed record:** judge `rag-judge` `d85f8193…` on `gemma2:9b` `ff02c370…`, with
   `num_ctx` 8192 and served 8192. The largest prompt was 3,620 tokens and the largest
   answer 513; the mode was `json`, and the embedder MiniLM. The aggregates are as in the
   table above.
5. **CI: green on PR #14.** Run 38135431966, on `da4b96b`, took 5 min 13 s.
   - `check` passed, including the new **Tier-2 check** step. On GitHub's runner all six
     fingerprint parts were `ok` and every floor was met. That is a fresh checkout on
     another machine reproducing the corpus hashes, the rendered-prompt probe and the
     judge hash.
   - `eval-retrieval` passed.
   - The two timing-based Ctrl-C tests did not flake. The earlier run on the PR
     (38135061271) was green too.

## Review (2026-10-11)

**First review (Opus 5.5, over the `/code-review` pass): no blockers; 4 findings, all in
the docs, fixed on the PR.**

- **Checked and sound:**
  - the two run files. The aggregates recompute from the items, which are in the answers
    file's order. The scores' `answers` sha matches `answers-latest.json`. The timings match
    the file: 69.1 min, a median of 159 s, at most 274 s, and a median ttft of 148 s;
  - the floors: each is the baseline minus 0.05, rounded down;
  - verifications 2, 3 and 3b, re-run here:
    - exit 0;
    - exit 1, with only `query` CHANGED;
    - exit 1, with only `references` CHANGED;
    - exit 0, with all three digests live on Ollama 0.22.1;
  - CI's Tier-2 check step: the runner ran the whole check, and all six parts were `ok`;
  - the hand check's wrong verdicts on `glider-slm` and `lusk-overbeek-itp`, against their
    contexts;
  - the resume code and the `/code-review`'s 7 fixes: the torn-line cut, the identity and
    kind checks, the restore from JSON (covered end to end), the failure counts taken from
    the items, and the totals carried across sessions. Nothing new was found;
  - the package version is not hashed, so S2-9's bump to 0.3.0 leaves the run fresh.
- **Not checkable from the repo:** scoring 2 is on the maintainer's Drive. That it matched
  scoring 1 is the maintainer's attestation. The smoke run, which matched both, crossed a
  judge load.
- **Not re-run:** the test suite. CI was green on `da4b96b`, the HEAD under review.

**Findings:**

- **A third context-recall overcredit: `glider-purpose`.** The review read the five items
  whose retrieval missed an expected page.
  - Three deserve their 1.0: `glider-training-data`, `cade7-venue` and `yolo-objectives`.
  - `glider-purpose` does not. The reference's third sentence, LiveBench's 0.654 against
    0.481 and 0.485, is only on p. 7, which was not retrieved. Context 2 (p. 5) describes
    LiveBench but gives no scores, so its context recall should be about 0.67.
  - With the three corrected, this baseline's context recall is about 0.94, below its own
    floor. So the number is relative to this judge, as the assessment says.
  - Corrected here, in STATUS.md, ARCHITECTURE.md and `thresholds.yaml`.
- **`decline_rate`'s floor allows no change.** Its 5 items are 0.2 each, so the 0.05
  minimum tolerates none. The comments said "one item in 20". **Kept at 0.55**, on the
  review's recommendation: it follows the marker decision, and generation here is
  deterministic. The comments now say so.
- **The `/code-review`'s two other findings were not named.** They are now, under
  Discovered.
- **The model deviation cited CLAUDE.md's Fable note,** which does not route S2-5b.
  Reworded.

**`context_recall` stays gated, on the review's recommendation.**

- This judge's errors all leaned generous, so a fall it reports is likely real: the alarm
  is quiet, but not false.
- A `null` floor would drop that alarm and still print the 1.000.
- A re-score on other hardware risks a false alarm on all four metrics alike.
- It is not read as retrieval coverage: tier 1 gates retrieval.

## Deviation from plan

- **Model:** Opus 5.5 without `/fast`, where the story said opus-fast. CLAUDE.md's Fable
  note does not cover S2-5b. The resumable-run code was moderately design-heavy, so it got
  a `/code-review` pass (9 findings: 7 fixed, 2 needed no change).
- **`--limit` is part of a checkpoint's identity.** The plan left it out, as items are
  independent. The review showed that a smoke run into the same `--out` would then
  resume from a full run's checkpoint, and delete it when done.
- **The ignore rule is a global `*.partial.jsonl`,** not `eval/runs/` only: `--out` can
  point anywhere.
- **The judge's Modelfile has no `num_thread`.** The baseline scored on a T4, where every
  layer is on the GPU. `judge` is unchanged.
- **Step 5's "check three scored answers by hand" became five items.** Tier 1's miss and the
  two answer-relevancy zeros needed a look.
- **The two scorings measured no noise at all** (above). The floors are the rule's 0.05
  minimum, and the cross-hardware spread is recorded as unmeasured.
- **The story's step 5 predates `max_unscored`.** Its value (0.1) follows the rule the
  maintainer approved in the plan.
