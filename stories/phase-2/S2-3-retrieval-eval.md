# S2-3: Score retrieval against the golden set, in CI on every push

| | |
| --- | --- |
| **Status** | In review (2026-10-03). Verification 1–3 run and shown; 4 (CI) runs on the branch's push |
| **Closes** | FR-7 (tier 1 of the quality gate) |
| **Depends on** | S2-1 (`build_retrieve`), S2-2 (the golden set) |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `feat/s2-3-retrieval-eval`. Risky: it adds the first CI job with network access, and a gate that can turn PRs red. |

## Goal

Retrieval quality becomes numbers that CI recomputes on every push. For each answerable
golden question, three are taken over what the prompt would actually receive:

- **hit rate**: an expected page, or one of its `also_pages`, appears anywhere in the
  context;
- **MRR**: the reciprocal rank of the first such page;
- **recall**: the share of the expected `pages` the context covers (`also_pages` are not
  in the denominator).

`also_pages` came from S2-2's second review: three records name pages that repeat the
whole answer. Without them, retrieving the repeat scores as a miss (ARCHITECTURE.md §2.1,
DEC-15 tier 1).

This is the half of the quality bar that can be recomputed: deterministic, with no LLM,
and taking minutes. Design: ARCHITECTURE.md §2.1 DEC-15 (tier 1), CI in §2.5.

## Scope

- **`evaluation/retrieval.py`.**
  - For each answerable item, run the question through the retrieval half that S2-1's
    `build_retrieve` produces.
  - Match the returned chunks to `expected_sources`. A chunk's `source` is made relative
    to `paths.data`, which works both before and after S2-6, and its `page_label` is
    compared with the record's printed pages.
  - Compute hit, reciprocal rank and recall for each item, and their means.
  - It needs the embedder and the index only, with no LLM, so it runs without Ollama.
- **`evaluation/cli.py`, the `rag-eval` entry point.** It is registered in
  `[project.scripts]`, and its other subcommands arrive in S2-5.
  - Usage:
    `rag-eval retrieval [--config PATH] [--dataset PATH] [--thresholds PATH] [--json OUT]`.
  - It prints a table for each item (misses first) and the aggregates.
  - Exit codes:

    | Code | Meaning |
    | --- | --- |
    | 0 | every floor is met |
    | 1 | a floor is missed |
    | 2 | it cannot start: no index, or a bad dataset or config |
- **`eval/thresholds.yaml`, `retrieval:` section.**
  - Measure the baseline of the current config: dense search, k=5, no reranker.
  - Set each floor to the baseline minus a 0.05 tolerance. That is one question on about
    20, so the first CI run must reproduce the local baseline on the runner's CPU before
    the floors are trusted.
  - Record both numbers here.
- **The CI job `eval-retrieval`**, which runs after `check` (`needs: check`):
  1. checkout and setup-uv, with the same pins as `check`;
  2. `uv sync --locked`;
  3. `actions/cache`, pinned by SHA, on `~/.cache/huggingface`, keyed on the model names
     (the embedder now; S2-4 adds the cross-encoder), not on `config.yaml`, so a prompt
     edit does not drop 180 MB of weights;
  4. `rag-ingest` into `$RUNNER_TEMP/index` through `RAG_VECTOR_STORE_PATH`. This step
     may use the network, and fills the cache on a miss;
  5. `rag-eval retrieval` under `HF_HUB_OFFLINE=1`.

  A comment in `ci.yml` says why this job alone has network access (DEC-15, caveat 19).
  The `check` job keeps `HF_HUB_OFFLINE` exactly as it is.
- **Tests.**
  - Metric arithmetic on hand-built documents: a hit at rank 1, at rank 3 and no hit;
    partial recall; `page_label` matching; absolute versus relative `source`. An
    `also_pages` hit counts for hit rate and MRR but leaves recall unchanged; an
    expected page ranked below an also-page sets the rank by the also-page.
  - `rag-eval retrieval` exit codes, against a tiny index built with
    `DeterministicFakeEmbedding`.

## Out of scope

- The reranker: S2-4 adds it, and re-baselines with these metrics.
- Generation metrics (S2-5).
- Tuning k, chunk size or the embedder. Note what the numbers suggest under Discovered;
  each is a later, measured change.

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. the real baseline, on a scratch index. The real index is untouched (caveat 9).
#    The corpus is the real one, so only the index path is overridden.
RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-ingest                 # ~2 min, 1,708 chunks
RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-eval retrieval; echo "exit $?"
#    → the item table and the aggregates; exit 0 against the new floors

# 3. negative control: a floor above what was measured must fail
printf 'retrieval: {hit_rate: 1.01, mrr: 0, recall: 0}\n' > <scratch>/strict.yaml
RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-eval retrieval --thresholds <scratch>/strict.yaml; echo "exit $?"   # → 1

# 4. CI
gh run list --limit 2
#    → both jobs green on the branch. Re-run once: the eval-retrieval log should show a
#      cache miss on the first run and a hit on the second.
```

## Review notes for the human

- **The CI job is the first with network access.** Check three things:
  - the action SHAs are pinned;
  - the cache key is the model names, with a prefix restore key;
  - `HF_HUB_OFFLINE=1` is set on the eval step, and the `check` job is unchanged.
- **Hand-compute one item.** Take its retrieved pages from the table, and check the hit,
  the rank and the recall against its golden record.

## Discovered

- **The baseline (2026-10-03): hit rate 0.800 (16 of 20), MRR 0.5875, recall 0.7167.**
  The scratch index built for verification 2 and the real index built on 2026-10-02 give
  identical results, question by question. That shows reproducibility on this host only;
  the runner's CPU is verification 4's question.
- **The four misses are real retrieval misses, not partial repeats.** S2-2 says to read the
  notes before calling a miss a retrieval failure, and none of these retrieved a page that
  its notes name as a repeat:
  - Three are GLIDER's opening pages: `glider-name` (p. 1, also p. 2), `glider-slm`
    (p. 2) and `glider-training-data` (pp. 1–2). The retriever prefers the paper's later
    pages (7, 8, 9, 4, 5). `glider-name` even retrieves p. 8, the trap its notes name.
  - `glider-slm` is S0-6's own failure, now measured: "none of its five retrieved chunks
    defined SLM". The model's invented expansion was at least partly a retrieval failure.
  - `ohlbach-wrightson-mkrp` (p. 496) retrieves p. 500, which its notes do not name.
- **Input for S2-4: what a reranker over 20 candidates can reach.** The same scratch index,
  with a scratch config at k=20 (a diagnostic, not a config change), scores hit rate 0.95,
  MRR 0.602 and recall 0.90.
  - Three of the four misses have their page in the top 20, at ranks 10, 10 and 11.
  - `glider-slm`'s p. 2 is not in the top 20 at all. No reranker over 20 candidates can
    recover it; that takes a chunking, embedder or hybrid-retrieval change, each a later
    and measured one.
- **Input for S2-4: a copied config needs `--dataset` and `--thresholds`.** The eval files
  are anchored to `eval/` beside the config file (ARCHITECTURE.md §2.3), as `paths` are. S2-4's
  step 3 runs two scratch copies of `config.yaml`. Without those flags each exits 2 with
  `cannot read the golden set <scratch>/eval/eval_dataset.jsonl` (seen here). Pass
  `--dataset eval/eval_dataset.jsonl --thresholds eval/thresholds.yaml`, beside
  `RAG_DATA_PATH`. Both S2-4 inputs are in the STATUS.md backlog.

## Deviation from plan

Drafted for review; finalised at close-out.

- **The floors live in `evaluation/gate.py`.** The story names only `retrieval.py` and
  `cli.py`, but ARCHITECTURE.md §2.2 gives the floors to `gate.py` ("freshness + floors"),
  and the thresholds file is shared with tier 2. Tier 2's check runs in CI's model-free
  job, so the floors cannot live in `retrieval.py`, which imports the chain and FAISS.
  S2-5 adds the `generation:` floors and the freshness check to `gate.py`.
  - `tests/test_architecture.py` asserts that importing `gate` loads none of ragas,
    openai, torch, sentence-transformers or langchain_huggingface. S2-5's story lists
    that test; it is added here because the module exists from here.
- **The floors are rounded down to 3 decimals**: 0.75, 0.537 and 0.666. A floor of exactly
  "baseline minus 0.05" could fail on floating-point rounding at the boundary. Rounding
  loosens each floor by less than 0.001.
- **The cache uses `actions/cache/restore` and `actions/cache/save`.** These come from the
  same repository and SHA as `actions/cache`. The save runs right after the ingest step, not
  at the end of the job. With the combined action, a first run that missed a floor would
  save nothing, and verification 4's "miss, then hit" would fail on the re-run.
- **The new job has `timeout-minutes: 20`.** Without it, a stalled download would run to
  GitHub's 6-hour default.
- **Smaller additions:**
  - `--json` holds every question in golden-set order, for S2-4's question-by-question
    comparison.
  - Exit 2 also covers a `--json` file that cannot be written.
  - Errors flush stdout first, so in a CI log the verdict comes after the table.
  - The deprecation gate imports `rag_qa.evaluation.cli`, so its "every module" claim
    stays true.
- **Model:** this session ran on Opus 5.5, as the story's opus-fast routing asks.
