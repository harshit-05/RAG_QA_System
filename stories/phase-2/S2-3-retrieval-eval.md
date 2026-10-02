# S2-3: Score retrieval against the golden set, in CI on every push

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-7 (tier 1 of the quality gate) |
| **Depends on** | S2-1 (`build_retrieve`), S2-2 (the golden set) |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `feat/s2-3-retrieval-eval`. Risky: it adds the first CI job with network access, and a gate that can turn PRs red. |

## Goal

Retrieval quality becomes numbers that CI recomputes on every push. For each answerable
golden question, three are taken over what the prompt would actually receive:

- **hit rate**: an expected page appears anywhere in the context;
- **MRR**: the reciprocal rank of the first expected page;
- **recall**: the share of the expected pages the context covers.

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
    partial recall; `page_label` matching; absolute versus relative `source`.
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
