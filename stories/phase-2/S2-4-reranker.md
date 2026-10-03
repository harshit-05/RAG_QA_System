# S2-4: Rerank with our own cross-encoder, switched on by the numbers

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-4, ISS-03 (both halves), DEC-5 step 2; backlog: declaring `langchain-classic` (closes as not needed), Sources relevance (adds the knob) |
| **Depends on** | S2-1 (`build_retrieve`), S2-3 (the metrics that decide) |
| **Model** | fable |
| **Plan-first** | no |
| **Branch** | `feat/s2-4-reranker`. Risky: it changes the retrieval every answer depends on, and it shrinks the `_target_` allowlist. |

## Goal

FR-4 finally holds: the reranker runs end to end, on about 40 lines of our own over
`sentence-transformers`, instead of a disabled entry that spans two maintenance-mode
packages. Whether it is on by default is decided by tier-1 numbers and measured latency,
not by taste.

Design: ARCHITECTURE.md §2.1 DEC-16, config in §2.3.

## Scope

- **`rag_qa/rerankers.py`: `CrossEncoderReranker(BaseDocumentCompressor)`.**
  - Settings: `model_name`, `top_n=5`, `device="cpu"` and `min_score: float | None = None`.
  - It loads `sentence_transformers.CrossEncoder` once, at construction.
  - `compress_documents` scores each (question, chunk) pair. It keeps the best `top_n`
    at or above `min_score`, and returns copies with `rerank_score` in their metadata.
    The input documents are never modified.
- **Wiring.**
  - `components.build_reranker(config)` returns the reranker, or `None`.
  - `chain.build_retrieve` applies it after the retriever, replacing the
    `ContextualCompressionRetriever` path.
  - `SourceRef.score` carries `rerank_score`.
- **`config.yaml`.**
  - Add `pipeline.query.reranker_candidates: 20`. `build_retrieve` overrides the
    retriever's `k` with it when a reranker is on, so one switch flips both and 20 chunks
    can never reach the prompt without a reranker (caveat 7). No second retriever entry.
  - `schema.py`: `reranker_candidates` is required when `reranker` is set, forbidden when
    it is not, and at least the reranker's `top_n`.
  - Replace `cross_encoder` with `rerankers.ms_marco_minilm_cpu` and
    `ms_marco_minilm_cuda`, named by both axes like the embedders.
  - Set `pipeline.query.reranker` according to the decision below, and put the numbers
    in the comment.
- **The allowlist.**
  - `registry.py`: `ALLOWED_PREFIXES` loses `langchain_classic.` and
    `langchain_community.`, and the docstring says why (an allowlist allows only what is
    used, ADR-020).
  - Update CLAUDE.md's environment-facts line that says the reranker entry and the
    allowlist keep `langchain_classic` named until Phase 2.
- **CI (`ci.yml`).**
  - Add a warm-up step to `eval-retrieval`, before the offline step, that builds the
    reranker from the config through `components.build_reranker` (not a download by
    name), so it caches exactly the files the offline step opens. Add the model's name to
    the cache key. `rag-ingest` loads only the embedder, so without this the offline step
    fails on a cache miss.
- **Tests.**
  - The reranker, with a stub in place of the model (no download): ordering, `top_n`,
    `min_score`, metadata copied rather than mutated, empty input.
  - `build_retrieve` with the reranker on and off: on fetches `reranker_candidates`,
    off fetches the retriever's own k. The schema rule for `reranker_candidates`.
  - **ISS-03 regression, rewritten.**
    - Every test that names `components.rerankers.cross_encoder` has to move:
      `tests/test_registry.py:151` and `tests/test_config_regressions.py:124-152` (two
      cases). The one at :143 uses a `langchain_classic.` path, which now fails the
      prefix check before `check_imports` is reached.
    - Move the case to a misspelled class under an allowed prefix,
      `rag_qa.rerankers.CrossEncoderRerank`, so it still tests what ISS-03 was.
    - Update the docstring at :211.
  - Allowlist tests: a `langchain_classic.` or `langchain_community.` target is now
    refused at load, with the hint. Attack the change too, not just the happy path
    (caveat 14).
- **The decision.** On S2-3's scratch index, run `rag-eval retrieval` for two configs:

  | | Retrieval |
  | --- | --- |
  | (a) | dense, k=5: S2-3's baseline |
  | (b) | k=20 candidates → reranker → top 5 |

  Also time the rerank step on CPU: the median over the golden questions.
  - **Compare question by question, not only the means.** On about 20 questions, one
    question moves a mean by 0.05. List which questions (b) wins (a better first-hit
    rank) and which it loses.
  - **Turn it on** if (b) wins more questions than it loses, loses no hit that (a) had,
    does not lower recall, and its latency is small next to generation (seconds against
    minutes). A tie stays off: the simpler pipeline wins a tie.
  - **Record** the table here and in the config comment.
  - **Re-baseline** the `retrieval:` floors for the chosen config.

## Out of scope

- Tuning `min_score`. S0-6 saw a distance gap between in-corpus and out-of-corpus hits,
  which is prior evidence that a cutoff can work; the tuning is a follow-up once tier 2
  exists (S2-5).
- Hybrid retrieval (Phase 3) and anything on the LLM side.

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q
uv run pytest tests/test_config_regressions.py -v      # ISS-03 still caught, now under rag_qa.

# 2. the allowlist really shrank
grep -n "langchain_classic\|langchain_community" src/rag_qa/registry.py config.yaml   # → no allowlist or _target_ hits

# 3. the decision, on S2-3's scratch index (both configs; the table goes below)
RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-eval retrieval --config <scratch>/dense.yaml \
  --dataset eval/eval_dataset.jsonl --thresholds eval/thresholds.yaml
RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-eval retrieval --config <scratch>/reranked.yaml \
  --dataset eval/eval_dataset.jsonl --thresholds eval/thresholds.yaml
#    Two copies of config.yaml that differ only in pipeline.query. Copied configs anchor
#    their paths to <scratch>, so also set RAG_DATA_PATH=$PWD/corpus. The eval files are
#    found beside the config file, so --dataset and --thresholds point the copies at the
#    real ones; without them each run exits 2 with "cannot read the golden set
#    <scratch>/eval/..." (S2-3). The flags are spelled out on purpose: zsh does not
#    word-split an unquoted $VAR, so a shared variable would arrive as one argument.

# 4. real end to end. The first use downloads the cross-encoder, about 90 MB.
uv run rag-query       # one question: the answer, then the sources in reranked order

# 5. CI
gh run list --limit 2  # check and eval-retrieval green; the warm-up step fetched the cross-encoder
#    Re-run once: the log should show a cache hit for both models.
```

## Review notes for the human

- **The allowlist diff is security-relevant.**
  - Confirm that every `_target_` still loads.
  - Confirm that the two dropped prefixes now fail at load with the hint.
  - Ask whether anything else can now slip through.
- **The decision table sets the default.** Check that it matches the config comment and
  the new floors.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
