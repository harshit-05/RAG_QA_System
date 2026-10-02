# S2-5: Measure generation with RAGAs, and fail CI on a drop or a stale run

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-7, ISS-15; backlog: the prompt placeholder check (FR-8 follow-up), pointing the RAGAs judge at local Ollama |
| **Depends on** | S2-2 (the golden set), S2-4 (the final retrieval config); S2-1 for `stream_answer` |
| **Model** | fable |
| **Plan-first** | yes |
| **Branch** | `feat/s2-5-ragas-gate`. Risky: it adds a CI gate that can turn every PR red, and a new optional-dependency surface (ragas, openai). |

## Goal

Generation quality is measured, committed and enforced:

- RAGAs faithfulness, answer relevancy, context precision and context recall, with a
  local `gemma2:9b` judge;
- a deterministic decline rate on questions the corpus cannot answer;
- `rag-eval check`, which fails CI when a score drops below its floor, or when the
  committed run no longer matches the repository.

This completes the "enforced quality bar" half of the Phase 2 exit. Design:
ARCHITECTURE.md §2.1 DEC-15, config in §2.3, file shapes in §2.4.

## Scope

- **Spike first, in plan mode.** Wire ragas 0.4.3 to `gemma2:9b` through Ollama's
  OpenAI-compatible endpoint:
  `llm_factory("gemma2:9b", client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))`.
  - Settle three things: the four `ragas.metrics.collections` metrics, the instructor
    mode Ollama needs, and the embedder that answer relevancy needs.
  - Done when, on one question, every metric returns a number, and a judge parse failure
    is counted rather than crashing the run.
  - Record the working recipe here before building on it.
- **`schema.py`.**
  - An optional `evaluation` section: `decline_marker`, plus `judge` with `model`,
    `base_url`, and whatever else the spike shows the judge needs.
  - `decline_marker` must appear in `pipeline.query.prompt.system`.
  - The `Prompt` placeholder rules from §2.3:
    - `human` has `{context}` and `{question}`;
    - `system` has no `{context}`;
    - no other `{placeholder}` appears;
    - parsing uses `string.Formatter`.
- **`evaluation/generation.py`: `rag-eval generate`.**
  - It consumes `stream_answer` for every golden question, so it measures exactly what
    users get.
  - It writes `eval/runs/answers-latest.json`, holding:
    - the fingerprint, without `judge`, and the generator model;
    - for each item: the answer, contexts, sources, `ttft_ms` and `total_ms`.
- **`evaluation/ragas_scoring.py`: `rag-eval score`.**
  - It is the only module that imports ragas or openai, both from the `eval` extra, and
    only inside functions.
  - It is omitted from coverage, and gets a mypy no-stub override if needed.
  - It scores the answerable items with the four metrics.
  - It computes the decline rate over the unanswerable ones: the answer contains
    `decline_marker` (case-insensitive), and none of the record's `must_not_contain`.
  - It writes `eval/runs/generation-latest.json`.
- **`evaluation/gate.py`: `rag-eval check [--config PATH] [--eval-dir DIR]`.** This is
  pure Python.
  - It recomputes the fingerprint from the repo: the config, corpus and dataset.
  - It compares that with both committed files, and requires the two files to agree with
    each other.
  - When the run is stale, it names the part that moved and the re-run that fixes it.
  - It checks the `generation:` floors.
  - Exit 0 when everything passes, 1 for a floor breach or a stale run, 2 when it cannot
    start.
- **The fingerprint**, exactly as §2.4 describes:
  - parts: `query`, `ingestion`, `corpus`, `dataset`, `judge`;
  - canonical JSON with sorted keys, then sha256;
  - paths and `base_url` are excluded;
  - the `ingestion` part is computed by `manifest.spec_identity` and
    `manifest.embedder_identity`, both pure.
  - **Whichever of S2-5 and S2-6 lands first creates those two functions in
    `rag_qa/manifest.py`, and the other reuses them unchanged.** Two implementations
    would move the fingerprint when the second one lands, and that means an hours-scale
    re-run of tier 2.
- **Packaging.**
  - `rag-eval` gets its `generate`, `score` and `check` subcommands.
  - Coverage `omit` moves from `evaluate.py` to `evaluation/ragas_scoring.py`.
  - `src/rag_qa/evaluate.py` is deleted.
- **The baseline run.** It is hours-scale, so surface the estimate first and let the
  maintainer choose local or Colab/Kaggle (CLAUDE.md GPU policy, caveats 16–18).
  1. Run `rag-eval generate` here with mistral, about 40 min. Check `ollama ps` and
     `free -h` first (caveat 8).
  2. Run `ollama stop mistral`.
  3. Run `rag-eval score`: locally that is about 3–4 h. On Colab/Kaggle, run Ollama with
     the same `gemma2:9b` tag; record that recipe here.
  4. Check three scored answers by hand against the judge's verdicts.
  5. Set the `generation:` floors to the baseline minus a stated tolerance, and record
     both numbers here.
  6. Commit both run files.
- **CI:** `uv run rag-eval check` becomes the last step of the `check` job.
- **Tests.**
  - Fingerprint sensitivity: each input moves its own part and only that one, while
    paths and `base_url` move nothing.
  - Stale messages name the right re-run.
  - Floors, and the decline-rate arithmetic, including a paraphrased refusal and a
    `must_not_contain` leak.
  - Placeholder validator cases: `{context}` missing, `{context}` in system, a stray
    `{x}`, and escaped `{{x}}` allowed.
  - `decline_marker` missing from the system prompt.
  - `generate` against a fake pipeline, with no Ollama.

## Out of scope

- **The DEC-2 revisit** (mistral vs phi3 vs qwen2): it stays on the backlog, after this
  story.
- **Tuning the prompt or `min_score`**: follow-ups, each measured by this harness.
- **Running tier 2 in CI**: excluded by DEC-15.

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
uv run pytest tests/test_eval_gate.py -v         # every fingerprint part, and the floors

# 4. what was committed
uv run python -c "import json; r = json.load(open('eval/runs/generation-latest.json')); print(r['judge'], r['aggregate'])"

# 5. CI
gh run list --limit 2                            # check (including rag-eval check) and eval-retrieval green
```

## Review notes for the human

- **Before trusting any floor, read the three hand-checked answers.** A judge that
  misparses quietly produces confident numbers.
- **Then read a stale message.** It must say which re-run fixes it, or the gate will be
  resented and bypassed.
- **The two run files are the maintainer's attestation.** Their fingerprints must match
  the tree being merged.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
