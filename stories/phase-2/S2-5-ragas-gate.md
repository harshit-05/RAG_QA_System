# S2-5: Build the RAGAs harness and the freshness gate

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-7 (the harness; the baseline and the CI step are S2-5b), ISS-15; backlog: the prompt placeholder check (FR-8 follow-up), pointing the RAGAs judge at local Ollama |
| **Depends on** | S2-2 (the golden set), S2-4 (the final retrieval config), S2-6 (`manifest.py`'s identity functions); S2-1 for `stream_answer` |
| **Model** | fable |
| **Plan-first** | yes |
| **Branch** | `feat/s2-5-ragas-gate`. Risky: it adds a CI gate that can turn every PR red, and a new optional-dependency surface (ragas, openai). |

## Goal

This story builds the generation-quality harness. It makes no hours-long run, and it adds
no CI step yet: those are S2-5b, once the code is reviewed. It was split from one story
because a spike, a schema change, three modules and a multi-hour baseline do not fit one
session. After S2-5b, generation quality is measured, committed and enforced:

- RAGAs faithfulness, answer relevancy, context precision and context recall, with a
  local `gemma2:9b` judge;
- a deterministic decline rate on questions the corpus cannot answer;
- `rag-eval check`, which fails CI when a score drops below its floor, or when the
  committed run no longer matches the repository.

This completes the "enforced quality bar" half of the Phase 2 exit. Design:
ARCHITECTURE.md §2.1 DEC-15, config in §2.3, file shapes in §2.4.

## Scope

- **`eval/judge.Modelfile` first** (DEC-15): `FROM gemma2:9b`, `PARAMETER num_ctx 8192`,
  `PARAMETER temperature 0`, `PARAMETER seed 42`. Then
  `ollama create rag-judge -f eval/judge.Modelfile`, which reuses gemma2's weights, so
  there is no download. Ollama's OpenAI endpoint cannot set the context size, and its 4k
  default would silently truncate RAGAs' prompts (caveat 22).
- **Spike, in plan mode.** Wire ragas 0.4.3 to `rag-judge` through Ollama's
  OpenAI-compatible endpoint:
  `llm_factory("rag-judge", client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))`.
  - Settle three things: the four `ragas.metrics.collections` metrics, the instructor
    mode Ollama needs, and the embedder that answer relevancy needs.
  - **Measure the context.** Log `usage.prompt_tokens` for every metric call on the
    golden item with the longest contexts. Check that `ollama ps` shows CONTEXT 8192 for
    `rag-judge`, and note the resident memory: about 8 GB is expected with the 8k KV
    cache, but that is an estimate, so measure it.
  - Done when, on one question, every metric returns a number, a judge parse failure is
    counted rather than crashing the run, and no prompt comes within 256 tokens of
    `num_ctx`.
  - Record the working recipe here before building on it.
- **`schema.py`.**
  - An optional `evaluation` section: `decline_marker`, plus `judge` with `model`,
    `modelfile`, `base_url`, and whatever else the spike shows the judge needs.
  - `decline_marker` must appear in `pipeline.query.prompt.system`.
  - The `Prompt` placeholder rules from §2.3:
    - `human` has `{context}` and `{question}`;
    - `system` has no `{context}`;
    - no other `{placeholder}` appears;
    - parsing uses `string.Formatter`.
- **`evaluation/generation.py`: `rag-eval generate`.**
  - It consumes `stream_answer` for every golden question, so it measures exactly what
    users get.
  - It writes `eval/runs/answers-latest.json`, or `--out PATH`, holding:
    - the parts `query`, `ingestion`, `corpus` and `questions`, and the generator model
      with its Ollama digest (recorded, not hashed);
    - for each item: the answer, contexts, sources, `ttft_ms` and `total_ms`.
- **`evaluation/ragas_scoring.py`: `rag-eval score`.**
  - It is the only module that imports ragas or openai, both from the `eval` extra, and
    only inside functions.
  - It is omitted from coverage, and gets a mypy no-stub override if needed.
  - It scores the answerable items with the four metrics.
  - It computes the decline rate over the unanswerable ones: the answer contains
    `decline_marker`, and none of the record's `must_not_contain`. Both comparisons are
    case-insensitive: the answer and each entry are lower-cased, so "coco" leaks "COCO".
    A `must_not_contain` entry matches **as a whole word** (`\b` on both sides, the
    entry `re.escape`d), so "COCO" does not fire on "cocoa", nor "H100" on "H1000"
    (S2-2 second review). The marker stays a substring match, because it is a phrase.
    - Accepted cost: a refusal that names a leak word to explain itself ("the references
      mention Oxford, but not CADE-8's venue") counts as a leak. S2-5b's three hand-checked
      answers should include an unanswerable one, so this shows up if it happens.
  - Before scoring, it checks that every unanswerable record's `ground_truth` contains
    `decline_marker`, and exits 2 naming the record if not. The golden-set loader cannot
    check this: it knows nothing of the config (S2-2 review).
  - **It copies the generation parts from the answers file**, and refuses to score an
    answers file whose `query`, `ingestion`, `corpus` or `questions` part does not match
    the checkout it runs in. `references` and `judge` are computed at score time.
  - **It checks the judge before scoring:** the served `rag-judge`'s `num_ctx` (from
    `/api/show`) must equal the Modelfile's, or it exits 2. During the run it fails if
    any prompt comes within 256 tokens of `num_ctx`.
  - It records the judge's digest and its base model's (not hashed), `num_ctx`, the
    largest `prompt_tokens` seen, and the parse failures per metric.
  - It writes `eval/runs/generation-latest.json`, or `--out PATH`.
- **`evaluation/gate.py`: `rag-eval check [--config PATH] [--eval-dir DIR]
  [--with-ollama]`.** It needs no models and no Ollama.
  - It recomputes the fingerprint from the checkout: the config, the corpus, the golden
    set and the judge Modelfile.
  - It compares that with both committed files, and requires the two files to agree on
    the generation parts.
  - When the run is stale, it names the part that moved and the re-run that fixes it.
  - It checks the `generation:` floors.
  - `--with-ollama` (local only) also compares the recorded model digests with the live
    ones.
  - Exit 0 when everything passes, 1 for a floor breach or a stale run, 2 when it cannot
    start.
- **`evaluation/fingerprint.py`: the fingerprint**, exactly as ARCHITECTURE.md §2.1
  (DEC-15) tabulates it:
  - six parts: `query`, `ingestion`, `corpus`, `questions`, `references`, `judge`;
  - `query` includes a rendered-prompt probe: the chat prompt rendered with
    `chain.format_docs` over two fixed fake documents. A change to `format_docs` or
    `citation()` then moves it, not only a config edit. Code in general is not hashed
    (ARCHITECTURE.md §2.1, "Honest limit");
  - model digests are **recorded, never hashed**, because CI has no Ollama and cannot
    recompute them;
  - canonical JSON with sorted keys, then sha256;
  - paths, `base_url`, device keys, and the golden set's `notes` and `expected_sources`
    are excluded;
  - specs are hashed by content, not by ref name, so renaming an entry moves nothing;
  - the `ingestion` part is computed by `manifest.spec_identity` and
    `manifest.embedder_identity`, both pure.
  - **S2-6 created those two functions in `rag_qa/manifest.py`; reuse them unchanged.**
    A second implementation would move the fingerprint later, and that means an
    hours-scale re-run of tier 2.
  - It may import `langchain_core` and `rag_qa.chain`, and nothing heavier
    (ARCHITECTURE.md §2.2).
- **Packaging.**
  - `rag-eval` gets its `generate`, `score` and `check` subcommands. `generate` and
    `score` take `--limit N` (the first N items) and `--out PATH`, and `score` takes
    `--answers PATH`. These serve the spike and smoke runs, which must not overwrite the
    committed files.
  - Coverage `omit` moves from `evaluate.py` to `evaluation/ragas_scoring.py`.
  - `src/rag_qa/evaluate.py` is deleted.
- **Tests.**
  - Fingerprint sensitivity: each input moves its own part and only that one, while
    paths, `base_url`, `notes`, `expected_sources` and a renamed component entry move
    nothing. A `ground_truth` edit moves `references` only; a `format_docs` change moves
    `query`.
  - Stale messages name the right re-run.
  - `score` refuses a mismatched answers file, and a served judge whose `num_ctx`
    differs from the Modelfile (both with a fake Ollama).
  - `tests/test_architecture.py`: importing `evaluation.gate` loads none of ragas,
    openai, torch, sentence-transformers or langchain_huggingface.
  - Floors, and the decline-rate arithmetic, including a paraphrased refusal and a
    `must_not_contain` leak.
  - Placeholder validator cases: `{context}` missing, `{context}` in system, a stray
    `{x}`, and escaped `{{x}}` allowed.
  - `decline_marker` missing from the system prompt.
  - A `must_not_contain` leak in a different case ("coco" for "COCO") still counts as a
    leak, and an unanswerable `ground_truth` without the marker is refused.
  - Whole-word matching: "cocoa" is not a COCO leak, and "H100." at the end of a sentence
    is an H100 leak.
  - `generate` against a fake pipeline, with no Ollama.

## Out of scope

- **The baseline run, the `generation:` floors, the committed run files and the CI step**:
  S2-5b.
- **The DEC-2 revisit** (mistral vs phi3 vs qwen2): it stays on the backlog, after this
  story.
- **Tuning the prompt or `min_score`**: follow-ups, each measured by this harness.
- **Running tier 2 in CI**: excluded by DEC-15.

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. the harness, on one question, against real Ollama (the spike's recipe).
#    Check `ollama ps` and `free -h` first (caveat 8); never run both models together (caveat 17)
ollama create rag-judge -f eval/judge.Modelfile  # no download: reuses gemma2:9b's weights
uv run rag-eval generate --limit 1 --out <scratch>/answers.json
ollama stop mistral
uv run rag-eval score --limit 1 --answers <scratch>/answers.json --out <scratch>/generation.json
#    → every metric returns a number, or a counted parse failure; the max prompt_tokens is
#      below 8192 - 256; `ollama ps` shows rag-judge with CONTEXT 8192

# 3. the gate, on a fixture run (no committed baseline yet)
uv run pytest tests/test_eval_gate.py -v         # every fingerprint part, stale messages, floors
#    stale detection by hand: a one-word prompt change in a scratch config copy
#    → exit 1, and the message says "query changed: re-run generate and score"

# 4. CI
gh run list --limit 2                            # check and eval-retrieval green; no rag-eval step yet
```

## Review notes for the human

- **Read the spike's recipe and its parse-failure counting.** A judge that misparses
  quietly produces confident numbers, which S2-5b then turns into floors.
- **Read a stale message.** It must say which re-run fixes it, or the gate will be
  resented and bypassed.
- **Check that `score` refuses a mismatched answers file, and a judge served with the
  wrong context size.** The second is the one that fails silently if it is missed.
- **Check the import guard on `evaluation.gate`.** If ragas or torch ever leaks into it,
  CI's `check` job breaks.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
