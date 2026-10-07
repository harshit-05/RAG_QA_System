# S2-5: Build the RAGAs harness and the freshness gate

| | |
| --- | --- |
| **Status** | Approved (2026-10-07): PR #13, both reviews done, every finding fixed; CI green on `b6f4e74` (`check`, `eval-retrieval`). Next: merge |
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

## Spike: the working recipe (2026-10-06/07)

Plan mode could not install the `eval` extra, so the APIs were first read from the exact
sources `uv.lock` pins (ragas 0.4.3, instructor 1.17.0, openai 3.3.0, and Ollama 0.22.1's
server). After `uv sync --extra eval`, a script re-checked 35 of those facts against the
installed packages, and all held, once one blocker was dealt with.

**Blocker: ragas 0.4.3 does not import on this stack.** `ragas/llms/base.py` imports
`langchain_community.chat_models.vertexai`, which langchain-community 0.4.2 (the sunset
release, our pin) removed; 0.4 and 0.4.1 still had it. The lock resolves, but `import
ragas` fails. Upstream issue vibrantlabsai/ragas#2741 has been open since 2026-05-24, and
`main` still has the import. ragas uses the class only in an `isinstance()` list on its
legacy LangChain-wrapper path, which `InstructorLLM` never reaches. **Maintainer's call
(2026-10-06): a narrow import shim** in `ragas_scoring.py`. Only when that one module is
missing, a stand-in whose `ChatVertexAI` is a never-instantiated placeholder is
registered. A local test fails once a ragas release no longer needs it.

**The recipe:**

```python
os.environ["RAGAS_DO_NOT_TRACK"] = "true"     # ragas posts usage to its maker otherwise
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama",
                     max_retries=0, timeout=2400)
llm = llm_factory("rag-judge", client=client, temperature=0, seed=42, max_tokens=2048)
# ragas patches an openai client with instructor Mode.JSON (response_format json_object,
# which Ollama turns into grammar-constrained format "json"). gemma2 has no tool support,
# so instructor's default TOOLS mode would fail.
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
Faithfulness(llm=llm); AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
ContextPrecision(llm=llm)  # the with-reference variant
ContextRecall(llm=llm)
# usage, parse errors: llm.client.on("completion:response" | "parse:error", handler)
```

- **The sampling comes from the Modelfile.** ragas sends `temperature 0.01, top_p 0.1,
  max_tokens 1024` on every request, and a request overrides the Modelfile. So `score`
  sends the Modelfile's `temperature`, `seed` and `num_predict` (as `max_tokens`).
- **`max_retries=0`** on the client: openai's default of 2 would re-send a timed-out
  10-minute call twice in silence.
- **instructor wraps connection errors and timeouts in `InstructorRetryException`**, the
  same class as a parse failure that exhausted its retries. So failures are classified
  by cause, and a transport error aborts the run.

**Measured on `yolo-anomaly-model`** (the answerable item with the longest contexts,
4,893 characters over 5 chunks), with Ollama on 2 threads:

| | |
| --- | --- |
| generate (mistral) | ttft 245 s, total 262 s; a 280-character answer |
| judge calls | 11, **0 parse errors** in JSON mode |
| largest prompt | 2,753 tokens (context recall); faithfulness 2,124, precision about 1,410 each |
| largest completion | 258 tokens (faithfulness's verdicts); prompt + completion at most 2,908 |
| `ollama ps` | `rag-judge` CONTEXT 8192, **8.26 GB** resident |
| scores | faithfulness 1.0, answer relevancy 0.906, context precision 1.0, context recall 1.0 |
| time | **2,358 s (39 min)** for the four metrics; slowest call 768 s (context recall) |

Decisions it settled:

- **The instructor mode is JSON**, as `llm_factory` makes it. JSON_SCHEMA is not needed.
- **`strictness=1`.** Answer relevancy's three calls returned the identical question
  under greedy decoding, so a mean over three equals one.
- **`num_predict 2048`** goes into the Modelfile, and is sent as `max_tokens`. This answer
  was short; a 512-token answer could yield many more statements to judge. 2,048 plus the
  largest prompt stays under 5k tokens.
- **`timeout_s: 2400`**, about 3× the slowest call.

**For S2-5b:** at about 39 min per item on 2 threads, a full local scoring of the 20
answerable items is **about 13 h**, not the 3–4 h estimated (caveat 16). Generating took
4.4 min for this one answer. Colab/Kaggle is the clear default for scoring.

## Verification record (2026-10-07)

1. **Gates.** `ruff check`, `mypy` and `pytest --cov=rag_qa` passed with the eval extra
   installed: **638 passed**, 4 xfailed, coverage 96.95%. Then the same on CI's install
   (`uv sync --locked`, no extra): **634 passed, 4 skipped** (the four that need the extra),
   and mypy clean both ways. `HF_HOME=<empty> HF_HUB_OFFLINE=1 uv run pytest` gave the same,
   and the folder stayed empty (caveat 23).
2. **Smoke run against real Ollama**, run by the maintainer in a terminal.
   - `generate --limit 1`: `glider-purpose`, ttft 264 s, total 304 s, mistral's digest
     recorded.
   - `score --limit 1`: faithfulness 1.000, answer relevancy 0.828, context precision
     1.000, context recall 1.000. 0 parse failures, 0 truncated, 0 retried.
   - Largest prompt 3,017 tokens (below 7,936); largest judge answer 526 tokens.
   - `ollama ps`: `rag-judge` CONTEXT 8192, 8.3 GB, 100% CPU.
   - Scoring took about 54 min.

   The run preceded the reviews' fixes. Of those, the check of the served judge against
   every Modelfile PARAMETER was re-run against the real Ollama: the real `rag-judge`
   passes, and a Modelfile edited without `ollama create` is refused.
3. **`pytest tests/test_eval_gate.py -v`: 81 passed.** By hand, on a fixture run built
   from this checkout (all 25 ids):
   - `check` → exit 0;
   - S2-5b's recipe (a scratch copy of `config.yaml`, "helpful and precise" → "helpful,
     precise", `RAG_DATA_PATH=$PWD/corpus`, `--eval-dir`) → exit 1, "query changed: re-run
     generate and score (…)";
   - a ground-truth edit in a scratch golden copy → exit 1, "references changed: re-run
     score (…)".
4. **CI on PR #13: green.** The first run failed only at the Audit step, on the
   multidict advisory (Discovered). After the bump, `check` and `eval-retrieval` both
   passed (run 37547537746). `check` runs without the eval extra, so the import guard
   holds there too.

## Reviews (Opus 5.5, two passes, as for every Fable-routed story)

**First review: 10 findings; 9 fixed, 1 deferred.**

- **Fixed:**
  - Scores record the answers file's sha256. A re-generate with unchanged inputs moved no
    part, so `check` would have passed scores of answers no longer committed.
  - `classify` uses the final cause. instructor wraps a timeout on a retry with the
    earlier failed parse attempt, which would have counted as a parse failure.
  - `rag-eval` has an exit-2 backstop. An unforeseen error in `check` exited 1, which CI
    reads as a verdict.
  - The served judge is checked against every Modelfile PARAMETER and its FROM, not only
    `num_ctx`.
  - `retried` no longer counts a failed call's last attempt.
  - `check --with-ollama` asks no Ollama about a generator that is not one.
  - `/api/ps` is polled once.
  - `run_retrieval` shares `_crashed`.
  - `datasets` and `pandas` left the eval extra: nothing imports them now.
- **Deferred: resumable runs.** See Discovered.

**Second review: 10 findings, all fixed.**

- **The first review's `retried` fix did not work.** instructor also passes
  `attempt_number` and `max_attempts`, and falls back to calling a handler with the error
  alone when its signature cannot take them. A test now goes through instructor's
  `Hooks`.
- **A judge that misparsed most items could pass on the mean of the rest.** A required
  `max_unscored` in the `generation:` floors now fails a gated metric with more unscored
  items than that.
- **Other fixes:**
  - `score --answers <scratch>` without `--out` is refused, so it can never overwrite the
    committed scores;
  - a base digest that was never recorded no longer fails `--with-ollama` for good;
  - `top_p` is forwarded from the Modelfile;
  - an `$OLLAMA_HOST` without a port means 11434;
  - `write_run` uses a unique temporary file and removes it on failure;
  - the `ollama create` hint names the real Modelfile path;
  - a moved `corpus` or `ingestion` part says "re-run rag-ingest, then generate and
    score";
  - a test keeps the floors' keys in step with the metric set.

**The maintainer's review (2026-10-07): no blockers, five low findings, and notes for the
second reviewer.** CI and the gates were confirmed green on PR #13. Applied on the PR:

- Runtime-only llm keys no longer move `query`: `validate_model_on_init`, `keep_alive`
  and the HTTP clients' settings. Each would have forced a 13-hour re-score for an edit
  that changes no answer. `num_thread` stays hashed, as the backlog intends.
- `judge` hashes the Modelfile without its comment and blank lines. A SYSTEM or TEMPLATE
  edit still moves it, and a test pins both.
- `_Setup.config` is typed `RagConfig`; the stray `#:` comment above `metric_inputs` is a
  docstring now.

Recorded rather than changed:

- whether a Colab `ollama create` gives the same digest as a local one: an input to S2-5b;
- resumable runs: S2-5b's first backlog line;
- answer relevancy's embedder pinned by name, not revision: accepted, as DEC-17 did.

Left for the code-quality reviewer: splitting `score()`, and moving the Ollama client and
the run files out of `gate.py`.

**The second reviewer (2026-10-07, on the risky story): no blockers; 3 findings, all
fixed on the PR.**

- **Checked and sound:**
  - failure counting, against instructor 1.17's own retry code: the same exception
    classes, a bare truncation, and a wrapped parse error with its cause;
  - `max_unscored`, counted per item and metric;
  - a NaN becomes no score;
  - exit 2 for anything unforeseen;
  - `rag_qa.evaluation.cli` imports no model stack;
  - every Query field is hashed;
  - the corpus part is the same in a clean clone, because the untracked `.parquet` files
    have no loader.
- **instructor's version moved no part (low).**
  - In JSON mode instructor adds its own system message to every judge call
    (`v2/providers/openai/handlers.py:871`), and writes the re-ask after a parse failure.
  - So a `uv lock --upgrade` would change the judge's prompts while `check` still passed.
  - `fingerprint.INSTRUCTOR_VERSION` (1.17.0) is now hashed into `judge` and held to
    `uv.lock` by a test, and `score` refuses a mismatch.
  - This moves `judge`, which costs nothing before the first baseline.
- **The served judge was checked one way only (low).**
  - The check covered the FROM and each PARAMETER the Modelfile sets. It missed a SYSTEM
    or TEMPLATE, and a parameter left from an older Modelfile, each of which `judge`
    hashes as the Modelfile has it.
  - The real Ollama's `/api/show` shows that a created model serves its base's
    parameters and TEMPLATE, with the Modelfile's in their place. So `check_judge` now
    compares against both: the Modelfile, and the base for anything the Modelfile does
    not set.
  - Without the base in Ollama, only what the Modelfile sets is checked.
  - `parse_modelfile` now reads SYSTEM and TEMPLATE, including `"""` blocks, and any
    whitespace after a keyword.
  - The real `rag-judge` still passes. A negative control that switched the new check off
    failed all four of its refusal tests.
- **`--with-ollama` skipped a generator digest (nit).** Its rule for skipping a missing
  base digest matched by model name, so it also skipped the generator's when gemma2 both
  answers and underlies the judge. The base entry is now flagged instead.
- **Gates after the fixes:** ruff and mypy clean; 655 passed, 4 xfailed, 97% coverage,
  with the eval extra installed. CI's install without the extra is next, on the push.

## Discovered

- **CI's Audit step went red on the PR (2026-10-07)** with a new advisory: multidict 6.9.0,
  CVE-2026-104874, fixed in 6.9.1. It arrives through aiohttp and yarl
  (langchain-community), and `main` locked the same version, so every PR would have been
  red. It is unrelated to S2-5's code, and lint, types and tests had passed. The
  maintainer chose a separate commit on this PR: `uv lock --upgrade-package multidict`
  changes only that package. On CI's install, the Audit step now finds nothing, and the
  suite passes.
- **ragas 0.4.3 does not import with langchain-community 0.4.2** (the spike section above).
  The shim goes once a ragas release stops importing that module, and a test says when.
- **Scoring is not resumable.** A transport error or a timeout on item 19 aborts a 13-hour
  run with nothing written; so does a failure in `generate`. Backlog, for S2-5b to weigh
  before its baseline: write progress as it goes, and skip ids already scored.
- **Tier-2 cost on this CPU** is about 39–54 min per answerable item, so a full local
  scoring is about 13–18 h, not caveat 16's 3–4 h. Colab/Kaggle is the default for S2-5b.
- **Two Ctrl-C tests in `test_cli.py` flaked once under load** (the second reviewer).
  In one full run, while other `uv` commands ran beside it, `test_ctrl_c_while_a_chunk_is_processed_leaves_no_stream_open_at_the_prompt`
  and `test_two_ctrl_cs_during_an_answer_end_the_session` failed, along with one other
  whose name the log cut off. Three full runs since passed, two with coverage. S2-5
  touches none of their code. Backlog (STATUS.md).
- **For S2-5b, from the second reviewer:**
  - Run `rag-eval check` from a clean clone before committing the baseline. A file that
    git ignores but a loader reads, such as a stray `.md` in `corpus/`, would make a
    baseline taken here fail in CI.
  - `generate` checks the index, then builds the pipeline seconds later. Only a
    `rag-ingest` running at that moment could swap the generation in between, so do not
    ingest while generating.
- **The judge also runs on Ollama's 2 threads.** S2-5b's `num_thread` measurement could
  give the judge Modelfile a `PARAMETER num_thread` too. That moves `judge`, which is
  harmless before the first baseline.

## Deviation from plan

- **Model:** Opus 5.5 at max effort in place of Fable (CLAUDE.md, from 2026-10-04), with
  two review passes.
- **The spike ran after plan approval, not in plan mode.** Plan mode cannot install the
  eval extra or create the judge. The APIs were first read from the pinned sources, then
  re-checked against the installed packages (35 facts) before any code.
- **The ragas import shim** (maintainer, 2026-10-06), above.
- **Maintainer decisions taken during planning:** the Modelfile is resolved in the eval
  folder, not beside the config (§2.3 note); answer relevancy uses MiniLM on CPU;
  `generate` refuses an index that lags the corpus.
- **Beyond the fingerprint table:**
  - `CHUNKING_VERSION` is in `ingestion`;
  - `decline_marker`, the embedder, the instructor mode, the metric settings
    (`strictness=1`) and `RAGAS_VERSION` are in `judge`;
  - `max_unscored` and the answers hash came from the reviews.

  All are recorded as dated notes on DEC-15.
- **`num_predict 2048`** is in the Modelfile beyond the story's four lines, from the
  spike. The judge config gained `embedding_model` and `timeout_s`.
- **Whole-word matching uses lookarounds** rather than `\b`. It gives the same result for
  the golden set's words, and is right for an entry that ends in punctuation.
