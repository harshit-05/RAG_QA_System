# S2-4: Rerank with our own cross-encoder, switched on by the numbers

| | |
| --- | --- |
| **Status** | Done 2026-10-05 (PR #11). Verification 1–5 run and shown: CI reproduced the reranked baseline question by question, and its later runs hit the exact cache key with every model load offline. Reviewed twice; the follow-ups are `2f17e6c` and `739bf34`. Deviation in one line: the decision forced small changes outside the Scope list (the eval CLI names the candidates, three README lines), and the migration note was added in review |
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

- **The decision: on.** Verification 3 ran both configs on a fresh scratch index (1,708
  chunks, built offline from the cached MiniLM, as S2-3 did). (a) reproduced S2-3's
  baseline exactly, question by question.

  | | Retrieval | Hit rate | MRR | Recall |
  | --- | --- | --- | --- | --- |
  | (a) | dense, k=5 | 0.800 (16 of 20) | 0.5875 | 0.7167 |
  | (b) | 20 candidates → reranker → top 5 | **0.950** (19 of 20) | **0.7958** | **0.8583** |

  Question by question, by the rank of the first hit:

  | (b) against (a) | Questions |
  | --- | --- |
  | wins, 8 | `glider-name` (miss → 2), `glider-training-data` (miss → 1), `ohlbach-wrightson-mkrp` (miss → 1), `glider-human-study` (4 → 1), `lusk-overbeek-itp` (3 → 1), `stickel-ring-commutativity` (3 → 1), `wos-linked-inference` (3 → 1), `yolo-objectives` (2 → 1) |
  | losses, 4 | `glider-data-filtering` (2 → 4), `yolo-anomaly-model` (2 → 3), `siekmann-unification-hierarchy` (1 → 3), `yolo-acronym` (1 → 2): each still a hit |
  | ties, 8 | seven at rank 1 in both, and `glider-slm`, a miss in both |

  - **Every condition holds.** (b) wins 8 and loses 4. It loses no hit (a) had. Mean
    recall rises from 0.717 to 0.858. And its latency is seconds against minutes of
    generation (below).
  - **One question's recall fell.** `yolo-objectives` went from 2/2 to 1/2: its p. 4
    dropped out of the top 5, while its first hit rose to rank 1. The rule is on mean
    recall, which rose, but the drop is recorded here.
  - **0.95 is the ceiling S2-3 predicted.** `glider-slm`'s p. 2 is not among the 20
    candidates, so no reranker over them can recover it.
  - **Deterministic here.** A second reranked run gave identical items and aggregates.
- **Rerank latency on this CPU** (i7-1255U, torch on 10 threads, Ollama idle): over the 20
  questions × 3 rounds, reranking 20 candidates takes a median of **1.44 s** (min 0.72 s,
  max 2.01 s; round medians 1.34, 1.53 and 1.44 s). Fetching the 20 candidates takes
  19 ms. Loading the model adds 2.5 s once, at startup.
  - **For Phase 3's NFR-2 decision:** the rerank now sits before the first token of
    every answer, so it adds about 1.4 s to `ttft_ms` on this CPU.
  - Not tuned here: torch's thread count (10 includes the efficiency cores) and the
    batch size are the obvious levers.
- **Score units, for the `min_score` follow-up.** ms-marco-MiniLM-L-6-v2 loads with an
  `Identity` activation, so `rerank_score` is a raw logit. The kept chunks scored from
  -6.42 to 8.38 over the golden questions.
- **API facts, verified against the installed versions before writing code**
  (langchain-core 1.6.3, sentence-transformers 6.1.0, pydantic 2.13.5):
  - **`BaseDocumentCompressor`** has an empty `model_config`, so pydantic ignores unknown
    keys, and a misspelled `topn: 3` would be dropped silently. Our class sets
    `extra="forbid"`.
  - **`CrossEncoder(model_name_or_path, *, device=…)`.** The old spellings (`model_name=`,
    a positional `device`) are still accepted, but reported only by
    `logger.warning_once`, never as a warning. So the deprecation gate would not see
    them. We pass the name positionally and the device by keyword.
  - **`predict` reads a list of `(question, text)` tuples as a batch**, a one-candidate
    list included: a list counts as one pair only when its first element is not a list
    or tuple. `rank()` would sort internally and needs `num_labels == 1`. We call
    `predict`, so the ordering is our code and the stub tests it, and we check
    `num_labels` at construction.
  - **FAISS hands back the docstore's own `Document` objects** (`InMemoryDocstore.search`
    returns `self._dict[id]`). Writing `rerank_score` into them would change the loaded
    index; the reranker returns copies.
  - **A sync `RunnableLambda` runs in an executor thread** on `ainvoke`. So the reranker
    inside `retrieve` stays off the event loop, as DEC-14 assumes.
  - **A failing pydantic `mode="after"` validator stops the ones after it.** So the
    `top_n` check, defined after `check_references`, always finds the reference resolved.
- **The download and the cache.** The first build fetched the model under its configured
  repo id, `models--cross-encoder--ms-marco-MiniLM-L-6-v2`: 88 MB, as large as MiniLM. The
  warm-up command CI now runs did the download here. The offline eval runs after it
  passed with `HF_HUB_OFFLINE=1`.
- **Hermeticity can no longer be checked with the local cache.** The cross-encoder is
  now in this host's HF cache, so a test that loaded the real model would pass here and
  fail only in CI. CI's `check` job sees the world as
  `HF_HOME=<empty dir> HF_HUB_OFFLINE=1`: the suite passed that way (350 passed, 4
  xfailed), and the folder stayed empty. Added as STATUS.md caveat 23.
- **For S2-8: one more way for `rag-query` to fail at startup.** With the reranker on,
  startup also loads the cross-encoder, so a machine that is offline before its first
  query fails with a traceback, as S2-8's backlog line already describes for Ollama. The
  startup guard planned there should name this case as well: "the reranker's model is
  not in the HF cache; run once with network".
- **Verification 4's transcript** (2026-10-04, phi3, by caveat 8: 5.4 GiB available,
  1.9 GiB already in swap; a scratch copy differing only in `pipeline.query.llm`, against
  the real index, read-only). Asked "How many domains and how many evaluation criteria
  was GLIDER trained on?", one that dense retrieval missed: 685 domains and 183
  criteria, right. Sources came back in reranked order, p. 1, 12, 3, 7, 5, which is
  tier 1's (b) row for `glider-training-data`. 3 min 45 s in all, exit 0.

## Deviation from plan

- **Model:** this session ran on Opus 5.5 at max effort, not Fable, which the story
  names: Fable is unavailable from 2026-10-04 (CLAUDE.md, "Model routing").
- **Small additions the decision made necessary, outside the Scope list:**
  - **`rag-eval retrieval`'s header and `--json` name the candidates.** With a reranker
    they still showed the retriever's `k` of 5 while 20 were fetched, so a recorded run
    would misstate itself. Now: "reranker: …, over 20 candidates in place of k", and
    `reranker_candidates` in the JSON.
  - **`README.md`, three lines that turning the reranker on made false:** the pipeline
    diagram's "top 5 chunks", the stack table (a Reranker row), and the first-run
    download (two models of about 90 MB each). S2-9's README scope does not cover them.
- **The schema rule needs `top_n` stated in the reranker's entry.** Loading never
  imports the class to read its default (DEC-7), so the "at least `top_n`" check reads the
  entry. `reranker_candidates` must also be at least 1.
- **The reranker refuses what it cannot use, at construction:** unknown keys
  (`extra="forbid"`), `top_n` below 1, and a model with more than one score per pair. A
  classification cross-encoder would otherwise fail obscurely at the first question.
- **Tests:**
  - `use_fakes` switches the reranker off, so the hermetic fixtures never load its model.
  - Reranker tests switch it on with `use_fakes_and_reranker`, under
    `stub_cross_encoder`.
  - `test_registry` now import-checks every `_target_` in the real config, used or not.
    That makes the review note "every `_target_` still loads" executable.
  - The allowlist attack tests show that a dropped package cannot come back through our
    own modules' re-exports (`rag_qa.vectorstore.FAISS`, `rag_qa.rerankers.CrossEncoder`).
- **Verification commands:**
  - Step 3 ran with `--json`, for the comparison, and offline after the warm-up, as CI
    does.
  - Step 4 used phi3, by caveat 8.
  - Step 5 ran on the PR, since CI runs on pull requests only. Three runs, all green with
    the same aggregates (0.950, 0.796, 0.858):
    - run 37180934412 on `1c38a09`: a prefix restore of the embedder-only cache, the
      cross-encoder downloaded by the warm-up, and both models saved under the new key;
    - its re-run (attempt 2): an exact hit, with ingest, warm-up and eval all offline;
    - run 37226206144 on the review commits (`739bf34`): an exact hit, all offline.
- **First review (2026-10-04).**
  - STATUS.md's DEC-7 row records that S2-4 removed the two prefixes, and that a v0.2
    config no longer loads until its old entry is deleted.
  - S2-7 is told to build the reranker once, in its lifespan, rather than on every swap
    of the retrieval half (`build_retrieve` builds one on each call: about 2.5 s and
    90 MB).
  - S2-9 owes three things: the README's v0.2 config migration, an annotation on §1.1's
    DEC-7 paragraph, and ADR-007's now-moot `langchain_classic` exception.
- **Second review (2026-10-04).**
  - **Verified:**
    - **CI's first PR run reproduced the reranked baseline** (0.950, 0.796, 0.858), with
      per-question tables identical to this host's, reranked order included. So the
      floors rest on numbers that hold across CPUs.
    - **Hermetic as CI runs it:** the suite passed with an empty HF cache and offline,
      and the cache folder stayed empty.
    - **Concurrent reranks are safe:** 40 concurrent reranks from 8 threads on one
      `CrossEncoderReranker` raised nothing, and matched sequential results exactly.
    - **No config mutation:** the `k` override edits a fresh copy (`model_dump`), and
      `search_kwargs` always exists.
    - **The allowlist holds:** beyond the committed tests, names `rerankers.py` imports
      (`CrossEncoder`, `Field`, `ConfigDict`, `PrivateAttr`) and model loading options
      are all refused.
    - **The CI cache** restored the old embedder-only cache by prefix, fetched the
      cross-encoder in the warm-up, and saved under the new key.
    - **Verification 5's re-run** (attempt 2 of run 37180934412) restored that exact key.
      Ingest, warm-up and eval all ran with `HF_HUB_OFFLINE: 1`, the save step was
      skipped, and the aggregates were identical.
  - **Changed:**
    - **A v0.2 config now says how to migrate.** v0.2's own `config.yaml` failed with
      "outside the import allowlist… deliberately not configurable" and no way forward,
      which could read as an invitation to add the prefixes back.
      - A refused `_target_` under `langchain_classic.` or `langchain_community.` now
        adds: delete the old entry, copy the `rerankers` block and the two
        `pipeline.query` lines from `config.yaml`; adding a prefix back is not the fix.
      - It does so at load (`schema.py`) and at import (`registry.py`), via
        `registry.removed_prefix_hint`.
      - Tests cover the v0.2 entry, each dropped target at import time, and other
        refusals, which stay without the note. The real v0.2 file still exits 2, now with
        the note.
    - **"Retrieval takes milliseconds" was stale** (DEC-14 in ARCHITECTURE.md, and a
      comment in `answering.py`): the rerank takes 1.4 s (up to 2 s). The wording is
      corrected. S2-7's scope gains the consequence: a disconnected request's rerank must
      not overlap the next request's. The plan picks whether to hold the slot until the
      thread ends, or to serialize retrieval with a lock.
  - **Not changed here:** PR #11's title lost its `!:` to shell history expansion. The
    commits keep it, and the PRs are rebase-merged; the recipe fixes the title.
