# S0-6: End-to-end proof — ingest the corpus, answer a question

| | |
| --- | --- |
| **Status** | Done (2026-09-25) — commit and `v0.1` tag pending, maintainer commits manually |
| **Closes** | — (Phase 0 exit criterion, SRS §12) |
| **Depends on** | S0-5 (DEC-2 resolved 2026-09-18: `mistral`, `phi3` fallback) |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

The Phase 0 exit criterion, demonstrated and recorded: from the current
tree, ingestion builds a fresh index from `docs/` and a real question gets
a grounded, cited answer from the configured Ollama model. Not "it should
work" — a transcript.

## Scope

- Memory pre-check: `free -h`. With ≥ ~6 GB available use `mistral` (the
  DEC-2 choice, already pulled); below that, switch `pipeline.query.llm` to
  a `phi3_ollama` entry for this run (2.2 GB) and record which model
  produced the transcript. `ollama list` confirms the model is present;
  `validate_model_on_init: true` will fail fast otherwise.

- Delete the stale index dir `vectorstore/db_faiss/` (already untracked
  since S0-1; it was pickled under LangChain 0.2 and will not load); run
  `uv run rag-ingest` against `docs/`; record document/chunk counts. First
  run downloads the ~90 MB MiniLM embedder; expect minutes on CPU.

- Latency expectations (CPU, 7B): 15–30 s to first token, 1–2 min per full
  answer. Slow is not broken. If an answer looks ungrounded, check `num_ctx`
  and `k` in config before anything else (STATUS.md caveat 7).

- Run `uv run rag-query`, ask 2 questions with known answers in the corpus
  (e.g. from the YOLOv8 surveillance paper) and 1 question NOT in the
  corpus (must refuse per the prompt contract, FR-5).

- Paste the actual transcript (answers + cited sources) into this story
  file under Verification.

- Update `README.md` with the real quickstart: uv sync → ollama serve +
  model → rag-ingest → rag-query (closes the worst of ISS-20; full docs
  remain Phase 1).

- **Release v0.1** (this story closes Phase 0): bump `pyproject.toml`
  version to `0.1.0`, commit, `git tag v0.1`, and update the release
  mapping row in STATUS.md to Done. (Push + tag push once `gh auth login`
  is done — see STATUS.md prerequisites.)

## Out of scope

- Performance measurement (NFR-1/2 — needs instrumentation, Phase 3).
- Evaluation harness (Phase 2).

## Verification

```bash
uv run rag-ingest                       # completes; prints doc + chunk counts
uv run rag-query                        # interactive: 2 in-corpus Qs answered with sources; 1 out-of-corpus Q refused
git status --short                      # vectorstore/ untracked (ignored), only intended changes
```

### Ingestion (2026-09-25) — done

Stale 0.2-era index (July 2025 pickle, untracked) deleted, then `uv run
rag-ingest` on the real corpus, foreground, exit 0:

| Measure | Value |
| --- | --- |
| Pages loaded | 561 (32 + 516 + 13 across the three PDFs) |
| Chunks embedded | 1,708 |
| Failed / skipped | 0 / 2 (the parquet files, by design) |
| Embedding | 54 batches in 77 s, about 22 chunks/s on CPU |
| Wall time | 125 s |

These are the project's first throughput numbers (NFR-3). Extrapolated, a
full rebuild of a corpus ten times this size takes about 20 minutes and a
hundred times takes hours — the point where Phase 2's incremental manifest
(ISS-14) or a GPU offload starts to matter.

### Retrieval on the real index — done

The index reloads under LangChain 1.x: 1,708 vectors, dimension 384 (MiniLM).
The corpus is lopsided — the proceedings hold 88% of chunks, the GLIDER paper
9%, the YOLOv8 paper 3% — so the meaningful tests are the two small papers.

| Question | Top-5 from expected doc | Best distance |
| --- | --- | --- |
| YOLOv8 crowd/threat detection | 5/5, the 3% paper | 0.548 |
| How GLIDER grades LLM outputs | 5/5, the 9% paper | 0.858 |
| Resolution in theorem proving | 5/5, the proceedings | 0.521 |
| Capital of Australia (not in corpus) | noise, all proceedings | 1.645 |

Lower distance is closer. Two findings:

- **`page_label` citations proved their worth**: one proceedings hit cites
  "p. v", a roman-numeral front-matter page. Raw `page + 1` would have shown a
  wrong arabic number there.
- **A relevance cutoff looks viable.** Every in-corpus hit fell at or below
  1.004; the out-of-corpus question's best hit was 1.645. A threshold in that
  gap would stop the CLI listing "sources" for a refusal (S0-5 backlog item).
  Four questions are not a calibration; the Phase 2 eval harness should set
  the actual value.

### CLI transcript (2026-09-25, model: `mistral`, run by the maintainer)

Run as `uv run rag-query` with 6.6 GB available. Pasted as produced:

```text
Question: How does the YOLOv8 system detect crowds and threats?

Answer:
 The YOLOv8 system detects crowds and threats through object detection and anomaly
recognition. It is trained to identify people in high-resolution video streams, count
them, and track their movement across frames. The system can also detect behaviors that
deviate from predefined norms, which may indicate potential threats. Additionally, the
YOLOv8 model is equipped with an anomaly detection framework that identifies suspicious
activities with 92.7% accuracy, reducing false positives through context-aware filtering.
This helps in maintaining safety compliance in workplaces, managing crowds in
high-density environments, and facilitating timely interventions by generating instant
alerts when abnormal or suspicious activity is detected.

Sources:
  [1] Batch22_SmartSurveillanceSystemsUsingYOLOv8AScalableApproachforCrowdandThreatDetection.pdf, p. 10
  [2] Batch22_SmartSurveillanceSystemsUsingYOLOv8AScalableApproachforCrowdandThreatDetection.pdf, p. 3
  [3] Batch22_SmartSurveillanceSystemsUsingYOLOv8AScalableApproachforCrowdandThreatDetection.pdf, p. 9
  [4] Batch22_SmartSurveillanceSystemsUsingYOLOv8AScalableApproachforCrowdandThreatDetection.pdf, p. 3
  [5] Batch22_SmartSurveillanceSystemsUsingYOLOv8AScalableApproachforCrowdandThreatDetection.pdf, p. 1

Question: What is GLIDER and what does it evaluate?

Answer:
 GLIDER is a small, explainable, and performant SLM-as-judge model that has been
fine-tuned and aligned from a phi-3.5-mini model. It is designed to evaluate models in
the context of Software Language Model (SLM) evaluations. The evaluation criteria for
GLIDER are multi-metric, as shown by its performance on the LiveBench dataset, where it
outperforms other models with F1 scores of 0.654, 0.481, and 0.485 respectively compared
to GPT-4o-mini, Qwen-2.5-72B, and other evaluation criteria provided in the LiveBench
dataset. The dataset used for evaluating GLIDER contains questions related to coding,
problem solving, and more essential for a holistic evaluation of the model.

Sources:
  [1] 2412.14140v2.pdf, p. 7
  [2] 2412.14140v2.pdf, p. 5
  [3] 2412.14140v2.pdf, p. 4
  [4] 2412.14140v2.pdf, p. 8
  [5] 2412.14140v2.pdf, p. 9

Question: What is the capital of Australia?

Answer:
 I could not find the answer to your question in the provided documents as they do not
contain information about the capital of Australia.

Sources:
  [1] 7thInternationalConferenceonAutomatedDeduction.pdf, p. 163
  [2] 7thInternationalConferenceonAutomatedDeduction.pdf, p. 195
  [3] 7thInternationalConferenceonAutomatedDeduction.pdf, p. 161
  [4] 7thInternationalConferenceonAutomatedDeduction.pdf, p. 93
  [5] 7thInternationalConferenceonAutomatedDeduction.pdf, p. 182
```

### Fact-check against the source PDFs

Every claim was checked against the PDF text, and the failures were then
traced to see whether retrieval or generation caused them.

| Question | Verdict | Detail |
| --- | --- | --- |
| YOLOv8 | **Correct** | Every claim is in the paper, several nearly verbatim: 92.7% anomaly-recognition accuracy, context-aware filtering, tracking across frames, deviation from predefined norms, instant alerts. Key facts on p. 3 and p. 9, both cited. |
| GLIDER | **Two errors** | Base model phi-3.5-mini and the LiveBench study are correct. But "Software Language Model" is invented — the paper defines SLM as **Small** Language Model (p. 2). And the scores are misattributed: the paper gives GLIDER 0.654, GPT-4o-mini 0.481, Qwen-2.5-72B 0.485 (p. 7); the answer maps the three numbers onto the wrong models. |
| Capital of Australia | **Correct refusal, paraphrased** | Did not leak "Canberra". Reworded the fixed refusal sentence rather than using it verbatim (phi3 used it exactly in S0-5). |

**Where the GLIDER answer went wrong — generation, not retrieval:**

- The top-ranked chunk (p. 7) contained the paper's score sentence **intact**,
  word for word. The model had the right answer and scrambled the attribution.
- **None** of the five retrieved chunks defines "SLM". Instead of omitting it,
  the model invented an expansion, despite "do not use any prior knowledge".
  The grounding rule held for a fully out-of-corpus question, but not for a gap
  _inside_ an answerable one.
- A hypothesis that the paper's flattened results table fed the model number
  soup was checked and ruled out: the retrieved table chunk had 2% digit
  density.

Both are faithfulness failures, the property the Phase 2 RAGAs harness
measures. This question, with its known ground truth, is the obvious first
entry for the golden dataset.

### Exit criterion, literally: a fresh clone (2026-09-25)

SRS §12's Phase 0 exit reads "a fresh clone runs ingestion and answers a
query end to end". Every earlier run used the working copy, which could hide a
dependence on uncommitted or ignored files. So the repo was cloned at
`55344bd` into the scratchpad and run from nothing:

| Step | Result |
| --- | --- |
| Clone | no `.venv`, no index — confirmed absent |
| `uv sync` | 3 s, all wheels from uv's cache |
| `uv run rag-ingest` | exit 0; 561 pages, **1,708 chunks**, 0 skipped, 0 failed; 128 s |
| `uv run rag-query`, one question, `mistral` | exit 0; 299 s for the session |

- **Same 1,708 chunks** as the working copy's independent build — sorted file
  order makes ingestion deterministic.
- **0 skipped** because the ignored parquet files never reach a clone,
  confirming S0-1's cleanup from the outside.
- **The answer was word for word identical** to the maintainer's run, with
  the same five sources. Temperature 0 plus deterministic ingestion makes the
  pipeline reproducible, which Phase 2's evaluation baselines depend on.
- **Latency, measured:** about five minutes for the first answer of a
  `mistral` session, cold model load included. The board's pre-flight
  estimate of 5–15 s to first token and 1–2 minutes per answer was optimistic
  for a cold start; the README quotes the measurement.

The clone was at the last _committed_ state, so it ran without this story's
uncommitted changes (README, version bump, ingestion progress output). None
of those affect the pipeline's behaviour.

**Phase 0 exit criterion: met.**

## Review notes for the human

Read the three answers yourself — this is the one story where the human
review is about output quality, not code. If the in-corpus answers are
wrong or uncited, that's a real failure even if every command exited 0.

## Discovered

- **Faithfulness is the real quality gap**, not retrieval. Retrieval chose the
  right document 5/5 for every question, including a paper holding 3% of the
  chunks. The errors came from the model misreading intact text and filling a
  gap with invention. → Backlog: seed the Phase 2 golden dataset with the
  GLIDER question and its verified ground truth.
- **Refusals are not reliably verbatim.** mistral paraphrased the fixed
  refusal sentence; phi3 used it exactly. Anything that detects a refusal by
  exact string match will be flaky. This also argues for suppressing "Sources"
  by retrieval distance (S0-6 retrieval check found a clear gap) rather than by
  matching the refusal text. → Backlog, Phase 2.
- **Streamed answers start with a stray space** (" The YOLOv8…"): mistral's
  first token carries a leading space. Cosmetic; the CLI could strip leading
  whitespace from the first answer chunk. → Backlog, Phase 1.
- **The old README misdescribed the project**: image support, an API,
  Streamlit/Gradio and `pip install` from the deleted `v1/`. Rewritten from what
  actually works, with today's measured numbers and an honest limitations
  section. It states "MIT License", as the old README did, but **the repo has
  no `LICENSE` file**, so on GitHub the code is legally all-rights-reserved
  until one is added. Left for the maintainer to decide.

## Deviation from plan

- **One small code change, at the maintainer's request to watch progress.**
  `ingest.py` now prints each file's page count as it loads, and switches on
  the embedder's own progress bar for the ingestion instance only. Without it,
  embedding (the longest step) printed nothing for over a minute. The query
  path builds a separate embedder instance, so `rag-query` output is unchanged,
  and the config is never modified (ADR-022's rule). Embedders without a
  progress switch run as before.
