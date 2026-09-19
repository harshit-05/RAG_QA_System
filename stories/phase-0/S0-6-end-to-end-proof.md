# S0-6: End-to-end proof — ingest the corpus, answer a question

| | |
| --- | --- |
| **Status** | Todo |
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

Transcript: (pasted here at close-out)

## Review notes for the human

Read the three answers yourself — this is the one story where the human
review is about output quality, not code. If the in-corpus answers are
wrong or uncited, that's a real failure even if every command exited 0.

## Discovered

—

## Deviation from plan

—
