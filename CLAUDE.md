# RAG_QA_System — session bootstrap

Config-driven RAG engine (LangChain / FAISS / Ollama) being brought from
prototype to production via a spec-driven story loop.

## Read these before doing anything

1. `stories/STATUS.md` — the board: current story, open decisions, backlog.
2. The active story file in `stories/phase-N/` — your scope for this session.
3. `docs/ARCHITECTURE.md` (if it exists for the current phase) — design already
   confirmed; follow it, don't re-litigate it.

4. `docs/SRS.md` — full spec with FR/NFR/ISS IDs; consult when a story cites an ID.
5. `docs/WORKFLOW.md` — the process rules; follow steps 3–5 for implementation.

## Hard rules

- Implement ONLY the active story. Out-of-scope findings → story's
  "Discovered" section or STATUS.md Backlog, never a detour.

- A story is done only when every command in its Verification section has
  been run and its output shown. Stop before committing; wait for approval.

- One commit per story; message format: `S0-3: <summary> (ISS-10, FR-2)`.
- Python tooling is **uv only** (`uv add`, `uv run`, `uv sync`) — never
  bare pip, never conda. Interpreter is pinned via `.python-version`.

- `corpus/` is the RAG document corpus: everything in it gets embedded, and
  the text loader reads `.md`. Never put project docs there; never ingest
  repo docs. Project docs live in `docs/`, except `README.md` and
  `CLAUDE.md`, which stay at the repo root. (DEC-4.)

- Never commit binaries, indexes (`vectorstore/`), `__pycache__`, or media.
- Before ending a session: update the story file status + STATUS.md.

## Environment facts (verified 2026-08-01)

- Arch Linux; system Python 3.14.6 — do NOT use it; uv has 3.12/3.11 managed.
- `uv` 0.11.8 at `~/.local/bin/uv`. No conda/pyenv/poetry.
- **No NVIDIA GPU** — CPU-only torch and embeddings; never select
  `device: cuda` components. 15 GB RAM (7B models fit, tightly).

- Ollama daemon running; models pulled: phi3, codellama, mistral.
  `qwen2:7b` (referenced in config) is NOT pulled — see DEC-2 in STATUS.md.

- Fresh resolve of the stack lands LangChain 1.x (`langchain-classic` holds
  the legacy chains) — the code was written against 0.2.x; see DEC-1.

- Docker 29.6.2 installed. `gh` CLI 2.96.0 installed but NOT authenticated
  (`gh auth login` pending — see STATUS.md prerequisites).

## Releases

- Phase exit ⇒ git tag: v0.1 (Phase 0), v0.2, v0.3, v1.0 (Phase 3 / SRS
  complete). Package version tracks the tag; `.dev0` suffix between tags.

## Model routing (which Claude model for which work)

The story file's **Model** field is authoritative per story. The convention:

| Work | Model |
| --- | --- |
| Architecture passes (WORKFLOW Step 1), phase sharding | fable (plan mode) |
| Mechanical stories: hygiene, moves, deps, wiring (S0-1/2/3/6) | opus + /fast |
| LangChain 1.x API surface, config `_target_` migration (S0-4/5) | fable |
| Any story after the obvious fix failed twice | escalate to fable |
| Library/design choices (e.g. Qdrant vs pgvector, Phase 3) | fable |

## GPU policy

- This host is CPU-only. Nothing in Phase 0–3 *requires* a GPU — do not
  block on one. If a job would clearly benefit (bulk re-embedding of a
  grown corpus, long RAGAs eval sweeps), flag it to the user instead of
  running it for hours: they can offload to Google Colab / Kaggle.
- Never propose CPU-hour-scale runs silently; surface the estimate first.
