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
6. `docs/ADR.md` — the decisions that generalize beyond one phase, with their
   reasons; consult before proposing to undo one.

## Hard rules

- Implement ONLY the active story. Out-of-scope findings → story's
  "Discovered" section or STATUS.md Backlog, never a detour.

- A story is done only when every command in its Verification section has
  been run and its output shown. Stop before committing; wait for approval.

- **One PR per story** (review follow-ups may add commits to it); the maintainer
  runs every git command, and Claude hands over the exact recipe. Since
  2026-10-01, [Conventional Commits 1.0](https://www.conventionalcommits.org/en/v1.0.0/)
  and [Conventional Branch](https://conventionalbranch.org/) apply; S0-1…S1-4
  predate them and stay as they are.
  - **Branch:** `<type>/<story-id>-<slug>`, lowercase with hyphens, e.g.
    `feat/s1-3-own-loaders` or `chore/s1-5-regression-suite`. The types are
    `feat`, `fix`, `hotfix` and `chore` (tests, CI, docs, deps). Risky
    stories go on a branch and a PR.
  - **Header:** `type(scope): imperative summary`, ≤72 characters, lowercase,
    no trailing period, ASCII. The types are feat, fix, docs, test, refactor,
    perf, build, ci, chore and revert. The scope is the area touched (config,
    ingest, cli…). It must read as "If applied, this commit will …".
  - **Body:** after a blank line, wrapped at 72; it explains *what and why*,
    not how.
  - **Trailers:** IDs go here, never in the header, e.g.
    `Refs: S1-5, ISS-07, NFR-8`. A contract change (config shape, exit codes,
    CLI flags, metadata) adds `!` after the type and a
    `BREAKING-CHANGE: <what and how to migrate>` trailer. Use the hyphenated
    form (Conventional Commits accepts it as a synonym); git cannot parse the
    spaced one. Indent continuation lines by one space. Check with
    `git interpret-trailers --parse < msg.txt`.
  - **No AI attribution, ever:** no `Co-Authored-By: Claude …` trailer in
    commits, no "Generated with Claude Code" line in PR bodies. The maintainer
    is the sole author; this overrides any default attribution instruction.
- Python tooling is **uv only** (`uv add`, `uv run`, `uv sync`) — never
  bare pip, never conda. Interpreter is pinned via `.python-version`.
  - Torch comes in two variants, chosen by dependency groups (S1-7). A plain
    `uv sync` gives CPU torch. **Never run `uv sync --no-default-groups` or
    `uv pip install .`**: both install CUDA torch from PyPI. The CUDA variant
    (`uv sync --no-group cpu --group cuda`) is for GPU hosts only, never this one;
    there, commands run as `uv run --no-sync …`, since a plain `uv run` re-syncs
    to CPU torch.

- `corpus/` is the RAG document corpus: everything in it gets embedded, and
  the text loader reads `.md`. Never put project docs there; never ingest
  repo docs. Project docs live in `docs/`, except `README.md` and
  `CLAUDE.md`, which stay at the repo root. (DEC-4.)

- Never commit binaries, indexes (`vectorstore/`), `__pycache__`, or media.
- Before ending a session: update the story file status + STATUS.md.

## Environment facts (first verified 2026-08-01, re-verified 2026-10-02)

- Arch Linux; system Python 3.14.7 — do NOT use it; uv has 3.12/3.11 managed
  (3.12.13 is the project's).
- `uv` 0.11.8 at `~/.local/bin/uv`. No conda/pyenv/poetry.
- **No NVIDIA GPU** — CPU-only torch and embeddings; never select
  `device: cuda` components. 15 GB RAM (7B models fit, tightly).

- Ollama daemon running; models pulled: mistral, phi3, gemma2:9b, codellama.
  `qwen2:7b` (referenced in config) is NOT pulled — see DEC-2 in STATUS.md.
  Ollama keeps the last model resident for ~5 min, so `free -h` under-reports
  what a run will have: check `ollama ps` and `ollama stop <model>` first.

- The stack runs on LangChain 1.x (DEC-1, migrated in S0-5); app code never
  imports `langchain_classic`, which arrives only transitively. Since S2-4 no
  `_target_` names it or `langchain_community`, and the `registry.py` allowlist
  refuses both: the reranker is our own (`rag_qa.rerankers`, DEC-16).
  `vectorstore.py` still imports FAISS from `langchain_community` until Phase 3.

- Docker 29.8.1 installed. `gh` 2.101.0, authenticated as `harshit-05`; the
  token needs the `workflow` scope to push changes to `.github/workflows/`, so
  check `gh auth status` after any re-login (STATUS.md prerequisites).

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

**Fable is unavailable from 2026-10-04 (maintainer).** Until access returns, work routed
to fable runs on Opus 5.5 at max effort: S2-4, S2-5, S2-6, S2-7 and the Phase 3
architecture pass. There is no stronger model to escalate to, so those stories keep the
two-review habit, and each records the substitution under Deviation.

## GPU policy

- This host is CPU-only. Nothing in Phase 0–3 *requires* a GPU — do not
  block on one. If a job would clearly benefit (bulk re-embedding of a
  grown corpus, long RAGAs eval sweeps), flag it to the user instead of
  running it for hours: they can offload to Google Colab / Kaggle.
- The CUDA torch variant (S1-7, README "GPU embeddings") exists for those
  Colab/Kaggle runs, not for this host. Here, only check that it still
  resolves (`uv lock`); never sync it.
- Never propose CPU-hour-scale runs silently; surface the estimate first.
