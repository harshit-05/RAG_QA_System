# RAG_QA_System — session bootstrap

Config-driven RAG engine (LangChain / Qdrant / Ollama) being brought from
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
  - Python is ruff-formatted, and CI's Format step gates it (S3-1, DEC-28): run
    `uv run ruff format` before a commit, bare, without file names. Markdown is
    excluded only while ruff walks the tree: a `.md` file named on the command line
    gets its Python examples rewritten.
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

## Environment facts (first verified 2026-08-01, re-verified 2026-10-13)

- Arch Linux; system Python 3.14.7 — do NOT use it; uv has 3.12/3.11 managed
  (3.12.13 is the project's).
- `uv` 0.11.8 at `~/.local/bin/uv`. No conda/pyenv/poetry.
- **No NVIDIA GPU** — CPU-only torch and embeddings; never select
  `device: cuda` components. 15 GB RAM (7B models fit, tightly).

- Ollama 0.22.1 daemon running; models pulled: mistral, phi3, gemma2:9b, codellama,
  plus `rag-judge`, tier 2's judge, built from `eval/judge.Modelfile` on gemma2:9b
  (8k context). `qwen2:7b` (referenced in config) is NOT pulled — see DEC-2 in
  STATUS.md. Every Ollama llm in config sets `num_thread: 10`. Ollama keeps the last
  model resident for ~5 min, so `free -h` under-reports what a run will have: check
  `ollama ps` and `ollama stop <model>` first. The 5 min count from the end of the
  last request, so after a dropped prefill the model can unload while you wait, and
  the next question pays a reload and a full prefill (S2-9).

- **The hosted fallback (S3-6)** is off unless `pipeline.query.fallback_llm` names an entry.
  It needs `uv sync --extra fallback`, which a plain `uv sync` removes again, and its key in
  `RAG_FALLBACK_API_KEY`, read from the environment only (a shell may load a gitignored
  `.env` with `set -a; . ./.env; set +a`). Never write the key to a file, never print it.
  Without the extra, as in CI, `tests/test_fallback_openai.py` is skipped.

- Commands: `rag-ingest`, `rag-query`, `rag-serve` (the HTTP API), `rag-keys` (its API
  keys, in `var/api-keys.json`, since S3-8) and `rag-eval`
  (`retrieval` for tier 1; `generate`, `score` and `check` for tier 2). Servers and
  model runs go in the maintainer's foreground terminal, logged with `tee -i`: a
  plain `tee` dies with Ctrl-C and loses the shutdown lines. Since S3-7 every log goes to
  stderr, as JSON when it is a pipe, so prefix such a run with `RAG_LOG_FORMAT=console`
  (caveat 39).

- **Tier 2 is committed and fingerprinted.** An edit to the prompt, an llm, retriever
  or reranker spec, the corpus, the embedder, splitter or loaders, or a golden
  question fails CI's `rag-eval check` until tier 2 is re-run: about 70 min of
  `generate` here, then about 35 min of `score` on a Colab T4. Budget it before such
  an edit (STATUS.md caveat 24; README, "Evaluation").

- The stack runs on LangChain 1.x (DEC-1, migrated in S0-5); app code never
  imports `langchain_classic`. Since S2-4 no `_target_` names it or
  `langchain_community`, and the `registry.py` allowlist refuses both: the reranker is
  our own (`rag_qa.rerankers`, DEC-16). Since S3-3 neither package nor `faiss-cpu` is a
  dependency or in a plain `uv sync`; only the `eval` extra installs the two LangChain
  ones, through ragas, and `tests/test_architecture.py` keeps them out of `src`.

- **The index lives in Qdrant** (DEC-19, since S3-2): the `qdrant` service of
  `compose.yaml`, `qdrant/qdrant:v1.19.2-unprivileged` pinned by digest, on
  127.0.0.1:6333 with its data in the `qdrant-storage` volume. Start it with
  `docker compose up -d qdrant` before any `rag-ingest`, `rag-query`, `rag-serve` or
  `rag-eval retrieval|generate`: each exits 2 without it. The alias is `rag_qa`;
  a scratch index is `RAG_QDRANT_COLLECTION=<name>`, and `RAG_VECTOR_STORE_PATH` is
  refused (caveat 9). The writer lock is `var/ingest.lock`, and it excludes only writers
  sharing this `var/`: another checkout on this host (a fresh-clone e2e) gets
  `RAG_QDRANT_COLLECTION=<name>` of its own, or the two can delete each other's new
  generation (caveat 36). The unit suite needs no server;
  `RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server` runs the
  server tests. v0.3's `vectorstore/` folder is no longer read.

- Docker 29.9.0 installed (re-checked 2026-10-16). `gh` 2.101.0, authenticated as
  `harshit-05`; the token needs the `workflow` scope to push changes to
  `.github/workflows/`, so check `gh auth status` after any re-login (STATUS.md
  prerequisites).

## Releases

- Phase exit ⇒ git tag: v0.1 (Phase 0), v0.2, v0.3, v1.0 (Phase 3 / SRS
  complete). Package version tracks the tag; `.dev0` suffix between tags.
- Current: `1.0.0.dev0`, since S3-1 (2026-10-16). The last tag is `v0.3`, 0.3.0 on
  `aa8c19b` (Phase 2, 2026-10-14). S3-12 releases 1.0.0 and tags `v1.0`, setting the
  version in `pyproject.toml` and `src/rag_qa/__init__.py` (and `uv lock`).

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
to fable runs on Opus 5.5 at max effort: S2-4 to S2-7 did, and so did the Phase 3
architecture pass (2026-10-15); Phase 3's fable-routed stories will too. There is no
stronger model to escalate to, so that work keeps the two-review habit, and each story
records the substitution under Deviation.

## GPU policy

- This host is CPU-only. Nothing in Phase 0–3 *requires* a GPU — do not
  block on one. If a job would clearly benefit (bulk re-embedding of a
  grown corpus, long RAGAs eval sweeps), flag it to the user instead of
  running it for hours: they can offload to Google Colab / Kaggle.
- The CUDA torch variant (S1-7, README "GPU embeddings") exists for those
  Colab/Kaggle runs, not for this host. Here, only check that it still
  resolves (`uv lock`); never sync it.
- Never propose CPU-hour-scale runs silently; surface the estimate first.
