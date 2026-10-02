# S2-9: Phase 2 exit — docs, end-to-end proof, release 0.3.0

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | — (phase exit; SRS §12 Phase 2) |
| **Depends on** | S2-1 … S2-8, S2-5b |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `chore/s2-9-phase-2-exit`. A release: it lands through a PR, and `main` is tagged after the merge. |

## Goal

Close Phase 2 by showing its exit criterion holds, bringing the documentation in line with
the code, and cutting the release. Exit criterion (SRS §12): **the system is callable over
HTTP and has an enforced quality bar.**

- The first half is a manual end-to-end run over HTTP against real Ollama, on a fresh
  clone.
- The second half is the two eval gates, green on `main`.

Exit ⇒ version 0.3.0, tag `v0.3`.

## Scope

- **The end-to-end run**, on a fresh clone and real Ollama. Free memory first
  (caveat 8). The steps:
  1. `uv sync`;
  2. `rag-ingest`, a full run (`vectorstore/db_faiss` becomes a symlink to a
     generation);
  3. a second `rag-ingest`: nothing to embed;
  4. `rag-serve`;
  5. a query streamed over SSE with `curl -N`;
  6. an ingest job, then the documents list;
  7. a disconnect during the prefill and one mid-stream. Each closes the server's
     connection to Ollama within about 1 s, the runner idles by the end of the prefill
     at the latest (caveat 21), and the next query gets a slot;
  8. `rag-query`, including a Ctrl-C mid-answer.

  Record a table of the checks, as S1-8's manual test did.
- **Quality bar.**
  - `rag-eval check` is green on the release commit, and `eval-retrieval` is green in CI.
  - Re-run tier 2 only if its fingerprint moved after S2-5b. Surface the estimate first
    (caveat 16).
- **Docs.**
  - `README.md`: the HTTP API (`rag-serve`, the token, the endpoints, curl); evaluation
    (`rag-eval`, the two tiers, the Colab/Kaggle scoring recipe); incremental ingestion
    and `--rebuild`; the one-time re-ingest for v0.2 users.
  - `docs/ARCHITECTURE.md` §2.8, "As delivered": every place the code differs from
    §2.1–§2.7, with the story that records why. As in §1.8, earlier sections are
    annotated, not rewritten.
  - `docs/ADR.md`: entries for the decisions that generalize. They are indexed and not
    renumbered. Candidates:
    - cancelling an async stream means owning the task, not going async;
    - a two-tier quality gate, where the expensive tier is committed with a fingerprint
      whose parts name the re-run that fixes them;
    - publishing an index by an atomic symlink flip between generations;
    - versioning a local judge's settings as a Modelfile, because its API cannot set
      them per request;
    - an embedder identity that leaves out the keys that do not change the vectors.
  - `CLAUDE.md`: environment facts (the new commands; `gemma2:9b` as the judge); check
    the `langchain_classic` line S2-4 rewrote; anything a fresh session would get wrong.
  - `stories/STATUS.md`:
    - Phase 2 moves to Done;
    - the caveats learned on the way are added;
    - the backlog is reconciled, so every Phase 2 line is closed or re-deferred to
      Phase 3 with a reason;
    - the "Now" block points at the Phase 3 architecture pass.
- **Release.** `0.3.0.dev0` → `0.3.0`, in both `pyproject.toml` and
  `src/rag_qa/__init__.py`. Once the PR is merged, tag `main` with `git tag -a v0.3`, and
  push it. The maintainer runs every git command, from a recipe Claude provides.

## Out of scope

- Any Phase 3 work: Qdrant, Docker, OTel, rate limiting, the hosted fallback.
- The DEC-2 revisit (backlog).
- Rewriting the Phase 0 or Phase 1 sections of ARCHITECTURE.md.

## Verification

```bash
# 1. every gate, locally and in CI
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q
uv run rag-eval check; echo "exit $?"                     # → 0
gh run list --limit 3                                     # check and eval-retrieval green on the PR, then on main

# 2. the exit criterion, first half: a fresh clone over HTTP (S1-8's note applies: build
#    the "clone" from `git ls-files` with working-tree contents if the PR is not merged yet)
git clone . <scratch>/fresh && cd <scratch>/fresh && uv sync
uv pip list | grep -icE 'nvidia'                          # → 0
uv run rag-ingest && uv run rag-ingest                    # full, then nothing to embed
#    rag-serve runs in a second terminal, in the foreground, with its log tee'd:
#      cd <scratch>/fresh && uv run rag-serve 2>&1 | tee <scratch>/serve.log
curl -s localhost:8000/v1/health
curl -sN -X POST localhost:8000/v1/query -H 'content-type: application/json' -d '{"question": "What is GLIDER?"}'
#    plus the remaining checks in the Scope table, recorded below

# 3. docs tell the truth
for f in docs/*.md stories/*.md stories/phase-*/*.md CLAUDE.md README.md; do
  grep -o '](\([^)#]*\.md\)' "$f" | sed 's/^](//' | while read -r l; do
    test -e "$(dirname "$f")/$l" || echo "DANGLING in $f: $l"; done; done    # → prints nothing

# 4. release
grep '^version' pyproject.toml; grep __version__ src/rag_qa/__init__.py   # → 0.3.0
git tag -l | grep v0.3                                    # after the maintainer tags
```

## Review notes for the human

- **The fresh-clone run in step 2 matters most.** Phase 2 changed the index format, the
  query path and the dependencies, so the earlier exit criteria are the likeliest to have
  regressed silently: "a fresh clone answers a query" (v0.1) and "CI catches every
  Phase-0 bug" (v0.2).
- **Then read CLAUDE.md as a fresh session would.** It is loaded into every future
  session.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
