# S2-9: Phase 2 exit — docs, end-to-end proof, release 0.3.0

| | |
| --- | --- |
| **Status** | Done 2026-10-13, reviewed 2026-10-14; tag `v0.3` on `main` after the merge |
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
    and `--rebuild`; the one-time re-ingest for v0.2 users; and the v0.2 config migration
    (S2-4): a config that still has the old `components.rerankers.cross_encoder` entry no
    longer loads, so delete it, and the reranker is `ms_marco_minilm_cpu`.
  - `docs/ARCHITECTURE.md` §2.8, "As delivered": every place the code differs from
    §2.1–§2.7, with the story that records why. As in §1.8, earlier sections are
    annotated, not rewritten. Two annotations S2-4 owes:
    - §1.1's DEC-7 paragraph lists `langchain_community.` and `langchain_classic.` as
      allowed prefixes and calls the reranker entry disabled. Both prefixes left the
      allowlist in S2-4.
    - ADR-007's "Phase 2 may use a narrow named exception for `langchain_classic`'s
      `CrossEncoderReranker`" is moot: DEC-16 hand-rolled the reranker, so no exception
      was used.
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

## End-to-end run (2026-10-13)

The fresh clone was `~/dev/rag-s2-9-e2e/fresh`, on disk rather than in the RAM-backed
`/tmp`: `git clone .` of `main` at `2d12b34`, plus the three files of the version bump,
so it ran 0.3.0. The server fix was copied in later for row 9's re-check. It changes only
what `main()` does after uvicorn has shut down, so rows 1–8 stand. The real index was
never touched, and no `RAG_*` variable was set. mistral came from the shipped config,
with 9.5–10 GB available before the server started.

**How the drops were timed** (S2-7's method). A helper outside the repo,
`~/dev/rag-s2-9-e2e/e2e.py`, did the timing:

- **Our connection:** it polled `ss` every 20 ms for the connection carrying the request
  to Ollama, told apart from idle pooled connections by its byte counters. Each poll is
  stamped before `ss` runs, so a close reads as "within one poll" (a few hundredths of a
  second), and once as −0.03 s.
- **The runner:** it read the runner's CPU from `/proc` every 0.5 s.
- **The CLI:** it drove `rag-query` in a pty, where `0x03` passes the tty's line
  discipline as a keyboard's Ctrl-C does.
- **Ollama's journal** gave each request's status and duration.

| # | Check | Result |
| --- | --- | --- |
| 1 | `uv sync` | Pass: 1 s from uv's cache. 0 `nvidia-*` packages, torch `2.14.0+cpu`, `rag-qa-system 0.3.0` |
| 2 | `rag-ingest`, a full run | Pass: 3 documents, 561 pages, 1,708 chunks (the v0.1 count), exit 0, 1:59. `vectorstore/db_faiss` is a relative link to `db_faiss.gen-20261013T065022.700200Z-3b4acf8d`, with the lock beside it |
| 3 | `rag-ingest` again | Pass: "Nothing changed: the index is up to date.", "Documents: 3 unchanged.", exit 0, 0.6 s |
| 4 | `rag-serve`, then both health probes | Pass: it served that generation, 3 documents. `/v1/health` 200, ready (index serving, the llm answered); `/v1/health/live` 200 |
| 5 | `curl -N` "What is GLIDER?" | Pass: `sources` (5 chunks of the GLIDER paper, reranked), 10 `: ping`s through the prefill, 91 `token`s, then `done` with `ttft_ms` 154,613 and `total_ms` 173,499; curl exit 0. The answer is grounded (the paper's abstract) |
| 6 | Ingest jobs, then the documents list | Pass. A note added to the clone's corpus: 202, succeeded, `exit_code` 0, 7.7 s, 1 added and 3 unchanged, and the new generation served (4 documents, 1,709 chunks). Removed: 0.5 s, 1 removed, a new generation (3, 1,708). Nothing changed: 0.5 s, nothing published. `/v1/documents` listed the manifest each time, and no host path appeared in any body |
| 7a | Drop during the prefill, at 20 s (after the first `: ping`) | Pass. Our connection closed 0.01 s after curl left, and Ollama logged `/api/chat` 500 at that moment. The runner ran to the end of its prefill and idled at 153 s from the request (mistral had unloaded, so a reload came first). The next query got 200, a slot, and never reached Ollama. The server logged "Answer cancelled: its client disconnected" for both |
| 7b | Drop mid-stream, after 5 tokens | Pass. Closed 0.03 s after curl left; `/api/chat` 200 at that moment; the runner idle 0.68 s after the drop; the next query got 200. mistral had unloaded again, so this paid a full 151 s prefill (caveat 25) |
| 8 | `rag-query`: a full answer, a Ctrl-C mid-answer, then `exit` | Pass. "Answer:" came 153.3 s after Enter (reload and prefill), with 5 YOLOv8 sources, and the prompt was back at 172.2 s. The repeat hit Ollama's prompt cache ("Answer:" at 1.3 s). Ctrl-C 67 bytes in printed "(answer interrupted; Ctrl-C again at the prompt quits)", and the prompt came back at once. The stream closed within one poll, `/api/chat` 200 after 2.9 s, and the runner idled 1.0 s after the Ctrl-C. `exit` gave exit 0, and the pooled connection closed at exit |
| 9 | Ctrl-C on `rag-serve` (maintainer's terminal, `tee -i`) | **Failed, then fixed.** "Application shutdown complete" and "Finished server process", then a `KeyboardInterrupt` traceback; reproduced with exit status 130. After the fix, through `.venv/bin/rag-serve` with the real models and Ollama: exit 0, no traceback |

## Discovered

- **`rag-serve`'s Ctrl-C ended in a traceback and exit 130** (row 9), where its docstring
  and `--help` promise 0.
  - **The mechanism:** uvicorn's `capture_signals` re-raises the captured SIGINT after
    its graceful shutdown. The handler it restores is asyncio's runner's, which cancels
    the finished main task and raises `KeyboardInterrupt` out of `server.run()`.
    `main()` caught only `SystemExit`.
  - **Why nobody saw it:** it dates from S2-7, whose log was tee'd without `-i`, so the
    shutdown lines were lost.
  - **The fix (maintainer's call: in this PR):** `main()` catches it and returns the
    existing status, 0 once the server has started.
  - **The tests, red first, then green:** one sends a real SIGINT to `main()` in a child
    process, with a stub app and no preflight, so it stays hermetic. The other covers the
    branch in-process.
  - **SIGTERM already behaved:** the same graceful shutdown, then status 143 from the
    re-raised signal, with no traceback. The docstring now says so, and the backlog's
    Docker line has it.
- **Ollama's keep-alive counts from the end of the last request** (caveat 25). After 7a,
  Ollama logged the request finished at the drop, so mistral unloaded about 2.5 min after
  the runner went idle. Rows 7b and 8 each paid a reload and a full prefill. Row 8's
  tool call also waited about 5 min for approval, which used up the window.
- **While the runner holds the CPU, the API's retrieval is about 9× slower** (caveat 26).
  7a's slot probe got its 200 after 7.0 s. On an idle CPU (7b), the status line and
  `sources` both came at 0.78 s. So the status line waits for the first event, and the
  7 s was retrieval competing with the runner's 10 threads. It is not a slot problem: the
  probe got 200, not 503.
- **Ollama's prompt cache makes a repeated question cheap**, within keep-alive: row 8's
  repeat reached its first token in 1.3 s. That is the cheap way to reach "mid-stream" in
  a manual check, as S2-7 found.
- **The package docstring was stale:** it listed `evaluate`, deleted in S2-5, and lacked
  `rerankers`. Fixed with the version bump, as v0.2's release fixed its own.
- **README gaps beyond the story's list:**
  - the project layout lacked `rerankers.py`, `eval/` and `scripts/`;
  - the latency paragraph quoted v0.1's five minutes;
  - Development named neither tier 2's step nor `eval-retrieval`.
- **"Embedding 1 chunks..."** appeared in an API job's log: the known plural slip
  (backlog).
- **The branch did not exist at the start.** HEAD was `main` at `2d12b34`; the recipe
  creates `chore/s2-9-phase-2-exit` from it, and the uncommitted work carries over.

## Deviation from plan

- **Model:** Opus 5.5, where the story says opus-fast.
- **Code in a release story.** Row 9's fix is a `fix(api)` commit with its tests, by the
  maintainer's choice over a separate PR or a backlog line. It restores the documented
  exit code, so it carries no `!`.
- **Order:**
  - the version bump came first, so the clone ran 0.3.0;
  - the run came before the docs (the maintainer's instruction);
  - `rag-query` (row 8) ran while `rag-serve` was still up (4.9 GB available), so that
    the server's Ctrl-C check came last.
- **Row 6 ran three jobs, not one.** Adding and removing a note in the clone's corpus
  made the jobs do real work and switch generations.
- **Verification 1's `gh run list`** shows `main`'s runs, green on `2d12b34` (run
  38349373369). The PR's run, `main`'s after the merge and the tag's come later, and the
  close-out commit records them.
- **Beyond the scope's doc list:**
  - §2.2's layout was corrected in place: six `api` modules, `checkpoint.py`, and the
    `chain` and `components` lines;
  - caveat 10 gained S2-7's note that `uv sync` is exact;
  - the backlog's Docker line gained the SIGTERM status.

## Verification record (2026-10-13)

1. **The gates.**
   - `ruff check`: all checks passed. `mypy`: no issues in 30 source files.
   - `pytest --cov=rag_qa`: **890 passed, 5 xfailed**, coverage 98.37%. `server.py` is
     covered except its `if __name__` line.
   - **Caveat 23:** the same suite under `HF_HOME=<empty dir> HF_HUB_OFFLINE=1` gave 890
     passed, and the folder stayed empty.
   - `rag-eval check`: **exit 0**, all six parts ok, every floor met, so no tier-2
     re-run. Tier 1 here: 0.950, 0.796 and 0.858, exit 0, in 43 s.
   - `gh run list`: success on `main` (run 38349373369). The PR's run is to come.
2. **The fresh clone over HTTP:** the table above. Rows 1–8 pass; row 9 failed, and
   passes after the fix.
3. **The link check:** it prints nothing. All 28 `#anchors` in the docs also resolve
   with a GitHub-style slugger, the five new ADR index links included.
4. **The release:** `version = "0.3.0"` and `__version__ = "0.3.0"`. No `v0.3` tag yet:
   the maintainer tags `main` after the merge.

## Review (2026-10-14)

**Second review (Opus 5.5): no blockers; 4 doc findings and 4 nits, fixed on the PR.** It
covered the server fix, the docs against the code, and the run's own logs.

- **Checked and sound:**
  - **The gates, re-run on `6cec959`:** ruff and mypy clean; **890 passed, 5 xfailed**,
    coverage 98.37%; `rag-eval check` exit 0, all six parts ok.
  - **CI on the PR's head `6cec959`:** `check` and `eval-retrieval` green (run
    38396359569). Verification 1's PR run is done.
  - **The fix discriminates.** The new test's child, run against `main`'s `server.py`,
    died by SIGINT (status 130 in a shell) with a traceback; on the branch it exits 0
    with none. SIGTERM gives 143 and no traceback, as the docs say. A Ctrl-C while the
    lifespan loads the models is inside uvicorn's signal capture, so it exits 0 too.
  - **The upgrade guide works:** v0.2's own `config.yaml` is refused as shipped and loads
    once the `cross_encoder` entry is deleted.
  - **The README against the code:** the allowlist, `trust_remote_code`, the prompt
    rules, the chunk metadata, "item(s)" against v0.2's "file(s)", the job's fields, the
    golden set's 25/20/5, the thresholds, the Modelfile, the fingerprint's parts and
    runtime keys, and CI's triggers. With a token the Host check is off, so the
    "another machine" example works. Every name §2.2 cites exists.
  - **The run's logs** in `~/dev/rag-s2-9-e2e` match the table: 7a's close at +0.01 s
    and 7b's at +0.03 s, the runner idle at +133 s and +0.68 s, and the traceback in
    `serve-repro.log` but not in `serve-fixed.log`.
  - **The commits:** Conventional headers, `Refs:` trailers that git parses, and no
    attribution in the commits or the PR body.
- **Findings, fixed on the PR:**
  - **ADR-026's example was false.** It said that in `HuggingFaceBgeEmbeddings`,
    `encode_kwargs: {}` turns normalisation off. In the installed langchain-community
    0.4.2 that field already defaults to `{}`, and the class never normalises by itself;
    its quantized siblings do not either. The claim came from S2-6, where its test checked
    our identity function, not the library. The decision stands for its own reason.
    ADR-026's Cons and Lesson are reworded. ARCHITECTURE.md and S2-6's story get dated
    notes, and the comments in `manifest.py` and `test_manifest.py` are corrected. No
    hashed input moved.
  - **"CI's only job with network access"** (README) and **"CI's only network use"**
    (ADR-023) were wrong: the `check` job runs `uv sync --locked` and pip-audit's live
    lookup. What holds is that `eval-retrieval` is the only job that downloads models.
  - **A second Ctrl-C was undocumented,** and it is what uvicorn's log offers during the
    graceful wait ("CTRL+C to force quit"). Reproduced with a request in flight: it exits
    0, but does not shorten the 5 s wait (exit at 5.4 s after the first), skips the
    lifespan's shutdown, and uvicorn logs the cancelled tasks as errors. On `main` the
    same case exited 130, so the fix improves it. Now in the README, §2.8, and the
    backlog's Docker line: only a repeated SIGINT forces, so `docker stop` never does.
  - **Caveat 24 said about 70 min to score on a T4,** counting S2-5b's two scorings.
    CLAUDE.md, which cites it, the README, ADR-023 and caveat 16 say about 35 min for
    one. It now says 35.
- **Nits, fixed:** the checkpoint is `generation-latest.partial.jsonl`, not
  `<out>.partial.jsonl` (README; `<stem>` in §2.2); §2.8 put the floors in `gate.py`,
  where their checks live, and the numbers are in `eval/thresholds.yaml`; one overlong
  line in ADR-025; an odd break in README's upgrade step 5.

## Release, after the merge

Tag the merge commit on `main`, as `v0.2` was tagged, then check the tag's CI run:

```bash
git switch main && git pull --ff-only
git log --oneline -1          # the merge of this PR
git tag -a v0.3 -m 'v0.3: Phase 2 complete; callable over HTTP, with an enforced quality bar'
git push origin v0.3
gh run list --limit 3         # the tag's run: check and eval-retrieval green
```

Then the close-out commit records the PR number, the three CI runs (PR, `main`, tag) and
the tag's commit, in this file and in STATUS.md.
