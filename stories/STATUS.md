# Board

> Updated at every story close-out. This file + the story files ARE the
> project memory across sessions.

## Now

**Next action: [S2-4](phase-2/S2-4-reranker.md)**: our own cross-encoder reranker, switched
on or off by tier-1 numbers. Model fable, not plan-first, on branch `feat/s2-4-reranker`
(risky: it changes the retrieval every answer depends on, and shrinks the `_target_`
allowlist). Its inputs from S2-3 are in the backlog, under "S2-4 — inputs from S2-3",
including a CI step-order trap for the cross-encoder warm-up. The order of the Phase 2
stories, and what each depends on, are in the Phase 2 table below.

**S2-3 is done** (2026-10-04, PR #9). Tier 1 runs in CI's `eval-retrieval` job. Its
baseline is hit rate 0.800, MRR 0.5875 and recall 0.7167, against floors of 0.75, 0.537
and 0.666. GitHub's runner reproduced the baseline exactly, question by question.

**CI runs on pull requests, not on pushes to a story branch** (S2-3's first review). Push
runs are for `main` and `v*` tags only, so open the PR to get CI on a branch. `main`'s push
run is what writes the HF model cache that every PR run restores.

**The flaky cancel test is fixed** (PR #10, 2026-10-04). It was a real race in
`stream_answer`; the backlog line "Closed 2026-10-04" has the detail. A red `check` job is
real now.

**The Phase 2 architecture pass is done** (2026-10-02).

- `ARCHITECTURE.md` has a confirmed Phase 2 section (§2.1–§2.7).
- DEC-14 … DEC-18 are resolved below.
- The phase is sharded into S2-1 … S2-9.

Implementation sessions follow it and do not re-litigate it. The maintainer's answers are
recorded with it:

- a two-tier eval gate (DEC-15);
- a stale tier-2 run fails CI rather than warning;
- the API requires a token off localhost (DEC-18).

**The pass ran on Opus 5.5, not Fable.** ADR-018 routes architecture passes to Fable;
the maintainer chose to accept this pass rather than re-run it. Every backlog line now
names its owning Phase 2 story, or its re-deferral.

**Design reviews (2026-10-02, before any code).** Two reviews; the maintainer left the
second review's decisions to the reviewer, "useful before unusual".

- **First review**, already folded into the docs:
  - the SRS differences listed;
  - S2-5 split into S2-5/S2-5b, and S2-6 moved before S2-5;
  - `reranker_candidates` as one pipeline key;
  - the file's path in the chunk ID;
  - the tier-1 warm-up and cache key;
  - auth hardening (docs gated, Host check);
  - the API's start without an index;
  - the strict negative control.
- **Second review** (risky areas: auth, migration, concurrency). It found two problems
  in the first review's additions and changed four decisions:
  - **H1, fingerprint:** CI cannot recompute Ollama digests, and the prompt probe
    needs LangChain. Now: six parts computed in `evaluation/fingerprint.py`, which may
    import langchain_core; digests are recorded, not hashed, and checked by
    `check --with-ollama`; `notes` and `expected_sources` are left out; a
    `ground_truth` edit costs a re-score only.
  - **H2, judge context:** the OpenAI endpoint cannot set `num_ctx`, and Ollama's 4k
    default would silently truncate RAGAs' prompts. Now: `eval/judge.Modelfile`
    (`rag-judge`, 8k), hashed and checked at score time (caveat 22).
  - **H3, API disconnects:** uvicorn reports ASGI 2.3, so Starlette already cancels the
    stream on disconnect, and the planned producer task was the shape that would
    orphan generation. Now: `stream_answer` runs inside the SSE generator; refusals are
    dependencies; the slot has two release paths.
  - **H4, the prefill criterion:** Ollama reportedly finishes a prompt evaluation in
    progress. The criterion is now "our connection closes within about 1 s, and the
    runner idles by the end of the prefill" (caveat 21).
  - **Migration and concurrency:**
    - generations with an atomic symlink flip and `fsync`, replacing the two-rename
      swap;
    - an all-or-nothing apply phase, because FAISS's add is not atomic (verified);
    - a changed loader identity re-embeds the document;
    - the API reloads on the generation, not on mtime, and re-checks the embedder;
    - `flock`, not `lockf`.
  - **Auth and privacy:**
    - requests carrying `Origin` get 403, and every POST needs a JSON body (CSRF to
      localhost);
    - the token is compared as bytes;
    - ingest errors and logs are path-free.
  - **Smaller items:**
    - a second Ctrl-C closes the Runner;
    - tier-2 tolerance comes from two scorings;
    - the reranker is decided question by question;
    - S2-5 smoke runs get `--out`;
    - servers run in a foreground terminal.

**Phase 1 is complete and released as `v0.2`** (2026-10-02): tag on `db75439`, CI green
on the PR, `main` and the tag.

| Story | Commits | PR |
| --- | --- | --- |
| S1-1 | `b22a37a` | — |
| S1-2 | `1f1d6ff`, follow-up `f68eee3` | — |
| S1-3 | `3b0e77f`, follow-up `ddcd98f` | #1 |
| S1-4 | `eff50ea`, follow-up `a6f21b3`, DEC-13 `20c4a6a` | #2 |
| S1-5 | `e3927a4` … `4efa53d` (8 commits) | #3 |
| S1-6 | `ff61088` … `be545a4` (9 commits) | #4 |
| S1-7 | `6a50792` … `234f437` (7 commits) | #5 |
| S1-8 | `fe7c38e` … `db75439` (6 commits) | #6 |

**CI exists.** Every story closes on a green run. From S1-8 it gates ruff → mypy →
pytest + coverage → pip-audit.

**Commit and branch convention (maintainer, 2026-10-01):** Conventional Commits +
Conventional Branch, as set out in CLAUDE.md's hard rules. Story and SRS IDs go in a `Refs:` trailer,
not the header. History up to S1-4 predates it and is left as is.

**Branch rule (maintainer, 2026-09-30):** risky stories go on a feature branch
and a PR, not straight to `main`. The call is made and stated at the start of each
story.

**The Phase 1 architecture pass is done** (2026-09-25): `ARCHITECTURE.md` has a
confirmed Phase 1 section (§1.1–§1.7) and DEC-6 … DEC-12 are resolved below.
Implementation sessions follow it and do not re-litigate it. Every Phase 1
backlog item was resolved or re-confirmed in that pass — each line in the
Backlog now names its owning story or its re-deferral.

**Phase 0 is complete and released as `v0.1`** (2026-09-25): tag on
`ae15a23`, pushed to GitHub along with `main`. The exit criterion was met
literally: a fresh clone ran `uv sync`, ingested the corpus and answered
through `rag-query`.

**Known quality gap carried into Phase 1+:** retrieval is reliable, but the
7B model sometimes misstates retrieved facts (S0-6 fact-check: misattributed
scores, an invented acronym expansion). Faithfulness stays unmeasured until the
Phase 2 tier-2 harness (S2-5, baseline in S2-5b). The S0-6 story records the failing question and
its verified ground truth, which becomes the first golden record (S2-2).

**Commit history so far** (S0-1 → S0-4; ISS-09 verified fully closed, `git
ls-files` shows no parquet, FAISS index, cache, media or `.save` file):

| Commit | Contents |
| --- | --- |
| `ee86e1f` | pre-v0.1 baseline: workflow kit + Phase 0 arch pass |
| `f702dc8` | S0-1 hygiene (incomplete — missed four files) |
| `e781b17` | S0-1 fixup **fused with** the S0-3 file moves |
| `0987c60` | S0-2 uv project |
| `2bc7658` | S0-3 module split + entry points |
| `e16c07d` | S0-4 config repair + corpus rename |
| `ebcfc2e` | S0-4 follow-up: GPU and multilingual embedders restored as config axes |
| `6630222` | S0-5 LangChain 1.x LCEL chain, streaming CLI with sources |
| `55344bd` | S0-7 project docs into `docs/`, corpus rule inverted |
| `ae15a23` | S0-6 end-to-end proof, README, 0.1.0 — tagged `v0.1` |

`e781b17` deliberately carries two stories' worth of change: the S0-3 `git mv`
operations were staged when the fixup was committed, so the content merged, and
the message was amended to describe both rather than rewrite already-public
history. Read it as "S0-1 fixup + first half of S0-3". **Rule that came out of
it: Claude must not run staging git commands (`git mv`, `git rm`, `git add`)
while the maintainer commits manually — three collisions came from exactly
that.** S0-4's corpus rename therefore used a plain `mv`; rename detection is
computed from content at commit time, so history is unaffected.

The Phase 0 architecture pass is done: `ARCHITECTURE.md` Phase 0 is confirmed and
DEC-1/DEC-2/DEC-4 are resolved below.

## Release mapping (per SRS §12, rev 1.1)

| Phase exit | Git tag | Package version |
| --- | --- | --- |
| Phase 0 | `v0.1` | 0.1.0 — **released 2026-09-25** (`ae15a23`) |
| Phase 1 | `v0.2` | 0.2.0 — **released 2026-10-02** (`db75439`) |
| Phase 2 | `v0.3` | 0.3.0 |
| Phase 3 (SRS complete) | `v1.0` | 1.0.0 |

The final story of each phase performs the tag + version bump at close-out.

## Human prerequisites (do these anytime, no session needed)

- [x] `gh auth login` — done 2026-09-24 as `harshit-05` (HTTPS, keyring;
      scopes: repo, read:org, gist). Needed to push and for Phase 1 CI/PR work.

- [x] **`workflow` scope** — granted 2026-09-29, lost when `gh` was logged in
      again (by 2026-10-02 the token had `admin:public_key gist read:org repo`
      and S1-7's push was refused), and restored on 2026-10-02 with
      `gh auth refresh -h github.com -s workflow`. Git pushes to github.com
      through `gh auth git-credential`. Every push that changes
      `.github/workflows/*.yml` needs this scope, so check with `gh auth status`
      after any re-login. **Note:** the token now lives in
      `~/.config/gh/hosts.yml` (plain text), not the system keyring as before.
      Re-login with keyring storage if that was not intended.

- [x] Resolve DEC-1 / DEC-2 — done in the Phase 0 arch pass, 2026-09-18.

- [ ] **Phase 2 (S2-5b): somewhere to run tier-2 scoring.** It scores twice, to measure
  the judge's noise.
  - **Locally:** about 3–4 h of CPU per scoring with `gemma2:9b`, which is already
    pulled, so 6–8 h in all. The `rag-judge` variant (8k context) holds about 8 GB, an
    estimate S2-5 measures.
  - **On Colab/Kaggle:** needs an account, plus Ollama with the same `gemma2:9b` tag on
    the GPU runtime, and `ollama create rag-judge -f eval/judge.Modelfile`.

  S2-5b surfaces the estimate first, and the maintainer picks.

- [ ] Nothing else: uv ✓, Python 3.12 (uv-managed) ✓, Ollama daemon ✓,
      Docker 29.8.1 ✓ (Phase 3), disk 311 GB free ✓. **No MCP servers are
      required for any phase** — built-in tools cover the whole workflow.

## Pre-flight caveats (read before a story session)

Numbered because stories cite them by number; resolved ones stay as one line so
those references still land. Items 9 onward were learned in Phase 1; each one
cost a session something.

Items 15 onward come from the Phase 2 architecture pass (2026-10-02):

- Item 15 is a fact the pass verified.
- Items 16–21 are known costs of Phase 2 work.
- Item 22 comes from the second design review.

1. **Resolved (2026-09-18), historical.** The three pending commits (baseline,
   S0-1, S0-2) landed as `ee86e1f`, `f702dc8` + `e781b17` and `0987c60`. The
   lesson still holds: `git reset` does not untrack anything, and `.gitignore`
   never untracks an already-tracked file. `tests/test_repo_hygiene.py` now reads
   `git ls-files` on every CI run and fails if an artifact is tracked (ISS-09).
2. **Stale FAISS index trap (S0-5/S0-6).** `vectorstore/db_faiss/index.pkl`
   was pickled under the old LangChain — after the 1.x migration it will
   likely fail to unpickle. Any chain-construction check must re-ingest
   first; never debug an unpickle error there, just rebuild the index.
3. **CPU latency expectations.** 7B on CPU: Phase 0 measured ~5–15 s to first token
   and ~1–2 min full answers; the current small corpus ingests in minutes.
   Consequence: **NFR-2 (<2 s first-token p95) is not achievable CPU-only** — by v1.0
   either revise the SLO or plan GPU serving. Flagged in GPU offload notes below.
   - **S2-1 measured 254 s to first token** for mistral with k=5 (2026-10-02). Budget
     manual runs on that figure until it is re-measured.
   - **The cause is a setting, not this CPU** (S2-1 second review). Ollama's journal
     shows every mistral load with `NumThreads:2`: on this i7-1255U it counts only the
     2 performance cores, out of 10 cores and 12 threads. The service also recorded a
     2.3 GB swap peak, so the weights were partly swapped out. `ChatOllama` takes
     `num_thread`, and the backlog line "num_thread" sets it before S2-5b.
   - Before timing anything, check `free -h` and `ollama ps` (caveat 8), and read the
     `NumThreads:` value from `journalctl -u ollama`.
4. **First run downloads models.** Ingestion pulls the MiniLM embedder
   (~90 MB) from HuggingFace on first use — needs network once.
5. **Resolved (S0-1, S0-4), historical.** The parquet files were untracked,
   `docs/dataset.py` became `scripts/fetch_dataset.py`, and `corpus/` holds only
   the three PDFs.
6. **PyTorch index trap (S0-2).** `download.pytorch.org/whl/cpu` hosts stale
   `langchain-community` releases. Declared as a general index it makes uv
   silently resolve the whole stack to LangChain 0.3.x. It must be
   `explicit = true` and bound to torch via `[tool.uv.sources]` — see
   ARCHITECTURE.md §0.1. Check `langchain-core` is 1.x in `uv.lock`.
7. **Ollama context window.** Default `num_ctx` is 2048; the old config's
   k=10 × 1000-char chunks overflowed it silently. Config now sets
   `num_ctx: 4096`, `k: 5`. If answers look ungrounded, check these first.
8. **Memory at S0-6.** This host showed only 4.3 GB free with desktop apps
   open (2026-09-18). Run `free -h` before the proof; use `phi3` if under
   ~6 GB free, and record which model produced the transcript. Run `ollama ps`
   first: Ollama keeps the last model resident for ~5 min, so `free -h`
   under-reports, and `ollama stop <model>` gives that memory back (S1-1).

9. **Scratch ingests set both paths.** Any command that ingests a scratch corpus
   sets `RAG_DATA_PATH` **and** `RAG_VECTOR_STORE_PATH`. With only the first,
   the scratch build overwrites the real index (S1-3, S1-4).
10. **Check a guard the way CI runs it.** A plain `uv run` re-syncs the
    environment first, so it can repair the very state a check is meant to
    catch and then pass. Guards on the installed environment use
    `uv run --no-sync` (S1-7).
11. **Never gate warnings with `-W error` on the command line.**
    `langchain_core` resets the warning filters on import, so the gate cannot
    fail. Record warnings in-process, as `tests/test_deprecations.py` does (S0-5,
    S1-5).
12. **A plain `pip-audit` skips torch.** It looks up `2.14.0+cpu` on PyPI, finds
    no such release and only warns. CI's Audit step strips the local label, and a
    negative control showed that an old torch then fails the gate (S1-8).
13. **Pushing a workflow change needs `gh`'s `workflow` scope.** A re-login drops
    it, and the push is refused. Run `gh auth status` after any re-login (S1-7).
14. **Attack a security check; don't only confirm it accepts the good input.**
    The prefix-only allowlist passed every positive test and was bypassed twice
    (S1-2, ADR-020).
15. **Never stream an answer through `RunnablePassthrough.assign` where cancelling must
    stop it.** `RunnableParallel` waits on its step tasks with `asyncio.wait` and never
    cancels them. A cancelled consumer therefore returns while the generation runs on
    (verified; ARCHITECTURE.md §2.1, DEC-14). **Not through `prompt | llm | parser`
    either** (the second trap, fixed 2026-10-04). A sequence runs each chunk in its own
    task, so a cancel that arrives while a chunk is in flight leaves the model's stream
    open until the loop runs again. Render the prompt and stream the model itself, as
    `stream_answer` does.
16. **Tier-2 evaluation is hours-scale on CPU.** About 25 questions × 4 RAGAs metrics with
    `gemma2:9b` take about 3–4 h to score.
    - Generating takes about 40 min **if each answer takes ~1–2 min**. At S2-1's 254 s to
      first token, generating alone is about 1.8 h (25 × 254 s) before any decoding.
    - That figure ran on 2 threads with swapped weights (caveat 3). Re-measure after the
      `num_thread` backlog line, with memory freed, then surface the estimate before
      starting.
    - Scoring defaults to Colab/Kaggle (GPU offload notes).
17. **Never run the generator and the judge together.** mistral (~5 GB resident) and
    `gemma2:9b` do not both fit beside a desktop in 15 GB. Generate, run
    `ollama stop mistral`, then score. Caveat 8 applies to both.
18. **A local judge is noisy.** Judges of 7–9B parameters sometimes misparse RAGAs'
    structured prompts. Check a few scored answers by hand before a number becomes a
    threshold.
19. **CI's `eval-retrieval` job uses the network**: about 180 MB of HF models on a cache
    miss, and a few minutes of embedding. The unit job stays hermetic (DEC-11). If
    `eval-retrieval` is red while `check` is green, retrieval quality moved; the code did
    not break.
20. **After S2-6, every existing index must be rebuilt once.** v0.2 indexes have no
    manifest, and `rag-query` refuses them with exit 2. Run `rag-ingest`, and use
    `--rebuild` after an embedder change.
21. **Cancellation is proved only against real Ollama.** The hermetic tests show that our
    stream closes. Only the daemon shows that the model stopped: its log, or `top`
    falling to idle. S2-1 and S2-7 carry that manual step.
    - **Read the result in two parts.** Our connection must close within about 1 s.
    - Ollama reportedly finishes a prompt evaluation already in progress before it
      notices (third-party measurement, 2026-09-23; not yet seen here). So during the
      prefill, the runner may stay busy up to the end of the prefill (5–15 s on CPU).
      That is expected, not a bug in our code; generating tokens after the cancel would
      be one.
    - **Seen here in S2-1 (2026-10-02):** after a cancel during the prefill, the runner
      finished the abandoned prompt evaluation, about 4 min on this host, and generated
      nothing. A question asked during that time queues in Ollama behind it.
22. **Ollama's OpenAI endpoint cannot set the context size, and defaults to 4k below
    24 GiB of VRAM** (this host, and a Colab T4). Prompts over it are truncated silently.
    The judge is therefore `rag-judge`, built from `eval/judge.Modelfile` with
    `num_ctx 8192`, and `score` refuses a judge served with any other context. After a
    re-pull of `gemma2:9b`, run `ollama create` again. Caveat 7 is the same trap on the
    generator side.

## GPU offload notes (nothing *requires* GPU; Colab/Kaggle available)

- **Phase 0–1: no GPU work at all.**
- **Phase 2 — tier-2 RAGAs scoring (DEC-15)**: with a local CPU judge, a full
  metric sweep is an hours-scale run.
  - `rag-eval generate` runs here.
  - `rag-eval score` runs on Colab/Kaggle, with Ollama and `gemma2:9b` on their GPU.
  - The two exchange only `eval/runs/answers-latest.json`.
  - S2-5b records the recipe.
- **Bulk re-embedding** only if the corpus grows to thousands of docs.
- **Serving SLO (NFR-2)** — Colab/Kaggle are batch sandboxes with session
  limits, not hosting; if the <2 s SLO must hold at v1.0, that's a real
  GPU box or a hosted-LLM fallback, decided in the Phase 3 arch pass.

## Phase 0 — Make it run, make it honest

Exit criterion (SRS §12): a fresh clone runs ingestion and answers a query
end to end. Exit ⇒ tag `v0.1`.

| Story | Title | Closes | Depends | Status |
| --- | --- | --- | --- | --- |
| [S0-1](phase-0/S0-1-repo-hygiene.md) | Repo hygiene: gitignore + purge artifacts | ISS-09 | — | Done 2026-09-18 |
| [S0-2](phase-0/S0-2-uv-init.md) | uv project: pyproject, pinned 3.12, lockfile | ISS-08 | S0-1, DEC-1 | Done 2026-09-18 |
| [S0-3](phase-0/S0-3-collapse-trees.md) | Collapse v1/v2/temp into one package | ISS-10, ISS-12 | S0-2 | Done 2026-09-19 |
| [S0-4](phase-0/S0-4-config-repair.md) | Config repair: keys, paths, corpus rename | ISS-01, ISS-02, ISS-11, DEC-4 | S0-3 | Done 2026-09-25 |
| [S0-5](phase-0/S0-5-langchain-migration.md) | Migrate code to resolved LangChain version | ISS-17 | S0-4, DEC-1 | Done 2026-09-25 |
| [S0-7](phase-0/S0-7-docs-layout.md) | Project docs into `docs/`; invert the CLAUDE.md corpus rule | DEC-4 | S0-4 | Done 2026-09-25 |
| [S0-6](phase-0/S0-6-end-to-end-proof.md) | End-to-end proof: ingest + answered query | — (exit) | S0-5, S0-7, DEC-2 | Done 2026-09-25 |

Execution order follows the **Depends** column, not the story number: S0-7 was
added after the initial sharding (DEC-4) and runs between S0-5 and S0-6, so the
`v0.1` tag ships the final layout.

## Phase 1 — Make it trustworthy

Exit criterion (SRS §12): CI is green and would have caught every Phase-0 bug.
The second half is literal — `tests/test_config_regressions.py` (S1-5) encodes
each Phase-0 bug as a case. Exit ⇒ tag `v0.2`, version 0.2.0.
Design: ARCHITECTURE.md §1.1–§1.7.

| Story | Title | Closes | Depends | Status |
| --- | --- | --- | --- | --- |
| [S1-1](phase-1/S1-1-config-schema.md) | Validate the config with a frozen Pydantic model | FR-8, ISS-01, ISS-11, NFR-11 | — | Done 2026-09-26 |
| [S1-2](phase-1/S1-2-target-allowlist.md) | Allowlist `_target_` imports; stand up CI | ISS-04 | S1-1 | Done 2026-09-29; review follow-up 2026-09-30 pending commit |
| [S1-3](phase-1/S1-3-own-loaders.md) | Own loaders, extension map, recursive walk | ISS-13, FR-2, DEC-5 step 1 | S1-1 | Done 2026-09-30 (PR #1) |
| [S1-4](phase-1/S1-4-error-handling-cli.md) | Explicit error handling and a real CLI surface | ISS-05, ISS-06, ISS-18, NFR-7 | S1-3 | Done 2026-10-01 (PR #2) |
| [S1-5](phase-1/S1-5-regression-suite.md) | Phase-0 regression suite and coverage gate | ISS-07, NFR-8 | S1-2, S1-4 | Done 2026-10-01 (PR #3) |
| [S1-6](phase-1/S1-6-types-and-lint.md) | Type annotations, ruff and mypy configuration | ISS-19, NFR-9, ISS-12 | S1-5 | Done 2026-10-02 (PR #4) |
| [S1-7](phase-1/S1-7-cuda-extra.md) | GPU embedder path installable (CPU/CUDA torch variants) | FR-1 (GPU axis) | S1-1 | Done 2026-10-02 (PR #5) |
| [S1-8](phase-1/S1-8-phase-1-exit.md) | Phase 1 exit: audit gate, doc hygiene, 0.2.0 | — (exit) | S1-6, S1-7 | Done 2026-10-02 (PR #6, `v0.2`) |

CI lands in S1-2, not at the end, so every later story closes on green rather
than the whole phase arriving unverified at once. Each gate is added by the
story that makes it passable (ruff + pytest in S1-2, coverage in S1-5, mypy in
S1-6, pip-audit in S1-8) and blocks from that moment (DEC-11). S1-2 lands the ruff
gate on a tree that already has 4 default-rule errors, so it suppresses them with
`noqa` tags naming S1-4, and S1-4 removes them. S1-7 depends only on S1-1 and can
be run whenever convenient before S1-8.

## Phase 2 — Make it a service

Exit criterion (SRS §12): the system is callable over HTTP and has an enforced quality
bar. Both halves are executable:

- **Callable over HTTP:** the API's tests, plus a manual end-to-end run against real
  Ollama (S2-7, S2-9).
- **An enforced quality bar:** the two eval gates in CI (DEC-15).

Exit ⇒ tag `v0.3`, version 0.3.0. Design: ARCHITECTURE.md §2.1–§2.7.

| Story | Title | Closes | Depends | Status |
| --- | --- | --- | --- | --- |
| [S2-1](phase-2/S2-1-answer-stream.md) | Stream answers as events; Ctrl-C cancels generation | FR-5 (structured sources), backlog: Ctrl-C | — | Done 2026-10-03 (PR #7); cancel-race fix 2026-10-04 (PR #10) |
| [S2-2](phase-2/S2-2-golden-dataset.md) | Golden dataset, seeded with GLIDER | FR-7 (dataset), ISS-15 (part), SRS §7.4 | — | Done 2026-10-03 (PR #8) |
| [S2-3](phase-2/S2-3-retrieval-eval.md) | Tier-1 retrieval eval and its CI job | FR-7 (tier 1) | S2-1, S2-2 | Done 2026-10-04 (PR #9) |
| [S2-4](phase-2/S2-4-reranker.md) | Our own cross-encoder reranker, decided by the numbers | FR-4, ISS-03, DEC-5 step 2 | S2-1, S2-3 | Todo |
| [S2-5](phase-2/S2-5-ragas-gate.md) | Tier-2 RAGAs harness and the freshness gate (code only) | FR-7 (harness), ISS-15 | S2-2, S2-4, S2-6 | Todo |
| [S2-5b](phase-2/S2-5b-tier2-baseline.md) | Tier-2 baseline run, floors, and the CI check | FR-7 (enforced) | S2-5 | Todo |
| [S2-6](phase-2/S2-6-ingest-manifest.md) | Incremental ingestion with a hash manifest | FR-2, ISS-14, NFR-4 (part), SRS §7.3 | S2-1 | Todo |
| [S2-7](phase-2/S2-7-http-api.md) | FastAPI service with SSE streaming | FR-6, SRS §8.1 | S2-1, S2-6 | Todo |
| [S2-8](phase-2/S2-8-cli-polish.md) | CLI polish batch | backlog: CLI polish | S2-1, S2-6 | Todo |
| [S2-9](phase-2/S2-9-phase-2-exit.md) | Phase 2 exit: docs, end-to-end, 0.3.0 | — (exit) | S2-1 … S2-8, S2-5b | Todo |

**Order.** S2-1, S2-2, S2-3, S2-4, **S2-6**, S2-5, S2-5b, S2-7, S2-8, S2-9. It differs
from the numbering in one place, and the Depends column is satisfied throughout:

- S2-6 runs before S2-5, because S2-5's fingerprint reuses `manifest.py`'s identity
  functions, and those belong to S2-6 (risky, and reviewed there). Written once, they
  cannot drift and force an hours-scale re-run.
- The reranker (S2-4) is switched on or off by tier-1 numbers.
- Tier 2's hours-scale baseline (S2-5b) is taken once, on the final retrieval config.
- S2-5 was split: the harness and gate are code (one session), the baseline and the CI
  step are a long run (S2-5b).

**Story setup.**

- **Plan-first:** S2-1, S2-5, S2-6 and S2-7. Each is design-heavy or touches an API
  surface not used here before.
- **Branches:** every story goes on its own branch with a PR. Each story file states its
  risk call. S2-2 and S2-8 are the low-risk ones; they use branches only so that CI runs
  before the merge.
- **Version:** S2-1 bumps it to `0.3.0.dev0`, and S2-9 releases 0.3.0.

## Decisions log

| ID | Decision | Status | Notes |
| --- | --- | --- | --- |
| DEC-1 | LangChain: migrate to 1.x vs pin legacy 0.2.x | **Resolved 2026-09-18: migrate to 1.x** | Verified resolve: core 1.6.3, community 0.4.2, ollama 1.1.0, huggingface 1.2.2, text-splitters 1.1.2, classic 1.0.8 (transitive only). Rules: app code imports only `langchain_core` / `_text_splitters` / `_community` / `_huggingface` / `_ollama`; chain is hand-composed LCEL, no `langchain_classic` imports in Phase 0–1; PyTorch index `explicit = true`. Full rationale: ARCHITECTURE.md §0.1. |
| DEC-2 | Query LLM: pull `qwen2:7b` vs repoint to already-pulled `mistral` | **Resolved 2026-09-18: `mistral`, `phi3` fallback** | Same weight class as qwen2:7b, so the reason is zero download + known-good here, not RAM. `qwen2_ollama` entry stays in config unused. All Ollama entries: `temperature 0`, `num_ctx 4096`, `num_predict 512`, `validate_model_on_init true`; retriever `k 5`. Revisit with the Phase 2 eval harness. |
| DEC-4 | Repo layout: `docs/` currently holds the RAG corpus, colliding with the universal convention that `docs/` is project documentation | **Resolved 2026-09-18: corpus → `corpus/`, `docs/` becomes project documentation** | Maintainer decision. The old name needed a CLAUDE.md hard rule to stay safe, and the text loader claims `.md`, so a project doc dropped in there gets embedded into the index. Renamed in S0-4 (which rewrites every path anyway); `SRS.md` / `WORKFLOW.md` / `ARCHITECTURE.md` move into `docs/` in S0-7, which also inverts the CLAUDE.md rule. `CLAUDE.md` and `README.md` stay at root (auto-load; GitHub renders README from root only). `stories/` stays at root as working state. |
| DEC-5 | `langchain-community` was sunset 2026-05-22 (issue #674): frozen, unmaintained, warns on import. We use it for FAISS, the loaders, and the disabled cross-encoder | **Resolved 2026-09-25: staged exit, keep through Phase 0** | Nothing is broken; the lock pins it and `langchain-core<2` bounds drift. Exit rides planned work: loaders → ~40 lines of own code on `pypdf`/`docx2txt` (Phase 1, with the loader-selection item); cross-encoder → `sentence-transformers` directly (Phase 2 reranker); FAISS → `langchain-qdrant` (Phase 3). After Phase 3 both `langchain-community` and its transitive `langchain-classic` leave `pyproject.toml`. No official standalone FAISS/loader package exists; the unofficial `langchain-faiss` 0.1.1 is rejected on supply-chain grounds. |
| DEC-3 | Vector store for Phase 3: Qdrant vs pgvector | **Lean: Qdrant** (decide in Phase 3 arch pass) | Native hybrid dense+sparse, one container, no Postgres to run. pgvector only if a Postgres already exists in the deployment. Store-specific code is confined to `vectorstore.py` so either works. |
| DEC-6 | Config validation: how far Pydantic reaches into the call sites | **Resolved 2026-09-25: frozen `RagConfig`, attribute access everywhere** | `load_config()` returns a frozen model, not a dict; typed component kinds (`extra="forbid"`) make ISS-01 and ISS-11 unwritable; component leaves stay `extra="allow"` (ADR-010); retrievers get their own `RetrieverSpec` (no `_target_` — typing them as components fails the real config, verified). Non-mutation becomes structural via the access path: call sites only get fresh copies from `component(ref).spec()`/`.kwargs()` — `frozen=True` alone does not freeze nested dicts (verified). **Verified trap:** a field literally named `_target_` is silently dropped by Pydantic (leading underscore ⇒ private attribute; `model_dump()` returned `{}`) — the schema must use `target: str = Field(alias="_target_")` and dump `by_alias=True`. S1-1. ARCHITECTURE.md §1.1, §1.4. |
| DEC-7 | Where the `_target_` allowlist lives, and whether it is configurable | **Resolved 2026-09-25: hardcoded, inside `import_from_string`** | Enforcing at the import funnel rather than in `build_object` covers `ingest.py`'s direct call — the gap ADR-015 flagged. Not config-overridable and no env escape hatch: an allowlist the config can edit is not an allowlist. `langchain_classic.` is allowed as a *config* prefix (disabled reranker entry); ADR-007 governs *source* imports and is unaffected. Load string-checks prefixes (nested too) and never imports; a separate `check_imports` helper, used by tests, catches misspelled classes (ISS-03). S1-2. Closes ISS-04. |
| DEC-8 | DEC-5 step 1: replace the `langchain-community` loaders now or later | **Resolved 2026-09-25: now, in Phase 1** | Own `PdfLoader`/`DocxLoader`/`TextLoader` on `pypdf`/`docx2txt`; `extensions` leaves the component entries for `pipeline.ingestion.loaders`, giving one instantiation path; discovery becomes recursive (ISS-13). Afterwards `langchain_community` is imported only by `vectorstore.py`. Risk is citation metadata, so the story records the `v0.1` chunk-count + citation baseline before changing anything and must reproduce it. S1-3. |
| DEC-9 | NFR-6 (timeout / retry / circuit-breaking): Phase 1 or later | **Resolved 2026-09-25: Phase 1 does error handling only** | ISS-05 (collect failures, exit non-zero — NFR-7) and ISS-06 (REPL try/except) are S1-4. Timeout, retry-with-backoff and circuit-breaking (SRS §11 Resilience) need the service shape and move to Phase 3 with the serving decision. Deferred deliberately, not by omission. |
| DEC-10 | Device selection: collapse the `_cpu`/`_cuda` embedder entries into one env-driven setting | **Resolved 2026-09-25: no — the entries stay explicit** | The backlog item assumed pydantic-settings interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside YAML values. **It does not** — there is no `${VAR:-default}` expansion for arbitrary values. With the CUDA torch variant (DEC-12), one sync command plus repointing `pipeline.ingestion.embedder` is already a one-line switch, so the four entries restored in `ebcfc2e` stay as FR-1 axes. |
| DEC-11 | CI: host, hermeticity and how hard the gates bite | **Resolved 2026-09-25: GitHub Actions, hermetic suite, blocking gates** | No Ollama, no model download, no network in tests: `DeterministicFakeEmbedding` (verified present in core 1.6.3) + a fake chat model. Gate order: `ruff check` → `mypy` (`disallow_untyped_defs` on `src/rag_qa`) → `pytest --cov-fail-under=80` (omitting `evaluate.py`, Phase 2 + `eval` extra) → `pip-audit`. Each gate blocks from the story that adds it. **No `ruff format --check` in Phase 1** — the tree is unformatted and a whole-tree reformat is the diff that gets rubber-stamped → backlog. |
| DEC-12 | Making the declared GPU embedder entries actually installable | **Resolved 2026-09-25: uv-conflicting CPU/CUDA torch variants; plain `uv sync` stays CPU** | Each variant routed to its own index, `explicit = true` on both. **Mechanism chosen by S1-7's spike**: extras have no default, so torch-only-in-extras would make a plain sync pull PyPI CUDA torch via `sentence-transformers`. Preferred: dependency groups + `default-groups = ["dev", "cpu"]`; fallback: extras with `--extra cpu` on every install path. S1-7 re-runs the S0-2 smoke test on the **installed env** (the lock legitimately holds both variants): core 1.x, `torch 2.14.0+cpu`, zero `nvidia-*`. CUDA is verifiable here as a **resolve only** — this host is CPU-only. |
| DEC-13 | Partial ingestion failure: replace the index with what loaded, or keep the previous one | **Resolved 2026-10-01: replace, exit 1** | Maintainer's call, raised by S1-4's third review. Some documents or folders unreadable → index the rest, save over the old index, exit 1 ("rebuilt without them"). Keeping the old index is safer when unattended, but one persistently bad file would block every update. Phase 1 ingests are hand-run and watched. Phase 2's hash manifest dissolves it: a failed document keeps its previously indexed chunks. Exit codes: 0 all indexed · 1 document/folder unreadable, nothing indexed, or unexpected error (traceback) · 2 cannot start. ARCHITECTURE.md §1.1. |
| DEC-14 | Cancelling an answer: how the CLI's Ctrl-C and the API's client disconnect stop generation | **Resolved 2026-10-02: one answer-event stream (`answering.stream_answer`); stopping it means cancelling the task that consumes it** | **Verified trap:** async alone does not fix it. Under `RunnablePassthrough.assign`, `RunnableParallel` waits on its step tasks with `asyncio.wait` and never cancels them. In the trial, a cancel at 0.5 s during a simulated prefill left the stream open until 2.0 s; streaming `prompt \| llm \| parser` directly closed it at 0.5 s. **Design:** the CLI uses one `asyncio.Runner` per session. `QueryPipeline` holds the parts, and `build_rag_chain` stays for `invoke`. `aclose_llm` closes ChatOllama's clients, and `ttft_ms` is measured for NFR-2. S2-1. ARCHITECTURE.md §2.1. |
| DEC-15 | Phase 2 quality gate: what runs in CI when RAGAs on CPU takes hours | **Resolved 2026-10-02: two tiers** (maintainer) | **Tier 1:** retrieval hit rate, MRR and recall against `expected_sources`, recomputed on every pull request and every push to `main` by a CI job that caches the HF models. That job is CI's only network use. **Tier 2:** RAGAs with a `gemma2:9b` judge through Ollama's OpenAI endpoint, run offline (scoring can go to Colab/Kaggle) and committed with a fingerprint. `rag-eval check` fails CI on a floor breach **or a stale run** (maintainer: fail, not warn). Rejected: a hosted judge in CI; offline-only. **Second review:** the fingerprint has six parts (`query` with a rendered-prompt probe, `ingestion`, `corpus`, `questions`, `references`, `judge`) in `evaluation/fingerprint.py`; Ollama digests are recorded, not hashed; the judge is `rag-judge` from `eval/judge.Modelfile` (8k context, caveat 22); the tier-2 tolerance comes from two scorings. S2-2, S2-3, S2-5, S2-5b. |
| DEC-16 | Reranker: `langchain_classic`'s `ContextualCompressionRetriever`, or our own | **Resolved 2026-10-02: our own `CrossEncoderReranker` on `sentence-transformers`** (DEC-5 step 2) | A `langchain_core` `BaseDocumentCompressor` that writes `rerank_score` into metadata. k=20 candidates are cut to `top_n` 5, and tier-1 numbers decide whether it is on. `langchain_classic.` and `langchain_community.` leave `ALLOWED_PREFIXES`. The `min_score` knob is for the Sources-relevance backlog line. S2-4. |
| DEC-17 | Incremental ingestion: change detection, failure handling, write safety | **Resolved 2026-10-02: a sha256 manifest beside the index, generation directories published by an atomic symlink flip, one writer** (the flip replaced a two-rename swap in the second review) | **Updates:** only added and changed documents are re-embedded. A failed document keeps its chunks, which dissolves DEC-13. **Full rebuild** when the embedder identity or the splitter changes. Device keys are left out of the identity, so Colab's `_cuda` equals local `_cpu`. **IDs and metadata:** uuid5 chunk IDs, ready for Qdrant (DEC-3). `source` becomes corpus-relative, plus `source_sha256` and `ingested_at` (SRS §7.3). **Query side:** `rag-query` refuses a mismatched index. **Breaking:** re-ingest once. S2-6. |
| DEC-18 | HTTP API shape: SSE library, auth, concurrency, ingest jobs | **Resolved 2026-10-02: FastAPI ≥ 0.135 native SSE; a token is required off localhost (maintainer); one generation at a time** | **Surface:** `rag-serve`. `QueryRequest` is `{question}` with `extra="forbid"`, so the API never supplies components, and a `trust_remote_code` load guard comes with it. **Load:** 503 with `Retry-After` when busy. A disconnect cancels generation: under uvicorn (ASGI 2.3) Starlette cancels the stream, and `stream_answer` runs inside it, with no producer task (second review). Requests with an `Origin` header are refused. **Ingest:** single-flight jobs in memory, which swap only the retrieval half. **Privacy:** no host paths over HTTP. S2-7. |

## Backlog

Reconciled at the Phase 1 exit (S1-8, 2026-10-02). Every Phase 1 line is either
**closed** by the story named, or **re-deferred** with a phase and a reason. The
detail behind each closed line is in its story file.

Reconciled again in the Phase 2 architecture pass (2026-10-02). Every open line now
leads with one of two things:

- **its owning Phase 2 story**, as **S2-n —**, where the detail below is that story's
  input;
- **its phase or trigger**, when it stays deferred.

### Closed in Phase 2

- **S2-1:** Ctrl-C cancels generation (DEC-14): one answer stream, `stream_answer`,
  streamed outside `RunnablePassthrough.assign`, and the second Ctrl-C quits cleanly. The
  CLI closes ChatOllama's clients at exit (`aclose_llm`); the API half stays with S2-7.
  `py.typed` and `types-PyYAML` (found in S1-6).
- **S2-2:** the golden set, `eval/eval_dataset.jsonl`, seeded with GLIDER (S0-6, and the
  v0.2 manual test's two traps). It has 25 records, 20 answerable and 5 unanswerable, each
  signed off by the maintainer against its PDF. `evaluation/dataset.py` validates the
  file, and a committed test checks that every quote in the notes is on its cited page.
- **S2-3:** tier 1 of the quality bar. `rag-eval retrieval` scores hit rate, MRR and recall
  against the golden set. The `eval-retrieval` CI job recomputes them on every pull request
  and every push to `main`, against the floors in `eval/thresholds.yaml`. It is CI's only
  job with network access: the model weights are cached, and on a cache hit nothing
  reaches the network. Exit 1 means a floor was missed; exit 2 means it could not run.

### Closed in Phase 1

- **S1-1:** the config stops being mutable (`build_rag_chain` gets fresh copies
  only); `create_store` / `open_store` take a path, not the whole config;
  malformed configs (`paths: {data: }`, an empty YAML) fail with a `ConfigError`
  instead of a `TypeError` / `AttributeError`.
- **S1-2:** the `_target_` allowlist (ISS-04). The trust-boundary *doc lines*
  from its second review were written in S1-8: `registry.py` docstring,
  ARCHITECTURE.md §1.8, and README "Known limitations".
- **S1-3:** loaders selected by extension through `pipeline.ingestion.loaders`
  (DEC-8); embedder resolution in one place (`components.build_embedder`).
- **S1-4:** ISS-05 / ISS-06 error handling and exit codes (DEC-9, DEC-13);
  `argparse` with `--help` / `--config`; `fetch_dataset.py` (ISS-18); ignored
  files counted and symlinks never followed (two S1-3 leftovers).
- **S1-5:** the deprecation gate as a pytest; CI on uv-managed Python;
  two-dot extension keys rejected; ADR-009 as a runtime test.
- **S1-6:** ruff from its defaults plus `E501`; `warn_unreachable` (ISS-12's
  guard); the private-field test made behavioural.
- **S1-7:** the GPU embedder path installable (DEC-12).
- **S1-8:** the S1-1 doc items (Ollama `keep_alive` vs `free -h`, now in
  caveat 8 and CLAUDE.md; `gemma2:9b` in CLAUDE.md's environment facts). The
  "scratch ingests set both paths" habit is now pre-flight caveat 9.

### Re-deferred at the Phase 1 exit

- **Phase 3 — the Docker image installs with `uv sync` or `uv export`, never
  `pip install .` (found in S1-7's review).** `pip install .` and
  `uv pip install .` ignore dependency groups, so they skip the `cpu` group and
  pull torch from PyPI as the CUDA build, with GBs of `nvidia-*` wheels. `uv sync`
  and `uv export` honour the default groups. A constraint for the Phase 3
  container story.
- **Backlog — audit-gate policy (S1-8 review).** CI's Audit step does a live
  advisory lookup and blocks, so a new advisory with no fixed release turns every
  PR red, docs-only ones included. Decide an `--ignore-vuln` policy (ID + reason in
  `ci.yml`) or a scheduled audit beside a PR gate for new dependencies. Not needed
  until it first happens.
- **S2-5 — prompt placeholder check (FR-8 follow-up, found in S1-1).** A
  `human` prompt missing `{context}` silently answers without retrieval. One
  validator on `Prompt`. S1-5 left it open, as the board allowed. It belongs with
  the Phase 2 evaluation harness, which is what would catch an ungrounded
  answer anyway. Its rules are in ARCHITECTURE.md §2.3.
- **S2-7 — the API must never let a request supply or override components
  (S1-2 second review).** A constraint for the Phase 2 architecture pass, with
  the optional load-time guard that rejects `trust_remote_code` anywhere in the
  config. Re-deferred because it only matters once the config is reachable from
  outside the host. Designed in DEC-18: `QueryRequest` forbids extra keys, and the
  `trust_remote_code` guard lands in the same story.
- **S2-8 — CLI polish (found in S1-4).** No single item is worth a story, so they are
  batched as S2-8. It runs after S2-6, which rewrites `ingest.py`:
  - `rag-query` startup is unguarded: Ollama being down
    (`validate_model_on_init`) and Ctrl-C while models load both print a
    traceback. One startup try/except with a clean message.
  - The connection error doesn't name Ollama (`ConnectError: [Errno 111]`). Add
    the hint "Is Ollama running? `systemctl status ollama`", and stop the empty
    `Answer:` header printing before the error.
  - Zero-count summary lines ("Skipped 0 symlink(s)") print on every run; print
    them only when nonzero.
  - An unreadable corpus *root* reports `./: PermissionError`, exit 1. "Corpus
    directory not readable" with exit 2 would match "not found". "Failed N
    file(s)" also counts folders now; say "item(s)".
  - `fetch_dataset.py`: its corpus guard checks only the repo's `corpus/`, not
    `paths.data` / `RAG_DATA_PATH`; nothing checks the `PAR1` header; and
    `requests` is undeclared (transitive only).
  - **No line editing at the prompt** (v0.2 manual test). `input()` without
    `readline` turns the arrow keys into text: an up-arrow was sent to the
    model as the question `^[[A`. `import readline` in `cli.py` (stdlib) gives
    editing and history.
  - **pypdf's own warnings leak** (v0.2 manual test). A corrupt PDF prints
    `invalid pdf header…` / `EOF marker not found` above the banner, before
    the clean error. Quiet the `pypdf` logger to ERROR in `ingest.py`.
- **S2-6 — the walk descends into ignored trees** (found in S1-3 and S1-4).
  It walks all of `.git` before discarding it. Performance only. Phase 2's
  incremental ingestion rewrites discovery around the manifest, so prune there
  (`Path.walk` is top-down, and pruning `folders[:]` in place fixes it).
- **S2-7 — keep a client disconnect a cancel (found in S2-1).** Closing `stream_answer`
  at a `yield` does not close the model stream before `aclose()` returns: langchain-core
  1.6.3 never closes a chat model's inner generator, and the loop's finalizer closes it a
  loop iteration or two later. A cancel, which DEC-18 already relies on, closes it at once:
  since the fix in the next line, every time. Detail in the S2-1 story, under Discovered.
- **Closed 2026-10-04 (`fix/s2-1-cancel-race`) — a cancel sometimes closed the model
  stream late (found in S2-2's CI; diagnosed in S2-3's second review).**
  - **The cause:** `stream_answer` streamed `prompt | llm | StrOutputParser()`, and a
    sequence runs each chunk in its own task. A cancel arriving while a chunk was in
    flight ended that task, not the model's read, and the stream stayed open until the
    loop's finalizer closed it. In the CLI that was not until the next question:
    Ollama kept generating while the user sat at the prompt. That was shown with the
    real `repl()`, a real SIGINT and a fake model, not yet against real Ollama.
  - **The fix:** render the prompt and stream `llm.astream(messages)` itself
    (ARCHITECTURE.md §2.1, the second trap).
  - **The numbers:** one core, mid-stream, 66 of 300 failures before and 0 of 300
    after. Forcing the cancel while a chunk is processed is now a deterministic test, with
    a strict `xfail` control on the old shape. The CLI test checks that the stream is
    closed when the prompt returns.
  - **The history below is kept as found.**
  S2-1's `test_cancel_closes_the_stream_before_the_consumer_returns[mid-stream]` is
  flaky. CI tested `dd0f315` twice: the push run failed ("the consumer returned while the
  model was still streaming"), and the PR run passed.
  - **Reproduced locally (2026-10-03).** Calling the test function in a loop, it failed
    5 times in 200 runs, and 21 in 200 pinned to one core (`taskset -c 0`), so a slow CI
    runner meets it more often.
  - **What a failure is.** In all 33 of 300 one-core failures, the model's stream closed
    7 event-loop iterations after the cancelled consumer returned, and no token was
    generated after the cancel. So nothing is orphaned: the close is late, through the
    loop finalizer that the line above describes. A cancel landing at the wrong moment
    (likely after a chunk is delivered, before the consumer resumes) takes that path
    instead of unwinding through the model's `finally`.
  - **Why it matters beyond the test.** The CLI's `Runner.run()` returns once the
    cancelled task is done, and the loop then sits idle at the `input()` prompt. If the
    remaining 7 iterations wait for the next `run()`, the HTTP stream to Ollama can stay
    open while the user is at the prompt. That is not yet checked against real Ollama.
    DEC-18's disconnect relies on the same claim that a cancel closes the stream at once.
  - **Fix first, then re-measure:** make `stream_answer` close the model stream
    deterministically on cancel, then show the test passing in a few hundred one-core
    runs. Do not paper over it with a retry or a sleep in the test.
- **Phase 3 — the parity tests and the corpus PDF they read** (found in S1-3).
  They are accepted until `langchain-community` leaves. Removing that PDF before
  then fails them loudly, which is right.
- **On the first real GPU run — CUDA 12 fallback (found in S1-7).** `cu130`
  wheels need a recent NVIDIA driver, and this host cannot check that. If
  Colab/Kaggle refuse, add a `cuda12` variant on `cu128`, which caps torch at 2.11.
  This is triggered by an event, not by a phase.
- **Accepted, no action — torch parity between the variants (found in S1-7).**
  CUDA resolves `2.14.1+cu130`, CPU stays at `2.14.0+cpu`.
  `uv lock --upgrade-package torch` aligns them if it ever matters.

### Re-deferred or dropped in the Phase 1 architecture pass (2026-09-25)

- **Dropped — device selection as one env-driven setting.** The item assumed
  `pydantic-settings` interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside
  YAML values; it does not (no `${VAR:-default}` expansion for arbitrary
  values). The CUDA torch variant (DEC-12) makes the switch one line anyway. See **DEC-10**.
- **Phase 3 — hosted-LLM fallback** (`groq_llama3` entry behind an optional
  extra + `.with_fallbacks()` in `chain.py`). Moved out of Phase 1: it widens
  the DEC-1 import surface and needs an API key, for something Phase 1 could
  only exercise against a fake failing LLM. It belongs with the Phase 3 serving
  decision (NFR-2), which is the problem it actually solves.
- **Phase 3 — NFR-6** timeout / retry-with-backoff / circuit-breaking, per
  **DEC-9**.
- **Backlog — `ruff format`.** Not gated in Phase 1: the tree is unformatted and
  a whole-tree reformat is the kind of diff that gets rubber-stamped (DEC-11).
- **Not yet storied — per-loader splitter strategies** (SRS §7.2): chunking
  configurable per document type. The `_target_` mechanism already supports it;
  it needs more splitter entries and a per-loader default, not new architecture.

### Later phases

- **S2-6 —** chunk metadata: content hash of the source file + ingestion timestamp
  (SRS §7.3), with the manifest that needs them (DEC-17).
- **S2-4 — inputs from S2-3 (found in S2-3).** Detail in S2-3's Discovered section.
  - **What a reranker can reach.** At k=20, three of tier 1's four misses have their page
    in the candidates (ranks 10, 10 and 11). So the reranker can at best lift the hit rate
    from 0.80 to 0.95. `glider-slm`'s p. 2 is not in the top 20 at all, so no reranker can
    recover it.
  - **Step 3's scratch configs need `--dataset` and `--thresholds`.** The eval files are
    anchored beside the config file, as `paths` are, so without the flags a scratch copy
    of `config.yaml` exits 2: "cannot read the golden set". Already written into S2-4's
    step 3, with the flags spelled out: zsh does not word-split an unquoted `$VAR`.
  - **"S2-3's scratch index" was in a session scratchpad under `/tmp`.** Build a fresh
    one with `RAG_VECTOR_STORE_PATH=<scratch>/index uv run rag-ingest` (about 2 min).
  - **CI: the cross-encoder warm-up goes between "Ingest the corpus" and "Save the HF
    models"** (found at S2-3's close-out). S2-3 saves the cache right after ingest, so a
    warm-up placed after the save is never cached. The next exact-key hit then runs
    offline without the cross-encoder and fails, or downloads it on every run if left
    online. The warm-up also takes ingest's
    `HF_HUB_OFFLINE: ${{ steps.hf-models.outputs.cache-hit == 'true' && '1' || '0' }}`, so
    an exact hit stays offline (S2-3's second review). The first run after the key change
    is a prefix hit: MiniLM is restored, and only the cross-encoder downloads.
- **S2-4, closes as not needed —** `langchain-classic` is referenced by the (disabled)
  reranker component but is only a transitive dependency via `langchain-community`.
  The question was whether to declare it in `pyproject.toml`. DEC-16 hand-rolls the
  reranker instead, and `langchain_classic.` leaves the allowlist, so there is
  nothing to declare.
- **S2-5 —** the RAGAs judge must be pointed at local Ollama explicitly (its default
  is OpenAI). DEC-15 settles both halves: the judge is `gemma2:9b` through Ollama's
  OpenAI endpoint, and a full sweep on CPU takes hours, so scoring offloads to
  Colab/Kaggle.
- **S2-5b, first step — set Ollama's `num_thread` (found in S2-1's second review).**
  Ollama loads mistral with `NumThreads:2` on this i7-1255U, which has 2 performance
  cores, 10 cores and 12 threads. That is part of why S2-1 measured 254 s to first token
  (caveat 3).
  - Free memory first: `ollama stop`, close the desktop apps, `free -h`.
  - Time one fixed golden question at `num_thread` 2, 6 and 10, and record the
    time-to-first-token for each.
  - Set the fastest on the Ollama entries in `config.yaml`, with the numbers in a
    comment.
  - It must land **before** the tier-2 baseline. The llm spec is hashed into the
    fingerprint's `query` part, so adding it afterwards marks the baseline stale and
    forces another multi-hour run. A different thread count can also change the
    floating-point sums, and so, rarely, an answer.
  - A small change on its own: config plus a measured comment. It can also ride S2-3. S2-2
    closed without it.
- **S2-6 —** the ingestion manifest records the embedder's identity and dimension;
  refuse to open an index built with a different embedder (DEC-17).
- **Phase 3 —** NFR-2 (<2 s first token) is unachievable CPU-only: revise the SLO
  or plan GPU serving in the Phase 3 arch pass. Phase 2 measures it: `ttft_ms` on
  every answer (DEC-14) and in every tier-2 run.
- **S2-4, then a follow-up after S2-5 —** CLI "Sources" lists what was retrieved, not
  what the answer used, so a refusal still shows a source. S2-4 adds the reranker's
  `min_score` knob (DEC-16), defaulting to off. Tuning it with the eval harness comes
  after the tier-2 baseline exists.
- **S2-7 —** `ChatOllama` leaves its HTTP client open
  (`ResourceWarning: unclosed socket` at exit). S2-1 closed the CLI half: `aclose_llm`,
  called at exit. S2-7's API lifespan owns the client and closes it at shutdown.
- **DEC-5 exit tasks, one per phase:**
  - own loaders (**Phase 1 → S1-3**);
  - the cross-encoder via `sentence-transformers` (**Phase 2 → S2-4**);
  - dropping `langchain-community` and `langchain-classic` from `pyproject.toml`
    after the Qdrant move (Phase 3).

  Note: ragas pulls both into the `eval` extra, so Phase 3 removes them from our
  direct dependencies only.
- **Phase 3 — statelessness (NFR-5).** The Phase 2 API keeps ingest jobs in memory
  and the index in process (DEC-18). A horizontally scalable query tier needs Qdrant
  and a durable job queue.
- **Phase 3 — API deployment hardening:** rate limiting, TLS and multi-user auth.
  DEC-18 is localhost-first, with an optional bearer token.
- **After S2-5 — the DEC-2 revisit.** Run mistral vs phi3 (and `qwen2:7b`, if pulled)
  through the tier-2 harness. It is hours-scale, so it is a Colab/Kaggle sweep. Its
  result feeds the Phase 3 serving decision.
- Phase 2 was sharded on 2026-10-02 (S2-1 … S2-9). Phase 3 is sharded at the Phase 2
  exit, via WORKFLOW.md Step 2.

## Done

**Phase 0 — shipped as `v0.1` (`ae15a23`, 2026-09-25).** S0-1 repo hygiene ·
S0-2 uv project · S0-3 single package · S0-4 config repair + corpus rename ·
S0-5 LangChain 1.x LCEL chain with streaming · S0-7 docs layout · S0-6
end-to-end proof. Per-story detail in `stories/phase-0/`; the Phase 0 table
above is the index.

**Phase 1 — shipped as `v0.2` (`db75439`, 2026-10-02).**
S1-1 frozen Pydantic config · S1-2 `_target_` allowlist + CI · S1-3 own loaders,
recursive walk · S1-4 error handling, exit codes, real CLI · S1-5 Phase-0
regression suite + coverage gate · S1-6 types and lint · S1-7 CPU/CUDA torch
variants · S1-8 audit gate, doc hygiene, release. As delivered:
ARCHITECTURE.md §1.8; per-story detail in `stories/phase-1/`.

**Architecture passes.**

| Phase | Confirmed | Decisions | Sharded into |
| --- | --- | --- | --- |
| 0 | 2026-09-18 | DEC-1, DEC-2, DEC-4 (later DEC-5) | — |
| 1 | 2026-09-25 | DEC-6 … DEC-12 | S1-1 … S1-8 |
| 2 | 2026-10-02 | DEC-14 … DEC-18 | S2-1 … S2-9 |

The Phase 2 pass ran on Opus 5.5 by the maintainer's choice; ADR-018 routes
architecture passes to Fable.
