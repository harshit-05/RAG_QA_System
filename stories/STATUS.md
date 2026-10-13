# Board

> Updated at every story close-out. This file + the story files ARE the
> project memory across sessions.

## Now

**Next action: commit [S2-8](phase-2/S2-8-cli-polish.md), the CLI polish batch, open its
PR, and run the second review.** It is implemented on `fix/s2-8-cli-polish` and not yet
committed (2026-10-13). The first review is done.

- **Shown in the story:** Verification 1–5. The suite gave 887 passed, 5 xfailed,
  coverage 98.31%, after the first review's fixes. `rag-eval check` passed, so caveat 24
  held. Verification 3 passed by the maintainer's hand: arrow keys, history, and Ctrl-C
  while loading.
- **Owed after the commit:** the PR, for CI (Verification 6).
- **`preflight` and `OLLAMA_HINT` moved** from `api/server.py` to `components` (maintainer),
  so `rag-query` shares `rag-serve`'s start-up check without importing the `api` package.
  ARCHITECTURE.md §2.2 has a dated note, and an architecture test holds the rule.
- **New main dependency:** `requests`, declared rather than transitive. Its locked version
  did not change.
- **Breaking (exit code):** an unreadable corpus root makes `rag-ingest` exit **2**, where
  it exited 1, and an API ingest job reports the same `exit_code`. Its summary also drops
  the zero-count lines. The `rag-ingest` commit carries `!` and a `BREAKING-CHANGE:`
  trailer.
- **First review (2026-10-13): no blockers.** It re-ran the gates and Verification 2, 4 and
  5, and drove the prompt in a pseudo-terminal. Two code findings are fixed, test first: an
  `OSError` from the index check was blamed on the Hugging Face cache, and a read timeout
  got "Is Ollama running?". The suite is now 887 passed.

**S2-7 is done** (2026-10-12, PR #15, merged as `1ef599f`; on Opus 5.5 in place of
Fable).

- CI was green on the PR's head, `2eff33f` (run 38260306611), and on `main` after the
  merge (run 38262357171). Verification 1–6 are shown in the story.
- Two reviews, no blockers; each fixed one code and one doc finding on the PR.
- **Two commits in the middle are red:** `f12ef97` and `badb2cd` carry tests whose fix
  came in `4c37911`. They were merged as is, by the maintainer's choice; a `git bisect`
  should skip them.
- The second review left one decision open, now a backlog line: a back-off after a failed
  index reopen.

**Second review of S2-7 (2026-10-12): no blockers.** It re-ran the gates on the working
tree (848 passed), the uvicorn tests five times on one core, and probed the reopen. Both
findings are fixed on the PR.

- **Code:** after a failed reopen, every request queued behind it tried again in turn:
  five concurrent refreshes made five opens, one after another, and `/v1/health` needs no
  token. Now a request that waited for an attempt takes its result. Reproduced with a test
  first. **Your call (now a backlog line):** a few seconds' back-off after a failure, so
  probes during an outage cost nothing (not built).
- **Docs:** README called `/docs` interactive, but a browser's "Try it out" POST sends
  `Origin` and gets 403. The row now says "to read".

**First review of S2-7 (2026-10-12): no blockers.** It re-ran the gates, the six uvicorn
tests on one core, and the auth by probe. Both findings are fixed on the PR.

- **Docs:** malformed JSON gets its 422 before the 403 and the 401, because FastAPI
  parses the body before any dependency. The docs said a 422 comes last. They now describe
  two 422s, and a test pins the order.
- **Code:** a reopen that failed with anything but `NoIndexError` (a disk error, a failing
  embedder) gave 500s on query, documents and health. Reproduced with a test first. The
  server is now not ready with a fixed message, logs the exception, and tries again on the
  next request.
- **Backlog, reworded:** the body-size gap is pre-auth, so a token does not guard it.

**What S2-7 delivered:** `rag-serve`, the HTTP API (FR-6, SRS §8.1, DEC-18).

- **What it serves.**
  - Answers stream as SSE: `sources`, `token`…, then `done` or `error`, with `: ping`
    keep-alives during the prefill.
  - Ingest runs as background jobs, one at a time.
  - There is a document listing, and two health probes.
  - The server starts without an index, and picks up any new generation, its own jobs' or
    a CLI's.
- **A disconnect stops generation.** Against real Ollama, our connection closed **0.03 s**
  after curl left, mid-stream and during the prefill alike.
  - Mid-stream, the runner idled within 1.1 s.
  - During the prefill, Ollama finished the prompt evaluation first (188 s), as caveat 21
    predicts.
- **The spike changed one mechanism.** FastAPI's native SSE runs the route's generator in
  a producer task of its own. A drop between events left the stream to Ollama open until
  garbage collection: 1 of 300 drops, and every time when forced. The slot's teardown now
  closes it, and a test forces that window. DEC-18 has the dated note.
- **Access.**
  - **Without a token:** loopback only, and a request whose `Host` is not loopback is
    refused.
  - **With `RAG_API_TOKEN`:** a bearer token, compared in constant time on bytes.
  - **Always:** `Origin` gives 403, a POST that is not JSON gives 422, and a request
    cannot name a component.
- **Breaking (config):** a truthy `trust_remote_code` anywhere in `components` no longer
  loads. The real config does not set it.
- **Caveat 24 held:** `rag-eval check` was clean; no hashed input moved.
- **The suite:** 845 passed, 5 xfailed, coverage 98.30%. New main dependencies: fastapi
  0.143.0, starlette 1.7.0, uvicorn 0.54.0, and opentelemetry-api 1.45.1 through FastAPI.

**S2-5b is done** (2026-10-12, PR #14, merged as `e88c6f2`; on Opus 5.5, where the story
says opus-fast). CI was green on its last commit, `edd933b`. Tier 2 is enforced: the
baseline is committed, its floors are set, and `rag-eval check` is the last step of CI's
`check` job. Every verification is shown in the story file.

- **Runs resume.** `generate` and `score` keep each finished item in a checkpoint beside
  their output (`<name>.partial.jsonl`), and the same command resumes. A `/code-review`
  found 9 issues: 7 were fixed, and 2 needed no change.
- **`num_thread` 10** is set on every Ollama llm. Time to first token went from 251 s to
  168 s, with identical answers.
- **The baseline:**
  - generated here in 69 min;
  - scored twice on a Colab T4, about 35 min each;
  - faithfulness 0.908, answer relevancy 0.719, context precision 0.920, context recall
    1.0, decline rate 0.6;
  - the floors are each minus 0.05.
- **The two scorings were identical**, so the measured noise was 0. A re-score on other
  hardware remains unmeasured (caveat 16).
- **The digest question is answered.** With Ollama 0.22.1 pinned, Colab's `ollama create`
  gives this host's `d85f8193c353`, so `check --with-ollama` passes here on a Colab-scored
  run.
- **The hand check found the judge lenient.** It scored two wrong answers as faithful (a
  made-up expansion, and numbers moved between a table's rows), and gave context recall
  1.0 four times where the contexts held only part of the reference. The reviews found
  the last two credits and the second answer. That is recorded, not fixed (backlog).
- **The maintainer kept the decline marker:** two honest refusals that paraphrase it count
  as misses (backlog).
- **From now on, CI fails on any edit to a hashed input until tier 2 is re-run** (caveat
  24). S2-7 and S2-8 should leave the prompt, the llm, retriever and reranker specs, the
  corpus and the golden set alone.
- **First review (2026-10-11): no blockers; 4 findings, all in the docs, fixed on the PR.**
  It re-ran verifications 2, 3 and 3b, recomputed the run files, and checked the
  `/code-review`'s fixes.
  - **A third context-recall overcredit, `glider-purpose`:** its LiveBench scores are only
    on p. 7, which was not retrieved. With all three corrected, context recall would be
    about 0.94, below its own floor.
  - **`decline_rate`'s floor allows no change.** One flip of its 5 items gives 0.4. It
    stays at 0.55, which is strict on purpose.
  - The `/code-review`'s two unfixed findings are named, and the model deviation is
    reworded.
  - **`context_recall` stays gated.** The judge's errors all leaned generous, so a fall it
    reports is likely real.
- **Second review (2026-10-12): no blockers; 2 findings.** It re-ran the gates (677
  passed), recomputed the committed run from its items, and attacked the checkpoint with
  crafted files. The resume code holds. CI was green on `8ba2d9f`.
  - **`glider-flask-vs-gpt4o` is a misattribution the judge scored 1 / 0.92 / 1 / 1.** The
    answer gives GLIDER GPT-4o's FLASK score, and GPT-4o its BigGen Bench one: the golden
    record's own trap, and S0-6's misattributed scores. With both faithfulness misses
    corrected, faithfulness would be about 0.83–0.84, below its floor. Fixed in the docs;
    no floor changes.
  - It scored high on every metric, so a hand check that picks low scores cannot find
    errors like it. Hand-labelling all 20 can (backlog).
  - **`check` trusts the run file's aggregates** rather than recomputing them from its
    items (backlog).

**S2-5 is done** (2026-10-07, PR #13, on Opus 5.5 in place of Fable). CI was green on
`b6f4e74`. Tier 2's harness is code: `rag-eval generate`, `score` and `check`, the
six-part fingerprint, and the prompt placeholder rules.

- **ragas 0.4.3 does not import on our stack.** It imports a module langchain-community
  0.4.2 removed (vibrantlabsai/ragas#2741). The maintainer chose a narrow import shim,
  which a test says when to delete.
- **The spike's recipe:** JSON mode, 0 parse failures; MiniLM for answer relevancy;
  `num_predict 2048`; and `rag-judge` serves 8192 tokens in 8.3 GB.
- **Tier 2 costs about 39–54 min per item on this CPU**, so a full local scoring is
  13–18 h. S2-5b scores on Colab/Kaggle by default.
- **Two reviews, 10 and 10 findings; 19 fixed, 1 deferred** (resumable runs, which S2-5b
  closed). They added `max_unscored` to the floors and an answers hash to the scores, and
  `score` now checks the served judge against its whole Modelfile.
- **The second reviewer (2026-10-07): no blockers, 3 findings, all fixed on the PR.**
  - instructor's version is pinned and hashed into `judge`, as ragas's is: it writes
    part of every judge prompt.
  - The served judge is checked both ways against its Modelfile and base: a leftover
    parameter, SYSTEM or TEMPLATE is refused.
  - `--with-ollama` no longer skips a generator's missing digest when the generator is
    the judge's base.
- **Breaking:** a config whose prompt breaks the placeholder rules no longer loads (the
  real one passes).

**S2-6 is done** (2026-10-06, PR #12, on Opus 5.5 in place of Fable). CI is green: the
runner reproduced tier 1 question by question. The first `eval-retrieval` attempt
stalled in ingest with no log at all, and its re-run passed in the usual time; the story
file has the detail. Ingestion is incremental, tracked by a sha256 manifest, and published by
an atomic symlink flip between generation folders, under a one-writer `flock` (DEC-17).

- **What it does:**
  - re-running embeds only what changed, and a run that changes nothing takes under a
    second;
  - a failed document keeps its chunks;
  - `rag-query` and `rag-eval` refuse a v0.2 index, or one from another embedder,
    before loading any model.
- **Tier 1 did not move:** 0.950, 0.7958 and 0.8583, question by question.
  `index.faiss` is byte-identical to v0.2's.
- **Breaking (caveat 20).** The real index was migrated here; the old one is kept as
  `vectorstore/db_faiss.v02-*`.
- **Maintainer decisions:** `multi_process` stays in the embedder identity, and a full
  rebuild leaves failed documents out.
- **Two reviews, 10 and 15 findings, all fixed.** Two of them could lose data: any
  folder at the store path was renamed aside, and recovery misread a link written as
  an absolute path. Detail in the story file.
- **A third review (2026-10-06, second reviewer): 3 findings, fixed on PR #12.**
  - Code that makes chunks is now versioned (`CHUNKING_VERSION`, pinned by a test).
  - A default spelled out in a `HuggingFaceEmbeddings` spec no longer counts as another
    embedder.
  - `rag-query` no longer says "re-run rag-ingest" for something `rag-ingest` refuses.

  No existing index rebuilds: the real one still reads "nothing changed".

**S2-4 is done** (2026-10-05, PR #11). Our own cross-encoder reranker is in, and tier 1
switched it **on**: 20 candidates reranked to 5.

- It lifts hit rate from 0.800 to 0.950, MRR from 0.5875 to 0.7958, and recall from
  0.7167 to 0.8583, at 1.4 s per answer on this CPU.
- The floors are re-baselined to 0.9, 0.745 and 0.808. GitHub's runner reproduced the
  reranked baseline question by question.
- The allowlist lost `langchain_classic.` and `langchain_community.`. A v0.2 config's
  refusal now says how to migrate, rather than leaving a wider allowlist as the apparent
  fix.
- CI's `eval-retrieval` job warms the cross-encoder up before saving the HF cache. Its
  cache key names both models, and an exact hit runs every model load offline.
- The second review gave S2-7 a requirement: a disconnected request's 1.4 s rerank must
  not overlap the next request's.

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

- [x] **Phase 2 (S2-5b): somewhere to run tier-2 scoring.** Done 2026-10-11: a Colab T4,
  with Ollama pinned to 0.22.1 and outputs on Google Drive. That is about 35 min per
  scoring, against 13–18 h on this CPU at 2 threads. The notebook's cells are in S2-5b's
  story file.

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
- Item 23 comes from S2-4.
- Item 24 comes from S2-5b.

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
     2.3 GB swap peak, so the weights were partly swapped out.
   - **S2-5b set `num_thread: 10` (2026-10-11).** With memory free and no swap, time to
     first token was 251 s at 2 threads, 187 s at 6 and 168 s at 10, with identical
     answers. So the thread count was the cost, not swap. Budget manual runs on about
     150–170 s to first token and 2.5–3 min per answer.
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
    `gemma2:9b` take about 3–4 h to score. **Measured in S2-5 (2026-10-07): 39–54 min per
    answerable item** on Ollama's 2 threads, so 13–18 h for the 20; the judge holds 8.3 GB.
    - **Measured in S2-5b (2026-10-11):**
      - generating the 25 answers took **69 min** here at `num_thread` 10;
      - one scoring took **about 35 min on a Colab T4**, and two identical scorings
        showed no judge noise on that setup.
    - **Re-score on the same setup:** a Colab T4, Ollama pinned to 0.22.1, and the
      notebook in S2-5b's story. The floors' 0.05 tolerance was set with zero measured
      noise. A score from other hardware (this CPU, another GPU) may differ by an amount
      no one has measured.
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
23. **A cached model hides a test that is not hermetic.** This host's HF cache holds both
    models, MiniLM and S2-4's cross-encoder. A test that loads a real one therefore passes
    here, offline included, and fails only in CI's `check` job. Check the suite the way
    that job sees the world, with
    `HF_HOME=<an empty dir> HF_HUB_OFFLINE=1 uv run pytest`, and see that the folder
    stays empty.
24. **An edit to a hashed input fails CI until tier 2 is re-run** (S2-5b, DEC-15). The
    `check` job ends with `rag-eval check`, which fails on a stale committed run.
    - **A re-run of generate and score (about 70 min here, then about 70 min on a T4):**
      the prompt, an llm, retriever or reranker spec, `reranker_candidates`, a corpus file,
      the embedder, the splitter, the loader map or a golden question.
    - **A re-run of score only:** a `ground_truth`, `decline_marker` or the judge's
      Modelfile.
    - **Free:** `notes`, `expected_sources`, comments, and the runtime-only llm keys.

    So a story that touches one of these budgets the re-run, and the error names which
    re-run. Code changes are not hashed: after one that changes answers, re-run tier 2 by
    hand.

## GPU offload notes (nothing *requires* GPU; Colab/Kaggle available)

- **Phase 0–1: no GPU work at all.**
- **Phase 2 — tier-2 RAGAs scoring (DEC-15)**: with a local CPU judge, a full
  metric sweep is an hours-scale run.
  - `rag-eval generate` runs here.
  - `rag-eval score` runs on Colab/Kaggle, with Ollama and `gemma2:9b` on their GPU.
  - The two exchange only `eval/runs/answers-latest.json`.
  - S2-5b's story records the recipe as run: a T4, Ollama 0.22.1, outputs on Drive, and
    about 35 min per scoring.
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
| [S2-4](phase-2/S2-4-reranker.md) | Our own cross-encoder reranker, decided by the numbers | FR-4, ISS-03, DEC-5 step 2 | S2-1, S2-3 | Done 2026-10-05 (PR #11); reranker on |
| [S2-5](phase-2/S2-5-ragas-gate.md) | Tier-2 RAGAs harness and the freshness gate (code only) | FR-7 (harness), ISS-15 | S2-2, S2-4, S2-6 | Done 2026-10-07 (PR #13) |
| [S2-5b](phase-2/S2-5b-tier2-baseline.md) | Tier-2 baseline run, floors, and the CI check | FR-7 (enforced) | S2-5 | Done 2026-10-12 (PR #14) |
| [S2-6](phase-2/S2-6-ingest-manifest.md) | Incremental ingestion with a hash manifest | FR-2, ISS-14, NFR-4 (part), SRS §7.3 | S2-1 | Done 2026-10-06 (PR #12) |
| [S2-7](phase-2/S2-7-http-api.md) | FastAPI service with SSE streaming | FR-6, SRS §8.1 | S2-1, S2-6 | Done 2026-10-12 (PR #15) |
| [S2-8](phase-2/S2-8-cli-polish.md) | CLI polish batch | backlog: CLI polish | S2-1, S2-6 | In review 2026-10-13 |
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
| DEC-7 | Where the `_target_` allowlist lives, and whether it is configurable | **Resolved 2026-09-25: hardcoded, inside `import_from_string`** | Enforcing at the import funnel rather than in `build_object` covers `ingest.py`'s direct call — the gap ADR-015 flagged. Not config-overridable and no env escape hatch: an allowlist the config can edit is not an allowlist. `langchain_classic.` was allowed as a *config* prefix (the disabled reranker entry); ADR-007 governs *source* imports and is unaffected. **S2-4 (2026-10-04) removed `langchain_classic.` and `langchain_community.`** from the list: our own reranker replaced that entry (DEC-16), so no `_target_` names either. A v0.2 `config.yaml` no longer loads until its old `cross_encoder` entry is deleted. Load string-checks prefixes (nested too) and never imports; a separate `check_imports` helper, used by tests, catches misspelled classes (ISS-03). S1-2. Closes ISS-04. |
| DEC-8 | DEC-5 step 1: replace the `langchain-community` loaders now or later | **Resolved 2026-09-25: now, in Phase 1** | Own `PdfLoader`/`DocxLoader`/`TextLoader` on `pypdf`/`docx2txt`; `extensions` leaves the component entries for `pipeline.ingestion.loaders`, giving one instantiation path; discovery becomes recursive (ISS-13). Afterwards `langchain_community` is imported only by `vectorstore.py`. Risk is citation metadata, so the story records the `v0.1` chunk-count + citation baseline before changing anything and must reproduce it. S1-3. |
| DEC-9 | NFR-6 (timeout / retry / circuit-breaking): Phase 1 or later | **Resolved 2026-09-25: Phase 1 does error handling only** | ISS-05 (collect failures, exit non-zero — NFR-7) and ISS-06 (REPL try/except) are S1-4. Timeout, retry-with-backoff and circuit-breaking (SRS §11 Resilience) need the service shape and move to Phase 3 with the serving decision. Deferred deliberately, not by omission. |
| DEC-10 | Device selection: collapse the `_cpu`/`_cuda` embedder entries into one env-driven setting | **Resolved 2026-09-25: no — the entries stay explicit** | The backlog item assumed pydantic-settings interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside YAML values. **It does not** — there is no `${VAR:-default}` expansion for arbitrary values. With the CUDA torch variant (DEC-12), one sync command plus repointing `pipeline.ingestion.embedder` is already a one-line switch, so the four entries restored in `ebcfc2e` stay as FR-1 axes. |
| DEC-11 | CI: host, hermeticity and how hard the gates bite | **Resolved 2026-09-25: GitHub Actions, hermetic suite, blocking gates** | No Ollama, no model download, no network in tests: `DeterministicFakeEmbedding` (verified present in core 1.6.3) + a fake chat model. Gate order: `ruff check` → `mypy` (`disallow_untyped_defs` on `src/rag_qa`) → `pytest --cov-fail-under=80` (omitting `evaluate.py`, Phase 2 + `eval` extra) → `pip-audit`. Each gate blocks from the story that adds it. **No `ruff format --check` in Phase 1** — the tree is unformatted and a whole-tree reformat is the diff that gets rubber-stamped → backlog. |
| DEC-12 | Making the declared GPU embedder entries actually installable | **Resolved 2026-09-25: uv-conflicting CPU/CUDA torch variants; plain `uv sync` stays CPU** | Each variant routed to its own index, `explicit = true` on both. **Mechanism chosen by S1-7's spike**: extras have no default, so torch-only-in-extras would make a plain sync pull PyPI CUDA torch via `sentence-transformers`. Preferred: dependency groups + `default-groups = ["dev", "cpu"]`; fallback: extras with `--extra cpu` on every install path. S1-7 re-runs the S0-2 smoke test on the **installed env** (the lock legitimately holds both variants): core 1.x, `torch 2.14.0+cpu`, zero `nvidia-*`. CUDA is verifiable here as a **resolve only** — this host is CPU-only. |
| DEC-13 | Partial ingestion failure: replace the index with what loaded, or keep the previous one | **Resolved 2026-10-01: replace, exit 1** | Maintainer's call, raised by S1-4's third review. Some documents or folders unreadable → index the rest, save over the old index, exit 1 ("rebuilt without them"). Keeping the old index is safer when unattended, but one persistently bad file would block every update. Phase 1 ingests are hand-run and watched. Phase 2's hash manifest dissolves it: a failed document keeps its previously indexed chunks. Exit codes: 0 all indexed · 1 document/folder unreadable, nothing indexed, or unexpected error (traceback) · 2 cannot start. ARCHITECTURE.md §1.1. **Superseded in part by DEC-17 (S2-6, 2026-10-05):** in an update a failed document keeps its chunks, and exit 1 reads "The index was updated; the failed documents kept their previous chunks, if they had any"; a full rebuild still leaves them out ("rebuilt without them", maintainer). Exit 2 also covers another ingestion holding the lock, or something at the index path that is not an index. |
| DEC-14 | Cancelling an answer: how the CLI's Ctrl-C and the API's client disconnect stop generation | **Resolved 2026-10-02: one answer-event stream (`answering.stream_answer`); stopping it means cancelling the task that consumes it** | **Verified trap:** async alone does not fix it. Under `RunnablePassthrough.assign`, `RunnableParallel` waits on its step tasks with `asyncio.wait` and never cancels them. In the trial, a cancel at 0.5 s during a simulated prefill left the stream open until 2.0 s; streaming `prompt \| llm \| parser` directly closed it at 0.5 s. **Design:** the CLI uses one `asyncio.Runner` per session. `QueryPipeline` holds the parts, and `build_rag_chain` stays for `invoke`. `aclose_llm` closes ChatOllama's clients, and `ttft_ms` is measured for NFR-2. S2-1. ARCHITECTURE.md §2.1. |
| DEC-15 | Phase 2 quality gate: what runs in CI when RAGAs on CPU takes hours | **Resolved 2026-10-02: two tiers** (maintainer) | **Tier 1:** retrieval hit rate, MRR and recall against `expected_sources`, recomputed on every pull request and every push to `main` by a CI job that caches the HF models. That job is CI's only network use. **Tier 2:** RAGAs with a `gemma2:9b` judge through Ollama's OpenAI endpoint, run offline (scoring can go to Colab/Kaggle) and committed with a fingerprint. `rag-eval check` fails CI on a floor breach **or a stale run** (maintainer: fail, not warn). Rejected: a hosted judge in CI; offline-only. **Second review:** the fingerprint has six parts (`query` with a rendered-prompt probe, `ingestion`, `corpus`, `questions`, `references`, `judge`) in `evaluation/fingerprint.py`; Ollama digests are recorded, not hashed; the judge is `rag-judge` from `eval/judge.Modelfile` (8k context, caveat 22); the tier-2 tolerance comes from two scorings. S2-2, S2-3, S2-5, S2-5b. |
| DEC-16 | Reranker: `langchain_classic`'s `ContextualCompressionRetriever`, or our own | **Resolved 2026-10-02: our own `CrossEncoderReranker` on `sentence-transformers`** (DEC-5 step 2) | A `langchain_core` `BaseDocumentCompressor` that writes `rerank_score` into metadata. k=20 candidates are cut to `top_n` 5, and tier-1 numbers decide whether it is on. `langchain_classic.` and `langchain_community.` leave `ALLOWED_PREFIXES`. The `min_score` knob is for the Sources-relevance backlog line. S2-4. **Outcome (S2-4, 2026-10-04): on.** Hit rate 0.950, MRR 0.7958, recall 0.8583, against dense 0.800, 0.5875, 0.7167; 8 questions won, 4 lost (still hits), 1.4 s per answer on CPU. |
| DEC-17 | Incremental ingestion: change detection, failure handling, write safety | **Resolved 2026-10-02: a sha256 manifest beside the index, generation directories published by an atomic symlink flip, one writer** (the flip replaced a two-rename swap in the second review) | **Updates:** only added and changed documents are re-embedded. A failed document keeps its chunks, which dissolves DEC-13. **Full rebuild** when the embedder identity or the splitter changes. Device keys are left out of the identity, so Colab's `_cuda` equals local `_cpu`. **IDs and metadata:** uuid5 chunk IDs, ready for Qdrant (DEC-3). `source` becomes corpus-relative, plus `source_sha256` and `ingested_at` (SRS §7.3). **Query side:** `rag-query` refuses a mismatched index. **Breaking:** re-ingest once. S2-6. **As implemented (2026-10-05):** `multi_process` stays in the identity, since langchain-huggingface 1.2.2's multi-process path drops `encode_kwargs` (maintainer). An empty `model_kwargs` counts as none. Generation stamps carry microseconds and always sort after the live one. Only a folder holding exactly `index.faiss` and `index.pkl` is migrated as v0.2, and anything else at the path exits 2. Every update opens the live index, and an update probes the embedder's dimension. ARCHITECTURE.md DEC-17 has the dated notes. |
| DEC-18 | HTTP API shape: SSE library, auth, concurrency, ingest jobs | **Resolved 2026-10-02: FastAPI ≥ 0.135 native SSE; a token is required off localhost (maintainer); one generation at a time** | **Surface:** `rag-serve`. `QueryRequest` is `{question}` with `extra="forbid"`, so the API never supplies components, and a `trust_remote_code` load guard comes with it. **Load:** 503 with `Retry-After` when busy. A disconnect cancels generation: under uvicorn (ASGI 2.3) Starlette cancels the stream, and `stream_answer` runs inside it, with no producer task (second review). Requests with an `Origin` header are refused. **Ingest:** single-flight jobs in memory, which swap only the retrieval half. **Privacy:** no host paths over HTTP. S2-7. **As built (S2-7, 2026-10-12):** FastAPI's native SSE runs the generator in its own producer task. The cancel still reaches `stream_answer`, and the slot's teardown closes a stream that a drop between events left paused (the spike's window). Retrieval is serialized by a lock, not the slot. The API opens the index itself (`chain.open_retrieval`), and `require_index` is gone. `rag-serve` checks Ollama before uvicorn starts. ARCHITECTURE.md DEC-18 has the dated notes. |

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
- **S2-4:** the reranker (FR-4, ISS-03, DEC-5 step 2). `rag_qa.rerankers` holds our own
  `CrossEncoderReranker` on `sentence-transformers`, and tier 1's numbers switched it on.
  - S2-3's inputs were all used: the reach at k=20, the scratch configs' flags, a fresh
    scratch index, and the warm-up between ingest and the cache save.
  - Declaring `langchain-classic` closes as not needed: no `_target_` names it, and the
    allowlist refuses it.
  - The Sources-relevance line gets its knob, `min_score`, off by default. Its tuning
    stays below, after S2-5.

- **S2-5 (PR #13):** tier 2's harness (FR-7, ISS-15): `rag-eval generate`, `score` and
  `check`, the six-part fingerprint, and the prompt placeholder rules. It closes two lines
  below: the placeholder check, and the judge pointed at local Ollama.
- **S2-5b (PR #14):** tier 2 enforced (FR-7). It closes three lines below:
  - resumable runs;
  - `num_thread`, now 10;
  - the judge's `num_thread`, not set: the baseline scored on a GPU.

  It opens three follow-ups: two from its hand check and its answers, and one from its
  second review (`check` recomputing the aggregates).
- **S2-7 (PR #15):** the HTTP API (FR-6, SRS §8.1, DEC-18). It closes three lines below:
  requests never supply components (with the `trust_remote_code` guard), a disconnect
  stays a cancel, and ChatOllama's client closed at shutdown (the API half).
  - It opens two: request size and health-check load (Phase 3), and the `eval` extra's
    langchain advisory.
  - S2-8 can reuse its start-up check for `rag-query`.
  - S2-9 owes README's status line, §2.2's module list and §2.8.
- **S2-6:** incremental ingestion (FR-2, ISS-14, NFR-4 in place, SRS §7.3, DEC-17).
  - Three backlog lines close with it: walk pruning, the §7.3 chunk metadata, and the
    embedder check on open.
  - S2-8 gains two items: a text file's summary line reads "1 pages/sections", and the
    summary now has more zero-count lines.
  - S2-9 owes README's status line, which still lists incremental ingestion as
    planned.

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
  PR red, docs-only ones included. **First seen 2026-10-07 on S2-5's PR:** multidict
  6.9.0 (CVE-2026-104874). A fixed release existed, so a one-package lock bump was the
  fix and no policy was needed; the question stays open for an advisory with no fix. Decide an `--ignore-vuln` policy (ID + reason in
  `ci.yml`) or a scheduled audit beside a PR gate for new dependencies. Not needed
  until it first happens.
- **Closed by S2-5 (2026-10-07) — prompt placeholder check (FR-8 follow-up, found in S1-1).**
  `Prompt` now refuses a `human` turn without `{context}` or `{question}`, `{context}` in
  `system`, and any other placeholder; the history below is kept. A
  `human` prompt missing `{context}` silently answers without retrieval. One
  validator on `Prompt`. S1-5 left it open, as the board allowed. It belongs with
  the Phase 2 evaluation harness, which is what would catch an ungrounded
  answer anyway. Its rules are in ARCHITECTURE.md §2.3.
- **Closed by S2-7 (2026-10-12) — the API must never let a request supply or override
  components (S1-2 second review).** `QueryRequest` is `{question}` with `extra="forbid"`,
  and config load refuses a truthy `trust_remote_code` anywhere in `components`. The
  history below is kept. A constraint for the Phase 2 architecture pass, with
  the optional load-time guard that rejects `trust_remote_code` anywhere in the
  config. Re-deferred because it only matters once the config is reachable from
  outside the host. Designed in DEC-18: `QueryRequest` forbids extra keys, and the
  `trust_remote_code` guard lands in the same story.
- **S2-8 — CLI polish (found in S1-4).** No single item is worth a story, so they are
  batched as S2-8. It runs after S2-6, which rewrites `ingest.py`:
  - `rag-query` startup is unguarded: Ollama being down
    (`validate_model_on_init`) and Ctrl-C while models load both print a
    traceback. One startup try/except with a clean message. Since S2-4 it also
    covers the reranker's model missing from the HF cache on an offline machine:
    "run once with network" (S2-4, Discovered). S2-7 built this for `rag-serve`:
    `rag_qa.api.server.preflight` and `OLLAMA_HINT`. Reuse them.
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
    the clean error. Quiet the `pypdf` logger to ERROR in `ingest.py`. Still
    seen in S2-6's Verification 3.
  - **The summary after S2-6** (found in S2-6):
    - a text file reads "1 pages/sections";
    - an update prints a "Documents:" line with up to five zero counts. The
      zero-count rule above covers it.
- **Backlog — two more plural slips in `rag-ingest`'s log** (found in S2-8): "Embedding 1
  chunks..." and "Published …: 1 documents, 1 chunks.". S2-8 fixed the ones on its list:
  "1 page/section", and the summary's "1 chunk".
- **Backlog — the API's ingest jobs print pypdf's warnings** to `rag-serve`'s stderr (found
  in S2-8). `rag-ingest` sets the `pypdf` logger to ERROR for its own run only, as it does
  its log handler.
- **Closed by S2-6 (2026-10-05) — the walk descends into ignored trees** (found in S1-3 and S1-4). Hidden folders are pruned in place now; the history below is kept.
  It walks all of `.git` before discarding it. Performance only. Phase 2's
  incremental ingestion rewrites discovery around the manifest, so prune there
  (`Path.walk` is top-down, and pruning `folders[:]` in place fixes it).
- **Closed by S2-7 (2026-10-12) — keep a client disconnect a cancel (found in S2-1).** It
  is a cancel, through FastAPI's own SSE producer task. The slot's teardown closes a
  stream that a drop between events left paused; the spike found that window. Against
  real Ollama, our connection closed 0.03 s after the client left. The history below is
  kept. Closing `stream_answer`
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
- **Closed by S2-5b (2026-10-11) —** tier-2 runs resume: `generate` and `score` keep each
  finished item in `<name>.partial.jsonl`, and the same command skips them
  (`evaluation/checkpoint.py`).
- **After S2-5b — what the baseline's answers and hand check found.** Each fix is a change
  to measure with this harness (a re-run, caveat 24), not to make by eye.
  - **Two refusals paraphrase past the decline marker:** "I could not find information
    about…" (`unanswerable-yolo-dataset`, `unanswerable-cade8-venue`). That puts
    `decline_rate` at 0.6. The maintainer kept the marker. Tighten the prompt's refusal
    instruction, or revisit the rule.
  - **Generation misses:**
    - `glider-name` refuses although its title chunk is in context;
    - `glider-slm` makes up "Summarization as Judge Model" (tier 1's one miss: page 2 is
      not retrieved);
    - `glider-flask-vs-gpt4o` gives GLIDER GPT-4o's FLASK score, 0.610, and GPT-4o its
      BigGen Bench one, 0.614 (second review). GLIDER's own row is in no retrieved chunk.

    The prompt, `min_score` and the DEC-2 revisit are the levers.
  - **The judge is lenient.**
    - It scored two wrong answers as faithful (1.0): `glider-slm`'s fabrication, and
      `glider-flask-vs-gpt4o`'s misattribution (found in the second review).
    - It gave context recall 1.0 on four items whose contexts hold only part of the
      reference: `glider-slm`, `lusk-overbeek-itp`, `glider-purpose` (first review) and
      `glider-flask-vs-gpt4o` (second review).
    - It got 3 of the 20 hand-checked verdicts wrong, all too generous. So the baseline
      runs high, context recall sits at its ceiling, and the gate is a quieter alarm than
      its numbers suggest. The story file has the assessment.
    - **First, measure it:** hand-label faithfulness for the 20 answers (one to two hours
      of reading) and compute the judge's agreement. No re-run needed. It is the only way
      to find errors among the perfect scores: `glider-flask-vs-gpt4o` scored 1 / 0.92 /
      1 / 1.
    - **Then, if the agreement is poor:** a stronger judge (a 14B model at 4-bit should fit
      on a T4, not yet measured), or a second judge for spot checks, in the DEC-2 revisit or
      Phase 3. Either moves `judge` and needs a re-score.
  - **Answer relevancy is all-or-nothing on hedging.** ragas zeroes a "noncommittal"
    answer, so one hedge in an otherwise right answer costs about 0.04 of the mean.
- **Backlog — the `eval` extra's `langchain` 1.3.2 has PYSEC-2026-2192** (fixed in 1.3.9;
  found in S2-7's audit). ragas pulls it in. CI never installs the extra, so its audit
  never sees it. Bump it with the next ragas change (which re-scores), or decide whether
  the extra should be audited at all (the audit-gate policy line above).
- **Backlog — `rag-eval check` recomputes the aggregates from the items** (S2-5b's second
  review). It reads `aggregate` and `failures` from `generation-latest.json` as written.
  `score` writes them consistently, and the baseline recomputes exactly. But a hand edit,
  or a merge conflict in that file resolved by hand, could pass a gate its items fail. A
  few lines in `run_check`: refuse a file whose aggregates disagree with its items.
- **When a ragas release stops importing `langchain_community.chat_models.vertexai` —
  delete the import shim** in `ragas_scoring.import_ragas` (S2-5,
  vibrantlabsai/ragas#2741). `tests/test_eval_scoring.py` fails then and says so. A
  ragas upgrade also means bumping `fingerprint.RAGAS_VERSION`, which `uv.lock`'s test
  enforces, and re-scoring.
- **Closed by S2-5b (2026-10-11) —** the judge's `num_thread`: not set. The baseline scored
  on a Colab T4, where every layer runs on the GPU. A local CPU re-score would add
  `PARAMETER num_thread 10` to `eval/judge.Modelfile`, which moves `judge`.
- **Flaky under load — two Ctrl-C tests in `tests/test_cli.py`** (found in S2-5's second
  review). `test_ctrl_c_while_a_chunk_is_processed_leaves_no_stream_open_at_the_prompt`
  and `test_two_ctrl_cs_during_an_answer_end_the_session` failed once in a full run
  beside other `uv` commands, then passed in three full runs. They are timing-based. If
  either fails on CI, widen its waits or switch to event-driven synchronisation.
- **If it recurs — make a stalled CI step diagnosable** (found in S2-6's CI).
  - **The incident:** `eval-retrieval` stalled in "Ingest the corpus" until the job's
    20-minute limit, and GitHub kept no log. It was not reproducible here, and the
    re-run passed in the usual time.
  - **The remedy, if it happens again:** a step-level `timeout-minutes` on the ingest
    step, so the step fails and the job still uploads its log, and the ingest run as
    `timeout -s ABRT 540 python -X faulthandler -m rag_qa.ingest`, so Python dumps
    where every thread was.
  - It needs gh's `workflow` scope to push (caveat 13).
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

- **Closed by S2-6 (2026-10-05) —** chunk metadata: content hash of the source file + ingestion timestamp
  (SRS §7.3), with the manifest that needs them (DEC-17).
- **Closed by S2-5 (2026-10-07) —** `rag-eval score` judges with `rag-judge` through
  Ollama's OpenAI endpoint (`evaluation.judge` in `config.yaml`). As found: the RAGAs
  judge must be pointed at local Ollama explicitly (its default
  is OpenAI). DEC-15 settles both halves: the judge is `gemma2:9b` through Ollama's
  OpenAI endpoint, and a full sweep on CPU takes hours, so scoring offloads to
  Colab/Kaggle.
- **Closed by S2-5b (2026-10-11) —** Ollama's `num_thread` (found in S2-1's second review).
  It is now 10 on every Ollama llm, set before the baseline. The measured table is in
  `config.yaml` and caveat 3.
- **Closed by S2-6 (2026-10-05) —** the ingestion manifest records the embedder's identity and dimension;
  refuse to open an index built with a different embedder (DEC-17).
- **Phase 3 —** NFR-2 (<2 s first token) is unachievable CPU-only: revise the SLO
  or plan GPU serving in the Phase 3 arch pass. Phase 2 measures it: `ttft_ms` on
  every answer (DEC-14) and in every tier-2 run. Since S2-4 the rerank sits before the
  first token: about 1.4 s of every `ttft_ms` on this CPU, plus 2.5 s of model loading
  at startup.
- **After S2-5 — tune the reranker's `min_score`.** CLI "Sources" lists what was
  retrieved, not what the answer used, so a refusal still shows a source. S2-4 added the
  knob (DEC-16), off by default. It is in the model's own units: logits, from -6.42 to
  8.38 over the kept chunks of the golden questions. Tuning it with the eval harness
  comes after the tier-2 baseline exists.
- **Closed by S2-7 (2026-10-12) —** `ChatOllama` leaves its HTTP client open
  (`ResourceWarning: unclosed socket` at exit). S2-1 closed the CLI half: `aclose_llm`,
  called at exit. The API's lifespan owns the client and closes it at shutdown.
- **Maintainer's call — a back-off after a failed index reopen** (S2-7's second review).
  - **Now:** after a reopen fails, each later request tries again once. Concurrent requests
    share one attempt.
  - **The proposal:** wait a few seconds after a failure, so probes arriving one by one
    during an outage (an unreadable disk, say) cost nothing.
  - `/v1/health` refreshes without a token. Not built.
- **Phase 3 — request size and health-check load** (found in S2-7).
  - FastAPI reads a request body whole before validating it, so a huge body costs memory
    before the 422.
  - `/v1/health` is unauthenticated and asks Ollama on every call.

  Both belong with the rate limiting.
- **DEC-5 exit tasks, one per phase:**
  - own loaders (**Phase 1 → S1-3**);
  - the cross-encoder via `sentence-transformers` (**Phase 2 → S2-4**, done 2026-10-05);
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
