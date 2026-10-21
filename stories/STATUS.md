# Board

> Updated at every story close-out. This file + the story files ARE the
> project memory across sessions.

## Now

**Next action: S3-5, plan-first, on `feat/s3-5-hybrid-retrieval`.**

**S3-4 is merged** (2026-10-21, PR #22, merge commit `54944d4`). CI was green on the PR's
last head, `be44453` (run 39589681301), and is green on `main` after the merge (run
39590870310). The merge also cleared `main`'s red audit, through the certifi bump
(`a27d98a`). The live index is unchanged: the `rag_qa` alias names
`rag_qa--gen-20261019T153834837765Z-7f20891e`, and S3-2's parity baseline is still
`~/dev/rag-s3-2/contexts-qdrant.json` (sha256 `5a36e2f9…a074cbc`), which S3-4 reproduced
IDENTICAL.

**`main` is protected** by ruleset 25561360 (applied 2026-10-21, ahead of S3-9).

- **What it enforces:** no deletion and no force-push; a pull request for every change, with
  0 approvals; and `check` and `eval-retrieval` from GitHub Actions (integration 15368)
  passing, on a branch up to date with `main`. Squash and rebase merges are disabled in the
  repository settings, so a merge commit is the only method.
- **§11's rule that a red stage blocks a merge now holds** for the two existing jobs. S3-9
  adds `deploy-config` and `container` to this ruleset, by a PUT, and never creates a second
  one.
- **Records go on the story's branch.** `main` takes no direct push, so a close-out after
  the merge (S1-8's and S2-3's precedent) needs a pull request of its own. Record the merge
  of one story on the next story's branch, as S3-5 does for S3-4.
- **If a merge asks for an approval,** the likely cause is a pull-request default that
  GitHub filled in without documenting it, `require_extra_approval_for_unattributed_changes:
  true`. Every commit so far is attributed to `harshit-05`. Set it to false on ruleset
  25561360.

**S3-4 was done before its merge** (2026-10-21). CI was green on the second review's record,
`e0ee166` (run 39588185690, both jobs): the suite 1068 passed, coverage 98.40%, the audit
clean, tier 2 fresh, tier 1 0.950 / 0.796 / 0.858, and the 8 Qdrant server tests. Neither
review found a defect in the code.

**Second review of S3-4 (2026-10-21, on `fbf3d8d`): no defect in the code; 1 record
finding and 1 config comment, fixed after it.** The live `rag_qa` alias was only read;
probes ran on a throwaway Qdrant v1.19.2, removed afterwards. Detail in the story.

- **New evidence: an update equals a rebuild after every S3-4 transition,** on the real
  server: the markdown splitter changed, the default changed, the limit lowered below the
  PDF, the limit raised with the override dropped, and an override with the default's spec.
  At each, the same point IDs, texts and metadata as a fresh `--rebuild`, vectors within
  1.2e-7, and equal manifest records.
- **Held:** the gates on `fbf3d8d` (1068 passed, 98.40%; `rag-eval check` exit 0; tier 1
  IDENTICAL, `5a36e2f9…`); the S3-2-written live manifest under S3-4's rules (3 unchanged,
  no full rebuild); the identity rules read in the code; the Discovered claims against
  langchain-text-splitters 1.1.2 and the real functions.
- **Fixed:** this file and the story still named `d1e747f` as the head and the first
  review's record as uncommitted. And `config.yaml` now warns against `keep_separator:
  false` on the markdown entry: with regex separators, the library joins merged pieces with
  the pattern, `"#{1,6} "`, in the chunk text. Comments are not hashed.

S3-4's PR went red on `0e210a8` (run 39487923435) in Audit alone: GHSA-6q48-6c7w-9gw2, a
critical certifi advisory (compromised emSign intermediate CA keys) published at 19:35 UTC
on 2026-10-20, after `main`'s last green run. `main`'s lock fails it too. `uv lock
--upgrade-package certifi` moves certifi alone, 2026.7.22 → 2026.10.19, from the same
Trusted Publisher as before; the audit is then clean, and the suite and `rag-eval check`
are unchanged. As S3-2's aiohttp bump did, a `build(deps)` commit (`a27d98a`) on the PR, so
the merge clears `main` as well.

**S3-4 (2026-10-21): a file type can have its own splitter, and the corpus a maximum file
size; the real index, tier 2 and every answer are unchanged.**

- **Per-extension splitters:** `pipeline.ingestion.splitters`, empty in `config.yaml`. Each
  document records the splitter that chunked it, and a changed splitter re-embeds only the
  documents it chunks: DEC-17's full rebuild on a splitter change is gone. One entry is
  ready for use, `components.splitters.markdown`, referenced by nothing.
- **A maximum file size:** `max_file_mb: 50`. A larger file is skipped, reported with its
  size and the limit in the log, `IngestReport` and the API's job report, and never read.
  One indexed earlier is removed, as a deleted file is.
- **Parity:** `rag-eval check` exit 0, all six parts ok; `rag-ingest` "Nothing changed";
  the contexts file is IDENTICAL to S3-2's (`5a36e2f9…`); tier 1 is 0.950 / 0.796 / 0.858.
- **The suite:** 1068 passed, 12 skipped, 5 xfailed, coverage 98.40%. 16 deliberate
  breakages of the new guards were all caught.
- **The scratch drill** in a collection of its own showed each rule on the real server: the
  Markdown file alone re-embedded when its splitter changed, and the PDF skipped and its
  chunks removed below the limit.
- **Discovered:** `MarkdownTextSplitter` in langchain-text-splitters 1.1.2 never splits at
  a heading unless `is_separator_regex` is set, which the entry now does. A float limit
  missed the decimal one by a byte, and a rounded size could read as under the limit;
  both are fixed with tests. The fingerprint's relative tests could not see a change to a
  part's shape, so a new test pins `ingestion` to the committed run. Detail in the story.
- **First review (2026-10-21):** everything reproduced on `d1e747f`, including a drill on a
  throwaway Qdrant (overlapping chunk IDs after a splitter change on the same bytes, the
  size limit's three cases), a fuzz of the size line, and the fingerprint under nine edits.
  No defect in the code; this record was the one stale thing, and one cosmetic line went to
  the backlog (`exit_status` on a run the limit emptied). Detail in the story.

**S3-3 is merged** (2026-10-20, PR #21, merge commit `9542c9b`). CI was green on the PR's
last head, `72ca896` (run 39464788422, both jobs), and is green on `main` after the merge
(run 39466730259). The commits: `0ba74fb` the dependency change, `b30a669` the story
record, `72ca896` the record of CI.

**Second review of S3-3 (2026-10-20): no blockers; 5 doc and record findings and 1 nit,
fixed in the working tree.** It re-ran verification 1–11, rebuilt the lock from `HEAD` in
a scratch copy (byte-identical), and broke each new guard on purpose. Detail in the story.

- **Held:** the lock, the 14 packages, tier-1 parity (`5a36e2f9…`, IDENTICAL), and the
  negative control against langchain-core 1.6.3's source. The predicate flags every
  deprecation kind and the old sunset text, and the install guard fails on a stand-in for
  each removed package.
- **Fixed:** this file's DEC-5 row and ARCHITECTURE.md's DEC-5 text still described the
  exit as pending and the gate's allowance; verification 10's count (158); the two package
  counts' flags; and the read timeout's note. Qdrant's access log and telemetry record only
  completed requests, so they cannot say whether the query arrived (S3-6 backlog line).
- **The suite:** 1020 passed, 12 skipped, 5 xfailed, coverage 98.37%; `rag-eval check`
  exit 0.

**S3-3 (2026-10-20): `langchain-community` and `faiss-cpu` are out of the dependencies, and
retrieval is unchanged.**

- **The lock** loses `faiss-cpu` alone. `langchain-community` and `langchain-classic` stay
  for ragas, in the `eval` extra. The main environment loses 14 packages, aiohttp and
  multidict among them.
- **Parity:** the contexts file is IDENTICAL to S3-2's (`5a36e2f9…`), tier 1 is unchanged
  (0.950, 0.796, 0.858), and `rag-eval check` exits 0.
- **The suite:** 1020 passed, 12 skipped, 5 xfailed, coverage 98.37%, and no warnings: the
  sunset notice is gone.
- **The gates that guard it:** the deprecation gate's negative control is now
  `ChatOllama.dict()`, deprecated in langchain-core 1.4.2, and the sunset allowance is
  gone. The architecture test scans `src` for the three packages and checks that a plain
  sync installs none.
- **Discovered:** the first tier-1 run hit a Qdrant read timeout with Qdrant up. It did
  not reproduce (backlog line for S3-6).
- **First review (2026-10-20):** everything reproduced, including the contexts digest and
  the 102 → 88 package count (`uv export --no-dev`). No blocker; four findings fixed: STATUS's backlog and DEC-5
  exit lines, CLAUDE.md's two stale LangChain claims, two test nits (one redundant scan
  entry and predicate clause, one tautological test), and `.claude/` kept out of the commit.
  Detail in the story.

**S3-2 is merged** (2026-10-20, PR #20, merge commit `123e171`). CI was green on the PR's
last head, `5276dca` (run 39381661063), and is green on `main` after the merge (run
39381665898). v0.3's `vectorstore/` folder is deleted on this host. The live index is in
Qdrant under the `rag_qa` alias (`docker compose up -d qdrant` before any command), and
S3-2's parity baseline, which S3-3 to S3-10 diff against, is
`~/dev/rag-s3-2/contexts-qdrant.json`.

- **The commits:** `390b228` the parity tool, `84600e9` its review record, `c49aeea` the
  dependency and compose file, `3cc393a` the store (breaking), `09475d4` CI, `25c32ff` the
  README, `24c5780` the story record; `46a8b80`, `65464a6`, `a2b7d38` the first round's
  follow-up; `21260b9` aiohttp 3.14.5, `79411fc` the store fixes, `5ab1dee` the lock's scope
  and round 2's record.
- **CI:** green on `24c5780` (run 39283008142); red on `a2b7d38` (run 39294075683), in
  Audit, on aiohttp advisories published between the two runs, which `main`'s lock fails
  too; green again on `5ab1dee` once `21260b9` bumped aiohttp. The story's "Verification
  record" has every result.

**Second review of S3-2, round 2 (2026-10-20, on `a2b7d38`): no blockers; 1 concurrency
finding and 2 exit-code findings, fixed on the PR, and CI's red run explained and fixed.**
It ran the hermetic suite against a dead Qdrant URL, the server tests on a throwaway
v1.19.2, and two writers at once there; the real `rag_qa` alias was only read. Detail in
the story.

- **CI went red on the advisory database, not the review.** Three aiohttp 3.14.3
  advisories were published at 17:07 UTC on 2026-10-19, between the green run (16:07) and
  the red one (17:26); `main`'s lock fails the same audit. aiohttp 3.14.5 moves alone, and
  the main environment audits clean. As S2-5's multidict bump did, a `build(deps)` commit.
- **The writer lock excludes only writers that share `var/`** (caveat 36). A second checkout
  on this host writes the same alias under its own lock. Its recovery deleted the other
  run's new generation, and, forced, the generation just switched to, with the alias. The
  docs now say so; a lock Qdrant holds is on the backlog.
- **A wrong `RAG_QDRANT_URL` that reaches another HTTP server** crashed `rag-ingest`,
  `rag-query` and `rag-serve` with exit 1. `check()` now refuses it: exit 2, and the URL
  to check.
- **A switch the server applied but whose answer was lost** said "published nothing", exit
  2. The alias is read back now, and such a run is published.
- **The suite:** 994 passed, 8 skipped, 5 xfailed, coverage 98.37%; `rag-eval check` exit 0.

**Second review of S3-2 as a whole (2026-10-19): no blockers; 1 test gap and 3 doc
findings, fixed in the follow-up commit.** It reproduced parity on the live index, ran the
hermetic suite against a dead Qdrant URL, and attacked the store on a throwaway server.
Detail in the story.

- **No test said search is exact:** `exact=False` passed everything, since local mode
  ignores it and a small collection is scanned in full. A new test spies on the request.
- **The cut rule is necessary:** with 40 exact ties and a restart, the server's own cut
  gave 6 answers; ours gave one.
- **Held under attack:** update equals rebuild, a reader on a deleted generation, a lost
  alias (and the README's recovery recipe), Qdrant stopped mid-run, and the API going 503
  and back.
- **The embedder varies by about 1e-7 with batch composition** (not the store): a story
  claim of "no drift from a fresh ingest" is reworded.
- **A backlog line:** the Qdrant URL reaches API clients inside an ingest job's error
  text; S3-8 decides on redacting it.

**S3-2 phases B to H (2026-10-19): the index is in Qdrant, and retrieval is unchanged.**

- **Parity holds:** `contexts-qdrant.json` is IDENTICAL to FAISS's (sha256 `5a36e2f9…`), a
  second run is STABLE, and tier 1 equals FAISS's question by question (0.950, 0.7958,
  0.8583). `rag-eval check` exits 0: no tier-2 re-run.
- **The hermetic suite:** 987 passed, 8 skipped (the `qdrant_server` tests), 5 xfailed,
  coverage 98.36%, with Qdrant stopped and an empty HF cache. The server tests: 8 passed.
  Verification 3 caught one test whose child process reached the real Qdrant; fixed.
- **The crash drill ran twice:** a kill while embedding leaves nothing (a generation is
  created after embedding), and a kill timed into the window before the switch leaves an
  orphan the next run removes. The alias held both times.
- **The API** picked up a CLI `--rebuild` with no restart.
- **Before working on the store:** start Qdrant, `docker compose up -d qdrant` (CLAUDE.md).
  v0.3's `vectorstore/` folder is still on disk here, unread: delete it after the merge.

**Second review of S3-2 phase A, the parity tool (2026-10-19): no blockers; 2 findings, 1
wording fix and 2 nits, fixed before the first commit.** All 25 golden questions, run
through the real `stream_answer` with a recording fake model, got exactly the prompt whose
context the FAISS file digests. The FAISS run reproduced byte for byte. Detail in the story.

- **The tool now retrieves through `ainvoke`,** as every answer does. Equal on FAISS (step 1
  re-run: the same sha256s), but a Qdrant retriever with its own async path would otherwise
  have been proven on a path no answer takes.
- **A test that checked only `context_record`'s definition** is replaced by one that runs
  the answer path, plus one that tier 1 uses `ainvoke`. Each failed with its guard broken.
- **For phase B:** chunks carry `ingested_at`, which the Qdrant re-ingest sets anew, so
  "the same metadata as FAISS" holds except for that key (Discovered).
- **The suite:** 899 passed, 5 xfailed, coverage 98.38%; `rag-eval check` exit 0.

- **The maintainer, in parallel** (no session needed):
  - hand-label faithfulness for the 20 answerable v0.3 answers, before S3-11 (format in
    DEC-26);
  - pull `qwen2:7b` (4.4 GB) if it is to be a candidate in S3-11, and pick a current model,
    if one;
  - choose a provider for the hosted fallback, and have its key ready for S3-6 (an
    environment variable, never a file in the repository);
  - check `gh auth status` for the `workflow` scope before S3-2, S3-7 and S3-9, which edit
    `ci.yml`. S3-1's push had it (2026-10-16);
  - run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once per clone, so a
    local `git blame` skips the reformat.

**S3-1 is done** (2026-10-16, PR #19; on Opus 5.5 without /fast, where the story says
opus-fast). Phase 3 is open at 1.0.0.dev0, the code is formatted, and CI gates it.

- **Three commits:**
  - `5281b0e`: the version;
  - `01854a1`: the reformat alone, 30 Python files and 868 lines, plus the Markdown
    exclusion;
  - `a2bbe98`: CI's Format step, `.git-blame-ignore-revs` and README.
- **CI** was green on `a2bbe98` (run 38927352954), Format step included, and on the
  close-out commit `177ffbf` (run 38935030666). Check the PR's head before the merge.
- **The reformat is mechanical, checked two ways.** Each file parses to the same syntax
  tree as before, docstrings included, and each is byte-identical to ruff 0.16.8 run over
  `main`'s version.
- **Caveat 24 held:** `rag-eval check` exited 0. Its prompt probe runs through the
  reformatted `chain.py`.
- **The suite:** 890 passed, 5 xfailed, coverage 98.37%.
- **From now on, CI fails on unformatted Python:** run `uv run ruff format` before a
  commit.
- **Discovered:** the Markdown exclude holds only when ruff walks a folder. A file named on
  the command line is still formatted (backlog line).

**Second review of S3-1 (2026-10-16): no blockers; 2 doc findings, fixed on the PR.** It
formatted a clean snapshot of `main` with the project's ruff 0.16.8: the 30 files are
byte-identical to `01854a1` and parse to the same trees. It re-ran the gates (890 passed,
`rag-eval check` exit 0), counted the 4 Markdown files the exclude protects, and tried
`force-exclude = true` (it closes the backlog line). Detail in the story.

- **The story's verification 2** used `HEAD~1`, which the close-out commit moves; it now
  names `01854a1`.
- **"CI green on the PR's head"** is now "on `a2bbe98`", since the close-out commit is the
  fourth on the PR.
- **Keep `.claude/` out of the commit:** it is untracked and not in `.gitignore`.

**Second review of S3-1, round 2 (2026-10-16, on `177ffbf`): no blockers; 1 doc finding and
1 nit, fixed on the PR.** Beyond round 1, it checked what the syntax-tree test cannot see,
the comments: the same order in all 30 files, and the 22 `noqa`, `type: ignore` and
`pragma` comments on the same code. It also checked that the version reaches only the
OpenAPI document. The gates gave the same results on the head. Detail in the story.

- **CLAUDE.md's format rule** said "over what you change", which invites naming files, and
  a Markdown file named on the command line is formatted despite the exclude. It now says
  to run `uv run ruff format` bare.
- **CI on the head:** the board and the story now name `177ffbf`'s green run as well.
- **Squash and rebase merges are allowed** on the repository, and `main` has no
  protection: pick "Create a merge commit" by hand.

**The Phase 3 design is merged** (PR #18, `d491dfb`, 2026-10-16), with both design reviews
folded in. Their records are below.

**The Phase 3 architecture pass is done** (2026-10-15, decisions taken 2026-10-14).

- `ARCHITECTURE.md` has a confirmed Phase 3 section (§3.1–§3.7), with the gap-closure table
  that makes the exit executable and Phase 3's own SRS differences.
- DEC-3 and DEC-9 are resolved, DEC-2 and DEC-5 point at their stories, and DEC-19 … DEC-28
  are new (Decisions log).
- The phase is sharded into S3-1 … S3-12 (below), with S3-11b split off in design review 1,
  and every Phase 3 backlog line names its story.
- **The maintainer's decisions:** Qdrant, with three additions (backup and restore tested
  against a real server; the sparse vectors' source decided, and switched on only in the
  batched re-run; the quickstart change stated); NFR-2 revised; the hosted fallback as an
  opt-in seam plus one real check; named, scoped API keys; `ruff format` once, gated.
- **Tier 2 is re-run once in the phase,** in S3-11b, after S3-11's sweep (DEC-26). Every
  earlier story shows that it moves no hashed input.
- **The pass ran on Opus 5.5 at max effort,** in place of Fable (ADR-018).
- **Versions were checked** by resolving in a scratch copy of the project; the real
  `uv.lock` was not touched.

**Design reviews of the Phase 3 design (before any code).** Both are folded into the pull
request.

**Design review 2 (2026-10-16, depth): no blockers, 18 findings and 9 nits, all fixed in the
docs. None changes one of the maintainer's five decisions.** It ran against a throwaway
Qdrant v1.19.2 and Caddy 2.11.7 (`review-*` containers, removed afterwards), the project's
own venv read-only, and a scratch venv with the Phase 3 packages. It checked:

- **the store:** a prototype of the generation protocol killed at each step, readers during
  alias switches, metadata, names, the payload round trip and parity on the real index, ties,
  the shared `var/`, and snapshots end to end;
- **keys, limits and jobs:** the key check's timing, the keys file in a container, the 413
  with a chunked body, `X-Forwarded-For`, the rate tables' memory, and the abandoned-job rule;
- **cancellation:** the OTel 0.66b1 source and a real-uvicorn disconnect with it, the context
  rules, tenacity, the fallback's stream, and a Caddy spike;
- **the sweep's** statistics, cost and judge, from the committed run;
- **the supply chain:** Trivy's and Prometheus's releases, and every image the design pins.

What it found, and what changed:

- **Findings that changed the design:**
  - **Supply chain (DEC-24).** Trivy's malicious v0.69.4 went through the real release
    pipeline (GHSA-69fq-xp46-6x23), so a signature alone would have passed it. Every pin is
    now verified once: provenance pinned to the workflow, the tag and its commit; that
    commit on `main`; and a release at least 14 days old. promtool runs from the Prometheus
    image by digest, and the weekly run scans every pinned third-party image. DEC-24 has the
    table of digests.
  - **The client's address (DEC-23).** uvicorn trusts `X-Forwarded-For` from loopback by
    default, so a local client could name any address. A published port's connections arrive
    from the compose gateway, inside the network's range. `rag-serve` now passes the trusted
    address itself: Caddy's, and no range.
  - **The rate tables (DEC-23)** grew by 180 bytes per address, and one IPv6 /64 holds 2^64.
    They are bounded now, and the failed-authentication bucket gates only failures.
  - **"No credential" (DEC-23)** means no key and no token. With a keys file alone, the Host
    check would have refused Caddy's public name and its health probe.
  - **One uid (DEC-24).** Under another uid, the host's CLI could not open a container's lock
    file or write its WAL database, and the container could not read the 0600 keys file.
    Compose now runs `app` and `backup` as the host user.
  - **The keys file (DEC-23)** lives in `var/`. A single-file bind mount kept the old inode
    after an atomic replace, so a revocation never reached the container.
  - **The backup service (DEC-25)** now mounts `./var`, which holds the writer lock.
  - **Restores (DEC-25).** An upload to a taken name, the alias's included, replaced the live
    generation's data in place. Restore now refuses a taken name.
  - **Snapshot files (DEC-25)** outlive their collections, and then cannot be reached through
    the API. Restore, `create` and `recover` delete them.
  - **A lost live collection (DEC-19)** takes its alias with it, and recovery would then
    delete every generation, the previous one included. With no alias, it now deletes only
    incomplete generations.
  - **Ties (DEC-19, DEC-20).** At any cut, the server kept different tied points for each
    insertion order and after a restart, so review 1's fusion-cut fix was incomplete. Our
    code now makes every cut, and fuses the hybrid branches itself. Today's corpus has no
    such tie, so parity is unaffected.
  - **The abandoned-job rule (DEC-23)** called a just-finished job abandoned in 92 of 400
    simulated runs. The row is now read again after the probe.
  - **The 413 (DEC-23).** FastAPI turned our own exception into a 400 for a chunked body. The
    middleware raises `HTTPException(413)` instead.
  - **Retries (DEC-21)** run inside the retrieval lock. The breaker wraps each attempt, with
    one deadline per retrieval.
  - **The sweep (DEC-26).**
    - Stage 1's re-score is certain.
    - "Better," and every stage's rule, are margins fixed before the first run.
    - A stage 0 measures the T4 against this CPU.
    - Colab restores this host's index, and runs Ollama only.
    - It needs about 5–9 h of T4 time, not 4–6.
    - About 5 labels outside the 20 come after S3-11b.
- **The nits:**
  - the manifest under one metadata key, its 32 MiB limit, and hot paths that never read it;
  - how "in the way" is told, and the alias batch's semantics;
  - restore's full check, and the fallback's own retries;
  - Qdrant's unprivileged image, Docker 29.9.0 in CLAUDE.md, and a manual switch-back.
- **What held,** each claim run against the real server or the installed source:
  - **the write protocol,** killed after each of 6 steps and at 8 random moments: the alias
    kept the old generation, and one recovery left the live generation and the previous one;
  - **readers:** 2,973 queries during 500 alias switches, each from one generation. A
    deleted generation answers 404, mid-search too;
  - **parity:** the live FAISS generation's 1,708 points, in Qdrant in three insertion
    orders, gave FAISS's top 20 in its order for all 25 questions, and the same contexts.
    Every payload came back with its types;
  - **a snapshot round trip** carried the manifest, the payload index and every point;
  - **shared state:** a `flock` and SQLite's WAL across the host and containers, with one
    uid;
  - **cancellation:** a client dropped during 30 s of silence cancelled the stream within
    milliseconds, with the OTel instrumentation off, excluded or full, and through Caddy over
    HTTP/1.1, TLS and HTTP/2. The detach trap and the executor-thread rule are real, and
    tenacity and the fallback's stream create no task;
  - **the rest:** the key check leaks nothing useful, gemma3:12b fits a T4 at 8k, and review
    1's routing of the fallback moves nothing hashed.
- **The maintainer's five decisions stand.** The one mechanism changed, where hybrid fusion
  runs, is DEC-20's, not one of the five.

**Design review 1 (2026-10-15, breadth): 4 blockers, 16 findings and 11 nits, all fixed in
the docs.** It checked:

- the SRS coverage, and Phase 2's decisions;
- the tier-2 promise, story by story;
- the backlog, and the stories;
- every claim marked "checked", against installed source, Qdrant's own code at v1.19.2,
  release lists and a scratch resolve. The real `uv.lock` was not touched.

What it found, and what changed:

- **The blockers:**
  - **Hybrid retrieval would not repeat.** Qdrant's server sorts fused points by score alone,
    in hash-map order ("Does not break ties"), and ties are common in RRF. Our code now asks
    for the whole union and makes the cut itself, ties by point ID (DEC-20, caveat 31).
  - **A restore raced recovery.** It took the writer lock only to switch, so an ingest
    starting mid-restore would delete the uploaded generation. The lock is now held from the
    upload to the end (DEC-25).
  - **The fallback contradicted the breaker.** Ollama down meant a 503 and an exit 2, in the
    very case the fallback is for, so S3-6's one real check could not run. With a fallback
    configured, Ollama's breaker now routes to it (DEC-21).
  - **The `eval` extra has four advisories, not one,** all published before the pass.
    langchain's and langgraph-sdk's have fixes; diskcache's and ragas's do not. The policy
    allows no ignore for a fixed one, so the langchain bump moves into S3-9 (DEC-24).
- **Findings that changed the design:**
  - readers resolve the alias again per question, since a Qdrant generation is gone two
    switches later (DEC-19);
  - each Qdrant call's retry semantics are stated (DEC-21);
  - an abandoned job is told by a lock of its own, and the health routes are exempt from
    rate limits (DEC-23);
  - S3-2 writes manifest version 2 whole;
  - every story from S3-3 to S3-10 diffs the contexts digest against S3-2's;
  - NFR-1 is benchmarked at k=20, and NFR-2a is measured as a p95 (DEC-22, DEC-27);
  - the gap-closure table has one row per requirement, adding §9's error handling, §11's
    hygiene, NFR-4 and the whole of §2.4;
  - the SRS differences are corrected and extended to 16;
  - a host shell loads Qdrant's key after S3-9 (caveat 32).
- **Two claims failed:** the advisories, and the bump. The langchain bump moves five
  packages, not one, among them websockets 17.1 → 16.1.1 in the main environment, which
  still audits clean.
- **The rest held:** every version and release, qdrant-client's local mode, the BM25
  arithmetic, the torch variant, the Trivy compromise, and `ruff format`'s 30 files and
  868 lines.
- **The maintainer's calls (2026-10-15):**
  - the bump in S3-9;
  - `max_file_mb: 50`;
  - S3-12 runs the SRS amend session;
  - S3-11 is split into the sweep and S3-11b, the commit.
- **The backlog lost nothing:** all 19 of `main`'s Phase 3 lines have an owner.

**The release is done:** `v0.3` is tagged and its CI run is green (below).

**Phase 2 is complete and released as `v0.3`** (2026-10-14): the tag is on `aa8c19b`, and
CI is green on the PR, on `main` and on the tag.

**S2-9 is done** (2026-10-14, PR #17, merged as `aa8c19b`; on Opus 5.5, where the story
says opus-fast). The exit criterion holds; one bug was found and fixed.

- **CI**, `check` and `eval-retrieval` both green each time:
  - the PR's head before the review, `6cec959` (run 38396359569);
  - its head after the review, `69aa991` (run 38517570599);
  - `main` after the merge (run 38518198641);
  - the `v0.3` tag (run 38518816307).
- **The tag** is annotated, sits on `origin/main`, and reads 0.3.0 in both version files.
  The six commits carry `Refs:` trailers and no attribution.

**Second review of S2-9 (2026-10-14): no blockers; 4 doc findings and 4 nits, fixed on the
PR.** It re-ran the gates (890 passed, `rag-eval check` exit 0), ran the new test's child
against `main`'s `server.py` (status 130 and a traceback there, 0 on the branch), and
loaded v0.2's own config through the upgrade guide. Detail in the story.

- **ADR-026's example was false:** `HuggingFaceBgeEmbeddings` already defaults
  `encode_kwargs` to `{}`, so `{}` changes nothing. It dates from S2-6, whose test checked
  our function, not the library. The decision stands; the ADR, two dated notes and two
  comments are corrected, and no hashed input moved.
- **"CI's only network use"** (README, ADR-023) was wrong: the `check` job syncs and
  audits over the network. `eval-retrieval` is the only job that downloads models.
- **A second Ctrl-C on `rag-serve`** still exits 0 but does not shorten the 5 s wait. It
  skips the lifespan's shutdown, and uvicorn logs errors. It is documented now, and the
  backlog's Docker line has it.
- **Caveat 24's "70 min on a T4"** counted two scorings; one is about 35 min.

- **The fresh clone over HTTP** (`~/dev/rag-s2-9-e2e/fresh`, `main` plus the version bump,
  then the server fix):
  - `uv sync` gave 0 `nvidia-*` packages and torch `2.14.0+cpu`;
  - `rag-ingest` built 1,708 chunks in 1:59, and a second run had nothing to embed (0.6 s);
  - the GLIDER query streamed over SSE: first token at 154.6 s, done at 173.5 s;
  - three API ingest jobs (add a file, remove it, nothing) each switched the served
    generation without a restart;
  - a drop during the prefill, a drop mid-stream and a `rag-query` Ctrl-C each closed our
    connection to Ollama within 0.03 s. The runner idled at the end of the prefill, and
    within 1 s mid-stream;
  - every next query got a slot.
- **The quality bar:** `rag-eval check` exit 0, fingerprint fresh; tier 1 here gave 0.950,
  0.796 and 0.858 in 43 s. CI's `check` and `eval-retrieval` were green on the PR.
- **Found by the run, fixed in this PR (maintainer's call):** Ctrl-C on `rag-serve` shut
  down cleanly, then printed a `KeyboardInterrupt` traceback and exited 130, where its
  docs say 0. It dates from S2-7, whose log lost its shutdown lines to a plain `tee`.
  `main()` now catches it, and a test sends a real SIGINT to a child process.
- **The docs:**
  - README: status, all five upgrade steps from v0.2, evaluation and the Colab recipe, and
    API examples;
  - ARCHITECTURE.md: §2.2 corrected, §2.8 written, and DEC-7 annotated;
  - ADR-007's moot exception noted, and ADR-022 to ADR-026 added;
  - CLAUDE.md re-verified.
- **Release 0.3.0.** Caveats 25–28 are added, and the backlog is reconciled.

**S2-8 is done** (2026-10-13, PR #16, merged as `5961de8`; on Opus 5.5, where the story
says opus-fast).

- **CI** was green on the PR's head, `573e733` (run 38343630988), and on `main` after the
  merge (run 38344128852). Verification 1–6 are shown in the story, 3 by the maintainer's
  hand.
- **What it delivered:**
  - **`rag-query` starts cleanly.** It checks the index from its files, then the model's
    server, then loads the models. Ollama down, a model missing from the HF cache offline,
    or Ctrl-C while loading is one message and exit 2.
  - **`rag-query`'s prompt** has line editing and history (`readline`, libedit here). A
    refused or dropped connection names Ollama with a hint, and `Answer:` waits for the
    first token.
  - **`rag-ingest`** leaves out zero counts, says "item(s)" and "1 page/section", and
    quiets pypdf's own warnings for its run.
  - **`fetch_dataset.py`** refuses `paths.data` and `RAG_DATA_PATH` too, and writes only a
    download that starts with `PAR1`.
- **Breaking (exit code):** an unreadable corpus root exits **2**, where it exited 1, from
  `rag-ingest` and in an API ingest job's `exit_code`. The failure lines read "N item(s)".
  `86788b5` carries `!` and a `BREAKING-CHANGE:` trailer.
- **Moved (maintainer):** `preflight` and `OLLAMA_HINT` moved from `api/server.py` to
  `components`, so the CLI never imports the `api` package. An architecture test holds
  the rule.
- **New main dependency:** `requests`, declared rather than transitive, at its locked
  version.
- **Two reviews, no blockers.**
  - The first fixed two code findings: an `OSError` from the index check got the HF cache
    hint, and a read timeout got "Is Ollama running?".
  - The second fixed one. The new exception escaped tier 2's corpus scan, so `rag-eval`
    printed a traceback, still with exit 2. Implementation had missed that caller.
  - The second also drove the prompt under readline through Ctrl-C and Ctrl-D: the
    terminal stayed sane.
- **Caveat 24 held:** `rag-eval check` was clean, and no hashed input moved.
- **The suite:** 888 passed, 5 xfailed, coverage 98.37%. The state after each of the first
  four commits was checked green before they were made.

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
| Phase 2 | `v0.3` | 0.3.0 — **released 2026-10-14** (`aa8c19b`) |
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

- [x] **A ruleset on `main`** — ruleset 25561360, applied 2026-10-21 ("Now"). It requires
      `check` and `eval-retrieval`; S3-9's recipe extends it with `deploy-config` and
      `container`.

- [x] **Phase 2 (S2-5b): somewhere to run tier-2 scoring.** Done 2026-10-11: a Colab T4,
  with Ollama pinned to 0.22.1 and outputs on Google Drive. That is about 35 min per
  scoring, against 13–18 h on this CPU at 2 threads. The notebook's cells are in S2-5b's
  story file.

- [ ] **Phase 3 (S3-11): the hand labels.** Faithfulness for the 20 answerable answers in
      `eval/runs/answers-latest.json`, as `eval/hand_labels.jsonl` (DEC-26). One to two
      hours of reading; it decides whether the judge stays.
- [ ] **Phase 3 (S3-11): the DEC-2 candidates.** `ollama pull qwen2:7b` (4.4 GB) if it is to
      be swept, and the name of a current model, if one should be.
- [ ] **Phase 3 (S3-6): a hosted provider and its key,** for the fallback's one real check.
      Any OpenAI-compatible provider; the key in an environment variable only.
- [ ] **Phase 3 (S3-11): the Colab notebook** gains cells that run Qdrant's static binary,
      restore this host's index into it from a snapshot (`rag-backup restore`), and run
      `rag-eval generate` with CPU torch, beside S2-5b's scoring cells (design review 2).
- [ ] **Phase 3 (S3-11b): about 5 more labels,** on the committed answers the judge scored
      below 1, or that the two judges disagree on: a check outside the 20 it was chosen on
      (design review 2).
- [ ] Nothing else: uv ✓, Python 3.12 (uv-managed) ✓, Ollama daemon ✓,
      Docker 29.9.0 ✓ (Phase 3), disk 211 GB free ✓. **No MCP servers are
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
- Items 25–28 come from the Phase 2 exit: S2-9's end-to-end run, and S2-8's review.
- Items 29–31 come from the Phase 3 architecture pass (2026-10-15). Items 9 and 20 change
  when S3-2 lands; each says how.
- Item 32, and the second half of 31, come from design review 1 of Phase 3 (2026-10-15).
- Items 33–35, and the last parts of 30 and 31, come from design review 2 (2026-10-16).

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
   - **From S3-2:** a scratch index is a collection name, so set `RAG_DATA_PATH` **and**
     `RAG_QDRANT_COLLECTION`. `RAG_VECTOR_STORE_PATH` is refused while set, so the old habit
     fails loudly instead of writing the real index (DEC-19).
10. **Check a guard the way CI runs it.** A plain `uv run` re-syncs the
    environment first, so it can repair the very state a check is meant to
    catch and then pass. Guards on the installed environment use
    `uv run --no-sync` (S1-7). And `uv sync` is exact: it removes an extra it was not
    asked for, such as `eval`, while a plain `uv run` keeps it. Restore it with
    `uv sync --extra eval` (S2-7).
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
    - **After S3-2, once more, into Qdrant:** start it (`docker compose up -d qdrant`),
      then `rag-ingest`. v0.3's `vectorstore/` folder is no longer read; delete it once the
      new index answers.
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
    - **A re-run of generate and score (about 70 min here, then about 35 min on a T4):**
      the prompt, an llm, retriever or reranker spec, `reranker_candidates`, a corpus file,
      the embedder, the splitter, the loader map or a golden question.
    - **A re-run of score only:** a `ground_truth`, `decline_marker` or the judge's
      Modelfile.
    - **Free:** `notes`, `expected_sources`, comments, and the runtime-only llm keys.

    So a story that touches one of these budgets the re-run, and the error names which
    re-run. Code changes are not hashed: after one that changes answers, re-run tier 2 by
    hand.
25. **Ollama's keep-alive counts from the end of the last request** (S2-9). After a
    dropped prefill, Ollama ends the request at once (`/api/chat` 500 in its journal),
    while the runner evaluates on for minutes. So the 5 minutes can run out soon after the
    runner finishes.
    - In S2-9, mistral unloaded twice that way. Each time the next question paid a reload
      and a full prefill, and lost the prompt cache that a repeated question relies on.
    - Run a check that relies on the cache within 5 min of the previous request's end, or
      budget a full prefill. A tool call waiting for approval uses up those minutes too.
26. **While the runner holds the CPU, everything else on this host is slow** (S2-9). At
    `num_thread` 10, Ollama takes about 1,000% of the 12 threads.
    - The API's retrieval (the query embedding and a 20-candidate rerank) took 7 s rather
      than 0.78 s. A request right after a dropped prefill is slow for that reason, not
      because of its slot.
    - Time the API and retrieval with the runner idle.
    - The SSE status line arrives with `sources`, after retrieval. So a slot probe that
      leaves on its status line has already done its retrieval, and the 7 s showed up as
      a slow 200.
27. **A server's Ctrl-C is exercised only on a main thread, and seen only with `tee -i`**
    (S2-7, S2-9).
    - uvicorn installs its signal handlers only on the main thread, and re-raises the
      captured signal after its graceful shutdown. asyncio's runner turns a re-raised
      SIGINT into `KeyboardInterrupt`.
    - The real-uvicorn test fixture runs the server in a thread, so it never takes that
      path. Test a server command's Ctrl-C in a child process, as `test_api_server.py`
      now does.
    - Log a foreground server with `tee -i`. A plain `tee` dies with the Ctrl-C and loses
      the shutdown lines, which is how `rag-serve`'s traceback went unseen from S2-7 to
      S2-9.
28. **A new exception from a shared function means checking every caller** (S2-8).
    `CorpusUnreadableError` reached the two callers the story named, `rag-ingest` and the
    API's jobs, but not tier 2's `corpus_scan`, which then printed a traceback. Find the
    callers with `grep`; the story's list is not the full list.
29. **Qdrant's local mode is for tests only** (DEC-19). `QdrantClient(path=...)` takes an
    exclusive, non-blocking lock on its folder (`portalocker`, in `qdrant_local.py`), so
    `rag-serve` and `rag-ingest` could not share it. Tests use `":memory:"` through the
    fixture; everything else talks to the server. Local mode is also a reimplementation in
    Python, with no snapshots: a test that must hold for the server carries the
    `qdrant_server` marker and runs in CI's `eval-retrieval` job.
30. **Pin a security tool by a hash we hold** (DEC-24). `aquasecurity/trivy-action`'s tags
    were hijacked in March 2026 (CVE-2026-33634), and Trivy v0.69.4 and some Docker Hub
    images were malicious too. A tag, or a checksum file fetched from beside the binary, is
    only as trustworthy as the account that publishes it. Trivy runs as its release binary,
    checked against a sha256 written in `ci.yml`, and promtool from the Prometheus image,
    pinned by digest.
    - **A signature proves the pipeline, not the release** (design review 2). v0.69.4 was
      made by tagging the attacker's commit and running the real release pipeline, so it was
      signed as usual. Verify a pin once, with provenance pinned to the signer workflow and
      the tag; check that its source commit is on `main`; and take only a release 14 days
      old or more. S3-9 has the commands, and DEC-24 the image table.
31. **Qdrant's default RRF constant is 2, over zero-based ranks** (DEC-20), not the usual
    60 (checked in `qdrant_client/hybrid/fusion.py`, and in the server's
    `reciprocal_rank_fusion.rs`). Always set `rrf_k`, or a top rank in one list will
    outweigh agreement between the two.
    - **And the server breaks no ties** (design review 1). It sorts the fused points by
      score alone, in hash-map order. Never let Qdrant make a cut through a fused list: ask
      for the whole union, and order it by score, then point ID, as `vectorstore` does. Local
      mode sorts stably, so only a test against the server shows this.
    - **Never let the server make any cut through tied points** (design review 2). Each
      prefetch still cut its own list on the server, which decides a tie by the collection's
      layout. Three insertion orders kept three different sets of 10 from 60 tied points, and
      a restart changed each, while every repeat within one process agreed. So our code asks
      for more than each cut, extends while the cut ties, and fuses the two branches itself.
      Qdrant's RRF constant above then no longer applies. A tie test must compare builds in
      different insertion orders, or runs either side of a restart: two runs in one process
      prove nothing.
32. **A host shell talking to the compose stack needs Qdrant's key** (design review 1).
    - From S3-9, the stack's Qdrant requires `RAG_QDRANT_API_KEY`, which compose reads from
      `.env`. `settings.py` reads the environment only, never `.env`.
    - So load it first, `set -a; . ./.env; set +a`, and pass
      `-H "api-key: $RAG_QDRANT_API_KEY"` to curl.
    - A host `rag-query` also talks to the host's Ollama, while the stack's holds mistral
      too: stop one first (caveat 8).
33. **The host and the containers share `var/`, so they share one uid** (design review 2).
    - A `flock` and SQLite's WAL both work across the bind mount.
    - But under another uid, a lock file a container created is a `PermissionError` to the
      host's CLI, its WAL database is read-only to the host, and the host's 0600 keys file
      is unreadable from the container. Compose sets `user: "${UID}:${GID}"` from `.env`.
    - Never bind-mount a file on its own when it is replaced atomically, the keys file above
      all. The mount keeps the old inode, so a revocation never reaches the container. Mount
      its folder.
34. **Qdrant's snapshots have three traps** (design review 2, each checked on v1.19.2).
    - An upload to a name that exists replaces that collection's data in place, and an
      alias's name reaches its target: the live generation, with no switch. Restore only to
      a free name.
    - Snapshot files outlive their collection, and cannot be listed or downloaded once it
      goes (404). Delete them before the collection.
    - Uploads land in `snapshots/tmp/` first: remove it, and every upload fails with a 500.
35. **Deleting the alias's target deletes the alias too** (design review 2). From S3-2, with
    no alias, recovery keeps every complete generation, so the previous one survives a lost
    live collection. `rag-query` names it, and README has the line that points the alias
    back.
36. **The writer lock excludes only the writers that share `var/`** (S3-2's second review).
    The lock is a file and the index a Qdrant collection, so a second checkout on this host
    (a fresh-clone e2e) writes the same `rag_qa` alias under a lock of its own. On a
    throwaway server, one run's recovery deleted the other's new generation, and that run
    died on a 404. In a forced interleaving it deleted the generation just switched to, and
    the alias went with it until the next publish. Give another checkout
    `RAG_QDRANT_COLLECTION=<name>`, or stop this checkout's Qdrant while it runs its own.

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
  **Decided (DEC-27, 2026-10-14): the SLO is revised.** The `sources` event in under 2 s,
  and the time to first token reported per class of hardware: this CPU's p95 from the
  committed run, and a T4's from S3-11. A GPU host later is a `base_url` edit, with no code.
- **Phase 3 — the tier-2 sweep (S3-11 and S3-11b, DEC-26):** candidates are generated and
  scored on a Colab T4. Qdrant's static binary runs there, holding this host's index
  restored from a snapshot, with CPU torch, since the T4 runs Ollama only (design review 2).
  Only the committed run, S3-11b's, is generated here. About 5–9 h of T4 time across 3–4
  sessions, plus about 70 min here (design review 2; the first draft said 4–6 h).
- **Phase 3 — the NFR-1 benchmark (S3-7)** uses 100k synthetic vectors on the local Qdrant
  server: no GPU, and minutes, not hours.

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
| [S2-8](phase-2/S2-8-cli-polish.md) | CLI polish batch | backlog: CLI polish | S2-1, S2-6 | Done 2026-10-13 (PR #16) |
| [S2-9](phase-2/S2-9-phase-2-exit.md) | Phase 2 exit: docs, end-to-end, 0.3.0 | — (exit) | S2-1 … S2-8, S2-5b | Done 2026-10-14 (PR #17, `v0.3` on `aa8c19b`) |

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

## Phase 3 — Make it production-grade

Exit criterion (SRS §12): the gaps in §9 and §11 are closed. Phase 3 completes the SRS, so
its remaining requirements close too, or are amended in S3-12's amend session. The exit is
executable: the gap-closure table in ARCHITECTURE.md's Phase 3 intro names, for every §9 and
§11 row and every remaining requirement, the story that closes it and what S3-12 shows from a
fresh clone. Exit ⇒ tag `v1.0`, version 1.0.0. Design: ARCHITECTURE.md §3.1–§3.7.

| Story | Title | Closes | Depends | Status |
| --- | --- | --- | --- | --- |
| [S3-1](phase-3/S3-1-open-phase-3.md) | Open Phase 3: 1.0.0.dev0, format the code once and gate it | DEC-28, backlog: `ruff format` | — | Done 2026-10-16 (PR #19) |
| [S3-2](phase-3/S3-2-qdrant-store.md) | Move the index to Qdrant: generations behind an alias | DEC-3, DEC-19, NFR-4, §11 (interrupted ingestion) | S3-1 | Done 2026-10-19 (PR #20, merged 2026-10-20) |
| [S3-3](phase-3/S3-3-leave-langchain-community.md) | Leave langchain-community and FAISS | DEC-5 step 3, ISS-16, §9 (deserialization) | S3-2 | Done 2026-10-20 (PR #21, merged 2026-10-20) |
| [S3-4](phase-3/S3-4-ingestion-config.md) | Per-loader splitters and a maximum file size | SRS §7.2, §7.1 | S3-2 | Done 2026-10-21 (PR #22, merged 2026-10-21) |
| [S3-5](phase-3/S3-5-hybrid-retrieval.md) | Hybrid retrieval and source filters, measured by tier 1 | FR-3, DEC-20 | S3-4 | Todo |
| [S3-6](phase-3/S3-6-resilience.md) | Timeouts, retries, circuit breakers and the hosted fallback | NFR-6, DEC-9, DEC-21, SRS §2.4 | S3-2 | Todo |
| [S3-7](phase-3/S3-7-observability.md) | Structured logs, traces, metrics and alerts | NFR-10, NFR-1, NFR-3, DEC-22 | S3-6 | Todo |
| [S3-8](phase-3/S3-8-api-hardening.md) | API keys, rate limits and a stateless query tier | NFR-5, DEC-23 | S3-7 | Todo |
| [S3-9](phase-3/S3-9-containers.md) | Containers, compose, TLS and CI scanning | §11 (CI, containers, config), §8.3, §9 LLM05, DEC-24 | S3-8 | Todo |
| [S3-10](phase-3/S3-10-backup-restore.md) | Backup and restore | §11 (backup and DR), DEC-25 | S3-9 | Todo |
| [S3-11](phase-3/S3-11-tier2-sweep.md) | The tier-2 sweep: measure every lever on a T4 | DEC-2, DEC-26 (the sweep) | S3-5, S3-10 | Todo |
| [S3-11b](phase-3/S3-11b-tier2-commit.md) | Commit the tier-2 re-run: v1.0's configuration and floors | DEC-26 (the re-run), §9 LLM01, NFR-2 (numbers) | S3-11 | Todo |
| [S3-12](phase-3/S3-12-phase-3-exit.md) | Phase 3 exit: end to end on the stack, the SRS amended, docs, 1.0.0 | — (exit); the SRS amend session | S3-1 … S3-11b | Todo |

**Order.** The numbering is the order, and the Depends column holds throughout.

- **S3-1 first:** a phase boundary is the one moment no story branch is open to conflict
  with a whole-tree reformat.
- **S3-2 before everything that touches the store.** It proves parity with FAISS while FAISS
  is still installed; S3-3 removes it afterwards.
- **S3-5 measures; S3-11 switches.** Hybrid retrieval lands with the real config unchanged,
  so its tier-1 table becomes S3-11's input.
- **S3-11 and S3-11b come last before the exit,** so the one committed tier-2 run describes
  the code that ships as v1.0. S3-11 measures from its branch and merges nothing hashed;
  S3-11b commits the choices with the run (design review 1).

**Tier 2 is re-run once** (DEC-26), in S3-11b. From S3-1 to S3-11, each story's verification
shows `rag-eval check` exiting 0, and from S3-2 to S3-10 the contexts digest unchanged against
S3-2's (design review 1).

**Story setup.**

- **Plan-first:** S3-2, S3-5, S3-6, S3-7, S3-8, S3-9, S3-10 and S3-11.
- **Branches:** every story has its own branch and PR, named in its file. The risky ones are
  S3-2 (the store), S3-8 (authentication), S3-9 (CI and the image) and S3-10 (a restore
  switches the live index).
- **Models:** the fable-routed stories (S3-2, S3-5, S3-6, S3-7, S3-8, S3-11) run on Opus 5.5
  at max effort while Fable is unavailable, and record it under Deviation.
- **Workflow scope (caveat 13):** S3-1, S3-2, S3-7 and S3-9 edit `ci.yml`.
- **Version:** S3-1 bumps it to `1.0.0.dev0`, and S3-12 releases 1.0.0.

## Decisions log

| ID | Decision | Status | Notes |
| --- | --- | --- | --- |
| DEC-1 | LangChain: migrate to 1.x vs pin legacy 0.2.x | **Resolved 2026-09-18: migrate to 1.x** | Verified resolve: core 1.6.3, community 0.4.2, ollama 1.1.0, huggingface 1.2.2, text-splitters 1.1.2, classic 1.0.8 (transitive only). Rules: app code imports only `langchain_core` / `_text_splitters` / `_community` / `_huggingface` / `_ollama`; chain is hand-composed LCEL, no `langchain_classic` imports in Phase 0–1; PyTorch index `explicit = true`. Full rationale: ARCHITECTURE.md §0.1. |
| DEC-2 | Query LLM: pull `qwen2:7b` vs repoint to already-pulled `mistral` | **Resolved 2026-09-18: `mistral`, `phi3` fallback** | Same weight class as qwen2:7b, so the reason is zero download + known-good here, not RAM. `qwen2_ollama` entry stays in config unused. All Ollama entries: `temperature 0`, `num_ctx 4096`, `num_predict 512`, `validate_model_on_init true`; retriever `k 5`. Revisit with the Phase 2 eval harness. **Revisit placed (Phase 3 pass, 2026-10-14): S3-11**, the tier-2 sweep (DEC-26): mistral and phi3, plus qwen2:7b if pulled, plus a current model if the maintainer wants one, generated and scored on a Colab T4; faithfulness and the decline rate choose, and the time to first token here breaks a tie. **Design review 2 (2026-10-16):** the rule is a margin fixed before the first run. mistral stays unless a candidate leads by 0.10 in faithfulness or 0.2 in the decline rate, with no metric behind by more than 0.05 (DEC-26). |
| DEC-4 | Repo layout: `docs/` currently holds the RAG corpus, colliding with the universal convention that `docs/` is project documentation | **Resolved 2026-09-18: corpus → `corpus/`, `docs/` becomes project documentation** | Maintainer decision. The old name needed a CLAUDE.md hard rule to stay safe, and the text loader claims `.md`, so a project doc dropped in there gets embedded into the index. Renamed in S0-4 (which rewrites every path anyway); `SRS.md` / `WORKFLOW.md` / `ARCHITECTURE.md` move into `docs/` in S0-7, which also inverts the CLAUDE.md rule. `CLAUDE.md` and `README.md` stay at root (auto-load; GitHub renders README from root only). `stories/` stays at root as working state. |
| DEC-5 | `langchain-community` was sunset 2026-05-22 (issue #674): frozen, unmaintained, warns on import. We use it for FAISS, the loaders, and the disabled cross-encoder | **Resolved 2026-09-25: staged exit, keep through Phase 0** | Nothing is broken; the lock pins it and `langchain-core<2` bounds drift. Exit rides planned work: loaders → ~40 lines of own code on `pypdf`/`docx2txt` (Phase 1, with the loader-selection item); cross-encoder → `sentence-transformers` directly (Phase 2 reranker); FAISS → `langchain-qdrant` (Phase 3). After Phase 3 both `langchain-community` and its transitive `langchain-classic` leave `pyproject.toml`. No official standalone FAISS/loader package exists; the unofficial `langchain-faiss` 0.1.1 is rejected on supply-chain grounds. **Step 3's mechanism amended (Phase 3 pass, 2026-10-14):** our own module on `qdrant-client`, not `langchain-qdrant`, whose `add_texts` would embed again in batches of 64 and could break the parity proof (DEC-19). The goal stands. **Done (S3-3, 2026-10-20):** `langchain-community` and `faiss-cpu` left `pyproject.toml` (`langchain-classic` was never in it), and a plain `uv sync` installs neither LangChain package. ragas still brings both into the `eval` extra, so `uv.lock` keeps them, and the import scan in `tests/test_architecture.py`, not the environment, keeps them out of `src`. The deprecation gate allows no warning; its negative control is `ChatOllama.dict()`. ARCHITECTURE.md DEC-5 has the dated note. |
| DEC-3 | Vector store for Phase 3: Qdrant vs pgvector | **Resolved 2026-10-14: Qdrant** (maintainer) | Native hybrid dense+sparse, one container, no Postgres to run. pgvector only if a Postgres already exists in the deployment. Store-specific code is confined to `vectorstore.py` so either works. **Resolved in the Phase 3 pass**, with three additions from the maintainer: backup and restore are tested against a real server, because local mode has no snapshots (DEC-25); the source of the sparse vectors is decided explicitly, and they are switched on only in the batched re-run (DEC-20); the quickstart change is stated, a Qdrant container, not local mode with `path=` (DEC-19). The design is DEC-19. S3-2. |
| DEC-6 | Config validation: how far Pydantic reaches into the call sites | **Resolved 2026-09-25: frozen `RagConfig`, attribute access everywhere** | `load_config()` returns a frozen model, not a dict; typed component kinds (`extra="forbid"`) make ISS-01 and ISS-11 unwritable; component leaves stay `extra="allow"` (ADR-010); retrievers get their own `RetrieverSpec` (no `_target_` — typing them as components fails the real config, verified). Non-mutation becomes structural via the access path: call sites only get fresh copies from `component(ref).spec()`/`.kwargs()` — `frozen=True` alone does not freeze nested dicts (verified). **Verified trap:** a field literally named `_target_` is silently dropped by Pydantic (leading underscore ⇒ private attribute; `model_dump()` returned `{}`) — the schema must use `target: str = Field(alias="_target_")` and dump `by_alias=True`. S1-1. ARCHITECTURE.md §1.1, §1.4. |
| DEC-7 | Where the `_target_` allowlist lives, and whether it is configurable | **Resolved 2026-09-25: hardcoded, inside `import_from_string`** | Enforcing at the import funnel rather than in `build_object` covers `ingest.py`'s direct call — the gap ADR-015 flagged. Not config-overridable and no env escape hatch: an allowlist the config can edit is not an allowlist. `langchain_classic.` was allowed as a *config* prefix (the disabled reranker entry); ADR-007 governs *source* imports and is unaffected. **S2-4 (2026-10-04) removed `langchain_classic.` and `langchain_community.`** from the list: our own reranker replaced that entry (DEC-16), so no `_target_` names either. A v0.2 `config.yaml` no longer loads until its old `cross_encoder` entry is deleted. Load string-checks prefixes (nested too) and never imports; a separate `check_imports` helper, used by tests, catches misspelled classes (ISS-03). S1-2. Closes ISS-04. |
| DEC-8 | DEC-5 step 1: replace the `langchain-community` loaders now or later | **Resolved 2026-09-25: now, in Phase 1** | Own `PdfLoader`/`DocxLoader`/`TextLoader` on `pypdf`/`docx2txt`; `extensions` leaves the component entries for `pipeline.ingestion.loaders`, giving one instantiation path; discovery becomes recursive (ISS-13). Afterwards `langchain_community` is imported only by `vectorstore.py`. Risk is citation metadata, so the story records the `v0.1` chunk-count + citation baseline before changing anything and must reproduce it. S1-3. |
| DEC-9 | NFR-6 (timeout / retry / circuit-breaking): Phase 1 or later | **Resolved 2026-09-25: Phase 1 does error handling only** | ISS-05 (collect failures, exit non-zero — NFR-7) and ISS-06 (REPL try/except) are S1-4. Timeout, retry-with-backoff and circuit-breaking (SRS §11 Resilience) need the service shape and move to Phase 3 with the serving decision. Deferred deliberately, not by omission. **Resolved 2026-10-14 by DEC-21** (Phase 3 pass): timeouts, retries where a retry is safe, and circuit breakers around the network calls (Ollama, Qdrant, the fallback). S3-6. |
| DEC-10 | Device selection: collapse the `_cpu`/`_cuda` embedder entries into one env-driven setting | **Resolved 2026-09-25: no — the entries stay explicit** | The backlog item assumed pydantic-settings interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside YAML values. **It does not** — there is no `${VAR:-default}` expansion for arbitrary values. With the CUDA torch variant (DEC-12), one sync command plus repointing `pipeline.ingestion.embedder` is already a one-line switch, so the four entries restored in `ebcfc2e` stay as FR-1 axes. |
| DEC-11 | CI: host, hermeticity and how hard the gates bite | **Resolved 2026-09-25: GitHub Actions, hermetic suite, blocking gates** | No Ollama, no model download, no network in tests: `DeterministicFakeEmbedding` (verified present in core 1.6.3) + a fake chat model. Gate order: `ruff check` → `mypy` (`disallow_untyped_defs` on `src/rag_qa`) → `pytest --cov-fail-under=80` (omitting `evaluate.py`, Phase 2 + `eval` extra) → `pip-audit`. Each gate blocks from the story that adds it. **No `ruff format --check` in Phase 1** — the tree is unformatted and a whole-tree reformat is the diff that gets rubber-stamped → backlog. |
| DEC-12 | Making the declared GPU embedder entries actually installable | **Resolved 2026-09-25: uv-conflicting CPU/CUDA torch variants; plain `uv sync` stays CPU** | Each variant routed to its own index, `explicit = true` on both. **Mechanism chosen by S1-7's spike**: extras have no default, so torch-only-in-extras would make a plain sync pull PyPI CUDA torch via `sentence-transformers`. Preferred: dependency groups + `default-groups = ["dev", "cpu"]`; fallback: extras with `--extra cpu` on every install path. S1-7 re-runs the S0-2 smoke test on the **installed env** (the lock legitimately holds both variants): core 1.x, `torch 2.14.0+cpu`, zero `nvidia-*`. CUDA is verifiable here as a **resolve only** — this host is CPU-only. |
| DEC-13 | Partial ingestion failure: replace the index with what loaded, or keep the previous one | **Resolved 2026-10-01: replace, exit 1** | Maintainer's call, raised by S1-4's third review. Some documents or folders unreadable → index the rest, save over the old index, exit 1 ("rebuilt without them"). Keeping the old index is safer when unattended, but one persistently bad file would block every update. Phase 1 ingests are hand-run and watched. Phase 2's hash manifest dissolves it: a failed document keeps its previously indexed chunks. Exit codes: 0 all indexed · 1 document/folder unreadable, nothing indexed, or unexpected error (traceback) · 2 cannot start. ARCHITECTURE.md §1.1. **Superseded in part by DEC-17 (S2-6, 2026-10-05):** in an update a failed document keeps its chunks, and exit 1 reads "The index was updated; the failed documents kept their previous chunks, if they had any"; a full rebuild still leaves them out ("rebuilt without them", maintainer). Exit 2 also covers another ingestion holding the lock, or something at the index path that is not an index. |
| DEC-14 | Cancelling an answer: how the CLI's Ctrl-C and the API's client disconnect stop generation | **Resolved 2026-10-02: one answer-event stream (`answering.stream_answer`); stopping it means cancelling the task that consumes it** | **Verified trap:** async alone does not fix it. Under `RunnablePassthrough.assign`, `RunnableParallel` waits on its step tasks with `asyncio.wait` and never cancels them. In the trial, a cancel at 0.5 s during a simulated prefill left the stream open until 2.0 s; streaming `prompt \| llm \| parser` directly closed it at 0.5 s. **Design:** the CLI uses one `asyncio.Runner` per session. `QueryPipeline` holds the parts, and `build_rag_chain` stays for `invoke`. `aclose_llm` closes ChatOllama's clients, and `ttft_ms` is measured for NFR-2. S2-1. ARCHITECTURE.md §2.1. |
| DEC-15 | Phase 2 quality gate: what runs in CI when RAGAs on CPU takes hours | **Resolved 2026-10-02: two tiers** (maintainer) | **Tier 1:** retrieval hit rate, MRR and recall against `expected_sources`, recomputed on every pull request and every push to `main` by a CI job that caches the HF models. That job is CI's only network use. **Tier 2:** RAGAs with a `gemma2:9b` judge through Ollama's OpenAI endpoint, run offline (scoring can go to Colab/Kaggle) and committed with a fingerprint. `rag-eval check` fails CI on a floor breach **or a stale run** (maintainer: fail, not warn). Rejected: a hosted judge in CI; offline-only. **Second review:** the fingerprint has six parts (`query` with a rendered-prompt probe, `ingestion`, `corpus`, `questions`, `references`, `judge`) in `evaluation/fingerprint.py`; Ollama digests are recorded, not hashed; the judge is `rag-judge` from `eval/judge.Modelfile` (8k context, caveat 22); the tier-2 tolerance comes from two scorings. S2-2, S2-3, S2-5, S2-5b. |
| DEC-16 | Reranker: `langchain_classic`'s `ContextualCompressionRetriever`, or our own | **Resolved 2026-10-02: our own `CrossEncoderReranker` on `sentence-transformers`** (DEC-5 step 2) | A `langchain_core` `BaseDocumentCompressor` that writes `rerank_score` into metadata. k=20 candidates are cut to `top_n` 5, and tier-1 numbers decide whether it is on. `langchain_classic.` and `langchain_community.` leave `ALLOWED_PREFIXES`. The `min_score` knob is for the Sources-relevance backlog line. S2-4. **Outcome (S2-4, 2026-10-04): on.** Hit rate 0.950, MRR 0.7958, recall 0.8583, against dense 0.800, 0.5875, 0.7167; 8 questions won, 4 lost (still hits), 1.4 s per answer on CPU. |
| DEC-17 | Incremental ingestion: change detection, failure handling, write safety | **Resolved 2026-10-02: a sha256 manifest beside the index, generation directories published by an atomic symlink flip, one writer** (the flip replaced a two-rename swap in the second review) | **Updates:** only added and changed documents are re-embedded. A failed document keeps its chunks, which dissolves DEC-13. **Full rebuild** when the embedder identity or the splitter changes. Device keys are left out of the identity, so Colab's `_cuda` equals local `_cpu`. **IDs and metadata:** uuid5 chunk IDs, ready for Qdrant (DEC-3). `source` becomes corpus-relative, plus `source_sha256` and `ingested_at` (SRS §7.3). **Query side:** `rag-query` refuses a mismatched index. **Breaking:** re-ingest once. S2-6. **As implemented (2026-10-05):** `multi_process` stays in the identity, since langchain-huggingface 1.2.2's multi-process path drops `encode_kwargs` (maintainer). An empty `model_kwargs` counts as none. Generation stamps carry microseconds and always sort after the live one. Only a folder holding exactly `index.faiss` and `index.pkl` is migrated as v0.2, and anything else at the path exits 2. Every update opens the live index, and an update probes the embedder's dimension. ARCHITECTURE.md DEC-17 has the dated notes. **S3-4 (2026-10-21):** a splitter change no longer rebuilds in full. Each document records the splitter that chunked it (its extension's own, or the default), and a document whose splitter changed re-embeds, as one whose loader changed does; the manifest's top-level splitter decides nothing. |
| DEC-18 | HTTP API shape: SSE library, auth, concurrency, ingest jobs | **Resolved 2026-10-02: FastAPI ≥ 0.135 native SSE; a token is required off localhost (maintainer); one generation at a time** | **Surface:** `rag-serve`. `QueryRequest` is `{question}` with `extra="forbid"`, so the API never supplies components, and a `trust_remote_code` load guard comes with it. **Load:** 503 with `Retry-After` when busy. A disconnect cancels generation: under uvicorn (ASGI 2.3) Starlette cancels the stream, and `stream_answer` runs inside it, with no producer task (second review). Requests with an `Origin` header are refused. **Ingest:** single-flight jobs in memory, which swap only the retrieval half. **Privacy:** no host paths over HTTP. S2-7. **As built (S2-7, 2026-10-12):** FastAPI's native SSE runs the generator in its own producer task. The cancel still reaches `stream_answer`, and the slot's teardown closes a stream that a drop between events left paused (the spike's window). Retrieval is serialized by a lock, not the slot. The API opens the index itself (`chain.open_retrieval`), and `require_index` is gone. `rag-serve` checks Ollama before uvicorn starts. ARCHITECTURE.md DEC-18 has the dated notes. |
| DEC-19 | The Qdrant store: how an index is written, published, recovered and found | **Resolved 2026-10-14: a generation is a collection behind an atomic alias switch; the manifest lives in the collection's metadata; our own module on `qdrant-client`** | Qdrant v1.19.2, `qdrant-client` 1.19.1 (resolves with the stack; adds grpcio, protobuf, portalocker, h2; no locked version moves). ADR-024's pattern carries over: build `<collection>--gen-<stamp>-<hex>`, then one `update_collection_aliases` call, then recovery keeps the target and the newest older one. Manifest version 2: identities plus per-document records, with chunk counts in place of ID lists. Search is always exact, by cosine (MiniLM's vectors are unit length). `paths.state` (`var/`) holds the lock and S3-8's job database. **Breaking:** `paths.vector_store` and `RAG_VECTOR_STORE_PATH` are refused at load; a scratch index is `RAG_QDRANT_COLLECTION`. **Quickstart:** `docker compose up -d qdrant` first; local mode with `path=` is not offered (its folder lock admits one process). Parity with FAISS is proven by a contexts digest of all 25 golden questions, so no tier-2 re-run. **Design review 1 (2026-10-15):** readers resolve the alias again per question, since a generation is deleted two switches later; S3-2 writes manifest version 2 whole; a retry counts a create that finds our collection, or a delete that finds none, as done; S3-3 … S3-10 repeat the digest. **Design review 2 (2026-10-16):** the protocol held, killed at every step; our code makes every cut, since the server decides a tie by the collection's layout; with no alias (deleting the target deletes it), recovery deletes only incomplete generations; the manifest sits under one metadata key, written whole; and `var/` is shared under one uid. **As implemented (S3-2, 2026-10-19/20):** parity with FAISS proven, the contexts digest of all 25 golden questions identical (`5a36e2f9…`), so no tier-2 re-run. A generation is created after the embedding, so a kill while embedding leaves nothing to recover. A retriever takes `k` only: the schema refuses FAISS's other options. Readers resolve the alias per question (`build_retrieve`), the API per request, and go not-ready when the alias cannot be read; `rag-eval generate` holds a generation it moves to against its corpus scan. `check()` refuses a server older than 1.16 or an answer that is not Qdrant's (exit 2), and a switch whose answer was lost is read back. The writer lock excludes only writers that share `var/` (caveat 36). Local mode differs from the server (a failing alias batch applied halfway, vector precision), so those properties are `qdrant_server` tests, which CI runs against its service container. S3-2. ARCHITECTURE.md §3.1. |
| DEC-20 | Hybrid retrieval: where the sparse vectors come from, how they are fused, how it is switched on | **Resolved 2026-10-14: our own BM25 encoder; Qdrant prefetch + RRF with an explicit constant; then the cross-encoder; tier 1 decides, S3-11 switches** | `rag_qa/sparse.py`, about 80 lines with no dependency; IDF applied by Qdrant. Rejected: fastembed's `Qdrant/bm25` (onnxruntime, a model download, a tokenizer that moves with releases) and Qdrant's own BM25 (needs `cloud_inference=True`, which local mode refuses; its tokenizer changes with the server). Qdrant's default RRF constant is 2, so `rrf_k` is always set (60). A `sources` filter by document path (API and CLI) is FR-3's metadata filtering. The real config sets no sparse encoder until S3-11b, so nothing hashed moves before the batched re-run. **Design review 1:** Qdrant's server breaks no RRF ties, so the fusion returns its whole union and our code cuts it, ties by point ID; S3-5 runs hybrid tier 1 twice. **Design review 2 (2026-10-16):** each prefetch still cut its own list on the server, which kept other tied points for each insertion order and after a restart. So the two branches run as separate searches, each cut by our code, and our code fuses them by RRF. S3-5's second run follows a restart. S3-5, S3-11, S3-11b. |
| DEC-21 | NFR-6 and the hosted fallback | **Resolved 2026-10-14: timeouts, safe retries and circuit breakers on network calls; an opt-in OpenAI-compatible fallback, with one real check** (maintainer) | Ollama: connect 5 s, read 600 s; retries only before the first token. Qdrant: 10 s; every call retried (all idempotent). Our own breaker per dependency; open means 503 with `Retry-After` before the first byte. The fallback (`pipeline.query.fallback_llm`, a `fallback` extra with `langchain-openai`) answers only while Ollama's breaker is open or after a connection error before the first token; streamed itself, never `.with_fallbacks()`; its key from the environment only; never used by tier 2. It absorbs the backlog's reopen back-off. **Design review 1:** with a fallback configured, an open Ollama breaker routes to it, and the start-up checks warn instead of exiting 2; retry semantics are stated per call; the retry before the first token is a hand-written loop. **Design review 2 (2026-10-16):** Qdrant's breaker wraps each attempt, with one deadline per retrieval, since retrieval holds the retrieval lock; a 404 is never retried; the fallback sets `max_retries: 0` and a timeout, as the OpenAI SDK retries twice on its own. S3-6. |
| DEC-22 | NFR-10: logs, traces, metrics, alerts | **Resolved 2026-10-14: structlog; OpenTelemetry spans we create ourselves; prometheus-client; burn-rate alerts; Jaeger** | No LangChain callback handlers: one that suspends reopens DEC-14's cancel window. Spans are never current across a `yield`, and end in a `finally`. Metrics on their own port. Alert rules tested by `promtool` in CI. NFR-1 by a 100k synthetic benchmark; NFR-3 reported by `rag-ingest`. **Design review 2 (2026-10-16):** it held. With the instrumentation off, excluded or full, a client dropped during 30 s of silence cancelled the stream within milliseconds, and every span ended. promtool runs from the Prometheus image, by digest. S3-7. |
| DEC-23 | The API beyond localhost (DEC-18's Phase 3 half) and NFR-5 | **Resolved 2026-10-14: named, scoped API keys (maintainer); token-bucket rate limits; a 64 KiB body limit; job records in SQLite; a stateless query tier; TLS at the proxy** | A keys file of sha256 hashes with scopes `query` and `ingest`, managed by `rag-keys`; `RAG_API_TOKEN` still works as one all-scope key. A job holds a lock of its own, so an unfinished row whose job lock is free reads as abandoned. Limits per replica, documented. **Design review 1:** the job's own lock replaces probing the writer lock, which could call a live job abandoned and make a starting writer exit 2; the health routes are exempt from rate limits; failed authentication is limited to 10 a minute per address. **Design review 2 (2026-10-16):** `X-Forwarded-For` is trusted from Caddy's own address only, since uvicorn trusts loopback by default; the rate tables are bounded; the keys file lives in `var/`, mounted as a folder, and fails closed; "no credential" means no key and no token; a streamed body over the limit raises `HTTPException(413)`; and the abandoned-job rule reads the row again after the probe. S3-8. |
| DEC-24 | Containers, compose, overlays and CI scanning | **Resolved 2026-10-14: one image (uv, CPU torch, models baked in, offline), one compose file with Caddy for TLS and balancing, `RAG_CONFIG_OVERLAY`, Trivy run as a binary pinned by sha256** | Not `aquasecurity/trivy-action` (tags hijacked, CVE-2026-33634). SIGTERM ends `rag-serve` with 143 under an init, and that stays. The audit policy: ignores only for advisories with no fix, each with a reason and a date; a weekly run; the `eval` extra audited. A ruleset makes every stage block a merge. **Design review 1:** the `eval` extra has four advisories, not one. The `langchain` bump moves into S3-9 (maintainer) and clears two; diskcache's and ragas's have no fix, and are the policy's first ignores. The repository is public, so rulesets are available. **Design review 2 (2026-10-16):** a signature would not have caught Trivy's pipeline-built v0.69.4, so every pin is verified once, by provenance pinned to its workflow and tag, its source commit on `main`, and an age rule. promtool runs from the image, by digest, and the weekly run scans the third-party images. Compose runs one uid, Qdrant's unprivileged image, and Caddy as the only trusted proxy; the disconnect through Caddy held. S3-9. |
| DEC-25 | Backup and restore | **Resolved 2026-10-14: snapshots of immutable generations to a directory; a restore is a new generation and an alias switch** | Checksums verified both ways; a snapshot from another embedder is refused; RPO and RTO stated; tested against a real server in CI and by drills (maintainer's addition). **Design review 1:** a restore holds the writer lock from the upload to the end, or an ingest could delete its generation mid-restore; the sidecar keeps the whole manifest. **Design review 2 (2026-10-16):** the manifest survived a real round trip. Restore refuses a taken name, since an upload overwrites in place, and runs `check_index`'s whole rule. Snapshot files outlive their collections, so restore, `create` and `recover` delete them. The backup service mounts `./var`. S3-10. |
| DEC-26 | Tier-2 re-runs in Phase 3 | **Resolved 2026-10-14: one, batching every lever; S3-11 measures them and S3-11b commits** | The judge (from the hand labels), the prompt fixes and LLM01's delimiting, the hybrid switch, `min_score`, the model (DEC-2), `check` recomputing aggregates, and a per-question context digest. Staged so the grid stays small: about 4–6 h of T4 time plus about 70 min here. **Design review 1:** split into S3-11 (the sweep) and S3-11b (the commit), as S2-5 and S2-5b were (maintainer); the `langchain` bump moved to S3-9; the answers file records Qdrant's version. **Design review 2 (2026-10-16):** a stage 0 measures the T4 against this CPU; stage 1's re-score is certain; every rule is a margin committed before the first run; Colab restores this host's index and runs Ollama only; about 5–9 h of T4 time, where this row first said 4–6; and about 5 labels outside the 20 after S3-11b. S3-11, S3-11b. |
| DEC-27 | NFR-2: revise the SLO, serve from a GPU, or a hosted model | **Resolved 2026-10-14: revised** (maintainer) | (a) the `sources` event in under 2 s at p95 with the model idle; (b) the time to first token reported per class of hardware, this CPU's from the committed run and a T4's from S3-11. A GPU host later is a `base_url` edit. Resolves ADR-016 at the amend session, which S3-12 runs. **Design review 1:** (a) is measured as a p95 over the golden questions (S3-7, and S3-11b if hybrid retrieval is on). |
| DEC-28 | `ruff format` | **Resolved 2026-10-14: format the code once and gate it** (maintainer) | 30 Python files, about 860 lines, in one commit of S3-1; Markdown excluded (ruff 0.16 would rewrite examples in two docs and two closed stories); then a CI gate and `.git-blame-ignore-revs`. **Delivered in S3-1 (2026-10-16):** the reformat is `01854a1`, 30 files and 868 lines, alone; the gate is the `check` job's Format step, after Lint. The exclude holds when ruff walks a folder, not for a Markdown file named on the command line (backlog line). |

## Backlog

Reconciled at the Phase 1 exit (S1-8, 2026-10-02). Every Phase 1 line is either
**closed** by the story named, or **re-deferred** with a phase and a reason. The
detail behind each closed line is in its story file.

Reconciled again in the Phase 2 architecture pass (2026-10-02). Every open line now
leads with one of two things:

- **its owning Phase 2 story**, as **S2-n —**, where the detail below is that story's
  input;
- **its phase or trigger**, when it stays deferred.

Reconciled at the Phase 2 exit (S2-9, 2026-10-13). Every Phase 2 line is closed by the
story named, or re-deferred to Phase 3, or to a named trigger, with a reason. A line
reading **Phase 3 —** is an input to the Phase 3 architecture pass.

Reconciled in the Phase 3 architecture pass (2026-10-15). Every line that read **Phase 3 —**
now leads with its owning story, **S3-n —**, or with the decision that settled it, and the
detail below is that story's input. The lines waiting on a trigger keep it: the ragas import
shim (ragas 0.4.3 is still the newest release, checked 2026-10-14), the timing-based Ctrl-C
tests, a stalled CI step, the CUDA 12 fallback and torch parity. Design review 1 (2026-10-15)
diffed this backlog against `main`'s: no line was dropped.

### Opened in the Phase 3 architecture pass (2026-10-15)

- **After v1.0, or inside S3-11 if a candidate needs it — upgrading Ollama** past 0.22.1
  (0.40.4 is current). Every digest changes, and answers may, so it rides a tier-2 re-run.
  ADR-025's portable digests hold for one Ollama version, which Colab's recipe pins.
- **After v1.0 — ingestion on several hosts.** The writer lock and the job records live on
  one host's state volume (DEC-19, DEC-23). A second ingesting host needs a lock both can
  see. So does a second checkout on one host, which has a `var/` of its own (caveat 36): a
  lock Qdrant itself holds would cover both.
- **After v1.0, for a much larger corpus — a clone from a snapshot.** An update copies the
  live generation point by point, linear in its size (DEC-19): about 1–2 s here.
- **If tier 1 shows misses it would fix — stemming in the BM25 encoder** (DEC-20). A
  `stemmer` option and a `SPARSE_VERSION` bump.

### Opened in Phase 3 stories

- **S3-9 — `ubuntu-latest` becomes Ubuntu 26 from 2026-10-19** (found in S3-2's CI run
  39283008142, as an annotation). Both CI jobs run on it, and the Qdrant service container
  and the system tools come from the image. S3-9, which pins the CI images, pins the
  runner too, or records why not.

- **If a tool formats files by name (an editor's format-on-save, a pre-commit hook) —
  `force-exclude = true` under `[tool.ruff]`** (found in S3-1). `[tool.ruff.format]`'s
  `exclude = ["*.md"]` applies only while ruff walks a folder, so `ruff format docs/ADR.md`
  would still rewrite its Python examples. CI's bare `ruff format --check` is unaffected,
  and CLAUDE.md has sessions run it bare for this reason (S3-1's second review, round 2).

- **S3-6 — a Qdrant read timeout ended a tier-1 run with Qdrant up** (found in S3-3).
  `rag-eval retrieval` retrieved 9 of 20 questions, then a query hit the client's 10 s
  timeout: exit 2, and "Is Qdrant running?". Not reproduced, and the host showed no
  pressure. Qdrant's access log and telemetry record only completed requests, so they show
  that the query never completed, not whether it arrived (S3-3's second review). DEC-21's
  retries cover it. S3-6 can use it as a real case, turn on `httpx`/`httpcore` debug
  logging to tell a lost request from a stalled one, and decide whether a timeout with the
  server up should say so rather than ask if it runs.

- **Any story touching ingestion's log — "Published …: 1 documents"** (found in S3-4's
  drill). `_update`'s publish line does not agree in number for one document, where S2-8
  fixed the other lines. Cosmetic; no hashed input.

- **Any story touching `exit_status` — name `max_file_mb` when it emptied the run** (found
  in S3-4's first review). When every file is above the limit, the run exits 1 with "No
  documents were loaded."; the cause is only in the log lines above it. The line is shared
  with the API's job report, so the wording is a small contract. Cosmetic; no hashed input.

- **S3-8 — the Qdrant URL in an ingest job's error text** (found in S3-2's second review).
  `describe_error` passes `cannot use Qdrant at http://host:6333 (…)` through to
  `GET /v1/ingest/<job>` clients. Not a path, so DEC-18 holds; once S3-9 puts Qdrant on an
  internal network, decide whether to redact it.

### Closed in Phase 3

- **S3-1 (PR #19):** Phase 3 opened at 1.0.0.dev0, and the `ruff format` line closed
  (DEC-28). The reformat is `01854a1`, alone, and CI's Format step keeps the tree formatted.
  It opens one line: the exclude and a file named on the command line.
- **S3-3 (2026-10-20):** the parity tests against the `langchain-community` loaders, and
  the corpus PDF line that rode with them (found in S1-3). DEC-5's exit is complete.
- **S3-4 (2026-10-21):** per-loader splitter strategies (SRS §7.2), as
  `pipeline.ingestion.splitters`; and SRS §7.1's declared limit, `max_file_mb: 50`. It
  opens one line: the publish line's "1 documents".

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
- **S2-8 (PR #16):** the CLI polish batch. It closes the "CLI polish" line below, all
  eight of its items.
  - It opens two lines: the log's plural slips, and the API's ingest jobs printing pypdf's
    warnings.
  - S2-9 owes §2.2's `components` line.
- **S2-6:** incremental ingestion (FR-2, ISS-14, NFR-4 in place, SRS §7.3, DEC-17).
  - Three backlog lines close with it: walk pruning, the §7.3 chunk metadata, and the
    embedder check on open.
  - S2-8 gains two items: a text file's summary line reads "1 pages/sections", and the
    summary now has more zero-count lines.
  - S2-9 owes README's status line, which still lists incremental ingestion as
    planned.
- **S2-9:** the Phase 2 exit. It paid every doc debt above: README's status line and
  limitations, §2.2's module and dependency lists, §2.8 with DEC-18's notes, §1.1's DEC-7
  annotation and ADR-007's moot exception. Its end-to-end run found one bug and fixed it:
  `rag-serve`'s Ctrl-C traceback and exit 130.

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

- **S3-9 — the Docker image installs with `uv sync` or `uv export`, never
  `pip install .` (found in S1-7's review).** `pip install .` and
  `uv pip install .` ignore dependency groups, so they skip the `cpu` group and
  pull torch from PyPI as the CUDA build, with GBs of `nvidia-*` wheels. `uv sync`
  and `uv export` honour the default groups. A constraint for the Phase 3
  container story.
  - **And its stop signal (S2-9).** `docker stop` sends SIGTERM. `rag-serve` shuts down
    gracefully on it, then uvicorn ends the process by that signal: status 143, no
    traceback. That is conventional, and only Ctrl-C promises exit 0. The container story
    decides whether the entrypoint should map SIGTERM to 0.
  - **And a forced stop (S2-9's review).** A second Ctrl-C during the 5 s graceful wait
    (uvicorn's log offers "CTRL+C to force quit") does not shorten it. It skips the
    lifespan's shutdown, which closes the model clients and reports an abandoned ingest
    job, and uvicorn logs the cancelled tasks as errors; the exit is still 0. Harmless at
    exit. Only a repeated SIGINT forces: a second SIGTERM does not, so `docker stop`
    (SIGTERM, then SIGKILL after 10 s) never takes this path.
- **S3-9, decided in DEC-24 — audit-gate policy (S1-8 review).** CI's Audit step does a live
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
- **Closed by S2-8 (2026-10-13) — CLI polish (found in S1-4).** Every item below is done;
  the two the story left alone are separate lines (the log's plurals, and the API's pypdf
  warnings). The history below is kept. No single item is worth a story, so they are
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
- **S3-7 — two more plural slips in `rag-ingest`'s log** (found in S2-8): "Embedding 1
  chunks..." and "Published …: 1 documents, 1 chunks.". S2-8 fixed the ones on its list:
  "1 page/section", and the summary's "1 chunk". Seen again in S2-9's API job log.
  **Re-deferred at the Phase 2 exit:** cosmetic, and Phase 3's move to structlog (NFR-10)
  rewrites these messages anyway.
- **S3-7 — the API's ingest jobs print pypdf's warnings** to `rag-serve`'s stderr (found
  in S2-8). `rag-ingest` sets the `pypdf` logger to ERROR for its own run only, as it does
  its log handler. **Re-deferred at the Phase 2 exit:** the fix is logging configuration,
  which the same structlog move owns.
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
- **S3-11 (measured) and S3-11b (committed), the batched re-run (DEC-26) — what the
  baseline's answers and hand check found.**
  Each fix is a change to measure with this harness (a re-run, caveat 24), not to make by
  eye. **Re-deferred at the Phase 2 exit:** each lever (the prompt, `min_score`, the model)
  moves a hashed input, so batching them into the DEC-2 revisit's sweep costs one tier-2
  re-run, not one per fix. Hand-labelling faithfulness needs no re-run, and decides
  whether the judge stays; it is a good first input to the Phase 3 pass.
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
- **S3-9, the bump and the policy (design review 1) — the `eval` extra's advisories.**
  Design review 1 found four, all published before the Phase 3 pass:
  - `langchain` 1.3.2, PYSEC-2026-2192, fixed in 1.3.9;
  - `langgraph-sdk` 0.3.15, CVE-2026-104873, fixed in 0.4.4;
  - `diskcache` 5.6.3, PYSEC-2026-2447, with no fix;
  - `ragas` 0.4.3, PYSEC-2026-3046, with no fix.

  S3-9 bumps `langchain` to 1.4.3 (maintainer), which clears the first two and moves five
  packages: websockets drops to 16.1.1 in the main environment, which still audits clean. The
  other two become the policy's first ignore entries, each with its reason. The history
  below is kept, as first found: `langchain` 1.3.2 has
  PYSEC-2026-2192 (fixed in 1.3.9; found in S2-7's audit). ragas pulls it in. CI never
  installs the extra, so its audit never sees it. Bump it with the next ragas change
  (which re-scores), or decide whether the extra should be audited at all (the audit-gate
  policy line above). **Re-deferred at the Phase 2 exit:** the extra runs only on a
  maintainer's machine or a notebook, against local Ollama, and a bump alone costs a
  re-score.
- **S3-11 — `rag-eval check` recomputes the aggregates from the items** (S2-5b's second
  review). It reads `aggregate` and `failures` from `generation-latest.json` as written.
  `score` writes them consistently, and the baseline recomputes exactly. But a hand edit,
  or a merge conflict in that file resolved by hand, could pass a gate its items fail. A
  few lines in `run_check`: refuse a file whose aggregates disagree with its items.
  **Re-deferred at the Phase 2 exit:** no hand edit has happened, and the DEC-2 revisit,
  which rewrites that file, is the natural story for it.
- **When a ragas release stops importing `langchain_community.chat_models.vertexai` —
  delete the import shim** in `ragas_scoring.import_ragas` (S2-5,
  vibrantlabsai/ragas#2741). `tests/test_eval_scoring.py` fails then and says so. A
  ragas upgrade also means bumping `fingerprint.RAGAS_VERSION`, which `uv.lock`'s test
  enforces, and re-scoring.
- **Closed by S2-5b (2026-10-11) —** the judge's `num_thread`: not set. The baseline scored
  on a Colab T4, where every layer runs on the GPU. A local CPU re-score would add
  `PARAMETER num_thread 10` to `eval/judge.Modelfile`, which moves `judge`.
- **If either fails in CI — two timing-based Ctrl-C tests in `tests/test_cli.py`** (found
  in S2-5's second review). Not seen failing again through the Phase 2 exit. `test_ctrl_c_while_a_chunk_is_processed_leaves_no_stream_open_at_the_prompt`
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
- **Closed by S3-3 (2026-10-20) — the parity tests and the corpus PDF they read** (found
  in S1-3). The parity tests are deleted with `langchain-community`. The PDF stays in
  `corpus/`, where it is corpus content (README and the tier-2 run cite it); no test reads it.
- **On the first real GPU run — CUDA 12 fallback (found in S1-7).** `cu130`
  wheels need a recent NVIDIA driver, and this host cannot check that. If
  Colab/Kaggle refuse, add a `cuda12` variant on `cu128`, which caps torch at 2.11.
  This is triggered by an event, not by a phase. S3-11 installs no CUDA torch on Colab
  (design review 2), so the sweep does not trigger it.
- **Accepted, no action — torch parity between the variants (found in S1-7).**
  CUDA resolves `2.14.1+cu130`, CPU stays at `2.14.0+cpu`.
  `uv lock --upgrade-package torch` aligns them if it ever matters.

### Re-deferred or dropped in the Phase 1 architecture pass (2026-09-25)

- **Dropped — device selection as one env-driven setting.** The item assumed
  `pydantic-settings` interpolates `device: ${RAG_EMBED_DEVICE:-cpu}` inside
  YAML values; it does not (no `${VAR:-default}` expansion for arbitrary
  values). The CUDA torch variant (DEC-12) makes the switch one line anyway. See **DEC-10**.
- **S3-6, decided in DEC-21 (an opt-in seam plus one real check, maintainer) — hosted-LLM fallback** (`groq_llama3` entry behind an optional
  extra + `.with_fallbacks()` in `chain.py`). Moved out of Phase 1: it widens
  the DEC-1 import surface and needs an API key, for something Phase 1 could
  only exercise against a fake failing LLM. It belongs with the Phase 3 serving
  decision (NFR-2), which is the problem it actually solves.
- **S3-6, decided in DEC-21 — NFR-6** timeout / retry-with-backoff / circuit-breaking, per
  **DEC-9**.
- **Closed by S3-1 (2026-10-16), as DEC-28 decided (format the code once, Markdown excluded, and gate it) — `ruff format`.** Not gated in Phase 1: the tree is
  unformatted and a whole-tree reformat is the kind of diff that gets rubber-stamped
  (DEC-11). **Re-deferred at the Phase 2 exit:** a phase boundary is the one time no
  story branch is open to conflict with it, so that pass decides: reformat once, alone,
  and gate it, or drop the line.
- **Closed by S3-4 (2026-10-21), as `pipeline.ingestion.splitters` — per-loader splitter
  strategies.** A changed splitter re-embeds only the documents it chunks, and an empty
  map moves no fingerprint part, so neither cost below remains. (SRS §7.2): chunking
  configurable per document type. The `_target_` mechanism already supports it;
  it needs more splitter entries and a per-loader default, not new architecture.
  **Re-deferred at the Phase 2 exit:** a splitter change moves tier 2's `ingestion` part
  and forces a full re-embed, and the move to Qdrant re-ingests everything anyway.

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
- **Decided in DEC-27 (the SLO is revised); its numbers come from S3-7, S3-11 and S3-11b —** NFR-2 (<2 s first token) is unachievable CPU-only: revise the SLO
  or plan GPU serving in the Phase 3 arch pass. Phase 2 measures it: `ttft_ms` on
  every answer (DEC-14) and in every tier-2 run. Since S2-4 the rerank sits before the
  first token: about 1.4 s of every `ttft_ms` on this CPU, plus 2.5 s of model loading
  at startup. **S2-9's numbers:** first token at 154.6 s over HTTP, on `num_thread` 10;
  and while the runner holds 10 of 12 threads, the API's own retrieval slows from 0.78 s
  to 7 s (caveat 26). A serving box would need headroom for both.
- **S3-5 (a tier-1 table of cutoffs), S3-11 (the choice) and S3-11b (the setting) — tune the reranker's `min_score`.** **Re-deferred at
  the Phase 2 exit:** a cutoff changes what reaches the prompt, so it moves `query` and
  needs a tier-2 re-run, which the revisit's sweep provides. CLI "Sources" lists what was
  retrieved, not what the answer used, so a refusal still shows a source. S2-4 added the
  knob (DEC-16), off by default. It is in the model's own units: logits, from -6.42 to
  8.38 over the kept chunks of the golden questions. Tuning it with the eval harness
  comes after the tier-2 baseline exists.
- **Closed by S2-7 (2026-10-12) —** `ChatOllama` leaves its HTTP client open
  (`ResourceWarning: unclosed socket` at exit). S2-1 closed the CLI half: `aclose_llm`,
  called at exit. The API's lifespan owns the client and closes it at shutdown.
- **S3-6, absorbed by DEC-21 (an open Qdrant breaker short-circuits the reopen) — a back-off after a failed
  index reopen** (S2-7's second review). **Re-deferred at the Phase 2 exit:** it is the
  same concern as the health check's load, the next line: what an unauthenticated probe
  may cost.
  - **Now:** after a reopen fails, each later request tries again once. Concurrent requests
    share one attempt.
  - **The proposal:** wait a few seconds after a failure, so probes arriving one by one
    during an outage (an unreadable disk, say) cost nothing.
  - `/v1/health` refreshes without a token. Not built.
- **S3-8 — request size and health-check load** (found in S2-7).
  - FastAPI reads a request body whole before validating it, so a huge body costs memory
    before the 422.
  - `/v1/health` is unauthenticated and asks Ollama on every call.

  Both belong with the rate limiting.
- **DEC-5 exit tasks, one per phase:**
  - own loaders (**Phase 1 → S1-3**);
  - the cross-encoder via `sentence-transformers` (**Phase 2 → S2-4**, done 2026-10-05);
  - dropping `langchain-community` and `langchain-classic` from `pyproject.toml`
    after the Qdrant move (**Phase 3 → S3-3**, done 2026-10-20).

  Note: ragas pulls both into the `eval` extra, so Phase 3 removes them from our
  direct dependencies only.
- **S3-8, shown with two replicas in S3-9 — statelessness (NFR-5).** The Phase 2 API keeps ingest jobs in memory
  and the index in process (DEC-18). A horizontally scalable query tier needs Qdrant
  and a durable job queue.
- **S3-8 (keys, scopes, rate limits) and S3-9 (TLS at Caddy) — API deployment hardening:** rate limiting, TLS and multi-user auth.
  DEC-18 is localhost-first, with an optional bearer token.
- **S3-11, the tier-2 sweep (DEC-26) — the DEC-2 revisit.** Run mistral vs phi3 (and `qwen2:7b`, if pulled)
  through the tier-2 harness. It is hours-scale, so it is a Colab/Kaggle sweep. Its
  result feeds the Phase 3 serving decision, and the tier-2 follow-ups above ride the
  same re-run.
- Phase 2 was sharded on 2026-10-02 (S2-1 … S2-9) and delivered on 2026-10-13. Phase 3
  was sharded on 2026-10-15 (S3-1 … S3-12), after its architecture pass; design review 1
  split S3-11b off the same day.

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

**Phase 2 — shipped as `v0.3` (`aa8c19b`, 2026-10-14).**
S2-1 answer events, Ctrl-C cancels generation · S2-2 golden set · S2-3 tier-1 retrieval
gate in CI · S2-4 own cross-encoder reranker, on · S2-6 incremental ingestion, generations
and a symlink flip · S2-5 tier-2 harness and freshness gate · S2-5b tier-2 baseline,
enforced · S2-7 HTTP API with SSE · S2-8 CLI polish · S2-9 end-to-end over HTTP, docs,
release. As delivered: ARCHITECTURE.md §2.8; the decisions that generalize are ADR-022 to
ADR-026; per-story detail in `stories/phase-2/`.

**Architecture passes.**

| Phase | Confirmed | Decisions | Sharded into |
| --- | --- | --- | --- |
| 0 | 2026-09-18 | DEC-1, DEC-2, DEC-4 (later DEC-5) | — |
| 1 | 2026-09-25 | DEC-6 … DEC-12 | S1-1 … S1-8 |
| 2 | 2026-10-02 | DEC-14 … DEC-18 | S2-1 … S2-9 |
| 3 | 2026-10-15 | DEC-3, DEC-9 resolved; DEC-19 … DEC-28 | S3-1 … S3-12, with S3-11b split off in design review 1 |

The Phase 2 pass ran on Opus 5.5 by the maintainer's choice; ADR-018 routes
architecture passes to Fable. The Phase 3 pass ran on Opus 5.5 at max effort, since Fable is
unavailable (CLAUDE.md, "Model routing").
