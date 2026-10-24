# S3-12: Phase 3 exit: end to end on the stack, docs, release 1.0.0

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | — (the Phase 3 exit, and v1.0: SRS complete); the SRS amend session (revision 2.0) |
| **Depends on** | S3-1 … S3-11b |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `chore/s3-12-phase-3-exit`. Low code risk; it ends in the release tag. |

## Goal

The exit criterion is shown, not asserted: every row of the gap-closure table in
ARCHITECTURE.md's Phase 3 intro is run from a fresh clone, on the compose stack. The SRS is
amended to the system as delivered, in the explicit session WORKFLOW requires (maintainer,
design review 1). The docs then describe v1.0 as delivered, and 1.0.0 is released and tagged
`v1.0` with CI green on the tag, as each phase exit was.

## Scope

- **The end-to-end run**, from a fresh clone of `main` in `~/dev/rag-s3-12-e2e`, logged with
  `tee -i`. First stop this checkout's Qdrant (`docker compose stop qdrant` here): the
  clone's stack publishes the same port, and a host command in the clone would otherwise
  write this checkout's `rag_qa` alias under a writer lock of its own (caveat 36, S3-2's
  second review):
  1. `cp .env.example .env`, keys made with `rag-keys`, `docker compose up -d`, and every
     service healthy;
  2. an ingest over HTTPS through Caddy, then a question answered over SSE, then the same
     through `rag-query` on the host. The host shell loads `.env` first, for Qdrant's key
     (caveat 32), and one Ollama is stopped before the host run: `rag-query` talks to the
     host's, while the stack's holds mistral too (caveat 8);
  3. `--scale app=2`: both replicas serve the new generation, and either one reports the
     job;
  4. a client dropped through Caddy, during the prefill and mid-stream: our connection to
     Ollama closes within about 1 s each time (DEC-14);
  5. Ollama stopped: a 503 with `Retry-After` before the first byte, and a recovery when it
     returns;
  6. a backup, the live generation deleted, a restore: tier 1 and the contexts digest are
     unchanged (DEC-25). The previous generation survives the loss (DEC-19), and Qdrant's
     `snapshots/` folder holds nothing of ours afterwards (design review 2);
  7. a `kill -9` during an ingest: the live generation is untouched, and the next run
     recovers (§11, resilience);
  8. the observability profile: a trace in Jaeger, the metrics scraped, and the alert rules
     loaded;
  9. the gates: `rag-eval check` exits 0, and tier 1 meets its floors.
- **The gap-closure table,** filled in with each row's evidence, in this file.
- **Docs.**
  - **ARCHITECTURE.md:** §3.8, "As delivered", indexing where the delivery differs from
    §3.1–§3.7, with the stories that record it, and the Phase 3 header marked delivered.
  - **ADR.md:** new ADRs for the decisions that generalize, for example generations behind
    an alias, a single batched re-run per phase, our own spans over callbacks for
    cancel-safety, security tools pinned by a hash we hold, and identities that hash only
    what is set. Statuses updated for ADR-007, ADR-011, ADR-013, ADR-014, ADR-016 and
    ADR-024.
  - **README:** status, requirements, the quickstart with Qdrant, upgrading from v0.3 (every
    step), deployment, observability, backup and restore, capacity planning, known
    limitations and the roadmap.
  - **CLAUDE.md:** the environment facts re-verified, the release line, the commands.
- **The SRS amend session** (maintainer, design review 1), WORKFLOW's explicit one, in a
  commit of its own before the release commit.
  - **The list:** Phase 2's differences (the Phase 2 intro), Phase 3's (the Phase 3 intro),
    and a sweep of the ADRs and §1.8 for earlier departures that neither phase list holds.
    Among those are ADR-008's hand-composed chain and DEC-14's stream in place of §6.3's
    `create_retrieval_chain`, §6.3's allowlist prefixes, and §10's coverage targets.
  - **NFR-2 (a)'s target** (DEC-27; added at S3-7's close-out): S3-7 missed it (a p95 of
    2.09 s, then 2.02 s), and S3-11's stage 2 owns the levers. If S3-11b's `sources` p95
    still misses 2 s, the amendment sets the target from that measured p95.
  - **The amendment:** the maintainer approves the merged list, and `docs/SRS.md` becomes
    revision 2.0, with a revision block naming each change.
  - **If the session runs long,** the release moves to S3-12b, as S2-5b split from S2-5.
- **The backlog** is reconciled: every Phase 3 line is closed by its story or re-deferred
  with a reason.
- **Release:**
  - 1.0.0 in `pyproject.toml` and `__init__.py`, then `uv lock`;
  - the tag recipe for the maintainer: annotated `v1.0` on the merge commit, then a push;
  - CI green on the PR, on `main` and on the tag.

## Out of scope

- New features. Anything the end-to-end run finds is either fixed here, as S2-9 fixed
  `rag-serve`'s Ctrl-C, by the maintainer's call, or recorded for after v1.0.

## Verification

```bash
git clone <repo> ~/dev/rag-s3-12-e2e/fresh && cd ~/dev/rag-s3-12-e2e/fresh
# the nine steps above, each with its output pasted here
uv run rag-eval check; echo "exit $?"                     # 0
grep -n 'Version\|Revision 2.0' docs/SRS.md | head -3     # 2.0, with its revision block
grep -n '^version' pyproject.toml                          # 1.0.0
gh run list --limit 5                                      # green: the PR, main, the tag
```

## Review notes for the human

- **The gap-closure table:** every row needs evidence pasted here.
- **The SRS amendment:** you approve the merged list before `docs/SRS.md` changes; every
  difference in it is either amended or still listed.
- **The upgrade guide from v0.3,** followed literally on a v0.3 checkout, as S2-9 did for
  v0.2.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
