# S0-7: Project docs into `docs/`, invert the corpus rule

| | |
| --- | --- |
| **Status** | Done (2026-09-25) — commit pending, maintainer commits manually |
| **Closes** | DEC-4 (move half) |
| **Depends on** | S0-4 (the corpus rename must land first — `docs/` has to be free) |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

`docs/` means what it means in every other repository: project documentation.
The standing rule that made the old layout safe ("never put markdown in
`docs/`") is deleted rather than maintained, because after S0-4 the corpus
lives in `corpus/` and the name no longer lies. Runs before S0-6 so the
`v0.1` tag ships the final layout.

## Scope

- `git mv SRS.md WORKFLOW.md ARCHITECTURE.md ADR.md docs/` (creates the
  directory). `ADR.md` is the decision-record log added 2026-09-24; it is
  project documentation like the other three.
- `README.md` and `CLAUDE.md` stay at repo root: CLAUDE.md auto-loads from the
  working directory upward, and GitHub renders README from root only.
- `stories/` stays at repo root — it is working state, not documentation.
- **Invert the CLAUDE.md hard rule.** Replace the `docs/`-is-the-corpus bullet
  with: `corpus/` is the RAG document corpus, never put project docs there and
  never ingest repo docs; project docs live in `docs/`, except README.md and
  CLAUDE.md at root. Update the "Read these before doing anything" paths to
  `docs/SRS.md`, `docs/WORKFLOW.md`, `docs/ARCHITECTURE.md`.
- Fix every cross-reference to the three moved files. Known referrers:
  `CLAUDE.md`, `stories/STATUS.md`, all `stories/phase-0/S0-*.md`, `README.md`,
  and the moved files' references to each other. Relative links inside
  `docs/` become siblings (`SRS.md`), links from `stories/` become `../docs/SRS.md`.
- No content changes to the three documents beyond link fixes. In particular
  do not amend the SRS (WORKFLOW.md forbids casual SRS edits).

## Out of scope

- README rewrite / quickstart (S0-6 does that, against the final layout).
- Moving `stories/` or adding any new documentation.
- The corpus rename itself (S0-4).

## Verification

```bash
ls docs/                                  # → ADR.md  ARCHITECTURE.md  SRS.md  WORKFLOW.md
ls *.md                                   # → CLAUDE.md  README.md   (only these two at root)
git status --short | grep -c '^R'         # → 4 (moves detected as renames)
grep -rn "docs/" CLAUDE.md | head         # points at docs/SRS.md etc., corpus rule inverted
grep -c "corpus/" CLAUDE.md               # → ≥1 (the inverted hard rule)
# no dangling links to the old root paths, and no link to a file that does not exist:
grep -rn --include='*.md' -E '\]\((\.\./)*(SRS|WORKFLOW|ARCHITECTURE)\.md' . \
  | while IFS=: read -r f _ rest; do
      t=$(printf '%s' "$rest" | sed -E 's/.*\]\(([^)]+)\).*/\1/');
      [ -e "$(dirname "$f")/$t" ] || echo "DANGLING: $f -> $t";
    done                                  # → no output
uv run python -c "import rag_qa.registry; print('package untouched')"
```

## Review notes for the human

Two things only: that the three moves are renames rather than delete-plus-add,
so `git log --follow` still works, and that the inverted rule in `CLAUDE.md`
now says `corpus/` where it used to say `docs/`. A stale rule here is worse
than no rule, because every future session reads it as authoritative.

## Verification results (2026-09-25)

| Check | Result |
| --- | --- |
| `ls docs/` | `ADR.md ARCHITECTURE.md SRS.md WORKFLOW.md` |
| `ls *.md` at root | `CLAUDE.md README.md` only |
| Renames the commit will record | 4: `ADR`, `ARCHITECTURE`, `SRS` at R100, `WORKFLOW` at R090 |
| CLAUDE.md read list | all three paths now under `docs/` |
| CLAUDE.md corpus rule | inverted: names `corpus/`, sends project docs to `docs/` |
| Dangling markdown links | none |
| Ingestion data path | `corpus/`, which contains no `.md` files |
| Package imports | unaffected |

The three R100 renames double as proof that `SRS.md`, `ADR.md` and
`ARCHITECTURE.md` are byte-identical to their committed versions, so the
"no SRS edits" rule held.

**How renames were checked without staging.** Nothing is staged (the
no-staging rule in STATUS.md), so `git status` shows deletes plus untracked
files, and the story's original `git status | grep '^R'` check cannot pass
before commit. Instead the commit was built in a throwaway index
(`GIT_INDEX_FILE` pointing into the scratchpad), diffed with `-M`, and
discarded. The real index was confirmed empty afterwards.

## Discovered

- **Which references to change was a real decision**, not a mechanical
  find-and-replace. The rule applied: change a reference when someone will
  _act_ on it as a location, and leave it when it is a citation by name.
  Changed: CLAUDE.md's read-first list, and WORKFLOW.md's copy-paste prompt
  templates. Left: citations such as "ARCHITECTURE.md §0.4" in stories and
  the board, historical notes about the pre-rename `docs/` corpus, and
  everything inside `ADR.md`. Each doc name exists exactly once, so
  citations stay unambiguous, and rewriting dozens of them would be churn.
- **Two WORKFLOW.md templates were write instructions**, not just reads:
  "write it to ARCHITECTURE.md" and "it goes into ARCHITECTURE.md". Left at
  the root path, a future session following them literally could create a
  second, empty `ARCHITECTURE.md` at the root and split the design record in
  two. Those were the most important edits in this story.
- `ADR.md` is not in CLAUDE.md's read-first list, so a new session will not
  find the decision log unless something points it there. Adding a pointer is
  an editorial change outside this story's scope; left for the maintainer.
- `README.md` still documents a `docs/` of "Documentation and reports" and
  the deleted `v1/`/`v2/` layout. S0-6 rewrites it.

## Deviation from plan

One content change beyond link fixes: WORKFLOW.md's closing rule said
"`docs/` is the RAG corpus... Project docs live at repo root". The move makes
that rule false, and a false rule in a process doc is worse than none, so it
now names `corpus/` and sends project docs to `docs/`. No other wording in
the moved docs changed.
