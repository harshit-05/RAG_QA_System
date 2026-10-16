# S3-1: Open Phase 3: version 1.0.0.dev0, format the code once and gate it

| | |
| --- | --- |
| **Status** | Done (2026-10-16): PR #19, in three commits: `5281b0e` the version, `01854a1` the reformat, `a2bbe98` the gate, plus two commits of docs: the close-out, `177ffbf`, and the second review's round 2. CI was green on `a2bbe98` (run 38927352954), the new Format step included, and on `177ffbf` (run 38935030666); check `gh pr checks` on the head before the merge. Verification 1–4 are run and shown below. Merge it with a merge commit: `.git-blame-ignore-revs` names `01854a1` by its SHA. |
| **Closes** | DEC-28; backlog: `ruff format` |
| **Depends on** | — (the Phase 3 design reviews, merged) |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `chore/s3-1-open-phase-3`. Low risk: no behaviour changes. A branch so CI runs before the merge. |

## Goal

Phase 3 opens with its version and a formatted tree. The reformat lands now because a phase
boundary is the one moment no story branch is open to conflict with it (DEC-11 deferred it;
DEC-28 decides it). After this story, `ruff format --check` is a CI gate, so the tree stays
formatted, and `git blame` skips the reformat. Design: ARCHITECTURE.md §3.1, DEC-28.

## Scope

Three commits, in this order. The maintainer makes each one.

1. **Version.** `1.0.0.dev0` in `pyproject.toml` and `src/rag_qa/__init__.py`, then
   `uv lock`. Only the project's own version line moves in `uv.lock`.
2. **The reformat**, alone:
   - `[tool.ruff.format] exclude = ["*.md"]` in `pyproject.toml`. ruff 0.16 also formats the
     Python blocks inside Markdown, which would rewrite examples in `docs/ADR.md`,
     `docs/ARCHITECTURE.md` and two closed story files (S0-5, S2-5);
   - `uv run ruff format`, which changes 30 Python files (11 in `src/rag_qa`, 19 in
     `tests`; about 860 changed lines, measured 2026-10-14).

   Nothing else goes in this commit, so a reviewer can trust it as mechanical.
3. **The gate:**
   - a `Format` step in CI's `check` job, after `Lint`: `uv run ruff format --check`;
   - `.git-blame-ignore-revs`, naming commit 2's full SHA, with a comment saying what it
     is. GitHub's blame view reads the file; locally,
     `git config blame.ignoreRevsFile .git-blame-ignore-revs`;
   - README's Development section lists `uv run ruff format --check` beside the other gates,
     and mentions the blame setting.

## Out of scope

- Any lint rule change, and any other cleanup the formatter happens to show.
- Formatting Markdown (DEC-28).

## Verification

```bash
# 1. the version, in both places and in the lock
grep -n '^version' pyproject.toml && grep -n '__version__' src/rag_qa/__init__.py
git diff main -- uv.lock | grep '^[-+]version' # one line each way: rag-qa-system's own

# 2. the reformat touched only Python files, and the tree is now formatted
git show --stat 01854a1 | tail -3                 # 30 .py files plus pyproject.toml
uv run ruff format --check                        # "N files already formatted", none to reformat
uv run ruff format --check docs stories; echo $?  # Markdown excluded: nothing to check

# 3. nothing behaves differently
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q
uv run rag-eval check; echo "exit $?"             # 0: the rendered-prompt probe did not move

# 4. CI is green on the PR, Format step included (gh auth status shows `workflow` first)
gh pr checks
```

## Review notes for the human

- Commit 2 should be whitespace, quotes, trailing commas and line breaks only. Skim the diff
  for anything that is not.
- The `.git-blame-ignore-revs` SHA must be commit 2's. A merge commit keeps it; a squash or a
  rebase would not.

## Discovered

- **The exclude holds only when ruff walks a folder.** `force-exclude` is off by default, so
  a Markdown file named on the command line is still formatted: `ruff format --check
  docs/ADR.md` reports "1 file would be reformatted", and with `--force-exclude` it checks
  nothing. CI's bare `ruff format --check` walks the tree, so the gate is sound. A tool that
  formats files by name (an editor's format-on-save, a pre-commit hook) would not be
  (backlog line).
- **Half the Python was already formatted.** `ruff format` left 29 of the 59 Python files
  as they were (19 in `src`, 9 in `tests`, and `scripts/fetch_dataset.py`), so DEC-11's
  "the tree is unformatted" was true of 30 files.
- **The suite's one warning is not new:** langchain-community's sunset notice, from
  `vectorstore.py`'s FAISS import (DEC-5). S3-3 removes the import.

## Verification record (2026-10-16)

Each commit's state was checked green before the commit was made: the gates ran on the
version bump alone (890 passed, 5 xfailed, coverage 98.37%), and again on the final tree.

1. **The version.** `pyproject.toml:3` and `src/rag_qa/__init__.py:27` read `1.0.0.dev0`.
   `git diff main -- uv.lock | grep '^[-+]version'` gave one line each way,
   `rag-qa-system`'s own: `-version = "0.3.0"`, `+version = "1.0.0.dev0"`. `uv lock`
   printed "Updated rag-qa-system v0.3.0 -> v1.0.0.dev0" and moved nothing else.
2. **The reformat.** `git show --stat HEAD~1 | tail -3` (run on `a2bbe98`, before the close-out commit; use `01854a1`) ended with
   `31 files changed, 594 insertions(+), 279 deletions(-)`: the 30 Python files (11 in
   `src`, 19 in `tests`, 868 lines, as design review 1 measured), plus `pyproject.toml`'s
   five-line exclusion. `uv run ruff format --check` printed "59 files already formatted".
   `uv run ruff format --check docs stories; echo $?` printed nothing, then `0`.

   **Checked two more ways, beyond the skim the review notes ask for:**
   - each of the 30 files parses to the same syntax tree before and after (`ast.dump`,
     without positions), so string values and docstrings are unchanged too;
   - each is byte-identical to ruff 0.16.8 run over `origin/main`'s version of it.
3. **Nothing behaves differently.** `ruff check`: "All checks passed!". `mypy`: no issues in
   30 source files. `pytest --cov=rag_qa -q`: **890 passed, 5 xfailed, 1 warning**,
   coverage 98.37%, as on the version bump alone. `uv run rag-eval check` exited **0**:
   all six fingerprint parts ok, and every floor met. Its `query` part renders the prompt
   through `chain.py`, one of the reformatted files.
4. **CI.** `gh auth status` showed the `workflow` scope. `gh pr checks` on PR #19:
   `check pass 2m20s` and `eval-retrieval pass 2m14s` (run 38927352954, on `a2bbe98`).
   The `check` job's steps ran Lint, then **Format**, then Types, Test, Audit and Tier-2
   check, each a success.

**The blame file.** `.git-blame-ignore-revs` names
`01854a1164dd39b6dbd665a0047da5b85ce23bb8`, which was `HEAD~1` on `a2bbe98`. On
`tests/test_vectorstore.py:68`, a line the reformat split, a plain
`git blame` names `01854a1`; with `--ignore-revs-file .git-blame-ignore-revs` it names
`3be65a5` (2026-10-05), the line's real author.

## Deviation from plan

Ran on Opus 5.5 without /fast, where the story says opus-fast: fast mode needs usage
credits.

Otherwise none. The two `pyproject.toml` changes went into commits 1 and 2 by
`git add -p`, and the recipe appended commit 2's SHA to `.git-blame-ignore-revs` between
commits 2 and 3, since it could not exist earlier.

## Review (2026-10-16)

**Second review (Opus 5.5): no blockers; 2 doc findings, fixed on the PR.** It re-ran the
claims above against a clean snapshot of `main` and the installed ruff.

- **Checked and sound:**
  - **The reformat is mechanical, reproduced.** `main`'s 59 Python files, formatted by the
    project's ruff 0.16.8 in a scratch copy: all 30 reformatted files are byte-identical
    to `01854a1`'s, and each parses to the same syntax tree (`ast.dump`). The one other
    difference, `__init__.py`, is the version string. 11 + 19 changed, 19 + 9 + 1 untouched.
  - **The gates, re-run on the working tree:** `ruff check` clean; `ruff format --check`
    "59 files already formatted"; mypy clean (30 files); **890 passed, 5 xfailed**,
    coverage 98.37%; `rag-eval check` exit 0, all six parts ok and every floor met.
  - **The exclude:** with `format.exclude` emptied, `ruff format --check` flags 4 Markdown
    files; with it, `docs` and `stories` check nothing. A named `docs/ADR.md` is still
    flagged, and `--force-exclude` checks nothing, as Discovered says.
  - **The backlog fix works:** `force-exclude = true` under `[tool.ruff]`, tried in a
    scratch copy of `pyproject.toml`, makes `ruff format --check docs/ADR.md` exit 0.
  - **The lock:** `git diff main -- uv.lock` is the one version line. CI runs
    `uv sync --locked`, so it formats with the lock's ruff 0.16.8.
  - **Blame:** `.git-blame-ignore-revs` names a real commit; `tests/test_vectorstore.py:68`
    blames `01854a1` plainly and `3be65a5` with the ignore file.
  - **Commits and PR:** each commit's `Refs:` trailer parses, none carries an attribution
    line, and the PR body has none. PR #19 is mergeable and clean, and merge commits are
    allowed on the repository.
- **Findings, fixed:**
  - Verification 2 read `git show --stat HEAD~1`, true only while `a2bbe98` is the tip; the
    close-out commit moves it. It now names `01854a1`.
  - The Status line said CI was green "on the PR's head", which the close-out commit
    changes; it now says "on `a2bbe98`" and counts the close-out commit.
- **Notes, no change:** `.claude/` is untracked and not ignored, so keep it out of the
  commit. `ARCHITECTURE.md` lines 325 and 1848 call the code unformatted; they sit in the
  Phase 1 and Phase 2 sections and read as history.

**Second review, round 2 (Opus 5.5, 2026-10-16, on the PR's head `177ffbf`): no blockers;
1 doc finding and 1 nit, fixed on the PR.** It took in the close-out commit, and treated
round 1's record as claims to check.

- **Checked and sound:**
  - **The comments, which a syntax tree leaves out.** In all 30 reformatted files the
    comments come in the same order, and each of the 22 `noqa`, `type: ignore` and
    `pragma` comments sits on the same code as before, so none now suppresses another line.
  - **The reformat, again** from a clean `main` with ruff 0.16.8: byte-identical to
    `01854a1` but for commit 1's two version lines, with the same syntax trees. Commit 2
    touches only `.py` files and `pyproject.toml`.
  - **The gates on `177ffbf`:** `ruff check` clean, "59 files already formatted", mypy
    clean, **890 passed, 5 xfailed**, coverage 98.37%, and `rag-eval check` exit 0.
  - **The version's reach:** `__version__` feeds only FastAPI's OpenAPI `version`. No
    index manifest or tier-2 run file records it, so nothing goes stale.
  - **The rest:** the blame on `tests/test_vectorstore.py:68`; the four commits' headers,
    bodies, trailers, and no attribution; CI green on `177ffbf` (run 38935030666); and
    CLAUDE.md's "S3-12 releases 1.0.0", against the S3-12 story.
- **Finding, fixed:** CLAUDE.md said to run `uv run ruff format` "over what you change",
  which reads as naming the changed files, the one case the exclude misses: with it in
  place, `ruff format --check docs/ADR.md docs/ARCHITECTURE.md` would rewrite both. It now
  says to run it bare, and why. The backlog line points at that rule.
- **Nit, fixed:** the Status line and the board named CI on `a2bbe98` only. They now name
  `177ffbf`'s run too, and say to check the head before the merge.
- **Note, no change:** the repository allows squash and rebase merges, and `main` has no
  protection or ruleset, so only the docs ask for a merge commit. PR #18 was merged with
  one; check the merge button's choice all the same.
