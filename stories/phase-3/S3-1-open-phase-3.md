# S3-1: Open Phase 3: version 1.0.0.dev0, format the code once and gate it

| | |
| --- | --- |
| **Status** | Todo |
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
git show --stat HEAD~1 | tail -3                  # 30 .py files plus pyproject.toml
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

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
