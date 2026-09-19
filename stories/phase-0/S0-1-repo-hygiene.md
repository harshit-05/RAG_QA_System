# S0-1: Repo hygiene — .gitignore and purge tracked artifacts

| | |
| --- | --- |
| **Status** | Done (2026-09-18) — commit pending, maintainer commits manually |
| **Closes** | ISS-09 |
| **Depends on** | — (can run before the Phase 0 architecture pass) |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

The git tree stops carrying build artifacts, binaries, and media. After this
story, `git ls-files` shows only source, config, and project docs — and
nothing ignorable can be committed again.

## Scope

- Add root `.gitignore`: `__pycache__/`, `*.pyc`, `.venv/`, `vectorstore/`,
  `temp/`, `*.save`, `*.webm`, `docs/*.parquet`, `.python-version` NOT
  ignored (it's config), `eval artifacts`, OS/editor litter.

- `git rm -r --cached` (files stay on disk where useful):
  `__pycache__/`, `v1/__pycache__/`, `vectorstore/db_faiss/`,
  `Screencast from 30-07-25 04_25_00 PM IST.webm`, `v1/docx_processor.py.save`.

- Delete `temp/` entirely (untracked stale copy of v2; verified diverged —
  superseded by git history).

- Delete the `.webm` and `.py.save` from disk too (they serve no purpose).
- `docs/` is the ingested corpus, so only ingestible files belong there
  (arch pass 2026-09-18): `git rm --cached docs/0000.parquet docs/train.parquet`
  (no loader can read them; ignored thereafter, files stay on disk) and
  `git mv docs/dataset.py scripts/fetch_dataset.py` with **no content
  change** (its ISS-18 fix is a Phase 1 story). The three PDFs stay tracked:
  they are the test corpus and are already in history.

## Out of scope

- Collapsing `v1`/`v2` (S0-3).
- Any dependency or code changes.

## Verification

```bash
git status --short                      # no __pycache__/vectorstore/webm entries staged as tracked
git ls-files | grep -E '__pycache__|\.pyc|vectorstore/|\.webm|\.save' | wc -l   # → 0
ls temp/ 2>&1                           # → No such file or directory
git ls-files docs/                      # → exactly the three .pdf files
ls scripts/fetch_dataset.py             # exists; git shows it as a rename of docs/dataset.py
cat .gitignore                          # shows the rules above
```

## Review notes for the human

The `git rm --cached` list — confirm nothing source-like is being untracked.
The commit removes ~4.3 MB of binaries from the tree tip (history rewrite is
deliberately NOT done — low value for a repo this young).

## Discovered

- `temp/v2/` was byte-identical to `HEAD:v2/` except one line (a
  `_build_object` → `build_object` rename in `pipeline_builder.py`), so the
  stale copy was already in history before deletion.
- Git does not support trailing inline comments in `.gitignore`; a first
  draft with `pattern   # comment` lines silently ignored nothing. Comments
  now sit on their own lines. Worth remembering for any future ignore/attr
  files.

## Deviation from plan

None in scope. Executed before the `pre-v0.1 baseline` commit (maintainer
chose to commit manually); the untrack operations are staged, so the
baseline and S0-1 commits must be separated at commit time — recipe in the
session hand-off.
