# S3-3: Leave langchain-community and FAISS

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | DEC-5 step 3; ISS-16; SRS §9 (deserialization only of what the system produced); backlog: the parity tests and the corpus PDF they read |
| **Depends on** | S3-2 |
| **Model** | opus-fast |
| **Plan-first** | no |
| **Branch** | `chore/s3-3-leave-langchain-community`. Low risk once S3-2 has merged: nothing in `src` calls either package any more. |

## Goal

The last step of DEC-5. `langchain-community` was sunset on 2026-05-22, and FAISS's pickle
was the one place the system deserialized code-bearing data (ISS-16). After S3-2 nothing in
the application uses either, so both leave the direct dependencies. ragas still brings
`langchain-community` and `langchain-classic` into the `eval` extra (DEC-5's note), so they
stay in `uv.lock` but not in a plain `uv sync`. Design: ARCHITECTURE.md §3.1, DEC-19 and
"Earlier decisions".

## Scope

- **Dependencies:** `uv remove langchain-community faiss-cpu`. Record the lock's diff:
  `faiss-cpu` leaves it, while `langchain-community` and `langchain-classic` stay for ragas.
- **Tests that exist only for the old packages:**
  - `tests/test_loaders.py`: the parity tests against the `langchain_community` loaders
    (found in S1-3, "accepted until `langchain-community` leaves"). The contract tests stay.
  - `tests/test_registry.py`: the cases that name `rag_qa.vectorstore.FAISS` or
    `langchain_community.*` targets. Keep the meaning of each: a re-export from an allowed
    module still fails check 2, with a `rag_qa` re-export of some other package standing in.
  - `tests/test_deprecations.py`: its negative control imports
    `langchain_community.llms.Ollama`. Replace it with another known-deprecated import that
    the main environment still installs, or with a module the test writes itself, so the
    gate still proves it can fail. Its allowance for the sunset notice goes.
- **The architecture test:** no module under `src` imports `langchain_community`, `faiss` or
  `langchain_classic` (parsed imports, as the existing legacy-import test does). The main
  environment (`uv sync`, no extras) does not have `langchain_community` installed.
- **Docs:**
  - `registry.py`'s docstring and `ALLOWED_PREFIXES` comment (FAISS is no longer imported);
  - README's dependency notes and its pickle limitation, if S3-2 has not removed it already;
  - ARCHITECTURE.md §0.5's invariant note gets a dated line: the invariant is moot, the
    pickle is gone.
- **The `eval` extra still works:** ragas imports with the import shim (S2-5) still in
  place.

## Out of scope

- The ragas import shim (`ragas_scoring.import_ragas`): it goes when a ragas release stops
  importing `langchain_community.chat_models.vertexai`, which 0.4.3, the newest, still does.
- The `eval` extra's advisories: S3-9 bumps `langchain` and writes the audit policy.

## Verification

```bash
uv sync                                                       # exact: the main env only
uv run --no-sync python -c "import langchain_community" ; echo "exit $?"   # 1: not installed
uv run --no-sync python -c "import faiss" ; echo "exit $?"                 # 1
grep -rnE "^\s*(from|import) (langchain_community|faiss)|allow_dangerous_deserialization" src/ ; echo "exit $?"  # 1
# (S3-2's hint that v0.3's vectorstore/db_faiss folder is no longer read may still name it)
uv run ruff check && uv run ruff format --check && uv run mypy
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run rag-eval check; echo "exit $?"                         # 0
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-3/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-3/contexts.json && echo IDENTICAL
# the eval extra still imports ragas (the shim), then back to the main env
uv sync --extra eval && uv run --no-sync python -c "from rag_qa.evaluation.ragas_scoring import import_ragas; import_ragas(); print('ragas ok')"
uv sync
gh pr checks                                                  # check and eval-retrieval green
```

## Review notes for the human

- The deprecation gate must still be able to fail: look at its negative control.
- `uv.lock` should lose `faiss-cpu` (and anything only it needed) and nothing else.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
