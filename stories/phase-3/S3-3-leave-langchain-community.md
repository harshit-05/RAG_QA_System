# S3-3: Leave langchain-community and FAISS

| | |
| --- | --- |
| **Status** | Done (2026-10-20), pending the PR: two commits on `chore/s3-3-leave-langchain-community`, the dependency change and this record. Reviewed twice; the first review's four findings and the second review's six are fixed (see "First review" and "Second review"). Verification 1–11 are run and shown below, all as expected, and the gates passed again on the final tree (1020 passed, 98.37%; `rag-eval check` exit 0). 12, `gh pr checks`, runs once the PR is open. |
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

- **The first tier-1 run ended on a Qdrant read timeout** (verification 8), not reproduced.
  The run connected, loaded both models, and retrieved 9 of the 20 questions at about 1 s
  each. Then the client was silent for 20 s, read the alias (logged, 1.6 ms), and sent its
  query: after the client's 10 s timeout, `cannot use Qdrant at http://localhost:6333
  (ReadTimeout: timed out). Is Qdrant running?`, exit 2, with Qdrant up. The host showed
  no CPU, memory or IO pressure; no Ollama model was loaded; Qdrant logged no warning;
  `::1:6333` refuses at once, so the `localhost` lookup is not it. The re-run 3 minutes
  later was clean. S3-6 retries every Qdrant call (DEC-21), which would have absorbed it
  (backlog line).
  - **What the server's records can say** (second review). Neither of Qdrant's records has
    a line for that query, and neither could: v1.19.2's access log (actix-web 4.15.0's
    `Logger`) is written when a response body is dropped, and its telemetry
    (`actix_telemetry.rs`) after the handler returns. A query that arrived and was still
    running, or was cancelled, when the client gave up leaves no trace in either. Telemetry
    since the container's 04:41 UTC start: 84 completed `points/query` calls, all 200, the
    slowest 19 ms, which is this run's 9 plus three full runs of 25 (the re-run and both
    reviews'). So the query never completed on the server; whether it arrived is unknown.
    The 20 s before it held no Qdrant call, so the client side stalled first (an inference,
    not shown).
- **The main environment loses 14 packages, not 2** (131 → 117: distinct names in
  `uv export --frozen`, the dev group and the project included, torch's two platform lines
  counted once; with `--no-dev`, the first review's count, 102 → 88):
  `langchain-community`, `langchain-classic`, `faiss-cpu`, and what only the first two
  needed: aiohttp, aiohappyeyeballs, aiosignal, attrs, frozenlist, greenlet, httpx-sse,
  multidict, propcache, sqlalchemy and yarl. The Audit went red on multidict in S2-5 and on
  aiohttp in S3-2. Both now come only with the `eval` extra, which S3-9's audit policy
  covers. The `eval` extra loses `faiss-cpu` alone (157 → 156, counted the same way; 128 →
  127 with `--no-dev`).
- **The suite's one warning is gone.** S3-1 recorded the sunset notice as the suite's only
  warning; the main environment's run now has none. In the `eval` environment, ragas's own
  import still raises it, in `test_eval_scoring.py`. It is a pytest warning, not a gate
  failure, and it stays until ragas stops importing `langchain_community` (out of scope).
- **`tests/test_registry.py` was nearly done already.** S3-2 had replaced the
  `rag_qa.vectorstore.FAISS` re-export case with `rag_qa.vectorstore.QdrantClient`. The
  `langchain_community.*` targets left are refused by prefix before any import, and they
  back v0.2's migration message (`REMOVED_PREFIXES`), so they stay; a comment says why.
- **README had nothing left to change.** S3-2 had removed the pickle limitation, and no
  dependency note names `langchain-community` or FAISS. Its v0.2 upgrade note, that a
  `langchain_community.` target is refused, is still true.
- **`tests/test_manifest.py` names `langchain_community.embeddings.HuggingFaceBgeEmbeddings`**
  as an unlisted embedder class. It is a string and never imported, so it stays.

## Verification record (2026-10-20)

Run in this order, on the working tree, with Qdrant up and serving the live index under
the `rag_qa` alias (`rag_qa--gen-20261019T153834837765Z-7f20891e`).

1. **`uv sync`:** "Resolved 178 packages … Checked 114 packages": the main environment,
   no extras. `uv remove langchain-community faiss-cpu` made it.
2. **`import langchain_community`:** `ModuleNotFoundError`, exit **1**.
3. **`import faiss`:** `ModuleNotFoundError`, exit **1**.
4. **The grep** printed nothing, exit **1**. The legacy-folder hint in `cli.py` names
   `db_faiss` only in a glob pattern, which the pattern does not match.
5. **Lint, format, types:** "All checks passed!", "60 files already formatted", "no issues
   found in 30 source files".
6. **The suite** (`HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1`): **1021 passed, 12 skipped, 5
   xfailed**, coverage **98.37%**, no warnings (1020 passed after the review dropped one
   test). The 12 skips are the 8 `qdrant_server` tests
   and 4 ragas tests that need the `eval` extra; those 4 ran in step 10.
7. **`rag-eval check`:** exit **0**; all six fingerprint parts ok, every floor met.
8. **`rag-eval retrieval --contexts-out ~/dev/rag-s3-3/contexts.json`:** the first run exited
   2 on a Qdrant read timeout (Discovered). The re-run exited **0**: hit rate 0.950, MRR
   0.796, recall 0.858, S3-2's numbers, and "Wrote the contexts of 25 questions".
9. **`diff`:** **IDENTICAL**. Both files hash to `5a36e2f9d00bbdaf…a074cbc`, S3-2's digest.
10. **The `eval` extra:** `uv sync --extra eval`, then `import_ragas()` printed **ragas ok**.
    In that environment `langchain_community` and `langchain_classic` are installed and
    `faiss` is not. `test_eval_scoring.py`, `test_architecture.py` and
    `test_deprecations.py` there: 159 passed, no skips (158 after the review dropped one
    test).
11. **`uv sync`:** back to the main environment; none of `langchain_community`,
    `langchain_classic`, `faiss` or `ragas` is installed.
12. **`gh pr checks`:** "no pull requests found for branch "main"". It waits for the PR.

**The lock** (`git diff uv.lock`, 23 lines removed, none added): `faiss-cpu` 1.15.1's
package block, and the project's two requirement lines. No other package and no version
moves. `langchain-community` 0.4.2 and `langchain-classic` 1.0.8 stay, for ragas.

**The new guards were checked against breakage:**

- the deprecation gate's negative control, `ChatOllama(model="x").dict()`, raises a
  `LangChainDeprecationWarning` in a fresh interpreter. It is deprecated in langchain-core
  1.4.2 and removed in 2.0, which `pyproject.toml`'s `<2` keeps out;
- with a stand-in `faiss.py` on `PYTHONPATH`, `test_a_plain_sync_installs_none_of_the_removed_packages`
  failed with `assert {'faiss'} == set()`;
- the import scan's negative control flags `import faiss` and
  `from langchain_community.vectorstores import FAISS`, and not a docstring that names the
  package.

## First review (2026-10-20)

Run by the first reviewer on the working tree, over the primary review's head. Reproduced
independently, and every claim held:

- verifications 2–7 and 9–11: the same results; `uv.lock` loses `faiss-cpu` and the
  project's two requirement lines;
- `uv export` of HEAD against the tree, main environment: 102 → 88 packages, the 14 above;
  the `eval` extra: 128 → 127, `faiss-cpu` alone;
- tier 1 re-run into a scratch file: 0.950 / 0.796 / 0.858, and the contexts digest
  `5a36e2f9…074cbc`, identical to S3-2's; no Qdrant timeout in that run;
- the negative control against langchain-core 1.6.3's source: `BaseChatModel.dict` carries
  `@deprecated("1.4.2", removal="2.0.0")`; the install guard fails with a stand-in `faiss.py`
  or `langchain_community.py` on `PYTHONPATH`;
- an AST scan of `src/`, `scripts/` and `eval/` against the declared dependencies: nothing
  relied on the removed packages transitively.

Findings, all fixed (no blocker):

1. **STATUS.md was half closed:** the backlog line for the parity tests and the corpus PDF,
   the "DEC-5 exit tasks" bullet and "Closed in Phase 3" now record S3-3.
2. **CLAUDE.md stated two false things:** `langchain_classic` "arrives only transitively"
   (it is not installed in the main environment), and "S3-3 removes it and `faiss-cpu`" in
   the future tense. Its first line also named FAISS. All three are corrected.
3. **Test nits:** `langchain_classic` is out of the import scan (ADR-007's
   `_legacy_imports` test already covers it) and stays in the install check only;
   `violations()` lost its redundant `LangChainDeprecationWarning` clause
   (a `DeprecationWarning` subclass, MRO checked); `test_the_sunset_notice_is_no_longer_allowed`
   is deleted, since it restated the predicate.
4. **`.claude/` is untracked and not ignored** (local permission lists, absolute paths). Not
   in the diff: the commit recipe names its paths, never `git add -A`.

After the fixes: ruff, format and mypy clean; **1020 passed, 12 skipped, 5 xfailed**,
coverage **98.37%**.

## Second review (2026-10-20, Opus 5.5)

**No blockers; 5 doc and record findings and 1 nit, all fixed in the working tree.** Run on
the working tree after the first review's fixes, with both earlier records treated as
claims to check. Probes ran in the scratchpad; the live `rag_qa` alias and Qdrant's
telemetry were only read.

- **Reproduced:** verification 1–11 (the suite 1020 passed, 98.37%, no warnings; `rag-eval
  check` exit 0; a fresh tier-1 run 0.950 / 0.796 / 0.858 with the contexts digest
  `5a36e2f9…`, IDENTICAL; the `eval` leg: ragas ok, both LangChain packages back, FAISS
  not; the main environment restored).
- **The lock:** `uv remove langchain-community faiss-cpu` on a scratch copy of `HEAD` gives
  a `uv.lock` byte-identical to this one. The 14 names leave the main environment, the
  `eval` extra loses `faiss-cpu` alone, nothing is added, and `uv pip check` is clean. What
  still names a removed package is an extra we never enable (fsspec's `http`,
  transformers' `retrieval`).
- **The negative control, against langchain-core 1.6.3:** `BaseChatModel.dict` carries
  `@deprecated("1.4.2", alternative="asdict", removal="2.0.0")`, and `ChatOllama` does not
  override it. The real predicate flags it, a plain `DeprecationWarning`, LangChain's
  pending kind and the old sunset text (so the allowance is gone), and not a
  `FutureWarning` or a clean body. Every module under `src`, `rag_qa.api` included,
  imports with no warning under it.
- **The install guard:** a stand-in for each of the three packages on `PYTHONPATH` fails
  it; with a stand-in `ragas` it still fails on `faiss` and passes the two LangChain ones.
- **No contract lost with the parity tests:** the PDF contract test pins the whole metadata
  dict, `page_label` and `total_pages` included, and the old oracle wrapped the same
  pypdf. Both models load from `model.safetensors`, no `.bin` cached, so §9's "no pickle
  anywhere" holds for the weights too. A `langchain_community.` target is refused by
  prefix before any import. S3-2's cited runs, 39381661063 and 39381665898, are green on
  both jobs.

Findings, all fixed:

1. **STATUS.md's DEC-5 row** still ended "S3-3 removes `langchain-community` and
   `faiss-cpu`". It now carries a "Done (S3-3)" note, as DEC-19's row carries S3-2's.
2. **ARCHITECTURE.md's DEC-5** said the deprecation checks "allow only the sunset notice".
   A dated line now records the end of the exit, the gate's empty allowance and its new
   control (Deviation).
3. **Verification 10's count** was 159, from before the first review dropped a test; it is
   158.
4. **Two package counts, no flags:** 131 → 117 and 102 → 88 are both right (the dev group
   in or out). Discovered now says how they are counted.
5. **The read timeout's note read more into Qdrant's logs than they hold.** "A query that
   Qdrant never logged" invites "never arrived", but the access log and the telemetry both
   record only completed requests (actix-web 4.15.0's `Logger`, Qdrant's
   `actix_telemetry.rs`, both read at their tags). Discovered and the S3-6 backlog line now
   say what the telemetry shows, and suggest client-side debug logging for next time.
6. **Nit:** `violations()`'s docstring said "the second test" for its second clause.

Out of scope, not recorded as backlog lines: the deprecation gate's import list leaves out
`rag_qa.api` (it imports clean today), and `config.py` imports `pydantic_core`, which
`pyproject.toml` does not declare (pydantic pins it exactly).

After the fixes: ruff, format and mypy clean; `uv lock --check` clean; **1020 passed, 12
skipped, 5 xfailed**, coverage **98.37%**; `rag-eval check` exit 0.

## Deviation from plan

- **CLAUDE.md was edited,** which Scope's Docs list does not name: its LangChain note
  became false the moment the packages left (first review, finding 2).
- **ARCHITECTURE.md's DEC-5 got a dated line,** which the Docs list does not name either:
  its "allow only the sunset notice" became false when the gate's allowance went (second
  review, finding 2).
- **The sunset-notice test was dropped** after the review: the predicate has no allowance
  left to pin.
- Model: opus-fast, as routed.
