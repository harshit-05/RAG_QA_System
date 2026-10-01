# S1-6: Type annotations, ruff and mypy configuration

| | |
| --- | --- |
| **Status** | Done (2026-10-02) — PR #4, rebase-merged into `main` as `ff61088` … `be545a4` (9 commits incl. the review follow-ups); CI green on the branch, the PR and `main` |
| **Closes** | ISS-19, NFR-9, ISS-12 (via `warn_unreachable`) |
| **Depends on** | S1-5 |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

Every public function in `rag_qa` carries type hints and the tree passes
`ruff check` and `mypy` with zero errors, both blocking in CI (NFR-9). Typed
config from S1-1 is what makes this cheap: `config.pipeline.query.llm` has a type
where `config["pipeline"]["query"]["llm"]` had `Any`.

## Scope

- Annotate every function in `src/rag_qa/` — parameters and returns. LangChain
  return types come from `langchain_core` (`Embeddings`, `BaseChatModel`,
  `Runnable`, `VectorStore`, `Document`), not from `langchain_community`.
- **`[tool.ruff]`** in `pyproject.toml`: `line-length = 100` (the existing code's
  actual shape). **Start from ruff's default rule set**, then `extend-select`
  anything wanted that it lacks, and `ignore` with a reason inline. Record any
  per-file ignores the same way.
  *(Amended at S1-5's close-out, from the board's S1-1 finding. This bullet
  first said "an explicit rule set rather than the default, at minimum `E`, `F`,
  `I`, `B`, `UP`". Ruff 0.16's default is 788 rules, so that would have
  **narrowed** the gate CI already enforces. Check `ruff check --show-settings`
  before choosing.)*
- **`[tool.mypy]`**: `disallow_untyped_defs = true` scoped to `src/rag_qa`,
  `ignore_missing_imports = true` for third-party packages without stubs.
  `evaluate.py` may be excluded — its imports live in the optional `eval` extra.
  **Also `warn_unreachable = true`: it is ISS-12's only guard** (code after
  `return`). Verified in S1-5's review: ruff 0.16.8 has no unreachable-code rule,
  even in preview, and mypy *without* this option reports nothing. Afterwards,
  update ISS-12's row in `tests/test_config_regressions.py`'s issue map from
  "not caught yet" to the mypy gate.
- `tests/test_components.py` reads a private field (`splitter._chunk_size`).
  Assert behaviour instead: split a long text and check the chunk lengths.
  (Board item from S1-3.)
- Add both to CI, ordered cheapest-first: `ruff check` → `mypy` → `pytest`.
- Fix what they find; if a fix is behavioural rather than cosmetic, it is a
  Discovered note, not a silent change.

## Out of scope

- `ruff format` / `ruff format --check` (**DEC-11**): the tree is not formatted,
  and a whole-tree reformat is exactly the diff that gets rubber-stamped in a
  review-the-diff workflow. Backlog.
- `tests/` under `disallow_untyped_defs` — lint them, don't force annotations on
  fixtures.
- Strict mode (`--strict`), `Any` elimination, generics on the registry.
  `build_object` is `Any`-in/`Any`-out by nature (ADR-010).

## Verification

```bash
uv run ruff check src tests          # → All checks passed
uv run mypy src/rag_qa               # → Success: no issues found in N source files
uv run pytest -q                     # unchanged, still green
uv run rag-ingest && echo "What is this corpus about?" | uv run rag-query   # behaviour unchanged
gh run list --limit 1                # CI green with both gates blocking
```

### Results (2026-10-02)

**Risk call: branch** (`chore/s1-6-types-and-lint`). It adds a blocking mypy
gate, changes ruff's configuration, and touches signatures in most modules.

**Measured first:** ruff's default set is ~790 rules across 38 families (`F`,
`B`, `UP`, `I`, `SIM`, `RUF`, `PL*`, `PERF`…), but only 2 `E` rules, so `E501`
was off and the line length was ruff's 88. mypy, under the story's settings plus
`warn_unreachable`, reported only **7 errors** (in `chain.py`, `vectorstore.py`
and `evaluate.py`), because S1-1…S1-5 annotated what they wrote.

| Check | Result |
| --- | --- |
| 1. `ruff check src tests` | All checks passed. The config **extends** the default with `E501` at 100, instead of replacing it with a narrower list. 27 lines were wrapped by `ruff format --range` (statement-scoped; the formatter is AST-preserving), and 2 long string literals by hand. The default set then caught a `zip(x, x[1:])` in a new test (`RUF007` → `itertools.pairwise`) |
| 2. `mypy src/rag_qa` | `Success: no issues found in 12 source files`. **`evaluate.py` is included, not excluded**: it cost one `-> None` |
| 2b. `warn_unreachable` bites (ISS-12) | In a scratch copy, code after `build_object`'s last `return` → `registry.py:121: error: Statement is unreachable`, from the project config alone |
| 3. `pytest` | **184 passed**, coverage 97.92% (gate 80%) |
| 3b. the private-field test, rewritten (board item) | now asserts behaviour: every chunk but the last is 900–1000 characters, and each overlap is 100–150. Negative controls (review follow-up): the real config passes; `chunk_size` 200 or 5000, and `chunk_overlap` 10 or 100, each fail. The first version only bounded from above, so a smaller size or overlap passed |
| 4. `rag-ingest` + `rag-query`, real config, mistral | pass, **behaviour unchanged**: ingest 561 pages → 1,708 chunks, 0 failed; the answer is **word-for-word identical** to the S1-1, S1-2 and S1-3 mistral runs, with the same 5 sources (p. 9, 340, 477, 233, 81). The session ends `Exiting...`, exit 0 (S1-4's end-of-input fix, seen in a real run). 646 s on CPU. So the signatures, the `VectorStore` return type and the statement re-wrapping changed nothing a user sees |
| ruff, whole project (as CI) | All checks passed |
| 5. CI | green on the branch: `check` passed in 1m41s with ruff, mypy and pytest all blocking |

## Review notes for the human

Annotation passes are where behaviour changes sneak in disguised as type fixes —
scan the diff for anything that is not purely a signature or an import. Check
the ruff ignore list: each entry should carry a reason, and `BLE001` should now
be *gone* rather than ignored, since S1-4 replaced the bare
`except Exception` it flagged.

## Discovered

- **The seam's return type leaked FAISS.** `create_store` / `open_store`
  returned `FAISS`, so every caller's types named the store implementation. They
  now return `langchain_core`'s `VectorStore`, and mypy then passed. That is
  machine evidence that no code in `src/` uses anything FAISS-specific, which is
  ADR-013's seam checked by a tool rather than by reading the code. The Phase 3
  swap changes no caller's types.
- **`create_store` needs a `list`, not any sequence.** `FAISS.from_documents`
  requires `list[Document]`. The one caller passes `split_documents()`' list, so
  the annotation says `list`: a real constraint mypy surfaced, not an
  annotation gap.
- **The review note's "`BLE001` should now be *gone*" predates S1-4.** S1-4
  deliberately kept two broad catches, both requirements: the per-document read
  (NFR-7) and the REPL boundary (ISS-06). Each `# noqa: BLE001` carries its
  reason in the comment above it. There are no blanket ignores and no
  per-file ignores in `[tool.ruff]`. The other suppressions in `src` are S1-1's
  and S1-2's `TRY004`s, also explained in place.
- **`ignore_missing_imports` was global, which hid first-party typos.** Found in
  review: `from rag_qa.vectorstor import ...` passed mypy. It is now a per-module
  override for the five imports that lack stubs (`yaml`, `docx2txt`, `pandas`,
  `datasets`, `ragas.*`), and the same typo is an error. `pyyaml` has stubs
  available (`types-PyYAML`); adding them would type-check the YAML boundary
  instead of treating it as `Any`. Small, optional → backlog.

## Deviation from plan

- **`evaluate.py` is type-checked** (the story allowed excluding it); it took
  one annotation.
- **`E501` is enabled** at the story's line length of 100. Without it,
  `line-length = 100` is a number nothing enforces.
- **`warn_unused_ignores = true` added** next to the story's settings, so a
  `# type: ignore` that stops being needed gets removed rather than piling up.
- **`files = ["src/rag_qa"]` is set in `[tool.mypy]`,** so CI and a local run
  both use a plain `mypy` with no arguments and check the same files.
- **The three board items folded in at S1-5's close-out** are all done: ruff
  starts from its defaults, `warn_unreachable` is on and ISS-12's row in the
  issue map points at it, and the private-field test is behavioural.
