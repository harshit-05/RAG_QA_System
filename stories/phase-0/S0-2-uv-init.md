# S0-2: uv project — pyproject.toml, pinned Python 3.12, lockfile

| | |
| --- | --- |
| **Status** | Done (2026-09-18) — commit pending, maintainer commits manually |
| **Closes** | ISS-08 |
| **Depends on** | S0-1 (DEC-1 resolved 2026-09-18 — ARCHITECTURE.md §0.1) |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

The project declares and locks every dependency it actually imports, on an
interpreter with full ML wheel coverage. `uv sync` on a fresh clone yields a
working environment — the first time this has been true.

## Scope

- `uv init` (or hand-write `pyproject.toml`) with package metadata and
  `version = "0.1.0.dev0"` (release scheme: Phase 0 exit ships 0.1.0 /
  tag `v0.1` — see STATUS.md release mapping); pin interpreter via
  `uv python pin 3.12` (3.12.13 already uv-managed locally).

- Runtime deps, exactly (DEC-1; verified to lock together 2026-09-18):
  `langchain-core>=1,<2`, `langchain-community>=0.4,<0.5`,
  `langchain-text-splitters>=1,<2`, `langchain-ollama>=1,<2`,
  `langchain-huggingface>=1,<2`, `faiss-cpu>=1.15`, `sentence-transformers>=5`,
  `torch>=2.6`, `pypdf>=6`, `docx2txt>=0.9`, `pyyaml>=6`, `pydantic>=2.11`,
  `pydantic-settings>=2.10`. **Not** the `langchain` meta-package and
  **not** `langchain-classic` (it arrives transitively; we never import it).

- **CPU-only torch, explicit index** (ARCHITECTURE.md §0.1 rule 2): the
  PyTorch CPU index hosts stale `langchain-community` releases, and as a
  general index it silently pins the stack to LangChain 0.3.x. Use exactly:

  ```toml
  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }
  ```

- Eval extra (`[project.optional-dependencies] eval`): `ragas>=0.4`,
  `datasets>=3`, `pandas>=2.2`.

- Dev group (`[dependency-groups] dev`): `pytest>=8`, `ruff>=0.13`, `mypy>=1.17`.
- Build system: hatchling with `packages = ["src/rag_qa"]`; create
  `src/rag_qa/__init__.py` containing `__version__ = "0.1.0.dev0"` in this
  story so the package builds and `uv sync` succeeds before S0-3 moves code in.
- Commit `pyproject.toml`, `uv.lock`, `.python-version`, `src/rag_qa/__init__.py`.
- Delete `v1/requirements.txt` only if S0-3 hasn't already retired `v1/`
  (ordering tolerance); otherwise leave for S0-3.

## Out of scope

- Importing/using any of these in code (S0-5).
- CI wiring (Phase 1).

## Verification

```bash
uv sync                                             # resolves + installs clean
uv run python -c "import langchain_core, faiss, yaml, pydantic; print('imports ok')"   # NOT `import langchain`: the meta-package is excluded by DEC-1 rule 1
uv run python --version                             # → 3.12.x
uv run python -c "import langchain_core; print(langchain_core.__version__)"   # → 1.x (the index trap check)
grep -A1 '^name = "torch"' uv.lock | head -2        # version ends in +cpu or source is the pytorch-cpu index
grep -c 'name = "nvidia' uv.lock                    # → 0
git ls-files | grep -E 'pyproject.toml|uv.lock|.python-version|src/rag_qa/__init__.py' | wc -l  # → 4
```

## Review notes for the human

Two lines in `uv.lock` decide this story: `langchain-core` must be **1.x**
(if it is 0.3.x the PyTorch index was not `explicit` — see STATUS.md
pre-flight caveat 6), and there must be **no `nvidia-*` wheels** (if there
are, torch came from PyPI and ~3–4 GB of CUDA packages are being pulled onto
a GPU-less host).

## Resolved versions (2026-09-18)

126 packages, resolved in 3.9 s, installed in 54 s, `.venv` 1.3 GB.

| | |
| --- | --- |
| langchain-core / -community / -text-splitters | 1.6.3 / 0.4.2 / 1.1.2 |
| langchain-ollama / -huggingface | 1.1.0 / 1.2.2 |
| langchain-classic | 1.0.8 (transitive via community; never imported) |
| torch | 2.14.0+cpu, `cuda.is_available() == False` |
| faiss-cpu / sentence-transformers / transformers | 1.15.1 / 6.1.0 / 5.17.0 |
| pydantic / pydantic-settings / numpy | 2.13.5 / 2.15.0 / 2.5.3 |
| pytest / ruff / mypy | 9.1.1 / 0.16.8 / 2.3.1 |
| ragas / datasets / pandas (eval extra, not installed by default) | 0.4.3 / 5.0.1 / 3.0.6 |

The `langchain` meta-package is **absent** from the environment, as DEC-1
rule 1 requires. Zero `nvidia-*` packages and no `cu12`/`cu13` strings in
the lock.

## Discovered

- The story's original import check was `import langchain`, which contradicts
  DEC-1 rule 1 (the meta-package is deliberately not a dependency and is not
  installed). Corrected in place to `import langchain_core`.
- `[project.scripts]` deliberately not added here — S0-3 wires the entry
  points when the modules they point at exist.
- The eval extra is declared but not installed by default; `uv sync --extra
  eval` adds ragas, tiktoken and scikit-network on top. Phase 2 installs it.
- `grep -c` exits non-zero on zero matches, which breaks `&&` command chains
  in verification blocks. Future stories should use `(grep -c ... || true)`.

## Deviation from plan

None in scope. Executed before the baseline commit, as with S0-1; see
STATUS.md pre-flight caveat 1 for the three-commit recipe.
