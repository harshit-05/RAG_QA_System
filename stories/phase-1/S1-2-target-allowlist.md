# S1-2: Allowlist `_target_` imports and stand up CI

| | |
| --- | --- |
| **Status** | In review (2026-09-26) — local verification passed; CI (step 5) runs after the push |
| **Closes** | ISS-04 (OWASP LLM07/08) |
| **Depends on** | S1-1 (ARCHITECTURE.md §1.1 DEC-7, DEC-11) |
| **Model** | opus-fast |
| **Plan-first** | no |

## Goal

The config stops being arbitrary code execution. `import_from_string` refuses any
dotted path outside an explicit module-prefix allowlist, and every `_target_` in
the system — including `ingest.py`'s direct call, which bypasses `build_object`
today — goes through that check. The same story stands up `.github/workflows/ci.yml`
so that from here on every story lands on green CI rather than all of Phase 1
arriving at once at the end.

## Scope

- **`registry.py`**: `ALLOWED_PREFIXES` module constant —
  `langchain_core.`, `langchain_community.`, `langchain_huggingface.`,
  `langchain_ollama.`, `langchain_text_splitters.`, `langchain_classic.`,
  `rag_qa.`. Enforced **inside `import_from_string`**, not `build_object`:
  that is the choke point both call sites share, and it is the gap ADR-015
  flagged. A blocked path raises with the offending target and the allowed
  prefixes listed.
- The constant is **not** config-overridable and takes no env escape hatch — an
  allowlist the config can edit is not an allowlist. Document that in the
  module docstring alongside the existing ADR-015 note, which this story
  resolves.
- **`config.py` / `schema.py`**: string-check every component's prefix at load
  (no import), so a bad `_target_` fails before anything is built. Imports still
  happen only at build time.
- `langchain_classic.` is allowed as a **config** prefix so the disabled
  reranker entry validates. ADR-007 constrains source imports under
  `src/rag_qa/`; the existing grep check in S0-5 still holds and stays.
- **`config.py` / `schema.py`**: the prefix string-check recurses into nested
  `_target_`s (ISS-03's typo was nested inside the reranker). Also add
  `check_imports(config, refs)` — imports the named components' targets and
  raises `ConfigError` on failure. It is **not** called by `load_config` (DEC-7:
  load never imports); S1-5's regression suite and real-config test use it.
- **`.github/workflows/ci.yml`**: push + pull_request; checkout →
  `astral-sh/setup-uv` (pinned, cache on) → `uv sync --locked` → `ruff check` →
  `pytest`. Blocking from this commit. `--locked` doubles as a lockfile-drift
  check.
- **Make `ruff check` green before the gate exists.** ruff 0.16.8's defaults
  already flag 4 errors in the tree (verified 2026-09-25): `BLE001` at
  `src/rag_qa/ingest.py:66`, and `I001`, `PLR1722`, `BLE001` in
  `scripts/fetch_dataset.py`. S1-4 owns the real fixes. Here: auto-fix the
  import order, and add targeted `# noqa: BLE001  # S1-4 (ISS-05/ISS-18)`
  suppressions for the rest — each one names the story that removes it. No
  blanket ignores.
- Tests: `tests/test_registry.py` — `import_from_string` on a valid path, a bad
  module, a bad attribute, and a **blocked** prefix (`os.system`,
  `subprocess.Popen`); `build_object` over nested `_target_`s, lists and plain
  dicts.

## Out of scope

- mypy and the coverage gate (S1-6, S1-5): CI runs ruff + pytest only for now.
- pip-audit (S1-8).
- Widening the allowlist for anything not already in `config.yaml`. If a new
  prefix seems needed, that is a Discovered note, not a quiet addition.

## Verification

```bash
# 1. the allowlist bites, including through the ingest call site
uv run python -c "
from rag_qa.registry import import_from_string
for bad in ['os.system', 'subprocess.Popen', 'builtins.eval']:
    try: import_from_string(bad); print('LEAK:', bad)
    except ImportError as e: print('blocked:', bad, '|', e)
print(import_from_string('rag_qa.registry.build_object').__name__)  # allowed
"

# 2. a config naming a blocked target fails at LOAD, before any build
uv run pytest tests/test_registry.py -q

# 3. nothing in the real config is blocked; the system still runs
uv run rag-ingest && echo "What is this corpus about?" | uv run rag-query

# 4. ADR-007 source-import rule still holds. Import lines only: the allowlist
#    constant in registry.py names "langchain_classic." as a string, by design.
grep -rnE "^\s*(from|import) langchain_classic|^\s*from langchain import|^\s*import langchain$" src/rag_qa/ | wc -l   # → 0

# 6. the S1-4 suppressions are the only ones, and each names its owner
grep -rn "noqa" src scripts

# 5. CI is green on the pushed branch
gh run list --limit 1 && gh run view --log-failed 2>/dev/null | head -20
```

### Results (2026-09-26)

| Check | Result |
| --- | --- |
| 1. allowlist bites | pass: `os.system`, `subprocess.Popen` and `builtins.eval` refused by prefix; `rag_qa.registry.import_module` refused by the defining-module check (see Deviation); `rag_qa.registry.build_object` allowed |
| 2. `pytest tests/test_registry.py` | pass: **23 passed in 3.66 s**. Full suite: **65 passed**, run with `HF_HUB_OFFLINE=1` as CI does |
| 3a. `rag-ingest`, real config | pass: 561 pages → 1,708 chunks, 0 failed. The loaders go through the direct `import_from_string` call site, so this is the ADR-015 path, now covered |
| 3b. `rag-query`, real config, **mistral** | pass: answer word-for-word identical to S1-1's mistral run (`temperature: 0`), same 5 sources, 8 m 48 s including ingest |
| every target in `config.yaml` | all 14 pass both checks, including the unreferenced reranker's three; each class is defined inside its own allowlisted package |
| 4. ADR-007 import grep | 0 |
| 6. `grep -rn noqa src scripts` | the three S1-4-tagged suppressions, plus S1-1's `TRY004` in `schema.py` (reason in the comment above it) |
| `uv sync --locked` / `ruff check` (whole project, as CI runs them) | pass / All checks passed |
| 5. CI on GitHub | **pending the push**; the `workflow` token scope is needed first (Discovered) |

The two refusal messages a maintainer can hit:

```text
_target_ 'os.system' is outside the import allowlist (ISS-04). Allowed module prefixes: 'langchain_core.', 'langchain_community.', 'langchain_huggingface.', 'langchain_ollama.', 'langchain_text_splitters.', 'langchain_classic.', 'rag_qa.'. The list is fixed in rag_qa/registry.py and deliberately not configurable (DEC-7).

_target_ 'rag_qa.registry.import_module' resolves to <function import_module ...>, defined in 'importlib', which is outside the import allowlist (ISS-04). A name an allowed module re-exports does not count; name the object where it is defined. Allowed module prefixes: ...
```

## Review notes for the human

Check the enforcement point: it must be `import_from_string`, so that
`ingest.py`'s direct call is covered — if it lands in `build_object` the story
has missed its own reason for existing. Then read the blocked-path error
message: it should name the target and the allowed prefixes, because the person
hitting it is usually a maintainer adding a legitimate component, not an
attacker. In the workflow file, confirm no step needs Ollama, a model download
or network beyond the package index.

## Discovered

- **A prefix-only allowlist was bypassable from inside our own package.**
  `rag_qa.registry` does `from importlib import import_module`, so
  `rag_qa.registry.import_module` passes a prefix check. `build_object` with
  `{name: subprocess}` then imported any module on the path, running its
  top-level code — arbitrary code execution from inside the allowlist (verified
  2026-09-26, before the fix). A scan of the allowlisted LangChain packages
  found them re-exporting `import_module` too. Closed by the defining-module
  check (Deviation).
- **Pushing this story needs the GitHub `workflow` token scope.** GitHub rejects
  a push that adds or changes `.github/workflows/*.yml` unless the token has
  it. The `gh` token has `gist`, `read:org`, `repo`, and git pushes through a
  `cache` credential helper whose token scopes can't be inspected. The fix is
  `gh auth refresh -h github.com -s workflow`, then `gh auth setup-git` →
  STATUS prerequisites.
- **S1-1's hash changed on amend**: `801ddea` → `b22a37a`. The story file and
  board are corrected in this commit, since a commit cannot record its own
  hash.
- `setup-uv` v10's inputs were checked at the pinned SHA, not assumed:
  `version`, `enable-cache` (default now `auto`) and `cache-python` all exist.

## Deviation from plan

- **Addition: a second allowlist check, on where the object is defined.**
  After resolving a `_target_`, `import_from_string` also requires the object's
  `__module__` to be under an allowed prefix. It refuses re-exported names and
  module objects (which have no `__module__`). The story specified a prefix
  check only, and that alone does not meet its own goal ("the config stops
  being arbitrary code execution"); see Discovered. It cost the real config
  nothing: all 14 targets pass.
- **Addition: `RagConfig.references()` and `RagConfig.targets(ref)`**, the
  public helpers `check_imports` needed (retrievers yield no targets).
  `references()` is also what S1-5's real-config import test will iterate.
- **Addition: `HF_HUB_OFFLINE=1` in CI**, a tripwire that turns an accidental
  model download in a test into a loud failure instead of a silent ~90 MB fetch
  (DEC-11's "no model download").
- **Test fixture change in S1-1's `test_schema.py`**: the fake `pkg.*` targets
  moved to `rag_qa.stub.*`. The new load-time check correctly rejected them;
  they are never imported, since loading is a string check.
- The step-3 run used mistral on the real config: 7.4 GB was available once RAM
  was freed. The first ingest ran through a phi3 scratch config at 5.5 GB,
  per pre-flight caveat 8.
