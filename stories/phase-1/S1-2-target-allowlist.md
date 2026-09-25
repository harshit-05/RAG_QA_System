# S1-2: Allowlist `_target_` imports and stand up CI

| | |
| --- | --- |
| **Status** | Todo |
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

## Review notes for the human

Check the enforcement point: it must be `import_from_string`, so that
`ingest.py`'s direct call is covered — if it lands in `build_object` the story
has missed its own reason for existing. Then read the blocked-path error
message: it should name the target and the allowed prefixes, because the person
hitting it is usually a maintainer adding a legitimate component, not an
attacker. In the workflow file, confirm no step needs Ollama, a model download
or network beyond the package index.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
