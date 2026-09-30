# S1-2: Allowlist `_target_` imports and stand up CI

| | |
| --- | --- |
| **Status** | Done (2026-09-29) — `1f1d6ff`; CI green on first run. Review follow-up 2026-09-30 (targets must be classes) — pending commit |
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
print(import_from_string('rag_qa.config.ConfigError').__name__)  # allowed (a class; functions are refused since the follow-up)
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
| 5. CI on GitHub | pass (2026-09-29): run `36610223687` on `1f1d6ff`, 47 s, every step green. Log checked, not just the badge: 97 packages installed, `torch==2.14.0+cpu`, **0** `nvidia-*` wheels, `ruff` All checks passed, **65 passed** in 14.9 s with `HF_HUB_OFFLINE=1` |

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
  check (Deviation) — **only partly**: the scan looked for re-exports of
  `import_module`, not for functions that wrap it. See the second review below.
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
- **CI runs on the runner's system Python 3.12.3, not uv-managed 3.12.13**
  (found in the run log, 2026-09-29). `.python-version` pins only `3.12`, and
  uv prefers a matching system interpreter, so CI tests a patch release ten
  versions behind this host. `cache-python: true` therefore does nothing, and
  its comment in `ci.yml` ("the uv-managed 3.12") is wrong. Harmless today.
  Fix: set `UV_PYTHON_PREFERENCE: only-managed` in the job env and correct the
  comment, in the next story that edits `ci.yml` (S1-5) → STATUS backlog.

### Second review (2026-09-30)

- **Import helpers *defined* in an allowed package passed both checks.**
  `langchain_core.utils.utils.guard_import` (public name
  `langchain_core.utils.guard_import`) wraps `importlib.import_module`, and
  `langchain_core._import_utils.import_attr` returns any attribute of any module.
  Both are defined in `langchain_core`, so the defining-module check let them
  through, and the load-time prefix check did too. Run against `1f1d6ff`:
  `build_object({_target_: …guard_import, module_name: colorsys})` imported
  `colorsys`; `import_attr(attr_name=Popen, module_name="", package=subprocess)`
  returned the live `subprocess.Popen`. That is the same import-anything
  capability the first Discovered item treated as arbitrary code execution.
  **Closed by the follow-up** (check 3, below).
- **Instances were not refused, despite the code comment.** `registry.py` said
  instances have no `__module__`; they inherit it from their class. A
  module-level callable instance of an allowlisted class would have passed check
  2 and been called by `build_object`. No abusable instance was found. **Closed
  by the same follow-up**, and the comment is corrected.
- **Still open: allowed classes with arbitrary kwargs are code execution.**
  `HuggingFaceEmbeddings` passes `model_kwargs` unchanged into
  `SentenceTransformer(model_name, **model_kwargs)`
  (`langchain_huggingface/embeddings/huggingface.py:98`), which accepts
  `trust_remote_code`. An embedder entry shaped like the four in `config.yaml`,
  pointed at someone else's HF repo with `trust_remote_code: true`, downloads and
  runs that repo's Python. Confirmed from source only (running it would execute
  remote code). The SRS LLM07/08 requirement (code-loading from config is
  allowlisted) is met as written; this story's Goal line ("the config stops being
  arbitrary code execution") and ISS-04's "any kwargs" half are not. The config
  stays trusted input, like code → STATUS backlog.

### Follow-up: `_target_`s must be classes (2026-09-30)

`import_from_string` gains **check 3**: the resolved object must be a class
(`isinstance(obj, type)`). `build_object` instantiates what it gets, and every
legitimate target is a class, so this refuses every module-level function and
every instance with one rule, instead of blocking gadgets one at a time. It raises
`ImportError` like the other two checks (targeted `noqa: TRY004` with the reason
beside it), because `check_imports` and the callers catch only that. Tests were
written first and failed against `1f1d6ff` (6 failed, 23 passed), including
`build_object` actually running `guard_import`.

Verification step 1's "allowed" example changed from the function
`rag_qa.registry.build_object` to the class `rag_qa.config.ConfigError`, and so
did `test_imports_an_allowed_path`, since functions are now refused.

| Check | Result |
| --- | --- |
| `ruff check` | All checks passed |
| full `pytest`, `HF_HUB_OFFLINE=1` | **71 passed** in 3.80 s (6 new cases) |
| 1. allowlist bites | `os.system`, `subprocess.Popen`, `builtins.eval`, `guard_import`, `import_attr` all blocked; `rag_qa.config.ConfigError` allowed |
| every target in `config.yaml` via `check_imports` | 14 targets import cleanly, all classes |
| 4. ADR-007 import grep | 0 |
| 6. `grep -rn noqa src scripts` | the three S1-4 tags, S1-1's `TRY004` in `schema.py`, and this follow-up's `TRY004` in `registry.py` |
| 3. `rag-ingest` + `rag-query` | not re-run: the change only refuses non-class targets, and the `check_imports` row shows all 14 real targets are classes |
| 5. CI | pending the push |

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
