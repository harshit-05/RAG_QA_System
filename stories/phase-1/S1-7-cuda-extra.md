# S1-7: Make the GPU embedder path installable (`cuda` variant of torch)

| | |
| --- | --- |
| **Status** | In review (2026-10-02) — spike done, option (a) chosen and approved in plan mode; local verification passed; CI after the push; branch `feat/s1-7-cuda-torch-variant` |
| **Closes** | FR-1 (the GPU axis, declarable → installable) |
| **Depends on** | S1-1 (ARCHITECTURE.md §1.1 DEC-12; DEC-1 rule 2) |
| **Model** | fable |
| **Plan-first** | yes — the mechanism is chosen by the step-0 spike |

## Goal

The `minilm_cuda` and `multilingual_mpnet_cuda` config entries are declarable
today but not installable: the lockfile pins CPU-only torch on purpose. This
story adds mutually exclusive CPU / CUDA torch variants so one documented
command produces a CUDA torch build, making the Colab/Kaggle bulk re-embedding
path real — **without changing what a plain `uv sync` installs on this host.**

## Scope

- **Step 0 — spike, before editing `pyproject.toml` for real.** In a scratch
  copy of the repo, confirm what a *plain* `uv sync` installs once torch is
  split. The expected trap (uv's documented behaviour, not yet run here): with
  torch declared only inside `cpu`/`cuda` **extras**, a sync that names neither
  extra has no source rule for torch, so `sentence-transformers` pulls it from
  PyPI — the CUDA build with `nvidia-*` wheels. Extras have no default. Pick the
  mechanism from what the spike shows:
  - **(a) Dependency groups** `cpu` / `cuda`, declared conflicting in
    `[tool.uv] conflicts`, sources scoped per group, and
    `[tool.uv] default-groups = ["dev", "cpu"]` so a plain `uv sync` stays CPU.
    GPU: `uv sync --no-group cpu --group cuda`. Preferred if the spike confirms
    group-scoped sources work on uv 0.11.8.
  - **(b) Conflicting extras**, per uv's PyTorch guide. Only acceptable if every
    install path names an extra: CI (`uv sync --locked --extra cpu`), README,
    and the fresh-clone check in S1-8. A plain `uv sync` changing meaning is a
    regression of the v0.1 exit criterion, so (b) must be stated in README and
    CLAUDE.md, not left implicit.
  Record the spike output and the choice in this file before continuing.


### Step 0: spike results (2026-10-02, uv 0.11.8; scratch copies only)

| Finding | Evidence |
| --- | --- |
| Only **`cu130`** carries our torch 2.14 | index listings: `cu128` stops at 2.11.0; `cu130` has `2.14.0+cu130` and `2.14.1+cu130` (cp312, manylinux x86_64) |
| Locking downloads no CUDA wheel | the index publishes `data-core-metadata`; the `.whl.metadata` is 38 KB, against a ~1 GB wheel |
| **(a) dependency groups** — chosen | lock 2.4 s; `langchain-core` stays 1.6.3; plain `uv sync` → `torch 2.14.0+cpu`, 0 nvidia, cuda False; `--no-group cpu --group cuda` → `torch 2.14.1+cu130` + 15 nvidia-*; `--group cuda` → clear conflict error; `--no-dev` keeps CPU; `--locked` (CI) unchanged |
| **(b) conflicting extras** — rejected | a plain `uv sync` plans 15 nvidia-* (CUDA from PyPI): exactly the trap this story predicted. It changes the meaning of the most common command |
| (a2) base torch from the CPU index plus a `cuda` group override — impossible | uv refuses: "Requirements contain conflicting indexes for package `torch`". A conflict needs two *named* sides, and base dependencies are not one |
| The one trap left in (a): `--no-default-groups` | plans 15 nvidia-* (CUDA torch from PyPI via `sentence-transformers`). Mitigated by a CI guard plus documentation, since it cannot be designed out |

The choice and the design were approved in plan mode before any real file
changed.
- `explicit = true` stays on **both** torch indexes — the load-bearing part of
  DEC-1 rule 2: as a general index, `download.pytorch.org` also serves stale
  `langchain-community` releases and uv silently resolves the whole stack down
  to LangChain 0.3.x.
- `.github/workflows/ci.yml`: whatever sync command the chosen mechanism needs
  for CPU; CI must never download `nvidia-*` wheels (~GBs, and a silent
  wrong-variant install).
- `README.md`: one short section — CPU is the default; the GPU command is for a
  CUDA host (Colab/Kaggle); switching embedder entries invalidates the index, so
  re-run `rag-ingest`.
- `config.yaml`: update the comment on the `_cuda` entries — they now need the
  GPU sync command rather than "changing the torch source in pyproject.toml".

## Out of scope

- Collapsing the four embedder entries into a model + device setting
  (**DEC-10**: the axis stays explicit; pydantic-settings does not interpolate
  `${VAR:-default}` in YAML values, which is what that backlog item assumed).
- Any GPU run on this host. CLAUDE.md: CPU-only, never select a `cuda` component
  here.

## Verification

Check the **installed environment**, not the lockfile: with two variants
declared, `uv.lock` legitimately records both resolutions, so `nvidia-*` lines
and a non-`+cpu` torch *will* appear in it.

```bash
# 0. spike output recorded above (plain `uv sync` in the scratch copy → CPU torch?)

# 1. the DEC-1 rule 2 smoke test — a PLAIN sync still installs CPU torch
uv sync
uv pip list 2>/dev/null | grep -iE '^torch |nvidia'    # → torch 2.14.0+cpu, no nvidia-* lines
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # → 2.14.0+cpu False
uv run python -c "import langchain_core; print(langchain_core.__version__)"            # → 1.x
uv lock --check                                          # lockfile consistent with pyproject

# 2. the CUDA variant RESOLVES (resolve only — this host cannot run it).
#    (`uv lock` has no --extra flag; the lock is universal. Dry-run the sync.)
uv sync --dry-run --no-group cpu --group cuda 2>&1 | grep -iE 'torch|nvidia' | head   # option (a)
#   or: uv sync --dry-run --extra cuda ...                                            # option (b)

# 3. the two variants are actually exclusive
uv sync --dry-run --group cpu --group cuda 2>&1 | tail -3     # → conflict error, by design
#   or the --extra cpu --extra cuda form for option (b)

# 4. nothing else moved
uv run pytest -q && uv run rag-ingest && echo "What is this corpus about?" | uv run rag-query
gh run list --limit 1          # CI green, and its sync log shows no nvidia-* downloads
```

### Results (2026-10-02)

**Risk call: branch** (`feat/s1-7-cuda-torch-variant`). It changes how torch is
declared, which is the load-bearing DEC-1 rule 2 setup, and adds a CI guard.

| Check | Result |
| --- | --- |
| 1. plain `uv sync`, real environment | `torch 2.14.0+cpu`, **0 nvidia**, `cuda.is_available()` False, `langchain-core` 1.6.3; only the project itself was rebuilt. `uv lock --check` consistent |
| 2. the CUDA variant resolves (dry run; never run here) | `- torch==2.14.0+cpu`, `+ torch==2.14.1+cu130`, 15 nvidia-* planned |
| 3. the variants exclude each other | `uv sync --group cuda` → `error: Groups cpu (enabled by default) and cuda are incompatible …` |
| the documented trap | `--no-default-groups` plans 15 nvidia-*, as documented |
| CI's own command | `uv sync --locked --dry-run` → "Would make no changes" |
| the new CI guard | exit 0 on the real environment. **Negative controls:** exit 1 for `2.14.1+cu130`, and exit 1 for a plain `2.14.0`, the PyPI build the trap installs |
| 4. nothing else moved | ruff and mypy clean; **184 passed**, coverage 97.92%; `rag-ingest` 561 pages → 1,708 chunks, 0 failed; mistral's answer **word-for-word identical** to every earlier run, same 5 sources, exit 0 |
| CI | after the push |

## Review notes for the human

The review is what a plain `uv sync` installs, not the TOML. Confirm
`torch 2.14.0+cpu` and zero `nvidia-*` packages *in the environment* after a
plain sync and in the CI log, because the failure mode is silent — a successful
install of the wrong variant (ADR-005). Confirm `explicit = true` survived on
both index declarations. **Stated limit of this story: the CUDA path is verified
as a resolve only.** It has never been run, and the story must not imply
otherwise.

## Discovered

- **The CUDA variant resolves one patch ahead:** `2.14.1+cu130` against CPU's
  locked `2.14.0+cpu`. Accepted: GPU and CPU arithmetic already differ by more
  than a patch release. `uv lock --upgrade-package torch` aligns them if parity
  ever matters.
- **Whether Colab/Kaggle drivers meet CUDA 13's requirement is unverified.**
  This host has no NVIDIA GPU. If they don't, the fallback is a CUDA 12 variant
  on `cu128`, which caps torch at 2.11 → backlog.
- **The trap cannot be designed out on uv 0.11.8.** Base torch from the CPU
  index plus a `cuda` group override is rejected as conflicting indexes. So the
  defence is a CI guard (fails unless the installed torch is `+cpu`) plus the
  rule in CLAUDE.md, README and `pyproject.toml`.

## Deviation from plan

- **Mechanism (a), dependency groups,** chosen by the spike, as the story
  preferred. The GPU command is `uv sync --no-group cpu --group cuda`.
- **Additions:** the CI guard step, and one line in CLAUDE.md's uv rule (never
  `--no-default-groups`). The rest of CLAUDE.md's GPU wording stays S1-8's.
- **The `cu130` index specifically:** the story did not name a CUDA version, and
  `cu130` is the only one carrying torch 2.14.
