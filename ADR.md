# Architecture Decision Records — RAG_QA_System

This file records the significant decisions made while turning three overlapping
prototypes (`v1/`, `v2/`, a stale `temp/v2/`) into a single, production-track RAG
service. It exists for two audiences at once: **future sessions** (human or
Claude) who need to know *why* something is the way it is before touching it,
and **you**, as a record of the reasoning patterns that generalize past this
one project.

Each entry follows the same shape:

- **Context** — the situation that forced a choice.
- **Decision** — what was chosen, precisely.
- **Consequences** — what you gain and what you now owe, in exchange.
- **Alternatives considered** — the other options on the table, and the
  specific reason each lost.
- **Lesson** — the general engineering principle this instance teaches, with
  a comparison to how the same mistake/pattern shows up elsewhere.

ADRs are numbered in the order the decisions were made, not by how "important"
they sound — the ordering itself is a lesson (§ADR-001). Status is one of
`Accepted`, `Accepted (provisional)` — right for now, explicitly scheduled to
be revisited — or `Superseded`.

## Index

| ADR | Title | Status | Phase |
| --- | --- | --- | --- |
| [001](#adr-001-spec-driven-story-loop-instead-of-conversational-development) | Spec-driven story loop instead of conversational development | Accepted | Process |
| [002](#adr-002-repo-hygiene-before-any-feature-work) | Repo hygiene before any feature work | Accepted | 0 |
| [003](#adr-003-delete-v1-and-the-stale-temp-copy-rather-than-branch-them) | Delete v1 and the stale temp copy rather than branch them | Accepted | 0 |
| [004](#adr-004-uv-as-the-only-python-tool-with-a-pinned-interpreter) | uv as the only Python tool, with a pinned interpreter | Accepted | 0 |
| [005](#adr-005-explicit-scoped-pytorch-index-instead-of-a-general-one) | Explicit, scoped PyTorch index instead of a general one | Accepted | 0 |
| [006](#adr-006-migrate-to-langchain-1x-now-instead-of-pinning-legacy) | Migrate to LangChain 1.x now instead of pinning legacy | Accepted | 0 |
| [007](#adr-007-restricted-import-surface-for-langchain) | Restricted import surface for LangChain | Accepted (provisional) | 0–1 |
| [008](#adr-008-hand-composed-lcel-chain-instead-of-a-prebuilt-chain-class) | Hand-composed LCEL chain instead of a prebuilt chain class | Accepted | 0 |
| [009](#adr-009-module-split-with-an-enforced-one-way-dependency-direction) | Module split with an enforced one-way dependency direction | Accepted | 0 |
| [010](#adr-010-config-driven-object-graph-via-a-target-convention) | Config-driven object graph via a `_target_` convention | Accepted | 0 |
| [011](#adr-011-mistral-over-qwen27b-for-the-phase-0-proof) | `mistral` over `qwen2:7b` for the Phase 0 proof | Accepted (provisional) | 0 |
| [012](#adr-012-renaming-docs-to-corpus-instead-of-writing-a-standing-warning) | Renaming `docs/` to `corpus/` instead of writing a standing warning | Accepted | 0 |
| [013](#adr-013-faiss-now-a-swap-seam-for-qdrant-later) | FAISS now, a swap seam for Qdrant later | Accepted (provisional) | 0 → 3 |
| [014](#adr-014-accepting-allow_dangerous_deserializationtrue-with-a-documented-invariant) | Accepting `allow_dangerous_deserialization=True` with a documented invariant | Accepted | 0 |
| [015](#adr-015-no-target-import-allowlist-yet) | No `_target_` import allowlist yet | Accepted (provisional) | 0 → 1 |
| [016](#adr-016-accepting-an-unmet-latency-slo-instead-of-buying-a-gpu) | Accepting an unmet latency SLO instead of buying a GPU | Accepted (provisional) | 0 → 3 |
| [017](#adr-017-phase-exits-as-release-points-not-calendar-dates) | Phase exits as release points, not calendar dates | Accepted | Process |
| [018](#adr-018-model-routing-for-ai-assisted-implementation) | Model routing for AI-assisted implementation | Accepted | Process |

---

## ADR-001: Spec-driven story loop instead of conversational development

**Context.** The starting point was a repo with two full prototype rewrites of
the same idea (`v1/`, `v2/`) plus a silently-diverged stale copy (`temp/v2/`),
committed binaries, no tests, and two critical bugs (`ISS-01`, `ISS-02`) that
had already cost real debugging time (evidenced by a hand-written
`check_config.py` diagnostic script that didn't even check the keys that were
actually broken). That pattern — rewrite instead of fix, because the previous
version's reasoning was never written down — is what an unstructured,
chat-driven "just ask the AI to build a RAG system" workflow produces at scale.

**Decision.** Adopt a fixed chain of truth, each link a durable artifact, not a
chat transcript:

```text
SRS.md  (what & why)  →  ARCHITECTURE.md  (how, per phase)  →  stories/phase-N/*.md  (one unit of work)  →  code + one commit
```

`stories/STATUS.md` is the board. Every session starts by reading it, not by
re-explaining the project. A story is scoped to one session; anything
discovered outside that scope becomes a backlog line, never a mid-session
detour (see WORKFLOW.md, CLAUDE.md's hard rules).

**Consequences.**

- *Pros:* any session — today or six months from now — can answer "where was
  I and why" by reading three files instead of scrolling chat history.
  Decisions get written down once, at the moment they're made, instead of
  being re-derived (or contradicted) every time someone picks the work back
  up. Review is cheap: the story's Verification section says exactly what
  "done" means before code is written, so review is "did the commands pass,"
  not "re-read every line."
- *Cons:* overhead. Writing a story file for a five-line fix is real friction,
  and the process only pays for itself on multi-session work — for a
  one-off script you'd never do this.

**Alternatives considered.**

- *Ask for features conversationally, session by session, no artifacts* —
  this is exactly the process that produced `v1`/`v2`/`temp/v2`: three
  attempts because nobody could reconstruct why the previous attempt looked
  the way it did, so it was faster to start over. Rejected because it's the
  problem, not a solution to compare against.
- *Full BMAD-style ceremony (heavier agile-for-AI frameworks)* — rejected as
  disproportionate for a single-maintainer project; WORKFLOW.md explicitly
  frames this as "not BMAD ceremony, not YOLO autopilot."

**Lesson.** This is the same fix as "write an ADR" or "write a design doc
before the PR" in any team — the artifact isn't bureaucracy, it's what lets
work be resumed by someone (including future-you) who wasn't in the room
when the decision was made. The tell that you need this: you're about to make
the *same* design decision for the second time, or you're rewriting something
instead of fixing it because you don't trust your own past reasoning.

---

## ADR-002: Repo hygiene before any feature work

**Context.** The git tree tracked `__pycache__/`, a 341 KB screen recording,
a `.py.save` editor backup, and FAISS index binaries — none of which are
source (`ISS-09`). No `.gitignore` existed at all. This is Phase 0's very
first story (S0-1), run even before the architecture pass.

**Decision.** Add `.gitignore` and `git rm --cached` every non-source
artifact, deleting genuinely useless files (the screen recording, the
`.save` backup) from disk entirely, but keeping useful-but-generated ones
(the FAISS index, `__pycache__`) on disk while untracking them. History
itself was **not** rewritten — the story explicitly notes this is "low value
for a repo this young."

**Consequences.**

- *Pros:* every commit from this point on is reviewable by a human — a diff
  that's 90% binary noise gets rubber-stamped instead of read. Clone size and
  time stay small. A `.gitignore` also documents, for free, what the project
  considers "not source" — a fast onboarding signal.
- *Cons:* stale binaries already in history remain in history (repo size
  doesn't shrink until/unless a history rewrite happens later); doing this
  as its own commit before any feature work means one more commit for
  reviewers to look at.

**Alternatives considered.**

- *Rewrite history to purge the binaries retroactively* (`git filter-repo` /
  BFG) — rejected: it invalidates every existing clone and commit hash for a
  repo that's days old with a single maintainer, i.e. real cost for a
  benefit (disk space) that doesn't matter yet.
- *Ignore it and move on to features* — rejected because every subsequent
  `git status`/`git diff` stays noisy, and — the concrete trap that actually
  happened here — a `git reset` silently **un-ignores already-tracked
  files** (`.gitignore` does not retroactively untrack anything), so if this
  isn't fixed early, it quietly regresses every time someone resets. This bit
  the project once mid-Phase-0 (recorded in STATUS.md's pre-flight caveats)
  even *after* the fix, because the fix hadn't been committed yet.

**Lesson.** `.gitignore` is necessary but not sufficient — it only stops
*future* additions. A file already tracked stays tracked until you explicitly
`git rm --cached` it, and a `git reset` restores tracked-ness from HEAD. This
trips up experienced engineers, not just beginners: the failure mode is
"I added the ignore rule, why is this file back?" The general principle —
fix structural rot before building on top of it, because every layer above
inherits and amplifies the mess below — is why "boy scout" cleanup stories
exist at the start of a project, not just at the end.

---

## ADR-003: Delete v1 and the stale temp copy, rather than branch them

**Context.** Three folders implemented overlapping versions of the same
system: `v1/` (a linear script), `v2/` (a config-driven redesign — the right
instinct, but broken by `ISS-01`), and `temp/v2/`, an untracked copy that had
silently diverged from `v2/` by exactly one line (a `_build_object` →
`build_object` rename). Nobody could say with confidence which copy was
"live" without diffing all three.

**Decision.** Verify `temp/v2/` was genuinely superseded (byte-diff against
`HEAD:v2/`, confirmed one-line divergence, confirmed it predates that rename
in git history), then delete `v1/` and `temp/` entirely. Git history — not a
folder — is the record of what v1 looked like.

**Consequences.**

- *Pros:* one source tree means one place to fix a bug, one set of imports
  that resolve unambiguously, and no risk of shipping code from the wrong
  copy. The dependency-direction contract (ADR-009) as a single-tree
  guarantee has real teeth: it can't be undermined by a fourth secret copy.
- *Cons:* this is a one-way door if it turns out `v1` had logic nobody
  remembered to check. The mitigation was doing the verification *before*
  deleting (S0-3's Review notes explicitly call out: "confirm nothing from
  v1 was silently needed") — the only unique v1 logic (`LOADER_MAPPING`) was
  confirmed superseded by v2's config-driven loaders before the delete.

**Alternatives considered.**

- *Keep both trees, pick one at runtime via a flag* — rejected: this is how
  you get a *fourth* copy's worth of maintenance burden (two live code paths
  to keep in sync) instead of zero. Nobody was asking to run v1 in
  production; it was dead weight, not a supported option.
- *Move v1 to a `legacy/` folder instead of deleting* — rejected for the same
  reason `temp/v2/` was a problem in the first place: a second folder starts
  fresh and drifts the moment anyone touches the main tree without touching
  it too. Git history is the "keep it around" mechanism that doesn't drift.

**Lesson.** "Delete it, git history remembers" is the correct instinct almost
every time you're tempted to keep an old version around "just in case" —
version control exists precisely so a deletion is reversible without the
ongoing cost of a second copy sitting in the working tree, silently
diverging (as `temp/v2/` proves happens in practice, not just in theory).
The discipline is verifying *before* deleting, not skipping the delete.

---

## ADR-004: uv as the only Python tool, with a pinned interpreter

**Context.** `v1/requirements.txt` had its pins commented out and listed
conflicting `faiss-cpu` **and** `faiss-gpu`; `v2/` declared no dependencies
at all (`ISS-08`). The host has no conda/pyenv/poetry, system Python is
3.14.6 (too new for some ML wheel coverage at the time), and there's no
NVIDIA GPU.

**Decision.** `uv` only — `uv add`/`uv run`/`uv sync`, never bare `pip`,
never conda — with the interpreter pinned to 3.12 via `.python-version`,
and every runtime dependency declared with a version range in
`pyproject.toml`, locked in `uv.lock`.

**Consequences.**

- *Pros:* `uv sync` on a fresh clone reproduces the exact same dependency
  graph every time — "works on my machine" stops being a risk. A pinned
  interpreter avoids the class of bug where a library hasn't published
  wheels for the newest CPython yet. One lockfile is one source of truth for
  "what's actually installed," closing `ISS-08` outright.
- *Cons:* another tool to have installed and know the invocation for; `uv`
  is younger than pip/poetry, so less Stack-Overflow-era tribal knowledge
  exists for edge cases.

**Alternatives considered.**

- *pip + requirements.txt* — the status quo that produced `ISS-08` in the
  first place: no lockfile means no reproducibility, and nothing stops a
  contradictory pin (`faiss-cpu` + `faiss-gpu` together) from being merged.
- *conda* — rejected: heavier, and this project has zero non-Python native
  dependencies that would justify conda's package management beyond what
  wheels already cover.
- *Poetry* — a reasonable alternative in the abstract, but not installed on
  this host and `uv` is materially faster at resolution (relevant later —
  see ADR-005, where a resolution trap needed several fast iterate-and-check
  cycles to diagnose).

**Lesson.** A dependency file without a lockfile is a *suggestion*, not a
guarantee — `>=` ranges resolve differently on different days as new
versions publish. The generalizable rule: any project you expect to still
build in six months needs a lockfile, full stop, regardless of which tool
produces it (`uv.lock`, `poetry.lock`, `package-lock.json`, `Cargo.lock` —
same principle everywhere).

---

## ADR-005: Explicit, scoped PyTorch index instead of a general one

**Context.** `torch`'s CPU-only wheels live at `download.pytorch.org/whl/cpu`
— but that index *also* hosts stale `langchain-community` releases. Declared
as a general package index (the naive way to add a custom index in most
tools), `uv` would silently resolve the **entire** dependency stack against
it, downgrading the whole LangChain stack to 0.3.x without any error — a
silent, wrong resolution, not a crash.

**Decision.**

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

`explicit = true` means this index is *only* consulted for packages
explicitly routed to it via `tool.uv.sources` — here, just `torch`. Every
other package resolves from PyPI as normal.

**Consequences.**

- *Pros:* CPU-only torch (`2.14.0+cpu`, confirmed zero `nvidia-*` wheels in
  the lock) without dragging the rest of the stack to a stale index's
  versions. The fix is two TOML stanzas, not a workaround in application
  code.
- *Cons:* this is a non-obvious trap — nothing about a "general index"
  declaration *looks* wrong; it fails silently (a successful resolve, just
  the wrong one) rather than loudly. Anyone maintaining this later has to
  know to check `uv.lock` for `langchain-core` version and absence of
  `nvidia-*` strings as a smoke test (which is exactly what S0-2's
  Verification section does).
- The general lesson has a mirror-image failure mode worth naming: if this
  index were *not* scoped to CPU-only intentionally, a GPU-equipped host
  could silently get CPU wheels instead of CUDA ones — same mechanism,
  opposite unwanted outcome.

**Alternatives considered.**

- *Add the index as a general fallback index* — this is the trap itself,
  not an alternative; included only because it's the default, easy-to-reach
  configuration and had to be actively avoided.
- *Pin `torch==2.x+cpu` directly from PyPI without a custom index* — PyPI
  does not host the `+cpu`-suffixed CPU-only builds at all for every
  platform/version combination; the PyTorch index exists precisely because
  PyPI can't serve this artifact. Not a real option.

**Lesson.** "Which index does a package actually resolve from" is an
under-examined question in most dependency graphs until it silently breaks
something. Any time you add a custom/private package index to a resolver
(uv, pip, npm scopes, Maven repositories), ask explicitly: does this index
apply to *one* package, or *every* package the resolver might look up? The
default is usually "every package," and that default is usually wrong.

---

## ADR-006: Migrate to LangChain 1.x now instead of pinning legacy

**Context.** The existing code (`v2/pipeline_builder.py`) targets LangChain
0.2.x APIs. A fresh dependency resolve on this host — the only kind
possible, since no lockfile existed yet (ADR-004) — lands LangChain 1.x by
default (`langchain-core 1.6.3`), with 0.2.x-era code (`langchain_classic`)
demoted to a legacy compatibility package that ships transitively. This is
the single highest-leverage decision in Phase 0: it determines the shape of
every chain-construction line written afterward (ADR-008).

**Decision.** Migrate to 1.x now, not later, with two hard rules: (1)
application code imports only from `langchain_core`,
`langchain_text_splitters`, `langchain_community`, `langchain_huggingface`,
`langchain_ollama` — never the `langchain` meta-package, and not
`langchain_classic` in Phase 0–1 (see ADR-007); (2) the query chain is
hand-composed LCEL (ADR-008), not the legacy `RetrievalQA` class or the
1.x-era `create_retrieval_chain` classic wrapper.

**Consequences.**

- *Pros:* the code and the lockfile agree from day one — no "we'll migrate
  later" debt accruing interest while the ecosystem moves further from
  0.2.x. Every future dependency bump stays on the maintained line. The
  decision closes `ISS-17` (deprecated APIs) as a side effect of being made
  early rather than as separate cleanup work.
- *Cons:* real, immediate cost — the existing `v2/pipeline_builder.py`
  logic has to be rewritten against a different chain-construction paradigm
  (`RetrievalQA.from_chain_type(...)` → LCEL pipe composition) before
  *anything* runs end-to-end. This is why S0-3 explicitly *expects*
  `import rag_qa.chain` to fail with `ModuleNotFoundError` until the
  migration story (S0-5) lands — the plan accepts a known-broken
  intermediate state rather than pretending migration is free.

**Alternatives considered.**

- *Pin `langchain-core<1` (stay on 0.2.x)* — rejected: 0.2.x is unmaintained
  going forward, still carries pydantic-v1 compatibility shims, and
  critically, `ragas` (needed for the Phase 2 evaluation gate, `FR-7`)
  requires `langchain-core>=1`. Pinning old would make the eval extra
  **unable to share a single lockfile** with the runtime dependencies —
  a hard technical blocker, not just a preference.
- *Pin to 0.3.x (the intermediate release line)* — rejected as "same import
  rewrite later, no gain": 0.3.x is itself being superseded, so choosing it
  buys nothing but defers the exact same migration cost to a second future
  session instead of paying it once, now, while the code is already being
  touched for other reasons.

**Lesson.** When you must rewrite something anyway (here: touching every
chain-construction call site regardless of target version), evaluate
against the version you'd want to be on in a year, not the version that
requires the smallest immediate diff. "We'll upgrade later" long-lived
dependencies are usually never upgraded later — the cost only grows, and
here it would have blocked a downstream requirement (`ragas`) outright, not
just cost extra effort.

---

## ADR-007: Restricted import surface for LangChain

**Context.** LangChain 1.x ships `langchain_classic` as a compatibility
package holding the pre-1.x chain classes (`RetrievalQA`,
`create_retrieval_chain`) — installed transitively via
`langchain-community` whether or not application code ever imports it.
Nothing stops a future contributor from reaching for the familiar
`RetrievalQA` out of habit and quietly reintroducing the exact deprecated
pattern the migration (ADR-006) was meant to leave behind.

**Decision.** A explicit allowlist of importable LangChain packages
(`langchain_core`, `langchain_text_splitters`, `langchain_community`,
`langchain_huggingface`, `langchain_ollama`), with `langchain_classic`
named as **not imported** in Phase 0–1, verified mechanically:
`uv run python -W error::DeprecationWarning -c "import rag_qa..."` plus a
`grep -c "langchain_classic\|^from langchain import\|^import langchain$"`
check in S0-5's own Verification section. Phase 2 is explicitly permitted a
narrow, named exception: `langchain_classic.retrievers.document_compressors
.CrossEncoderReranker`, if a hand-rolled Runnable proves larger than reusing
the wrapper — a decision deferred, not a rule broken.

**Consequences.**

- *Pros:* the rule is checkable by machine (a grep), not just written prose
  that erodes over time — it lives in a story's Verification section, so
  violating it fails the story, not a code review someone might skip.
  It keeps a stated architectural intent (LCEL-only chains) from silently
  regressing one convenient import at a time.
- *Cons:* it's a "for now" rule with a named escape hatch already built in
  (the Phase 2 reranker exception) — meaning it requires active maintenance
  to keep meaningful rather than becoming a stale comment nobody enforces.
  Status is marked "provisional" for exactly this reason.

**Alternatives considered.**

- *Trust code review to catch a stray `langchain_classic` import* — rejected
  as the same class of failure the whole SRS is a reaction to: `ISS-07`
  ("zero automated tests — nothing would have caught ISS-01 before a
  human did") is the general argument against relying on human vigilance
  for anything mechanically checkable.
- *Uninstall/exclude `langchain_classic` entirely* — not possible; it's a
  transitive dependency of `langchain-community` itself, not something the
  project chose to add.

**Lesson.** An architectural boundary that only exists as a sentence in a
document erodes the first time someone's under deadline pressure and the
deprecated-but-familiar API is one import away. Where a rule can be
expressed as a grep, a lint rule, or an import-linter config, do that
instead of (not in addition to, though both is fine) writing it down as
prose — this is the same principle behind ArchUnit rules in Java codebases
or `import-linter` contracts in Python monorepos.

---

## ADR-008: Hand-composed LCEL chain instead of a prebuilt chain class

**Context.** With LangChain 1.x chosen (ADR-006) and `langchain_classic`
off-limits by default (ADR-007), the query chain needs a construction
pattern that doesn't reach for the deprecated `RetrievalQA` class the old
code used, and doesn't reach for `create_retrieval_chain` either (which
lives in `langchain_classic`).

**Decision.** Compose the chain directly from LCEL primitives:

```python
retriever = store.as_retriever(**retriever_cfg)
prompt = ChatPromptTemplate.from_messages([("system", cfg_system), ("human", cfg_human)])
to_prompt = RunnableLambda(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
chain = (
    RunnablePassthrough.assign(context=itemgetter("question") | retriever)
    | RunnablePassthrough.assign(answer=to_prompt | prompt | llm | StrOutputParser())
)
# chain.invoke({"question": q}) -> {"question": str, "context": list[Document], "answer": str}
```

`format_docs` numbers each chunk and prefixes it with `source`/`page`
metadata so the model can cite what it read, and the returned dict always
carries `context` alongside `answer` — fixing the specific bug where the old
code set `return_source_documents=True` but `main2.py` never read
`result["source_documents"]`, silently discarding citations that were
already being computed (part of `FR-5`).

**Consequences.**

- *Pros:* the resulting `chain` object is a plain LangChain `Runnable` —
  `.invoke()`, and later (Phase 2) `.astream()` for token streaming, come
  for free from the LCEL contract rather than needing bespoke code per
  entry point. The same object serves the REPL (`cli.py`) today and the
  planned FastAPI layer tomorrow without a rewrite (`ARCHITECTURE.md §0.4`
  states Phase 2's API calls `chain.astream` on "the same object"). It's
  also explicit: every step of context-formatting → prompting → generation
  is a visible line, not hidden inside a class's `from_chain_type`
  classmethod.
- *Cons:* more code to write than `RetrievalQA.from_chain_type(...)` — a
  one-liner in the old shape becomes ~10 lines of explicit composition.
  Whoever maintains this needs to actually understand LCEL's
  `RunnablePassthrough.assign`/pipe semantics rather than treating chain
  construction as a black box.

**Alternatives considered.**

- *Keep `RetrievalQA.from_chain_type`* — the least-effort option, but it's
  the deprecated shape ADR-006 is migrating away from, and it's what
  silently dropped citations in the first place (`result["result"]` vs the
  richer shape needed for sources).
- *`create_retrieval_chain` (the 1.x-era prebuilt replacement)* — rejected
  specifically because it lives in `langchain_classic`, which ADR-007
  restricts; using it here would undercut the import-surface rule in the
  very module that's supposed to demonstrate the target pattern.

**Lesson.** Prebuilt "batteries included" abstractions (a `from_chain_type`
classmethod, a Django generic view, a Spring Boot starter) trade explicitness
for speed — fine until you need something the abstraction didn't anticipate
(here: actually reading the sources it already computed). When a framework's
convenience wrapper is being deprecated industry-wide in favor of composable
primitives (LCEL over Chain classes; middleware chains over monolithic
frameworks elsewhere), that's usually a signal the wrapper was hiding
exactly the kind of control you eventually need.

---

## ADR-009: Module split with an enforced one-way dependency direction

**Context.** `v2/pipeline_builder.py` mixed pure config-resolution logic
(`import_from_string`, `build_object`), FAISS calls, and chain construction
in one file, and — critically — `v2/file_processor.py` (ingestion)
transitively pulled in the same module that builds the query chain,
meaning running ingestion dragged the entire LangChain query stack along
with it, whether or not it was needed or even working.

**Decision.** Split into `registry.py` (pure Python, zero LangChain
imports — `import_from_string`, `build_object`, `resolve_ref`), `config.py`
(YAML loading), `vectorstore.py` (the only FAISS calls), `chain.py` (query
chain construction), `ingest.py`, and `cli.py`, with one architectural rule
enforced across all of them: **`ingest.py` must never import `chain.py`.**
Dependency direction: `cli → chain → {vectorstore, registry, config}`;
`ingest → {vectorstore, registry, config}`.

**Consequences.**

- *Pros:* ingestion and querying are now genuinely independent failure
  domains. This was proven, not assumed — after the split, `rag-ingest` ran
  all the way to a (separate, expected) `FileNotFoundError`, while
  `rag-query` still failed on the LangChain-migration-pending
  `ModuleNotFoundError` — different failures for different commands is the
  actual evidence the coupling is gone, not just a code-review claim. It
  also makes `registry.py`/`config.py` importable and testable without the
  ML stack installed at all (they have zero LangChain imports), which
  matters directly for `NFR-8`'s coverage goal — you can unit-test config
  resolution in milliseconds instead of paying for a model load every test
  run.
- *Cons:* more files to navigate than one big module; the boundary has to
  be actively maintained — a later change (see the code-review findings on
  this same codebase) can still quietly duplicate logic across the boundary
  (e.g., both `chain.py` and `ingest.py` independently re-deriving the
  embedder config path) even when the *import* direction stays correct.
  Splitting files doesn't automatically eliminate duplication; it just
  makes the duplication visible if you look.

**Alternatives considered.**

- *One `pipeline.py` module, keep everything together* — this is the status
  quo that caused the actual bug (ingestion dragging in the query stack);
  rejected on direct evidence, not preference.
- *Split by LangChain "layer" (all component construction in one file,
  regardless of ingest vs. query use)* — considered and rejected because it
  wouldn't have prevented the specific coupling that mattered: the
  dependency-direction violation was about *what imports what*, and a
  layer-based split doesn't guarantee acyclic imports the way a
  responsibility-based split with an explicit "X must never import Y" rule
  does.

**Lesson.** "Split into modules" is not by itself an architecture — the
value is in the *rule about which modules may depend on which others*,
stated explicitly enough to grep for and verify (`grep -c "chain" ingest.py`
should count docstring mentions only, never an import). This is the same
principle behind layered-architecture rules ("controllers may not import
the database layer directly") or Conway's-Law-aware module boundaries in
larger codebases: the boundary is worthless if it's only a folder name and
nobody checks that imports actually respect it.

---

## ADR-010: Config-driven object graph via a `_target_` convention

**Context.** `v2/pipeline_builder.py` already had the right instinct: a YAML
"component library" where each entry names a class via a dotted import path
under a `_target_` key, and a recursive builder instantiates the whole graph
from config. SRS.md calls this "a legitimate, well-known design — a
home-grown Hydra" (referencing Meta's Hydra config framework) and explicitly
says to preserve it through the rewrite rather than replace it.

**Decision.** Keep the pattern, isolated into `registry.py`:

```python
def build_object(config_dict):
    if isinstance(config_dict, dict) and "_target_" in config_dict:
        cls = import_from_string(config_dict["_target_"])
        args = {k: build_object(v) for k, v in config_dict.items() if k != "_target_"}
        return cls(**args)
    elif isinstance(config_dict, dict):
        return {k: build_object(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [build_object(item) for item in config_dict]
    else:
        return config_dict
```

Any component — an embedder, an LLM, a splitter, eventually a reranker —
is swapped by editing YAML, never code. `resolve_ref` separately navigates
dotted pipeline references (`components.llms.mistral_ollama`) to the
component dict a pipeline stage should use.

**Consequences.**

- *Pros:* this is real dependency injection without a DI framework —
  swapping the embedder or the LLM backend is a one-line YAML edit, and the
  vector-store swap planned for Phase 3 (ADR-013) is designed around this
  same mechanism (`vectorstore.py` becomes a `_target_`-built component,
  "nothing else changes" per `ARCHITECTURE.md §0.5`). It also means
  `registry.py` has zero LangChain imports and is trivially unit-testable.
- *Cons:* it's config-as-code — `import_from_string` will import and
  instantiate **any** dotted path with **any** kwargs from the YAML file,
  which SRS.md flags directly: "config is effectively arbitrary code
  execution" (`ISS-04`, High severity). This is a real, named security
  tradeoff, not an oversight — see ADR-015.

**Alternatives considered.**

- *Hardcode component choices in Python, branch on a config flag* — this is
  what `v1/` did (hardcoded model choice and paths); rejected because every
  new component (a new embedder, a new LLM provider) requires a code change
  and a redeploy instead of a config edit, and it's exactly the pattern the
  rewrite was meant to leave behind.
- *A heavier DI framework (dependency-injector, or a full Hydra
  dependency)* — not adopted; the existing hand-rolled recursive builder
  already does what's needed in ~25 lines with no new dependency, and
  SRS.md's judgment ("legitimate, well-known design... worth preserving")
  is that reinventing it with an external library buys little here.

**Lesson.** Config-driven instantiation (the pattern behind Hydra, Spring's
XML/annotation-based beans, Kubernetes' declarative manifests) trades
compile-time safety for runtime flexibility — you can reconfigure without
redeploying, but a typo in a string (`ISS-01`'s `llmS`/`llms` mismatch) is
now a runtime `KeyError` instead of a type error caught before ship. The
pattern is a legitimate, common tradeoff — the discipline it demands in
return is schema validation *before* first use (deferred here to Phase 1,
explicitly, not silently) and, if the config source isn't fully trusted, an
allowlist on what it's allowed to import (ADR-015).

---

## ADR-011: `mistral` over `qwen2:7b` for the Phase 0 proof

**Context.** The committed config referenced `qwen2:7b` as the query LLM,
but the environment only had `phi3`, `codellama`, and `mistral` pulled via
Ollama — `qwen2:7b` was never actually downloaded (CLAUDE.md's environment
facts note this explicitly).

**Decision.** Repoint `pipeline.query.llm` to a new `mistral_ollama`
component entry for the Phase 0 proof, keep `qwen2_ollama` in the config as
an unused, still-defined entry, and use `phi3` as an even-lighter fallback
if a memory check at proof time shows less than ~6 GB free.

**Consequences.**

- *Pros:* zero download time, and `mistral` is a known-good, already-tested
  model on this host — the decision explicitly notes the reason is
  "known-good here, not RAM" (both models are the same ~4.4–5 GB weight
  class), i.e. it optimizes for proof velocity in Phase 0, not for a
  permanent model choice.
- *Cons:* it's an explicit stand-in, not a validated "this is the right
  model for the job" decision — quality/accuracy comparison between
  `mistral` and `qwen2:7b` was never done and is deliberately deferred
  ("DEC-2 is revisited when the Phase 2 eval harness exists"). Shipping
  this without that caveat recorded would risk it calcifying into a
  permanent choice nobody consciously made.

**Alternatives considered.**

- *Pull `qwen2:7b` to match the config as-shipped* — rejected for Phase 0:
  it's a multi-GB download for a model with no evidence it's better suited
  to this task than what's already available, on a host where "don't run
  CPU-hour-scale jobs silently" is a standing policy (GPU/CPU policy in
  CLAUDE.md). Pulling it is cheap to do later, once Phase 2's evaluation
  harness can actually compare it against `mistral` on real metrics instead
  of a guess.
- *Block Phase 0 on resolving which model is "correct"* — rejected: Phase
  0's exit criterion is "a fresh clone runs ingestion and answers a query
  end to end," not "the optimal model is selected." Model selection is a
  quality question that needs the Phase 2 eval harness to answer
  rigorously; blocking on it here would gate a plumbing milestone on a
  research question.

**Lesson.** Not every decision needs to be the *final* decision to be worth
making explicitly — the key discipline is naming a choice as provisional
and stating exactly what would need to exist to revisit it properly (here:
an eval harness with real metrics), rather than either (a) blocking
progress until every open question is resolved, or (b) making the
placeholder choice silently and letting it become permanent by default
through nobody ever revisiting it.

---

## ADR-012: Renaming `docs/` to `corpus/` instead of writing a standing warning

**Context.** The RAG document corpus (the PDFs to be ingested) lived in
`docs/` — the name every other convention in the software world reserves
for project documentation. The project's text loader claims `.md` files,
so any project documentation someone dropped into `docs/` following normal
convention would get silently chunked and embedded into the vector index.
The interim fix was a CLAUDE.md hard rule: "never put markdown docs there."

**Decision.** Rename the corpus directory to `corpus/` (S0-4), then
repurpose the now-empty `docs/` for actual project documentation — moving
`SRS.md`, `WORKFLOW.md`, `ARCHITECTURE.md` there (S0-7) — and **delete**
the standing warning rather than keep maintaining it, replacing it with the
inverted, now-true statement: `corpus/` is the RAG corpus, `docs/` is
documentation.

**Consequences.**

- *Pros:* the directory name now means what every contributor, tool, and
  future Claude session will assume it means by convention, so the failure
  mode (a doc silently ingested into the vector index) becomes structurally
  impossible rather than "prevented by everyone remembering a rule."
  `ARCHITECTURE.md`'s own reasoning: "the old layout needed a standing rule
  to stay safe... deleting the rule rather than maintaining it" is the
  point — a correct rule you have to keep re-stating is worse than a
  correct default nobody has to remember.
- *Cons:* it's a two-story migration (S0-4 renames the corpus, S0-7 moves
  the docs in) with a real ordering dependency (`docs/` has to be empty
  before it can be repurposed) and a cross-reference-fixing pass (every file
  linking to the old `SRS.md`/`WORKFLOW.md`/`ARCHITECTURE.md` paths needs
  updating, verified in S0-7 by checking for dangling links).

**Alternatives considered.**

- *Keep `docs/` as the corpus permanently, keep the CLAUDE.md warning
  forever* — the status quo; rejected because a rule that has to be
  remembered by every future session/contributor is a rule that will
  eventually be forgotten once. `ARCHITECTURE.md` states this outright:
  "a stale rule here is worse than no rule, because every future session
  reads it as authoritative."
- *Rename `docs/` to something else non-standard for the corpus (e.g.
  `data/`, `source_docs/`) but leave project docs at root* — considered,
  but doesn't fix the root problem as cleanly: `docs/` remains a landmine
  name for the *next* person who doesn't know the local convention,
  whereas `corpus/` for RAG source material and `docs/` for documentation
  are both immediately legible without reading any project-specific rule
  at all.

**Lesson.** When a bug's fix is "remember not to do X," prefer to instead
make X structurally impossible or renamed out of the way of common
convention — a naming collision with an industry-standard convention
(`docs/` almost universally means project documentation) is a trap for
every newcomer, not a one-time mistake. This generalizes past this project:
if your fix for a footgun is a comment or a wiki page saying "don't," ask
whether a rename, a type system, or a directory structure could make the
mistake unrepresentable instead.

---

## ADR-013: FAISS now, a swap seam for Qdrant later

**Context.** The immediate Phase 0 need is "persist and search embeddings
locally," which FAISS (`faiss-cpu`, on-disk index + pickle) already does
and the existing code already used. The target architecture (`SRS.md §6.2`)
calls for a store supporting native upsert and metadata filtering, which
on-disk FAISS does not — motivating a lean-but-deferred decision (`DEC-3`)
toward Qdrant for Phase 3.

**Decision.** Use FAISS for Phase 0–2, but confine every FAISS-specific call
to exactly one module, `vectorstore.py`, exposing only two functions —
`create_store(chunks, embeddings, cfg)` and `open_store(embeddings, cfg)` —
with every caller interacting only through `.as_retriever()` /
`.add_documents()`, never a FAISS-specific method directly. Qdrant is the
lean pick for Phase 3 (over pgvector) specifically because it offers native
hybrid dense+sparse retrieval in one container, with no Postgres dependency
to operate, unless a Postgres instance already exists in the target
deployment.

**Consequences.**

- *Pros:* Phase 3's store migration is scoped, by design, to rewriting two
  function bodies and nothing else (`ARCHITECTURE.md §0.5`) — no caller
  code changes because callers were never allowed to depend on
  FAISS-specific behavior. This is a real, load-bearing seam, not aspirational:
  a code review of the actual codebase confirmed FAISS imports appear
  nowhere outside `vectorstore.py`.
- *Cons:* the seam only holds as long as its contract (config in, generic
  vector-store interface out) doesn't leak FAISS-specific assumptions.
  A code review of this exact module found a live risk to that contract:
  both `create_store`/`open_store` currently take the *entire* app config
  dict rather than just the path/connection info they need — meaning a
  future backend needing different config shape (a Qdrant host/collection
  instead of a file path) will have to reach into that same full dict
  anyway, partially undercutting the "rewrite two bodies, nothing else"
  promise. A stricter seam would resolve the path/connection value once, in
  the caller, and pass only that in.

**Alternatives considered.**

- *Adopt Qdrant now instead of deferring* — rejected for Phase 0: it adds an
  operational dependency (a running Qdrant container) before the immediate
  goal (get *anything* running end to end, "make it run, make it honest")
  is met. Phase 0's exit criterion doesn't need upsert or hybrid search.
- *pgvector instead of Qdrant, decided now* — the lean pick is Qdrant unless
  a Postgres instance is already part of the target deployment; deferred to
  the Phase 3 architecture pass rather than locked in now, because that's a
  deployment-environment fact this project doesn't have yet.

**Lesson.** "Defer the expensive decision, but build the seam that makes
deferring safe" is different from just procrastinating — the difference is
verifiable: can you point to the one module that would need to change, and
confirm nothing else references the thing you're deferring? If yes, you've
bought real optionality. If a "seam" module still leaks its underlying
implementation's assumptions into its interface (as the full-config-dict
parameter here does, partially), the seam is weaker than it looks and is
worth tightening before the swap, not after.

---

## ADR-014: Accepting `allow_dangerous_deserialization=True` with a documented invariant

**Context.** FAISS's `load_local` requires `allow_dangerous_deserialization
=True` to unpickle a saved index — pickle deserialization of untrusted data
is a well-known remote-code-execution vector. The old code set this flag
with no comment explaining why it's safe here.

**Decision.** Keep the flag (there's no supported alternative for loading a
local FAISS index), but document the safety invariant explicitly in
`vectorstore.py`'s own module docstring: the index directory is *only ever*
produced by this project's own `rag-ingest` command, on this host — never
point it at an index from an untrusted source.

**Consequences.**

- *Pros:* the risk is named and scoped instead of silently present. Anyone
  extending the system later (e.g., accepting an index upload from an
  external source, or downloading a pre-built index from a shared location)
  has a documented tripwire telling them exactly why that specific change
  would break a safety assumption the rest of the system relies on.
- *Cons:* it's a real, live constraint on how the system can ever be
  extended — a "download a shared pre-built index to save re-ingestion
  time" feature (which would otherwise be a reasonable performance
  optimization) is foreclosed by this invariant unless a non-pickle
  serialization format is adopted instead. The mitigation is documentation,
  not elimination of the risk — this is a deliberate, accepted tradeoff,
  not a fix.

**Alternatives considered.**

- *Refuse to set the flag / find a non-pickle FAISS load path* — not
  available in the LangChain FAISS integration used here; the flag is a
  hard requirement of `FAISS.load_local`, not an optional toggle with a
  safer default.
- *Switch to a vector store with a non-pickle persistence format now* —
  this is effectively pulling ADR-013's Phase 3 decision forward; rejected
  for the same reason (operational cost before Phase 0's goal needs it).

**Lesson.** Not every security risk can be eliminated at the point you find
it — sometimes the correct action is to *name the invariant that makes the
current use safe*, in the exact place a future change is most likely to
violate it, so the next person extending the system has to consciously
break a stated rule rather than stumble into a silent vulnerability. This
is the same reasoning behind a code comment on `eval()` or `pickle.load()`
anywhere: the risk is real, the constraint is documented, and the burden
shifts to "don't violate this documented boundary" instead of "hope nobody
notices this is dangerous."

---

## ADR-015: No `_target_` import allowlist yet

**Context.** `registry.build_object` (ADR-010) will import and instantiate
*any* dotted path named in the config file, with any kwargs — SRS.md calls
this "config is effectively arbitrary code execution" (`ISS-04`, High
severity, mapped to OWASP LLM07/08, insecure design / excessive agency).

**Decision.** Do not add an allowlist in Phase 0. `registry.py`'s own
module docstring states the plan explicitly: "Phase 1 adds the import
allowlist (ISS-04) and Pydantic validation *here*, not in callers, which is
why every `_target_` funnels through `build_object`" — i.e., the fix has a
named home and a named phase, it's just not built yet.

**Consequences.**

- *Pros:* Phase 0 stays scoped to "make it run" without also solving a
  design problem (what should the allowlist actually contain? how does it
  interact with third-party loader classes?) that needs more thought than
  a single story session should absorb. The single funnel point
  (`build_object`) is *already* built, meaning Phase 1's allowlist is a
  localized change to one function, not a hunt across the codebase.
- *Cons:* this is a real, live vulnerability in the interim — the config
  file is currently a trusted, maintainer-controlled input, so the risk is
  theoretical *today*, but the moment config becomes attacker-influenced
  (e.g., a future multi-tenant feature letting users supply their own
  pipeline config) this becomes exploitable. A code review of the current
  codebase found a related, concrete gap worth flagging: `ingest.py`'s
  document-loader construction calls `import_from_string` **directly**,
  bypassing `build_object` entirely — meaning Phase 1's allowlist, landing
  inside `build_object` as planned, would not automatically cover loader
  construction unless that call site is also routed through the funnel.

**Alternatives considered.**

- *Ship a minimal allowlist now, even if incomplete* — rejected: a
  half-designed allowlist (e.g., "anything starting with `langchain_`")
  gives false confidence without real analysis of what's actually needed,
  which is worse than an documented, explicit gap with a scheduled fix.
- *Restrict config to a fixed, hardcoded set of components instead of
  dynamic `import_from_string`* — this would eliminate the risk but also
  eliminates the entire point of the `_target_` pattern (ADR-010) — trading
  the whole benefit of config-driven swappability to close one gap that has
  a scoped, less-destructive fix already planned.

**Lesson.** Accepting a known risk for a bounded, documented period is a
legitimate call — the difference between "acceptable technical debt" and
"negligence" is whether the risk is named, scoped to who can currently
exploit it (a trusted maintainer-only config, today), and has a concrete
plan with a phase attached, versus being an unexamined assumption. The
follow-up finding here (loaders bypassing the funnel) is exactly the kind
of thing this kind of explicit tracking is meant to catch before the "fix"
phase arrives and turns out to be incomplete.

---

## ADR-016: Accepting an unmet latency SLO instead of buying a GPU

**Context.** `SRS.md` proposes `NFR-2`: time-to-first-token SHALL be <2s at
p95 with a local 7B-class model. This host has no NVIDIA GPU, 15 GB RAM
shared with desktop applications, and CPU inference of a 7B model measures
at 15–30 seconds to first token — an order of magnitude over the target,
not a rounding error.

**Decision.** Accept the SLO as unachievable on this host for Phase 0–2,
document it explicitly rather than quietly missing it, and carry the
decision ("revise the SLO, or plan GPU serving") forward to the Phase 3
architecture pass as an explicit open question — not a silently-dropped
requirement. For work that would benefit from a GPU but doesn't need to be
a permanent serving decision (bulk re-embedding of a much larger corpus,
full RAGAs evaluation sweeps), the standing policy is to flag the estimate
to the maintainer for a Colab/Kaggle offload rather than either running it
for hours on CPU or silently buying/provisioning GPU infrastructure.

**Consequences.**

- *Pros:* nobody discovers the latency gap by surprise in a demo — it's a
  known, recorded fact with an owner (the Phase 3 pass) and a next action.
  Phase 0's actual exit criterion ("fresh clone ingests and answers a
  query end to end") doesn't require the SLO to be met, so the project
  correctly doesn't block basic functionality on a performance target that
  needs a different kind of infrastructure decision entirely.
- *Cons:* every interactive session on this host is genuinely slow (1–2
  minutes for a full answer) — that's a real UX cost paid every single
  time the system is used for local development, not just a number in a
  spec. The eventual fix (a GPU box, or a hosted-LLM fallback) is real
  infrastructure spend or a new external dependency, not a code change.

**Alternatives considered.**

- *Silently drop or don't measure NFR-2* — rejected: SRS.md is explicit that
  "the codebase has never been benchmarked, which is itself a gap" —
  measuring and recording an unmet target is more honest and more useful
  than omitting the target to avoid an uncomfortable number.
- *Provision a GPU now to hit the SLO in Phase 0* — rejected as
  disproportionate: Phase 0's job is proving the pipeline works at all;
  buying/renting GPU infrastructure to hit a latency target before the
  pipeline is even correct would be optimizing a system that doesn't fully
  exist yet.
- *Route every inference call through Colab/Kaggle transparently* — rejected
  for interactive serving specifically: those are batch sandboxes with
  session limits, explicitly noted as unsuitable for hosting a live query
  path (only for offline, batchable work like eval sweeps or bulk
  re-embedding).

**Lesson.** An NFR you can't currently meet is more valuable recorded and
missed than silently dropped from the spec — "we know NFR-2 fails and here's
why, and here's what would fix it" is a position you can make a real
decision from later (revise the target, or invest in infrastructure);
"NFR-2 was never mentioned again" hides a decision that got made by default,
for nobody, by omission. The Colab/Kaggle-offload framing is itself a
useful pattern: distinguish between latency you need in the live serving
path (which sandboxes with session limits can't solve) and throughput you
need in a batch job (which they're a legitimately cheap answer to) rather
than treating "we need a GPU" as one undifferentiated need.

---

## ADR-017: Phase exits as release points, not calendar dates

**Context.** The project needed a versioning/release scheme, but is
single-maintainer, scattered-session work — a calendar-based release
cadence ("ship every two weeks") doesn't map to a workflow where sessions
happen whenever time is available, sometimes days apart, sometimes not.

**Decision.** Tie git tags and package versions to phase exit criteria, not
dates: Phase 0 exit → `v0.1`; Phase 1 → `v0.2`; Phase 2 → `v0.3`; Phase 3
(full SRS scope complete) → `v1.0`. The package version carries a `.dev0`
suffix between tags. The final story of each phase is responsible for the
version bump, commit, and tag.

**Consequences.**

- *Pros:* a version number now means something falsifiable — `v0.1` is not
  "whatever existed on some Tuesday," it's "the state where a fresh clone
  ingests and answers a query end to end" (Phase 0's exit criterion,
  verified by S0-6's actual transcript, not assumed). Anyone can look at a
  tag and know exactly which SRS-defined capability set it represents.
- *Cons:* release cadence is now unpredictable from the outside — there's no
  answer to "when's the next release" independent of "when is Phase N's
  exit criterion met," which is fine for a single-maintainer project but
  wouldn't satisfy a team/customer expecting calendar predictability.

**Alternatives considered.**

- *Calendar-based releases (weekly/monthly tags regardless of state)* —
  rejected: on scattered, part-time sessions this produces meaningless
  version numbers (a tag landing mid-way through a broken migration, e.g.
  the current state where `rag_qa.chain` is expected to fail import) that
  convey no information about what actually works.
- *No versioning at all until "done"* — rejected: intermediate proof points
  (Phase 0's "it runs at all," Phase 1's "CI would have caught every
  Phase-0 bug") are genuinely meaningful milestones worth being able to
  reference and roll back to individually.

**Lesson.** Version numbers are a communication tool, not a formality — the
question worth asking before adopting any release cadence is "what does
crossing this boundary actually *mean* to someone reading the tag later,"
and tying it to a testable exit criterion (SRS §12's phase definitions,
verified by an actual command output, not a feeling) keeps that meaning
honest instead of decorative.

---

## ADR-018: Model routing for AI-assisted implementation

**Context.** This project is implemented with Claude Code across many
scattered sessions, spanning both mechanical work (moving files, wiring
`pyproject.toml` entries) and judgment-heavy work (choosing LangChain 1.x
over legacy, designing the LCEL chain contract). Using one model
uniformly for both wastes either speed (an expensive, careful model on a
file move) or quality (a fast model on an irreversible library-migration
choice).

**Decision.** Route by task type, recorded per-story in each story file's
`Model` field (the story file is authoritative, per CLAUDE.md): plan-mode
architecture passes and phase sharding use a stronger, deliberate model;
mechanical stories (repo hygiene, dependency wiring, file moves — S0-1,
S0-2, S0-3, S0-6) use a faster mode; stories touching new API surface
(the LangChain 1.x migration itself, config `_target_` updates — S0-4,
S0-5) or any library/design choice use the deliberate model regardless of
how the story otherwise looks; and any story escalates to the deliberate
model after the obvious fix has failed twice.

**Consequences.**

- *Pros:* fast iteration on the 80% of stories that are genuinely
  mechanical (there's no judgment call in `git mv v2/main2.py
  src/rag_qa/cli.py`), reserved deliberation for the decisions that are
  actually hard to reverse (a wrong LangChain version choice costs a whole
  re-migration; a wrong file location costs a `git mv`). The "escalate
  after two failed attempts" rule catches the case where a story looked
  mechanical but wasn't — a cheap, mechanical safety net for a
  classification that can't be perfect upfront.
- *Cons:* misclassifying a story costs real rework — if a story marked
  mechanical turns out to hinge on an undocumented API-surface subtlety,
  the fast pass may produce a plausible-looking but subtly wrong result
  that a review has to catch instead of the model itself flagging
  uncertainty.

**Alternatives considered.**

- *Use the same (fast) model for everything* — rejected: the LangChain 1.x
  migration and library-choice decisions (ADR-006, ADR-013) are exactly the
  kind of judgment call where getting it right the first time avoids a
  second migration later; speed isn't the bottleneck there, correctness is.
- *Use the same (deliberate) model for everything* — rejected as wasteful
  for the genuinely mechanical majority of stories, where the deliberate
  model's extra care buys nothing a fast pass with a clear spec (the story
  file's Scope section) wouldn't already produce correctly.

**Lesson.** "Which tool/model/reviewer level does this task need" is a
triage decision worth making explicit and revisiting per-task, the same way
a human engineering team routes a one-line config change through a lighter
review process than a database migration. The generalizable tell for
"this needs the more careful pass" isn't "is this a big diff" — it's "is
this decision expensive or impossible to reverse if wrong" (a library
version choice, a schema choice) versus "is this decision cheap to redo if
wrong" (a file's location, a variable rename).

---

## How to extend this file

When a new decision of this weight gets made — choosing a technology,
reversing an earlier decision, accepting a tradeoff you want a future
session to know was deliberate — add a new `ADR-0NN` entry at the end (do
not renumber existing ones) and add it to the index table. If a later
decision **replaces** an earlier one, mark the old entry's status as
`Superseded by ADR-0NN` rather than deleting it — the point of this file is
that a wrong-in-hindsight decision, with its original reasoning intact, is
more useful to future-you than a clean history with no trace it was ever
considered.
