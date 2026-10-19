# RAG_QA_System — Software Requirements Specification

| | |
| --- | --- |
| **Document ID** | SRS-RAGQA-001 |
| **Version** | 1.1 |
| **Status** | Draft for review |
| **Date** | 2026-08-01 |
| **Baseline** | git `main` @ `9b274d0` |

_Revision 1.1 (2026-08-01): phase exits in §12 mapped to product release tags (v0.1 → v1.0) per maintainer decision._

> A polished, styled version of this document (with architecture diagrams) is available as a published artifact. This file is the git-tracked plain-text reference.

---

## 1. Introduction

### 1.1 Purpose

This document specifies the current state, target requirements, and target architecture for **RAG_QA_System** — a locally-hosted retrieval-augmented generation (RAG) question-answering engine built on LangChain, FAISS, and Ollama. It is written against a full read of the repository as checked out: two prototype generations (`v1/`, a linear script; `v2/`, a config-driven "pipeline builder"), a stale duplicate (`temp/v2/`), and supporting assets (`docs/`, `vectorstore/`).

### 1.2 Scope

In scope: document ingestion/chunking, embedding/indexing, retrieval and reranking, grounded answer generation, evaluation, configuration management, and the operational surface (deployment, observability, security) needed to run this as a service.

Out of scope for this revision: a web front-end, multi-modal ingestion, LLM fine-tuning, and multi-tenant SaaS concerns — noted as future extension points in §12.

### 1.3 Definitions & Acronyms

| Term | Definition |
| --- | --- |
| RAG | Retrieval-Augmented Generation |
| Chunk | A splitter-produced fragment of a source document |
| Embedding | Dense vector representation of text for similarity search |
| ANN | Approximate Nearest Neighbor search |
| Reranker | Cross-encoder model that re-scores retrieved candidates |
| RRF | Reciprocal Rank Fusion — merges ranked lists from multiple retrievers |
| LCEL | LangChain Expression Language — composable pipe interface superseding legacy Chain classes |
| SSE | Server-Sent Events — standard transport for token streaming |
| RBAC | Role-Based Access Control |
| OTel | OpenTelemetry — vendor-neutral standard for traces/metrics/logs, with a Gen AI semantic convention |
| RAGAs | Open-source RAG evaluation framework (faithfulness, answer relevancy, context precision/recall) |
| `_target_` | This codebase's config key naming the fully-qualified class to instantiate |

### 1.4 References & Standards Consulted

- OWASP Top 10 for LLM Applications (2025) — structures §9.
- OpenTelemetry Semantic Conventions for Generative AI — structures §11.
- LangChain migration guides (LCEL, `langchain_ollama`, retriever/reranker interfaces).
- RAGAs documentation.
- The repository itself: `v1/main.py`, `v1/config.py`, `v1/docx_processor.py`, `v1/requirements.txt`, `v2/main2.py`, `v2/pipeline_builder.py`, `v2/file_processor.py`, `v2/evaluate.py`, `v2/check_config.py`, `v2/config.yaml`, `docs/dataset.py`, and current git status.

### 1.5 Severity & Status Taxonomy

| Severity | Meaning |
| --- | --- |
| Critical | System does not run, or trivial exploit yields code execution / data loss |
| High | Will cause an incident or block adoption in any multi-user/production setting |
| Medium | Should fix before scaling usage or team size; not an immediate blocker |
| Low | Quality-of-implementation issue; fix opportunistically |
| Nitpick | Style/consistency only |

| Status | Meaning |
| --- | --- |
| Met | Implemented and correct as written |
| Partial | Implemented but incomplete, unvalidated, or broken in a specific way |
| Not met | Does not exist in the current code |

---

## 2. Overall Description

### 2.1 Product Perspective

RAG_QA_System is not one product today — it's three overlapping attempts at the same product in one repo:

- **v1** — a linear script (`main.py` + `config.py` + `docx_processor.py`) with hardcoded model choice and paths.
- **v2** — a config-driven redesign (`pipeline_builder.py`) instantiating components from a YAML "component library" via a `_target_` dotted-path convention — the right architectural instinct, currently broken by two config key typos (§3, ISS-01).
- **temp/v2** — an untracked, stale copy of v2 that has silently diverged from the live version.

Target: a single versioned service exposing ingestion and query as first-class operations, config-validated at load, with the three prototype trees collapsed into one package.

### 2.2 Product Functions

| Function | Current (v2) | Target |
| --- | --- | --- |
| Document ingestion | Full-corpus rebuild, top-level directory only | Incremental, recursive, hash-tracked |
| Chunking | Fixed recursive character splitter | Configurable strategy by document type |
| Embedding & indexing | HuggingFace MiniLM → FAISS on local disk | Pluggable embedder → vector store with metadata filtering |
| Retrieval | Dense vector top-k only | Hybrid (lexical + dense) with RRF |
| Reranking | Configured but non-functional | Cross-encoder reranking, working end-to-end |
| Generation | Legacy `RetrievalQA`, blocking, no citations surfaced | LCEL chain, streaming, cited sources |
| Access | Interactive terminal REPL only | REPL + REST/streaming API |
| Evaluation | Script exists, dataset missing, no gate | RAGAs regression suite gating CI |

### 2.3 User Classes and Characteristics

**Today:** a single developer operating from a terminal on the machine that built the index. **Target:** (a) API consumers — internal apps calling the query endpoint; (b) ML/pipeline engineers tuning retrieval/reranking/prompt config; (c) operators deploying/monitoring. Neither (a) nor (c) is servable by the current CLI-only shape.

### 2.4 Operating Environment

Target: Linux containers, Python 3.11+, a locally-hosted Ollama instance as default LLM runtime (with a hosted-provider fallback abstraction), single-node to start with a documented path to multi-node retrieval serving.

### 2.5 Design & Implementation Constraints

- Self-hosted/local-first by inference from the existing Ollama dependency.
- Open-source component stack (LangChain, FAISS/Qdrant, HuggingFace models).
- Single active maintainer today — requirements are scoped so CI enforcement carries the quality bar.

### 2.6 Assumptions and Dependencies

- An Ollama daemon with the configured model (`qwen2:7b`) is reachable at query time; no fallback exists if not.
- The document corpus is trusted content — no adversarial-document threat model assumed for v1 scope, though prompt-injection-via-retrieved-content is addressed in §9.
- No multi-tenancy or per-document ACL required yet; §8's API is designed so this can be added without a rewrite.

---

## 3. Current State Findings

> Condensed from a full-repository read. Each finding has a stable ID (`ISS-##`) carried through to Appendix B. Full table with fixes in Appendix A.

**Headline:** the v2 pipeline does not start. `v2/config.yaml` defines the LLM component library under the key `llmS` (capital S, line 48) while `v2/pipeline_builder.py` resolves it via the query pipeline's reference to `components.llms` (line 95) — a plain `KeyError` on the first run of `main2.py`. The existence of `check_config.py` as a hand-written diagnostic script is itself evidence this has already cost real debugging time.

### 3.1 What's structurally right

The `_target_` / `build_object` pattern in `pipeline_builder.py` (a recursive object-graph builder driven by dotted import paths in YAML) is a legitimate, well-known design — a home-grown Hydra. Separating a reusable "component library" from a per-pipeline "assembly" section is the correct instinct and worth preserving through the rewrite.

### 3.2 What blocks everything else

| ID | Severity | Finding |
| --- | --- | --- |
| ISS-01 | Critical | Config key typo (`llmS` vs `llms`) — system fails to start |
| ISS-02 | Critical | Both v1 and v2 hardcode absolute paths that don't match this checkout |
| ISS-04 | High | `_target_` resolution has no allowlist — config is effectively arbitrary code execution |
| ISS-07 | High | Zero automated tests — nothing would have caught ISS-01 before a human did |
| ISS-09 | High | No `.gitignore`; `__pycache__`, FAISS binaries, a 341 KB screen recording are committed |

The remaining findings (ISS-03, ISS-05, ISS-06, ISS-08, and thirteen more) are catalogued in full in Appendix A.

---

## 4. Functional Requirements

Stated as **SHALL** (mandatory) or **SHOULD** (strongly recommended, deferrable).

**FR-1 — Declarative pipeline assembly** · `Partial`
The system SHALL assemble ingestion and query pipelines from a single declarative config file, with every stage swappable without code changes.
_Current:_ the `_target_` pattern works for the happy path, but two of its own referenced keys are misspelled (ISS-01), and it performs no schema validation.

**FR-2 — Ingestion** · `Partial`
The system SHALL recursively discover documents, chunk, embed, and persist to a vector index. SHOULD support incremental ingestion without a full rebuild.
_Current:_ non-recursive `os.listdir` (ISS-13); always full rebuild, no manifest (ISS-14).

**FR-3 — Retrieval** · `Partial`
SHALL retrieve top-k via dense similarity. SHOULD combine with lexical (BM25) retrieval fused via RRF, and support metadata filtering.
_Current:_ dense-only, no filtering, no hybrid path even as a config option.

**FR-4 — Reranking** · `Not met`
Where enabled, SHALL rerank candidates with a cross-encoder before generation.
_Current:_ config targets a nonexistent class name and passes a string where an object is expected (ISS-03) — never executed successfully.

**FR-5 — Grounded generation** · `Partial`
SHALL constrain answers to retrieved context and decline when insufficient. SHOULD return source chunks attributed to their document.
_Current:_ prompt enforces grounding correctly; `return_source_documents=True` is set, but `main2.py` never reads or prints `result["source_documents"]` — citations are computed and discarded.

**FR-6 — Access surface** · `Partial`
SHALL provide an interactive interface (retained for local dev) and a programmatic API (REST, streaming).
_Current:_ REPL only, synchronous, no external callability.

**FR-7 — Evaluation** · `Partial`
SHALL support automated, repeatable quality evaluation against a versioned golden question set, runnable in CI.
_Current:_ `evaluate.py` correctly wires RAGAs, but reads `eval_dataset.jsonl` which does not exist, and RAGAs' default judge is OpenAI — unconfigured and undocumented.

**FR-8 — Config validation** · `Not met`
SHALL validate the full config graph at load time and fail with a specific, actionable error before building any component.
_Current:_ raw dictionary indexing with no validation; `check_config.py` is a manual, separate script that doesn't even check the keys that are actually broken.

---

## 5. Non-Functional Requirements

> Targets below are proposed SLOs to design against and validate once instrumentation exists — the codebase has never been benchmarked, which is itself a gap.

### 5.1 Performance

- **NFR-1:** Retrieval (top-k=10, corpus ≤100k chunks) SHALL return in <300ms at p95.
- **NFR-2:** Time-to-first-token SHALL be <2s at p95 with a local 7B-class model.
- **NFR-3:** Ingestion throughput SHALL be documented for capacity planning.

### 5.2 Scalability

- **NFR-4:** Vector store SHALL support incremental upsert without full-index rebuild.
- **NFR-5:** Query path SHALL be stateless and horizontally scalable behind a load balancer.

### 5.3 Reliability

- **NFR-6:** LLM/embedding calls SHALL have timeout, retry-with-backoff, circuit-breaking.
- **NFR-7:** Ingestion failures on individual documents SHALL NOT abort the run; run SHALL end with a pass/fail summary and non-zero exit on any failure.

### 5.4 Security

See §9 in full.

### 5.5 Maintainability

- **NFR-8:** Core pipeline modules SHALL carry ≥80% unit test coverage, enforced in CI.
- **NFR-9:** Codebase SHALL pass ruff + mypy/pyright in CI with zero errors.

### 5.6 Observability

- **NFR-10:** Every query/ingestion run SHALL emit structured logs and OTel Gen AI-convention spans (retrieval, rerank, generation latency, token counts).

### 5.7 Portability

- **NFR-11:** No absolute filesystem paths in committed config; all paths relative to config file or env-overridden. Closes ISS-02.

---

## 6. Target System Architecture

The target keeps the config-driven component philosophy from v2 and wraps it in the layers a 2026-era production RAG service is expected to have.

### 6.1 Current-state data flow

```text
config.yaml (unvalidated, hardcoded paths)
        │
        ├──▶ file_processor.py (full-rebuild ingestion) ──▶ HuggingFace MiniLM embeddings ──▶ FAISS index.faiss/.pkl (local disk)
        │
        └──▶ main2.py (blocking REPL) ──▶ pipeline_builder.py (RetrievalQA, legacy chain) ──▶ [FAISS index] + [Ollama qwen2:7b]
```

All in a single process, single machine.

### 6.2 Target architecture

```text
API consumers / CLI
        │
        ▼
Auth + rate limiting
        │
        ▼
FastAPI service (async) ── /v1/query (SSE stream) · /v1/ingest · /v1/health
        │
        ▼
Orchestration (LCEL / LangGraph)
  Hybrid retriever (BM25 + dense, RRF) ──▶ Cross-encoder reranker ──▶ LLM gateway (Ollama primary, hosted fallback)
        │
        ▼
Data layer: Vector store (Qdrant/pgvector, metadata filtering + upsert) + Ingestion manifest (content-hash tracked)
        ▲
        │
Ingestion worker (incremental, recursive)

Platform (cross-cutting): Pydantic config + allowlisted _target_ registry · OpenTelemetry traces/metrics → Langfuse/Jaeger · RAGAs regression suite in CI
```

### 6.3 Layer-by-layer targets

**Config & component registry** — Keep the `_target_` pattern, but load it through a Pydantic v2 `BaseSettings` schema so every field is typed and validated before any component is built, and restrict `import_from_string` to an explicit allowlist of module prefixes (`langchain_community.`, `langchain_huggingface.`, the project's own package). Closes ISS-04.

**Ingestion** — Recursive corpus walk; content-hash manifest (path → hash → chunk IDs) so re-running only re-embeds changed files; failures collected into a summary report.

**Retrieval & reranking** — Move from FAISS-on-disk-pickle to a store supporting upsert and metadata filtering natively (Qdrant or pgvector). Add BM25 alongside dense, fused with RRF. Fix the reranker to target the real `CrossEncoderReranker` class with a nested `HuggingFaceCrossEncoder` object — the recursive `build_object` already supports this nesting, it's a config bug, not an architecture gap.

**Generation** — Replace deprecated `langchain_community.llms.Ollama` with `langchain_ollama.ChatOllama`, and the legacy `RetrievalQA` chain with an LCEL `create_retrieval_chain` pipeline supporting token streaming and structured citation output.

**API layer** — A FastAPI service, async throughout, streaming over SSE, auto-generated OpenAPI schema, versioned under `/v1`. Full shape in §8.

**Observability** — Structured JSON logs (structlog) plus OpenTelemetry spans following Gen AI semantic conventions (`gen_ai.*` attributes), exported to a trace backend (Langfuse, Phoenix, or Jaeger). Metrics scraped by Prometheus with alerting on SLO burn.

**Evaluation** — RAGAs (or DeepEval) against a versioned golden dataset committed to the repo, with a regression threshold failing CI on drop below baseline.

---

## 7. Data Requirements

### 7.1 Source corpus

Supported formats: PDF, DOCX, TXT, Markdown (already configured via the loader library). Target adds recursive directory traversal, a declared max file size, and explicit skip-and-report for unsupported types.

### 7.2 Chunking

Current: fixed `RecursiveCharacterTextSplitter`, 1000 chars / 150 overlap, uniform across document types. Target: chunking strategy configurable per loader, wired through the existing `_target_` mechanism — no architecture change needed, just more splitter entries and a per-loader default.

### 7.3 Metadata schema

Every chunk SHALL carry: source file path, page/section number (where applicable), content hash of the source file at ingestion time, ingestion timestamp, loader used.

### 7.4 Evaluation dataset schema

A committed `eval_dataset.jsonl` (currently absent) with one record per line: `question`, `ground_truth`, optionally `expected_sources`. Versioned alongside code for reproducible evaluation.

---

## 8. Interface Requirements

### 8.1 REST API (target)

| Method & path | Purpose |
| --- | --- |
| `POST /v1/query` | Submit a question; streams tokens over SSE, final event includes cited source chunks |
| `POST /v1/ingest` | Trigger (incremental) ingestion; returns a job ID for async status polling |
| `GET /v1/ingest/{job_id}` | Ingestion job status and per-document pass/fail summary |
| `GET /v1/documents` | List indexed documents with metadata |
| `GET /v1/health` | Liveness/readiness — checks vector store and LLM reachability |

### 8.2 CLI

Retained for local development as a thin client over the same internal pipeline objects used by the API — not a separately-maintained code path. Closes the duplication between `v1/main.py`'s query loop and `v2/main2.py`'s near-identical copy.

### 8.3 Configuration schema

A single `config.yaml` (or split base + environment overlay) validated against a Pydantic schema at process start, with a machine-specific `config.local.yaml` pattern (gitignored) for paths and secrets, replacing hardcoded absolute paths currently committed to source control.

---

## 9. Security Requirements

Mapped to the OWASP Top 10 for LLM Applications where applicable, plus standard application-security requirements this codebase currently misses.

| Ref | Requirement | Current gap |
| --- | --- | --- |
| LLM01 | Prompt injection: retrieved content SHALL be treated as untrusted; system prompt SHALL be structurally separated from context | No isolation beyond prompt wording; acceptable for a trusted corpus today, insufficient once ingestion accepts external documents |
| LLM05 | Supply chain: dependencies SHALL be pinned and scanned (pip-audit) in CI | v1 pins commented out, lists conflicting `faiss-cpu`+`faiss-gpu`; v2 has no requirements file at all (ISS-08) |
| LLM07/08 | Insecure design / excessive agency: dynamic code-loading from config SHALL be allowlisted | `import_from_string` imports and instantiates any dotted path with any kwargs — config is effectively `eval` (ISS-04) |
| — | Deserialization SHALL only occur on artifacts the system itself produced | `allow_dangerous_deserialization=True` set on every FAISS load with no invariant documented or enforced (ISS-16) |
| — | No secrets in source control | None found in this repo today — a genuine positive to preserve |
| — | Errors SHALL be handled explicitly, not swallowed | Bare `except Exception` in ingestion (ISS-05); no error handling around the query-time LLM call (ISS-06) |

---

## 10. Testing & QA Requirements

- **Unit:** `import_from_string` (valid path, bad module, bad attribute), `build_object` (nested `_target_`, lists, plain dicts), config schema validation against a valid and a deliberately broken config.
- **Integration:** ingestion against 2–3 fixture documents with a deterministic fake embedder, asserting chunk counts and index round-trip; a config-resolution test loading the real `config.yaml` and resolving every referenced path — this single test would have caught ISS-01, the reranker class mismatch, and the dead `vector_stores` block before any shipped.
- **Evaluation-as-test:** RAGAs run in CI against the committed golden dataset, failing the build on regression past a set threshold.
- **Coverage gate:** ≥80% on `pipeline_builder.py`, `file_processor.py`, and the config-validation module, enforced in CI.

---

## 11. DevOps & Deployment Requirements

- **CI:** lint (ruff), type-check (mypy/pyright), unit + integration tests, dependency audit (pip-audit), container build + vulnerability scan (Trivy), evaluation gate — one pipeline, blocking merge on any red stage.
- **Containerization:** multi-stage `Dockerfile`; `docker-compose.yml` wiring app + Ollama + vector store for local/dev; readiness/liveness probes backing `/v1/health`.
- **Config management:** environment-specific overlays, no absolute/machine-specific paths committed (closes ISS-02).
- **Resilience:** retry-with-backoff and circuit-breaking around embedding and LLM calls; graceful handling of an interrupted ingestion run so the vector store is never left half-written undetected.
- **Backup/DR:** scheduled snapshot of vector store + manifest to object storage, documented restore runbook, stated RPO/RTO once beyond a local prototype.
- **Repository hygiene:** `.gitignore` covering `__pycache__/`, vector store binaries, editor artifacts; tracked screen recording, FAISS index, and `.py.save` backup removed from history (ISS-09).

---

## 12. Migration Roadmap

Sequenced by dependency, not calendar time — each phase's exit criteria gate the next. **Each phase exit is a release point**, tagged in git and reflected in the package version: the repaired, runnable system is **v0.1**; the completed SRS scope is **v1.0**.

**Phase 0 — Make it run, make it honest** *(exit → tag `v0.1`)*
Fix ISS-01 (config keys) and ISS-02 (hardcoded paths); collapse `v1/`, `v2/`, `temp/v2/` into one source tree; add `.gitignore` and purge tracked binaries.
_Exit:_ a fresh clone runs ingestion and a query end to end.

**Phase 1 — Make it trustworthy**
Pydantic config validation replacing `check_config.py`; allowlisted `_target_` resolution; unit + config-resolution tests in CI; pinned dependencies in one `pyproject.toml`; error handling around ingestion and the query loop.
_Exit:_ CI is green and would have caught every Phase-0 bug.

**Phase 2 — Make it a service**
FastAPI wrapper with streaming; fix and exercise the reranker; incremental ingestion with a hash manifest; commit `eval_dataset.jsonl` and wire RAGAs into CI as a gate.
_Exit:_ the system is callable over HTTP and has an enforced quality bar.

**Phase 3 — Make it production-grade**
Migrate FAISS-on-disk to Qdrant/pgvector; add hybrid retrieval; OpenTelemetry tracing + metrics + alerting; containerize and add CI vulnerability scanning; document backup/restore.
_Exit:_ the gaps in §9 and §11 are closed.

---

## Appendix A — Full Issue Log

21 findings · 2 critical · 6 high · 8 medium · 4 low · 1 nitpick

| ID | Severity | Location | Finding | Fix |
| --- | --- | --- | --- | --- |
| ISS-01 | Critical | `v2/config.yaml:48,95` | Component key defined as `llmS`, referenced as `llms` — `KeyError` on startup | Rename the key; add config-resolution test |
| ISS-02 | Critical | `v2/config.yaml:110-111`, `v1/config.py:4-5` | Hardcoded absolute paths to another user's home directory; unrunnable on this checkout | Paths relative to config file location, override via env var |
| ISS-03 | High | `v2/config.yaml:62-67` | Reranker targets `CrossEncoderRerank` (doesn't exist) with a bare model string instead of an object | Target `CrossEncoderReranker`; nest a `HuggingFaceCrossEncoder` `_target_` under `model` |
| ISS-04 | High | `v2/pipeline_builder.py:11-20` | `import_from_string` imports and instantiates any dotted path with any kwargs from config | Allowlist of importable module prefixes |
| ISS-05 | High | `v2/file_processor.py:33-34` | Bare `except Exception` in ingestion; corrupt file silently drops from corpus | Collect failures, print summary, exit non-zero if any document failed |
| ISS-06 | High | `v2/main2.py:22` | No error handling around `qa_chain.invoke`; one Ollama hiccup crashes the REPL | Wrap in try/except, print error, continue loop |
| ISS-07 | High | repository-wide | Zero automated tests | Unit + integration suite, enforced in CI |
| ISS-08 | High | `v1/requirements.txt`, v2 (missing) | v1 pins commented out, conflicting `faiss-cpu`+`faiss-gpu`; v2 declares no dependencies at all | Single pinned `pyproject.toml`; pip-audit in CI |
| ISS-09 | High | git tree | No `.gitignore`; `__pycache__/*.pyc`, FAISS binaries, a 341 KB screencast tracked | Add `.gitignore`; `git rm --cached` the offenders |
| ISS-10 | Medium | `v1/`, `v2/`, `temp/v2/` | Three parallel copies of the system instead of git history/branches | Collapse to one tree; delete `temp/`, retire `v1/` |
| ISS-11 | Medium | `v2/config.yaml:42-46,84` | `components.vector_stores` never read; path duplicated as top-level key | Delete dead block; read path through pipeline reference |
| ISS-12 | Medium | `v2/pipeline_builder.py:72-117` | ~45 lines of unreachable code (prior implementation pasted after `return`) | Delete; git history preserves it |
| ISS-13 | Medium | `v2/file_processor.py:19` | Non-recursive `os.listdir` — subdirectories silently ignored | Recursive traversal (`os.walk`/`Path.rglob`) |
| ISS-14 | Medium | `v2/file_processor.py` | Ingestion is full-rebuild-only; no dedup/incremental update | Content-hash manifest; upsert only changed files |
| ISS-15 | Medium | `v2/evaluate.py:15` | Reads `eval_dataset.jsonl`, which doesn't exist; RAGAs defaults to unconfigured OpenAI judge | Commit golden dataset; point RAGAs at local Ollama explicitly |
| ISS-16 | Medium | `v2/pipeline_builder.py:53`, `v1/main.py:36` | `allow_dangerous_deserialization=True` with the safety invariant undocumented | Document invariant; never load an index from an untrusted source |
| ISS-17 | Medium | `v2/pipeline_builder.py`, `config.yaml` | Deprecated APIs: `langchain_community.llms.Ollama`, legacy `RetrievalQA` chain | Migrate to `langchain_ollama.ChatOllama` and LCEL `create_retrieval_chain` |
| ISS-18 | Low | `docs/dataset.py:13` | Downloads a HuggingFace `/blob/` URL (returns HTML, not parquet); no timeout | Use `/resolve/` URL form; add `timeout=` |
| ISS-19 | Low | `v2/*.py` | No type hints anywhere in v2 | Annotate core functions; add mypy/pyright to CI |
| ISS-20 | Low | `README.md` | Two lines; no setup, no Ollama prerequisite, no workflow docs | Setup guide, prerequisites, `_target_` extension example |
| ISS-21 | Nitpick | `v2/main2.py:7`, misc. | Typos, hand-rolled ANSI escapes, duplicated REPL logic | Fix in passing; consolidate REPL into one shared CLI module |

## Appendix B — Requirements Traceability Matrix

| Requirement | Blocked by | Resolved by |
| --- | --- | --- |
| FR-1 | ISS-01, ISS-04 | Pydantic config schema + allowlisted registry (§6.3) |
| FR-2 | ISS-13, ISS-14 | Recursive walk + hash manifest (§6.3, §7.1) |
| FR-3 | — | Hybrid retriever, RRF (§6.2) |
| FR-4 | ISS-03 | Corrected reranker config (§6.3) |
| FR-5 | — | LCEL chain surfacing `source_documents` (§6.3) |
| FR-6 | — | FastAPI service (§8.1) |
| FR-7 | ISS-15 | Committed eval dataset + CI gate (§7.4, §10) |
| FR-8 | ISS-01, ISS-11 | Pydantic config schema (§6.3) |
| NFR-4/5 | — | Qdrant/pgvector migration (§6.3) |
| NFR-6/7 | ISS-05, ISS-06 | Retry/circuit-break + explicit error handling (§11) |
| NFR-8/9 | ISS-07, ISS-19 | CI: pytest, ruff, mypy (§10, §11) |
| NFR-10 | — | OpenTelemetry Gen AI spans (§6.3) |
| NFR-11 | ISS-02 | Relative/env-driven config paths (§8.3) |

---

_RAG_QA_System — Software Requirements Specification v1.0. Compiled from a full-repository engineering review, 2026-08-01. Severity ratings follow the taxonomy defined in §1.5._
