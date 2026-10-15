# S3-4: Per-loader splitters and a maximum file size

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | SRS §7.2 (chunking configurable per document type); SRS §7.1 (a declared maximum file size); backlog: per-loader splitter strategies |
| **Depends on** | S3-2 (manifest version 2) |
| **Model** | opus (not fast: it changes identity rules that tier 2's fingerprint reuses) |
| **Plan-first** | no |
| **Branch** | `feat/s3-4-ingestion-config`. Moderate risk: an identity rule wrong in either direction either re-embeds for nothing or keeps stale chunks. |

## Goal

Two SRS targets that need configuration, not architecture.

- **§7.2:** a document type can have its own chunking.
- **§7.1:** the corpus can declare how large a file may be.

Both default to today's behaviour, so the real index, tier 2's fingerprint and every answer
stay as they are. The identity rules follow ADR-026: an identity holds exactly what
determines the output. Design: ARCHITECTURE.md §3.3; DEC-17's identities.

## Scope

- **Per-extension splitters.**
  - `pipeline.ingestion.splitters`, a map from extension to a splitter reference (for
    example `".md": components.splitters.markdown`), overrides `pipeline.ingestion.splitter`
    for those files. It is empty by default.
  - **Schema rules:** keys follow `loaders`' (a lowercase suffix with one dot), each key must
    also be in `loaders`, and each reference must resolve.
  - **The manifest** records each document's effective splitter identity
    (`DocumentRecord.splitter`, made room for in version 2).
  - **What re-embeds:** a document whose effective splitter changed is re-embedded, as a
    changed loader identity is (DEC-17). A change to the default splitter changes every
    document that uses it.
  - **One entry for use:** `components.splitters.markdown`
    (`langchain_text_splitters.MarkdownTextSplitter`, 1,000 / 150), referenced by nothing
    by default, as the GPU embedders are (keep the alternatives as config axes).
- **A maximum file size.** `pipeline.ingestion.max_file_mb`, a positive number, unset by
  default.
  - A file above it is skipped and reported, the way an unmapped suffix is, with its size
    and the limit in the line and in `IngestReport`. The API job's report carries the same.
  - The walk is shared, so tier 2's `corpus` part leaves a skipped file out, exactly as
    ingest does.
  - A file that was indexed and later exceeds a lowered limit is removed, like a deleted
    file. README says so.
- **The fingerprint stays put.** `ingestion` hashes the per-extension map only when it is
  not empty. Unit tests hold the real config's six parts unchanged with the new keys unset,
  and show that each one moves `ingestion` or `corpus` once it is set.
- **Docs:** README's Configuration section shows both keys; `config.yaml` comments them.

## Out of scope

- Sparse vectors and their identity (S3-5).
- Using the markdown splitter for this corpus: it has no Markdown files, and any change
  that moves a hashed input waits for S3-11.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                 # 0: the real config's parts unchanged
uv run rag-ingest                                     # "Nothing changed: the index is up to date."
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

# a scratch corpus with a .md file (caveat 9, the S3-2 way: a scratch collection)
export RAG_DATA_PATH=$(mktemp -d) RAG_QDRANT_COLLECTION=s3_4_scratch
# ... a .pdf and a .md in it; a scratch config with ".md": components.splitters.markdown
uv run rag-ingest --config <scratch config>           # the manifest records the md splitter
# change the md splitter's chunk_size, ingest again: only the .md document is re-embedded
# set max_file_mb below the PDF's size: it is skipped and reported, and its chunks removed
unset RAG_DATA_PATH RAG_QDRANT_COLLECTION
gh pr checks
```

## Review notes for the human

- Two rules carry the risk: the per-document splitter identity in `diff`, and the
  fingerprint leaving the empty map out. Check both against ADR-026's rule: nothing more
  and nothing less than what determines the chunks.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
