# S3-4: Per-loader splitters and a maximum file size

| | |
| --- | --- |
| **Status** | Implemented (2026-10-21), not yet committed: verification 1–7 are run and shown below, all as expected; 8, `gh pr checks`, waits for the PR. Stopped before committing, for the maintainer's approval. |
| **Closes** | SRS §7.2 (chunking configurable per document type); SRS §7.1 (a declared maximum file size); backlog: per-loader splitter strategies |
| **Depends on** | S3-2 (manifest version 2) |
| **Model** | opus (not fast: it changes identity rules that tier 2's fingerprint reuses) |
| **Plan-first** | no |
| **Branch** | `feat/s3-4-ingestion-config`. Moderate risk: an identity rule wrong in either direction either re-embeds for nothing or keeps stale chunks. |

## Goal

Two SRS targets that need configuration, not architecture.

- **§7.2:** a document type can have its own chunking.
- **§7.1:** the corpus can declare how large a file may be.

The splitter map defaults to today's behaviour, and the 50 MB limit skips no corpus file, so
the real index, tier 2's fingerprint and every answer stay as they are. The identity rules follow ADR-026: an identity holds exactly what
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
  - **Which record decides** (design review 1): the per-document splitter identities, which
    S3-2 writes as the default's. The manifest's top-level splitter record names the default
    and decides nothing, so DEC-17's full rebuild on a splitter change gives way to this.
  - **One entry for use:** `components.splitters.markdown`
    (`langchain_text_splitters.MarkdownTextSplitter`, 1,000 / 150), referenced by nothing
    by default, as the GPU embedders are (keep the alternatives as config axes).
- **A maximum file size.** `pipeline.ingestion.max_file_mb`, a positive number.
  `config.yaml` sets 50: a declared limit, as SRS §7.1 asks, above the largest corpus file
  (25.4 MB) (maintainer, design review 1). Unset means no limit.
  - A file above it is skipped and reported, the way an unmapped suffix is, with its size
    and the limit in the line and in `IngestReport`. The API job's report carries the same.
  - The walk is shared, so tier 2's `corpus` part leaves a skipped file out, exactly as
    ingest does.
  - A file that was indexed and later exceeds a lowered limit is removed, like a deleted
    file. README says so.
- **The fingerprint stays put.** `ingestion` hashes the per-extension map only when it is
  not empty. `max_file_mb` is never hashed: what it changes reaches `corpus` through the
  shared walk, so a limit that skips nothing moves nothing (ADR-026). Unit tests hold the real
  config's six parts unchanged with the 50 MB limit set, and show a non-empty map moving
  `ingestion` and a limit below a file's size moving `corpus`.
- **Docs:** README's Configuration section shows both keys; `config.yaml` comments them.

## Out of scope

- Sparse vectors and their identity (S3-5).
- Using the markdown splitter for this corpus: it has no Markdown files, and any change
  that moves a hashed input waits for S3-11b.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                 # 0: the real config's parts unchanged
uv run rag-ingest                                     # "Nothing changed: the index is up to date."
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-4/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-4/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

# a scratch corpus with a .md file (caveat 9, the S3-2 way: a scratch collection)
export RAG_DATA_PATH=$(mktemp -d) RAG_QDRANT_COLLECTION=s3_4_scratch
# ... a .pdf and a .md in it; a scratch config with ".md": components.splitters.markdown
uv run rag-ingest --config <scratch config>           # the manifest records the md splitter
# change the md splitter's chunk_size, ingest again: only the .md document is re-embedded
# set max_file_mb below the PDF's size: it is skipped and reported, and its chunks removed
# clean up: deleting the scratch generations removes their alias too
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_4_scratch--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}
unset RAG_DATA_PATH RAG_QDRANT_COLLECTION
gh pr checks
```

## Review notes for the human

- Two rules carry the risk: the per-document splitter identity in `diff`, and the
  fingerprint leaving out the empty map and `max_file_mb`. Check both against ADR-026's rule: nothing more
  and nothing less than what determines the chunks.
- **As built, against that rule:**
  - **`diff`** compares a document's bytes, loader identity and splitter identity: the
    three config-side inputs of its chunks. The embedder and `CHUNKING_VERSION`, the other
    two, still rebuild in full. The splitter identity is the effective one, the extension's
    own or the default, as `spec_identity` of its spec, so renaming an entry, or an entry
    with the default's spec, re-embeds nothing.
  - **The fingerprint** hashes only the entries whose spec differs from the default's
    (Deviation). Its one over-inclusion is kept on purpose: the default is hashed even when
    every extension has its own, which costs a needless re-run, never a stale pass, and
    keeps the part's shape. A test says so.
  - **`max_file_mb`** is never hashed; the walk leaves the file out, so `corpus` moves only
    when a file is skipped.
- **One record decides nothing, and can lag.** An edit of the default splitter that no
  document uses re-embeds nothing and publishes nothing, so the live manifest's top-level
  record still names the old default until the next publish. It is documented as "the
  default when this generation was published". Nothing reads it to decide since this
  story: `rag-ingest`'s full rebuild on it and `generate`'s staleness check on it are both
  gone.

## Verification record (2026-10-21)

Run on the working tree, on `main` at `9542c9b` plus this story's changes, with Qdrant up
and the live index under `rag_qa` (`rag_qa--gen-20261019T153834837765Z-7f20891e`).

1. **`rag-eval check`:** exit **0**; all six fingerprint parts ok, every floor met.
2. **`rag-ingest`:** "Nothing changed: the index is up to date.", "Documents: 3
   unchanged.", the two parquet files skipped for want of a loader, exit **0**. The
   per-document splitter identities S3-2 wrote are the ones the new rule computes, and the
   50 MB limit skips nothing.
3. **`rag-eval retrieval --contexts-out ~/dev/rag-s3-4/contexts.json`:** hit rate 0.950,
   MRR 0.796, recall 0.858, S3-2's numbers; every floor met, exit **0**.
4. **`diff`:** **IDENTICAL**. Both files hash to `5a36e2f9d00bbdaf…a074cbc`, S3-2's digest.
5. **The suite** (`HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1`): **1068 passed, 12 skipped, 5
   xfailed**, coverage **98.40%**; the HF folder stayed empty. The 12 skips are the 8
   `qdrant_server` tests and 4 ragas tests that need the `eval` extra. (S3-3 ended at 1020
   passed; this story adds 48.)
6. **Lint, format, types:** "All checks passed!", "60 files already formatted", "no issues
   found in 30 source files".
7. **The scratch drill** (`RAG_QDRANT_COLLECTION=s3_4_scratch`, a corpus of a 615,023-byte
   corpus PDF and a 2.6 KB Markdown file with three headed sections, a scratch config with
   `".md": components.splitters.markdown`):
   - **The first ingest** built 56 chunks, 52 from the PDF and 4 from the Markdown file. The
     manifest gives `notes.md` the markdown splitter's identity (`53e67a36…`) and
     `yolo.pdf` the default's (`86c72e0c…`), and its top level names the default. Every
     Markdown chunk starts at a heading. The third section, at 1,007 characters, is over
     `chunk_size`, so the splitter's next separator parts the heading from its body.
   - **The markdown splitter's `chunk_size` to 500:** "Updating the index: 1 changed, 1
     unchanged", "Embedding 10 chunks", no full rebuild. `yolo.pdf` kept its 52 chunks,
     splitter and `ingested_at`.
   - **`max_file_mb: 0.5`:** "Skipped yolo.pdf (0.615 MB, above the 0.5 MB limit)", "1
     removed, 1 unchanged", "Skipped 1 file(s) above the 0.5 MB limit (max_file_mb).",
     exit **0**. The new generation holds 10 points, none from `yolo.pdf`.
   - **Cleanup:** the story's `curl … | xargs` deleted the two scratch generations left
     (recovery had pruned the first), and the `s3_4_scratch` alias went with them. `rag_qa`
     still names `…7f20891e`, and its two generations are untouched.
8. **`gh pr checks`:** "no pull requests found for branch "main"", exit 1: there is no PR
   yet. To be run on the PR.

**The new guards were checked against breakage.** A script broke each one in turn in the
working tree, ran the tests meant to catch it, and restored the file, checked by sha256.
All 16 were caught:

- `diff` ignoring the splitter;
- the limit compared with `>=`, and as the float's binary value (0.000249 MB);
- the file hashed before its size is checked, and no size check at all;
- a splitter change rebuilding in full again;
- every document chunked by the default splitter, or recorded with the default's
  identity, or given it by the walk;
- the fingerprint hashing an override equal to the default, never hashing the map, or
  hashing the empty map as `{}`;
- `generate` checking the top-level splitter again;
- a splitter key with no loader, a lax `max_file_mb` (`true` read as 1), and the
  per-extension slot left out of the checked references.

The first pass caught 15. The survivor was a test of the Markdown chunks that the default
splitter also passed: its 150-character overlap starts the second chunk at "## Second"
too. The test now checks the first chunk's end. One "caught" was a syntax error the
mutation itself made; it was rewritten, and then failed the test it should.

## Discovered

- **`MarkdownTextSplitter` never splits at a heading as configured** (langchain-text-splitters
  1.1.2). Its separators are patterns (`"\n#{1,6} "`, `"\n---+\n"`, …), but it leaves
  `is_separator_regex` off, so they are escaped and matched literally. Apart from code
  fences it then splits as `RecursiveCharacterTextSplitter` does. The config entry sets
  `is_separator_regex: true`, and `tests/test_components.py` fails without it.
- **A float product misses a decimal limit.** 0.000249 × 10⁶ is 248.99999999999997, which
  would turn away the 249-byte file the limit admits. The limit is compared as the
  `Decimal` of what the config says, and a test holds the 249-byte boundary.
- **A rounded size can read as under the limit.** At three significant figures a
  12,345,001-byte file is "12.3 MB, above the 12.345 MB limit". The line adds digits until
  it reads as above ("12.35"), and never prints an exponent.
- **The fingerprint's relative tests cannot see a change to a part's shape.** Each compares
  two configs through the same code, so hashing the empty map as `{}` passed them all, and
  only `rag-eval check` would have failed, in CI. A unit test now pins the real config's
  `ingestion` part to the committed run's.
- **Backlog: "Published …: 1 documents"** (found in the drill). `_update`'s publish line
  does not agree in number for one document. It predates this story; S2-8 fixed the other
  lines.
- **For S3-11 or later, if the markdown splitter is used:** a section longer than
  `chunk_size` leaves its heading as a chunk of its own, 16 characters in the drill: the
  library's recursive fallback. This corpus has no Markdown files.

## Deviation from plan

- **The fingerprint hashes the overrides that differ from the default, not "the map when
  it is not empty".** An entry with the default's spec makes the default's chunks and
  re-embeds nothing in `rag-ingest`, so by ADR-026's rule it should move nothing either.
  An empty map is the case the story names, and it still moves nothing. ARCHITECTURE.md
  §3.3 has the dated note.
- **The markdown entry sets `is_separator_regex: true`**, beyond the story's "1,000 / 150"
  (Discovered). It is referenced by nothing, so no hash moves.
- **`rag-eval generate`'s staleness check dropped its top-level splitter comparison**, which
  the story does not name. It was the same rule as `rag-ingest`'s full rebuild, and would
  have refused an index `rag-ingest` calls up to date. The per-document records decide
  there too, through `diff`. Found by caveat 28's check of every reader of the identity.
- **The report's names:** `IngestReport.too_large`, as `(name, size in bytes)`, and
  `IngestReport.max_file_mb`, the limit the run applied. `IngestReportOut` mirrors them, so
  the API's job report carries both.
- **One more unit test than the story lists:** the real config's `ingestion` part against
  the committed run's (Discovered).
- **ARCHITECTURE.md** gains two dated notes, under DEC-17's full-rebuild list and §3.3.
- **The board's S3-3 record** is updated to merged, at the maintainer's request.
- **Model:** Opus 5.5, as the story says (opus, not fast).
- **The branch** `feat/s3-4-ingestion-config` is not created: the maintainer runs every git
  command, and the recipe starts with it.
