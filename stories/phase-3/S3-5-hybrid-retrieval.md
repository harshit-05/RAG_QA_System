# S3-5: Hybrid retrieval and source filters, measured by tier 1

| | |
| --- | --- |
| **Status** | In review (2026-10-22): built and measured on `feat/s3-5-hybrid-retrieval`, not committed. Verification 1–13 run and shown below, all as expected; the two HTTP checks (rag-serve in the maintainer's terminal) and `gh pr checks` wait for the maintainer. Tier 1 recommends hybrid on, with `rrf_k` 60 and `prefetch` 40, and `min_score` −4, for S3-11 ("Tier 1"). On Opus 5.5 at max effort (Deviation). |
| **Closes** | FR-3 (lexical retrieval fused by RRF; metadata filtering); DEC-20 |
| **Depends on** | S3-4 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-5-hybrid-retrieval`. Moderate risk: new retrieval code and identity rules. The real config does not change, so no answer can. |

## Goal

FR-3's two SHOULDs.

- **Hybrid retrieval.** BM25 sparse vectors beside the dense ones, fused by our own RRF,
  then reranked by the cross-encoder.
- **Metadata filtering.** A question can be limited to named documents.

Tier 1 measures hybrid retrieval against dense on the golden set, with S2-4's discipline:
question by question, on a scratch config and collection. The measurements become S3-11's
input. **The real config keeps dense retrieval and sets no sparse encoder**, so neither tier
2's `query` part nor its `ingestion` part moves, and the switch, if tier 1 earns it, rides
S3-11b's one re-run (DEC-26). Design: ARCHITECTURE.md §3.1, DEC-20; §3.3, §3.4.

## Scope

- **`rag_qa/sparse.py`: `Bm25Encoder`**, standard library only (an architecture test holds
  it to that).
  - **Tokens:** lowercase Unicode word tokens, minus a fixed English stopword list, with no
    stemming. Each token becomes a uint32 through blake2b (a 4-byte digest).
  - **Document weights:** `tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))`. **Query
    weights:** 1 per distinct token.
  - **Output:** sparse vectors as our own `(indices, values)` type; `vectorstore` converts
    them for Qdrant.
  - **Settings:** `k1` (1.2), `b` (0.75), and `avg_len`. Measure `avg_len` first, as the
    mean token count of the corpus's chunks, and fix it in `config.yaml`.
  - **Versioning:** `SPARSE_VERSION`, pinned by a test over the sample corpus as
    `CHUNKING_VERSION` is.
- **Schema.**
  - The new component kind `sparse_encoders`, and `pipeline.ingestion.sparse_encoder`
    (optional).
  - `RetrieverSpec` gains `mode` (`dense` when absent, or `hybrid`), `prefetch` and `rrf_k`.
    Unset, each stays out of `kwargs()`, so `vector_search`'s dump does not change.
  - `search_type` narrows to `similarity`.
  - A `hybrid` retriever requires an ingestion sparse encoder (a load error naming both).
- **Ingest and the store.**
  - With a sparse encoder set, the prepare phase also encodes each chunk.
  - **The generation** gains a sparse vector `bm25` with `modifier=IDF`, and the copy step
    carries sparse vectors over.
  - **The manifest** records the encoder's identity (its spec and `SPARSE_VERSION`). A
    changed identity, or one appearing or disappearing, rebuilds in full: the vector
    configuration changes.
- **The hybrid query** (DEC-20, as design review 2 changed it):
  - two `query_points` calls, dense (exact) and sparse, each with the filter and each cut to
    `prefetch` (40) by S3-2's cut rule. No cut is left to the server, which decides a tie by
    the collection's layout: review 1's whole-union fusion still let each prefetch cut its
    own list there;
  - fused in `vectorstore` by our own RRF: `1 / (rrf_k + rank)` summed over the lists a point
    is in, with one-based ranks and `rrf_k` always explicit (60). Qdrant's own fusion, and its
    default constant of 2 (caveat 31), are no longer used;
  - then ordered by score, ties by point ID, and cut to `reranker_candidates`.
- **The `sources` filter.**
  - **The retrieval half** takes a `RetrievalQuery(question, sources=())`; settle the exact
    type in the plan, with tier 1's call sites in mind.
  - **`stream_answer(pipeline, question, *, sources=())`.**
  - **HTTP:** `QueryRequest` gains an optional `sources`: at most 50 corpus-relative paths,
    each in the served manifest, else 422 naming the unknown ones. It stays
    `extra="forbid"`.
  - **CLI:** `rag-query --source PATH`, repeatable, checked against the manifest the same way
    at start-up.
  - **In Qdrant:** a `MatchAny` on the indexed `metadata.source`, in each branch's search
    (and as the query filter in dense mode).
- **The fingerprint.** `ingestion` includes the sparse encoder's identity only when one is
  set: an absent key is left out, so v0.3's hash holds. Unit tests show the real config's
  six parts unchanged, and the scratch config's `ingestion` and `query` parts moved.
- **Config.** Add `components.sparse_encoders.bm25` (with the measured `avg_len`) and
  `components.retrievers.hybrid`, both referenced by nothing in `pipeline`. Comment them
  with tier 1's table, as S2-4 commented the reranker.
- **Tier 1, measured and recorded here,** on a scratch config that differs from the real one
  only in `retriever: components.retrievers.hybrid` and `sparse_encoder`, ingested into a
  scratch collection:
  - dense against hybrid, both reranked to 5: hit rate, MRR, recall, and question by
    question (wins, losses, ties), as S2-4 did;
  - `rrf_k` 60 against one alternative, and `prefetch` 20 against 40;
  - the reach: how often an expected page is among the fused candidates before the rerank;
  - **the `min_score` table, for S3-11:** for cutoffs off, −4, −2, 0 and 2, on the
    configured retrieval, the answerable questions' hit rate and recall, and the chunks
    that survive on answerable and on unanswerable questions;
  - a recommendation: hybrid on or off, and a `min_score` value.
- **Server tests** (`qdrant_server`):
  - a hybrid query and a filtered query on the real server;
  - a fused list with a tie across its two lists, cut where the tie falls;
  - a tie at a branch's own cut, which must give the same points from two collections built
    in different insertion orders (design review 2). Within one process the server always
    agrees with itself, so two runs there prove nothing.
- **README:** the `sources` filter (HTTP and CLI) and what hybrid retrieval is, marked as
  measured but not yet on.

## Out of scope

- Switching the real config to hybrid, setting the sparse encoder there, or setting
  `min_score`: S3-11 chooses, and S3-11b commits the choice with the re-run.
- Stemming (a backlog line); weighted RRF (unless the table makes a case, recorded for
  S3-11).

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0: nothing hashed moved
uv run rag-ingest                                         # the real index: nothing changed
uv run rag-eval retrieval                                 # unchanged: 0.950 / 0.7958 / 0.8583
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-5/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-5/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q
uv run ruff check && uv run ruff format --check && uv run mypy

# tier 1, dense against hybrid, on a scratch collection (caveat 9)
RAG_QDRANT_COLLECTION=s3_5_hybrid uv run rag-ingest --config <scratch hybrid config>
RAG_QDRANT_COLLECTION=s3_5_hybrid uv run rag-eval retrieval --config <scratch hybrid config> \
  --contexts-out ~/dev/rag-s3-5/hybrid-1.json
docker compose restart qdrant   # the second run after a restart, where the server's own tie
                                # order changes (design review 2)
RAG_QDRANT_COLLECTION=s3_5_hybrid uv run rag-eval retrieval --config <scratch hybrid config> \
  --contexts-out ~/dev/rag-s3-5/hybrid-2.json
diff ~/dev/rag-s3-5/hybrid-1.json ~/dev/rag-s3-5/hybrid-2.json && echo STABLE   # the cut rule
# ... the variants and the min_score cutoffs; every table pasted here
# clean up: deleting the scratch generations removes their alias too
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_5_hybrid--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}

# the filter, over HTTP (rag-serve in a foreground terminal, logged with tee -i)
curl -sN -X POST 127.0.0.1:8000/v1/query -H 'content-type: application/json' \
  -d '{"question": "What is GLIDER?", "sources": ["2412.14140v2.pdf"]}' | head -3   # one file cited
curl -s -o /dev/null -w '%{http_code}\n' -X POST 127.0.0.1:8000/v1/query \
  -H 'content-type: application/json' -d '{"question": "x", "sources": ["nope.pdf"]}'  # 422
gh pr checks
```

## Verification record (2026-10-21/22)

Run on the working tree on `main`'s `54944d4`, before the branch's first commit. The
`<scratch hybrid config>` is `~/dev/rag-s3-5/configs/hybrid.yaml` ("Tier 1"), run with
`RAG_DATA_PATH=$PWD/corpus` and `--dataset eval/eval_dataset.jsonl --thresholds
eval/thresholds.yaml`, since a copied config anchors its paths to its own folder (S2-4).

| # | Command | Result |
| --- | --- | --- |
| 1 | `uv run rag-eval check` | exit 0: all six parts `ok`, every floor met (also after the config comments) |
| 2 | `uv run rag-ingest` | "Nothing changed: the index is up to date." 3 unchanged, exit 0 |
| 3 | `uv run rag-eval retrieval` | 0.950 (19 of 20) / 0.796 / 0.858, every floor met |
| 4 | `--contexts-out ~/dev/rag-s3-5/contexts.json`, `diff` | IDENTICAL; both `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc` |
| 5 | `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q` | 1180 passed, 15 skipped (the server tests), 5 xfailed; coverage 98.55% |
| 6 | `RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q` | 11 passed (8 earlier, 3 new) |
| 7 | `uv run ruff check && uv run ruff format --check && uv run mypy` | all checks passed; 62 files formatted; no issues in 31 source files |
| 8 | the scratch `rag-ingest` into `s3_5_hybrid` | 1,708 chunks published as `s3_5_hybrid--gen-20261021T191342911065Z-4227d960`, 2 min 24 s; its collection has `bm25` with `modifier: idf`, and its manifest the sparse record |
| 9 | hybrid tier 1, `--contexts-out hybrid-1.json` | 1.000 (20 of 20) / 0.833 / 0.933, exit 0 |
| 10 | `docker compose restart qdrant`, then again `hybrid-2.json`; `diff` | STABLE; both `bdc45c4886411ce09cc874d6b785507b469f3243d4d644fa062fbf86700d5682` |
| 11 | the variants, the reach and the `min_score` runs (12) | the tables in "Tier 1" |
| 12 | the cleanup | the scratch generation deleted, and its alias with it; the server holds `rag_qa`'s two generations and its alias only |
| 13 | `rag-query --source 2412.14140v2.pdf --source 2412.14140.pdf --source nope.pdf` | exit 2 before any model: "--source names documents not in the index: '2412.14140.pdf' (did you mean '2412.14140v2.pdf'?), 'nope.pdf'…" |
| 14 | the two HTTP checks, `rag-serve` in the maintainer's terminal | pending: the maintainer's |
| 15 | `gh pr checks` | pending: after the push |

## Review notes for the human

- **The BM25 arithmetic:** compare one chunk's weights with the formula by hand, and check
  that the query side sends 1 per distinct token. The IDF must come from Qdrant, not twice.
- **Our code makes every cut, and the fusion** (design review 2): two searches, each cut by
  S3-2's rule, fused by our RRF with `rrf_k` explicit, then ordered by score, then point ID.
- **The identity rules:** the sparse identity is in the manifest always, and in the
  fingerprint only when set.
- **Tier 1's table** is a decision input: make sure the scratch config differs from the real
  one only where it says.

## Plan (approved 2026-10-21)

Approved in plan mode, after a first presentation in chat. This section is the record to
cross-check the work against; the full plan, with the implementation by module, is
`~/.claude/plans/make-it-formal-as-polymorphic-hare.md`. A departure from it goes under
"Deviation from plan".

| # | Decision |
| --- | --- |
| D1 | **`RetrievalQuery`** is a frozen dataclass in `chain.py`, `RetrievalQuery(question: str, sources: tuple[str, ...] = ())`. The retrieval half becomes `Runnable[RetrievalQuery, list[Document]]`, and tier 1 passes `RetrievalQuery(item.question)`. No `str` union: every wrapper would have to handle two input types. |
| D2 | **Stopwords:** NLTK's English list less its 26 forms with an apostrophe, which `\w+` never yields: 153 words, frozen in `sparse.py`. It drops question words (what, how, which), and every query is a question whose distinct tokens weigh 1 each. |
| D3 | **Tokens:** remove U+00AD (soft hyphen) with the whitespace after it, then `.lower()`, then `re.findall(r"\w+")`, less the stopwords. No NFKC, no stemming. The live index has 823 soft hyphens, all at line-break hyphenations in `7thInternationalConferenceonAutomatedDeduction.pdf`, and no ligature. |
| D4 | **Index:** `int.from_bytes(blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")`. Term frequency is counted per index, so a collision merges two tokens' weights and never sends Qdrant a duplicate index. Today's 19,013-token vocabulary has no collision. |
| D5 | **`Bm25Encoder(*, k1, b, avg_len)`:** all three required, no defaults in code, so the hashed spec determines the encoder. `k1 > 0`, `0 ≤ b ≤ 1`, `avg_len > 0`. A document's length counts the tokens kept after stopwords. |
| D6 | **`RetrieverSpec`:** `mode` is `dense` (also when absent; an explicit `dense` dumps as absent) or `hybrid`. `prefetch` and `rrf_k`, whole numbers of at least 1, are required with `hybrid` and refused without it. A retriever in use whose `prefetch` is below its candidates is refused at load. DEC-20 said "60 by default": the default lives in config instead, where it is hashed. |
| D7 | **A query with no tokens** (all stopwords) skips the sparse branch; the dense list is fused alone. |
| D8 | **`avg_len`** is measured with the final tokenizer over the live index's 1,708 chunk texts (a read-only scroll), rounded to 2 decimals, and fixed in `config.yaml`. About 101.14 before D3. |
| D9 | **The variants:** `rrf_k` 10 against 60, `prefetch` 20 against 40. |
| D10 | **`check_index`** refuses a hybrid retriever over an index without the matching sparse record ("re-run rag-ingest"); `check_index_current`, `generate`'s staleness check, compares the sparse record too. |
| D11 | **Sources:** checked in a FastAPI dependency after the index check: 422 naming the unknown ones, matched exactly (no stripping), duplicates collapsed, 50 at most. The CLI exits 2 with a did-you-mean, and a session prints its filter. A source that a later ingest removes yields no chunks, and the model declines. |

**Tier 1's rules, fixed before the first run.** Hybrid is recommended on only if it wins more
questions than it loses, loses no hit dense had, and its MRR is not lower; a tie stays off.
A variant replaces its default only by the same rule. `min_score`: the highest cutoff that
loses no answerable hit and no recall, else off. Weighted RRF or stemming is recommended for
S3-11 only if the table makes the case.

## Tier 1 (2026-10-21)

Measured on the scratch collection `s3_5_hybrid`, ingested from the real corpus with the
scratch hybrid config (1,708 chunks, 2 min 24 s), and on the real `rag_qa` index for dense.
The scratch configs are `config.yaml` with exact text edits, made by
`~/dev/rag-s3-5/make_configs.py` (each pattern asserted to occur once). The main one,
`configs/hybrid.yaml`, differs from `config.yaml` in the two lines the story names and in
nothing else:

```text
226c226
<     #   sparse_encoder: components.sparse_encoders.bm25
---
>     sparse_encoder: components.sparse_encoders.bm25
230c230
<     retriever: components.retrievers.vector_search
---
>     retriever: components.retrievers.hybrid
```

Each run is `rag-eval retrieval --json --contexts-out` (`~/dev/rag-s3-5/run.sh`), and
`~/dev/rag-s3-5/compare.py` makes the tables from the run files. A win is a better first-hit
rank, a miss ranking below every hit (S2-4's rule). sha256: `make_configs.py` `7562a140…`,
`run.sh` `35ae1a8f…`, `compare.py` `ef640f93…`, `configs/hybrid.yaml` `74067544…`.

**Dense against hybrid**, both reranked to 5, on the 20 answerable questions:

| | Retrieval | Hit rate | MRR | Recall |
| --- | --- | --- | --- | --- |
| (a) | dense, `vector_search` (the real config) | 0.950 (19 of 20) | 0.7958 | 0.8583 |
| (b) | hybrid, `rrf_k` 60, `prefetch` 40 | **1.000** (20 of 20) | **0.8333** | **0.9333** |

| (b) against (a) | Questions |
| --- | --- |
| wins, 2 | `glider-slm` (miss → 2), `glider-data-filtering` (4 → 2) |
| losses, 0 | — |
| ties, 18 | 14 at rank 1 in both; `glider-name` (2), `yolo-acronym` (2), `yolo-anomaly-model` (3), `siekmann-unification-hierarchy` (3) |

- **No hit (a) had is lost.** Recall rises on two questions, `glider-slm` 0/1 → 1/1 and
  `glider-training-data` 1/2 → 2/2, and falls on none.
- **`glider-slm`, tier 1's one miss since S2-3, is found:** "SLM" is a word the embedding
  blurs, the case DEC-20 was made for.
- 14 of the 25 contexts differ (unanswerable questions included): the switch moves tier 2,
  as DEC-26 planned.

**The variants**, each against hybrid 60/40:

| Variant | Hit rate | MRR | Recall | Wins / losses / ties | Contexts that differ |
| --- | --- | --- | --- | --- | --- |
| `rrf_k` 10, `prefetch` 40 | 1.000 | 0.8333 | 0.9333 | 0 / 0 / 20 | 1 of 25 (`cade7-venue`: its 4th and 5th chunks) |
| `rrf_k` 60, `prefetch` 20 | 1.000 | 0.8333 | 0.9333 | 0 / 0 / 20 | 4 of 25, each below rank 3 |

**The reach:** an expected page among the 20 candidates before the rerank (the reranker off,
`search_kwargs.k` 20):

| | Hit rate | Recall |
| --- | --- | --- |
| dense | 0.950 (19 of 20) | 0.9000 |
| hybrid | **1.000** (20 of 20) | **0.9750** |

Only `glider-slm` differs (dense misses it), and recall@20 rises on `glider-slm` and
`glider-training-data`. Dense's miss was a miss of the candidates, which no rerank could fix,
as S2-4 found.

**The `min_score` table, for S3-11:** the reranker's cutoff, in its logits. Chunks are the
mean per question; there are 20 answerable and 5 unanswerable questions.

| Retrieval | `min_score` | Hit rate | Recall | Chunks, answerable | Chunks, unanswerable | Unanswerable with none |
| --- | --- | --- | --- | --- | --- | --- |
| dense | off | 0.950 (19) | 0.8583 | 5.00 | 5.00 | 0 |
| dense | −4 | 0.950 (19) | 0.8583 | 4.90 | 4.00 | 1 |
| dense | −2 | 0.900 (18) | 0.8083 | 4.35 | 4.00 | 1 |
| dense | 0 | 0.850 (17) | 0.7083 | 3.40 | 3.80 | 1 |
| dense | 2 | 0.800 (16) | 0.6917 | 2.40 | 3.20 | 1 |
| hybrid | off | 1.000 (20) | 0.9333 | 5.00 | 5.00 | 0 |
| hybrid | −4 | 1.000 (20) | 0.9333 | 5.00 | 4.00 | 1 |
| hybrid | −2 | 0.950 (19) | 0.8833 | 4.50 | 4.00 | 1 |
| hybrid | 0 | 0.900 (18) | 0.7833 | 3.65 | 4.00 | 1 |
| hybrid | 2 | 0.800 (16) | 0.7167 | 2.50 | 3.20 | 1 |

- **At −4,** hybrid drops nothing but the five chunks of `unanswerable-capital-australia`,
  which then reaches the model with no context. Dense also trims `glider-slm` (5 → 3), a
  question it misses anyway.
- **At −2,** each loses `glider-name` (5 → 1 chunk). Dense also trims `cade7-venue`,
  `glider-data-filtering`, `glider-slm` and `ketonen-ekl`, and hybrid `glider-slm` and
  `ketonen-ekl`.
- **Higher cutoffs** lose more hits in both: dense `glider-data-filtering` at 0, hybrid
  `glider-slm` at 0, and `siekmann-unification-hierarchy` by 2 in both.

**The recommendation, by the rules fixed before the first run (Plan):**

- **Hybrid on.** It wins 2 and loses none, loses no hit dense had, and raises MRR (0.7958 →
  0.8333) and recall (0.8583 → 0.9333).
- **`rrf_k` 60 and `prefetch` 40:** neither variant beats them; each ties all 20.
- **`min_score` −4,** with either retrieval: the highest cutoff that loses no answerable hit
  and no recall. Its effect is one unanswerable question with an empty context, which tier
  2's decline rate measures. S3-11 confirms it there.
- **No case for weighted RRF or stemming:** hybrid loses no question to fix.

The cost is one more search per question, a sparse one. S3-7 measures it (NFR-1).

## Discovered

- **`search_type` was narrowed to `similarity` in S3-2 already** (`SEARCH_TYPES` in
  `schema.py`, checked again in `vectorstore.retriever`). Nothing to do here.
- **The CADE proceedings' text has 823 soft hyphens,** all at line-break hyphenations
  ("con\u00ad\nstruction"), and no ligature. The BM25 tokenizer joins them (D3). The chunk
  text keeps them, so the embedder and the model see the split words as they always have.
  Joining them in the loader would change the chunks, a `CHUNKING_VERSION` bump with a full
  re-embed and a tier-2 re-run: backlog, for S3-11b to weigh.
- **Local mode and the server count IDF's N differently** when a point has an empty sparse
  vector. Local mode counts every point; the server counts only points with a sparse
  vector (a probe, 2026-10-21, of 3 points with 1 empty: idf 0.47 local, 0.18 server).
  No chunk of the real corpus is empty (3 tokens at least), and no hermetic test relies on
  an IDF value, so nothing depends on it.
- **FastAPI keeps a body declared only inside a dependency as the request body,** unembedded,
  and resolves that dependency's own dependencies first (checked on FastAPI 0.143). The
  route declares its body through `known_sources`, after the index's check, so the refusal
  order is as before: the index's 503 comes before any 422.
- **At `min_score` −4 one unanswerable question gets no context at all** (above), an input
  for S3-11's decline rate.
- **`search_kwargs.k`'s refusal was never tested** (`True`, `"5"`, `0`, `2.0`). It is now,
  with the shared `_whole` check.
- **Two probes touched the live Qdrant server outside the alias, read-only on `rag_qa`:** a
  collection `zz_s35_probe` (sparse vectors, the filter, an empty query), created and
  deleted, and the read-only scrolls that measured `avg_len` and found the soft hyphens.
- **A source that a later ingest removes, mid-session, yields no chunks** (D11). The CLI
  checks the sources once, at start-up. The model then declines, which is the honest answer.

## Deviation from plan

- **Model:** fable-routed, run on Opus 5.5 at max effort, since Fable is unavailable
  (CLAUDE.md, "Model routing"). The plan was presented in chat first, then made formal in
  plan mode at the maintainer's request; the board edits the maintainer asked for first
  were made before plan mode.
- **D6 against DEC-20:** `prefetch` and `rrf_k` are required with `mode: hybrid` rather than
  defaulted to 40 and 60 in code, so the hashed dump says everything (as approved).
- **`SPARSE_VERSION` is pinned twice:** over the sample corpus's chunks
  (`tests/test_ingest_incremental.py`), as planned, and over probe texts that exercise every
  tokenizer rule (`tests/test_sparse.py`).
- **The third server test fuses the two server cuts with `fuse` directly,** which is what
  the hybrid retriever runs: the tied collection's points carry no text to make documents
  of.
- **Small things:** the API's dependency is `known_sources` (the plan said
  `checked_query`); `_rerank_after` is folded into `_bound`, which now passes the sources
  to the retriever and the question to the reranker; `rag-ingest` builds the sparse encoder
  at the start of its update step, before embedding, so a bad spec costs no embedding.
