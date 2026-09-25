# S0-5: Migrate pipeline code to the resolved LangChain version

| | |
| --- | --- |
| **Status** | Done (2026-09-25) — commit pending, maintainer commits manually |
| **Closes** | ISS-17 |
| **Depends on** | S0-4 (DEC-1 resolved 2026-09-18 — ARCHITECTURE.md §0.1, §0.4, §0.5) |
| **Model** | fable |
| **Plan-first** | yes |

## Goal

`rag_qa` imports and constructs its chain against the locked (per DEC-1)
LangChain version using non-deprecated APIs: modern Ollama binding, current
retrieval-chain construction, current FAISS/embedding interfaces. After this
story the code and the lockfile agree.

## Scope

- `langchain_community.llms.Ollama` → `langchain_ollama.ChatOllama` (config
  `_target_` already updated in S0-4). Chat model returns messages, so the
  chain ends in `StrOutputParser()`.

- Replace legacy `RetrievalQA.from_chain_type` with the hand-composed LCEL
  chain in ARCHITECTURE.md §0.4 — **not** `create_retrieval_chain` and no
  `langchain_classic` import (DEC-1 rule 1). Shape:

  ```python
  retriever = store.as_retriever(**retriever_cfg)
  prompt = ChatPromptTemplate.from_messages([("system", cfg_system), ("human", cfg_human)])
  to_prompt = RunnableLambda(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
  chain = (
      RunnablePassthrough.assign(context=itemgetter("question") | retriever)
      | RunnablePassthrough.assign(answer=to_prompt | prompt | llm | StrOutputParser())
  )
  # invoke({"question": q}) -> {"question": str, "context": list[Document], "answer": str}
  ```

  `format_docs` numbers the chunks and prefixes each with its `source` and
  `page` metadata. `build_rag_chain` must not mutate the config dict.

- Prompt in `config.yaml` becomes two keys, `pipeline.query.prompt.system`
  and `.human`; `{context}` and `{question}` live in the human part. Keep
  the existing wording (grounding + refusal sentence) — this is a
  restructuring, not a rewrite.

- Update `cli.py` for the new result shape (`result["answer"]`, not
  `result["result"]`).

- Surface sources: after the answer, print one line per `result["context"]`
  document — source file name + page (closes the "citations computed then
  discarded" gap in FR-5 — one loop, not a feature).

- `vectorstore.py`: same two functions from S0-3, now against the 1.x
  imports; document the ISS-16 invariant in the module docstring (index dir
  is only ever produced by `rag-ingest` on this host).

- `ingest.py`: return an `IngestReport` (documents, chunks, skipped,
  failed counts) from `ingest(cfg)`; `main()` prints it. Keep the current
  print-and-continue behavior on loader errors — the error policy is Phase 1.

- Keep the REPL behavior otherwise identical.

**Added in the plan-first pass (2026-09-25, maintainer-approved):**

- **CLI streams answers** via `chain.stream`, printing tokens as they arrive,
  then the numbered sources. Exercises the streaming contract Phase 2's SSE
  endpoint depends on, and replaces a 1–2 minute blank terminal on CPU.
- `evaluate.py` moves to the new contract too — it is a caller of
  `build_rag_chain`, and leaving it on `result["result"]` would contradict
  "the code and the lockfile agree".
- `phi3_ollama` added to `components.llms` so the DEC-2 fallback is a
  one-line switch, consistent with the S0-4 embedder modularity.
- **The verification gate below replaces the original `-W error` gate**,
  which was structurally unable to pass or fail correctly (see Discovered).

## Out of scope

- FastAPI / SSE (Phase 2). Error handling/retries (Phase 1 — NFR-6/7).
- Reranker activation (Phase 2).
- Leaving `langchain-community` (DEC-5: staged across Phases 1–3).

## Verification

Run in isolation from the repo's stale index: a scratch corpus and index in
the session scratchpad, selected with `RAG_CONFIG` (which also exercises the
S0-4 fix through the real entry points). The scratch corpus is **fictional**
on purpose, so a correct answer can only have come from retrieval.

```bash
# 1. ingestion through the real entry point
RAG_CONFIG=<scratch>/config.yaml uv run rag-ingest

# 2. deprecation gate: record warnings in-process over imports AND full chain
#    construction. Fail on any LangChainDeprecationWarning, and on any
#    DeprecationWarning except the package-level langchain-community sunset
#    notice (DEC-5). Negative control: the same predicate over the legacy
#    langchain_community.llms.Ollama(model="x") MUST flag a violation.
# 3. contract: invoke keys == ['answer','context','question'], context is
#    Documents; stream order is question → context → answer, answer in >1 chunk
# 4. real CLI: one in-corpus question answered with sources, one question the
#    model knows from training (not in corpus) → refusal sentence
grep -rn "langchain_classic\|^from langchain import\|^import langchain$\|from langchain\." src/rag_qa/ | wc -l   # → 0
uv run ruff check src/        # → only the known ISS-05 BLE001
```

### Results (2026-09-25, model: `phi3` — 4.6 GB free, below the ~6 GB mistral needs)

| Check | Result |
| --- | --- |
| `rag-ingest` via `RAG_CONFIG` from `/tmp` | pass; 1 doc, 1 chunk; embedder downloaded (34 s total) |
| Gate over imports + construction | pass: 1 warning recorded, the allowed sunset notice; 0 violations |
| Negative control | pass: caught `LangChainDeprecationWarning` from legacy `Ollama` |
| `invoke` contract | pass: `['answer', 'context', 'question']`, context is `Document`s |
| `stream` contract | pass: `question → context → answer`, answer in 19 pieces |
| Legacy-import grep | 0 |
| ruff | only ISS-05 `BLE001` |

CLI transcript (ANSI stripped; "Question:" and the next line share a line
only because stdin was piped):

```text
Question: Where is Station Kestrel, and how many staff can it house?
Answer:
Station Kestrel is located at an altitude of 5,050 metres in northern Chile,
specifically positioned about 3 kilometres away from the nearest dish on The
Halvorsen Array. It has accommodations for up to 40 staff members who work
there and rotate on eight-day shifts due to the high altitude conditions.

Sources:
  [1] halvorsen.txt

Question: What is the capital of Australia?
Answer:
I could not find the answer in the provided documents.

Sources:
  [1] halvorsen.txt
```

Every in-corpus detail is correct and fictional. The refusal is the meaningful
one: phi3 knows the answer from training and still declined, which is the
grounding instruction working.

**Not proven here:** retrieval _choosing_ among chunks. The scratch file fits
in one chunk, so every question gets the same context. S0-6 proves retrieval
on the real three-PDF corpus.

## Review notes for the human

The chain-construction diff is the whole review: compare the old
`RetrievalQA.from_chain_type(...)` block against the new LCEL composition
and confirm the prompt wording, `k`, and return-sources behavior carried
over 1:1. Check `format_docs` includes source + page so the model can cite.
The invoke in verification will take 30–90 s on CPU; that is expected.

## Discovered

- **`langchain-community` was sunset on 2026-05-22** (official issue #674):
  frozen, working, unmaintained, and it emits a `DeprecationWarning` on
  import. We use it for FAISS, the three loaders, and the disabled
  cross-encoder. No official standalone FAISS or loader package exists (the
  only `langchain-faiss` on PyPI is an unofficial 0.1.1, rejected on supply-
  chain grounds). Resolved as **DEC-5**, a staged exit riding planned work.
- **The original deprecation gate could never have worked.** On import,
  `langchain_core` calls `warnings.filterwarnings("default", ...)` for
  `LangChainDeprecationWarning`, which overrides a command-line
  `-W error::DeprecationWarning`. Verified: legacy `Ollama` emits a
  `LangChainDeprecationWarning` and still passes `-W error`. So the gate was
  blind to exactly the deprecated-API use ISS-17 exists to remove, and it
  would only ever have failed on the unrelated sunset notice. Replaced by an
  in-process recording gate with a negative control proving it bites.
  Lesson: a check with no known-bad case run against it is an assumption,
  not a check.
- **"Sources" on a refusal is misleading.** The list shows what was
  _retrieved_, not what the answer _used_, and retrieval always returns
  something, so a refusal still prints a source. Proper fix is a relevance
  threshold or citation parsing, both Phase 2 evaluation work → Backlog.
- **`ChatOllama` leaves its HTTP client open**: a `ResourceWarning: unclosed
  socket` to port 11434 appears at interpreter exit. Harmless for a CLI; for
  the Phase 2 API the chain's lifecycle should own and close the client.
  → Backlog.
- The legacy-import grep is prose-sensitive, like S0-4's `cuda` grep: the
  module docstring naming the classic package tripped it. Reworded the prose,
  kept the check strict.
- `build_rag_chain` now takes a loaded config and is silent; the CLI prints
  progress. The Phase 2 API will call the same function without noise.
- PDF citations use `page_label` (the printed page number), falling back to
  0-indexed `page + 1`; docx/txt citations have no page.

## Deviation from plan

None from the approved plan. Verification used the documented `phi3` fallback
because only 4.6 GB was free, below mistral's ~5 GB resident need.
