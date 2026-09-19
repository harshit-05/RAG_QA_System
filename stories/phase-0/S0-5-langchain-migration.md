# S0-5: Migrate pipeline code to the resolved LangChain version

| | |
| --- | --- |
| **Status** | Todo |
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

## Out of scope

- Streaming, FastAPI (Phase 2). Error handling/retries (Phase 1 — NFR-6/7).
- Reranker activation (Phase 2).

## Verification

```bash
uv run python -c "
from rag_qa.config import load_config
from rag_qa.chain import build_rag_chain
chain = build_rag_chain(load_config('config.yaml'))
print(type(chain).__name__)                 # a Runnable* type
print(sorted(chain.invoke({'question': 'What is this corpus about?'}).keys()))   # → ['answer', 'context', 'question']
"
uv run python -W error::DeprecationWarning -c "import rag_qa.chain, rag_qa.ingest, rag_qa.cli, rag_qa.vectorstore"  # no deprecated imports
grep -rn "langchain_classic\|^from langchain import\|^import langchain$\|from langchain\." src/rag_qa/ | wc -l   # → 0 (DEC-1 rule 1)
```

(Full query verification is S0-6 — this story only proves construction.)

**Caveat:** `build_rag_chain` loads the FAISS store, and the committed
index was pickled under old LangChain — it will likely fail to unpickle
under 1.x. Run a fresh `uv run rag-ingest` before the construction check;
do not debug unpickle errors on the stale index (see STATUS.md
pre-flight caveat 2).

## Review notes for the human

The chain-construction diff is the whole review: compare the old
`RetrievalQA.from_chain_type(...)` block against the new LCEL composition
and confirm the prompt wording, `k`, and return-sources behavior carried
over 1:1. Check `format_docs` includes source + page so the model can cite.
The invoke in verification will take 30–90 s on CPU; that is expected.

## Discovered

—

## Deviation from plan

—
