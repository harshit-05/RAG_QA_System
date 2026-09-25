# RAG_QA_System

A local, config-driven question-answering engine over your own documents. It uses
retrieval-augmented generation: your files are split into chunks and embedded into a
vector index, and at question time the most relevant chunks go to a local LLM, which
answers from them and cites its sources.

Everything runs on your machine. There are no API keys, your documents never leave the
host, and no GPU is required.

**Status: v0.1 (Phase 0).** Ingestion and the interactive CLI work end to end. An HTTP
API, answer-quality evaluation and incremental ingestion are planned; see the
[roadmap](#roadmap).

## How it works

```text
corpus/   --rag-ingest-->  chunks --MiniLM embeddings-->  FAISS index (vectorstore/)
question  --rag-query--->  top 5 chunks --> local LLM via Ollama --> streamed answer + sources
```

| Piece | Default | Alternatives in `config.yaml` |
| --- | --- | --- |
| Loaders | PDF, DOCX, TXT, Markdown | — |
| Chunking | 1,000 characters, 150 overlap | — |
| Embeddings | `all-MiniLM-L6-v2` on CPU | multilingual model; GPU variants |
| Vector store | FAISS on local disk | Qdrant or pgvector planned (Phase 3) |
| LLM | `mistral` via Ollama | `phi3` (low memory), `qwen2:7b` |

## Requirements

- Linux (developed and tested on Arch Linux)
- [uv](https://docs.astral.sh/uv/), which also installs the pinned Python 3.12
- [Ollama](https://ollama.com), running locally, with a model pulled
- About 6 GB of free RAM for `mistral`, or about 3 GB for the `phi3` fallback
- Network access on the first run, to download the embedding model (about 90 MB)

## Quickstart

```bash
git clone https://github.com/harshit-05/RAG_QA_System.git
cd RAG_QA_System
uv sync                      # builds .venv from the lockfile, with CPU-only PyTorch
ollama pull mistral          # or: ollama pull phi3  (see Configuration)

# put your PDF / DOCX / TXT / MD files in corpus/, then:
uv run rag-ingest            # builds the index in vectorstore/
uv run rag-query             # ask questions; type 'exit' to quit
```

A session looks like this. The answer streams in as it is generated:

```text
Question: How does the YOLOv8 system detect crowds and threats?

Answer:
The YOLOv8 system detects crowds and threats through object detection and anomaly
recognition. It is trained to identify people in high-resolution video streams, count
them, and track their movement across frames. [...]

Sources:
  [1] Batch22_SmartSurveillanceSystemsUsingYOLOv8...pdf, p. 10
  [2] Batch22_SmartSurveillanceSystemsUsingYOLOv8...pdf, p. 3
```

Expect slow answers on CPU. In testing, the first question of a `mistral` session took
about five minutes end to end, including loading the 4.4 GB model into memory; the
answer then streams in as it is generated. `phi3` is much faster, answering in well
under a minute, at some cost in answer quality.

## Configuration

All wiring lives in `config.yaml`, which has two halves. `components` is a library of
things that can be built, and `pipeline` picks which ones to use. Switching a model or an
embedder means changing one pipeline reference; no code changes.

Common changes:

- **Low on memory:** set `pipeline.query.llm` to `components.llms.phi3_ollama`.
- **Non-English documents:** set `pipeline.ingestion.embedder` to
  `components.embedders.multilingual_mpnet_cpu`, then run `rag-ingest` again.

Changing the embedder invalidates the existing index, so always re-run `rag-ingest`
afterwards.

Paths inside `config.yaml` are relative to the file itself, so the project runs from any
directory. Environment variables override them:

| Variable | Overrides |
| --- | --- |
| `RAG_CONFIG` | which config file to load |
| `RAG_DATA_PATH` | the documents folder, default `corpus/` |
| `RAG_VECTOR_STORE_PATH` | the index folder, default `vectorstore/db_faiss` |

## Adding documents

Drop files into `corpus/` and run `uv run rag-ingest` again. Ingestion currently rebuilds
the whole index each time. The sample three-PDF corpus, 561 pages, takes about two
minutes on CPU. Incremental ingestion, which embeds only new or changed files, is planned
for Phase 2.

## Known limitations

- **Answers can misstate facts.** In testing, retrieval found the right passages, but the
  7B model sometimes attributed numbers to the wrong item, or filled a gap in the
  retrieved text with an invented detail. Answer faithfulness is not measured yet; an
  automated evaluation gate is planned for Phase 2. Check important answers against the
  cited pages.
- **"Sources" shows what was retrieved, not what was used.** Retrieval always returns five
  chunks, so even a refusal lists sources. A relevance cutoff is planned.
- **Latency.** On CPU, answers take minutes, not seconds. The long-term target of under
  2 seconds to the first word needs GPU serving.
- **The index is a pickle.** FAISS saves the index with Python's pickle format, which can
  run code when loaded. Only load indexes you built yourself with `rag-ingest`.

## Project layout

```text
config.yaml        component library + pipeline assembly
corpus/            your documents; everything in here gets ingested
src/rag_qa/
  config.py        loads config.yaml, resolves paths and environment overrides
  registry.py      builds objects from the `_target_` entries in the config
  vectorstore.py   the only FAISS code, and the swap point for Phase 3
  ingest.py        the rag-ingest command
  chain.py         the retrieval and answer chain (LangChain LCEL)
  cli.py           the rag-query command
  evaluate.py      RAGAs evaluation (Phase 2)
docs/              specification, architecture, workflow and decision records
stories/           the delivery board and a record of each change
```

## Roadmap

| Release | Phase | Adds |
| --- | --- | --- |
| v0.1 | 0 | Runs end to end (this release) |
| v0.2 | 1 | Validated config, allowlisted `_target_` imports, tests and CI |
| v0.3 | 2 | HTTP API with streaming, reranking, incremental ingestion, evaluation gate |
| v1.0 | 3 | Production vector store, hybrid search, tracing, containers |

The full plan is in [docs/SRS.md](docs/SRS.md), section 12. Image and other multi-modal
ingestion are recorded there as future extension points, not yet scheduled.

## Contributing

Issues and pull requests are welcome. Changes follow a one-story-per-change workflow,
described in [docs/WORKFLOW.md](docs/WORKFLOW.md).

## License

Distributed under the MIT License.
