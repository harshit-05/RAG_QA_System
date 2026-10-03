# RAG_QA_System

A local, config-driven question-answering engine over your own documents. It uses
retrieval-augmented generation: your files are split into chunks and embedded into a
vector index, and at question time the most relevant chunks go to a local LLM, which
answers from them and cites its sources.

Everything runs on your machine. There are no API keys, your documents never leave the
host, and no GPU is required.

**Status: v0.2 (Phase 1).** Ingestion and the interactive CLI work end to end. The
config is validated before anything runs, failures are reported with clear exit codes,
and a test suite runs in CI. An HTTP API, answer-quality evaluation and incremental
ingestion are planned; see the [roadmap](#roadmap).

## How it works

```text
corpus/   --rag-ingest-->  chunks --MiniLM embeddings-->  FAISS index (vectorstore/)
question  --rag-query--->  top 5 chunks --> local LLM via Ollama --> streamed answer + sources
```

| Piece | Default | Alternatives in `config.yaml` |
| --- | --- | --- |
| Loaders | PDF, DOCX, TXT, Markdown, chosen by file extension | add an extension in `pipeline.ingestion.loaders` |
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

# put your PDF / DOCX / TXT / MD files in corpus/ (subfolders too), then:
uv run rag-ingest            # builds the index in vectorstore/
uv run rag-query             # ask questions; Ctrl-C stops an answer, 'exit' quits
```

Both commands take `--config PATH` to use another config file, and `--help`, which
also lists their exit codes. `rag-ingest` exits 0 when every document was indexed, 1
when something could not be read (it indexes the rest and names what it skipped), and 2
when it cannot start, for example because the config is invalid.

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

### GPU embeddings (Colab / Kaggle)

A plain `uv sync` installs CPU-only PyTorch, and nothing here needs a GPU. For bulk
re-embedding on a CUDA machine, install the CUDA variant instead:

```bash
uv sync --no-group cpu --group cuda   # CUDA 13 build of torch; needs a recent NVIDIA driver
```

Then point `pipeline.ingestion.embedder` in `config.yaml` at a `_cuda` entry, for
example `components.embedders.minilm_cuda`, and rebuild the index. A different
embedder invalidates the existing one. From here on, run every command with
`--no-sync`:

```bash
uv run --no-sync rag-ingest
uv run --no-sync rag-query
```

A plain `uv run` first syncs the environment to the default groups, which puts CPU
torch back, and the `_cuda` embedder then fails with "Torch not compiled with CUDA
enabled". In a shell session, `export UV_NO_SYNC=1` does the same for every
command. In a Colab or Kaggle notebook, use `%env UV_NO_SYNC=1`, because each `!`
line runs in a fresh shell and forgets an `export`.

Three things to avoid:

- **Asking for both variants at once.** `uv sync --group cuda` is refused,
  because `cpu` is already on by default.
- **`uv sync --no-default-groups`.** With no variant selected, torch arrives from
  PyPI as the CUDA build, with several GB of `nvidia-*` wheels, without a
  word. CI fails if the installed torch is not the CPU build.
- **`pip install .` or `uv pip install .`.** These ignore dependency groups, so
  they hit the same trap. Install with `uv sync`, or with `uv export` for a
  requirements file; both honour the groups.

## Configuration

All wiring lives in `config.yaml`, which has two halves. `components` is a library of
things that can be built, and `pipeline` picks which ones to use. Switching a model or an
embedder means changing one pipeline reference; no code changes.

Common changes:

- **Low on memory:** set `pipeline.query.llm` to `components.llms.phi3_ollama`.
- **Non-English documents:** set `pipeline.ingestion.embedder` to
  `components.embedders.multilingual_mpnet_cpu`, then run `rag-ingest` again.
- **Another file type:** `pipeline.ingestion.loaders` maps each file extension to a
  loader, for example `".md": components.loaders.txt`. Files with an unlisted
  extension are skipped and reported.

The whole config is checked when it loads. A typo in a key, a reference to a component
that does not exist, or a `_target_` outside the allowed packages stops the command
with an error naming the problem, before any model loads.

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
- **Stopping an answer does not free Ollama at once.** Ctrl-C stops generation and returns
  you to the prompt. If it lands while the model is still reading the prompt, Ollama
  finishes that step first, and your next question waits behind it. How long that takes
  depends on your CPU, the threads Ollama uses and free memory: up to about 4 minutes in
  one test on a laptop CPU where Ollama ran on 2 threads with little free memory.
- **Latency.** On CPU, answers take minutes, not seconds. The long-term target of under
  2 seconds to the first word needs GPU serving.
- **The index is a pickle.** FAISS saves the index with Python's pickle format, which can
  run code when loaded. Only load indexes you built yourself with `rag-ingest`.
- **The config is trusted like code.** The `_target_` allowlist limits which classes a
  config can build, not the arguments it passes them; an embedder given
  `trust_remote_code: true` and someone else's model repository runs that repository's
  Python. Only use config files you wrote or reviewed.

## Project layout

```text
config.yaml        component library + pipeline assembly
corpus/            your documents; everything in here gets ingested
src/rag_qa/
  config.py        loads config.yaml and resolves its paths
  schema.py        what a valid config is; checked at load
  settings.py      the RAG_* environment overrides
  registry.py      builds objects from `_target_` entries, within an import allowlist
  components.py    builds the pipeline's loaders, splitter, embedder and LLM
  loaders.py       PDF, DOCX and text loaders
  vectorstore.py   the only FAISS code, and the swap point for Phase 3
  ingest.py        the rag-ingest command
  chain.py         the retrieval and answer chain (LangChain LCEL)
  cli.py           the rag-query command
  evaluate.py      RAGAs evaluation (Phase 2)
tests/             the test suite, including one regression case per Phase 0 bug
docs/              specification, architecture, workflow and decision records
stories/           the delivery board and a record of each change
```

## Development

The test suite is hermetic: no Ollama, no model download, no network. CI
(`.github/workflows/ci.yml`) runs these gates on every pull request and every push to `main`, and each
one blocks:

```bash
uv run ruff check                       # lint
uv run mypy                             # types
uv run pytest --cov=rag_qa              # tests, with an 80% coverage floor
(                                       # known vulnerabilities, as CI's Audit step
  set -o pipefail
  uv pip freeze --python .venv --exclude-editable \
    | sed 's/^\(torch==.*\)+cpu$/\1/' > /tmp/audit.txt &&
  test -s /tmp/audit.txt
) && uv run --no-sync pip-audit --no-deps --disable-pip -r /tmp/audit.txt
```

The audit reads the installed packages rather than running a plain `pip-audit`, which
would skip torch: the CPU build's `+cpu` version label does not exist on PyPI. Each
guard stops a pass that audited nothing:

- `pipefail` makes a failed freeze fail the whole step;
- `test -s` rejects an empty list, for which `pip-audit` reports "No known
  vulnerabilities";
- `--python .venv` errors when there is no project environment, instead of freezing
  whatever other Python uv finds.

The parentheses keep `pipefail` out of your shell. CI also fails if the installed torch
is not the CPU build.

## Roadmap

| Release | Phase | Adds |
| --- | --- | --- |
| v0.1 | 0 | Runs end to end |
| v0.2 | 1 | Validated config, allowlisted `_target_` imports, own loaders, exit codes, tests and CI (this release) |
| v0.3 | 2 | HTTP API with streaming, reranking, incremental ingestion, evaluation gate |
| v1.0 | 3 | Production vector store, hybrid search, tracing, containers |

The full plan is in [docs/SRS.md](docs/SRS.md), section 12. Image and other multi-modal
ingestion are recorded there as future extension points, not yet scheduled.

## Contributing

Issues and pull requests are welcome. Changes follow a one-story-per-change workflow,
described in [docs/WORKFLOW.md](docs/WORKFLOW.md).

## License

Distributed under the MIT License.
