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
question  --rag-query--->  20 nearest chunks --reranker--> best 5 --> local LLM via Ollama --> streamed answer + sources
```

| Piece | Default | Alternatives in `config.yaml` |
| --- | --- | --- |
| Loaders | PDF, DOCX, TXT, Markdown, chosen by file extension | add an extension in `pipeline.ingestion.loaders` |
| Chunking | 1,000 characters, 150 overlap | — |
| Embeddings | `all-MiniLM-L6-v2` on CPU | multilingual model; GPU variants |
| Vector store | FAISS on local disk | Qdrant or pgvector planned (Phase 3) |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder on CPU: rescores 20 candidates, keeps 5 | off (the 5 nearest chunks); GPU variant |
| LLM | `mistral` via Ollama | `phi3` (low memory), `qwen2:7b` |

## Requirements

- Linux (developed and tested on Arch Linux)
- [uv](https://docs.astral.sh/uv/), which also installs the pinned Python 3.12
- [Ollama](https://ollama.com), running locally, with a model pulled
- About 6 GB of free RAM for `mistral`, or about 3 GB for the `phi3` fallback
- Network access on the first run, to download the embedding model and the reranker
  (about 90 MB each)

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
also lists their exit codes. `rag-ingest` exits 0 when the index matches the corpus, 1
when something could not be read (it names what failed, and a document indexed before
keeps its previous version), and 2 when it cannot start: for example, an invalid config,
or another `rag-ingest` already running.

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

### Upgrading from v0.2

An index now carries a manifest recording what is in it and which embedder made it.
v0.2 indexes have none, so `rag-query` refuses them, exiting 2 with "re-run rag-ingest".
Run `uv run rag-ingest` once: it rebuilds the index (about two minutes for the sample
corpus) and keeps the old one beside it as `vectorstore/db_faiss.v02-<timestamp>`.
Delete that folder once the new index works.

### GPU embeddings (Colab / Kaggle)

A plain `uv sync` installs CPU-only PyTorch, and nothing here needs a GPU. For bulk
re-embedding on a CUDA machine, install the CUDA variant instead:

```bash
uv sync --no-group cpu --group cuda   # CUDA 13 build of torch; needs a recent NVIDIA driver
```

Then point `pipeline.ingestion.embedder` in `config.yaml` at a `_cuda` entry, for
example `components.embedders.minilm_cuda`, and re-embed everything on the GPU with
`rag-ingest --rebuild`. A `_cuda` entry and its `_cpu` twin count as the same embedder,
so the index works with either afterwards. From here on, run every command with
`--no-sync`:

```bash
uv run --no-sync rag-ingest --rebuild
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

Changing the embedder invalidates the existing index. `rag-query` then refuses it,
before loading any model, until `rag-ingest` has rebuilt it, which a changed embedder
makes a full rebuild. Switching only between a `_cpu` entry and its `_cuda` twin keeps
the index.

Paths inside `config.yaml` are relative to the file itself, so the project runs from any
directory. Environment variables override them:

| Variable | Overrides |
| --- | --- |
| `RAG_CONFIG` | which config file to load |
| `RAG_DATA_PATH` | the documents folder, default `corpus/` |
| `RAG_VECTOR_STORE_PATH` | the index folder, default `vectorstore/db_faiss` |

## Adding documents

Add, edit or delete files in `corpus/`, then run `uv run rag-ingest` again. Only new and
changed files are loaded and embedded, and the chunks of deleted ones are removed, so a
run where little changed takes seconds. Embedding the whole sample corpus (three PDFs,
561 pages) takes about two minutes on CPU.

- A file that fails to load keeps the chunks it had, and the run exits 1 naming it. The
  next run tries it again.
- `rag-ingest --rebuild` re-embeds everything into a fresh index. A changed embedder or
  splitter does that by itself.
- Each run that changes something publishes a new version of the index in one atomic
  step, so a crash or an error never leaves a half-written index.
  `vectorstore/db_faiss` is a link to the newer of the two versions kept beside it
  (`db_faiss.gen-*`).
- One `rag-ingest` runs at a time. A second one exits 2 at once, saying another is
  running; the lock is `vectorstore/db_faiss.lock`.

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
  manifest.py      what each index version holds: file hashes, the embedder, chunk IDs
  vectorstore.py   the only FAISS code: index versions and their atomic switch; the
                   swap point for Phase 3
  ingest.py        the rag-ingest command: updates the index with what changed
  chain.py         the retrieval and answer chain (LangChain LCEL)
  cli.py           the rag-query command
  evaluation/      the rag-eval command: retrieval scores (tier 1), and generation
                   scored by RAGAs offline, then checked against the floors (tier 2)
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
