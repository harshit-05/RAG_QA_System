# RAG_QA_System

A local, config-driven question-answering engine over your own documents. It uses
retrieval-augmented generation: your files are split into chunks and embedded into a
vector index, and at question time the most relevant chunks go to a local LLM, which
answers from them and cites its sources.

Everything runs on your machine. There are no API keys, your documents never leave the
host, and no GPU is required.

**Status: v0.3 (Phase 2).** Ingestion, the interactive CLI and an HTTP API with streamed
answers work end to end. Ingestion embeds only what changed, retrieved chunks are
reranked by a cross-encoder, and two evaluation gates in CI hold retrieval and answer
quality to measured floors. A production vector store, hybrid search, tracing and
containers come next; see the [roadmap](#roadmap). On `main` since v0.3, the index lives
in a Qdrant server: see [Upgrading from v0.3](#upgrading-from-v03).

## How it works

```text
corpus/   --rag-ingest-->  chunks --MiniLM embeddings-->  Qdrant collection (a local container)
question  --rag-query--->  20 nearest chunks --reranker--> best 5 --> local LLM via Ollama --> streamed answer + sources
```

`rag-serve` answers the same way over HTTP, and `rag-eval` measures retrieval and answers
against a hand-checked set of questions.

| Piece | Default | Alternatives in `config.yaml` |
| --- | --- | --- |
| Loaders | PDF, DOCX, TXT, Markdown, chosen by file extension | add an extension in `pipeline.ingestion.loaders` |
| Chunking | 1,000 characters, 150 overlap | — |
| Embeddings | `all-MiniLM-L6-v2` on CPU | multilingual model; GPU variants |
| Vector store | Qdrant server, exact search; each index version a collection behind an alias | another server or collection: `RAG_QDRANT_URL`, `RAG_QDRANT_COLLECTION` |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder on CPU: rescores 20 candidates, keeps 5 | off (the 5 nearest chunks); GPU variant |
| LLM | `mistral` via Ollama | `phi3` (low memory), `qwen2:7b` |

## Requirements

- Linux (developed and tested on Arch Linux)
- [uv](https://docs.astral.sh/uv/), which also installs the pinned Python 3.12
- [Ollama](https://ollama.com), running locally, with a model pulled
- [Docker](https://docs.docker.com/engine/install/) with Compose, to run Qdrant, the vector
  store. Without Docker, Qdrant's static binary from its
  [releases](https://github.com/qdrant/qdrant/releases) works too: run it on
  `127.0.0.1:6333`
- About 6 GB of free RAM for `mistral`, or about 3 GB for the `phi3` fallback
- Network access on the first run, to download the embedding model and the reranker
  (about 90 MB each)

## Quickstart

```bash
git clone https://github.com/harshit-05/RAG_QA_System.git
cd RAG_QA_System
uv sync                      # builds .venv from the lockfile, with CPU-only PyTorch
ollama pull mistral          # or: ollama pull phi3  (see Configuration)
docker compose up -d qdrant  # the vector store, on 127.0.0.1:6333; it restarts with Docker
curl -s 127.0.0.1:6333/readyz   # "all shards are ready"

# put your PDF / DOCX / TXT / MD files in corpus/ (subfolders too), then:
uv run rag-ingest            # builds the index in Qdrant
uv run rag-query             # ask questions; Ctrl-C stops an answer, 'exit' quits
```

The index lives in Qdrant's Docker volume, `qdrant-storage`: `docker compose down` keeps
it, and `docker compose down -v` deletes it, after which `rag-ingest` builds it again.
Every command checks Qdrant first. If it cannot be reached, the command stops at once with
exit 2 and `Is Qdrant running? docker compose up -d qdrant`.

Every command (`rag-ingest`, `rag-query`, `rag-serve` and `rag-eval`'s subcommands) takes
`--config PATH` to use another config file, and `--help`, which also lists its exit codes.
`rag-ingest` exits 0 when the index matches the corpus, 1 when something could not be
read (it names what failed, and a document indexed before keeps its previous version),
and 2 when it cannot start: for example, an invalid config, a corpus folder that is
missing or cannot be read, another `rag-ingest` already running, or Qdrant unreachable.

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

Expect slow answers on CPU. In the v0.3 end-to-end run on a laptop CPU, `mistral`'s first
words came after about 155 seconds, most of it the model reading the prompt, and the
answer was complete at about 175 seconds. The answer streams in as it is generated.
`phi3` is much faster, answering in well under a minute, at some cost in answer quality.

### Upgrading from v0.3

The index has moved from a FAISS folder to a Qdrant server. Answers do not change: for
all 25 questions of the golden set, retrieval gives the same chunks in the same order as
before. What needs something from you:

1. **Start Qdrant, then re-ingest once.** `docker compose up -d qdrant`, then
   `uv run rag-ingest`: it builds the index in Qdrant, which takes about two minutes for the
   sample corpus. Once `rag-query` answers, delete the old folder, `vectorstore/`. Nothing
   reads it any more, and while it is there, `rag-query` mentions it when it finds no index.
2. **Edit your `config.yaml`, if you keep your own.** In `paths`, replace `vector_store:`
   with `state: var` (the folder for the writer lock). Then copy the
   `components.vector_stores` block and the `pipeline.vector_store` line from this
   release's `config.yaml`. A config that still has `paths.vector_store` stops loading, and
   says how to fix it.
3. **Scratch indexes are collection names now.** Where you set `RAG_VECTOR_STORE_PATH` for a
   scratch index, set `RAG_QDRANT_COLLECTION` to a name of your own instead, for example
   `RAG_QDRANT_COLLECTION=scratch_1` (lowercase letters, digits and `_`). Every command
   refuses to start while `RAG_VECTOR_STORE_PATH` is set, so an old script cannot write to
   the real index by mistake.
4. **A retriever takes `k` only.** `components.retrievers` entries may set
   `search_kwargs.k`, and `search_type` only as `similarity`. Other options were FAISS's
   and are refused at load.
5. **Exit 2 also means Qdrant could not be used**, for every command that reads or writes
   the index: `rag-ingest`, `rag-query`, `rag-serve`, `rag-eval retrieval` and
   `rag-eval generate`. `rag-eval check` and `score` never touch it.

### Upgrading from v0.2

Five changes since v0.2 need something from you. The first two apply to every v0.2 user.

1. **Re-ingest once.** An index now carries a manifest recording what is in it and which
   embedder made it. v0.2 indexes have none, so `rag-query` and `rag-eval` refuse them,
   exiting 2 with "re-run rag-ingest". Run `uv run rag-ingest` once: it rebuilds the index
   (about two minutes for the sample corpus) and keeps the old one beside it as
   `vectorstore/db_faiss.v02-<timestamp>`. Delete that folder once the new index works.
   Each chunk now records its corpus-relative `source`, with `source_sha256` and
   `ingested_at`.
2. **Replace the reranker entry in your `config.yaml`.** A `_target_` under
   `langchain_classic.` or `langchain_community.` is now refused, so a config that still
   has v0.2's `components.rerankers.cross_encoder` entry no longer loads. Delete that
   entry. The reranker is now our own cross-encoder: to switch it on, copy the `rerankers`
   block and the two reranker lines of `pipeline.query` from this release's `config.yaml`:

   ```yaml
   reranker: components.rerankers.ms_marco_minilm_cpu
   reranker_candidates: 20      # at least the entry's top_n
   ```

3. **The prompt's placeholders are checked**, if you edited the prompt. `{context}` and
   `{question}` must both be in the `human` turn, `{context}` must not be in the `system`
   turn, and no other `{placeholder}` may appear anywhere; write a literal brace doubled,
   `{{` or `}}`. A prompt that breaks a rule stops the config loading, naming the problem.
   The shipped prompt passes.
4. **`trust_remote_code` is refused.** A config that sets it to a true value anywhere
   under `components` no longer loads. Remove the key or set it to false: a model that
   needs to run its own code is not supported.
5. **`rag-ingest` exits 2 for a corpus folder it cannot read**, where it exited 1. It also
   exits 2 while another ingestion runs, or when something that is not an index sits at
   the index path. Its failure lines now say "N item(s)" where they said "N file(s)", and
   summary lines with a zero count are left out. Update any script that matches the old
   codes or wording.

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

## HTTP API

`rag-serve` makes the same pipeline callable over HTTP. Answers stream as Server-Sent
Events, and ingestion runs as a background job.

```bash
uv run rag-serve                       # http://127.0.0.1:8000; --host, --port, --config
```

It checks that Qdrant and Ollama answer, then loads the models, which takes a few seconds.
It starts without an index too: it then reports not-ready until an ingest builds one. If
Qdrant goes away while it runs, it reports not-ready (503) rather than failing, and serves
again on the first request after Qdrant is back.

| Method and path | Does |
| --- | --- |
| `POST /v1/query` `{"question": "..."}` | streams an answer: `sources`, then `token`…, then `done` (or `error`) |
| `POST /v1/ingest` `{"rebuild": false}` | starts an ingest job, as `rag-ingest` (`--rebuild` with `true`); 202 with its `job_id` |
| `GET /v1/ingest/{job_id}` | the job: `status`, `rag-ingest`'s `exit_code`, the `report` and the `log` |
| `GET /v1/documents` | what the index holds, from its manifest |
| `GET /v1/health` | 200 when an index is served and Ollama answers; 503, saying what is missing |
| `GET /v1/health/live` | 200 while the process runs |
| `GET /docs`, `/openapi.json` | the docs and the schema, to read: a browser's "Try it out" POST sends `Origin` and gets `403` |

```bash
curl -N -X POST localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"question": "How does the YOLOv8 system detect crowds and threats?"}'
```

`-N` turns off curl's buffering, so events print as they arrive. From a run on this
host's CPU (file names shortened, and four of the five sources left out):

```text
event: sources
data: {"sources": [{"n": 1, "citation": "Batch22_...pdf, p. 10", "source": "Batch22_...pdf", "page": "10", "score": 8.016295433044434}, ...]}

: ping

event: token
data: {"text": " The"}

event: token
data: {"text": " YO"}

event: done
data: {"ttft_ms": 157033.62593899964, "total_ms": 176273.471257}
```

`sources` comes first, after about a second of retrieval. While the model reads the prompt
(minutes on CPU; 157 s above), the stream carries a `: ping` comment every 15 seconds.
Each `token` is a chunk of the answer. A stream that ends with neither `done` nor `error`
was cut off.

An ingest job, then its result:

```bash
curl -s -X POST localhost:8000/v1/ingest -H 'content-type: application/json' -d '{"rebuild": false}'
# {"job_id":"a5a336a2…","status":"queued"}
curl -s localhost:8000/v1/ingest/a5a336a2…    # status, exit_code, report and log, when done
curl -s localhost:8000/v1/documents           # what the index now holds
```

How it behaves:

- **One answer at a time.** A question that arrives while another is being answered gets
  `503` with `Retry-After`, rather than waiting in a queue for minutes.
  `RAG_API_MAX_CONCURRENT` raises the limit.
- **Leaving stops the answer.** Closing the connection (Ctrl-C on `curl`) stops
  generation, as Ctrl-C does in `rag-query`, and frees the slot at once. If it lands while
  the model reads the prompt, Ollama may finish that step first (see Known limitations).
- **Ingest jobs run one at a time** (`409` while one runs), and a running `rag-ingest`
  holds them off as well. When a job publishes a new index, the server switches to it;
  answers already streaming finish on the old one. An index that `rag-ingest` publishes
  outside the server is picked up on the next request.
- **A request cannot choose a component.** The query body is the question and nothing
  else; an extra key such as `"llm"` is refused with `422`.
- **No host paths in any response**: only corpus-relative ones.

### Access

| Variable | Default | |
| --- | --- | --- |
| `RAG_API_TOKEN` | unset | when set, every endpoint but the health probes needs `Authorization: Bearer <token>` |
| `RAG_API_HOST`, `RAG_API_PORT` | `127.0.0.1`, `8000` | `--host` and `--port` override them |
| `RAG_API_MAX_CONCURRENT` | `1` | answers streamed at once |

- **Without a token, the server listens only on this machine.** `rag-serve` refuses any
  other `--host`, and the server refuses requests whose `Host` header is not loopback,
  which stops DNS rebinding.
- **With a token, use plain HTTP only on a network you trust.** The token travels in
  cleartext until Phase 3 adds TLS, and `rag-serve` warns about it when it listens beyond
  this machine. Use a long random token, for example `openssl rand -hex 32`.
- **Browsers are refused.** Any request with an `Origin` header gets `403`, and every POST
  must be `application/json`. So a web page you visit cannot reach the server, even on
  localhost. Scripts, `curl` and the CLI send no `Origin`.

Serving another machine on your network, with a token:

```bash
export RAG_API_TOKEN=$(openssl rand -hex 32)
uv run rag-serve --host 0.0.0.0
# from the other machine, with the same token:
curl -N -X POST http://<server>:8000/v1/query -H "Authorization: Bearer $RAG_API_TOKEN" \
     -H 'content-type: application/json' -d '{"question": "What is GLIDER?"}'
```

Ctrl-C stops `rag-serve`: answers in flight get up to 5 seconds, then it exits 0. A
second Ctrl-C in those seconds does not hurry it, though uvicorn offers one: the 5
seconds still run out, then the server's own shutdown (which reports an ingest job it
abandons) is skipped, and uvicorn logs the cancelled tasks as errors. It still exits 0.
A SIGTERM stops it the same way as one Ctrl-C, and the process then ends by that signal
(status 143), as service managers expect. It exits 2, with the reason, when it cannot
start: invalid settings, a non-loopback host without a token, a configuration error,
Qdrant or Ollama not answering, or the port in use.

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
directory. Environment variables override them, and the store's settings:

| Variable | Overrides |
| --- | --- |
| `RAG_CONFIG` | which config file to load |
| `RAG_DATA_PATH` | the documents folder, default `corpus/` |
| `RAG_STATE_PATH` | `paths.state`, default `var/`: the writer lock, never the index |
| `RAG_QDRANT_URL` | the Qdrant server, default `http://localhost:6333` |
| `RAG_QDRANT_COLLECTION` | the index's name, default `rag_qa`; another name is a separate index (a scratch one, say) |
| `RAG_QDRANT_API_KEY` | Qdrant's API key, when it has one; read from the environment only, never from a file |

`RAG_VECTOR_STORE_PATH`, v0.3's index folder, is refused while it is set
([Upgrading from v0.3](#upgrading-from-v03)).

## Adding documents

Add, edit or delete files in `corpus/`, then run `uv run rag-ingest` again. Only new and
changed files are loaded and embedded, and the chunks of deleted ones are removed, so a
run where little changed takes seconds. Embedding the whole sample corpus (three PDFs,
561 pages) takes about two minutes on CPU.

- A file that fails to load keeps the chunks it had, and the run exits 1 naming it. The
  next run tries it again.
- `rag-ingest --rebuild` re-embeds everything into a fresh index. A changed embedder or
  splitter does that by itself.
- Each run that changes something builds a new version of the index, a *generation*, as a
  Qdrant collection of its own: `rag_qa--gen-<time>-<id>`. Then it points the alias
  `rag_qa` at it, in one atomic step. So a crash or an error never leaves a half-written
  index, and a question is answered from the old version or the new one, never a mix.
  The previous generation is kept for questions still using it; older ones are deleted.
- One `rag-ingest` runs at a time on a machine. A second one exits 2 at once, saying
  another is running; the lock is `var/ingest.lock`.

### Pointing the alias at a generation

Deleting the collection the alias points at, by hand or by accident, deletes the alias
with it. `rag-query` then finds no index, and names the newest generation that is still
complete. To serve it again without re-ingesting, point the alias at it (here at
`rag_qa--gen-20261017T120000123456Z-1a2b3c4d`):

```bash
curl -s -X POST 127.0.0.1:6333/collections/aliases -H 'content-type: application/json' \
     -d '{"actions": [{"create_alias": {"collection_name": "rag_qa--gen-20261017T120000123456Z-1a2b3c4d", "alias_name": "rag_qa"}}]}'
```

Or run `rag-ingest`, which publishes a new generation, and keeps the complete ones it finds
when there is no alias. `curl -s 127.0.0.1:6333/aliases` shows where the alias points, and
`curl -s 127.0.0.1:6333/collections` lists the generations.

## Evaluation

`rag-eval` measures the pipeline against a golden set, `eval/eval_dataset.jsonl`: 25
questions about the sample corpus, each checked by hand against its PDF. 20 can be
answered from the corpus, and each names the pages that hold its answer. For the other 5,
the right answer is a refusal. The quality bar has two tiers, and CI enforces both
against the floors in `eval/thresholds.yaml`.

**Tier 1, retrieval**, recomputed by CI on every pull request and every push to `main`:

```bash
uv run rag-eval retrieval        # hit rate, MRR and recall over the 20 answerable questions
```

It needs the index and the embedder but not Ollama, and takes under a minute here. The
v0.3 baseline is hit rate 0.950, MRR 0.796 and recall 0.858, with the reranker on (0.800,
0.588 and 0.717 without it). Exit 1 means a floor was missed, and 2 that it could not run.

**Tier 2, the answers**, scored offline, committed, then checked by CI:

| Command | Does | Cost |
| --- | --- | --- |
| `rag-eval generate` | answers all 25 questions the way users get answers, and records the answers, the chunks each prompt got and the timings in `eval/runs/answers-latest.json` | about 70 min on this laptop CPU; needs the index and `mistral` |
| `rag-eval score` | scores those answers with RAGAs (faithfulness, answer relevancy, context precision, context recall) using a local judge, plus the decline rate on the 5 unanswerable questions; writes `eval/runs/generation-latest.json` | 13–18 h on a CPU, about 35 min on a Colab T4: score on a GPU notebook (below) |
| `rag-eval check` | compares the committed run with this checkout and the floors | instant; no models, no Ollama. The last step of CI's `check` job |

The judge is `rag-judge`, built from `eval/judge.Modelfile` on `gemma2:9b`
(`ollama create rag-judge -f eval/judge.Modelfile`). `score` needs the `eval` extra,
`uv sync --extra eval`, which CI never installs. Do not run the generator and the judge
together on a 16 GB machine: `ollama stop mistral` before scoring locally.

**A committed run goes stale when its inputs change.** It records a fingerprint of what
produced it, and `check` fails on any part that moved, naming the re-run that fixes it:

- **generate and score:** the prompt, an llm, retriever or reranker setting,
  `reranker_candidates`, a corpus file, the embedder, the splitter, the loaders or a
  golden question;
- **score only:** a reference answer, the decline marker or the judge's Modelfile;
- **nothing:** comments, a question's `notes` or `expected_sources`, and runtime-only llm
  keys such as `keep_alive`.

Code is not fingerprinted: after a change that alters answers, re-run tier 2 by hand.
The v0.3 baseline is faithfulness 0.908, answer relevancy 0.719, context precision
0.920, context recall 1.000 and decline rate 0.6, and each floor is 0.05 below. A hand
check found the judge lenient: it scored two wrong answers as faithful. So read tier 2
as a regression alarm rather than a measure of absolute quality.

**Scoring on Colab or Kaggle.** The v0.3 baseline was scored this way, on a T4:

1. On a GPU runtime, install Ollama at the pinned version, so that the judge's digest
   matches this machine's: `apt-get install zstd`, then
   `curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.22.1 sh`. Start
   `ollama serve` in the background (a notebook has no systemd).
2. `ollama pull gemma2:9b`. Clone the repository at the commit holding your
   `answers-latest.json`, install uv, and run `uv sync --extra eval`.
3. `ollama create rag-judge -f eval/judge.Modelfile`.
4. `uv run --no-sync rag-eval score --out <a mounted Drive folder>/generation-latest.json`.
   A run that disconnects resumes from its checkpoint beside the output,
   `generation-latest.partial.jsonl`, when you run the same command again.
5. Commit the scores as `eval/runs/generation-latest.json`. On your machine,
   `uv run rag-eval check --with-ollama` also compares the recorded model digests with your
   local ones.

## Known limitations

- **Answers can misstate facts.** In testing, retrieval found the right passages, but the
  7B model sometimes attributed numbers to the wrong item, or filled a gap in the
  retrieved text with an invented detail. Tier 2 measures faithfulness (0.908 on the
  golden set), but its local judge is lenient: a hand check found it scoring two such
  answers as faithful. Check important answers against the cited pages.
- **"Sources" shows what was retrieved, not what was used.** Retrieval always returns five
  chunks, so even a refusal lists sources. The reranker has a relevance cutoff,
  `min_score`, which stays off until it is tuned against the evaluation.
- **Stopping an answer does not free Ollama at once.** Ctrl-C (or closing the HTTP
  connection) stops generation. If it lands while the model is still reading the prompt,
  Ollama finishes that step first, and your next question waits behind it: about 2.5
  minutes on a laptop CPU in the v0.3 run, and up to 4 minutes in an earlier test where
  Ollama ran on 2 threads with little free memory. Meanwhile Ollama holds most of the CPU,
  so the next question's retrieval is slow too: 7 seconds rather than under 1 in the v0.3
  run.
- **Latency.** On CPU, answers take minutes, not seconds. The long-term target of under
  2 seconds to the first word needs GPU serving.
- **The config is trusted like code.** The `_target_` allowlist limits which classes a
  config can build, not the arguments it passes them. A config that sets
  `trust_remote_code` anywhere is refused at load, since that runs Python from someone
  else's model repository. Other arguments are not checked, so only use config files you
  wrote or reviewed. Nothing in an HTTP request can choose or change a component.

## Project layout

```text
config.yaml        component library + pipeline assembly
compose.yaml       Qdrant, the vector store (`docker compose up -d qdrant`)
corpus/            your documents; everything in here gets ingested
var/               paths.state: the writer lock (not tracked)
src/rag_qa/
  config.py        loads config.yaml and resolves its paths
  schema.py        what a valid config is; checked at load
  settings.py      the RAG_* environment overrides
  registry.py      builds objects from `_target_` entries, within an import allowlist
  components.py    builds the pipeline's loaders, splitter, embedder and LLM
  loaders.py       PDF, DOCX and text loaders
  rerankers.py     the cross-encoder that rescores retrieved chunks
  manifest.py      what each index version holds: file hashes, the embedder, chunk counts
  vectorstore.py   the only module that talks to Qdrant: index versions, the alias
                   switch, recovery, and exact search
  ingest.py        the rag-ingest command: updates the index with what changed
  chain.py         the retrieval and answer chain (LangChain LCEL)
  cli.py           the rag-query command
  answering.py     one answer as a stream of events, for the CLI, the API and evaluation
  api/             the rag-serve command: the HTTP API, its ingest jobs and its checks
  evaluation/      the rag-eval command: retrieval scores (tier 1), and generation
                   scored by RAGAs offline, then checked against the floors (tier 2)
eval/              the golden set, the floors, the judge's Modelfile and the committed
                   tier-2 run
scripts/           fetch_dataset.py, a standalone helper that downloads a Hugging Face
                   dataset to downloads/, never into a corpus
tests/             the test suite, including one regression case per Phase 0 bug
docs/              specification, architecture, workflow and decision records
stories/           the delivery board and a record of each change
```

## Development

The test suite is hermetic: no Ollama, no model download, no network, and no Qdrant
server: each test gets its own in-memory Qdrant. The few tests that only a real server can
settle are marked `qdrant_server` and skipped unless `RAG_TEST_QDRANT_URL` names one:

```bash
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server
```

Each works under a random collection name of its own, and removes it afterwards. CI
(`.github/workflows/ci.yml`) runs on every pull request, every push to `main` and every
release tag. Its `check` job runs these gates, and each one blocks:

```bash
uv run ruff check                       # lint
uv run ruff format --check              # formatting; `uv run ruff format` fixes it
uv run mypy                             # types
uv run pytest --cov=rag_qa              # tests, with an 80% coverage floor
uv run rag-eval check                   # tier 2: the committed run is fresh and meets its floors
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

The code was formatted once, in a single commit that `.git-blame-ignore-revs` names, so
that `git blame` can skip it. GitHub's blame view reads that file. For a local
`git blame` to skip it too, run this once per clone:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

CI's second job, `eval-retrieval`, starts Qdrant as a service, ingests the corpus into
it, runs tier 1 against its floors, and runs the `qdrant_server` tests. It is CI's only
job that downloads models: it fetches the two from Hugging Face and caches them, and on a
cache hit it runs offline. When it is red and `check` is green, retrieval quality moved,
not the code.

## Roadmap

| Release | Phase | Adds |
| --- | --- | --- |
| v0.1 | 0 | Runs end to end |
| v0.2 | 1 | Validated config, allowlisted `_target_` imports, own loaders, exit codes, tests and CI |
| v0.3 | 2 | HTTP API with streaming, reranking, incremental ingestion, two evaluation gates (this release) |
| v1.0 | 3 | Production vector store, hybrid search, tracing, containers |

The full plan is in [docs/SRS.md](docs/SRS.md), section 12. Image and other multi-modal
ingestion are recorded there as future extension points, not yet scheduled.

## Contributing

Issues and pull requests are welcome. Changes follow a one-story-per-change workflow,
described in [docs/WORKFLOW.md](docs/WORKFLOW.md).

## License

Distributed under the MIT License.
