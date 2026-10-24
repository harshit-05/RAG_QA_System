"""NFR-1 and NFR-2a, measured (S3-7, DEC-22, DEC-27).

    uv run python scripts/bench_retrieval.py [--points 100000] [--questions 200]

**NFR-1** bounds retrieval at 300 ms at p95, for a corpus of up to 100k chunks. This fills a
scratch collection on the configured Qdrant server with ``--points`` synthetic points, built as
a generation is built: a dense vector by cosine at the embedder's dimension, a sparse one with
Qdrant's IDF, and the ``metadata.source`` index. It then times the query embedding plus an
exact search, the store's own cut rule included, through ``QdrantRetriever``, the class every
answer retrieves with. That is ``--questions`` real-looking questions at k=20, the reranker's
candidates, which is stricter than NFR-1's 10 (design review 1); first dense, then hybrid with
``components.retrievers.hybrid``'s settings. **The rerank is not timed:** NFR-1 bounds
retrieval, and FR-4 added the rerank after it.

Synthetic dense vectors are honest here, since an exact search costs the same whatever the
vectors mean. Sparse ones would not be: a BM25 search's cost follows its terms' postings, so
the sparse vectors encode texts resampled from the live index's own chunks.

**NFR-2a** bounds what the first answer event waits for: the sources, in under 2 s at p95,
with the model server idle (DEC-27). The script then times the real retrieval half, the rerank
included, on the live index, over the golden questions. It warns when Ollama holds a model,
since a model generating takes the CPU (STATUS.md caveat 26).

**The scratch collection** (``bench_s3_7`` by default) is never the alias, nor one of its
generations: a name that is either, or that exists already, is refused. It is deleted when the
run ends, however it ends; ``--cleanup`` deletes one a killed run left. Nothing else on the
server is written.

Progress goes to stderr, the results to stdout. Exit status: **0** both targets held; **1** a
target was missed; **2** it could not run (Qdrant unusable, no index, a bad option).
"""

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_qa import resilience
from rag_qa.chain import NoIndexError, RetrievalQuery, open_retrieval
from rag_qa.components import build_embedder, build_store
from rag_qa.config import ConfigError, load_config, resolve_config_path
from rag_qa.evaluation.dataset import GoldenSetError, load_golden
from rag_qa.evaluation.gate import ollama_root
from rag_qa.registry import build_object
from rag_qa.schema import RagConfig, collection_problem
from rag_qa.sparse import Bm25Encoder
from rag_qa.vectorstore import (
    SOURCE_KEY,
    SPARSE,
    VECTOR,
    QdrantStore,
    StoreError,
    generation_pattern,
    live_index,
    retriever,
)

EXIT_OK = 0
EXIT_TARGET_MISSED = 1
EXIT_CANNOT_RUN = 2

#: NFR-1's target, at p95; and NFR-2a's (DEC-27).
NFR1_TARGET_MS = 300.0
NFR2A_TARGET_S = 2.0
#: The reranker's candidates: what every retrieval fetches with the reranker on.
K = 20
#: The retrievers timed: the dense one the real config answers with, and S3-5's hybrid one.
DENSE = "components.retrievers.vector_search"
HYBRID = "components.retrievers.hybrid"
#: The sparse encoder a hybrid generation is written with (S3-5).
BM25 = "components.sparse_encoders.bm25"
#: Points per upsert: far below Qdrant's 32 MiB request limit at 384 dimensions.
BATCH = 512
#: Untimed questions before each timed series: the model's and the connections' first use.
WARM_UP = 10
#: How long Qdrant may take to index the points before the timing starts.
INDEXING_TIMEOUT_S = 1800.0
#: The words a resampled text takes from a chunk: about a chunk's length in BM25 tokens.
WINDOW = (80, 120)
#: Question shapes for the synthetic questions, around phrases of the corpus.
TEMPLATES = (
    "What does the corpus say about {}?",
    "How is {} evaluated?",
    "Which results are reported for {}?",
    "Why does {} matter?",
    "What is {} used for?",
    "How does {} compare with the baselines?",
    "Who introduced {}, and when?",
    "What are the limitations of {}?",
)


def say(message: str) -> None:
    """Progress, on stderr: the results alone go to stdout."""
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Timing:
    """A series of timings, in seconds, and what they say against a target."""

    seconds: list[float]

    def percentile(self, q: float) -> float:
        """The nearest-rank percentile: the smallest timing at or above ``q`` of them."""
        ordered = sorted(self.seconds)
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    @property
    def p50(self) -> float:
        return self.percentile(0.50)

    @property
    def p95(self) -> float:
        return self.percentile(0.95)

    @property
    def max(self) -> float:
        return max(self.seconds)


def timed(run: Callable[[str], Any], questions: Sequence[str]) -> Timing:
    """Run ``run`` over ``questions`` after :data:`WARM_UP` untimed ones, and time each."""
    for question in questions[:WARM_UP]:
        run(question)
    seconds = []
    for question in questions:
        started = time.perf_counter()
        run(question)
        seconds.append(time.perf_counter() - started)
    return Timing(seconds)


# --- the inputs ----------------------------------------------------------------------------


def corpus_texts(store: QdrantStore, generation: str) -> list[str]:
    """Every chunk's text in ``generation``: what the resampled texts are made of."""
    texts: list[str] = []
    offset: Any = None
    while True:
        points, offset = store.client.scroll(
            generation, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        texts += [str((point.payload or {}).get("page_content", "")) for point in points]
        if offset is None:
            return [text for text in texts if text.strip()]


def resampled(texts: Sequence[str], rng: random.Random) -> str:
    """A window of words from one of ``texts``: a chunk of the corpus's own vocabulary."""
    words = rng.choice(texts).split()
    length = rng.randint(*WINDOW)
    start = rng.randint(0, max(0, len(words) - length))
    return " ".join(words[start : start + length])


#: A phrase worth asking about: words of letters, three or more long.
_WORD = re.compile(r"^[A-Za-z][A-Za-z-]{2,}$")


def questions(golden: Sequence[str], texts: Sequence[str], count: int, seed: int) -> list[str]:
    """``count`` real-looking questions: the golden ones first, then phrases of the corpus in
    question shapes, chosen by ``seed``."""
    rng = random.Random(seed)
    asked = list(golden[:count])
    while len(asked) < count:
        words = [word for word in rng.choice(texts).split() if _WORD.match(word)]
        if len(words) < 2:
            continue
        length = rng.randint(2, min(4, len(words)))
        start = rng.randint(0, len(words) - length)
        phrase = " ".join(words[start : start + length]).lower()
        asked.append(rng.choice(TEMPLATES).format(phrase))
    return asked


def ollama_busy(config: RagConfig) -> list[str]:
    """The models Ollama holds now (``/api/ps``): a model generating takes the CPU the timing
    needs (caveat 26). Empty when it holds none, or does not answer."""
    spec = config.component(config.pipeline.query.llm).spec()
    root = ollama_root(spec.get("base_url") if isinstance(spec.get("base_url"), str) else None)
    try:
        with urllib.request.urlopen(f"{root}/api/ps", timeout=5) as response:
            loaded = json.load(response).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [str(model.get("name")) for model in loaded]


# --- the scratch collection ------------------------------------------------------------------


def refusal(store: QdrantStore, name: str) -> str | None:
    """Why ``name`` cannot be the scratch collection, or ``None``."""
    if name == store.collection or generation_pattern(store).fullmatch(name):
        return f"{name} is the index's alias, or one of its generations: never written here"
    problem = collection_problem(name)
    if problem is not None:
        return problem
    if name in {c.name for c in store.client.get_collections().collections}:
        return (
            f"a collection {name} exists already, perhaps left by a killed run: delete it with "
            f"--cleanup, or name another with --collection"
        )
    return None


def create(store: QdrantStore, name: str, dimension: int) -> bool:
    """Create ``name`` as a generation is created: a dense vector by cosine, a sparse one with
    Qdrant's IDF, and the ``metadata.source`` index. ``False`` when it exists by now: another
    run took the name after :func:`refusal` looked, so this run must leave it alone."""
    try:
        store.client.create_collection(
            name,
            vectors_config={
                VECTOR: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    except UnexpectedResponse as e:  # the server's 409
        if e.status_code == 409:
            return False
        raise
    except ValueError:  # local mode's "already exists"
        if name in {c.name for c in store.client.get_collections().collections}:
            return False
        raise
    store.client.create_payload_index(name, SOURCE_KEY, models.PayloadSchemaType.KEYWORD, wait=True)
    return True


def fill(
    store: QdrantStore,
    name: str,
    *,
    points: int,
    dimension: int,
    texts: Sequence[str],
    encoder: Bm25Encoder,
    seed: int,
) -> None:
    """Upsert ``points`` synthetic points into ``name``, in batches, so no more than one batch
    is ever held."""
    client = store.client
    vectors = np.random.default_rng(seed)
    words = random.Random(seed)
    started = time.perf_counter()
    for first in range(0, points, BATCH):
        size = min(BATCH, points - first)
        dense = vectors.standard_normal((size, dimension), dtype=np.float32)
        dense /= np.linalg.norm(dense, axis=1, keepdims=True)  # unit length, as MiniLM's
        batch_texts = [resampled(texts, words) for _ in range(size)]
        sparse = encoder.encode_documents(batch_texts)
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=first + i,
                    vector={
                        VECTOR: dense[i].tolist(),
                        SPARSE: models.SparseVector(
                            indices=list(sparse[i].indices), values=list(sparse[i].values)
                        ),
                    },
                    payload={
                        "page_content": batch_texts[i],
                        "metadata": {"source": f"bench/{(first + i) % 100}.txt"},
                    },
                )
                for i in range(size)
            ],
            wait=True,
        )
        if (first + size) % 10_000 < BATCH or first + size == points:
            say(f"  {first + size:,} of {points:,} points ({time.perf_counter() - started:.0f} s)")


def indexed(store: QdrantStore, name: str) -> None:
    """Wait until Qdrant has finished optimising ``name``: its background indexing would take
    the CPU the timing measures."""
    deadline = time.monotonic() + INDEXING_TIMEOUT_S
    while True:
        status = store.client.get_collection(name).status
        if status == models.CollectionStatus.GREEN:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Qdrant did not finish indexing {name} in {INDEXING_TIMEOUT_S:g} s")
        time.sleep(2)


# --- the command -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_retrieval.py",
        description="Measure NFR-1 (retrieval at 100k points) and NFR-2a (what the sources "
        "event waits for, over the golden questions).",
        epilog="Exit status: 0 both targets held; 1 a target was missed; 2 could not run.",
    )
    parser.add_argument("--config", metavar="PATH", help="config file (default: ./config.yaml)")
    parser.add_argument("--points", type=int, default=100_000, help="synthetic points (100,000)")
    parser.add_argument("--questions", type=int, default=200, help="timed questions (200)")
    parser.add_argument("--collection", default="bench_s3_7", help="scratch collection's name")
    parser.add_argument("--seed", type=int, default=7, help="what the synthetic data is drawn from")
    parser.add_argument(
        "--golden", metavar="PATH", help="the golden set (default: eval/ beside the config)"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="only delete the scratch collection a run left"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.points < 1 or args.questions < 1:
        print("Error: --points and --questions must be at least 1", file=sys.stderr)
        return EXIT_CANNOT_RUN
    try:
        config = load_config(args.config)
        store = build_store(config)
        store.check()
    except (ConfigError, StoreError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    try:
        if args.cleanup:
            return _cleanup(store, args.collection)
        return _run(args, config, store)
    finally:
        store.close()


def _cleanup(store: QdrantStore, name: str) -> int:
    if name == store.collection or generation_pattern(store).fullmatch(name):
        print(f"Error: {name} is the index's alias or one of its generations", file=sys.stderr)
        return EXIT_CANNOT_RUN
    deleted = store.client.delete_collection(name)
    say(f"{'Deleted' if deleted else 'There was no'} collection {name}.")
    return EXIT_OK


def _run(args: argparse.Namespace, config: RagConfig, store: QdrantStore) -> int:
    found = live_index(store)
    if isinstance(found, str):
        print(f"Error: no index to read the corpus from: {found}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    generation, manifest = found
    golden_path = (
        Path(args.golden)
        if args.golden
        else resolve_config_path(args.config).resolve().parent / "eval" / "eval_dataset.jsonl"
    )
    try:
        golden = [item.question for item in load_golden(golden_path)]
    except GoldenSetError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    problem = refusal(store, args.collection)
    if problem is not None:
        print(f"Error: {problem}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    busy = ollama_busy(config)
    if busy:
        say(
            f"Warning: Ollama holds {', '.join(busy)}. If it is generating, it takes the CPU, "
            f"and these are not the model-idle numbers NFR-2a is written for (caveat 26)."
        )

    say("Loading the embedder...")
    embeddings = build_embedder(config)
    encoder = build_object(config.component(BM25).spec())
    texts = corpus_texts(store, generation)
    asked = questions(golden, texts, args.questions, args.seed)
    dimension = len(embeddings.embed_query("dimension"))
    say(f"Filling {args.collection} with {args.points:,} points of {dimension} dimensions...")
    # Only a collection this run created is ever deleted by it: two runs started together
    # once both passed the check above, and the loser's cleanup deleted the winner's.
    if not create(store, args.collection, dimension):
        print(
            f"Error: {args.collection} was created by another run after this one checked: "
            f"leaving it alone. Wait for that run, or name another with --collection",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    try:
        fill(
            store,
            args.collection,
            points=args.points,
            dimension=dimension,
            texts=texts,
            encoder=encoder,
            seed=args.seed,
        )
        say("Waiting for Qdrant to finish indexing them...")
        indexed(store, args.collection)
        say(f"Timing {len(asked)} questions, dense then hybrid...")
        dense, hybrid = _nfr1(config, store, args.collection, embeddings, encoder, asked)
    finally:
        store.client.delete_collection(args.collection)
        say(f"Deleted {args.collection}.")
    nfr1 = _report_nfr1(dense, hybrid, config, args.points, args.collection, len(asked))

    say("Timing the golden questions on the live index, the rerank included...")
    try:
        opened = open_retrieval(config, embeddings, store=store)
    except NoIndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_RUN

    def retrieve(question: str) -> Any:
        return opened.retrieve.invoke(RetrievalQuery(question))

    sources = timed(retrieve, golden)
    nfr2a = _report_nfr2a(sources, opened.generation, manifest, len(golden))
    return EXIT_OK if nfr1 and nfr2a else EXIT_TARGET_MISSED


def _nfr1(
    config: RagConfig,
    store: QdrantStore,
    name: str,
    embeddings: Embeddings,
    encoder: Bm25Encoder,
    asked: Sequence[str],
) -> tuple[Timing, Timing]:
    """The dense and the hybrid series, each searched within a retrieval's deadline, as the
    API's are (DEC-21)."""
    dense = retriever(store, name, embeddings, config.retriever(DENSE).kwargs(), k=K)
    hybrid = retriever(
        store, name, embeddings, config.retriever(HYBRID).kwargs(), sparse=encoder, k=K
    )

    def within_deadline(search: Any) -> Callable[[str], Any]:
        def run(question: str) -> Any:
            with resilience.deadline(store.deadline_s):
                return search.invoke(question)

        return run

    return timed(within_deadline(dense), asked), timed(within_deadline(hybrid), asked)


def _line(label: str, timing: Timing, unit: str, scale: float, note: str) -> str:
    def shown(seconds: float) -> str:
        return f"{seconds * scale:7.1f} {unit}" if unit == "ms" else f"{seconds:5.2f} {unit}"

    return (
        f"  {label:<8} p50 {shown(timing.p50)}   p95 {shown(timing.p95)}   max "
        f"{shown(timing.max)}   ({note})"
    )


def _report_nfr1(
    dense: Timing, hybrid: Timing, config: RagConfig, points: int, name: str, asked: int
) -> bool:
    spec = config.retriever(HYBRID).kwargs()
    held = all(t.p95 * 1000 < NFR1_TARGET_MS for t in (dense, hybrid))
    print(
        f"NFR-1: the query embedding plus an exact search, at k={K} (the reranker's "
        f"candidates; NFR-1 names 10), over {points:,} synthetic points in {name}. The rerank "
        f"is not timed: NFR-1 bounds retrieval alone."
    )
    print(_line("dense", dense, "ms", 1000, f"{asked} questions"))
    print(
        _line(
            "hybrid",
            hybrid,
            "ms",
            1000,
            f"{asked} questions; prefetch {spec['prefetch']}, rrf_k {spec['rrf_k']}",
        )
    )
    print(f"  target: p95 under {NFR1_TARGET_MS:g} ms, dense and hybrid: {_verdict(held)}")
    return held


def _report_nfr2a(sources: Timing, generation: str, manifest: Any, asked: int) -> bool:
    held = sources.p95 < NFR2A_TARGET_S
    chunks = sum(record.chunks for record in manifest.documents.values())
    print(
        f"NFR-2a: the retrieval half the sources event waits for, the rerank included, on "
        f"the live index ({generation}, {chunks:,} chunks), over {asked} golden questions."
    )
    print(_line("sources", sources, "s", 1, f"{asked} questions"))
    print(f"  target: p95 under {NFR2A_TARGET_S:g} s, with the model idle: {_verdict(held)}")
    return held


def _verdict(held: bool) -> str:
    return "held" if held else "MISSED"


if __name__ == "__main__":
    sys.exit(main())
