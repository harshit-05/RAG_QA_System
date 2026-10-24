"""``scripts/bench_retrieval.py`` (S3-7): NFR-1's and NFR-2a's benchmark, on a small scale.

It runs against the test's in-memory Qdrant, over the sample corpus indexed with the fakes:
a few hundred synthetic points rather than 100k, which only the real run on the local server
measures. What these tests hold is the rest: that it times the dense and the hybrid search and
the golden questions, says what it measured, never writes the alias, and leaves its scratch
collection behind under no ending. The script is not a package module, so it is loaded from its
path, as ``scripts/fetch_dataset.py``'s test does.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import REPO_ROOT, generations, live_generation
from qdrant_client import QdrantClient, models

GOLDEN = [
    {
        "id": "notes",
        "question": "What do the notes say?",
        "ground_truth": "Plain text notes.",
        "answerable": True,
        "expected_sources": [{"source": "notes.txt"}],
    },
    {
        "id": "capital",
        "question": "What is the capital of Australia?",
        "ground_truth": "I could not find the answer in the provided documents.",
        "answerable": False,
        "must_not_contain": ["Canberra"],
    },
]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bench_retrieval", REPO_ROOT / "scripts" / "bench_retrieval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_script()


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in GOLDEN))
    return path


def run(config: Path, golden: Path, *args: str) -> int:
    return int(
        bench.main(["--config", str(config), "--golden", str(golden), "--points", "300", *args])
    )


def collections(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def test_it_times_dense_hybrid_and_the_golden_questions_then_deletes_its_collection(
    fake_rag: Path, golden: Path, qdrant: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    before = collections(qdrant)
    assert run(fake_rag, golden, "--questions", "12") == bench.EXIT_OK
    out = capsys.readouterr().out
    assert "NFR-1: the query embedding plus an exact search, at k=20" in out
    assert "over 300 synthetic points in bench_s3_7" in out
    assert "The rerank is not timed: NFR-1 bounds retrieval alone." in out
    assert "  dense    p50" in out and "(12 questions)" in out
    assert "  hybrid   p50" in out and "prefetch 40, rrf_k 60" in out
    assert "NFR-2a: the retrieval half the sources event waits for" in out
    assert "over 2 golden questions" in out
    assert out.count(": held") == 2
    assert collections(qdrant) == before  # the scratch collection is gone; nothing else moved


def test_a_name_that_is_the_alias_or_one_of_its_generations_is_refused(
    fake_rag: Path, golden: Path, qdrant: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("rag_qa", live_generation(qdrant)):
        assert run(fake_rag, golden, "--collection", name) == bench.EXIT_CANNOT_RUN
        assert "never written here" in capsys.readouterr().err
    assert len(generations(qdrant)) == 1


def test_a_collection_left_by_a_killed_run_is_refused_and_cleanup_deletes_it(
    fake_rag: Path, golden: Path, qdrant: QdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    qdrant.create_collection("bench_s3_7", vectors_config={})
    assert run(fake_rag, golden) == bench.EXIT_CANNOT_RUN
    assert "exists already, perhaps left by a killed run" in capsys.readouterr().err
    assert run(fake_rag, golden, "--cleanup") == bench.EXIT_OK
    assert "bench_s3_7" not in collections(qdrant)


def test_the_collection_goes_even_when_the_run_fails_midway(
    fake_rag: Path, golden: Path, qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fails(*args: object, **kwargs: object) -> None:
        raise TimeoutError("indexing never finished")

    monkeypatch.setattr(bench, "indexed", fails)
    with pytest.raises(TimeoutError):
        run(fake_rag, golden)
    assert "bench_s3_7" not in collections(qdrant)


def test_a_run_that_loses_the_name_to_another_leaves_the_others_collection_alone(
    fake_rag: Path,
    golden: Path,
    qdrant: QdrantClient,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Found in S3-7's own run: two runs started 4 s apart both passed the check, which looks
    # about 10 s before the create, while the embedder loads. The second's create got a 409,
    # and its cleanup deleted the first's collection mid-timing (a 404 there).
    real_refusal = bench.refusal

    def raced(store: Any, name: str) -> str | None:
        found = real_refusal(store, name)  # nothing in the way yet...
        qdrant.create_collection(name, vectors_config={})  # ...then the other run creates it
        qdrant.upsert(name, points=[models.PointStruct(id=1, vector={}, payload={"x": 1})])
        return found

    monkeypatch.setattr(bench, "refusal", raced)
    assert run(fake_rag, golden) == bench.EXIT_CANNOT_RUN
    assert "was created by another run after this one checked" in capsys.readouterr().err
    assert qdrant.count("bench_s3_7").count == 1  # the other run's, untouched


def test_the_percentile_is_the_nearest_rank() -> None:
    timing = bench.Timing([float(n) for n in range(1, 26)])  # 25 samples, as the golden set
    assert (timing.p50, timing.p95, timing.max) == (13.0, 24.0, 25.0)


def test_the_synthetic_questions_follow_the_golden_ones_and_the_seed() -> None:
    texts = ["Explainable ranking of judge models across many domains and criteria"]
    asked = bench.questions(["A golden one?"], texts, 5, seed=7)
    assert asked[0] == "A golden one?" and len(asked) == 5
    assert asked == bench.questions(["A golden one?"], texts, 5, seed=7)  # reproducible
    assert all(question.endswith("?") for question in asked)
