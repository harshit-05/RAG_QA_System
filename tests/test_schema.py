"""The schema's own guarantees, on a minimal hand-built config (no file, no env)."""

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from rag_qa.schema import ComponentSpec, RagConfig, RetrieverSpec

# Targets sit under an allowed prefix (the load-time allowlist check, ISS-04) but name
# modules that don't exist: loading is a string check and never imports (DEC-7).
MINIMAL: dict[str, Any] = {
    "components": {
        "loaders": {"txt": {"_target_": "rag_qa.stub.Loader"}},
        "splitters": {"split": {"_target_": "rag_qa.stub.Splitter", "chunk_size": 10}},
        "embedders": {
            "embed": {"_target_": "rag_qa.stub.Embedder", "model_kwargs": {"device": "cpu"}}
        },
        "llms": {"llm": {"_target_": "rag_qa.stub.LLM"}},
        "retrievers": {"search": {"search_kwargs": {"k": 5}}},
    },
    "pipeline": {
        "ingestion": {
            "loaders": {".txt": "components.loaders.txt"},
            "splitter": "components.splitters.split",
            "embedder": "components.embedders.embed",
        },
        "query": {
            "llm": "components.llms.llm",
            "retriever": "components.retrievers.search",
            "prompt": {"system": "Be precise.", "human": "{context}\n{question}"},
        },
    },
}


@pytest.fixture
def config() -> RagConfig:
    return RagConfig.model_validate(copy.deepcopy(MINIMAL))


def test_target_round_trips_under_its_real_key() -> None:
    # The trap this guards: a field named literally `_target_` is silently dropped.
    spec = ComponentSpec.model_validate({"_target_": "rag_qa.stub.Cls", "model": "m"})
    assert spec.target == "rag_qa.stub.Cls"
    assert spec.spec() == {"_target_": "rag_qa.stub.Cls", "model": "m"}


def test_component_without_target_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ComponentSpec.model_validate({"model": "m"})
    assert exc.value.errors()[0]["loc"] == ("_target_",)


def test_config_is_frozen_at_every_level(config: RagConfig) -> None:
    with pytest.raises(ValidationError):
        config.pipeline = config.pipeline  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.pipeline.query.llm = "components.llms.other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.component("components.llms.llm").target = "evil.Thing"  # type: ignore[misc]


def test_mutating_a_spec_cannot_reach_the_config(config: RagConfig) -> None:
    ref = "components.embedders.embed"
    before = config.component(ref).spec()

    handed_out = config.component(ref).spec()
    handed_out["model_kwargs"]["device"] = "cuda"  # nested dict, the deep case
    handed_out["_target_"] = "evil.Thing"

    assert config.component(ref).spec() == before


def test_retriever_kwargs_are_plain_and_a_copy(config: RagConfig) -> None:
    kwargs = config.retriever("components.retrievers.search").kwargs()
    assert kwargs == {"search_kwargs": {"k": 5}}  # no _target_, no unset options

    kwargs["search_kwargs"]["k"] = 99
    assert config.retriever("components.retrievers.search").kwargs()["search_kwargs"]["k"] == 5


def test_retriever_entry_rejects_a_target() -> None:
    with pytest.raises(ValidationError, match="unknown key '_target_'"):
        RetrieverSpec.model_validate({"_target_": "rag_qa.stub.Retriever", "search_kwargs": {}})


def test_accessors_refuse_the_wrong_kind(config: RagConfig) -> None:
    with pytest.raises(TypeError, match="retriever setting"):
        config.component("components.retrievers.search")
    with pytest.raises(TypeError, match="not a retriever setting"):
        config.retriever("components.llms.llm")


@pytest.mark.parametrize(
    "ref",
    ["components.llms.nope", "components.nokind.llm", "llms.llm", "components.llms"],
)
def test_lookup_of_a_bad_reference_names_it(config: RagConfig, ref: str) -> None:
    with pytest.raises(KeyError, match="components"):
        config.component(ref)


def test_optional_parts_default(config: RagConfig) -> None:
    assert config.pipeline.query.reranker is None
    assert config.components.rerankers == {}


RERANK = "components.rerankers.rerank"


def with_reranker(entry: dict[str, Any] | None = None, **query: Any) -> dict[str, Any]:
    """MINIMAL plus a reranker entry, ``top_n: 5`` unless ``entry`` replaces it, and these
    ``pipeline.query`` keys."""
    data = copy.deepcopy(MINIMAL)
    data["components"]["rerankers"] = {
        "rerank": entry if entry is not None else {"_target_": "rag_qa.stub.Reranker", "top_n": 5}
    }
    data["pipeline"]["query"].update(query)
    return data


@pytest.mark.parametrize("candidates", [5, 20])
def test_reranker_candidates_of_at_least_top_n_load(candidates: int) -> None:
    data = with_reranker(reranker=RERANK, reranker_candidates=candidates)
    assert RagConfig.model_validate(data).pipeline.query.reranker_candidates == candidates


@pytest.mark.parametrize(
    ("query", "problem"),
    [
        pytest.param({"reranker": RERANK}, "'reranker_candidates' is required", id="missing"),
        pytest.param(
            {"reranker_candidates": 20}, "set but 'reranker' is not", id="without a reranker"
        ),
        pytest.param(
            {"reranker": RERANK, "reranker_candidates": 4},
            "reranker_candidates is 4, below the reranker's top_n of 5",
            id="below top_n",
        ),
        pytest.param(
            {"reranker": RERANK, "reranker_candidates": 0},
            "greater than or equal to 1",
            id="zero",
        ),
    ],
)
def test_reranker_candidates_rules(query: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        RagConfig.model_validate(with_reranker(**query))


@pytest.mark.parametrize(
    "top_n",
    [pytest.param(None, id="left to the class default"), "5", True],
)
def test_the_reranker_entry_must_state_top_n_as_a_whole_number(top_n: Any) -> None:
    # Loading never imports the reranker, so a default in its class cannot be read here.
    entry: dict[str, Any] = {"_target_": "rag_qa.stub.Reranker"}
    if top_n is not None:
        entry["top_n"] = top_n
    data = with_reranker(entry, reranker=RERANK, reranker_candidates=20)
    with pytest.raises(ValidationError, match="must state top_n as a whole number"):
        RagConfig.model_validate(data)


def test_a_broken_reranker_reference_is_reported_as_one() -> None:
    # Not a KeyError from the top_n check: that check runs only once references resolve.
    data = with_reranker(reranker="components.rerankers.nope", reranker_candidates=20)
    with pytest.raises(ValidationError, match="has no entry 'nope'"):
        RagConfig.model_validate(data)


def test_without_a_path_context_paths_are_taken_as_given(config: RagConfig) -> None:
    # Programmatic construction: no config file, so nothing to anchor to.
    assert str(config.paths.data) == "corpus"
    assert str(config.paths.vector_store) == "vectorstore/db_faiss"
