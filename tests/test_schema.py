"""The schema's own guarantees, on a minimal hand-built config (no file, no env)."""

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from rag_qa.schema import ComponentSpec, RagConfig, RetrieverSpec

MINIMAL: dict[str, Any] = {
    "components": {
        "loaders": {"txt": {"_target_": "pkg.Loader", "extensions": [".txt"]}},
        "splitters": {"split": {"_target_": "pkg.Splitter", "chunk_size": 10}},
        "embedders": {"embed": {"_target_": "pkg.Embedder", "model_kwargs": {"device": "cpu"}}},
        "llms": {"llm": {"_target_": "pkg.LLM"}},
        "retrievers": {"search": {"search_kwargs": {"k": 5}}},
    },
    "pipeline": {
        "ingestion": {
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
    spec = ComponentSpec.model_validate({"_target_": "pkg.Cls", "model": "m"})
    assert spec.target == "pkg.Cls"
    assert spec.spec() == {"_target_": "pkg.Cls", "model": "m"}


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
        RetrieverSpec.model_validate({"_target_": "pkg.Retriever", "search_kwargs": {}})


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


def test_without_a_path_context_paths_are_taken_as_given(config: RagConfig) -> None:
    # Programmatic construction: no config file, so nothing to anchor to.
    assert str(config.paths.data) == "corpus"
    assert str(config.paths.vector_store) == "vectorstore/db_faiss"
