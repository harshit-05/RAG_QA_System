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


# --- the prompt's placeholders (S2-5; the FR-8 follow-up found in S1-1) ------------------


def with_prompt(system: str, human: str) -> dict[str, Any]:
    data = copy.deepcopy(MINIMAL)
    data["pipeline"]["query"]["prompt"] = {"system": system, "human": human}
    return data


@pytest.mark.parametrize(
    ("system", "human", "problem"),
    [
        pytest.param("Be precise.", "{question}", "it has no {context}", id="no context"),
        pytest.param("Be precise.", "{context}", "it has no {question}", id="no question"),
        pytest.param(
            "Use {context}.", "{context}\n{question}", "system must not contain {context}",
            id="context in system",
        ),
        pytest.param(
            "Be precise.", "{context} {question} {x}", r"unknown placeholder\(s\) \{x\}",
            id="a stray placeholder",
        ),
        pytest.param(
            "Answer as {persona}.", "{context} {question}", r"\{persona\}",
            id="a stray placeholder in system",
        ),
        pytest.param(
            "Be precise.", "{} {context} {question}", r"unknown placeholder\(s\) \{\}",
            id="positional",
        ),
        pytest.param(
            "Be precise.", "{context.text} {question}", r"\{context.text\}", id="attribute access"
        ),
        pytest.param(
            "Be precise.", "{context:{width}} {question}", r"\{width\}", id="nested in a spec"
        ),
        pytest.param(
            "Be precise.", "{context} {question} }", "human: Single '}'", id="unbalanced brace"
        ),
    ],
)
def test_prompt_placeholders_are_checked_at_load(system: str, human: str, problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        RagConfig.model_validate(with_prompt(system, human))


def test_escaped_braces_are_literal_text_and_load() -> None:
    data = with_prompt('Reply as JSON, e.g. {{"answer": "..."}}.', "{context}\n{{x}}\n{question}")
    assert RagConfig.model_validate(data).pipeline.query.prompt.human.count("{{x}}") == 1


def test_question_may_also_appear_in_system() -> None:
    # Only {context} is kept out of the system turn; the question is the user's own text.
    RagConfig.model_validate(with_prompt("Answer: {question}", "{context}\n{question}"))


# --- the evaluation section (S2-5) ---------------------------------------------------------

EVALUATION: dict[str, Any] = {
    "decline_marker": "could not find the answer",
    "judge": {
        "model": "rag-judge",
        "modelfile": "judge.Modelfile",
        "base_url": "http://localhost:11434/v1",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "timeout_s": 2400,
    },
}
REFUSAL = 'Answer from the context. Otherwise say "I could not find the answer."'


def with_evaluation(system: str = REFUSAL, **edits: Any) -> dict[str, Any]:
    data = with_prompt(system, "{context}\n{question}")
    evaluation = copy.deepcopy(EVALUATION)
    for key, value in edits.items():
        target = evaluation["judge"] if key in evaluation["judge"] else evaluation
        target[key] = value
    data["evaluation"] = evaluation
    return data


def test_evaluation_is_optional(config: RagConfig) -> None:
    assert config.evaluation is None


def test_the_evaluation_section_loads() -> None:
    evaluation = RagConfig.model_validate(with_evaluation()).evaluation
    assert evaluation is not None
    assert evaluation.judge.modelfile == "judge.Modelfile"


def test_the_marker_is_matched_in_any_case() -> None:
    RagConfig.model_validate(with_evaluation(system=REFUSAL.upper()))


def test_a_marker_missing_from_the_system_prompt_is_refused() -> None:
    with pytest.raises(ValidationError, match="does not appear in pipeline.query.prompt.system"):
        RagConfig.model_validate(with_evaluation(system="Be precise."))


@pytest.mark.parametrize("marker", ["", "   ", " could not find the answer"])
def test_a_blank_or_padded_marker_is_refused(marker: str) -> None:
    with pytest.raises(ValidationError, match="non-empty phrase"):
        RagConfig.model_validate(with_evaluation(decline_marker=marker))


@pytest.mark.parametrize(
    "modelfile", ["/etc/judge.Modelfile", "../judge.Modelfile", "~/judge.Modelfile", "", "a\\b"]
)
def test_the_modelfile_must_be_a_file_in_the_eval_folder(modelfile: str) -> None:
    with pytest.raises(ValidationError, match="relative to the eval folder"):
        RagConfig.model_validate(with_evaluation(modelfile=modelfile))


def test_the_judge_section_is_closed() -> None:
    data = with_evaluation()
    data["evaluation"]["judge"]["num_ctx"] = 8192
    with pytest.raises(ValidationError, match="unknown key 'num_ctx'"):
        RagConfig.model_validate(data)
