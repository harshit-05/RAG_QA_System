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
        "vector_stores": {"store": {"_target_": "rag_qa.stub.Store", "collection": "rag_qa"}},
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
        "vector_store": "components.vector_stores.store",
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


# --- hybrid retrieval (S3-5, DEC-20) ------------------------------------------------------

HYBRID = {"mode": "hybrid", "search_kwargs": {"k": 5}, "prefetch": 40, "rrf_k": 60}
BM25 = {"_target_": "rag_qa.sparse.Bm25Encoder", "k1": 1.2, "b": 0.75, "avg_len": 100.0}


def with_hybrid(retriever: dict[str, Any] | None = None, *, encoder: bool = True) -> dict[str, Any]:
    """MINIMAL with a hybrid retriever in use (``retriever`` replaces its entry), and the
    sparse encoder set for ingestion unless ``encoder`` is false."""
    data = copy.deepcopy(MINIMAL)
    data["components"]["retrievers"]["hybrid"] = retriever if retriever is not None else HYBRID
    data["components"]["sparse_encoders"] = {"bm25": BM25}
    data["pipeline"]["query"]["retriever"] = "components.retrievers.hybrid"
    if encoder:
        data["pipeline"]["ingestion"]["sparse_encoder"] = "components.sparse_encoders.bm25"
    return data


def test_a_hybrid_retriever_with_its_encoder_loads_and_dumps_every_setting() -> None:
    config = RagConfig.model_validate(with_hybrid())
    spec = config.retriever("components.retrievers.hybrid")
    assert spec.is_hybrid
    assert spec.kwargs() == HYBRID
    assert config.references()["pipeline.ingestion.sparse_encoder"] == (
        "components.sparse_encoders.bm25"
    )


def test_a_dense_retriever_s_dump_does_not_move(config: RagConfig) -> None:
    # Tier 2 hashes the dump (DEC-20): the new keys, unset, stay out of it, and an explicit
    # `mode: dense` is what no mode means.
    assert config.retriever("components.retrievers.search").kwargs() == {"search_kwargs": {"k": 5}}
    explicit = RetrieverSpec.model_validate({"mode": "dense", "search_kwargs": {"k": 5}})
    assert explicit.kwargs() == {"search_kwargs": {"k": 5}}
    assert not explicit.is_hybrid


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        pytest.param({"mode": "sparse"}, "mode 'sparse' is not supported", id="another mode"),
        pytest.param(
            {"mode": "hybrid", "rrf_k": 60}, "mode 'hybrid' needs prefetch", id="no prefetch"
        ),
        pytest.param(
            {"mode": "hybrid", "prefetch": 40}, "mode 'hybrid' needs rrf_k", id="no rrf_k"
        ),
        pytest.param(
            {"mode": "hybrid", "prefetch": "40", "rrf_k": 60},
            "prefetch must be a whole number",
            id="a prefetch as text",
        ),
        pytest.param(
            {"mode": "hybrid", "prefetch": 40, "rrf_k": 0},
            "rrf_k must be a whole number of at least 1",
            id="rrf_k 0",
        ),
        pytest.param(
            {"mode": "hybrid", "prefetch": 40, "rrf_k": 60.5},
            "rrf_k must be a whole number",
            id="a fractional rrf_k",
        ),
        pytest.param(
            {"prefetch": 40}, "prefetch is for a hybrid retriever only", id="dense prefetch"
        ),
        pytest.param(
            {"mode": "dense", "prefetch": 40, "rrf_k": 60},
            "prefetch and rrf_k are for a hybrid retriever only",
            id="dense, both",
        ),
    ],
)
def test_hybrid_settings_go_with_mode_hybrid_and_only_with_it(
    entry: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValidationError, match=problem):
        RetrieverSpec.model_validate({"search_kwargs": {"k": 5}, **entry})


@pytest.mark.parametrize("k", [0, "5", True, 2.0])
def test_a_retriever_s_k_is_a_whole_number_of_at_least_1(k: Any) -> None:
    # The same rule as prefetch and rrf_k: True is not 1, and "5" is not read as 5.
    with pytest.raises(ValidationError, match="search_kwargs.k must be a whole number"):
        RetrieverSpec.model_validate({"search_kwargs": {"k": k}})


def test_a_hybrid_retriever_without_a_sparse_encoder_names_both_slots() -> None:
    with pytest.raises(ValidationError) as exc:
        RagConfig.model_validate(with_hybrid(encoder=False))
    message = str(exc.value)
    assert "pipeline.query.retriever 'components.retrievers.hybrid' is hybrid" in message
    assert "pipeline.ingestion.sparse_encoder is not set" in message


def test_a_sparse_encoder_with_a_dense_retriever_loads(config: RagConfig) -> None:
    # The index then carries sparse vectors nobody searches yet: legitimate (DEC-20).
    data = with_hybrid()
    data["pipeline"]["query"]["retriever"] = "components.retrievers.search"
    loaded = RagConfig.model_validate(data)
    assert loaded.pipeline.ingestion.sparse_encoder == "components.sparse_encoders.bm25"


@pytest.mark.parametrize(
    ("prefetch", "reranker", "problem"),
    [
        (4, False, "prefetch is 4, below the 5 chunks it returns \\(search_kwargs.k\\)"),
        (19, True, "prefetch is 19, below the 20 chunks it returns \\(reranker_candidates\\)"),
    ],
)
def test_each_search_keeps_at_least_the_chunks_the_retriever_returns(
    prefetch: int, reranker: bool, problem: str
) -> None:
    data = with_hybrid({**HYBRID, "prefetch": prefetch})
    if reranker:
        data["components"]["rerankers"] = {"r": {"_target_": "rag_qa.stub.Reranker", "top_n": 5}}
        data["pipeline"]["query"].update(reranker="components.rerankers.r", reranker_candidates=20)
    with pytest.raises(ValidationError, match=problem):
        RagConfig.model_validate(data)
    data["components"]["retrievers"]["hybrid"]["prefetch"] = 20 if reranker else 5
    RagConfig.model_validate(data)  # equal is enough


def test_the_sparse_encoder_slot_must_name_a_sparse_encoder() -> None:
    data = with_hybrid()
    data["pipeline"]["ingestion"]["sparse_encoder"] = "components.embedders.embed"
    with pytest.raises(ValidationError, match="needs an entry of components.sparse_encoders"):
        RagConfig.model_validate(data)
    data["pipeline"]["ingestion"]["sparse_encoder"] = "components.sparse_encoders.nope"
    with pytest.raises(ValidationError, match="has no entry 'nope'"):
        RagConfig.model_validate(data)


def test_a_sparse_encoder_target_is_held_to_the_allowlist() -> None:
    data = with_hybrid()
    data["components"]["sparse_encoders"]["bm25"]["_target_"] = "os.system"
    with pytest.raises(ValidationError, match="outside the import allowlist"):
        RagConfig.model_validate(data)


def test_without_a_path_context_paths_are_taken_as_given(config: RagConfig) -> None:
    # Programmatic construction: no config file, so nothing to anchor to.
    assert str(config.paths.data) == "corpus"
    assert str(config.paths.state) == "var"


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
            "Use {context}.",
            "{context}\n{question}",
            "system must not contain {context}",
            id="context in system",
        ),
        pytest.param(
            "Be precise.",
            "{context} {question} {x}",
            r"unknown placeholder\(s\) \{x\}",
            id="a stray placeholder",
        ),
        pytest.param(
            "Answer as {persona}.",
            "{context} {question}",
            r"\{persona\}",
            id="a stray placeholder in system",
        ),
        pytest.param(
            "Be precise.",
            "{} {context} {question}",
            r"unknown placeholder\(s\) \{\}",
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


# --- trust_remote_code (S2-7, DEC-18) -------------------------------------------------------


def with_entry(kind: str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(MINIMAL)
    data["components"].setdefault(kind, {})[name] = entry
    return data


@pytest.mark.parametrize(
    ("kind", "entry", "where"),
    [
        pytest.param(
            "embedders",
            {"_target_": "rag_qa.stub.Embedder", "trust_remote_code": True},
            "embedders.evil.trust_remote_code",
            id="top level",
        ),
        pytest.param(
            "embedders",
            {"_target_": "rag_qa.stub.Embedder", "model_kwargs": {"trust_remote_code": True}},
            "embedders.evil.model_kwargs.trust_remote_code",
            id="nested in model_kwargs",
        ),
        pytest.param(
            "rerankers",
            {"_target_": "rag_qa.stub.R", "config_kwargs": [{"trust_remote_code": 1}]},
            "rerankers.evil.config_kwargs[0].trust_remote_code",
            id="inside a list",
        ),
        pytest.param(
            "llms",
            {"_target_": "rag_qa.stub.LLM", "trust_remote_code": "false"},
            "llms.evil.trust_remote_code",
            id="the string false is truthy",
        ),
        pytest.param(
            "vector_stores",
            {"_target_": "rag_qa.stub.Store", "trust_remote_code": True},
            "vector_stores.evil.trust_remote_code",
            id="a store, the kind S3-2 added",
        ),
    ],
)
def test_a_truthy_trust_remote_code_anywhere_is_refused_and_located(
    kind: str, entry: dict[str, Any], where: str
) -> None:
    # It runs Python from the model's repository; the allowlist cannot see kwargs (S1-2).
    with pytest.raises(ValidationError, match=rf"{where.replace('[', r'\[')}: trust_remote_code"):
        RagConfig.model_validate(with_entry(kind, "evil", entry))


def test_a_retriever_takes_no_kwarg_but_k_so_none_can_smuggle_trust_remote_code() -> None:
    # Before S3-2 a retriever's kwargs went to FAISS's as_retriever, and the walk above
    # covered them. Now any search option but k is refused on its own (DEC-19).
    entry = {"search_kwargs": {"k": 5, "trust_remote_code": True}}
    with pytest.raises(ValidationError, match="'trust_remote_code' not supported"):
        RagConfig.model_validate(with_entry("retrievers", "evil", entry))


@pytest.mark.parametrize("value", [False, 0, None, ""])
def test_a_falsy_trust_remote_code_loads(value: Any) -> None:
    entry = {"_target_": "rag_qa.stub.Embedder", "model_kwargs": {"trust_remote_code": value}}
    RagConfig.model_validate(with_entry("embedders", "fine", entry))


def test_every_offending_entry_is_named_at_once() -> None:
    data = with_entry("embedders", "a", {"_target_": "rag_qa.stub.E", "trust_remote_code": True})
    data["components"]["llms"]["b"] = {"_target_": "rag_qa.stub.L", "trust_remote_code": True}
    with pytest.raises(ValidationError) as exc:
        RagConfig.model_validate(data)
    assert "embedders.a.trust_remote_code" in str(exc.value)
    assert "llms.b.trust_remote_code" in str(exc.value)


# --- API keys, timeouts and the hosted fallback (S3-6, DEC-21) -------------------------------


@pytest.mark.parametrize(
    ("kind", "entry", "where"),
    [
        pytest.param(
            "llms",
            {"_target_": "langchain_openai.ChatOpenAI", "api_key": "sk-sentinel"},
            "llms.evil.api_key",
            id="top level",
        ),
        pytest.param(
            "llms",
            {"_target_": "rag_qa.stub.LLM", "openai_api_key": "sk-sentinel"},
            "llms.evil.openai_api_key",
            id="a provider's own name for it",
        ),
        pytest.param(
            "llms",
            {"_target_": "rag_qa.stub.LLM", "default_headers": {"GROQ_API_KEY": "sk-sentinel"}},
            "llms.evil.default_headers.GROQ_API_KEY",
            id="nested, in another case",
        ),
        pytest.param(
            "vector_stores",
            {"_target_": "rag_qa.stub.Store", "api_key": "sk-sentinel"},
            "vector_stores.evil.api_key",
            id="a store",
        ),
        pytest.param(
            "embedders",
            {"_target_": "rag_qa.stub.E", "items": [{"api_key": "sk-sentinel"}]},
            "embedders.evil.items[0].api_key",
            id="inside a list",
        ),
    ],
)
def test_an_api_key_anywhere_in_the_components_is_refused_without_its_value(
    kind: str, entry: dict[str, Any], where: str
) -> None:
    # A config file is committed, and the repository is public: a key comes from the
    # environment only (§9, DEC-21). Refused whatever its value, which the message never
    # repeats. The messages are what load_config renders (from None): pydantic's own dump of
    # the error repeats its input, value included, and is never shown.
    with pytest.raises(ValidationError) as exc:
        RagConfig.model_validate(with_entry(kind, "evil", entry))
    message = " ".join(detail["msg"] for detail in exc.value.errors())
    assert f"{where}: an API key in a config file would be committed" in message
    assert "RAG_FALLBACK_API_KEY" in message and "RAG_QDRANT_API_KEY" in message
    assert "sk-sentinel" not in message


@pytest.mark.parametrize("name", ["api_keys", "key", "token_api", "apikey"])
def test_other_names_are_not_api_keys(name: str) -> None:
    RagConfig.model_validate(with_entry("llms", "fine", {"_target_": "rag_qa.stub.L", name: 1}))


TIMEOUT = {"connect": 5, "read": 600, "write": 30, "pool": 30}


@pytest.mark.parametrize(
    "entry",
    [
        {"client_kwargs": {"timeout": TIMEOUT}},
        {"async_client_kwargs": {"timeout": 9.5}},
        {"sync_client_kwargs": {"timeout": TIMEOUT}},
        {"timeout": TIMEOUT},
        {"timeout": 60},
    ],
)
def test_a_timeout_is_seconds_or_all_four_phases(entry: dict[str, Any]) -> None:
    RagConfig.model_validate(with_entry("llms", "fine", {"_target_": "rag_qa.stub.L", **entry}))


@pytest.mark.parametrize(
    ("timeout", "said"),
    [
        ({"connect": 5, "read": 600}, "it has no write, pool"),
        ({**TIMEOUT, "total": 9}, "'total' is not a phase"),
        ({**TIMEOUT, "read": 0}, "read must be a positive number of seconds"),
        ({**TIMEOUT, "read": True}, "read must be a positive number of seconds"),
        ({**TIMEOUT, "pool": float("inf")}, "pool must be a positive number of seconds"),
        (-1, "a timeout is a positive number of seconds"),
        ("600", "a timeout is a positive number of seconds"),
        (None, "a timeout is a positive number of seconds"),
    ],
)
def test_a_timeout_of_another_shape_is_refused_at_load(timeout: Any, said: str) -> None:
    entry = {"_target_": "rag_qa.stub.L", "client_kwargs": {"timeout": timeout}}
    with pytest.raises(ValidationError, match="llms.slow.client_kwargs.timeout: ") as exc:
        RagConfig.model_validate(with_entry("llms", "slow", entry))
    assert said in str(exc.value)


SPARE = {
    "_target_": "langchain_openai.ChatOpenAI",
    "model": "openai/gpt-oss-120b",
    "base_url": "https://api.example.test/v1",
    "max_retries": 0,
    "timeout": TIMEOUT,
}


def with_fallback(entry: dict[str, Any] | None = None, ref: str = "components.llms.spare") -> Any:
    data = with_entry("llms", "spare", SPARE if entry is None else entry)
    data["pipeline"]["query"]["fallback_llm"] = ref
    return data


def test_a_fallback_on_chat_openai_with_its_address_and_no_retries_loads() -> None:
    config = RagConfig.model_validate(with_fallback())
    assert config.pipeline.query.fallback_llm == "components.llms.spare"
    assert config.references()["pipeline.query.fallback_llm"] == "components.llms.spare"


def test_no_fallback_is_the_default(config: RagConfig) -> None:
    assert config.pipeline.query.fallback_llm is None
    assert "pipeline.query.fallback_llm" not in config.references()


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda e: e.update(_target_="rag_qa.stub.LLM"), "a hosted fallback is a langchain_openai"),
        (lambda e: e.pop("base_url"), "it states no base_url"),
        (lambda e: e.update(base_url="  "), "it states no base_url"),
        (lambda e: e.pop("max_retries"), "it must set max_retries: 0"),
        (lambda e: e.update(max_retries=2), "it must set max_retries: 0"),
        (lambda e: e.update(max_retries=False), "it must set max_retries: 0"),
        (lambda e: e.pop("timeout"), "it must set a timeout"),
    ],
    ids=[
        "another class",
        "no address",
        "a blank address",
        "retries unset",
        "two retries",
        "retries false",
        "no timeout",
    ],
)
def test_a_fallback_that_could_hide_a_failure_or_its_destination_is_refused(
    change: Any, said: str
) -> None:
    entry = copy.deepcopy(SPARE)
    change(entry)
    with pytest.raises(
        ValidationError, match="pipeline.query.fallback_llm 'components.llms.spare'"
    ) as exc:
        RagConfig.model_validate(with_fallback(entry))
    assert said in str(exc.value)


def test_the_fallback_must_be_another_model_than_the_one_it_backs() -> None:
    data = with_fallback()
    data["pipeline"]["query"]["llm"] = "components.llms.spare"
    with pytest.raises(ValidationError, match="it names pipeline.query.llm itself"):
        RagConfig.model_validate(data)


def test_the_fallback_must_name_an_llms_entry() -> None:
    with pytest.raises(ValidationError, match="this slot needs an entry of components.llms"):
        RagConfig.model_validate(with_fallback(ref="components.rerankers.spare"))
