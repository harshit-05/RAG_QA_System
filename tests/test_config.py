"""Loading real files: the actionable-error contract (FR-8) and path anchoring (NFR-11)."""

from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_CONFIG, REPO_ROOT, MakeConfig

from rag_qa.config import ConfigError, load_config
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_STATE_PATH, ENV_VECTOR_STORE_PATH


def test_real_config_loads_and_every_reference_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_STATE_PATH)  # the suite points it at each test's own folder
    config = load_config(REAL_CONFIG)

    assert isinstance(config, RagConfig)
    assert config.paths.data == REPO_ROOT / "corpus"
    assert config.paths.state == REPO_ROOT / "var"
    ingestion, query = config.pipeline.ingestion, config.pipeline.query
    for ref in (ingestion.splitter, ingestion.embedder, query.llm, config.pipeline.vector_store):
        assert config.component(ref).target
    assert config.retriever(query.retriever).kwargs()


# --- malformed configs: each one a small edit of the real file ----------------------


def _rename(section: dict[str, Any], old: str, new: str) -> None:
    section[new] = section.pop(old)


MALFORMED = [
    pytest.param(
        lambda c: _rename(c["components"], "llms", "llmS"),
        ["components:", "unknown key 'llmS'", "did you mean 'llms'"],
        id="ISS-01 component-kind typo",
    ),
    pytest.param(
        lambda c: c["pipeline"]["query"].update(llm="components.llms.mistral_olama"),
        ["pipeline.query.llm", "no entry 'mistral_olama'", "did you mean 'mistral_ollama'"],
        id="entry-name typo",
    ),
    pytest.param(
        lambda c: c["pipeline"]["query"].update(retriever="components.llms.mistral_ollama"),
        ["pipeline.query.retriever", "points into components.llms", "components.retrievers"],
        id="reference to the wrong kind",
    ),
    pytest.param(
        lambda c: c["pipeline"]["query"].update(llm="mistral"),
        ["pipeline.query.llm", "not a component reference", "components.llms.<name>"],
        id="bare name instead of a reference",
    ),
    pytest.param(
        lambda c: c["paths"].update(data="/home/someone-else/corpus"),
        ["paths:", "absolute path", "relative to the config file", "RAG_DATA_PATH"],
        id="ISS-02 absolute path",
    ),
    pytest.param(
        lambda c: c["paths"].update(data="~/corpus"),
        ["paths:", "absolute path", "RAG_DATA_PATH"],
        id="home-relative path",
    ),
    pytest.param(
        lambda c: c["paths"].update(data=None),
        ["paths:", "'data' is empty", "RAG_DATA_PATH"],
        id="empty path",
    ),
    # --- the store (S3-2, DEC-19) ---
    pytest.param(
        lambda c: c["paths"].update(vector_store="vectorstore/db_faiss"),
        ["paths:", "'vector_store' was removed in 1.0", "docker compose up -d qdrant"],
        id="v0.3's paths.vector_store",
    ),
    pytest.param(
        lambda c: c["pipeline"].pop("vector_store"),
        ["pipeline.vector_store: required key is missing"],
        id="no store in the pipeline",
    ),
    pytest.param(
        lambda c: c["pipeline"].update(vector_store="components.llms.mistral_ollama"),
        ["pipeline.vector_store", "points into components.llms", "components.vector_stores"],
        id="store reference to the wrong kind",
    ),
    pytest.param(
        lambda c: c["components"]["vector_stores"]["qdrant"].update(_target_="x.FAISS"),
        ["components.vector_stores.qdrant", "outside the import allowlist"],
        id="a store outside the allowlist",
    ),
    pytest.param(
        lambda c: c["components"]["vector_stores"]["qdrant"].update(collection="Rag-QA"),
        ["vector_stores.qdrant: collection 'Rag-QA' must be 1 to 48 lowercase"],
        id="a collection name with a hyphen",
    ),
    pytest.param(
        lambda c: c["components"]["retrievers"]["vector_search"].update(search_type="mmr"),
        ["search_type 'mmr' is not supported", "similarity only"],
        id="a search the store does not run",
    ),
    pytest.param(
        lambda c: c.update(vector_store_path="vectorstore/db_faiss"),
        ["unknown key 'vector_store_path'"],
        id="ISS-11 duplicated top-level path key",
    ),
    pytest.param(
        lambda c: c["pipeline"]["query"].pop("prompt"),
        ["pipeline.query.prompt: required key is missing"],
        id="missing required section",
    ),
    pytest.param(
        lambda c: c["components"]["llms"]["mistral_ollama"].pop("_target_"),
        ["components.llms.mistral_ollama._target_: required key is missing"],
        id="component without _target_",
    ),
    pytest.param(
        lambda c: _rename(
            c["components"]["retrievers"]["vector_search"], "search_kwargs", "serch_kwargs"
        ),
        ["unknown key 'serch_kwargs'", "did you mean 'search_kwargs'"],
        id="retriever option typo",
    ),
    # --- the S1-3 loader map ---
    pytest.param(
        lambda c: c["components"]["loaders"]["pdf"].update(extensions=[".pdf"]),
        ["components:", "'extensions' moved", "pipeline.ingestion.loaders"],
        id="pre-S1-3 extensions key left in a loader",
    ),
    pytest.param(
        lambda c: c["pipeline"]["ingestion"]["loaders"].update({"PDF": "components.loaders.pdf"}),
        ["pipeline.ingestion:", "'PDF'", "can never match", "lowercase"],
        id="extension key without its dot",
    ),
    pytest.param(
        lambda c: c["pipeline"]["ingestion"]["loaders"].update({".Md": "components.loaders.txt"}),
        ["'.Md'", "can never match"],
        id="uppercase extension key",
    ),
    pytest.param(
        lambda c: c["pipeline"]["ingestion"]["loaders"].update(
            {".tar.gz": "components.loaders.txt"}
        ),
        ["'.tar.gz'", "can never match", "one leading dot"],
        id="two-dot extension key (Path.suffix is only .gz)",
    ),
    pytest.param(
        lambda c: c["pipeline"]["ingestion"]["loaders"].update({".pdf": "components.loaders.pfd"}),
        ["pipeline.ingestion.loaders['.pdf']", "no entry 'pfd'", "did you mean 'pdf'"],
        id="loader reference typo",
    ),
    pytest.param(
        lambda c: c["pipeline"]["ingestion"]["loaders"].update(
            {".pdf": "components.llms.mistral_ollama"}
        ),
        ["pipeline.ingestion.loaders['.pdf']", "points into components.llms", "components.loaders"],
        id="loader map pointing at the wrong kind",
    ),
    pytest.param(
        lambda c: c["pipeline"]["ingestion"].pop("loaders"),
        ["pipeline.ingestion.loaders: required key is missing"],
        id="no loader map",
    ),
]


@pytest.mark.parametrize(("edit", "expected"), MALFORMED)
def test_malformed_config_fails_with_an_actionable_error(
    make_config: MakeConfig, edit: Any, expected: list[str]
) -> None:
    path = make_config(edit)
    with pytest.raises(ConfigError) as exc:
        load_config(path)

    message = str(exc.value)
    print(message)  # shown with -s: the evidence FR-8 asks for
    assert str(path.resolve()) in message
    for fragment in expected:
        assert fragment in message
    assert "validation error for" not in message  # never the raw Pydantic header


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", "is empty", id="empty file"),
        pytest.param("# only a comment\n", "is empty", id="comments only"),
        pytest.param("components: [unclosed\n", "line 2", id="invalid YAML"),
        pytest.param("- a\n- b\n", "must be a mapping", id="top level is a list"),
    ],
)
def test_unreadable_config_fails_with_an_actionable_error(
    make_config: MakeConfig, text: str, expected: str
) -> None:
    path = make_config(text=text)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    print(exc.value)
    assert str(path.resolve()) in str(exc.value)
    assert expected in str(exc.value)


def test_missing_config_file_fails_with_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found.*RAG_CONFIG"):
        load_config(tmp_path / "nope.yaml")


def test_config_error_does_not_chain_the_pydantic_dump(make_config: MakeConfig) -> None:
    path = make_config(lambda c: _rename(c["components"], "llms", "llmS"))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__


# --- path anchoring: the S0-4 semantics, unchanged -----------------------------------


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory that is not the config file's directory."""
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def test_relative_path_in_the_file_anchors_to_the_config_dir(
    make_config: MakeConfig, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_STATE_PATH)
    path = make_config()
    config = load_config(path)
    assert config.paths.data == (path.parent / "corpus").resolve()
    assert config.paths.state == (path.parent / "var").resolve()


def test_relative_env_override_anchors_to_the_working_dir(
    make_config: MakeConfig, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATA_PATH", "./mydocs")
    config = load_config(make_config())
    assert config.paths.data == (elsewhere / "mydocs").resolve()


def test_absolute_env_override_is_accepted(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = (tmp_path / "machine-specific" / "var").resolve()
    monkeypatch.setenv(ENV_STATE_PATH, str(target))
    assert load_config(make_config()).paths.state == target


def test_a_state_path_is_resolved_through_links_as_data_is(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # v0.3 kept the store path's last part unresolved, since it was the link to the live
    # generation (DEC-17). paths.state is a plain folder: the link is followed before ".."
    # applies, exactly as resolve() always did.
    (tmp_path / "deep" / "real").mkdir(parents=True)
    (tmp_path / "linked").symlink_to(tmp_path / "deep" / "real")
    monkeypatch.setenv(ENV_STATE_PATH, str(tmp_path / "linked" / ".." / "var"))
    assert load_config(make_config()).paths.state == tmp_path / "deep" / "var"


def test_rag_vector_store_path_is_refused_while_set(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never ignored (DEC-19): caveat 9's scratch ingest set it to stay off the real index,
    # and ignored, it would write the real index. Refused before the file is even read.
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "index"))
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "no-such-config.yaml")
    message = str(exc.value)
    assert "RAG_VECTOR_STORE_PATH is set" in message
    assert "RAG_QDRANT_COLLECTION" in message  # the migration
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, "")  # empty counts as unset, as for every path
    assert isinstance(load_config(make_config()), RagConfig)


@pytest.mark.parametrize("name", ["Rag-QA", "rag--gen-x", "a" * 49, "has space"])
def test_a_bad_collection_in_the_environment_is_refused_at_load(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", name)
    with pytest.raises(ConfigError) as exc:
        load_config(make_config())
    assert "RAG_QDRANT_COLLECTION: collection" in str(exc.value)
    assert "1 to 48 lowercase letters" in str(exc.value)


def test_absolute_path_in_the_file_is_fine_when_the_env_overrides_it(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The env wins, so the file's value is never used and is not an error.
    monkeypatch.setenv("RAG_DATA_PATH", str(tmp_path))
    path = make_config(lambda c: c["paths"].update(data="/home/someone-else/corpus"))
    assert load_config(path).paths.data == tmp_path.resolve()


def test_empty_env_override_counts_as_unset(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATA_PATH", "")
    path = make_config()
    assert load_config(path).paths.data == (path.parent / "corpus").resolve()


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda c: c.pop("paths"), id="no paths section"),
        pytest.param(lambda c: c.update(paths=None), id="empty paths section"),
    ],
)
def test_omitted_paths_fall_back_to_anchored_defaults(
    make_config: MakeConfig, edit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_STATE_PATH)
    path = make_config(edit)
    config = load_config(path)
    assert config.paths.data == (path.parent / "corpus").resolve()
    assert config.paths.state == (path.parent / "var").resolve()


def test_rag_config_selects_the_file(
    make_config: MakeConfig, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_config()
    monkeypatch.setenv("RAG_CONFIG", str(path))
    assert load_config().paths.data == (path.parent / "corpus").resolve()


def test_explicit_path_beats_rag_config(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_CONFIG", str(tmp_path / "does-not-exist.yaml"))
    assert load_config(REAL_CONFIG).paths.data == REPO_ROOT / "corpus"


def test_unrelated_rag_variables_are_ignored(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # e.g. a leftover RAG_EMBED_DEVICE from the dropped device-setting idea (DEC-10)
    monkeypatch.setenv("RAG_EMBED_DEVICE", "cuda")
    assert isinstance(load_config(make_config()), RagConfig)


def test_trust_remote_code_in_the_config_file_is_a_load_error_naming_where(
    make_config: MakeConfig,
) -> None:
    # S2-7 (DEC-18): the real embedder entry, given someone else's code to trust.
    def edit(c: dict[str, Any]) -> None:
        c["components"]["embedders"]["minilm_cpu"]["model_kwargs"]["trust_remote_code"] = True

    with pytest.raises(ConfigError) as exc:
        load_config(make_config(edit))
    assert "components: embedders.minilm_cpu.model_kwargs.trust_remote_code" in str(exc.value)
