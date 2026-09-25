"""Loading real files: the actionable-error contract (FR-8) and path anchoring (NFR-11)."""

from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_CONFIG, REPO_ROOT, MakeConfig

from rag_qa.config import ConfigError, load_config
from rag_qa.schema import RagConfig


def test_real_config_loads_and_every_reference_resolves() -> None:
    config = load_config(REAL_CONFIG)

    assert isinstance(config, RagConfig)
    assert config.paths.data == REPO_ROOT / "corpus"
    assert config.paths.vector_store == REPO_ROOT / "vectorstore" / "db_faiss"
    ingestion, query = config.pipeline.ingestion, config.pipeline.query
    for ref in (ingestion.splitter, ingestion.embedder, query.llm):
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
        lambda c: c["paths"].update(vector_store="~/indexes/faiss"),
        ["paths:", "absolute path", "RAG_VECTOR_STORE_PATH"],
        id="home-relative path",
    ),
    pytest.param(
        lambda c: c["paths"].update(data=None),
        ["paths:", "'data' is empty", "RAG_DATA_PATH"],
        id="empty path",
    ),
    pytest.param(
        lambda c: c["components"].update(vector_stores={"faiss": {"_target_": "x.FAISS"}}),
        ["components:", "unknown key 'vector_stores'"],
        id="ISS-11 dead vector_stores block",
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
        lambda c: _rename(c["components"]["retrievers"]["vector_search"], "search_kwargs", "serch_kwargs"),
        ["unknown key 'serch_kwargs'", "did you mean 'search_kwargs'"],
        id="retriever option typo",
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
    make_config: MakeConfig, elsewhere: Path
) -> None:
    path = make_config()
    config = load_config(path)
    assert config.paths.data == (path.parent / "corpus").resolve()
    assert config.paths.vector_store == (path.parent / "vectorstore" / "db_faiss").resolve()


def test_relative_env_override_anchors_to_the_working_dir(
    make_config: MakeConfig, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATA_PATH", "./mydocs")
    config = load_config(make_config())
    assert config.paths.data == (elsewhere / "mydocs").resolve()


def test_absolute_env_override_is_accepted(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = (tmp_path / "machine-specific" / "index").resolve()
    monkeypatch.setenv("RAG_VECTOR_STORE_PATH", str(target))
    assert load_config(make_config()).paths.vector_store == target


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
def test_omitted_paths_fall_back_to_anchored_defaults(make_config: MakeConfig, edit: Any) -> None:
    path = make_config(edit)
    config = load_config(path)
    assert config.paths.data == (path.parent / "corpus").resolve()
    assert config.paths.vector_store == (path.parent / "vectorstore" / "db_faiss").resolve()


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
