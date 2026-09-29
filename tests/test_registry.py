"""The ``_target_`` machinery and its import allowlist (ISS-04, DEC-7)."""

from typing import Any

import pytest
from conftest import REAL_CONFIG, MakeConfig
from langchain_core.documents import Document

from rag_qa.config import ConfigError, check_imports, load_config
from rag_qa.registry import ALLOWED_PREFIXES, build_object, import_from_string

# --- import_from_string ----------------------------------------------------------------


def test_imports_an_allowed_path() -> None:
    assert import_from_string("rag_qa.registry.build_object") is build_object
    assert import_from_string("langchain_core.documents.Document") is Document


@pytest.mark.parametrize(
    ("path", "cause"),
    [
        pytest.param("rag_qa.no_such_module.Thing", "No module named", id="bad module"),
        pytest.param("rag_qa.registry.NoSuchThing", "has no attribute", id="bad attribute"),
    ],
)
def test_unimportable_path_names_the_cause(path: str, cause: str) -> None:
    with pytest.raises(ImportError, match=f"Could not import {path!r}.*{cause}"):
        import_from_string(path)


@pytest.mark.parametrize(
    "path",
    [
        "os.system",
        "subprocess.Popen",
        "builtins.eval",
        "importlib.import_module",
        "rag_qa_evil.Payload",           # prefix boundary: not the rag_qa package
        "langchain_core_evil.Payload",
        "rag_qa",                        # no attribute part
        "",
    ],
)
def test_path_outside_the_allowlist_is_refused_before_import(path: str) -> None:
    with pytest.raises(ImportError, match="outside the import allowlist") as exc:
        import_from_string(path)
    for prefix in ALLOWED_PREFIXES:  # the message tells the maintainer what is allowed
        assert prefix in str(exc.value)


@pytest.mark.parametrize(
    "path",
    [
        # importlib.import_module, re-exported by our own module: building it with
        # {name: subprocess} imported any module on the path (verified 2026-09-26).
        pytest.param("rag_qa.registry.import_module", id="re-exported function"),
        # a module object (rag_qa.config does `import yaml`) has no __module__ at all
        pytest.param("rag_qa.config.yaml", id="re-exported module"),
    ],
)
def test_reexported_name_under_an_allowed_prefix_is_refused(path: str) -> None:
    with pytest.raises(ImportError, match="re-exports does not count"):
        import_from_string(path)


def test_non_string_target_is_refused() -> None:
    with pytest.raises(ImportError, match="outside the import allowlist"):
        import_from_string(None)  # type: ignore[arg-type]


# --- build_object ----------------------------------------------------------------------

DOC = "langchain_core.documents.Document"


def test_builds_nested_targets_lists_and_plain_dicts() -> None:
    built = build_object(
        {
            "plain": {"a": 1, "b": [1, "two"]},
            "docs": [
                {"_target_": DOC, "page_content": "first"},
                {
                    "_target_": DOC,
                    "page_content": "outer",
                    "metadata": {"inner": {"_target_": DOC, "page_content": "nested"}},
                },
            ],
        }
    )
    assert built["plain"] == {"a": 1, "b": [1, "two"]}
    first, outer = built["docs"]
    assert isinstance(first, Document) and first.page_content == "first"
    assert isinstance(outer.metadata["inner"], Document)
    assert outer.metadata["inner"].page_content == "nested"


def test_build_object_refuses_a_blocked_nested_target() -> None:
    with pytest.raises(ImportError, match="outside the import allowlist"):
        build_object(
            {"_target_": DOC, "page_content": "x", "metadata": {"_target_": "os.system"}}
        )


# --- the allowlist at config load: a string check, no imports --------------------------


def _set_llm_target(target: str) -> Any:
    return lambda c: c["components"]["llms"]["mistral_ollama"].update(_target_=target)


def _set_nested_reranker_target(target: str) -> Any:
    def edit(c: dict[str, Any]) -> None:
        reranker = c["components"]["rerankers"]["cross_encoder"]
        reranker["base_compressor"]["model"]["_target_"] = target

    return edit


@pytest.mark.parametrize(
    ("edit", "location"),
    [
        pytest.param(_set_llm_target("os.system"), "components.llms.mistral_ollama", id="top level"),
        pytest.param(
            _set_nested_reranker_target("subprocess.Popen"),
            "base_compressor.model._target_",
            id="nested, in an unreferenced component",
        ),
    ],
)
def test_blocked_target_fails_at_load(make_config: MakeConfig, edit: Any, location: str) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(make_config(edit))
    message = str(exc.value)
    print(message)
    assert location in message
    assert "outside the import allowlist" in message
    assert "rag_qa/registry.py" in message  # where to change the list, and that it's fixed


def test_load_never_imports(make_config: MakeConfig) -> None:
    # Allowed prefix, nonexistent module: a string check passes it, so load succeeds.
    # Only check_imports (or building the component) finds out.
    config = load_config(make_config(_set_llm_target("rag_qa.no_such_module.ChatModel")))
    with pytest.raises(ConfigError, match="No module named"):
        check_imports(config, [config.pipeline.query.llm])


# --- check_imports ---------------------------------------------------------------------


def test_every_referenced_target_in_the_real_config_imports() -> None:
    config = load_config(REAL_CONFIG)
    check_imports(config, config.references().values())  # raises on any failure


def test_check_imports_reports_a_nested_missing_class_with_its_location(
    make_config: MakeConfig,
) -> None:
    # ISS-03's shape: a nonexistent class name two levels down inside a component.
    def edit(c: dict[str, Any]) -> None:
        c["components"]["rerankers"]["probe"] = {
            "_target_": DOC,
            "page_content": "x",
            "metadata": {"model": {"_target_": "langchain_core.documents.CrossEncoderRerank"}},
        }

    config = load_config(make_config(edit))
    with pytest.raises(ConfigError) as exc:
        check_imports(config, ["components.rerankers.probe"])
    print(exc.value)
    assert "components.rerankers.probe.metadata.model._target_" in str(exc.value)
    assert "has no attribute 'CrossEncoderRerank'" in str(exc.value)


def test_check_imports_catches_a_reexport_the_load_check_cannot(make_config: MakeConfig) -> None:
    config = load_config(make_config(_set_llm_target("rag_qa.registry.import_module")))
    with pytest.raises(ConfigError, match="re-exports does not count"):
        check_imports(config, [config.pipeline.query.llm])


def test_retriever_settings_have_nothing_to_import() -> None:
    config = load_config(REAL_CONFIG)
    assert config.targets(config.pipeline.query.retriever) == []
