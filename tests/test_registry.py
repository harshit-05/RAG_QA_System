"""The ``_target_`` machinery and its import allowlist (ISS-04, DEC-7)."""

from typing import Any

import pytest
from conftest import REAL_CONFIG, MakeConfig
from langchain_core.documents import Document

from rag_qa import registry
from rag_qa.config import ConfigError, check_imports, load_config
from rag_qa.registry import ALLOWED_PREFIXES, allowlist_hint, build_object, import_from_string
from rag_qa.schema import Components

#: Targets under the two prefixes S2-4 removed: the old reranker entry's three classes,
#: and FAISS, which vectorstore.py still imports from langchain_community in source.
DROPPED = [
    "langchain_classic.retrievers.ContextualCompressionRetriever",
    "langchain_classic.retrievers.document_compressors.CrossEncoderReranker",
    "langchain_community.cross_encoders.HuggingFaceCrossEncoder",
    "langchain_community.vectorstores.FAISS",
]

# --- import_from_string ----------------------------------------------------------------


def test_imports_an_allowed_path() -> None:
    assert import_from_string("rag_qa.config.ConfigError") is ConfigError
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
        *DROPPED,                        # allowed until S2-4
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


@pytest.mark.parametrize(
    ("path", "defined_in"),
    [
        # vectorstore.py imports FAISS, so our own module re-exports a langchain_community
        # class. The prefix check passes it; only the defining-module check stops it.
        pytest.param("rag_qa.vectorstore.FAISS", "langchain_community", id="FAISS"),
        # rerankers.py imports the model class, from a package never on the list.
        pytest.param("rag_qa.rerankers.CrossEncoder", "sentence_transformers", id="CrossEncoder"),
    ],
)
def test_a_package_off_the_list_cannot_come_back_through_our_own_modules(
    path: str, defined_in: str
) -> None:
    with pytest.raises(ImportError, match="re-exports does not count") as exc:
        import_from_string(path)
    assert f"defined in '{defined_in}." in str(exc.value)


@pytest.mark.parametrize(
    "path",
    [
        # Functions *defined* in an allowed package that wrap importlib: they pass the
        # defining-module check, and building them imports any module on the path or
        # hands back any attribute, e.g. subprocess.Popen (verified 2026-09-30).
        pytest.param("langchain_core.utils.utils.guard_import", id="guard_import"),
        pytest.param("langchain_core.utils.guard_import", id="guard_import, public name"),
        pytest.param("langchain_core._import_utils.import_attr", id="import_attr"),
    ],
)
def test_function_under_an_allowed_prefix_is_refused(path: str) -> None:
    with pytest.raises(ImportError, match="is not a class"):
        import_from_string(path)


def test_building_an_import_helper_never_calls_it() -> None:
    # Had guard_import run, it would raise its own "Could not import ..." instead.
    with pytest.raises(ImportError, match="is not a class"):
        build_object(
            {
                "_target_": "langchain_core.utils.utils.guard_import",
                "module_name": "rag_qa_never_imported",
            }
        )


def test_instance_under_an_allowed_prefix_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # An instance inherits __module__ from its class, so the defining-module check
    # passes it, and build_object would then call it.
    probe = type("Probe", (), {"__module__": "rag_qa.stub", "__call__": lambda self: self})
    monkeypatch.setattr(registry, "PROBE", probe(), raising=False)
    with pytest.raises(ImportError, match="is not a class"):
        import_from_string("rag_qa.registry.PROBE")


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
        # The _cuda entry: never referenced on this host, whatever the pipeline uses.
        reranker = c["components"]["rerankers"]["ms_marco_minilm_cuda"]
        reranker["wrapper"] = {"model": {"_target_": target}}

    return edit


@pytest.mark.parametrize(
    ("edit", "location"),
    [
        pytest.param(
            _set_llm_target("os.system"), "components.llms.mistral_ollama", id="top level"
        ),
        pytest.param(
            _set_nested_reranker_target("subprocess.Popen"),
            "wrapper.model._target_",
            id="nested, in an unreferenced component",
        ),
        # The old reranker entry's own shape: a dropped prefix, two levels down.
        pytest.param(
            _set_nested_reranker_target(DROPPED[2]),
            "wrapper.model._target_",
            id="nested, a dropped prefix",
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


@pytest.mark.parametrize("target", DROPPED)
def test_a_dropped_prefix_fails_at_load_with_the_hint(make_config: MakeConfig, target: str) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(make_config(_set_llm_target(target)))
    message = str(exc.value)
    assert "outside the import allowlist" in message
    assert allowlist_hint() in message  # what is allowed, and where the fixed list lives


def test_the_hint_no_longer_offers_the_dropped_prefixes() -> None:
    assert "langchain_classic" not in allowlist_hint()
    assert "langchain_community" not in allowlist_hint()


#: v0.2's reranker entry, as its config.yaml shipped it (disabled, but defined).
V02_RERANKER = {
    "_target_": "langchain_classic.retrievers.ContextualCompressionRetriever",
    "base_compressor": {
        "_target_": "langchain_classic.retrievers.document_compressors.CrossEncoderReranker",
        "model": {
            "_target_": "langchain_community.cross_encoders.HuggingFaceCrossEncoder",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        "top_n": 3,
    },
}

#: What a refusal under a removed prefix adds: the way to migrate, not to widen the list.
MIGRATION = "left the allowlist in 0.3.0 (S2-4)"


def test_a_v02_config_is_refused_with_the_way_to_migrate(make_config: MakeConfig) -> None:
    # Upgrading from v0.2 meets this first. "Deliberately not configurable" alone could
    # read as an invitation to add the prefixes back (S2-4's second review).
    def v02(c: dict[str, Any]) -> None:
        c["components"]["rerankers"]["cross_encoder"] = V02_RERANKER

    with pytest.raises(ConfigError) as exc:
        load_config(make_config(v02))
    message = str(exc.value)
    print(message)
    assert "components.rerankers.cross_encoder" in message
    assert MIGRATION in message
    assert "delete it" in message
    assert "rag_qa.rerankers.CrossEncoderReranker" in message
    assert "Adding a prefix back is not the fix" in message
    assert message.count(MIGRATION) == 1  # once for the entry, not once per nested target


@pytest.mark.parametrize("target", DROPPED)
def test_an_import_time_refusal_under_a_removed_prefix_names_the_migration(target: str) -> None:
    # build_object and check_imports reach import_from_string without a load.
    with pytest.raises(ImportError, match=r"left the allowlist in 0\.3\.0 \(S2-4\)"):
        import_from_string(target)


@pytest.mark.parametrize(
    "target", ["subprocess.Popen", "builtins.eval", "langchain.chains.LLMChain"]
)
def test_other_refusals_carry_no_migration_note(make_config: MakeConfig, target: str) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(make_config(_set_llm_target(target)))
    assert MIGRATION not in str(exc.value)
    with pytest.raises(ImportError) as imp:
        import_from_string(target)
    assert MIGRATION not in str(imp.value)


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


def test_every_target_in_the_real_config_imports_used_or_not() -> None:
    # The allowlist shrank (S2-4), so every entry must still build, unreferenced ones too:
    # the _cuda devices and the spare LLMs. Before S2-4 the reranker entry was left out,
    # so that langchain_classic was never imported (ADR-007); nothing needs that now.
    config = load_config(REAL_CONFIG)
    every_entry = [
        f"components.{kind}.{name}"
        for kind in Components.model_fields
        for name in getattr(config.components, kind)
    ]
    check_imports(config, every_entry)  # retrievers have no _target_, so add nothing


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


@pytest.mark.parametrize(
    ("target", "refusal"),
    [
        pytest.param("rag_qa.registry.import_module", "re-exports does not count", id="re-export"),
        pytest.param("langchain_core.utils.utils.guard_import", "is not a class", id="function"),
        # Loads, since rag_qa. is allowed: a dropped package reached through our own module.
        pytest.param(
            "rag_qa.vectorstore.FAISS", "re-exports does not count", id="re-export of FAISS"
        ),
    ],
)
def test_check_imports_catches_what_the_load_check_cannot(
    make_config: MakeConfig, target: str, refusal: str
) -> None:
    config = load_config(make_config(_set_llm_target(target)))
    with pytest.raises(ConfigError, match=refusal):
        check_imports(config, [config.pipeline.query.llm])


def test_retriever_settings_have_nothing_to_import() -> None:
    config = load_config(REAL_CONFIG)
    assert config.targets(config.pipeline.query.retriever) == []
