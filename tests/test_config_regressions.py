"""Every Phase-0 config bug, pinned (S1-5, ISS-07): the Phase 1 exit criterion,
"CI would have caught every Phase-0 bug", as executable tests.

Each case reproduces the bug's *literal* shape from the original
``v2/config.yaml`` (git ``ee86e1f``; line numbers below), applied to today's real
``config.yaml``, and asserts a ``ConfigError`` whose message names the problem.
Names start with the SRS Appendix A issue ID, so ``pytest -k iss01`` finds one.

Every case fails at **load**, before anything is built, except ISS-03's misspelled
class name. That one is caught by ``check_imports``, **by design**: loading never
imports (DEC-7), and a misspelled class under an allowed prefix passes a string
check. Do not "fix" that by importing at load.

**Every Appendix A issue, and what catches it** (the exit criterion taken
literally; S1-8 checks this table, not the file count). Config bugs are pinned
here; the rest are pinned elsewhere, or say why CI cannot check them:

======  ==================================================================
ISS-01  here (load)
ISS-02  here (load)
ISS-03  here: kind key and meta-package at load, class name by
        ``check_imports``. Its bare model *string* is not caught before build
        (leaf kwargs are open, ADR-010) → Phase 2's reranker story builds it
ISS-04  here (prefix at load); ``test_registry.py`` (all three import checks)
ISS-05  ``test_ingest.py`` (a bad document fails the run; so does a folder)
ISS-06  ``test_cli.py`` (the REPL survives a failing chain)
ISS-07  this suite, gated in CI: ruff, pytest, coverage ≥ 80% (NFR-8)
ISS-08  CI's ``uv sync --locked`` (lockfile drift); pip-audit → S1-8
ISS-09  ``test_repo_hygiene.py`` (nothing build-, index- or media-like tracked)
ISS-10  not a CI check: structural (three copies of the tree), fixed in S0-3
ISS-11  here (load)
ISS-12  not caught yet: ruff has no unreachable-code rule; mypy's
        ``warn_unreachable`` flags code after ``return`` (verified) → S1-6
ISS-13  ``test_loaders.py`` (recursive walk); ``test_ingest.py`` (unreadable
        folders)
ISS-14  a missing feature, not a regression: incremental ingest → Phase 2
ISS-15  Phase 2: the eval harness and its golden dataset (FR-7)
ISS-16  not a CI check: a documented invariant (never open an untrusted index)
ISS-17  ``test_deprecations.py`` (with a negative control)
ISS-18  not in CI: a standalone script (S1-4's call); fixed and verified by hand
ISS-19  mypy → S1-6
ISS-20  not a CI check: the README (S0-6)
ISS-21  partly: the duplicated REPL is gone (one ``cli.py`` since S0-3, its
        loop pinned by ``test_cli.py``). Typos are style, not a CI check. The
        hand-rolled ANSI escapes remain in ``cli.py`` by choice: it is a
        Nitpick, and a terminal library is not worth a dependency
======  ==================================================================
"""

from typing import Any

import pytest
from conftest import MakeConfig

from rag_qa.config import ConfigError, check_imports, load_config


def _fails_at_load(make_config: MakeConfig, edit: Any = None, *, text: str | None = None) -> str:
    path = make_config(edit, text=text) if text is not None else make_config(edit)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    message = str(exc.value)
    print(message)  # with -s: the message a maintainer would see
    assert str(path.resolve()) in message, "the error must name the file"
    return message


def _rename(section: dict[str, Any], old: str, new: str) -> None:
    section[new] = section.pop(old)


# --- ISS-01: the key typo that stopped v2 from starting --------------------------------


def test_iss01_component_kind_typo_llmS_fails_at_load(make_config: MakeConfig) -> None:
    # v2 line 48: `llmS:`, while line 95 referenced `components.llms.qwen2_ollama`.
    message = _fails_at_load(make_config, lambda c: _rename(c["components"], "llms", "llmS"))
    assert "unknown key 'llmS'" in message
    assert "did you mean 'llms'" in message


# --- ISS-02: another machine's home directory, committed ---------------------------------


def test_iss02_absolute_corpus_path_fails_at_load(make_config: MakeConfig) -> None:
    # v2 line 110: data_path: "/home/harshit/RAG_System/docs"
    message = _fails_at_load(
        make_config, lambda c: c["paths"].update(data="/home/harshit/RAG_System/docs")
    )
    assert "absolute path" in message
    assert "relative to the config file" in message


def test_iss02_old_top_level_path_keys_fail_at_load(make_config: MakeConfig) -> None:
    # v2 lines 110-111 kept the paths at top level, outside any `paths:` section.
    def v2_paths(c: dict[str, Any]) -> None:
        c["data_path"] = "/home/harshit/RAG_System/docs"
        c["vector_store_path"] = "/home/harshit/RAG_System/vectorstore/db_faiss"

    message = _fails_at_load(make_config, v2_paths)
    assert "unknown key 'data_path'" in message
    assert "unknown key 'vector_store_path'" in message


# --- ISS-03: the reranker block, which was wrong three ways -------------------------------


def test_iss03_reranker_kind_key_typo_fails_at_load(make_config: MakeConfig) -> None:
    # v2 line 62: `reranker:` (singular) while the pipeline referred to `rerankers`.
    message = _fails_at_load(
        make_config, lambda c: _rename(c["components"], "rerankers", "reranker")
    )
    assert "unknown key 'reranker'" in message
    assert "did you mean 'rerankers'" in message


def test_iss03_reranker_on_the_langchain_meta_package_fails_at_load(make_config: MakeConfig) -> None:
    # v2 line 64 targeted `langchain.retrievers...`: the meta-package, which is not
    # a dependency (DEC-1) and is outside the import allowlist (ISS-04).
    def v2_target(c: dict[str, Any]) -> None:
        c["components"]["rerankers"]["cross_encoder"]["_target_"] = (
            "langchain.retrievers.ContextualCompressionRetriever"
        )

    message = _fails_at_load(make_config, v2_target)
    assert "components.rerankers.cross_encoder" in message
    assert "outside the import allowlist" in message


def test_iss03_misspelled_reranker_class_is_caught_by_check_imports_not_load(
    make_config: MakeConfig,
) -> None:
    # v2 line 66: `CrossEncoderRerank` (no such class; the real one ends in -er).
    # The prefix is allowed, so a load-time string check cannot see it: by design
    # (DEC-7, load never imports). check_imports, run over the referenced
    # components, is what catches it.
    def referenced_typo(c: dict[str, Any]) -> None:
        reranker = c["components"]["rerankers"]["cross_encoder"]
        reranker["base_compressor"]["_target_"] = (
            "langchain_classic.retrievers.document_compressors.CrossEncoderRerank"
        )
        c["pipeline"]["query"]["reranker"] = "components.rerankers.cross_encoder"

    config = load_config(make_config(referenced_typo))  # loads: a string check passes
    with pytest.raises(ConfigError) as exc:
        check_imports(config, config.references().values())
    message = str(exc.value)
    print(message)
    assert "components.rerankers.cross_encoder.base_compressor._target_" in message
    assert "CrossEncoderRerank" in message


# --- ISS-04: config as arbitrary code execution ------------------------------------------


def test_iss04_target_outside_the_allowlist_fails_at_load(make_config: MakeConfig) -> None:
    message = _fails_at_load(
        make_config,
        lambda c: c["components"]["llms"]["mistral_ollama"].update(_target_="os.system"),
    )
    assert "outside the import allowlist" in message
    assert "rag_qa/registry.py" in message  # where the list lives, and that it's fixed


# --- ISS-11: the dead vector_stores block --------------------------------------------------


def test_iss11_dead_vector_stores_block_fails_at_load(make_config: MakeConfig) -> None:
    # v2 lines 42-46: a `vector_stores:` library that nothing read.
    def v2_block(c: dict[str, Any]) -> None:
        c["components"]["vector_stores"] = {
            "faiss": {"path": "vectorstore/db_faiss"},
            "chroma": {"path": "vectorstore/db_chroma"},
        }

    assert "unknown key 'vector_stores'" in _fails_at_load(make_config, v2_block)


def test_iss11_ingestion_vector_store_reference_fails_at_load(make_config: MakeConfig) -> None:
    # v2 line 84: `vector_store: components.vector_stores.faiss` in the ingestion pipeline.
    message = _fails_at_load(
        make_config,
        lambda c: c["pipeline"]["ingestion"].update(vector_store="components.vector_stores.faiss"),
    )
    assert "unknown key 'vector_store'" in message


# --- Phase-0 crashes with raw internal errors (FR-8) -------------------------------------


def test_phase0_empty_path_was_a_pathlib_typeerror(make_config: MakeConfig) -> None:
    # `paths: {data: }` used to raise TypeError from inside pathlib.
    message = _fails_at_load(make_config, lambda c: c["paths"].update(data=None))
    assert "'data' is empty" in message


def test_phase0_empty_yaml_was_an_attributeerror(make_config: MakeConfig) -> None:
    # An empty file used to raise AttributeError: 'NoneType' has no attribute 'get'.
    assert "is empty" in _fails_at_load(make_config, text="")


# --- and the real config passes every check -------------------------------------------------


def test_the_real_config_loads_and_every_referenced_target_imports() -> None:
    """SRS §10's config-resolution test. Referenced targets are import-checked;
    unreferenced ones (the disabled reranker) are prefix-checked by loading only,
    so ``langchain_classic`` is never imported here and ADR-007 holds."""
    from conftest import REAL_CONFIG

    config = load_config(REAL_CONFIG)
    check_imports(config, config.references().values())
    assert "pipeline.query.reranker" not in config.references()  # still disabled
