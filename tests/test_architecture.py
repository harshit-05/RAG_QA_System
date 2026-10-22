"""Architecture rules as tests, so a violation fails CI instead of waiting for review.

* ADR-009: ``rag_qa.ingest`` never imports ``rag_qa.chain`` or ``rag_qa.api``, nor
  ``rag_qa.cli`` the API (§2.2, S2-8), and the config layer
  (``registry``, ``schema``, ``settings``, ``config``) imports no LangChain at all.
  Nor does ``evaluation.dataset``, which CI's model-free eval gate imports (S2-2), nor
  ``manifest``, whose identity functions S2-5's fingerprint reuses there (S2-6).
  Checked at **runtime in a fresh interpreter**, replacing the grep
  ``^(from|import) .*chain``, which false-positives on ``langchain_core`` (S1-3).
  A fresh process is required: inside pytest, other tests have already loaded
  everything.
* ``evaluation.gate`` loads no model stack: it holds the floors and the freshness check
  that ``rag-eval check`` runs in CI's model-free job (ARCHITECTURE.md §2.2), and imports
  the fingerprint, which may bring in ``langchain_core`` and ``rag_qa.chain`` but nothing
  heavier. Nor does ``evaluation.cli``, which that CI step runs (S2-5b): it imports the
  scoring module, whose ragas imports must stay inside functions. Checked the same way,
  in a fresh interpreter.
* ADR-007 / DEC-1: application code never imports ``langchain_classic`` or the
  ``langchain`` meta-package. Checked on the **parsed imports**, so a docstring that
  merely names the package cannot trip it.
* DEC-19 (S3-2): only ``rag_qa.vectorstore`` imports ``qdrant_client``, the store's seam.
  Checked on the parsed imports too, ``from`` and plain, nested modules included.
* DEC-5 (S3-3): no module imports ``langchain_community`` or ``faiss``, again on the parsed
  imports (``langchain_classic`` is the ADR-007 rule above), and a plain ``uv sync`` installs
  none of the three. Only the ``eval`` extra brings the two LangChain packages back, through
  ragas.
* DEC-20 (S3-5): ``rag_qa.sparse``, the BM25 encoder, imports the standard library only, on
  the parsed imports, and loads no LangChain.
"""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "rag_qa"


def modules_loaded_by(module: str) -> set[str]:
    """Every module a fresh interpreter has loaded after importing ``module``."""
    probe = f"import json, sys, {module}; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_ingest_never_loads_the_query_chain() -> None:
    loaded = modules_loaded_by("rag_qa.ingest")
    assert "rag_qa.chain" not in loaded
    assert "rag_qa.cli" not in loaded
    # Nor the HTTP API (ARCHITECTURE.md §2.5): the API imports ingest, never the reverse.
    assert sorted(m for m in loaded if m == "rag_qa.api" or m.startswith("rag_qa.api.")) == []


def test_rag_query_never_loads_the_api_and_does_load_readline() -> None:
    # §2.2: the CLI never imports the API package. The start-up check both use lives in
    # components for that reason (S2-8). readline is imported for its side effect alone:
    # input() gains line editing and history, so nothing else would notice it missing.
    loaded = modules_loaded_by("rag_qa.cli")
    assert sorted(m for m in loaded if m == "rag_qa.api" or m.startswith("rag_qa.api.")) == []
    assert "readline" in loaded


@pytest.mark.parametrize(
    "module",
    [
        "rag_qa.registry",
        "rag_qa.schema",
        "rag_qa.settings",
        "rag_qa.config",
        "rag_qa.evaluation.dataset",
        "rag_qa.manifest",
        "rag_qa.sparse",
    ],
)
def test_the_config_layer_loads_no_langchain(module: str) -> None:
    # What lets config validation (and its tests) run without the ML stack. The golden
    # set's loader is held to the same rule: the tier-2 gate runs it with no models. So
    # is the manifest, whose identities S2-5's fingerprint computes in that same gate.
    langchain = sorted(m for m in modules_loaded_by(module) if m.startswith("langchain"))
    assert langchain == []


@pytest.mark.parametrize("module", ["rag_qa.evaluation.gate", "rag_qa.evaluation.cli"])
def test_the_gate_loads_without_the_model_stack(module: str) -> None:
    # The fingerprint may bring in langchain_core, but never any of these: `rag-eval check`
    # must stay a CPU-only, model-free step, and CI never installs the eval extra.
    heavy = {
        "ragas",
        "openai",
        "instructor",
        "torch",
        "sentence_transformers",
        "langchain_huggingface",
    }
    loaded = modules_loaded_by(module)
    assert sorted(m for m in loaded if m.split(".")[0] in heavy) == []


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _legacy_imports(path: Path) -> set[str]:
    return {
        name
        for name in _imported_modules(path)
        if name == "langchain" or name.startswith(("langchain.", "langchain_classic"))
    }


def application_files(root: Path) -> list[Path]:
    """Every Python file of the package under ``root``, subpackages included."""
    return sorted(root.rglob("*.py"))


@pytest.mark.parametrize(
    "path", application_files(SRC), ids=lambda p: p.relative_to(SRC).as_posix()
)
def test_no_legacy_langchain_imports_in_application_code(path: Path) -> None:
    forbidden = _legacy_imports(path)
    assert forbidden == set(), f"{path.name} imports {sorted(forbidden)} (ADR-007, DEC-1)"


def test_the_legacy_import_scan_reaches_subpackages(tmp_path: Path) -> None:
    # Phase 2's API will likely be a subpackage (rag_qa/api/); a top-level-only
    # scan would never read it.
    (tmp_path / "api").mkdir()
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "api" / "routes.py").write_text("from langchain_classic.chains import X\n")
    flagged = {
        p.relative_to(tmp_path).as_posix()
        for p in application_files(tmp_path)
        if _legacy_imports(p)
    }
    assert flagged == {"api/routes.py"}


def _qdrant_imports(path: Path) -> set[str]:
    return {
        name
        for name in _imported_modules(path)
        if name == "qdrant_client" or name.startswith("qdrant_client.")
    }


@pytest.mark.parametrize(
    "path", application_files(SRC), ids=lambda p: p.relative_to(SRC).as_posix()
)
def test_only_the_store_imports_the_qdrant_client(path: Path) -> None:
    # DEC-19: every Qdrant call, and every error it can raise, goes through one module, so
    # the store can change (DEC-3's pgvector, a cluster) behind a seam of one file.
    found = _qdrant_imports(path)
    if path.relative_to(SRC).as_posix() == "vectorstore.py":
        assert found, "vectorstore.py no longer imports qdrant_client: is the scan blind?"
    else:
        assert found == set(), f"{path.name} imports {sorted(found)}; only vectorstore may"


def test_the_qdrant_import_scan_catches_both_forms(tmp_path: Path) -> None:
    # A negative control: a plain import and a from-import of a submodule both count.
    (tmp_path / "a.py").write_text("import qdrant_client\n")
    (tmp_path / "b.py").write_text("from qdrant_client.http import models\n")
    (tmp_path / "c.py").write_text('"""Mentions qdrant_client in prose only."""\n')
    assert [bool(_qdrant_imports(tmp_path / n)) for n in ("a.py", "b.py", "c.py")] == [
        True,
        True,
        False,
    ]


#: The packages DEC-5's exit took out of our dependencies (S3-3), by import name.
#: ``langchain_classic`` is only in the install check: the ADR-007 scan above covers its imports.
DROPPED_PACKAGES = ("langchain_community", "faiss", "langchain_classic")
IMPORT_BANNED = ("langchain_community", "faiss")


def _dropped_imports(path: Path) -> set[str]:
    return {name for name in _imported_modules(path) if name.split(".")[0] in IMPORT_BANNED}


@pytest.mark.parametrize(
    "path", application_files(SRC), ids=lambda p: p.relative_to(SRC).as_posix()
)
def test_no_module_imports_a_package_dec_5_removed(path: Path) -> None:
    # The eval extra still installs langchain_community (ragas needs it), so there an
    # import would work: this scan, not the environment, is what keeps it out.
    found = _dropped_imports(path)
    assert found == set(), f"{path.name} imports {sorted(found)}, which S3-3 removed (DEC-5)"


def test_the_removed_package_scan_catches_both_forms(tmp_path: Path) -> None:
    # A negative control: a plain import and a from-import of a submodule both count.
    (tmp_path / "a.py").write_text("import faiss\n")
    (tmp_path / "b.py").write_text("from langchain_community.vectorstores import FAISS\n")
    (tmp_path / "c.py").write_text('"""Mentions langchain_community in prose only."""\n')
    assert [bool(_dropped_imports(tmp_path / n)) for n in ("a.py", "b.py", "c.py")] == [
        True,
        True,
        False,
    ]


def test_a_plain_sync_installs_none_of_the_removed_packages() -> None:
    installed = {name for name in DROPPED_PACKAGES if importlib.util.find_spec(name)}
    if importlib.util.find_spec("ragas"):
        # The eval extra: ragas brings the two LangChain packages back. FAISS stays out.
        installed -= {"langchain_community", "langchain_classic"}
    assert installed == set()


def test_the_bm25_encoder_imports_the_standard_library_only() -> None:
    # DEC-20: no dependency for the sparse vectors, so none whose release could move them.
    # The manifest and the store import it; it imports nothing of ours or anyone's.
    found = _imported_modules(SRC / "sparse.py")
    outside = sorted(name for name in found if name.split(".")[0] not in sys.stdlib_module_names)
    assert found, "sparse.py imports nothing: is the scan blind?"
    assert outside == [], f"sparse.py imports {outside}: the standard library only (DEC-20)"
