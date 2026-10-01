"""Architecture rules as tests, so a violation fails CI instead of waiting for review.

* ADR-009: ``rag_qa.ingest`` never imports ``rag_qa.chain``, and the config layer
  (``registry``, ``schema``, ``settings``, ``config``) imports no LangChain at all.
  Checked at **runtime in a fresh interpreter**, replacing the grep
  ``^(from|import) .*chain``, which false-positives on ``langchain_core`` (S1-3).
  A fresh process is required: inside pytest, other tests have already loaded
  everything.
* ADR-007 / DEC-1: application code never imports ``langchain_classic`` or the
  ``langchain`` meta-package. Checked on the **parsed imports**, so a docstring that
  merely names the package cannot trip it.
"""

import ast
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


@pytest.mark.parametrize(
    "module", ["rag_qa.registry", "rag_qa.schema", "rag_qa.settings", "rag_qa.config"]
)
def test_the_config_layer_loads_no_langchain(module: str) -> None:
    # What lets config validation (and its tests) run without the ML stack.
    langchain = sorted(m for m in modules_loaded_by(module) if m.startswith("langchain"))
    assert langchain == []


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
        p.relative_to(tmp_path).as_posix() for p in application_files(tmp_path) if _legacy_imports(p)
    }
    assert flagged == {"api/routes.py"}
