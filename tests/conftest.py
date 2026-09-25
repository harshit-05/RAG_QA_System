"""Shared fixtures. The suite is hermetic: no network, no Ollama, no model download."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_qa.settings import ENV_CONFIG, ENV_DATA_PATH, ENV_VECTOR_STORE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "config.yaml"

MakeConfig = Callable[..., Path]


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported RAG_* variables must not leak into any test."""
    for name in (ENV_CONFIG, ENV_DATA_PATH, ENV_VECTOR_STORE_PATH):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_config(tmp_path: Path) -> MakeConfig:
    """Write a scratch config and return its path.

    By default it is the repo's real ``config.yaml``, optionally changed by ``edit``
    (a function mutating the loaded dict), so every malformed case is one small change
    to the real file rather than a hand-written config that drifts from it. ``text``
    writes raw content instead, for cases YAML can't express as a dict.
    """

    def make(edit: Callable[[dict[str, Any]], None] | None = None, *, text: str | None = None) -> Path:
        path = tmp_path / "config.yaml"
        if text is not None:
            path.write_text(text)
            return path
        data = yaml.safe_load(REAL_CONFIG.read_text())
        if edit is not None:
            edit(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return make
