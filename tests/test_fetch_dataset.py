"""``scripts/fetch_dataset.py`` (S1-4, S2-8): where it refuses to write, and what.

The network is faked: ``requests.get`` returns a canned body, so nothing is downloaded.
The script is not a package module, so it is loaded from its path.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import requests
from conftest import REPO_ROOT, MakeConfig

from rag_qa.settings import ENV_CONFIG, ENV_DATA_PATH

PARQUET = b"PAR1" + b"\x00" * 12 + b"PAR1"  # the magic at both ends, as a real file has
HTML = b"<!DOCTYPE html><html><head><title>Hugging Face</title>"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fetch_dataset", REPO_ROOT / "scripts" / "fetch_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_dataset = load_script()


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


@dataclass
class Network:
    """The fake network: what it serves, and the URLs asked for."""

    body: bytes = PARQUET
    requested: list[str] = field(default_factory=list)


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> Network:
    """Serve ``Network.body`` for every ``requests.get``. The preview is skipped, as if
    ``datasets`` were not installed, so a fake body is never parsed."""
    net = Network()

    def get(url: str, timeout: float) -> FakeResponse:
        net.requested.append(url)
        return FakeResponse(net.body)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setitem(sys.modules, "datasets", None)  # its import fails: no preview
    return net


def test_a_parquet_download_is_written(network: Network, tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "downloads"
    assert fetch_dataset.main(["--out", str(out)]) == 0
    assert (out / "train.parquet").read_bytes() == PARQUET
    assert network.requested == [fetch_dataset.URL]
    assert "Preview skipped" in capsys.readouterr().out


def test_a_download_that_is_not_parquet_is_not_written(
    network: Network, tmp_path: Path, capsys: Any
) -> None:
    # The pre-S1-4 /blob/ URL served Hugging Face's HTML page with a 200, and it was saved
    # as train.parquet.
    network.body = HTML
    out = tmp_path / "downloads"
    assert fetch_dataset.main(["--out", str(out)]) == 1
    assert not (out / "train.parquet").exists()
    message = capsys.readouterr().out
    assert "not a parquet file" in message
    assert "<!DOCTYPE html>" in message


@pytest.mark.parametrize("inside", ["", "sub/folder"])
def test_it_refuses_the_repos_corpus(network: Network, inside: str, capsys: Any) -> None:
    out = fetch_dataset.CORPUS_DIR / inside
    assert fetch_dataset.main(["--out", str(out)]) == 2
    assert "Refusing to write into" in capsys.readouterr().out
    assert network.requested == []


def test_it_refuses_rag_data_path(
    network: Network, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    corpus = tmp_path / "my-corpus"
    corpus.mkdir()
    monkeypatch.setenv(ENV_DATA_PATH, str(corpus))
    monkeypatch.chdir(REPO_ROOT)  # where the script finds config.yaml
    assert fetch_dataset.main(["--out", str(corpus / "downloads")]) == 2
    assert f"Refusing to write into {corpus}" in capsys.readouterr().out
    assert network.requested == []
    assert not (corpus / "downloads").exists()


def test_it_refuses_rag_data_path_even_when_the_config_cannot_load(
    network: Network, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, "my-corpus")  # relative: anchored to the working dir
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    assert fetch_dataset.main(["--out", str(tmp_path / "my-corpus")]) == 2
    assert "Refusing to write into" in capsys.readouterr().out
    assert network.requested == []


def test_it_refuses_the_configs_paths_data(
    network: Network,
    make_config: MakeConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    # A config of its own, selected by RAG_CONFIG, whose corpus is not the repo's.
    path = make_config(lambda c: c.setdefault("paths", {}).update(data="elsewhere"))
    monkeypatch.setenv(ENV_CONFIG, str(path))
    assert fetch_dataset.main(["--out", str(tmp_path / "elsewhere")]) == 2
    assert f"Refusing to write into {tmp_path / 'elsewhere'}" in capsys.readouterr().out
    assert network.requested == []
