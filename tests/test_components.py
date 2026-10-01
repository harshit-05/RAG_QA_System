"""The shared assembly layer (S1-3): one way to build each component."""

from itertools import pairwise
from pathlib import Path

import pytest
from conftest import MakeConfig
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_qa.components import build_embedder, build_loader, build_splitter
from rag_qa.config import load_config
from rag_qa.loaders import DocxLoader, PdfLoader, TextLoader


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.pdf", PdfLoader),
        ("b.docx", DocxLoader),
        ("c.txt", TextLoader),
        ("d.md", TextLoader),
        ("E.PDF", PdfLoader),  # matched on the lowercased suffix
    ],
)
def test_build_loader_follows_the_extension_map(
    make_config: MakeConfig, tmp_path: Path, filename: str, expected: type
) -> None:
    loader = build_loader(load_config(make_config()), tmp_path / filename)
    assert type(loader) is expected
    assert loader.file_path == str(tmp_path / filename)


@pytest.mark.parametrize("filename", ["image.png", "old.doc", "no_suffix", "archive.tar.gz"])
def test_unmapped_files_get_no_loader(
    make_config: MakeConfig, tmp_path: Path, filename: str
) -> None:
    assert build_loader(load_config(make_config()), tmp_path / filename) is None


def test_loaders_are_built_through_the_allowlist(make_config: MakeConfig, tmp_path: Path) -> None:
    # The old ingest path called import_from_string directly; now loaders go through
    # build_object like everything else. A non-class target shows the checks apply.
    config = load_config(make_config(lambda c: c["components"]["loaders"]["pdf"].update(
        _target_="rag_qa.registry.build_object"
    )))
    with pytest.raises(ImportError, match="not a class"):
        build_loader(config, tmp_path / "a.pdf")


def test_build_embedder_always_reads_the_ingestion_embedder(make_config: MakeConfig) -> None:
    def use_fake_embedder(c: dict) -> None:
        c["components"]["embedders"]["fake"] = {
            "_target_": "langchain_core.embeddings.DeterministicFakeEmbedding", "size": 8
        }
        c["pipeline"]["ingestion"]["embedder"] = "components.embedders.fake"

    embedder = build_embedder(load_config(make_config(use_fake_embedder)))
    assert isinstance(embedder, DeterministicFakeEmbedding)
    assert len(embedder.embed_query("same model for index and query")) == 8


def test_build_splitter_uses_the_configured_splitter(make_config: MakeConfig) -> None:
    # Behaviour, not private fields: config.yaml says chunk_size 1000, overlap 150.
    splitter = build_splitter(load_config(make_config()))
    assert type(splitter).__name__ == "RecursiveCharacterTextSplitter"

    words = " ".join(f"word{i:04d}" for i in range(600))  # ~5,400 characters
    chunks = splitter.split_text(words)
    assert len(chunks) > 1
    assert max(len(chunk) for chunk in chunks) <= 1000
    # Neighbouring chunks share text (the overlap), and no chunk repeats wholesale.
    for left, right in pairwise(chunks):
        assert right.split()[0] in left
        assert left != right
