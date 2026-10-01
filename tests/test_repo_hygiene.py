"""ISS-09 as a test: build artifacts, indexes and media never get tracked again.

``.gitignore`` (S0-1) only keeps *untracked* files out; a ``git add -f`` or a file
committed before an ignore rule existed stays tracked, which is exactly how ISS-09
happened. So the guard reads what git actually tracks. It runs in the suite, not
as a shell step in ``ci.yml``, so it also runs locally and has a negative control.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import REPO_ROOT

#: What ISS-09 had committed, by kind: bytecode, the FAISS index, a screen
#: recording, an editor backup, and parquet data no loader reads. The corpus PDFs
#: are tracked on purpose (a fresh clone must be able to ingest, S0-6).
ARTIFACT = re.compile(
    r"(^|/)(__pycache__|vectorstore)/"
    r"|\.(pyc|faiss|pkl|parquet|webm|mp4|mkv|save)$"
)


def tracked_artifacts(repo: Path) -> list[str]:
    """Every path git tracks in ``repo`` that matches :data:`ARTIFACT`."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=True, timeout=30
    ).stdout.decode()
    return [path for path in listed.split("\0") if path and ARTIFACT.search(path)]


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git checkout")
def test_no_build_artifacts_index_or_media_are_tracked() -> None:
    assert tracked_artifacts(REPO_ROOT) == []


@pytest.mark.parametrize(
    ("path", "artifact"),
    [
        ("vectorstore/db_faiss/index.faiss", True),
        ("vectorstore/db_faiss/index.pkl", True),
        ("src/rag_qa/__pycache__/chain.cpython-312.pyc", True),
        ("corpus/train.parquet", True),
        ("Screencast from 30-07-25 04_25_00 PM IST.webm", True),
        ("v1/docx_processor.py.save", True),
        # Lookalikes that must stay allowed:
        ("src/rag_qa/vectorstore.py", False),  # the module, not the index folder
        ("corpus/2412.14140v2.pdf", False),    # corpus PDFs are tracked on purpose
        ("tests/test_registry.py", False),
    ],
)
def test_the_pattern_flags_iss09s_files_and_not_their_lookalikes(path: str, artifact: bool) -> None:
    assert bool(ARTIFACT.search(path)) is artifact


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_negative_control_a_force_added_index_is_caught(tmp_path: Path) -> None:
    # The ISS-09 path: .gitignore would have stopped a plain `git add`, but not -f.
    # A scratch repository of the test's own, never the real one.
    (tmp_path / ".gitignore").write_text("vectorstore/\n")
    (tmp_path / "vectorstore" / "db_faiss").mkdir(parents=True)
    (tmp_path / "vectorstore" / "db_faiss" / "index.faiss").write_bytes(b"\x00")
    (tmp_path / "vectorstore.py").write_text("")
    git = ["git", "-c", "init.defaultBranch=main"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True, timeout=30)
    subprocess.run([*git, "add", "-f", "."], cwd=tmp_path, check=True, timeout=30)

    assert tracked_artifacts(tmp_path) == ["vectorstore/db_faiss/index.faiss"]
