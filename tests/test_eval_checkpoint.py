"""A tier-2 run's checkpoint (S2-5b): what an interrupted run resumes from."""

import json
from pathlib import Path

import pytest

from rag_qa.evaluation.checkpoint import (
    Checkpoint,
    CheckpointError,
    checkpoint_path,
    discard,
)
from rag_qa.evaluation.gate import Generator

IDENTITY = {"fingerprint": {"query": "q1"}, "digests": ("a", "b")}


@pytest.fixture
def store(tmp_path: Path) -> Checkpoint:
    return Checkpoint(checkpoint_path(tmp_path / "runs" / "answers.json"), "answers", IDENTITY)


def test_it_sits_beside_the_run_file_it_leads_to(tmp_path: Path) -> None:
    out = tmp_path / "drive" / "scoring-1.json"
    assert checkpoint_path(out) == tmp_path / "drive" / "scoring-1.partial.jsonl"


def test_no_checkpoint_means_a_fresh_run(store: Checkpoint) -> None:
    assert store.load() is None


def test_what_was_appended_is_what_resumes(store: Checkpoint) -> None:
    store.append({"id": "one"}, started_at="2026-10-07T10:00:00Z")
    store.append({"id": "two"}, started_at="ignored once the header is written")
    resumed = store.load()
    assert resumed is not None
    assert resumed.started_at == "2026-10-07T10:00:00Z"
    assert resumed.records == [{"id": "one"}, {"id": "two"}]
    # One header, then a line per item: tuples in the identity compare as the lists
    # they read back as.
    assert len(store.path.read_text().splitlines()) == 3


def test_a_line_torn_by_a_crash_is_cut_from_the_file(store: Checkpoint) -> None:
    store.append({"id": "one"}, started_at="t")
    with store.path.open("a") as f:
        f.write('{"id": "tw')  # the process died mid-write
    resumed = store.load()
    assert resumed is not None and resumed.records == [{"id": "one"}]
    # From the file too: what is appended next must not run on from the torn line, or a
    # second interruption would find a bad line before the last (S2-5b's review).
    store.append({"id": "two"}, started_at="t")
    resumed = store.load()
    assert resumed is not None and resumed.records == [{"id": "one"}, {"id": "two"}]


def test_a_whole_last_line_without_its_newline_is_kept(store: Checkpoint) -> None:
    store.append({"id": "one"}, started_at="t")
    with store.path.open("a") as f:
        f.write('{"id": "two"}')  # every byte but the newline
    resumed = store.load()
    assert resumed is not None and resumed.records == [{"id": "one"}, {"id": "two"}]
    assert store.path.read_text().endswith('{"id": "two"}\n')
    assert list(store.path.parent.glob("*.tmp")) == []


def test_a_bad_line_before_the_last_is_refused(store: Checkpoint) -> None:
    store.append({"id": "one"}, started_at="t")
    with store.path.open("a") as f:
        f.write("not json\n" + json.dumps({"id": "three"}) + "\n")
    with pytest.raises(CheckpointError, match="not a checkpoint this version writes"):
        store.load()


def test_an_empty_file_is_refused(store: Checkpoint) -> None:
    store.path.parent.mkdir(parents=True)
    store.path.write_text("")
    with pytest.raises(CheckpointError, match="Delete it to start this run over"):
        store.load()


def test_a_checkpoint_of_another_setup_is_refused_by_name(store: Checkpoint) -> None:
    store.append({"id": "one"}, started_at="t")
    other = Checkpoint(store.path, "answers", {**IDENTITY, "fingerprint": {"query": "q2"}})
    with pytest.raises(CheckpointError) as exc:
        other.load()
    message = str(exc.value)
    assert "another fingerprint" in message and "digests" not in message
    assert str(store.path) in message and "Delete it" in message


def test_a_checkpoint_of_another_kind_of_run_is_refused(store: Checkpoint) -> None:
    store.append({"id": "one"}, started_at="t")
    with pytest.raises(CheckpointError, match="it is from rag-eval generate, not score"):
        Checkpoint(store.path, "generation", IDENTITY).load()


def test_a_record_this_version_would_not_write_is_refused(store: Checkpoint) -> None:
    with pytest.raises(CheckpointError, match="a recorded item is not one this version writes"):
        store.restore(Generator, {"ref": "x", "model": None})
    restored = store.restore(Generator, {"ref": "x", "model": None, "digest": None})
    assert restored == Generator(ref="x", model=None, digest=None)


def test_discard_deletes_it_and_tolerates_none(store: Checkpoint, tmp_path: Path) -> None:
    out = tmp_path / "runs" / "answers.json"
    store.append({"id": "one"}, started_at="t")
    discard(out)
    assert not store.path.exists()
    discard(out)  # nothing left: no error
