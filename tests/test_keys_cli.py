"""``rag-keys`` (S3-8, DEC-23): make, list and revoke the API's keys.

Run in process through :func:`rag_qa.api.keys_cli.main`; the concurrent writers are threads,
each with a lock file of its own open, as two processes would hold.
"""

import json
import stat
import threading
from pathlib import Path

import pytest

from rag_qa.api import auth, keys_cli
from rag_qa.api.keys_cli import EXIT_CANNOT_RUN, EXIT_NOTHING_DONE, EXIT_OK, main


@pytest.fixture
def keys_file(tmp_path: Path) -> Path:
    return tmp_path / "var" / "api-keys.json"


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    out, err = capsys.readouterr()
    return status, out, err


def test_add_prints_the_token_once_alone_on_stdout_and_keeps_only_its_hash(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out, err = run(
        capsys, "--file", str(keys_file), "add", "laptop", "--scope", "query,ingest"
    )
    assert status == EXIT_OK
    token = out.strip()
    assert out == f"{token}\n" and len(token) == 43  # 32 bytes, URL-safe base64
    assert "shown this once" in err and token not in err
    text = keys_file.read_text()
    assert token not in text
    (key,) = auth.read_keys(keys_file)
    assert (key.name, key.scopes, key.sha256) == ("laptop", {"query", "ingest"}, auth.digest(token))
    assert key.created.endswith("Z")
    # Only its owner reads it: the server's uid (DEC-24's one uid).
    assert stat.S_IMODE(keys_file.stat().st_mode) == 0o600


def test_the_file_comes_from_the_environment_too(
    keys_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_API_KEYS_FILE", str(keys_file))
    assert run(capsys, "add", "bot", "--scope", "query")[0] == EXIT_OK
    assert [key.name for key in auth.read_keys(keys_file)] == ["bot"]


def test_without_a_file_it_cannot_run(capsys: pytest.CaptureFixture[str]) -> None:
    status, _, err = run(capsys, "list")
    assert status == EXIT_CANNOT_RUN
    assert "RAG_API_KEYS_FILE" in err


def test_list_shows_names_scopes_and_dates_and_never_a_hash(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(capsys, "--file", str(keys_file), "add", "laptop", "--scope", "query,ingest")
    run(capsys, "--file", str(keys_file), "add", "bot", "--scope", "query")
    status, out, _ = run(capsys, "--file", str(keys_file), "list")
    assert status == EXIT_OK
    lines = out.splitlines()
    assert lines[0].split() == ["NAME", "SCOPES", "CREATED"]
    assert [line.split()[:2] for line in lines[1:]] == [
        ["laptop", "ingest,query"],
        ["bot", "query"],
    ]
    for key in auth.read_keys(keys_file):
        assert key.sha256.hex() not in out


def test_an_empty_or_missing_file_lists_no_keys(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out, err = run(capsys, "--file", str(keys_file), "list")
    assert (status, out, err.strip()) == (EXIT_OK, "", "No keys.")


def test_revoke_removes_the_key_and_an_unknown_name_is_nothing_done(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(capsys, "--file", str(keys_file), "add", "bot", "--scope", "query")
    run(capsys, "--file", str(keys_file), "add", "ops", "--scope", "ingest")
    inode = keys_file.stat().st_ino
    status, _, err = run(capsys, "--file", str(keys_file), "revoke", "bot")
    assert status == EXIT_OK and "within 5 s" in err
    assert [key.name for key in auth.read_keys(keys_file)] == ["ops"]
    assert keys_file.stat().st_ino != inode  # replaced, not rewritten: the server sees it move
    assert run(capsys, "--file", str(keys_file), "revoke", "bot")[0] == EXIT_NOTHING_DONE


def test_a_taken_name_is_nothing_done_and_the_reserved_one_a_usage_error(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(capsys, "--file", str(keys_file), "add", "bot", "--scope", "query")
    before = keys_file.read_text()
    status, out, err = run(capsys, "--file", str(keys_file), "add", "bot", "--scope", "ingest")
    assert (status, out) == (EXIT_NOTHING_DONE, "")
    assert "exists already" in err
    assert keys_file.read_text() == before
    for name in ("token", "Bot", "-bot", "a" * 65, "with space"):
        with pytest.raises(SystemExit) as usage:
            main(["--file", str(keys_file), "add", name, "--scope", "query"])
        assert usage.value.code == 2, name
    for scopes in ("admin", "query,", "", "query;ingest"):
        with pytest.raises(SystemExit) as usage:
            main(["--file", str(keys_file), "add", "x", "--scope", scopes])
        assert usage.value.code == 2, scopes


def test_a_file_that_does_not_parse_is_left_alone_and_cannot_run(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keys_file.parent.mkdir(parents=True)
    keys_file.write_text("{half")
    for argv in (["add", "bot", "--scope", "query"], ["list"], ["revoke", "bot"]):
        status, out, err = run(capsys, "--file", str(keys_file), *argv)
        assert (status, out) == (EXIT_CANNOT_RUN, ""), argv
        assert "not valid JSON" in err
    assert keys_file.read_text() == "{half"


def test_two_writers_at_once_lose_nothing(keys_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each thread opens the lock file itself, so their flocks conflict as two processes' would.
    # The first writer is held inside the lock until the second has asked for it: without the
    # lock, both would read the empty file and the last write would drop the first key.
    real_read = keys_cli._read
    inside = threading.Event()
    release = threading.Event()

    def slow_read(path: Path) -> list[auth.ApiKey]:
        keys = real_read(path)
        if threading.current_thread().name == "first":
            inside.set()
            assert release.wait(5)
        return keys

    monkeypatch.setattr(keys_cli, "_read", slow_read)
    statuses: dict[str, int] = {}

    def add(name: str) -> None:
        statuses[name] = main(["--file", str(keys_file), "add", name, "--scope", "query"])

    keys_file.parent.mkdir(parents=True)
    first = threading.Thread(target=add, args=("first",), name="first")
    first.start()
    assert inside.wait(5)
    second = threading.Thread(target=add, args=("second",), name="second")
    second.start()
    second.join(0.3)
    assert second.is_alive()  # waiting on the lock, not writing
    release.set()
    first.join(5)
    second.join(5)
    assert statuses == {"first": EXIT_OK, "second": EXIT_OK}
    assert sorted(key.name for key in auth.read_keys(keys_file)) == ["first", "second"]
    assert not list(keys_file.parent.glob(".*.tmp"))  # no temporary file left behind


def test_the_file_is_what_the_server_reads(
    keys_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out, _ = run(capsys, "--file", str(keys_file), "add", "bot", "--scope", "query")
    ring = auth.KeyRing(keys_file, None)
    key = ring.match(out.strip())
    assert key is not None and key.name == "bot"
    assert json.loads(keys_file.read_text())["version"] == auth.FILE_VERSION
