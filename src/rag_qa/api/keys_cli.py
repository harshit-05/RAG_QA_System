"""``rag-keys``: make, list and revoke the API's keys (DEC-23, S3-8).

- ``rag-keys add NAME --scope query[,ingest]`` makes a key: a token of 32 random bytes, printed
  once, alone on stdout, and its sha256 kept in the keys file. The token is never stored, so
  a lost one is revoked and replaced, never recovered.
- ``rag-keys list`` shows each key's name, scopes and creation time; never a hash.
- ``rag-keys revoke NAME`` removes a key. A running server refuses its token within 5 s.

The file is ``--file``, or ``RAG_API_KEYS_FILE``, which the server reads too; it belongs in
``var/`` (:mod:`rag_qa.api.auth` says why). **Every change is atomic and serialized:** it
holds a ``flock`` on ``<file>.lock`` while it reads the file and writes the new one, so two
``rag-keys`` at once lose nothing, and the new file is written beside the old, fsynced, then
put in its place by ``os.replace`` (and the folder fsynced), so a server never reads half a
file. The file is created with mode 0600: only its owner, the server's uid, reads it.

``rag-keys`` exit codes: **0** done; **1** nothing done: ``add`` of a name already taken, or
``revoke`` of a name the file does not hold; **2** it cannot run: no file named, a file that
does not parse, a file or lock that cannot be written, or a usage error.

Loads no web framework and no model stack (``tests/test_architecture.py``).
"""

import argparse
import fcntl
import os
import secrets
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from rag_qa.api.auth import (
    SCOPES,
    ApiKey,
    KeysFileError,
    digest,
    dump,
    name_problem,
    read_keys,
)
from rag_qa.settings import ENV_API_KEYS_FILE, KeysFileSettings, describe_settings_error

EXIT_OK = 0
EXIT_NOTHING_DONE = 1
EXIT_CANNOT_RUN = 2


def new_token() -> str:
    """A new key's token: 32 random bytes, URL-safe base64, 43 visible ASCII characters."""
    return secrets.token_urlsafe(32)


def _name(value: str) -> str:
    problem = name_problem(value)
    if problem is not None:
        raise argparse.ArgumentTypeError(problem)
    return value


def _scopes(value: str) -> frozenset[str]:
    scopes = frozenset(part.strip() for part in value.split(","))
    if not scopes <= set(SCOPES) or "" in scopes:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a list of scopes: some of {', '.join(SCOPES)}, separated by commas"
        )
    return scopes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-keys",
        description="Make, list and revoke the HTTP API's keys. A key's token is printed "
        "once, when it is made; the keys file keeps only its hash.",
        epilog="Exit status: 0 done; 1 nothing done (a name already taken, or one the file "
        "does not hold); 2 cannot run (no file named, a file that does not parse or cannot be "
        "written, or a usage error).",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        type=Path,
        help=f"the keys file (default: ${ENV_API_KEYS_FILE}); keep it in var/",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    add = commands.add_parser("add", help="make a key and print its token, once")
    add.add_argument("name", type=_name, help="the key's name, as logs and spans show it")
    add.add_argument(
        "--scope",
        required=True,
        type=_scopes,
        metavar="SCOPES",
        help=f"what the key may do: some of {', '.join(SCOPES)}, separated by commas",
    )
    commands.add_parser("list", help="show each key's name, scopes and creation time")
    revoke = commands.add_parser("revoke", help="remove a key; servers refuse it within 5 s")
    revoke.add_argument("name", type=_name)
    return parser


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold the keys file's lock, ``<file>.lock``, waiting for another ``rag-keys``."""
    lock_path = path.with_name(f"{path.name}.lock")
    with open(lock_path, "a") as lock:  # "a": created if missing, never truncated
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _read(path: Path) -> list[ApiKey]:
    """The file's keys; none when it does not exist yet. Raises :class:`KeysFileError`."""
    if not path.exists():
        return []
    return list(read_keys(path))


def _write(path: Path, keys: Sequence[ApiKey]) -> None:
    """Replace the keys file with one holding ``keys``, atomically (the module docstring)."""
    # mkstemp creates the file with mode 0600, in the file's own folder, so os.replace never
    # crosses a filesystem.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(dump(keys))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    folder = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(folder)
    finally:
        os.close(folder)


def _add(path: Path, name: str, scopes: frozenset[str]) -> int:
    with _locked(path):
        keys = _read(path)
        if any(key.name == name for key in keys):
            print(
                f"Error: a key named {name!r} exists already. Revoke it first to replace it.",
                file=sys.stderr,
            )
            return EXIT_NOTHING_DONE
        token = new_token()
        created = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        _write(path, [*keys, ApiKey(name, scopes, digest(token), created)])
    print(token)
    print(
        f"Added key {name!r} ({', '.join(sorted(scopes))}). Its token is above, shown this "
        f"once: only its hash is kept. Send it as Authorization: Bearer <token>.",
        file=sys.stderr,
    )
    return EXIT_OK


def _list(path: Path) -> int:
    keys = _read(path)
    if not keys:
        print("No keys.", file=sys.stderr)
        return EXIT_OK
    width = max(len(key.name) for key in keys)
    print(f"{'NAME':<{width}}  {'SCOPES':<12}  CREATED")
    for key in keys:
        print(f"{key.name:<{width}}  {','.join(sorted(key.scopes)):<12}  {key.created}")
    return EXIT_OK


def _revoke(path: Path, name: str) -> int:
    with _locked(path):
        keys = _read(path)
        kept = [key for key in keys if key.name != name]
        if len(kept) == len(keys):
            print(f"Error: no key named {name!r}.", file=sys.stderr)
            return EXIT_NOTHING_DONE
        _write(path, kept)
    print(f"Revoked key {name!r}. A running server refuses it within 5 s.", file=sys.stderr)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-keys``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)
    path: Path | None = args.file
    if path is None:
        try:
            path = KeysFileSettings().keys_file
        except ValidationError as e:
            print(f"Error: invalid settings:\n{describe_settings_error(e)}", file=sys.stderr)
            return EXIT_CANNOT_RUN
    if path is None:
        print(
            f"Error: no keys file: pass --file PATH, or set {ENV_API_KEYS_FILE} (for example "
            f"to $PWD/var/api-keys.json).",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    try:
        if args.command == "add":
            path.parent.mkdir(parents=True, exist_ok=True)
            return _add(path, args.name, args.scope)
        if args.command == "list":
            return _list(path)
        return _revoke(path, args.name)
    except KeysFileError as e:
        print(
            f"Error: {e} ({path}): fix it, or move it aside and add the keys again.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    except OSError as e:
        print(f"Error: cannot use the keys file {path}: {e.strerror or e}", file=sys.stderr)
        return EXIT_CANNOT_RUN


if __name__ == "__main__":
    sys.exit(main())
