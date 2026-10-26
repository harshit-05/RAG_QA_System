"""Who may ask: named API keys with scopes, and the one token (DEC-23, S3-8).

**The keys file,** named by ``RAG_API_KEYS_FILE``, is JSON: ``{"version": 1, "keys": [...]}``,
one entry per key with its ``name``, its ``scopes`` (some of ``query`` and ``ingest``), the
sha256 of its token, and when it was ``created``. It never holds a token: ``rag-keys`` prints a
new one once and keeps its hash (:mod:`rag_qa.api.keys_cli`). A token is 32 random bytes, so a
plain sha256 needs no slow hash: nothing can be guessed from it.

**How the server reads it** (:class:`KeyRing`). At most every :data:`RECHECK_S` seconds it
looks at the file's ``(st_ino, st_mtime_ns, st_size)``, and reads it again when they moved:
``rag-keys`` replaces the file atomically, so a new key or a revocation is seen within 5 s.
The file lives in ``var/``, which compose mounts as a folder: a file bind-mounted on its own
keeps its old inode when the host replaces it, and the container would never see a revocation
(DEC-23, design review 2).

**A file that cannot be used fails closed:** missing, unreadable, not JSON or not a keys file,
it holds no key at all until it parses again, and :attr:`KeyRing.problem` says why, in fixed
text that never names the path (it goes over HTTP). ``RAG_API_TOKEN`` keeps working. A file that
does not parse stays closed until it changes; one the operating system would not let be read is
tried again at every look, since nothing is wrong with it (S3-8's first review).

**The check** (:meth:`KeyRing.match`) hashes the presented token and compares the digest with
every key's through ``hmac.compare_digest``, on bytes, with no early exit: every failure takes
the same time and gets the same answer. ``RAG_API_TOKEN`` is the key named ``token``, with
both scopes; a keys file and the token may coexist.

Loads no web framework and no model stack, so ``rag-keys`` starts at once
(``tests/test_architecture.py``).
"""

import hashlib
import hmac
import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import SecretStr

logger = logging.getLogger(__name__)

#: The scopes: ``query`` asks questions and lists the documents, ``ingest`` runs and reads
#: ingest jobs.
QUERY: Final = "query"
INGEST: Final = "ingest"
SCOPES: Final = (QUERY, INGEST)
#: The name ``RAG_API_TOKEN`` goes by, which no key in the file may take.
TOKEN_NAME: Final = "token"
#: A key's name: what logs and spans carry, and what ``rag-keys revoke`` takes.
NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
#: The keys file's format.
FILE_VERSION: Final = 1
#: How often, at most, the server looks at the keys file for a change, in seconds.
RECHECK_S: Final = 5.0


def name_problem(name: str) -> str | None:
    """Why ``name`` cannot name a key, or ``None``."""
    if NAME.fullmatch(name) is None:
        return (
            "a key's name is 1 to 64 lowercase letters, digits, '-' or '_', starting with a "
            "letter or a digit"
        )
    if name == TOKEN_NAME:
        return f"{TOKEN_NAME!r} is the name RAG_API_TOKEN goes by"
    return None


def digest(token: str) -> bytes:
    """The sha256 of a token, as the keys file keeps it (in hex)."""
    return hashlib.sha256(token.encode("utf-8")).digest()


@dataclass(frozen=True)
class ApiKey:
    """One key: its name, its scopes, its token's sha256, and when it was made."""

    name: str
    scopes: frozenset[str]
    sha256: bytes
    created: str

    def entry(self) -> dict[str, Any]:
        """The key as the file holds it."""
        return {
            "name": self.name,
            "scopes": sorted(self.scopes),
            "sha256": self.sha256.hex(),
            "created": self.created,
        }


class KeysFileError(ValueError):
    """The keys file cannot be used. The message is fixed text with no path in it: the API's
    health reports it."""


class KeysFileUnreadableError(KeysFileError):
    """The operating system would not let the file be read (a full table of open files, an I/O
    error). Unlike a file that does not parse, nothing about the file is wrong, so whoever
    reads it tries again at its next look (:meth:`KeyRing.refresh`)."""


def _key(entry: object, at: int) -> ApiKey:
    where = f"the keys file's key {at + 1}"
    if not isinstance(entry, dict) or set(entry) != {"name", "scopes", "sha256", "created"}:
        raise KeysFileError(f"{where} is not a name, scopes, sha256 and created")
    name, scopes, sha256, created = (
        entry["name"],
        entry["scopes"],
        entry["sha256"],
        entry["created"],
    )
    if not isinstance(name, str) or name_problem(name) is not None:
        raise KeysFileError(f"{where} has no valid name")
    if (
        not isinstance(scopes, list)
        or not scopes
        or not all(isinstance(s, str) and s in SCOPES for s in scopes)
    ):
        raise KeysFileError(f"{where} ({name}) has no valid scopes: some of {', '.join(SCOPES)}")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise KeysFileError(f"{where} ({name}) has no valid sha256")
    if not isinstance(created, str):
        raise KeysFileError(f"{where} ({name}) has no valid created")
    return ApiKey(name, frozenset(scopes), bytes.fromhex(sha256), created)


def parse(text: str) -> tuple[ApiKey, ...]:
    """The keys in a keys file's ``text``. Raises :class:`KeysFileError`."""
    try:
        data = json.loads(text)
    except ValueError:
        raise KeysFileError("the keys file is not valid JSON") from None
    if (
        not isinstance(data, dict)
        or data.get("version") != FILE_VERSION
        or set(data) != {"version", "keys"}
    ):
        raise KeysFileError(f"the keys file is not a version {FILE_VERSION} keys file")
    if not isinstance(data["keys"], list):
        raise KeysFileError("the keys file's keys are not a list")
    keys = tuple(_key(entry, at) for at, entry in enumerate(data["keys"]))
    names = [key.name for key in keys]
    if len(set(names)) != len(names):
        raise KeysFileError("the keys file names a key twice")
    return keys


def dump(keys: Iterable[ApiKey]) -> str:
    """A keys file holding ``keys``, in their order."""
    return (
        json.dumps({"version": FILE_VERSION, "keys": [key.entry() for key in keys]}, indent=2)
        + "\n"
    )


def read_keys(path: Path) -> tuple[ApiKey, ...]:
    """The keys in the file at ``path``. Raises :class:`KeysFileError`, a missing file
    included: whoever reads it decides what that means."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise KeysFileError("the keys file does not exist") from None
    except OSError:
        raise KeysFileUnreadableError("the keys file cannot be read") from None
    except UnicodeDecodeError:  # the file's own content: reading it again changes nothing
        raise KeysFileError("the keys file cannot be read") from None
    return parse(text)


#: A file's identity and version, for telling a change: or why it could not be looked at.
Signature = tuple[int, int, int] | str


def _signature(path: Path) -> Signature:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


class KeyRing:
    """The keys the server accepts: the keys file's, read again when it changes, and
    ``RAG_API_TOKEN``'s (the module docstring). ``clock`` is for the tests.

    Used on the event loop alone (the API's async dependencies), so it takes no lock; a look
    at the file is one ``stat``, and a read happens only when it changed.
    """

    def __init__(
        self,
        path: Path | None,
        token: SecretStr | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self._clock = clock
        self._token = (
            None
            if token is None
            else ApiKey(TOKEN_NAME, frozenset(SCOPES), digest(token.get_secret_value()), "")
        )
        self._keys: tuple[ApiKey, ...] = ()
        self._signature: Signature | None = None
        self._looked = -math.inf
        #: Why the keys file holds no key just now, or ``None`` while it is read (or unset).
        self.problem: str | None = None
        self.refresh()

    @property
    def configured(self) -> bool:
        """Whether any credential is configured, a keys file or the token: DEC-23's "no
        credential" is the opposite."""
        return self.path is not None or self._token is not None

    def refresh(self) -> None:
        """Read the keys file again if it changed, looking at most every :data:`RECHECK_S`."""
        if self.path is None:
            return
        now = self._clock()
        if now - self._looked < RECHECK_S:
            return
        self._looked = now
        signature = _signature(self.path)
        if signature == self._signature:
            return
        self._signature = signature
        try:
            keys = read_keys(self.path)
        except KeysFileError as e:
            # A read the system refused says nothing about the file: forget its signature, so
            # the next look reads it again, instead of holding every key closed until the file
            # changes. Said once, not at every look.
            again = isinstance(e, KeysFileUnreadableError)
            repeated = again and self.problem == str(e)
            if again:
                self._signature = None
            self._keys = ()
            self.problem = str(e)
            if not repeated:
                # The server log may name the path; HTTP gets the fixed text alone.
                logger.warning(
                    "The keys file %s cannot be used: %s. Every key is refused with 503 until "
                    "it can%s.",
                    self.path,
                    self.problem,
                    "; RAG_API_TOKEN still works" if self._token is not None else "",
                )
            return
        if self.problem is not None or self._keys != keys:
            count = len(keys)
            logger.info(
                "Read the keys file %s: %d %s.", self.path, count, "key" if count == 1 else "keys"
            )
        self._keys = keys
        self.problem = None

    def match(self, presented: str) -> ApiKey | None:
        """The key whose token was presented, or ``None``. Every key's digest is compared, with
        no early exit, through ``hmac.compare_digest`` on bytes."""
        self.refresh()
        asked = digest(presented)
        found = None
        for key in (self._token, *self._keys):
            if key is not None and hmac.compare_digest(asked, key.sha256):
                found = key
        return found
