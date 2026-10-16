"""What a tier-2 run has finished so far, kept on disk so that an interrupted run resumes.

A full scoring is hours long, 13–18 h on this host's CPU, and generating is one to two.
Without this, one transport error near the end loses the whole run (S2-5's first review).
So ``rag-eval generate`` and ``score`` append each finished item to a checkpoint beside
the run file they will write, and a re-run of the same command skips what is in it. The
run file itself is still written once, at the end, atomically (:func:`.gate.write_run`),
and the checkpoint is deleted after it.

**A checkpoint resumes only the run that wrote it.** Its first line records the run's
*identity*: everything that decides an item's result and is known before the first item,
such as the fingerprint parts and the models' digests. A re-run with any of it changed is
refused and told to delete the file, so no run file ever mixes items from two setups.
The run's ``--limit`` is part of it too, so a smoke run into the same ``--out`` neither
resumes from a full run's checkpoint nor deletes it when it finishes.

The format is JSON Lines: the header ``{"kind", "identity", "started_at"}``, then one
record per finished item. Each line is flushed and ``fsync``'d as it is written. A crash
mid-write can tear only the last line, which a resume cuts from the file; a bad line
anywhere else means the file is not what this wrote, and is refused.

Light, like :mod:`.gate`: the standard library and pydantic.
"""

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

#: The checkpoint's name is the run file's, with this in place of its suffix.
SUFFIX = ".partial.jsonl"

#: The command whose run each kind of checkpoint holds, for messages.
COMMANDS = {"answers": "generate", "generation": "score"}


class CheckpointError(Exception):
    """The checkpoint is unreadable, or is of another run: the message says what to do."""


def checkpoint_path(out: Path) -> Path:
    """Where the checkpoint of a run written to ``out`` lives: beside it, so a run whose
    ``--out`` is on a mounted drive keeps its checkpoint there too."""
    return out.with_name(out.stem + SUFFIX)


def discard(out: Path) -> None:
    """Delete the checkpoint of the run written to ``out``, once that run file is written."""
    checkpoint_path(out).unlink(missing_ok=True)


@dataclass(frozen=True)
class Resumed:
    """What a checkpoint holds: when its run started, and the records of finished items."""

    started_at: str
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class Checkpoint:
    """The checkpoint at ``path`` of a run of ``kind`` (``answers`` or ``generation``) and
    ``identity`` (JSON-able, compared exactly)."""

    path: Path
    kind: str
    identity: Mapping[str, Any]

    def load(self) -> Resumed | None:
        """The finished items, or ``None`` when there is no checkpoint. Raises
        :class:`CheckpointError` for one of another run, or one that is not readable."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            raise CheckpointError(f"cannot read the checkpoint {self.path}: {e.strerror}") from e
        lines = text.split("\n")
        # A crash mid-write tears only the last line, which then has no newline yet.
        if lines[-1] == "" or not self._parses(lines[-1]):
            lines.pop()
        try:
            header, *records = [json.loads(line) for line in lines]
        except ValueError:  # an empty file, or a bad line before the last
            raise CheckpointError(
                self._unusable("it is not a checkpoint this version writes")
            ) from None
        if not isinstance(header, dict) or header.get("kind") != self.kind:
            got = header.get("kind") if isinstance(header, dict) else None
            raise CheckpointError(
                self._unusable(
                    f"it is from rag-eval {COMMANDS.get(str(got), got)}, not "
                    f"{COMMANDS.get(self.kind, self.kind)}"
                )
            )
        recorded = header.get("identity")
        recorded = recorded if isinstance(recorded, dict) else {}
        wanted = json.loads(json.dumps(self.identity))  # as it reads back: tuples are lists
        differ = sorted(key for key in {*wanted, *recorded} if wanted.get(key) != recorded.get(key))
        if differ:
            raise CheckpointError(
                self._unusable(f"it is of a run with another {', '.join(differ)}")
            )
        intact = "".join(line + "\n" for line in lines)
        if text != intact:
            # Cut the torn line from the file too: the next append would otherwise run on
            # from it, and leave a bad line before the last (S2-5b's review).
            self._replace(intact)
        return Resumed(started_at=str(header.get("started_at")), records=records)

    def _replace(self, text: str) -> None:
        """Rewrite the checkpoint as ``text``, atomically, as :func:`.gate.write_run` does."""
        fd, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise

    def append(self, record: Mapping[str, Any], *, started_at: str) -> None:
        """Add one finished item, writing the header first when the file is new."""
        lines = []
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = {"kind": self.kind, "identity": self.identity, "started_at": started_at}
            lines.append(json.dumps(header))
        lines.append(json.dumps(record))
        with self.path.open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
            f.flush()
            os.fsync(f.fileno())

    def restore[ModelT: BaseModel](self, model: type[ModelT], data: Any) -> ModelT:
        """One recorded item as ``model``. Raises :class:`CheckpointError` for a record
        this version would not have written."""
        try:
            return model.model_validate(data)
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            raise CheckpointError(
                self._unusable(
                    f"a recorded item is not one this version writes ({where}: {first['msg']})"
                )
            ) from None

    @staticmethod
    def _parses(line: str) -> bool:
        try:
            json.loads(line)
        except ValueError:
            return False
        return True

    def _unusable(self, why: str) -> str:
        return (
            f"the checkpoint {self.path} cannot be resumed: {why}. Delete it to start this run over"
        )
