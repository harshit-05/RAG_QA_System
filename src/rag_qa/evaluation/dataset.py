"""The golden set: what a valid ``eval/eval_dataset.jsonl`` is, and how it is loaded.

SRS §7.4, schema in ARCHITECTURE.md §2.4. One JSON object per line:

* ``id``: unique within the file;
* ``question`` and ``ground_truth``;
* ``answerable``: ``false`` marks a question the corpus cannot answer. Its
  ``ground_truth`` is the system prompt's refusal sentence, and it counts only towards
  the decline rate (DEC-15);
* ``expected_sources``: the corpus-relative file and the printed page labels the ground
  truth rests on. Required when answerable, forbidden otherwise. Tier 1 scores
  retrieval against it (S2-3);
* ``must_not_contain``: strings whose appearance in an answer would show the model
  answering from prior knowledge, e.g. ``["Canberra"]``. Unanswerable records only;
* ``notes``: free text for the human check, typically the passage quoted.

Both eval gates read this, and the S2-5 gate runs in CI's model-free job, so this
module is pure Python by contract: no LangChain, as ``tests/test_architecture.py``
checks.

A wrong record silently lowers the score of a correct answer, or raises a wrong one,
in every later run. So the loader rejects a malformed file outright, and names every
bad line rather than stopping at the first.
"""

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class GoldenSetError(ValueError):
    """The golden set file is malformed. The message lists each problem with its line."""


class _Frozen(BaseModel):
    # strict: a hand-edited file should say `true`, not `"true"` or `1`. Validated from
    # JSON text, where strict mode still accepts an array for a tuple.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ExpectedSource(_Frozen):
    """One file the ground truth rests on, and its pages there."""

    #: Relative to the corpus root (``paths.data``), with ``/`` separators.
    source: str = Field(min_length=1)
    #: Printed page labels, as ``citation()`` shows them: ``"7"``, ``"iii"``. Empty
    #: for formats without pages.
    pages: tuple[str, ...] = ()

    @field_validator("source")
    @classmethod
    def check_relative(cls, source: str) -> str:
        path = PurePosixPath(source)
        if path.is_absolute() or ".." in path.parts or "\\" in source:
            raise ValueError(
                f"{source!r} must be a path relative to the corpus root, with '/' "
                f"separators and no '..'"
            )
        return source

    @field_validator("pages")
    @classmethod
    def check_pages(cls, pages: tuple[str, ...]) -> tuple[str, ...]:
        if any(not page.strip() for page in pages):
            raise ValueError("page labels must be non-empty")
        return pages


class GoldenItem(_Frozen):
    """One golden record."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str
    answerable: bool
    expected_sources: tuple[ExpectedSource, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def check_kind(self) -> "GoldenItem":
        """The rules that differ between answerable and unanswerable records."""
        if self.answerable:
            if not self.ground_truth.strip():
                raise ValueError("an answerable record needs a non-empty ground_truth")
            if not self.expected_sources:
                raise ValueError("an answerable record needs expected_sources")
            if self.must_not_contain:
                raise ValueError(
                    "must_not_contain is only for unanswerable records: it lists what a "
                    "refusal must not leak"
                )
        elif self.expected_sources:
            raise ValueError("an unanswerable record has no expected_sources")
        if any(not text.strip() for text in self.must_not_contain):
            raise ValueError("must_not_contain entries must be non-empty")
        return self


def _describe(error: ValidationError) -> list[str]:
    problems = []
    for detail in error.errors():
        where = ".".join(str(part) for part in detail["loc"])
        message = detail["msg"].removeprefix("Value error, ")
        problems.append(f"{where}: {message}" if where else message)
    return problems


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Every record in a golden set file, in file order.

    Blank lines are skipped. Raises :class:`GoldenSetError` listing every problem,
    each with its line number, when any line is invalid, an id repeats, or the file
    holds no record. An unreadable file raises the ``OSError`` as is.
    """
    path = Path(path)
    items: list[GoldenItem] = []
    first_line: dict[str, int] = {}
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = GoldenItem.model_validate_json(line)
        except ValidationError as error:
            problems.extend(f"{path.name}:{number}: {problem}" for problem in _describe(error))
            continue
        if item.id in first_line:
            problems.append(
                f"{path.name}:{number}: id {item.id!r} repeats line {first_line[item.id]}"
            )
            continue
        first_line[item.id] = number
        items.append(item)
    if not items and not problems:
        problems.append(f"{path.name}: no records")
    if problems:
        raise GoldenSetError("invalid golden set:\n" + "\n".join(problems))
    return items
