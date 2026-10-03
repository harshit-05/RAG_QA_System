"""The quality bar's floors: ``eval/thresholds.yaml``, and which metrics fall below them.

DEC-15: each tier's floors are its first measured baseline minus a tolerance, and they
are only ever raised deliberately. The file has one section per tier::

    retrieval:  {hit_rate: ..., mrr: ..., recall: ...}     # tier 1, rag-eval retrieval (S2-3)

S2-5 adds the ``generation:`` section for tier 2, and the freshness check
(``rag-eval check``), to this module.

CI's model-free job will import this module (S2-5), so it stays light: pydantic and
YAML, no LangChain and no model stack.
"""

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ThresholdsError(ValueError):
    """The thresholds file is missing, unreadable or invalid. The message says where."""


class _Strict(BaseModel):
    # strict: a hand-edited floor should be a number, never "0.8" or `true`. Strict mode
    # still accepts an int for a float, so `mrr: 0` loads.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class RetrievalFloors(_Strict):
    """Tier 1's floors. A metric meets its floor when it is at or above it."""

    # ge=0: a negative floor could never fail, and a typo must not switch a gate off. No
    # upper bound: a floor above 1 can never be met, which is how a negative control
    # shows that the gate can fail.
    hit_rate: float = Field(ge=0)
    mrr: float = Field(ge=0)
    recall: float = Field(ge=0)


class Thresholds(_Strict):
    """The whole thresholds file."""

    retrieval: RetrievalFloors


def load_thresholds(path: str | Path) -> Thresholds:
    """The floors in ``path``. Raises :class:`ThresholdsError` naming every problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ThresholdsError(f"cannot read the thresholds file {path}: {e.strerror}") from e
    except yaml.YAMLError as e:
        raise ThresholdsError(f"thresholds file {path} is not valid YAML:\n{e}") from e
    if raw is None:
        raise ThresholdsError(f"thresholds file {path} is empty: it needs a retrieval: section")
    try:
        return Thresholds.model_validate(raw)
    except ValidationError as e:
        problems = []
        for detail in e.errors():
            where = ".".join(str(part) for part in detail["loc"])
            problems.append(f"  {where}: {detail['msg']}" if where else f"  {detail['msg']}")
        # `from None`: the lines above say everything the pydantic dump would.
        raise ThresholdsError(f"invalid thresholds file {path}:\n" + "\n".join(problems)) from None


def below_floors(values: Mapping[str, float], floors: BaseModel) -> list[str]:
    """The names of the metrics in ``values`` that are below their floor, in the floors'
    order. A metric exactly at its floor meets it."""
    return [
        name for name, floor in floors.model_dump().items() if values[name] < floor
    ]
