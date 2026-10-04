"""The config schema: what a valid ``config.yaml`` is, checked before anything is built.

FR-8: the whole config graph is validated at load time and fails with a specific,
actionable error. The schema mirrors the file's two halves (ARCHITECTURE.md §0.3, §1.4):

* the *skeleton* — the ``components`` kinds, ``pipeline`` and ``paths`` — is closed.
  An unknown key is an error that names the nearest valid key, which is what makes
  the ISS-01 typo (``llmS``) and the dead ``vector_stores`` block (ISS-11) unwritable;
* component *leaves* are open: any kwargs next to ``_target_`` pass straight through
  to :func:`rag_qa.registry.build_object` (ADR-010).

Pure Python by contract, like :mod:`rag_qa.registry`: no LangChain imports, so the
config machinery stays testable without the ML stack installed.

Every model is frozen. That blocks attribute assignment only — dicts inside the model
stay mutable — so call sites never receive those dicts directly: they go through
:meth:`ComponentSpec.spec` and :meth:`RetrieverSpec.kwargs`, which return fresh
copies. A caller mutating what it got back cannot reach the config.
"""

import difflib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from rag_qa.registry import allowlist_hint, is_allowed, removed_prefix_hint

#: Fallbacks used only when the config file omits a ``paths`` entry.
PATH_DEFAULTS = {
    "data": "corpus",
    "vector_store": "vectorstore/db_faiss",
}


def _did_you_mean(word: str, candidates: Iterable[str]) -> str:
    match = difflib.get_close_matches(word, list(candidates), n=1)
    return f" (did you mean {match[0]!r}?)" if match else ""


def iter_targets(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every ``_target_`` in a component spec, nested ones included, with its location.

    Locations are relative to the spec: ``_target_`` for the component itself,
    ``base_compressor.model._target_`` for one nested inside it. Nesting matters:
    ISS-03's broken class name sat two levels down inside the reranker.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "_target_":
                yield here, value
            else:
                yield from iter_targets(value, here)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from iter_targets(item, f"{path}[{index}]")


class _Strict(BaseModel):
    """Frozen, closed model: an unknown key fails with the nearest valid key named.

    ``extra="forbid"`` alone would report the ``llmS`` typo as an anonymous "extra
    inputs are not permitted" plus a separate "field required" for ``llms``; the
    validator below turns that pair into one line that says what to type instead.
    ``extra="forbid"`` stays as the backstop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def check_known_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        allowed = list(cls.model_fields)
        unknown = [str(key) for key in data if key not in allowed]
        if unknown:
            problems = "; ".join(
                f"unknown key {key!r}{_did_you_mean(key, allowed)}" for key in unknown
            )
            raise ValueError(f"{problems}. Expected one of: {', '.join(allowed)}")
        return data


class ComponentSpec(BaseModel):
    """One buildable component: a ``_target_`` class path plus open kwargs."""

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    # Aliased because Pydantic treats a leading-underscore name as a private
    # attribute: a field declared literally as `_target_` is silently dropped and
    # model_dump() comes back empty (verified 2026-09-25). Every build_object call
    # would then fail far from the cause.
    target: str = Field(alias="_target_")

    @model_validator(mode="after")
    def check_allowlist(self) -> "ComponentSpec":
        """Every ``_target_`` here, nested ones too, is under an allowed prefix (ISS-04).

        A string check only: loading never imports (DEC-7), so a defined-but-unused
        component (a ``_cuda`` entry on this host) costs nothing. Import-time checks,
        including where the object is defined, happen in ``registry.import_from_string``.
        """
        blocked = [
            (where, target) for where, target in iter_targets(self.spec())
            if not is_allowed(target)
        ]
        if blocked:
            # A v0.2 reranker entry also gets the way to migrate (S2-4's second review).
            migration = removed_prefix_hint(target for _, target in blocked)
            raise ValueError(
                f"{'; '.join(f'{where} {target!r}' for where, target in blocked)}: outside "
                f"the import allowlist (ISS-04). {allowlist_hint()}"
                + (f". {migration}" if migration else "")
            )
        return self

    def spec(self) -> dict[str, Any]:
        """A fresh ``build_object`` dict: ``_target_`` plus every kwarg, deep-copied."""
        return self.model_dump(by_alias=True)


class RetrieverSpec(_Strict):
    """Kwargs for ``VectorStore.as_retriever()``.

    Not a buildable component: a retriever is derived from the opened store, so this
    entry has no ``_target_``, and a ``_target_`` here is an unknown-key error.
    """

    search_type: str | None = None
    search_kwargs: dict[str, Any] = Field(default_factory=dict)

    def kwargs(self) -> dict[str, Any]:
        """A fresh kwargs dict for ``as_retriever``, with unset options left out."""
        return self.model_dump(exclude_none=True)


class Components(_Strict):
    """The component library. The kinds are fixed; the entries inside each are not."""

    loaders: dict[str, ComponentSpec]
    splitters: dict[str, ComponentSpec]
    embedders: dict[str, ComponentSpec]
    llms: dict[str, ComponentSpec]
    retrievers: dict[str, RetrieverSpec]
    rerankers: dict[str, ComponentSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_loader_shape(self) -> "Components":
        """Catch a pre-S1-3 config: ``extensions`` left inside a loader entry.

        Without this it would load fine (leaf kwargs are open) and then fail at build
        time as ``TypeError: unexpected keyword argument 'extensions'`` — far from the
        cause, for anyone upgrading an older config.
        """
        stale = [
            name for name, spec in self.loaders.items() if "extensions" in (spec.model_extra or {})
        ]
        if stale:
            raise ValueError(
                f"loaders {', '.join(map(repr, stale))}: 'extensions' moved out of the loader "
                f"entries into pipeline.ingestion.loaders (S1-3), which maps each extension "
                f"to a loader, e.g. \".pdf\": components.loaders.pdf. Remove it here"
            )
        return self


class Prompt(_Strict):
    system: str
    human: str


class Ingestion(_Strict):
    #: File extension → loader reference, e.g. ``".pdf": components.loaders.pdf``.
    #: A file whose lowercased suffix is not a key here is skipped, not failed.
    loaders: dict[str, str]
    splitter: str
    embedder: str

    @model_validator(mode="after")
    def check_extensions(self) -> "Ingestion":
        """Keys must be what a file's lowercased suffix can equal, or they never match.

        That means one dot, at the start: ``Path.suffix`` is only the last part, so a
        ``.tar.gz`` key could never match ``a.tar.gz`` (whose suffix is ``.gz``).
        """

        def matchable(ext: str) -> bool:
            return (
                ext.startswith(".") and len(ext) > 1 and ext == ext.lower() and ext.count(".") == 1
            )

        bad = [ext for ext in self.loaders if not matchable(ext)]
        if bad:
            raise ValueError(
                f"loaders: extension keys {', '.join(map(repr, bad))} can never match a file. "
                f"Use a lowercase suffix with one leading dot, e.g. '.pdf' (files are matched "
                f"on their lowercased last suffix, so '.pdf' also covers REPORT.PDF, and "
                f"'.gz' is what a.tar.gz matches)"
            )
        return self


class Query(_Strict):
    llm: str
    retriever: str
    prompt: Prompt
    reranker: str | None = None
    #: How many chunks the retriever fetches for the reranker, in place of its own ``k``.
    #: Only with a reranker, which cuts them down to its ``top_n`` (DEC-16).
    reranker_candidates: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_reranker_candidates(self) -> "Query":
        """``reranker_candidates`` comes with a reranker, and only with one.

        Missing with a reranker, the reranker would only reorder the retriever's own few
        chunks. Left behind without one, it would read as if more chunks were fetched,
        while the retriever's own ``k`` decides.
        """
        if self.reranker is not None and self.reranker_candidates is None:
            raise ValueError(
                "'reranker_candidates' is required with a 'reranker': how many chunks the "
                "retriever fetches for the reranker to cut down to its top_n, in place of "
                "the retriever's own k (config.yaml uses 20)"
            )
        if self.reranker is None and self.reranker_candidates is not None:
            raise ValueError(
                "'reranker_candidates' is set but 'reranker' is not. It sizes the reranker's "
                "candidates only; without a reranker, the retriever's own k decides how many "
                "chunks reach the prompt. Remove it, or set 'reranker'"
            )
        return self


class Pipeline(_Strict):
    ingestion: Ingestion
    query: Query


@dataclass(frozen=True)
class PathContext:
    """What :class:`Paths` needs to anchor relative paths. Supplied by ``load_config``.

    ``overrides`` maps each ``paths`` key to its environment override (``None`` when
    unset); ``env_vars`` maps it to the variable's name, for error messages.
    """

    base_dir: Path
    overrides: Mapping[str, str | None]
    env_vars: Mapping[str, str]


class Paths(_Strict):
    """Filesystem locations, absolute once loaded through ``load_config``.

    The anchoring rules are the S0-4 ones, and they differ by where a value came from:

    * a relative path **in the config file** anchors to the config file's directory,
      so a fresh clone runs from anywhere. An absolute path there is rejected (NFR-11:
      committed config must not carry machine-specific paths). ``~`` counts as
      absolute, since it expands to one user's home;
    * a relative path **in an environment variable** anchors to the working
      directory, because that is what ``RAG_DATA_PATH=./mydocs`` means to the person
      typing it. Absolute is allowed: the environment is the place for
      machine-specific paths.

    Without a :class:`PathContext` (programmatic construction) values are taken as
    given.
    """

    data: Path = Path(PATH_DEFAULTS["data"])
    vector_store: Path = Path(PATH_DEFAULTS["vector_store"])

    @model_validator(mode="before")
    @classmethod
    def anchor(cls, data: Any, info: ValidationInfo) -> Any:
        context = info.context
        if not isinstance(context, PathContext):
            return data
        if data is None:  # a `paths:` heading with nothing under it
            data = {}
        if not isinstance(data, dict):
            # ValueError, not TypeError: Pydantic turns only ValueError/AssertionError
            # into a ValidationError. A TypeError would escape as a raw traceback and
            # bypass ConfigError's rendering (FR-8).
            raise ValueError(  # noqa: TRY004
                f"must be a mapping of {', '.join(PATH_DEFAULTS)} to directories, "
                f"got {type(data).__name__}"
            )

        resolved = dict(data)  # unknown keys pass through to check_known_keys
        problems = []
        for key, default in PATH_DEFAULTS.items():
            env_var = context.env_vars[key]
            override = context.overrides.get(key)
            if override:
                resolved[key] = (Path.cwd() / Path(override).expanduser()).resolve()
                continue

            raw = data.get(key, default)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                problems.append(
                    f"{key!r} is empty. Give a directory relative to the config file, "
                    f"or set {env_var}"
                )
                continue
            if not isinstance(raw, str):
                problems.append(f"{key!r} must be a path string, got {type(raw).__name__}")
                continue
            path = Path(raw).expanduser()
            if path.is_absolute():
                problems.append(
                    f"{key!r} is an absolute path ({raw!r}) in the config file. Use a "
                    f"path relative to the config file's directory (NFR-11), or set "
                    f"{env_var} for a machine-specific location"
                )
                continue
            resolved[key] = (context.base_dir / path).resolve()

        if problems:
            raise ValueError("\n".join(problems))
        return resolved


#: Each pipeline slot, and the component kind it must reference.
_SLOT_KINDS = {
    ("ingestion", "splitter"): "splitters",
    ("ingestion", "embedder"): "embedders",
    ("query", "llm"): "llms",
    ("query", "retriever"): "retrievers",
    ("query", "reranker"): "rerankers",
}


class RagConfig(_Strict):
    """A validated, frozen config. Build one with :func:`rag_qa.config.load_config`."""

    components: Components
    pipeline: Pipeline
    # Optional in the file, in which case the defaults apply. validate_default makes
    # the defaults go through Paths.anchor too, so they come out absolute as well.
    paths: Paths = Field(default_factory=dict, validate_default=True)

    def _slots(self) -> Iterator[tuple[str, str, str]]:
        """Every pipeline reference in use: ``(location, reference, required kind)``.

        The one walk both :meth:`check_references` and :meth:`references` use, so a
        new slot (like the S1-3 loader map) cannot be validated but not listed, or
        listed but not validated.
        """
        for (stage, slot), kind in _SLOT_KINDS.items():
            ref = getattr(getattr(self.pipeline, stage), slot)
            if ref is not None:
                yield f"pipeline.{stage}.{slot}", ref, kind
        for extension, ref in self.pipeline.ingestion.loaders.items():
            yield f"pipeline.ingestion.loaders[{extension!r}]", ref, "loaders"

    @model_validator(mode="after")
    def check_references(self) -> "RagConfig":
        """Every pipeline reference names an existing component of the right kind."""
        problems = []
        for location, ref, kind in self._slots():
            problem = self._reference_problem(ref, kind)
            if problem:
                problems.append(f"{location}: {problem}")
        if problems:
            raise ValueError("\n".join(problems))
        return self

    # After check_references on purpose: a failing after-validator stops the ones after
    # it (verified, pydantic 2.13), so the reranker reference resolves here.
    @model_validator(mode="after")
    def check_reranker_top_n(self) -> "RagConfig":
        """``reranker_candidates`` gives the reranker at least its ``top_n`` to choose from.

        ``top_n`` is read from the reranker's entry, so the entry must state it: loading
        never imports a component to learn its defaults (DEC-7).
        """
        query = self.pipeline.query
        if query.reranker is None or query.reranker_candidates is None:
            return self  # Query has refused a reranker without candidates already
        top_n = self.component(query.reranker).spec().get("top_n")
        if not isinstance(top_n, int) or isinstance(top_n, bool):
            # ValueError, not TypeError, for the reason Paths.anchor gives (FR-8).
            raise ValueError(  # noqa: TRY004
                f"pipeline.query.reranker: {query.reranker!r} must state top_n as a whole "
                f"number, so that reranker_candidates can be checked against it at load. "
                f"Loading never imports the reranker to read its default (DEC-7)"
            )
        if query.reranker_candidates < top_n:
            raise ValueError(
                f"pipeline.query.reranker_candidates is {query.reranker_candidates}, below "
                f"the reranker's top_n of {top_n}. The reranker keeps the best top_n of the "
                f"candidates, so it needs at least that many"
            )
        return self

    def _reference_problem(self, ref: str, expected_kind: str) -> str | None:
        parts = ref.split(".")
        if len(parts) != 3 or parts[0] != "components":
            return (
                f"{ref!r} is not a component reference; "
                f"expected 'components.{expected_kind}.<name>'"
            )
        _, kind, name = parts
        if kind != expected_kind:
            return (
                f"{ref!r} points into components.{kind}, "
                f"but this slot needs an entry of components.{expected_kind}"
            )
        entries = getattr(self.components, kind)
        if name not in entries:
            return f"'components.{kind}' has no entry {name!r}{_did_you_mean(name, entries)}"
        return None

    def references(self) -> dict[str, str]:
        """Every pipeline reference in use, by slot, e.g. ``{'pipeline.query.llm': ...}``."""
        return {location: ref for location, ref, _ in self._slots()}

    def targets(self, ref: str) -> list[tuple[str, str]]:
        """The ``_target_``s a reference would import, with locations; none for retrievers."""
        spec = self._lookup(ref)
        if isinstance(spec, RetrieverSpec):
            return []
        return list(iter_targets(spec.spec()))

    def component(self, ref: str) -> ComponentSpec:
        """The buildable component a dotted reference names, e.g. ``components.llms.x``."""
        spec = self._lookup(ref)
        if not isinstance(spec, ComponentSpec):
            raise TypeError(
                f"{ref!r} is a retriever setting, not a buildable component; "
                f"use RagConfig.retriever()"
            )
        return spec

    def retriever(self, ref: str) -> RetrieverSpec:
        """The retriever settings a dotted reference names, e.g. ``components.retrievers.x``."""
        spec = self._lookup(ref)
        if not isinstance(spec, RetrieverSpec):
            raise TypeError(f"{ref!r} is not a retriever setting; use RagConfig.component()")
        return spec

    def _lookup(self, ref: str) -> ComponentSpec | RetrieverSpec:
        parts = ref.split(".")
        if len(parts) != 3 or parts[0] != "components" or parts[1] not in Components.model_fields:
            raise KeyError(f"{ref!r} is not a component reference ('components.<kind>.<name>')")
        _, kind, name = parts
        entries: Mapping[str, ComponentSpec | RetrieverSpec] = getattr(self.components, kind)
        if name not in entries:
            raise KeyError(f"'components.{kind}' has no entry {name!r}")
        return entries[name]
