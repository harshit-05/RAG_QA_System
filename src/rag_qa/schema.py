"""The config schema: what a valid ``config.yaml`` is, checked before anything is built.

FR-8: the whole config graph is validated at load time and fails with a specific,
actionable error. The schema mirrors the file's two halves (ARCHITECTURE.md §0.3, §1.4):

* the *skeleton* — the ``components`` kinds, ``pipeline`` and ``paths`` — is closed.
  An unknown key is an error that names the nearest valid key, which is what made the
  ISS-01 typo (``llmS``) and v2's dead ``vector_stores`` block (ISS-11) unwritable.
  ``vector_stores`` is a kind again since S3-2, and one is in use: ``pipeline.vector_store``
  must name it (DEC-19);
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
import re
import string
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rag_qa.registry import allowlist_hint, is_allowed, removed_prefix_hint

#: Fallbacks used only when the config file omits a ``paths`` entry.
PATH_DEFAULTS = {
    "data": "corpus",
    "state": "var",
}

#: What a store's ``collection`` may be: the alias its generations are published under,
#: which each generation's name extends with ``--gen-…``. With no hyphen allowed in it, a
#: generation's name parses one way only (DEC-19).
COLLECTION_NAME = re.compile(r"[a-z0-9_]{1,48}")

#: Why ``paths.vector_store`` is refused, and how to migrate (DEC-19).
VECTOR_STORE_REMOVED = (
    "'vector_store' was removed in 1.0: the index is no longer a folder but a Qdrant "
    "collection (DEC-19). Delete it here and add 'state: var'; copy the "
    "components.vector_stores block and the pipeline.vector_store line from the "
    "repository's config.yaml; start Qdrant (docker compose up -d qdrant), then run "
    'rag-ingest. README, "Upgrading from v0.3"'
)


def collection_problem(name: str) -> str | None:
    """Why ``name`` cannot be a store's collection, or ``None`` when it can."""
    if COLLECTION_NAME.fullmatch(name):
        return None
    return (
        f"collection {name!r} must be 1 to 48 lowercase letters, digits or underscores, "
        f"for example rag_qa or scratch_1: it names the alias, and each generation "
        f"extends it with '--gen-'"
    )


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


def iter_key(node: Any, key: str, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every value stored under ``key`` at any depth of a spec, with its location.

    Locations read as :func:`iter_targets`' do: ``model_kwargs.trust_remote_code``, or
    ``items[0].trust_remote_code`` inside a list.
    """
    if isinstance(node, dict):
        for name, value in node.items():
            here = f"{path}.{name}" if path else str(name)
            if name == key:
                yield here, value
            yield from iter_key(value, key, here)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from iter_key(item, key, f"{path}[{index}]")


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
            (where, target) for where, target in iter_targets(self.spec()) if not is_allowed(target)
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


#: The search the store's retriever runs, and the one search option it takes (S3-2).
SEARCH_TYPES = ("similarity",)
SEARCH_KWARGS = ("k",)
#: How many chunks a retriever returns when its ``search_kwargs`` name no ``k``: what
#: LangChain's ``as_retriever`` returned, kept so a v0.3 config means what it meant.
DEFAULT_K = 4


class RetrieverSpec(_Strict):
    """How the store's retriever searches: by similarity, for ``search_kwargs.k`` chunks.

    Not a buildable component: a retriever is derived from the opened store, so this
    entry has no ``_target_``, and a ``_target_`` here is an unknown-key error. Its shape
    is FAISS's ``as_retriever()`` keyword arguments, kept so that its dump, which tier 2
    hashes, does not move. Since S3-2 only the options the Qdrant retriever honours are
    accepted: one ignored silently would make a config say what it does not do.
    """

    search_type: str | None = None
    search_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_supported(self) -> "RetrieverSpec":
        problems = []
        if self.search_type is not None and self.search_type not in SEARCH_TYPES:
            problems.append(
                f"search_type {self.search_type!r} is not supported: the store searches by "
                f"similarity only (leave search_type out, or set it to 'similarity')"
            )
        unknown = sorted(set(self.search_kwargs) - set(SEARCH_KWARGS))
        if unknown:
            problems.append(
                f"search_kwargs {', '.join(map(repr, unknown))} not supported: the store's "
                f"retriever takes only 'k', how many chunks it returns"
            )
        k = self.search_kwargs.get("k", DEFAULT_K)
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            problems.append(f"search_kwargs.k must be a whole number of at least 1, not {k!r}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def kwargs(self) -> dict[str, Any]:
        """A fresh dict of the settings, with unset options left out: what tier 2 hashes."""
        return self.model_dump(exclude_none=True)

    def k(self) -> int:
        """How many chunks the retriever returns."""
        k: int = self.search_kwargs.get("k", DEFAULT_K)
        return k


class Components(_Strict):
    """The component library. The kinds are fixed; the entries inside each are not."""

    loaders: dict[str, ComponentSpec]
    splitters: dict[str, ComponentSpec]
    embedders: dict[str, ComponentSpec]
    llms: dict[str, ComponentSpec]
    retrievers: dict[str, RetrieverSpec]
    rerankers: dict[str, ComponentSpec] = Field(default_factory=dict)
    #: Where the index lives (DEC-19): ``rag_qa.vectorstore.QdrantStore`` entries.
    vector_stores: dict[str, ComponentSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_collections(self) -> "Components":
        """A store's ``collection`` is checked at load (FR-8), not when Qdrant refuses it."""
        problems = []
        for name, spec in self.vector_stores.items():
            collection = spec.spec().get("collection")
            if collection is None:
                continue
            problem = (
                collection_problem(collection)
                if isinstance(collection, str)
                else f"collection must be a name, got {type(collection).__name__}"
            )
            if problem is not None:
                problems.append(f"vector_stores.{name}: {problem}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

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
                f'to a loader, e.g. ".pdf": components.loaders.pdf. Remove it here'
            )
        return self

    @model_validator(mode="after")
    def check_trust_remote_code(self) -> "Components":
        """No component may load a model's own code (DEC-18, the S1-2 second review).

        ``trust_remote_code`` makes Hugging Face run Python from the model's repository.
        The allowlist limits which classes a config builds, not the kwargs it passes them,
        so this is the line it cannot draw. Every kind is walked, retrievers included, at
        any depth. A truthy value is refused, so the string ``"false"`` is too: it is
        truthy where it is used.
        """
        found = []
        for kind in type(self).model_fields:
            entries: Mapping[str, ComponentSpec | RetrieverSpec] = getattr(self, kind)
            for name, entry in entries.items():
                spec = entry.spec() if isinstance(entry, ComponentSpec) else entry.kwargs()
                found += [
                    f"{kind}.{name}.{where}"
                    for where, value in iter_key(spec, "trust_remote_code")
                    if value
                ]
        if found:
            raise ValueError(
                f"{', '.join(found)}: trust_remote_code is set. It runs Python from the "
                f"model's repository on this host, so no config may enable it. Remove it, "
                f"or set it to false"
            )
        return self


#: The placeholders the chat prompt is filled with, and the only ones it may use.
PROMPT_FIELDS = ("context", "question")


def placeholders(template: str) -> list[str]:
    """Every replacement field in an f-string template, nested format specs included.

    ``string.Formatter`` reads braces exactly as ``str.format`` and an f-string
    ``ChatPromptTemplate`` do: ``{{`` and ``}}`` are literal braces, and ``{}`` is a field
    named ``""``. Raises ``ValueError`` on an unbalanced brace. Using it keeps this module
    free of LangChain.
    """
    fields = []
    for _, field, spec, _ in string.Formatter().parse(template):
        if field is not None:
            fields.append(field)
        if spec:
            fields += placeholders(spec)
    return fields


class Prompt(_Strict):
    system: str
    human: str

    @model_validator(mode="after")
    def check_placeholders(self) -> "Prompt":
        """The template's placeholders are the ones the chain fills, where it fills them
        (ARCHITECTURE.md §2.3; the FR-8 follow-up found in S1-1).

        A ``human`` turn without ``{context}`` answers without retrieval, silently. Any
        other placeholder is a required variable that fails every question. ``{context}``
        in the system turn would put retrieved text among the instructions (OWASP LLM01).
        """
        found: dict[str, list[str]] = {}
        for role in ("system", "human"):
            try:
                found[role] = placeholders(getattr(self, role))
            except ValueError as e:
                raise ValueError(
                    f"{role}: {e}. A literal brace is written doubled, {{{{ or }}}}"
                ) from None
        problems = []
        stray = sorted({f for fields in found.values() for f in fields if f not in PROMPT_FIELDS})
        if stray:
            problems.append(
                f"unknown placeholder(s) {', '.join('{' + f + '}' for f in stray)}: only "
                f"{{context}} and {{question}} are filled, so each other one fails every "
                f"question. A literal brace is written doubled, {{{{ or }}}}"
            )
        missing = [f"{{{f}}}" for f in PROMPT_FIELDS if f not in found["human"]]
        if missing:
            problems.append(
                f"human must contain {{context}} and {{question}}; it has no {' or '.join(missing)}"
            )
        if "context" in found["system"]:
            problems.append(
                "system must not contain {context}: retrieved text belongs in the human "
                "turn, apart from the instructions (OWASP LLM01)"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self


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
    #: The store both halves share: ingestion writes it, queries read it (DEC-19).
    vector_store: str


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
    given. Absolute means symlinks resolved.

    ``state`` holds what the host keeps beside the index, never the index itself: the
    writer lock (DEC-19), and S3-8's job records. ``vector_store``, v0.3's index folder, is
    refused with the migration: the index lives in Qdrant now.
    """

    data: Path = Path(PATH_DEFAULTS["data"])
    state: Path = Path(PATH_DEFAULTS["state"])

    @model_validator(mode="before")
    @classmethod
    def refuse_vector_store(cls, data: Any) -> Any:
        """v0.3's ``paths.vector_store`` gets the migration, not "unknown key"."""
        if isinstance(data, dict) and "vector_store" in data:
            raise ValueError(VECTOR_STORE_REMOVED)
        return data

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


class Judge(_Strict):
    """The tier-2 judge (DEC-15): an Ollama model built from a committed Modelfile, reached
    through Ollama's OpenAI-compatible endpoint. Only ``rag-eval score`` builds it."""

    model: str = Field(min_length=1)
    #: A file in the eval folder (``eval/`` beside the config file, or ``--eval-dir``), so
    #: the eval files travel together. Its text is hashed into a run's ``judge`` part, and
    #: its ``num_ctx`` is checked against the served model's before scoring.
    modelfile: str
    base_url: str = Field(min_length=1)
    #: The sentence-transformers model answer relevancy compares questions with (on CPU).
    embedding_model: str = Field(min_length=1)
    #: Seconds one judge call may take. Runtime only, never hashed: a call on this CPU took
    #: up to 13 minutes in S2-5's spike.
    timeout_s: float = Field(gt=0)

    @field_validator("modelfile")
    @classmethod
    def check_modelfile(cls, modelfile: str) -> str:
        path = PurePosixPath(modelfile)
        if (
            not modelfile.strip()
            or path.is_absolute()
            or modelfile.startswith("~")
            or ".." in path.parts
            or "\\" in modelfile
        ):
            raise ValueError(
                f"{modelfile!r} must be a file name relative to the eval folder (eval/ beside "
                f"the config file, or --eval-dir), with no '..', e.g. judge.Modelfile"
            )
        return modelfile


class Evaluation(_Strict):
    """What only ``rag-eval`` reads (ARCHITECTURE.md §2.3). Optional in the file."""

    #: The core of the system prompt's refusal sentence. An unanswerable question's answer
    #: counts as a decline when it contains this, in any case (DEC-15).
    decline_marker: str
    judge: Judge

    @field_validator("decline_marker")
    @classmethod
    def check_marker(cls, marker: str) -> str:
        if not marker.strip() or marker != marker.strip():
            raise ValueError(
                "must be a non-empty phrase with no surrounding spaces, e.g. "
                "'could not find the answer'"
            )
        return marker


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
    #: Read by ``rag-eval`` only (S2-5). Optional, so a config without it still serves.
    evaluation: Evaluation | None = None

    @model_validator(mode="after")
    def check_decline_marker(self) -> "RagConfig":
        """The decline marker is part of the refusal the system prompt asks for.

        Edit the refusal sentence without the marker, and the decline rate would drop to
        zero without a word (ARCHITECTURE.md §2.3). Compared in any case, as answers are.
        """
        if self.evaluation is None:
            return self
        marker = self.evaluation.decline_marker
        if marker.lower() not in self.pipeline.query.prompt.system.lower():
            raise ValueError(
                f"evaluation.decline_marker {marker!r} does not appear in "
                f"pipeline.query.prompt.system. It must be part of the refusal sentence the "
                f"system prompt asks for, or no answer would ever count as a decline"
            )
        return self

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
        yield "pipeline.vector_store", self.pipeline.vector_store, "vector_stores"

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
