"""``_target_`` resolution and recursive object construction.

This module is pure Python by contract (ARCHITECTURE.md §0.2): it must not import
LangChain, so the config machinery stays testable without the ML stack installed.

Config validation lives in :mod:`rag_qa.schema`; dotted pipeline references are
resolved by :meth:`rag_qa.schema.RagConfig.component`.

**Import allowlist (ISS-04, DEC-7).** A ``_target_`` names code to import and call,
so an unrestricted config is arbitrary code execution. :func:`import_from_string`
is the one function every ``_target_`` reaches — through :func:`build_object` or
directly — and it enforces three checks:

1. the dotted path starts with one of :data:`ALLOWED_PREFIXES`;
2. the object it resolves to is *defined* under one of them too (its
   ``__module__``). Without this, a name an allowed module merely re-exports passes
   check 1: ``rag_qa.registry.import_module`` is :func:`importlib.import_module`,
   and building it with ``name: subprocess`` imports any module on the path
   (verified 2026-09-26). LangChain packages re-export it as well;
3. the object is a class. Check 2 does not stop helpers an allowed package defines
   itself: ``langchain_core.utils.utils.guard_import`` imports any module and
   ``langchain_core._import_utils.import_attr`` returns any attribute, e.g.
   ``subprocess.Popen`` (verified 2026-09-30). Every legitimate ``_target_`` is a
   class to instantiate, so this costs the real config nothing.

:data:`ALLOWED_PREFIXES` is a constant on purpose: not config-overridable, no env
escape hatch. An allowlist the config can edit is not an allowlist. Adding a
package means editing this file, in review. This resolves ADR-015's deferred gap,
including its follow-up: the loader call site that bypassed :func:`build_object`
still goes through :func:`import_from_string`, so it is covered too.

**Trust boundary: the config is trusted input, like code.** The allowlist restricts
which classes a config can build, not the kwargs it passes them.
``HuggingFaceEmbeddings`` forwards ``model_kwargs`` to ``SentenceTransformer``, so
``trust_remote_code: true`` plus someone else's model repository runs that
repository's Python. Never load a config from an untrusted source, and never let a
request supply or override components (a constraint on the Phase 2 API). ISS-16
draws the same line for the FAISS index.
"""

from collections.abc import Iterable
from importlib import import_module
from typing import Any

#: Module prefixes a ``_target_`` may import from. Each ends in "." so a prefix
#: cannot match a longer package name (``rag_qa_evil`` is not ``rag_qa``).
#:
#: Only packages a ``_target_`` actually uses: an allowlist allows what is used, not
#: what might be (ADR-020). S2-4 removed ``langchain_classic.`` and
#: ``langchain_community.``. The old reranker entry was the only ``_target_`` under
#: either, and our own reranker (:mod:`rag_qa.rerankers`, DEC-16) replaced it.
#: ``vectorstore.py`` still imports FAISS from ``langchain_community`` in source until
#: Phase 3; no config builds from that package.
ALLOWED_PREFIXES = (
    "langchain_core.",
    "langchain_huggingface.",
    "langchain_ollama.",
    "langchain_text_splitters.",
    "rag_qa.",
)

#: Prefixes 0.3.0 removed from :data:`ALLOWED_PREFIXES` (S2-4). A ``_target_`` under one
#: is almost certainly v0.2's disabled reranker entry, so its refusal also says how to
#: migrate: see :func:`removed_prefix_hint`. They stay refused; this only names them.
REMOVED_PREFIXES = ("langchain_classic.", "langchain_community.")


def is_allowed(dotted_path: object) -> bool:
    """Whether a ``_target_`` string is under one of :data:`ALLOWED_PREFIXES`."""
    return isinstance(dotted_path, str) and dotted_path.startswith(ALLOWED_PREFIXES)


def allowlist_hint() -> str:
    """The fix-it half of every allowlist error, for the maintainer who hits one."""
    return (
        f"Allowed module prefixes: {', '.join(map(repr, ALLOWED_PREFIXES))}. The list is fixed in "
        f"rag_qa/registry.py and deliberately not configurable (DEC-7)"
    )


def removed_prefix_hint(targets: Iterable[object]) -> str:
    """The migration note for refused ``targets`` under a prefix 0.3.0 removed, or ``""``.

    A v0.2 ``config.yaml`` still carries the old reranker entry under both prefixes, and
    upgrading turns its load into an allowlist error. Without this note, that error's
    "deliberately not configurable" could read as an invitation to add the prefixes back,
    re-opening what S2-4 closed. The note says what to do instead (S2-4's second review).
    """
    if not any(isinstance(t, str) and t.startswith(REMOVED_PREFIXES) for t in targets):
        return ""
    return (
        f"{' and '.join(map(repr, REMOVED_PREFIXES))} left the allowlist in 0.3.0 (S2-4). "
        f"If this is v0.2's components.rerankers.cross_encoder entry, delete it, then copy "
        f"the rerankers block, and pipeline.query's reranker and reranker_candidates lines, "
        f"from the repository's config.yaml: the reranker is now "
        f"rag_qa.rerankers.CrossEncoderReranker (DEC-16). Adding a prefix back is not the "
        f"fix: the allowlist allows only what is used (ADR-020)"
    )


def import_from_string(dotted_path: str) -> Any:
    """Import a dotted path and return the attribute it names, if the allowlist permits.

    Raises ``ImportError`` if the path is outside :data:`ALLOWED_PREFIXES`, cannot be
    imported, resolves to something defined outside them, or is not a class.
    """
    if not is_allowed(dotted_path):
        migration = removed_prefix_hint([dotted_path])
        raise ImportError(
            f"_target_ {dotted_path!r} is outside the import allowlist (ISS-04). "
            f"{allowlist_hint()}." + (f" {migration}." if migration else "")
        )

    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = import_module(module_path)
        obj = getattr(module, class_name)
    except (ValueError, AttributeError, ImportError) as e:
        raise ImportError(f"Could not import {dotted_path!r}: {e}") from e

    # Check 2: where the object is *defined*, not where it was found. Module objects
    # have no __module__, so they are refused here. Instances are not: they inherit
    # __module__ from their class, which is why check 3 exists.
    defined_in = getattr(obj, "__module__", None)
    if not is_allowed(f"{defined_in}."):
        raise ImportError(
            f"_target_ {dotted_path!r} resolves to {obj!r}, defined in {defined_in!r}, "
            f"which is outside the import allowlist (ISS-04). A name an allowed module "
            f"re-exports does not count; name the object where it is defined. "
            f"{allowlist_hint()}."
        )

    # Check 3: a class, since build_object instantiates it. This refuses the functions
    # allowed packages define themselves (langchain_core's guard_import imports any
    # module; import_attr returns any attribute) and instances with a __call__.
    # ImportError, not TypeError: every refusal here is one error type, the one
    # check_imports and callers catch; a TypeError would escape as a raw traceback.
    if not isinstance(obj, type):
        raise ImportError(  # noqa: TRY004
            f"_target_ {dotted_path!r} resolves to {obj!r}, which is not a class. A "
            f"_target_ must name a class to instantiate: a function or instance, even "
            f"one defined in an allowed package, can import or call anything (ISS-04)."
        )
    return obj


def build_object(config_dict: Any) -> Any:
    """Recursively build objects from a config dictionary."""
    if isinstance(config_dict, dict) and "_target_" in config_dict:
        class_to_instantiate = import_from_string(config_dict["_target_"])
        args = {key: build_object(value) for key, value in config_dict.items() if key != "_target_"}
        return class_to_instantiate(**args)
    elif isinstance(config_dict, dict):
        return {key: build_object(value) for key, value in config_dict.items()}
    elif isinstance(config_dict, list):
        return [build_object(item) for item in config_dict]
    else:
        return config_dict
