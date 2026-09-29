"""``_target_`` resolution and recursive object construction.

This module is pure Python by contract (ARCHITECTURE.md §0.2): it must not import
LangChain, so the config machinery stays testable without the ML stack installed.

Config validation lives in :mod:`rag_qa.schema`; dotted pipeline references are
resolved by :meth:`rag_qa.schema.RagConfig.component`.

**Import allowlist (ISS-04, DEC-7).** A ``_target_`` names code to import and call,
so an unrestricted config is arbitrary code execution. :func:`import_from_string`
is the one function every ``_target_`` reaches — through :func:`build_object` or
directly — and it enforces two checks:

1. the dotted path starts with one of :data:`ALLOWED_PREFIXES`;
2. the object it resolves to is *defined* under one of them too (its
   ``__module__``). Without this, a name an allowed module merely re-exports passes
   check 1: ``rag_qa.registry.import_module`` is :func:`importlib.import_module`,
   and building it with ``name: subprocess`` imports any module on the path
   (verified 2026-09-26). LangChain packages re-export it as well.

:data:`ALLOWED_PREFIXES` is a constant on purpose: not config-overridable, no env
escape hatch. An allowlist the config can edit is not an allowlist. Adding a
package means editing this file, in review. This resolves ADR-015's deferred gap,
including its follow-up: the loader call site that bypassed :func:`build_object`
still goes through :func:`import_from_string`, so it is covered too.
"""

from importlib import import_module
from typing import Any

#: Module prefixes a ``_target_`` may import from. Each ends in "." so a prefix
#: cannot match a longer package name (``rag_qa_evil`` is not ``rag_qa``).
#: ``langchain_classic.`` is here for the disabled reranker config entry (Phase 2);
#: ADR-007 still forbids importing it from source under ``src/rag_qa/``.
ALLOWED_PREFIXES = (
    "langchain_core.",
    "langchain_community.",
    "langchain_huggingface.",
    "langchain_ollama.",
    "langchain_text_splitters.",
    "langchain_classic.",
    "rag_qa.",
)


def is_allowed(dotted_path: object) -> bool:
    """Whether a ``_target_`` string is under one of :data:`ALLOWED_PREFIXES`."""
    return isinstance(dotted_path, str) and dotted_path.startswith(ALLOWED_PREFIXES)


def allowlist_hint() -> str:
    """The fix-it half of every allowlist error, for the maintainer who hits one."""
    return (
        f"Allowed module prefixes: {', '.join(map(repr, ALLOWED_PREFIXES))}. The list is fixed in "
        f"rag_qa/registry.py and deliberately not configurable (DEC-7)"
    )


def import_from_string(dotted_path: str) -> Any:
    """Import a dotted path and return the attribute it names, if the allowlist permits.

    Raises ``ImportError`` if the path is outside :data:`ALLOWED_PREFIXES`, cannot be
    imported, or resolves to something defined outside them.
    """
    if not is_allowed(dotted_path):
        raise ImportError(
            f"_target_ {dotted_path!r} is outside the import allowlist (ISS-04). "
            f"{allowlist_hint()}."
        )

    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = import_module(module_path)
        obj = getattr(module, class_name)
    except (ValueError, AttributeError, ImportError) as e:
        raise ImportError(f"Could not import {dotted_path!r}: {e}") from e

    # Check 2: where the object is *defined*, not where it was found. Module objects
    # and instances have no __module__, so they are refused too.
    defined_in = getattr(obj, "__module__", None)
    if not is_allowed(f"{defined_in}."):
        raise ImportError(
            f"_target_ {dotted_path!r} resolves to {obj!r}, defined in {defined_in!r}, "
            f"which is outside the import allowlist (ISS-04). A name an allowed module "
            f"re-exports does not count; name the object where it is defined. "
            f"{allowlist_hint()}."
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
