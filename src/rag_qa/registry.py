"""``_target_`` resolution and recursive object construction.

This module is pure Python by contract (ARCHITECTURE.md §0.2): it must not import
LangChain, so the config machinery stays testable without the ML stack installed.

Config validation lives in :mod:`rag_qa.schema` (S1-1); dotted pipeline references
are resolved by :meth:`rag_qa.schema.RagConfig.component`, which replaced
``resolve_ref``. The import allowlist (ISS-04) lands *here*, in
:func:`import_from_string`, in S1-2: every ``_target_`` in the system reaches that
one function, whether through :func:`build_object` or directly.
"""

from importlib import import_module


def import_from_string(dotted_path):
    """Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed."""

    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, AttributeError, ImportError) as e:
        raise ImportError(f"Could not import {dotted_path}") from e


def build_object(config_dict):
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
