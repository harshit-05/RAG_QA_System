"""Configuration loading and path resolution.

The only reader of YAML and of the ``RAG_*`` environment variables in the system
(ARCHITECTURE.md §0.2). Keeping it in one place is what lets Phase 1 swap the plain
dict for a validated Pydantic model (FR-8) without touching a single call site.

Paths in the config file are relative to *the config file's own directory*, never to
the process working directory. That is what makes a fresh clone runnable from
anywhere and closes ISS-02 / NFR-11 — the old config hardcoded absolute paths into
another machine's home directory.
"""

import os
from pathlib import Path

import yaml

#: Selects which config file to load.
ENV_CONFIG = "RAG_CONFIG"
#: Overrides ``paths.data`` (the document corpus).
ENV_DATA_PATH = "RAG_DATA_PATH"
#: Overrides ``paths.vector_store`` (the index directory).
ENV_VECTOR_STORE_PATH = "RAG_VECTOR_STORE_PATH"

DEFAULT_CONFIG_FILENAME = "config.yaml"

#: Fallbacks used only when the config file omits a ``paths`` entry.
_PATH_DEFAULTS = {
    "data": "corpus",
    "vector_store": "vectorstore/db_faiss",
}

_PATH_ENV_VARS = {
    "data": ENV_DATA_PATH,
    "vector_store": ENV_VECTOR_STORE_PATH,
}


def resolve_config_path(config_path=None):
    """Decide which config file to load.

    Precedence: explicit argument, then ``$RAG_CONFIG``, then ``config.yaml`` in
    the current directory.
    """
    chosen = config_path or os.environ.get(ENV_CONFIG) or DEFAULT_CONFIG_FILENAME
    return Path(chosen).expanduser()


def _resolve_path(base_dir, raw_value, env_var):
    """Resolve one configured path to an absolute string.

    An environment override wins over the config file. Relative values anchor to
    different places depending on where they came from, which is the conventional
    rule and the least surprising one:

    * relative path *in the config file* → relative to the config file's own
      directory, so a fresh clone runs from anywhere (NFR-11);
    * relative path *in an environment variable* → relative to the caller's
      working directory, so ``RAG_DATA_PATH=./mydocs`` means what a shell user
      typing it expects.
    """
    override = os.environ.get(env_var)
    if override:
        candidate, anchor = Path(override).expanduser(), Path.cwd()
    else:
        candidate, anchor = Path(raw_value).expanduser(), base_dir
    if not candidate.is_absolute():
        candidate = anchor / candidate
    return str(candidate.resolve())


def load_config(config_path=None):
    """Load the YAML config and return it as a dict with absolute ``paths``.

    Raises ``FileNotFoundError`` if the config file itself is missing, which is a
    clearer failure than the ``KeyError`` avalanche the old code produced.
    """
    path = resolve_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Pass a path, or set ${ENV_CONFIG}, or run from the repo root."
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = path.resolve().parent
    paths = dict(config.get("paths") or {})
    for key, default in _PATH_DEFAULTS.items():
        paths[key] = _resolve_path(base_dir, paths.get(key, default), _PATH_ENV_VARS[key])
    config["paths"] = paths

    return config
