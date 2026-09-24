"""Configuration loading.

The only reader of YAML in the system (ARCHITECTURE.md §0.2). Keeping it in one
place is what lets S0-4 add path resolution and environment overrides, and Phase 1
swap the plain dict for a validated Pydantic model (FR-8), without touching a
single call site.
"""

import yaml


def load_config(config_path="config.yaml"):
    """Load the YAML config file and return it as a plain dict.

    S0-4 extends this to resolve ``paths.*`` relative to the config file's own
    directory and to honour the ``RAG_CONFIG`` / ``RAG_DATA_PATH`` /
    ``RAG_VECTOR_STORE_PATH`` environment overrides (ISS-02, NFR-11).
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
