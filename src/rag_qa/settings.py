"""Environment overrides, read through pydantic-settings.

The only reader of the ``RAG_*`` environment variables (ARCHITECTURE.md §1.2). An
empty variable counts as unset, so ``RAG_DATA_PATH= rag-ingest`` means "use the config
file's value", as it always has.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Selects which config file to load.
ENV_CONFIG = "RAG_CONFIG"
#: Overrides ``paths.data`` (the document corpus).
ENV_DATA_PATH = "RAG_DATA_PATH"
#: Overrides ``paths.vector_store`` (the index directory).
ENV_VECTOR_STORE_PATH = "RAG_VECTOR_STORE_PATH"

#: Each ``paths`` key, and the variable that overrides it.
PATH_ENV_VARS = {
    "data": ENV_DATA_PATH,
    "vector_store": ENV_VECTOR_STORE_PATH,
}


class EnvSettings(BaseSettings):
    """The ``RAG_*`` overrides as they are *now*.

    Instantiated on every :func:`rag_qa.config.load_config` call, never at import, so
    a long-lived process and a test's ``monkeypatch`` both see the current environment.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_ignore_empty=True, frozen=True)

    config: str | None = None
    data_path: str | None = None
    vector_store_path: str | None = None

    def path_overrides(self) -> dict[str, str | None]:
        """Each ``paths`` key mapped to its override, ``None`` when unset."""
        return {"data": self.data_path, "vector_store": self.vector_store_path}
