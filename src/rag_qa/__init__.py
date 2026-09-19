"""RAG_QA_System: a config-driven local retrieval-augmented question-answering engine.

Module layout is fixed by ARCHITECTURE.md §0.2; the modules themselves arrive in S0-3:

- ``registry``    ``_target_`` resolution and recursive object construction (pure Python)
- ``config``      the only reader of YAML and environment overrides
- ``vectorstore`` the only FAISS touchpoints; the Phase 3 store-swap seam
- ``ingest``      corpus walk, chunking, index build
- ``chain``       ``build_rag_chain(cfg) -> Runnable``
- ``cli``         interactive REPL over that Runnable
- ``evaluate``    RAGAs harness (Phase 2)

Version tracks the release tags: ``.dev0`` between tags, bumped at each phase exit
(Phase 0 exit ships 0.1.0 / tag v0.1).
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
