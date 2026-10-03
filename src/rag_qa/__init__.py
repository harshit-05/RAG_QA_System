"""RAG_QA_System: a config-driven local retrieval-augmented question-answering engine.

Module layout is fixed by ARCHITECTURE.md §0.2 and §1.2:

- ``registry``    ``_target_`` resolution and recursive object construction (pure Python)
- ``schema``      what a valid config is: the frozen Pydantic ``RagConfig`` (pure Python)
- ``settings``    the ``RAG_*`` environment overrides
- ``config``      reads the config file, returns a validated ``RagConfig``
- ``loaders``     PDF / DOCX / text loaders on ``pypdf`` and ``docx2txt``
- ``components``  builds the embedder, splitter, LLM and loaders from a ``RagConfig``
- ``vectorstore`` the only FAISS touchpoints; the Phase 3 store-swap seam
- ``ingest``      corpus walk, chunking, index build
- ``chain``       ``build_query_pipeline(cfg) -> QueryPipeline``; ``build_rag_chain`` for invoke
- ``answering``   ``stream_answer``: one answer as typed events, for every front end
- ``cli``         interactive REPL over that stream
- ``evaluate``    RAGAs harness (Phase 2); replaced by ``evaluation`` in S2-5
- ``evaluation``  the golden set (``evaluation.dataset``) and, from S2-3, the eval gates

Version tracks the release tags: ``.dev0`` between tags, bumped at each phase exit
(Phase 0 shipped 0.1.0 / tag v0.1; Phase 1 shipped 0.2.0 / tag v0.2; Phase 2's
first story bumps to 0.3.0.dev0).
"""

__version__ = "0.3.0.dev0"

__all__ = ["__version__"]
