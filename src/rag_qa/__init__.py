"""RAG_QA_System: a config-driven local retrieval-augmented question-answering engine.

Module layout is fixed by ARCHITECTURE.md §0.2, §1.2 and §2.2:

- ``registry``    ``_target_`` resolution and recursive object construction (pure Python)
- ``schema``      what a valid config is: the frozen Pydantic ``RagConfig`` (pure Python)
- ``settings``    the ``RAG_*`` environment overrides
- ``config``      reads the config file, returns a validated ``RagConfig``
- ``loaders``     PDF / DOCX / text loaders on ``pypdf`` and ``docx2txt``
- ``components``  builds the embedder, splitter, LLM and loaders from a ``RagConfig``
- ``rerankers``   our own cross-encoder reranker on ``sentence-transformers`` (DEC-16)
- ``manifest``    what each index generation holds: file hashes, identities, chunk IDs
- ``vectorstore`` the only FAISS touchpoints, generations and their flip; the Phase 3 seam
- ``ingest``      corpus walk, diff against the manifest, incremental index update
- ``chain``       ``build_query_pipeline(cfg) -> QueryPipeline``; ``build_rag_chain`` for invoke
- ``answering``   ``stream_answer``: one answer as typed events, for every front end
- ``cli``         interactive REPL over that stream
- ``api``         ``rag-serve``: the HTTP API, SSE over that stream, ingest jobs (S2-7)
- ``evaluation``  ``rag-eval``: the golden set, tier-1 retrieval scores, and tier-2
                  generation, RAGAs scoring and the freshness-and-floors check

Version tracks the release tags: ``.dev0`` between tags, bumped at each phase exit
(Phase 0 shipped 0.1.0 / tag v0.1; Phase 1 shipped 0.2.0 / tag v0.2; Phase 2
shipped 0.3.0 / tag v0.3).
"""

__version__ = "1.0.0.dev0"

__all__ = ["__version__"]
