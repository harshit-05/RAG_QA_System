"""Evaluation: the golden set and the two quality gates (FR-7, DEC-15).

Phase 2 builds this package story by story (ARCHITECTURE.md §2.2):

- ``dataset``      the golden set's schema and loader (S2-2); pure Python, no LangChain
- ``retrieval``    tier-1 retrieval metrics (S2-3)
- ``gate``         the floors in ``eval/thresholds.yaml`` (S2-3); light, for CI's
                   model-free job. S2-5 adds tier 2's floors and the freshness check
- ``generation``, ``ragas_scoring``, ``fingerprint``: tier 2 (S2-5)
- ``cli``          ``rag-eval`` (S2-3, S2-5)

This ``__init__`` imports nothing, so ``rag_qa.evaluation.dataset`` stays importable
without the ML stack (``tests/test_architecture.py``).
"""
