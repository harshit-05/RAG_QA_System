"""The HTTP API: ``rag-serve`` (FR-6, SRS §8.1, ARCHITECTURE.md DEC-18).

- ``models``  what requests may say and what responses carry
- ``state``   what is served: the pipeline, its index generation, the slots
- ``jobs``    ingest jobs, one at a time across replicas, their records in SQLite
- ``auth``    API keys and their scopes, read from the keys file
- ``limits``  the body-size limit and the rate limits
- ``keys_cli`` the ``rag-keys`` command
- ``routes``  the endpoints, and the refusals that run before them
- ``app``     ``create_app``: the FastAPI application and its lifespan
- ``server``  the ``rag-serve`` command, under uvicorn

Dependency direction (§2.2): ``api → {answering, chain, ingest, manifest, components,
config, settings}``. Nothing outside this package imports it.
"""
