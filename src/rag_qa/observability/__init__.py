"""Observability (NFR-10, DEC-22): what the system says about itself while it runs.

- :mod:`.logs`: structured log lines for every command, ours and the libraries' alike (S3-7).
- :mod:`.metrics`: the numbers the SLOs are written in, for Prometheus to scrape (S3-7).
- :mod:`.genai`: what the GenAI semantic conventions call a chat model (S3-7).

Traces and alerts are S3-7b's. The package loads no model stack, so any module may import it
(ARCHITECTURE.md §3.2, ``tests/test_architecture.py``).
"""
