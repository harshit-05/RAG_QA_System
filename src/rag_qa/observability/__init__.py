"""Observability (NFR-10, DEC-22): what the system says about itself while it runs.

- :mod:`.logs`: structured log lines for every command, ours and the libraries' alike (S3-7).
- :mod:`.metrics`: the numbers the SLOs are written in, for Prometheus to scrape (S3-7).
- :mod:`.genai`: what the GenAI semantic conventions call a chat model, and its span (S3-7,
  S3-7b).
- :mod:`.tracing`: the spans of every answer and ingest run, for a collector to keep, while one
  is named (S3-7b).

The alert rules over the metrics live in ``deploy/prometheus/`` (S3-7b). The package loads no
model stack, so any module may import it (ARCHITECTURE.md §3.2, ``tests/test_architecture.py``).
"""
