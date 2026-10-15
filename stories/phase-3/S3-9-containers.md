# S3-9: Containers, compose, TLS and CI scanning

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | SRS §11 (CI: container build and vulnerability scan, merges blocked on red; containerization; config management); SRS §8.3 (environment overlays); SRS §9 LLM05 (the image scanned); DEC-24; backlog: the Docker image's install, its stop signal and a forced stop; the audit-gate policy; TLS |
| **Depends on** | S3-8 |
| **Model** | Opus 5.5 at max effort (Docker and CI surface new to this repository, with known traps: the torch variant, PID 1's signals, SSE through a proxy) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-9-containers`. Risky: CI changes (the `workflow` scope, caveat 13), a supply-chain decision, and the first network-facing deployment. |

## Goal

The system runs as a container stack from one command, with TLS at the edge, and CI builds
and scans what it ships.

- **The image** installs CPU torch only, carries its models, and runs offline.
- **Compose** wires the app, Qdrant and Ollama, as SRS §11 asks. Caddy terminates TLS and
  balances app replicas, which shows NFR-5 for real.
- **An overlay** adapts the config to the container network without editing it (§8.3).
- **CI scans the image and the repository** with Trivy, run as a binary pinned by its
  hash (caveat 30), under a written policy for advisories with no fix.

Design: ARCHITECTURE.md §3.1, DEC-24.

## Scope

- **`Dockerfile`**, multi-stage; every base image pinned by digest.
  - **Install:** `ghcr.io/astral-sh/uv:0.11.8` on `debian:trixie-slim`, with uv-managed
    CPython 3.12.13 (`.python-version`). `uv sync --locked --no-dev --no-install-project`,
    then the source, then `uv sync --locked --no-dev --no-editable`. **Never** `pip install
    .` or `--no-default-groups` (the backlog line; CLAUDE.md).
  - **A build step fails unless** torch's version ends in `+cpu`, and no `nvidia-*`
    distribution is installed.
  - **A `models` stage** runs `build_embedder` and `build_reranker` on `config.yaml`, as
    CI's warm-up does, then copies the Hugging Face cache into the runtime.
  - **Runtime:**
    - a non-root user;
    - `HF_HUB_OFFLINE=1`;
    - `HEALTHCHECK` on `/v1/health/live`;
    - `ENTRYPOINT ["rag-serve"]`, `CMD ["--host", "0.0.0.0"]`, which needs a key or a token:
      DEC-18's refusal stands.
- **`.dockerignore`:** `.git`, `var/`, `vectorstore/`, `backups/`, the virtual
  environment, and the caches.
- **Overlays (§8.3).**
  - `RAG_CONFIG_OVERLAY` (in `settings.py`) names a YAML file that `config.py` deep-merges
    over the base before validating: mappings merge, lists and scalars replace.
  - Paths resolve against the base config's folder.
  - Nothing is loaded unless named, and README says why.
  - **`config.docker.yaml`** (committed) sets `base_url: http://ollama:11434` on the llm
    entries (a runtime key) and the store's `url: http://qdrant:6333`.
  - **Tests:** the merge rules, and the six fingerprint parts unchanged by an overlay that
    sets only runtime keys.
- **`compose.yaml`**, extending S3-2's; every image pinned by digest.
  - **`qdrant`:** v1.19.2, published on 127.0.0.1 only, with its API key from `.env`
    (`RAG_QDRANT_API_KEY`).
  - **`ollama`:** 0.22.1, the host's version (ADR-025), with a named volume and a one-shot
    `ollama-pull` service for the configured model. README: unload the host daemon's model
    first (`ollama stop mistral`), since both would hold about 5 GB.
  - **`app`:**
    - the image, with `init: true` and an `env_file` of `.env`;
    - mounts: `./var` (the lock and the job database, shared with the host's CLI),
      `./corpus` read-only, and the overlay;
    - it waits for healthy dependencies (`depends_on`);
    - it scales with `--scale app=2`.
  - **`caddy`:** 2.11.7, with a `deploy/Caddyfile`.
    - **TLS:** `tls internal` for a local name; ACME when a public name is set.
    - **Upstreams:** `reverse_proxy` over the `app` replicas, health-checked on
      `/v1/health/live`. Liveness, not readiness: a stack with no index must still accept
      the ingest that builds one.
    - **SSE and bodies:** `flush_interval -1`, and `request_body max_size 64KB`.
  - **Profile `observability`:** Jaeger 2.22.0, Prometheus v3.15.0 (scraping the app's
    metrics port on the internal network) and Alertmanager v0.34.1, with S3-7's files.
  - **`.env.example`** (committed): every variable named, no values. `.env` stays gitignored.
- **Signals** (the backlog line). With `init: true`, `docker stop` ends `rag-serve` with
  status 143, which stays. Check, and record here:
  - the status with and without the init;
  - that the lifespan's shutdown lines appear in `docker compose logs`;
  - that a second SIGTERM does not force a stop.
- **CI** (needs the `workflow` scope, caveat 13).
  - **The `container` job:**
    - `docker/setup-buildx-action` v4.4.1 and `docker/build-push-action` v7.4.0, by SHA,
      with GitHub's cache, and no push;
    - a smoke run: `/v1/health/live` answers 200, `rag-ingest --help` exits 0, and torch is
      `+cpu` inside;
    - Trivy v0.75.0, as its release tarball checked against a sha256 written in `ci.yml`
      (never `aquasecurity/trivy-action`, caveat 30): `trivy image --severity
      HIGH,CRITICAL --ignore-unfixed --exit-code 1` and `trivy fs --scanners
      secret,misconfig --exit-code 1 .`.
  - **`deploy-config`** (S3-7's job) adds `docker compose config -q`, with `.env.example`
    as the environment.
  - **The audit policy,** as DEC-24 states it:
    - `--ignore-vuln` entries only for advisories with no fixed release, each with a reason
      and a date in `ci.yml` and a STATUS.md line;
    - `.trivyignore.yaml` entries, each with a statement and an `expired_at` at most 90 days
      out;
    - a weekly `schedule` run of the whole workflow on `main`;
    - an Audit step for the `eval` extra, from `uv export --extra eval` with no install. Its
      first entry is PYSEC-2026-2192 (`langchain` 1.3.2), "fixed by S3-11's bump".
  - **The ruleset recipe,** for the maintainer to run: a `gh api` call that requires
    `check`, `eval-retrieval`, `deploy-config` and `container` on `main`. Record whether
    the repository's plan allows it.
- **README:** a "Deployment" section covering:
  - `cp .env.example .env`, keys (`rag-keys`), `docker compose up -d`, the HTTPS address;
  - scaling to two replicas, and the memory note;
  - the observability profile;
  - what 143 means on stop.

  `rag-serve`'s cleartext warning now names this recipe.

## Out of scope

- Backups and their schedule (S3-10).
- Publishing the image to a registry.
- Kubernetes.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
uv run ruff check && uv run ruff format --check && uv run mypy

docker build -t rag-qa:s3-9 .                              # fails if torch is not +cpu
docker run --rm --entrypoint python rag-qa:s3-9 -c "import torch; print(torch.__version__)"   # +cpu
cp .env.example .env   # then fill in a token and Qdrant's key, by hand
docker compose up -d && docker compose ps                  # all healthy
curl -sk https://localhost/v1/health/live                  # 200, through Caddy's TLS
# ingest over HTTPS, then ask a question; with --scale app=2, both replicas serve the
# new generation, and either one reports the job
# a client dropped through Caddy during the prefill: our connection to Ollama closes
#   within ~1 s (the app's log and Ollama's journal), as DEC-14 requires
docker compose stop app; docker compose ps -a              # Exited (143); shutdown lines logged
docker compose --profile observability up -d               # Jaeger and Prometheus see the app
gh pr checks                                               # container and deploy-config green
```

## Review notes for the human

- **The Trivy step:** the sha256 is written in `ci.yml`, and the binary is checked before it
  runs.
- **The Dockerfile's install line, and its `+cpu` check.**
- **Caddy's health check is liveness,** and its body limit matches the app's.
- **`.env.example` holds names only.**

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
