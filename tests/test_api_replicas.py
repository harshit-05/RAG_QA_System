"""A stateless query tier (NFR-5, DEC-23, S3-8): any replica answers any request.

Two apps, each built as ``rag-serve`` builds it (its own lifespan, store client, models,
key ring and rate buckets), over one Qdrant (the test's in-memory client) and one state folder
(the test's ``var/``, which holds the job database and the job locks). Nothing a request
depends on lives in one of them alone:

- the index generation served follows the alias, read on every request (S3-2);
- the ingest jobs are rows in SQLite, and their locks files beside it;
- the keys are a file;
- what each caches (the generation and its manifest, the probes' answers, the rate buckets)
  is rebuilt from those: a third app, started afterwards, serves the same.

What stays per replica, and is documented as such (README, "Access"): the answer slots and
the rate limits, and the one Ollama every replica queues on. Ingestion runs on one host.
"""

import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pytest
from conftest import UNLIMITED, MakeConfig, use_fakes
from fastapi.testclient import TestClient

from rag_qa.api import auth
from rag_qa.api.app import create_app
from rag_qa.config import load_config
from rag_qa.ingest import ingest
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ServerSettings


@pytest.fixture
def config(
    make_config: MakeConfig, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> RagConfig:
    """The fakes' config over the sample corpus, with no index yet."""
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    return load_config(make_config(use_fakes))


@pytest.fixture
def keys(config: RagConfig) -> tuple[Path, str]:
    """A keys file in the shared state folder, with one key of both scopes, and its token."""
    path = config.paths.state / "api-keys.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    key = auth.ApiKey("laptop", frozenset(auth.SCOPES), auth.digest(token), "2026-10-26T00:00:00Z")
    path.write_text(auth.dump([key]))
    return path, token


@contextmanager
def replicas(config: RagConfig, keys_file: Path, count: int) -> Iterator[list[TestClient]]:
    """``count`` apps over the one store and state folder, each with its lifespan running."""
    settings = ServerSettings(keys_file=keys_file, rate_limit=UNLIMITED)
    with ExitStack() as stack:
        yield [
            stack.enter_context(
                TestClient(create_app(config, settings), base_url="http://127.0.0.1")
            )
            for _ in range(count)
        ]


def wait_for(client: TestClient, job_id: str, headers: dict[str, str]) -> dict[str, Any]:
    for _ in range(1500):
        job = client.get(f"/v1/ingest/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed"):
            return dict(job)
        time.sleep(0.02)
    raise AssertionError(f"ingest job {job_id} did not end")


def test_a_job_started_on_one_replica_is_seen_and_refused_on_the_other_and_its_index_served_by_both(
    config: RagConfig, keys: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    keys_file, token = keys
    auth_header = {"Authorization": f"Bearer {token}"}
    release = threading.Event()
    real_ingest = ingest

    def held_ingest(config: RagConfig, **kw: Any) -> Any:
        assert release.wait(30)
        return real_ingest(config, **kw)

    monkeypatch.setattr("rag_qa.api.jobs.ingest", held_ingest)
    with replicas(config, keys_file, 2) as (one, two):
        # Neither serves an index yet; the key, a file, works on both.
        for client in (one, two):
            assert client.get("/v1/documents", headers=auth_header).status_code == 503

        started = one.post("/v1/ingest", json={"rebuild": False}, headers=auth_header)
        assert started.status_code == 202
        job_id = started.json()["job_id"]

        # Visible on the other replica, and refused there with 409, naming it.
        seen = two.get(f"/v1/ingest/{job_id}", headers=auth_header).json()
        assert (seen["status"], seen["abandoned"]) == ("queued", False)
        refused = two.post("/v1/ingest", json={"rebuild": True}, headers=auth_header)
        assert refused.status_code == 409
        assert job_id in refused.json()["detail"]

        release.set()
        finished = wait_for(two, job_id, auth_header)  # read on the replica that did not run it
        assert (finished["status"], finished["exit_code"]) == ("succeeded", 0)
        assert finished["report"]["published"] is not None

        # The other replica serves what the first published, at its next request: it reads
        # the alias on every one.
        listed = two.get("/v1/documents", headers=auth_header)
        assert listed.status_code == 200
        documents = listed.json()["documents"]
        assert documents  # the sample corpus, as the first replica published it
        assert one.get("/v1/documents", headers=auth_header).json()["documents"] == documents
        health = two.get("/v1/health").json()
        assert health["index"]["generation"] == finished["report"]["published"]
        answer = two.post("/v1/query", json={"question": "How many staff?"}, headers=auth_header)
        assert answer.status_code == 200 and "event: done" in answer.text

        # Either may start the next job now; the first's record is the same on both.
        assert one.get(f"/v1/ingest/{job_id}", headers=auth_header).json() == finished
        assert (
            two.post("/v1/ingest", json={"rebuild": False}, headers=auth_header).status_code == 202
        )


def test_a_replica_started_afterwards_rebuilds_everything_from_the_shared_state(
    config: RagConfig, keys: tuple[Path, str]
) -> None:
    keys_file, token = keys
    auth_header = {"Authorization": f"Bearer {token}"}
    with replicas(config, keys_file, 1) as (one,):
        job_id = one.post("/v1/ingest", json={"rebuild": False}, headers=auth_header).json()[
            "job_id"
        ]
        finished = wait_for(one, job_id, auth_header)
        documents = one.get("/v1/documents", headers=auth_header).json()
    # The first is gone; a new one knows the job, the key and the index without it.
    with replicas(config, keys_file, 1) as (fresh,):
        assert fresh.get(f"/v1/ingest/{job_id}", headers=auth_header).json() == finished
        assert fresh.get("/v1/documents", headers=auth_header).json() == documents
        health = fresh.get("/v1/health").json()
        assert health["index"]["generation"] == finished["report"]["published"]
