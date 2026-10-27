"""The ingest jobs' records (S3-8, DEC-23): SQLite rows any replica reads, and the job's own
lock that tells a dead job from a live one.

Two :class:`rag_qa.api.jobs.JobStore` objects over one state folder stand for two replicas: a
``flock`` belongs to the open file, so one process's two opens conflict as two processes'
would. A job whose process dies is a real child process, killed with SIGKILL, since only the
kernel's release of its lock is the real thing. Through the HTTP API, ``test_api.py`` runs the
jobs, and ``test_api_replicas.py`` two apps over one state folder.
"""

import asyncio
import fcntl
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from conftest import UNLIMITED, MakeConfig, StreamingFakeChatModel, fake_serving, use_fakes
from fastapi.testclient import TestClient

from rag_qa.api import jobs as api_jobs
from rag_qa.api.app import create_app
from rag_qa.api.jobs import JobRunningError, JobStore, held
from rag_qa.config import load_config
from rag_qa.ingest import IngestReport
from rag_qa.observability import metrics
from rag_qa.settings import ENV_DATA_PATH, ServerSettings


def failed_runs() -> float:
    return metrics.REGISTRY.get_sample_value("rag_ingest_runs_total", {"status": "failed"}) or 0.0


@pytest.fixture
def state(tmp_path: Path) -> Path:
    return tmp_path / "var"


def test_the_database_is_in_wal_mode_in_the_state_folder(state: Path) -> None:
    JobStore(state)
    with sqlite3.connect(state / "jobs.sqlite3") as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_a_job_started_by_one_replica_is_read_and_refused_by_another(state: Path) -> None:
    one, two = JobStore(state), JobStore(state)
    lock = one.start("a" * 32, rebuild=True, api_key="laptop")
    job = two.get("a" * 32)
    assert job is not None
    assert (job.status, job.rebuild, job.api_key, job.abandoned) == (
        "queued",
        True,
        "laptop",
        False,
    )
    with pytest.raises(JobRunningError) as refused:
        two.start("b" * 32, rebuild=False, api_key=None)
    assert refused.value.job.job_id == "a" * 32
    assert two.get("b" * 32) is None  # the refused job left no row
    assert not two.lock_path("b" * 32).exists()  # nor its lock file
    one.running("a" * 32)
    one.finish("a" * 32, status="succeeded", exit_code=0, report={"chunks": 3})
    lock.release()
    finished = two.get("a" * 32)
    assert finished is not None
    assert (finished.status, finished.exit_code, finished.report) == ("succeeded", 0, {"chunks": 3})
    two.start("b" * 32, rebuild=False, api_key=None).release()  # one may start again


def test_two_replicas_starting_at_once_start_one_job(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BEGIN IMMEDIATE takes the write lock before the check, so the second start waits for the
    # first to commit, then sees its job and refuses. A deferred transaction would check before
    # the first committed, find nothing, and fail its own insert (or, without the lock, insert
    # a second job).
    one, two = JobStore(state), JobStore(state)
    inside, release = threading.Event(), threading.Event()
    real_prune = JobStore._prune

    def paused_prune(self: JobStore, db: sqlite3.Connection) -> None:
        if threading.current_thread().name == "one":
            inside.set()
            assert release.wait(5)
        real_prune(self, db)

    monkeypatch.setattr(JobStore, "_prune", paused_prune)
    outcomes: dict[str, object] = {}

    def start(store: JobStore, name: str) -> None:
        try:
            outcomes[name] = store.start(name * 16, rebuild=False, api_key=None)
        except Exception as e:  # noqa: BLE001  # the outcome is the test's to read
            outcomes[name] = e

    first = threading.Thread(target=start, args=(one, "one"), name="one")
    first.start()
    assert inside.wait(5)  # the first is inside its transaction, its row not yet committed
    second = threading.Thread(target=start, args=(two, "two"), name="two")
    second.start()
    second.join(0.3)
    assert second.is_alive()  # waiting for the write lock
    release.set()
    first.join(5)
    second.join(15)
    assert isinstance(outcomes["one"], api_jobs.JobLock)
    assert isinstance(outcomes["two"], JobRunningError), outcomes["two"]
    assert outcomes["two"].job.job_id == "one" * 16


def test_the_lock_is_held_from_before_the_row_until_after_its_end(state: Path) -> None:
    store = JobStore(state)
    lock = store.start("a" * 32, rebuild=False, api_key=None)
    assert held(store.lock_path("a" * 32))
    store.finish("a" * 32, status="failed", exit_code=2, report=None)
    assert held(store.lock_path("a" * 32))  # the end is written first
    lock.release()
    assert not held(store.lock_path("a" * 32))
    assert not store.lock_path("a" * 32).exists()


def test_a_probe_never_creates_the_lock_file(state: Path) -> None:
    store = JobStore(state)
    store.start("a" * 32, rebuild=False, api_key=None).release()  # the folder exists
    path = store.lock_path("b" * 32)
    assert path.parent.is_dir()
    assert not held(path)
    assert not path.exists()


def test_a_probe_never_takes_another_probe_for_the_lock_s_owner(state: Path) -> None:
    # Two replicas reading a dead job at once probe its lock together (S3-8's second review):
    # an exclusive probe took the other's lock for the owner's.
    path = JobStore(state).lock_path("a" * 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()  # the file a dead job's process left
    with open(path, "rb") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_SH)  # another probe, mid-way
        assert not held(path)
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)  # an owner
        assert held(path)


def test_the_log_is_appended_line_by_line_and_read_by_any_replica(state: Path) -> None:
    one, two = JobStore(state), JobStore(state)
    one.start("a" * 32, rebuild=False, api_key=None)
    for line in ("Loading 3 documents", 'a "quoted" line', "é"):
        one.log("a" * 32, line)
    job = two.get("a" * 32)
    assert job is not None and job.log == ["Loading 3 documents", 'a "quoted" line', "é"]


# --- abandoned jobs ------------------------------------------------------------------------------

_DIES = """
import os, signal, sys
from pathlib import Path
from rag_qa.api.jobs import JobStore
store = JobStore(Path(sys.argv[1]))
store.start(sys.argv[2], rebuild=False, api_key=None)
store.running(sys.argv[2])
print("started", flush=True)
os.kill(os.getpid(), signal.SIGKILL)
"""


def die_mid_job(state: Path, job_id: str) -> None:
    """A process that starts a job, marks it running, and is killed with SIGKILL."""
    result = subprocess.run(
        [sys.executable, "-c", _DIES, str(state), job_id],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.stdout.strip() == "started", result.stderr[-2000:]
    assert result.returncode == -signal.SIGKILL


def test_a_job_whose_process_was_killed_reads_failed_and_abandoned_once(state: Path) -> None:
    store = JobStore(state)
    die_mid_job(state, "d" * 32)
    before = failed_runs()
    job = store.get("d" * 32)
    assert job is not None
    assert (job.status, job.abandoned, job.exit_code, job.report) == ("failed", True, None, None)
    assert job.finished_at is not None
    assert failed_runs() - before == 1
    assert not store.lock_path("d" * 32).exists()  # the file the dead process left, removed
    assert store.get("d" * 32) == job  # the verdict is written: read again, it counts no more
    assert failed_runs() - before == 1


def test_a_dead_job_s_row_never_holds_the_409(state: Path) -> None:
    store = JobStore(state)
    die_mid_job(state, "d" * 32)
    before = failed_runs()
    store.start("e" * 32, rebuild=False, api_key=None).release()  # no 409
    assert not store.lock_path("d" * 32).exists()
    dead = store.get("d" * 32)
    assert dead is not None and (dead.status, dead.abandoned) == ("failed", True)
    assert failed_runs() - before == 1  # judged once, by the start


def test_a_start_is_not_refused_while_another_replica_probes_a_dead_job(state: Path) -> None:
    # The 409 a pair of exclusive probes gave (S3-8's second review): 42 of 300 starts while
    # another process probed the dead job's lock in a loop.
    store = JobStore(state)
    die_mid_job(state, "d" * 32)
    with open(store.lock_path("d" * 32), "rb") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_SH)  # another replica's probe, mid-way
        store.start("e" * 32, rebuild=False, api_key=None).release()  # no 409
    dead = store.get("d" * 32)
    assert dead is not None and (dead.status, dead.abandoned) == ("failed", True)


async def nothing() -> None:
    return None


def test_a_replica_judges_a_dead_job_as_it_starts_though_nobody_asks(jobs_config: Any) -> None:
    # A crash mid-ingest, and no client asks again (S3-8's second review): the replica that
    # starts next counts it, so RagIngestFailed fires, and removes the lock file it left.
    state = jobs_config.paths.state
    die_mid_job(state, "d" * 32)
    before = failed_runs()
    jobs = api_jobs.IngestJobs(jobs_config, after=nothing)
    assert failed_runs() - before == 1
    assert not jobs.store.lock_path("d" * 32).exists()
    with closing(sqlite3.connect(state / "jobs.sqlite3")) as db:
        assert db.execute("SELECT status, abandoned FROM jobs").fetchall() == [("failed", 1)]
    job = jobs.store.get("d" * 32)
    assert job is not None and (job.status, job.abandoned) == ("failed", True)
    assert failed_runs() - before == 1  # judged once


def test_a_replica_starting_beside_a_live_job_leaves_it_running(jobs_config: Any) -> None:
    store = JobStore(jobs_config.paths.state)
    lock = store.start("a" * 32, rebuild=False, api_key=None)
    store.running("a" * 32)
    before = failed_runs()
    api_jobs.IngestJobs(jobs_config, after=nothing)
    job = store.get("a" * 32)
    assert job is not None and (job.status, job.abandoned) == ("running", False)
    assert failed_runs() == before
    assert store.lock_path("a" * 32).exists()
    lock.release()


def test_a_job_that_ends_between_the_read_and_the_probe_is_not_abandoned(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The race design review 2 measured (92 of 400 runs without the second read): the row is
    # read unfinished, then the job writes its end and releases its lock, then the probe finds
    # the lock free. Reading the row again sees the end.
    store = JobStore(state)
    lock = store.start("a" * 32, rebuild=False, api_key=None)
    real_held = api_jobs.held

    def ends_first(path: Path) -> bool:
        store.finish("a" * 32, status="succeeded", exit_code=0, report=None)
        lock.release()
        return real_held(path)

    monkeypatch.setattr(api_jobs, "held", ends_first)
    before = failed_runs()
    job = store.get("a" * 32)
    assert job is not None
    assert (job.status, job.abandoned) == ("succeeded", False)
    assert failed_runs() == before


def test_a_verdict_is_written_once_and_never_over_a_final_status(state: Path) -> None:
    # Two guards close the race above: the second read, and this conditional write, each
    # enough alone. This one also makes the count exactly once when two replicas judge the
    # same dead row: only the write that changed it reports it.
    store = JobStore(state)
    for job_id in ("a" * 32, "b" * 32):
        lock = store.start(job_id, rebuild=False, api_key=None)
        if job_id == "a" * 32:
            store.finish(job_id, status="succeeded", exit_code=0, report=None)
        lock.release()
    with store._connect() as db:
        assert store._abandon(db, "a" * 32) is False  # ended: left as it is
        assert [store._abandon(db, "b" * 32) for _ in range(2)] == [True, False]
    ended, dead = store.get("a" * 32), store.get("b" * 32)
    assert ended is not None and (ended.status, ended.abandoned) == ("succeeded", False)
    assert dead is not None and (dead.status, dead.abandoned) == ("failed", True)


def test_the_newest_jobs_are_kept(state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_jobs, "KEEP", 3)
    store = JobStore(state)
    ids = [f"{n:032d}" for n in range(6)]
    for job_id in ids:
        lock = store.start(job_id, rebuild=False, api_key=None)
        store.finish(job_id, status="succeeded", exit_code=0, report=None)
        lock.release()
    assert [store.get(job_id) is not None for job_id in ids] == [False] * 3 + [True] * 3


# --- a job's run -------------------------------------------------------------------------------


@pytest.fixture
def jobs_config(
    make_config: MakeConfig, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    return load_config(make_config(use_fakes))


def test_a_job_is_queued_until_ingest_holds_the_writer_lock_then_running(
    jobs_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    waiting, locked, release = threading.Event(), threading.Event(), threading.Event()

    def fake_ingest(config: Any, *, rebuild: bool, on_locked: Any = None) -> IngestReport:
        waiting.wait(5)
        on_locked()
        locked.set()
        release.wait(5)
        return IngestReport(found=1)

    monkeypatch.setattr("rag_qa.api.jobs.ingest", fake_ingest)
    # A job writes its end while it still holds its lock (the review note): a reader that finds
    # the lock free must find the end.
    held_at_finish: list[bool] = []
    real_finish = JobStore.finish

    def finish(self: JobStore, job_id: str, **kw: Any) -> None:
        held_at_finish.append(held(self.lock_path(job_id)))
        real_finish(self, job_id, **kw)

    monkeypatch.setattr(JobStore, "finish", finish)

    async def scenario() -> list[str]:
        async def nothing() -> None:
            return None

        jobs = api_jobs.IngestJobs(jobs_config, after=nothing)
        job = await jobs.start(rebuild=False, api_key="laptop")
        statuses = []
        seen = await jobs.get(job.job_id)
        while seen is not None and seen.started_at is None:  # its thread has begun
            await asyncio.sleep(0.01)
            seen = await jobs.get(job.job_id)
        statuses.append(seen.status if seen else "?")
        waiting.set()
        await asyncio.to_thread(locked.wait, 5)
        seen = await jobs.get(job.job_id)
        statuses.append(seen.status if seen else "?")
        release.set()
        while jobs.running is not None or jobs._tasks:
            await asyncio.sleep(0.01)
        seen = await jobs.get(job.job_id)
        statuses.append(seen.status if seen else "?")
        assert seen is not None and seen.started_at is not None and seen.api_key == "laptop"
        assert not jobs.store.lock_path(job.job_id).exists()
        return statuses

    assert asyncio.run(scenario()) == ["queued", "running", "succeeded"]
    assert held_at_finish == [True]


def test_a_real_ingest_marks_its_job_running_inside_the_writer_lock(jobs_config: Any) -> None:
    # The real ingest() calls on_locked while it holds the writer lock (DEC-17), before it
    # changes anything.
    from rag_qa.ingest import LOCK_FILE, ingest

    seen: list[bool] = []

    def probe() -> None:
        fd = os.open(jobs_config.paths.state / LOCK_FILE, os.O_RDONLY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                seen.append(True)
            else:
                seen.append(False)
        finally:
            os.close(fd)

    ingest(jobs_config, on_locked=probe)
    assert seen == [True]


def test_over_http_a_killed_job_reads_failed_and_abandoned_and_another_may_start(
    jobs_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The verification's kill -9, hermetically: the dead process's job, read by a server that
    # never ran it.
    monkeypatch.setattr("rag_qa.api.jobs.ingest", lambda config, *, rebuild, **_: IngestReport())
    die_mid_job(jobs_config.paths.state, "d" * 32)
    serving = fake_serving(jobs_config, StreamingFakeChatModel())
    app = create_app(jobs_config, ServerSettings(rate_limit=UNLIMITED), serving=serving)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        job = client.get(f"/v1/ingest/{'d' * 32}").json()
        started = client.post("/v1/ingest", json={"rebuild": False})
    assert (job["status"], job["abandoned"], job["exit_code"], job["report"]) == (
        "failed",
        True,
        None,
        None,
    )
    assert started.status_code == 202
