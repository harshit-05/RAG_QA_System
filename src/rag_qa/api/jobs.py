"""Ingest jobs: ``POST /v1/ingest`` runs one at a time, in the background (DEC-18, DEC-23).

A job runs :func:`rag_qa.ingest.ingest`, the function behind ``rag-ingest``, and keeps
what that command would have shown: the report, the exit status and the log lines. All of
them are path-free (S2-6), since they go over HTTP.

**The records are durable and shared** (S3-8, NFR-5): a row per job in SQLite, at
``jobs.sqlite3`` in ``paths.state`` (``var/``), in WAL mode, so any replica that shares the
folder answers for any job, and the 409 holds whichever replica is asked
(:class:`JobStore`). The last :data:`KEEP` jobs are kept.

- **One at a time, across replicas.** :meth:`JobStore.start` checks for an unfinished job
  and inserts the new one in one ``BEGIN IMMEDIATE`` transaction, so two replicas asked at
  once cannot both start one. A ``rag-ingest`` running outside the server holds the writer
  lock (DEC-17), so a job started then fails at once with exit status 2 and that command's
  own message: a route cannot hold the ``flock`` on the job's behalf, because ``ingest()``
  takes it itself.
- **Queued, then running:** a row reads ``running`` only once ``ingest()`` holds the writer
  lock (its ``on_locked`` callback), so a job that waits on nothing reads ``queued``.
- **An abandoned job is told by a lock of its own** (DEC-23, design review 1). Before it
  inserts a job's row, the replica takes a ``flock`` on ``jobs/<id>.lock``, and holds it until
  the row's final status is written. The kernel releases a ``flock`` with its process, so an
  unfinished row whose job lock is free belongs to a process that died: it reads as
  ``failed``, with ``abandoned: true``. DEC-17's and DEC-19's recovery cleans up what the run
  left, at the next run. The writer lock is never probed: a writer starting at that moment
  would exit 2.
- **In this order** (design review 2): read the row, probe its job lock, and when the lock is
  free, read the row again; only a row still unfinished is abandoned, written back with
  ``UPDATE … WHERE status IN ('queued', 'running')``, which never overwrites a final status. A
  job that finished between the read and the probe would otherwise read as abandoned. The
  start transaction applies the same rule, so a dead job's row never holds the 409.
- **In a daemon thread.** An ingest cannot be interrupted and can take minutes, so stopping
  the server abandons a job rather than waiting for it. That is safe: a run never writes into
  the live index generation, and the next run cleans up what it left (DEC-17, DEC-19).
- **What it changes.** When a job ends, ``after`` runs (this replica's refresh), so a
  published generation is served at once here; another replica picks it up at its next
  request, as it does a ``rag-ingest`` run's.
- **What it reports** (S3-7): the job's log lines carry its ``job_id``, and the key that started
  it (S3-8). Each job counts in ``rag_ingest_runs_total`` by how it ended, its embedded chunks in
  ``rag_ingest_chunks_total``; an abandoned one counts as failed, once, on the replica that
  finds it, since the process it ran in can count nothing. A ``rag-ingest`` run outside the
  server has no metrics listener: it reports in its log.
- **Its trace** (S3-7b): while tracing is on, the job's ``rag.ingest`` span is the child of the
  ``POST /v1/ingest`` request that started it, in that request's trace, though it ends minutes
  after the request does. A job abandoned at shutdown never ends its spans: they go with it.
"""

import asyncio
import fcntl
import functools
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import closing, contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, cast

from opentelemetry.context import Context

from rag_qa.api.models import JobStatus
from rag_qa.config import ConfigError
from rag_qa.ingest import (
    EXIT_CANNOT_START,
    EXIT_OK,
    EXIT_RUN_FAILED,
    CorpusUnreadableError,
    IndexInTheWayError,
    IngestionRunningError,
    describe_error,
    ingest,
    outcome,
)
from rag_qa.ingest import logger as ingest_logger
from rag_qa.observability import logs, metrics, tracing
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import StoreUnavailableError

logger = logging.getLogger(__name__)

#: How many jobs the database keeps, the newest.
KEEP = 100
#: The job database and the folder of job locks, in ``paths.state``.
DATABASE = "jobs.sqlite3"
LOCKS = "jobs"
#: How long a write waits for another connection's, in seconds. Every transaction here is one
#: or two statements long.
BUSY_TIMEOUT_S = 10.0
#: A job's statuses before it ends.
UNFINISHED: tuple[JobStatus, ...] = ("queued", "running")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    rebuild INTEGER NOT NULL,
    api_key TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    exit_code INTEGER,
    report TEXT,
    log TEXT NOT NULL DEFAULT '[]',
    abandoned INTEGER NOT NULL DEFAULT 0
)
"""
_STILL_UNFINISHED = "status IN ('queued', 'running')"


@dataclass
class IngestJob:
    """One ``POST /v1/ingest``, as its row holds it."""

    job_id: str
    rebuild: bool
    api_key: str | None = None  # the name of the key that started it (S3-8), never a token
    status: JobStatus = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None  # what rag-ingest would have exited with
    #: :class:`rag_qa.ingest.IngestReport` as JSON holds it; ``None`` when the run could not
    #: start.
    report: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)
    #: Its process died before it ended (DEC-23): ``status`` is then ``failed``.
    abandoned: bool = False


class JobRunningError(Exception):
    """A job is running already. ``job`` is that one."""

    def __init__(self, job: IngestJob) -> None:
        super().__init__(f"ingest job {job.job_id} is running")
        self.job = job


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _job(row: sqlite3.Row) -> IngestJob:
    return IngestJob(
        job_id=row["job_id"],
        rebuild=bool(row["rebuild"]),
        api_key=row["api_key"],
        status=cast(JobStatus, row["status"]),
        started_at=_time(row["started_at"]),
        finished_at=_time(row["finished_at"]),
        exit_code=row["exit_code"],
        report=None if row["report"] is None else json.loads(row["report"]),
        log=json.loads(row["log"]),
        abandoned=bool(row["abandoned"]),
    )


def _abandoned(job_id: str, lock_path: Path) -> None:
    """Report the verdict this replica wrote on a dead job, once, by whoever wrote it, and
    remove the lock file its process could not: free, and never taken again, since no other job
    has its ID (found in S3-8's live run)."""
    lock_path.unlink(missing_ok=True)
    logger.warning(
        "Ingest job %s was abandoned: the process running it ended before it did. The live "
        "index changed only if it had published; the next ingest cleans up what it left.",
        job_id,
    )
    metrics.INGEST_RUNS.labels(status="failed").inc()


class JobLock:
    """A job's own lock, ``jobs/<id>.lock`` (the module docstring): taken before its row
    exists, released once its final status is written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: IO[bytes] | None = None

    def take(self) -> None:
        """Take it. The file is the job's own, new, so nothing else holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(self.path, "ab")  # noqa: SIM115  # held until release()
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock.close()
            raise
        self._file = lock

    def release(self) -> None:
        """Remove the file, then let the lock go: a probe that opens it after the removal
        finds no file, which reads as free, as the lock now is."""
        lock, self._file = self._file, None
        if lock is not None:
            self.path.unlink(missing_ok=True)
            lock.close()


def held(path: Path) -> bool:
    """Whether a job's lock is held now, by any process: this one's other threads included,
    since a ``flock`` belongs to the open file. Never creates the file: a missing one is free."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # releases the probe's own lock, if it got it
    return False


class JobStore:
    """The job records, in ``state/jobs.sqlite3``, and their locks in ``state/jobs/``.

    A connection per call, so any thread may use it: the event loop's (through
    :class:`IngestJobs`, in a worker thread) and each job's own. WAL mode lets readers go on
    while a job writes its log. ``synchronous=NORMAL``: a commit survives the process ending,
    however it ends, which is the case abandoned jobs are about; a power cut may lose the last
    few, never the database.
    """

    def __init__(self, state: Path) -> None:
        self.state = state
        self.path = state / DATABASE
        state.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection in autocommit mode: each statement is its own transaction, unless a
        ``BEGIN`` says otherwise."""
        with closing(
            sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
        ) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=NORMAL")
            yield db

    def lock_path(self, job_id: str) -> Path:
        return self.state / LOCKS / f"{job_id}.lock"

    def _row(self, db: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
        found: sqlite3.Row | None = db.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return found

    def _abandon(self, db: sqlite3.Connection, job_id: str) -> bool:
        """Write the verdict on a dead job's row, unless it ended meanwhile: whether this call
        wrote it. The caller reports it (:func:`_abandoned`) once it is committed."""
        cursor = db.execute(
            f"UPDATE jobs SET status = 'failed', abandoned = 1, finished_at = ? "
            f"WHERE job_id = ? AND {_STILL_UNFINISHED}",
            (_now(), job_id),
        )
        return cursor.rowcount == 1

    def start(self, job_id: str, *, rebuild: bool, api_key: str | None) -> JobLock:
        """Insert a queued job, holding its lock, and return the lock, which the job's thread
        releases once it has written the job's end. Raises :class:`JobRunningError` while
        another job runs, on any replica."""
        lock = JobLock(self.lock_path(job_id))
        lock.take()  # before the row exists, so no probe ever finds the row without it
        dead: list[str] = []
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")  # the write lock: no other start, or end, meanwhile
                try:
                    unfinished = db.execute(
                        f"SELECT * FROM jobs WHERE {_STILL_UNFINISHED}"
                    ).fetchall()
                    for row in unfinished:
                        # The write lock means the row cannot end meanwhile: a live job writes
                        # its end before it releases its lock, and cannot write now.
                        if held(self.lock_path(row["job_id"])):
                            raise JobRunningError(_job(row))
                        if self._abandon(db, row["job_id"]):
                            dead.append(row["job_id"])
                    db.execute(
                        "INSERT INTO jobs (job_id, rebuild, api_key, status, created_at) "
                        "VALUES (?, ?, ?, 'queued', ?)",
                        (job_id, int(rebuild), api_key, _now()),
                    )
                    self._prune(db)
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
                db.execute("COMMIT")
        except BaseException:
            lock.release()
            raise
        for job in dead:
            _abandoned(job, self.lock_path(job))
        return lock

    def _prune(self, db: sqlite3.Connection) -> None:
        """Keep the newest :data:`KEEP` jobs, and every unfinished one."""
        gone = [
            row["job_id"]
            for row in db.execute(
                f"SELECT job_id FROM jobs WHERE NOT ({_STILL_UNFINISHED}) AND seq NOT IN "
                f"(SELECT seq FROM jobs ORDER BY seq DESC LIMIT ?)",
                (KEEP,),
            )
        ]
        for job_id in gone:
            db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            self.lock_path(job_id).unlink(missing_ok=True)  # an abandoned job's, left behind

    def get(self, job_id: str) -> IngestJob | None:
        """A job as it is now, ``None`` if the database does not hold it. An unfinished one
        whose lock is free is judged abandoned, in the order the module docstring gives."""
        with self._connect() as db:
            row = self._row(db, job_id)
            if row is None or row["status"] not in UNFINISHED:
                return None if row is None else _job(row)
            if held(self.lock_path(job_id)):
                return _job(row)
            again = self._row(db, job_id)  # it may have ended between the read and the probe
            if again is not None and again["status"] in UNFINISHED:
                if self._abandon(db, job_id):  # autocommit: written already
                    _abandoned(job_id, self.lock_path(job_id))
                again = self._row(db, job_id)
            return None if again is None else _job(again)

    def started(self, job_id: str) -> None:
        """The job's thread began."""
        with self._connect() as db:
            db.execute("UPDATE jobs SET started_at = ? WHERE job_id = ?", (_now(), job_id))

    def running(self, job_id: str) -> None:
        """``ingest()`` holds the writer lock: the job runs."""
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = 'running' WHERE job_id = ? AND status = 'queued'",
                (job_id,),
            )

    def log(self, job_id: str, line: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET log = json_insert(log, '$[#]', ?) WHERE job_id = ?",
                (line, job_id),
            )

    def finish(
        self,
        job_id: str,
        *,
        status: JobStatus,
        exit_code: int | None,
        report: dict[str, Any] | None,
    ) -> None:
        """Write the job's end. Its lock is still held, so no probe has called it abandoned."""
        with self._connect() as db:
            db.execute(
                f"UPDATE jobs SET status = ?, exit_code = ?, report = ?, finished_at = ? "
                f"WHERE job_id = ? AND {_STILL_UNFINISHED}",
                (
                    status,
                    exit_code,
                    None if report is None else json.dumps(report),
                    _now(),
                    job_id,
                ),
            )


class _JobLog(logging.Handler):
    """Copies one thread's ``rag_qa.ingest`` records into a job's log, as plain text."""

    def __init__(self, store: JobStore, job_id: str, thread: int) -> None:
        super().__init__(logging.INFO)
        self._store = store
        self._job_id = job_id
        self._thread = thread

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread:
            try:
                self._store.log(self._job_id, record.getMessage())
            # Deliberately broad: a log line that cannot be stored must not end the run.
            except Exception:  # noqa: BLE001
                self.handleError(record)


class IngestJobs:
    """This replica's side of the jobs: it starts them, runs them in a thread, and reads any
    replica's from the shared :class:`JobStore`."""

    def __init__(self, config: RagConfig, *, after: Callable[[], Awaitable[None]]) -> None:
        self._config = config
        self._after = after
        self.store = JobStore(config.paths.state)
        self._running: IngestJob | None = None  # this replica's own, while its thread runs
        self._tasks: set[asyncio.Task[None]] = set()  # held, so none is garbage-collected

    @property
    def running(self) -> IngestJob | None:
        """The job this replica runs now, as it started; ``None`` while it runs none."""
        return self._running

    async def get(self, job_id: str) -> IngestJob | None:
        return await asyncio.to_thread(self.store.get, job_id)

    async def start(self, *, rebuild: bool, api_key: str | None = None) -> IngestJob:
        """Start a job now and return it. Raises :class:`JobRunningError` while one runs, on
        any replica. ``api_key`` is the name of the key that asked, which the job's log lines
        carry.

        Call it on the server's event loop; the job reports back to it."""
        job = IngestJob(job_id=uuid.uuid4().hex, rebuild=rebuild, api_key=api_key)
        lock = await asyncio.to_thread(
            self.store.start, job.job_id, rebuild=rebuild, api_key=api_key
        )
        self._running = job
        # The request's context, for the job's trace: read here, in the request's own task.
        parent = tracing.current()
        work = functools.partial(self._work, lock=lock, parent=parent)
        task = asyncio.create_task(self._run(job, work), name=f"ingest-{job.job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: IngestJob, work: Callable[[IngestJob], None]) -> None:
        try:
            await _in_daemon_thread(work, job)
        finally:
            self._running = None
        try:
            await self._after()
        # Deliberately broad: the job has finished either way, and a failure to reopen the
        # index is the next request's to report (each one refreshes first).
        except Exception:
            logger.exception("Could not serve what ingest job %s published", job.job_id)

    def _work(self, job: IngestJob, *, lock: JobLock, parent: Context | None = None) -> None:
        """The run itself, in the job's thread, under ``parent``, the request's context while
        tracing is on. Never raises: every ending is recorded, then the job's lock released."""
        # This thread's own context, which the request's never reaches: every line of the run.
        logs.bind(job_id=job.job_id, **({} if job.api_key is None else {"api_key": job.api_key}))
        store = self.store
        exit_code: int | None = None
        report: dict[str, Any] | None = None
        chunks = 0
        handler = _JobLog(store, job.job_id, threading.get_ident())
        level = ingest_logger.level
        try:
            store.started(job.job_id)
            ingest_logger.addHandler(handler)
            if ingest_logger.getEffectiveLevel() > logging.INFO:
                ingest_logger.setLevel(logging.INFO)  # progress is INFO, as rag-ingest shows it
            try:
                with tracing.attached(parent):
                    result = ingest(
                        self._config,
                        rebuild=job.rebuild,
                        on_locked=functools.partial(store.running, job.job_id),
                    )
            except (
                ConfigError,
                FileNotFoundError,
                CorpusUnreadableError,
                IndexInTheWayError,
                IngestionRunningError,
                StoreUnavailableError,
            ) as e:
                # What rag-ingest reports with exit status 2, having published nothing.
                exit_code = EXIT_CANNOT_START
                store.log(job.job_id, f"Error: {describe_error(e, self._config)}")
            # Deliberately broad: an unexpected error ends the job, never the server.
            # rag-ingest would print its traceback; here it goes to the server log only.
            except Exception as e:
                logger.exception("Ingest job %s failed", job.job_id)
                exit_code = EXIT_RUN_FAILED
                store.log(job.job_id, f"Error: {describe_error(e, self._config)}")
            else:
                report = asdict(result)
                chunks = result.chunks
                exit_code, error = outcome(result)
                if error is not None:
                    store.log(job.job_id, f"Error: {error}")
            finally:
                ingest_logger.removeHandler(handler)
                ingest_logger.setLevel(level)
            status: JobStatus = "succeeded" if exit_code == EXIT_OK else "failed"
            # Written before the lock goes: a reader that finds the lock free finds the end.
            store.finish(job.job_id, status=status, exit_code=exit_code, report=report)
            metrics.INGEST_RUNS.labels(status=status).inc()
            if report is not None:
                metrics.INGEST_CHUNKS.inc(chunks)
        # Deliberately broad: the database itself failed. The row stays unfinished, and reads
        # as abandoned once the lock below is free; the server log says why.
        except Exception:
            logger.exception("Could not record ingest job %s", job.job_id)
        finally:
            lock.release()


async def _in_daemon_thread(work: Callable[[IngestJob], None], job: IngestJob) -> None:
    """Run ``work(job)`` in a daemon thread and wait for it to end.

    Not ``asyncio.to_thread``: its pool threads are joined at exit, so a server stopped
    mid-ingest would wait out the whole run. A cancelled wait leaves the thread running;
    it is abandoned when the process ends.
    """
    loop = asyncio.get_running_loop()
    done: asyncio.Future[None] = loop.create_future()

    def finish() -> None:
        if not done.done():
            done.set_result(None)

    def target() -> None:
        try:
            work(job)
        finally:
            with suppress(RuntimeError):  # the loop has closed: the server stopped
                loop.call_soon_threadsafe(finish)

    threading.Thread(target=target, name=f"ingest-{job.job_id}", daemon=True).start()
    await done
