"""Ingest jobs: ``POST /v1/ingest`` runs one at a time, in the background (DEC-18).

A job runs :func:`rag_qa.ingest.ingest`, the function behind ``rag-ingest``, and keeps
what that command would have shown: the report, the exit status and the log lines. All of
them are path-free (S2-6), since they go over HTTP.

- **One at a time.** :meth:`IngestJobs.start` refuses a second job while one runs, which
  the route turns into a 409. A ``rag-ingest`` running outside the server holds the
  writer lock (DEC-17), so a job started then fails at once with exit status 2 and that
  command's own message. A route cannot hold the ``flock`` on the job's behalf, because
  ``ingest()`` takes it itself.
- **In a daemon thread.** An ingest cannot be interrupted and can take minutes, so
  stopping the server abandons a job rather than waiting for it. That is safe: a run never
  writes into the live index generation, and the next run cleans up what it left (DEC-17,
  DEC-19).
- **In memory.** The last :data:`KEEP` jobs are kept; Phase 3 brings a durable queue.
- **What it changes.** When a job ends, ``after`` runs (the server's refresh), so a
  published generation is served at once.
- **What it reports** (S3-7): the job's log lines carry its ``job_id``, and each job counts in
  ``rag_ingest_runs_total`` by how it ended, its embedded chunks in ``rag_ingest_chunks_total``.
  A ``rag-ingest`` run outside the server has no metrics listener: it reports in its log.
"""

import asyncio
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rag_qa.api.models import JobStatus
from rag_qa.config import ConfigError
from rag_qa.ingest import (
    EXIT_CANNOT_START,
    EXIT_OK,
    EXIT_RUN_FAILED,
    CorpusUnreadableError,
    IndexInTheWayError,
    IngestionRunningError,
    IngestReport,
    describe_error,
    ingest,
    outcome,
)
from rag_qa.ingest import logger as ingest_logger
from rag_qa.observability import logs, metrics
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import StoreUnavailableError

logger = logging.getLogger(__name__)

#: How many finished jobs the registry keeps.
KEEP = 20


@dataclass
class IngestJob:
    """One ``POST /v1/ingest``. Its worker thread fills it in; the routes only read it."""

    job_id: str
    rebuild: bool
    status: JobStatus = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None  # what rag-ingest would have exited with
    report: IngestReport | None = None  # None when the run could not start
    log: list[str] = field(default_factory=list)


class JobRunningError(Exception):
    """A job is running already. ``job`` is that one."""

    def __init__(self, job: IngestJob) -> None:
        super().__init__(f"ingest job {job.job_id} is running")
        self.job = job


class _JobLog(logging.Handler):
    """Copies one thread's ``rag_qa.ingest`` records into a job's log, as plain text."""

    def __init__(self, lines: list[str], thread: int) -> None:
        super().__init__(logging.INFO)
        self._lines = lines
        self._thread = thread

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread:
            self._lines.append(record.getMessage())


class IngestJobs:
    """The job registry, and the one job that may be running."""

    def __init__(self, config: RagConfig, *, after: Callable[[], Awaitable[None]]) -> None:
        self._config = config
        self._after = after
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._running: IngestJob | None = None
        self._tasks: set[asyncio.Task[None]] = set()  # held, so none is garbage-collected

    @property
    def running(self) -> IngestJob | None:
        return self._running

    def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def start(self, *, rebuild: bool) -> IngestJob:
        """Start a job now and return it. Raises :class:`JobRunningError` while one runs.

        Call it on the server's event loop; the job reports back to it."""
        if self._running is not None:
            raise JobRunningError(self._running)
        job = IngestJob(job_id=uuid.uuid4().hex, rebuild=rebuild)
        self._jobs[job.job_id] = job
        while len(self._jobs) > KEEP:
            self._jobs.popitem(last=False)  # the oldest; the running job is the newest
        self._running = job
        task = asyncio.create_task(self._run(job), name=f"ingest-{job.job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: IngestJob) -> None:
        try:
            await _in_daemon_thread(self._work, job)
        finally:
            self._running = None
        try:
            await self._after()
        # Deliberately broad: the job has finished either way, and a failure to reopen the
        # index is the next request's to report (each one refreshes first).
        except Exception:
            logger.exception("Could not serve what ingest job %s published", job.job_id)

    def _work(self, job: IngestJob) -> None:
        """The run itself, in the job's thread. Never raises: every ending is recorded."""
        logs.bind(job_id=job.job_id)  # this thread's own context: every line of the run
        job.started_at = datetime.now(UTC)
        job.status = "running"
        handler = _JobLog(job.log, threading.get_ident())
        level = ingest_logger.level
        ingest_logger.addHandler(handler)
        if ingest_logger.getEffectiveLevel() > logging.INFO:
            ingest_logger.setLevel(logging.INFO)  # progress is INFO, as rag-ingest shows it
        try:
            report = ingest(self._config, rebuild=job.rebuild)
        except (
            ConfigError,
            FileNotFoundError,
            CorpusUnreadableError,
            IndexInTheWayError,
            IngestionRunningError,
            StoreUnavailableError,
        ) as e:
            # What rag-ingest reports with exit status 2, having published nothing.
            job.exit_code = EXIT_CANNOT_START
            job.log.append(f"Error: {describe_error(e, self._config)}")
        # Deliberately broad: an unexpected error ends the job, never the server.
        # rag-ingest would print its traceback; here it goes to the server log only.
        except Exception as e:
            logger.exception("Ingest job %s failed", job.job_id)
            job.exit_code = EXIT_RUN_FAILED
            job.log.append(f"Error: {describe_error(e, self._config)}")
        else:
            job.report = report
            job.exit_code, error = outcome(report)
            if error is not None:
                job.log.append(f"Error: {error}")
        finally:
            ingest_logger.removeHandler(handler)
            ingest_logger.setLevel(level)
            job.finished_at = datetime.now(UTC)
            status: JobStatus = "succeeded" if job.exit_code == EXIT_OK else "failed"
            metrics.INGEST_RUNS.labels(status=status).inc()
            if job.report is not None:
                metrics.INGEST_CHUNKS.inc(job.report.chunks)
            job.status = status  # last: a reader that sees it sees the rest


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
