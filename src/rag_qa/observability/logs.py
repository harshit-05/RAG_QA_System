"""Structured log lines for every command (NFR-10, DEC-22, S3-7).

Our modules log through the standard library (``logging.getLogger(__name__)``), and so do
uvicorn, httpx and pypdf. :func:`configure_logging`, every command's first call, puts one
handler on the root logger, formatted by structlog's ``ProcessorFormatter``, so every record,
ours and the libraries', comes out alike:

- **JSON lines** with ``RAG_LOG_FORMAT=json``, or when it is unset and stderr is not a
  terminal: a pipe, a file, a container's log. **Console lines** otherwise, or with
  ``RAG_LOG_FORMAT=console``, which keeps a run piped into ``tee`` readable.
- **The console's style:** the message alone for the CLIs, exactly the progress
  ``rag-ingest`` always printed; the time, the level, the logger and the bound fields for
  ``rag-serve``.
- **Every log goes to stderr.** The commands print their results to stdout.
- **Fields:** a record's ``extra=`` keys become fields, and so do the context variables bound
  with :func:`bind` and :func:`bind_request`: a request's ID in the API, a job's ID in its
  ingest thread. While tracing is on, a line logged under a span carries its ``trace_id`` and
  ``span_id`` too, which join it to its trace (S3-7b). S3-8 adds the API key's name.
- **Levels:** our ``rag_qa`` loggers at INFO, every other at WARNING, so httpx's line per
  request stays out. pypdf at ERROR for the whole process: a corrupt PDF's own warnings
  never print ahead of the clean error the run reports for it (S2-8), the API's ingest jobs'
  included (S3-7).
- **Tracebacks are plain text** in every format. structlog's dictionary form and its console
  renderer (rich's, when rich is installed) record every frame's local variables by default,
  which could put a question or a key into the log.

No question, prompt or answer text is logged anywhere: the log goes wherever the operator
sends it.

Loads no model stack (``tests/test_architecture.py``): structlog, the standard library, the
OpenTelemetry API and :mod:`rag_qa.settings`.
"""

import logging
import sys
import uuid
from typing import Any, Final, Literal, TextIO

import structlog
from opentelemetry import trace
from pydantic import ValidationError
from structlog.typing import EventDict, Processor, WrappedLogger

from rag_qa.settings import ENV_LOG_PREFIX, LogSettings, describe_settings_error

type Format = Literal["json", "console"]
JSON: Final = "json"
CONSOLE: Final = "console"

#: The attribute that marks the root handler :func:`configure_logging` installs.
_MARK: Final = "_rag_qa_logs"
#: Our loggers, at INFO unless a command asks otherwise. uvicorn's are set by ``rag-serve``
#: through uvicorn itself.
INFO_LOGGERS: Final = ("rag_qa",)
#: The logger whose warnings repeat what a run reports more clearly (S2-8).
QUIET_LOGGERS: Final = ("pypdf",)

#: The format :func:`configure_logging` chose, or ``None`` before it ran (library use, tests).
_format: Format | None = None


class LogSettingsError(ValueError):
    """``RAG_LOG_FORMAT`` is set to something else than ``json`` or ``console``: the message
    names the variable, never its value."""


class _Stderr(logging.StreamHandler[TextIO]):
    """A handler on ``sys.stderr`` as it is at each record, as logging's own last resort is.
    pytest swaps stderr for every test, and a handler holding the stream it was built with
    would write to a closed one."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr  # under the handler's lock: Handler.handle holds it
        super().emit(record)

    def flush(self) -> None:
        self.stream = sys.stderr
        super().flush()


#: Record attributes that look like ``extra=`` fields but are not: what a formatter that saw
#: the record first left on it (pytest's, say), and uvicorn's coloured copy of its message.
_NOT_FIELDS: Final = ("message", "asctime", "color_message")


def _without_formatter_leftovers(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """Drop :data:`_NOT_FIELDS`: the message is already the event, and once is enough."""
    for name in _NOT_FIELDS:
        event_dict.pop(name, None)
    return event_dict


def _trace_ids(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """The trace and the span a record was logged under, when one is current (S3-7b): what
    joins a log line to its trace. Nothing while tracing is off, when no span ever is."""
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event_dict["trace_id"] = trace.format_trace_id(context.trace_id)
        event_dict["span_id"] = trace.format_span_id(context.span_id)
    return event_dict


def _message_only(_: WrappedLogger, __: str, event_dict: EventDict) -> str:
    """The CLIs' console line: the message, and a traceback under it when there is one."""
    message = str(event_dict.get("event", ""))
    exception = event_dict.get("exception")
    return f"{message}\n{exception}" if exception else message


def _formatter(chosen: Format, *, detailed: bool) -> structlog.stdlib.ProcessorFormatter:
    """The formatter of every record, for ``chosen``: see the module docstring."""
    foreign_pre_chain: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _trace_ids,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        _without_formatter_leftovers,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    remove_meta = structlog.stdlib.ProcessorFormatter.remove_processors_meta
    processors: list[Processor]
    if chosen == JSON:
        processors = [
            remove_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    elif detailed:
        # plain_traceback: ConsoleRenderer's default, rich's, prints every frame's local
        # variables when rich is installed, a question and its chunks among them.
        processors = [
            remove_meta,
            structlog.dev.ConsoleRenderer(
                colors=_is_terminal(sys.stderr),
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    else:
        processors = [remove_meta, structlog.processors.format_exc_info, _message_only]
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=foreign_pre_chain, processors=processors
    )


def _is_terminal(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):  # no such method, or a closed stream
        return False


def configure_logging(*, detailed: bool = False, level: int = logging.INFO) -> Format:
    """Format every log record of this process, ours and the libraries', and return the
    format chosen. Every command calls it first; ``rag-serve`` asks for ``detailed`` console
    lines. ``level`` is our loggers': ``rag-eval`` asks for WARNING, since its results are
    its printed tables and the corpus walk's progress would be noise beside them. A second
    call reconfigures the one handler rather than add another.

    Raises :class:`LogSettingsError` when ``RAG_LOG_FORMAT`` is neither ``json`` nor
    ``console``: the command exits 2 with its message.
    """
    global _format
    try:
        chosen = LogSettings().format
    except ValidationError as e:
        raise LogSettingsError(describe_settings_error(e, ENV_LOG_PREFIX)) from None
    if chosen is None:
        chosen = CONSOLE if _is_terminal(sys.stderr) else JSON
    root = logging.getLogger()
    handler = next((h for h in root.handlers if getattr(h, _MARK, False)), None)
    if handler is None:
        handler = _Stderr()
        setattr(handler, _MARK, True)
        root.addHandler(handler)
    handler.setFormatter(_formatter(chosen, detailed=detailed))
    root.setLevel(logging.WARNING)
    for name in INFO_LOGGERS:
        logging.getLogger(name).setLevel(level)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    _format = chosen
    return chosen


def current_format() -> Format | None:
    """The format :func:`configure_logging` chose, or ``None`` when no command configured
    logging in this process (a library caller, a test)."""
    return _format


def reset_logging() -> None:
    """Undo :func:`configure_logging`: its handler and its levels. For the tests, whose
    session runs many commands in one process."""
    global _format
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _MARK, False)]:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    for name in (*INFO_LOGGERS, *QUIET_LOGGERS):
        logging.getLogger(name).setLevel(logging.NOTSET)
    structlog.contextvars.clear_contextvars()
    _format = None


def bind(**fields: Any) -> None:
    """Add ``fields`` to every log line of this task or thread from now on."""
    structlog.contextvars.bind_contextvars(**fields)


def bind_request() -> str:
    """Start a request's log context: a new ``request_id``, and nothing left from before.
    Returns the ID. Call it in the request's own task (an ``async`` dependency), so the
    tasks it starts, FastAPI's SSE producer among them, inherit it."""
    structlog.contextvars.clear_contextvars()
    request_id = uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id)
    return request_id
