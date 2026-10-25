"""Structured logs (NFR-10, DEC-22, S3-7): ``rag_qa.observability.logs``.

Every command formats its log records, ours and the libraries', through one root handler on
stderr: JSON lines for a machine, console lines for a person. The conftest sets
``RAG_LOG_FORMAT=console`` for the rest of the suite; these tests choose for themselves, and
the autouse ``reset_logging`` fixture undoes every configuration after each.
"""

import io
import json
import logging
import sys
from typing import Any

import pytest
from conftest import Traced
from opentelemetry import trace

from rag_qa.observability import logs, tracing


def json_lines(text: str) -> list[dict[str, Any]]:
    """Every line of ``text`` that is a JSON object, parsed; fails on a line that is not."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture
def as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOG_FORMAT", "json")
    assert logs.configure_logging() == logs.JSON


def test_json_lines_carry_the_message_level_logger_time_and_extra_fields(
    as_json: None, capsys: pytest.CaptureFixture[str]
) -> None:
    logging.getLogger("rag_qa.ingest").info("Published %s.", "gen-1", extra={"points": 7})
    (line,) = json_lines(capsys.readouterr().err)
    assert line["event"] == "Published gen-1."
    assert (line["level"], line["logger"], line["points"]) == ("info", "rag_qa.ingest", 7)
    assert line["timestamp"].endswith("Z")  # ISO, in UTC


def test_a_json_traceback_is_text_and_carries_no_local_variable(
    as_json: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # structlog's dictionary form records each frame's locals by default (show_locals=True),
    # which could put a question or a key into the log.
    def fail() -> None:
        private = "a-local-value-never-logged"  # noqa: F841
        raise ValueError("boom")

    try:
        fail()
    except ValueError:
        logging.getLogger("rag_qa.api").exception("An answer failed")
    (line,) = json_lines(capsys.readouterr().err)
    assert isinstance(line["exception"], str)
    assert "Traceback (most recent call last)" in line["exception"]
    assert "ValueError: boom" in line["exception"]
    assert "a-local-value-never-logged" not in json.dumps(line)


def test_the_clis_console_line_is_the_message_alone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # rag-ingest's progress, exactly as it always read.
    monkeypatch.setenv("RAG_LOG_FORMAT", "console")
    assert logs.configure_logging() == logs.CONSOLE
    logging.getLogger("rag_qa.ingest").info("  - Loading %s with %s", "a.pdf", "PdfLoader")
    assert capsys.readouterr().err == "  - Loading a.pdf with PdfLoader\n"


def test_rag_serves_console_line_carries_the_level_logger_and_bound_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_LOG_FORMAT", "console")
    logs.configure_logging(detailed=True)
    request_id = logs.bind_request()
    logging.getLogger("rag_qa.api.routes").warning("Answer refused")
    err = capsys.readouterr().err
    assert "Answer refused" in err and "warning" in err and "rag_qa.api.routes" in err
    assert request_id in err


@pytest.mark.parametrize(("terminal", "chosen"), [(True, "console"), (False, "json")])
def test_unset_the_format_follows_whether_stderr_is_a_terminal(
    monkeypatch: pytest.MonkeyPatch, terminal: bool, chosen: str
) -> None:
    monkeypatch.delenv("RAG_LOG_FORMAT")
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: terminal)
    monkeypatch.setattr(sys, "stderr", stream)
    assert logs.configure_logging() == chosen
    assert logs.current_format() == chosen


def test_an_unknown_format_is_refused_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LOG_FORMAT", "yaml")
    with pytest.raises(logs.LogSettingsError) as caught:
        logs.configure_logging()
    assert str(caught.value) == "RAG_LOG_FORMAT: Input should be 'json' or 'console'"


def test_configuring_twice_keeps_one_handler_and_the_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    logs.configure_logging()
    logs.configure_logging(detailed=True)
    assert len([h for h in root.handlers if h not in before]) == 1
    assert root.level == logging.WARNING  # httpx's line per request stays out
    assert logging.getLogger("rag_qa").level == logging.INFO
    assert logging.getLogger("pypdf").level == logging.ERROR  # for the process (backlog)
    logs.reset_logging()
    assert root.handlers == before
    assert logging.getLogger("pypdf").level == logging.NOTSET
    assert logs.current_format() is None


def test_rag_eval_keeps_our_info_lines_out_of_its_printed_tables(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The corpus walk's progress ("Skipped 0000.parquet") beside tier 2's table is noise.
    logs.configure_logging(level=logging.WARNING)
    logging.getLogger("rag_qa.ingest").info("  - Skipped a.parquet")
    logging.getLogger("rag_qa.ingest").warning("  - Cannot read b.pdf")
    assert capsys.readouterr().err == "  - Cannot read b.pdf\n"


def test_a_librarys_info_line_stays_out_and_its_warning_comes_through(
    as_json: None, capsys: pytest.CaptureFixture[str]
) -> None:
    logging.getLogger("httpx").info("HTTP Request: POST http://localhost:11434/api/chat")
    logging.getLogger("httpx").warning("a warning")
    assert [line["event"] for line in json_lines(capsys.readouterr().err)] == ["a warning"]


def test_uvicorns_records_are_formatted_as_ours_without_their_coloured_copy(
    as_json: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # As rag-serve leaves uvicorn's loggers, with log_config=None: propagating to the root.
    # uvicorn's own default config, which the real-uvicorn tests run with in this process,
    # stops that and gives "uvicorn" a handler of its own.
    monkeypatch.setattr(logging.getLogger("uvicorn"), "propagate", True)
    monkeypatch.setattr(logging.getLogger("uvicorn"), "handlers", [])
    uvicorn = logging.getLogger("uvicorn.error")
    uvicorn.setLevel(logging.INFO)  # rag-serve sets it through uvicorn's log_level
    try:
        uvicorn.info("Started server process [%d]", 42, extra={"color_message": "\x1b[36m42"})
    finally:
        uvicorn.setLevel(logging.NOTSET)
    (line,) = json_lines(capsys.readouterr().err)
    assert (line["event"], line["logger"]) == ("Started server process [42]", "uvicorn.error")
    assert "color_message" not in line


def test_what_another_formatter_left_on_a_record_is_not_a_field(
    as_json: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # pytest's own handler formats a record before ours sees it, which sets record.message.
    record = logging.LogRecord("rag_qa.x", logging.INFO, __file__, 1, "hello %s", ("you",), None)
    logging.Formatter("%(asctime)s %(message)s").format(record)
    logging.getLogger("rag_qa.x").handle(record)
    (line,) = json_lines(capsys.readouterr().err)
    assert line["event"] == "hello you"
    assert "message" not in line and "asctime" not in line


def test_each_request_starts_its_own_context(
    as_json: None, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = logging.getLogger("rag_qa.api")
    first = logs.bind_request()
    logs.bind(job_id="j1")
    logger.info("one")
    second = logs.bind_request()
    logger.info("two")
    one, two = json_lines(capsys.readouterr().err)
    assert (one["request_id"], one["job_id"]) == (first, "j1")
    assert two["request_id"] == second != first
    assert "job_id" not in two  # nothing is left from before


def test_a_line_logged_under_a_span_carries_its_trace_and_span_ids(
    as_json: None, traced: Traced, capsys: pytest.CaptureFixture[str]
) -> None:
    # S3-7b: what joins a log line to its trace.
    with tracing.span("rag.test"):
        logging.getLogger("rag_qa.api").info("inside")
    logging.getLogger("rag_qa.api").info("outside")
    inside, outside = json_lines(capsys.readouterr().err)
    span = traced.one("rag.test")
    assert span.context is not None
    assert inside["trace_id"] == trace.format_trace_id(span.context.trace_id)
    assert inside["span_id"] == trace.format_span_id(span.context.span_id)
    assert "trace_id" not in outside and "span_id" not in outside


def test_with_tracing_off_no_line_carries_trace_ids(
    as_json: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with tracing.span("rag.test"):
        logging.getLogger("rag_qa.api").info("inside")
    (line,) = json_lines(capsys.readouterr().err)
    assert "trace_id" not in line and "span_id" not in line


def test_the_handler_writes_to_stderr_as_it_is_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pytest swaps stderr for every test; a handler holding the first one would write to a
    # closed stream.
    logs.configure_logging()
    later = io.StringIO()
    monkeypatch.setattr(sys, "stderr", later)
    logging.getLogger("rag_qa").warning("after the swap")
    assert "after the swap" in later.getvalue()
