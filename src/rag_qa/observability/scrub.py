"""What leaves this host in a span is scrubbed of exception text (NFR-10, DEC-22, S3-7b).

Traces are built to go to a collector that is not this host's, and an exception's message and
traceback may carry text: a document's, a question, a path DEC-18 keeps off HTTP, a config's
value (S3-6 found that pydantic's error text repeats them). The log keeps the full traceback,
on this host. A span keeps the class name alone.

Our own spans already do (:func:`rag_qa.observability.tracing.failed`). The contrib FastAPI
instrumentation's server span does not: when an exception passes through the app unhandled, it
records it itself (``span.record_exception`` in its middleware, and again in the ASGI one),
with its message and stacktrace, and sets the status description to ``"Type: message"``. It has
no switch to turn that off, and an instrumentation added later may do the same, so the one
place that covers every span is the exporter.

:class:`ScrubbingExporter` wraps the real exporter. For each span it keeps every
``exception`` event's ``exception.type`` and drops the rest of its attributes (the message, the
stacktrace), and drops the status description, leaving the status code. A span with neither
reaches the exporter as it is. Not ``SpanLimits(max_event_attributes=0)``: that drops
``exception.type`` too, and silently strips any attribute an event gains later.

Imports the SDK, so :func:`rag_qa.observability.tracing.configure` imports this module only
once tracing is on.
"""

from collections.abc import Sequence
from typing import Final

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.semconv.attributes.exception_attributes import EXCEPTION_TYPE
from opentelemetry.trace import Status

#: The event name the SDK and the instrumentations record an exception under.
EXCEPTION_EVENT: Final = "exception"


class _Scrubbed(ReadableSpan):
    """A span as it was, with other events and another status: what the exporter reads, and
    the counts of what the span dropped, which a plain copy would lose."""

    def __init__(self, span: ReadableSpan, events: Sequence[Event], status: Status) -> None:
        super().__init__(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=span.attributes,
            events=events,
            links=span.links,
            kind=span.kind,
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )
        self._dropped = (span.dropped_attributes, span.dropped_events, span.dropped_links)

    @property
    def dropped_attributes(self) -> int:
        return self._dropped[0]

    @property
    def dropped_events(self) -> int:
        return self._dropped[1]

    @property
    def dropped_links(self) -> int:
        return self._dropped[2]


def scrubbed(span: ReadableSpan) -> ReadableSpan:
    """``span`` without any exception text: see the module docstring. ``span`` itself when it
    has none to drop."""
    carries_text = span.status.description is not None or any(
        event.name == EXCEPTION_EVENT and set(event.attributes or {}) != {EXCEPTION_TYPE}
        for event in span.events
    )
    if not carries_text:
        return span
    events = [
        Event(
            event.name,
            {EXCEPTION_TYPE: event.attributes[EXCEPTION_TYPE]}
            if event.attributes and EXCEPTION_TYPE in event.attributes
            else {},
            event.timestamp,
        )
        if event.name == EXCEPTION_EVENT
        else event
        for event in span.events
    ]
    return _Scrubbed(span, events, Status(span.status.status_code))


class ScrubbingExporter(SpanExporter):
    """Passes every span to ``inner`` through :func:`scrubbed`."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([scrubbed(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
