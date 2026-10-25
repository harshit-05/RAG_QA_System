"""What the GenAI semantic conventions call a chat model (NFR-10, S3-7, S3-7b).

The token metric (:mod:`.metrics`) labels each answer with its provider and the model it
asked for, and the chat spans (S3-7b) carry the same. :func:`model_identity` reads both from
the model object itself, so the hosted fallback (S3-6) and the tests' fakes need no config
lookup, and :func:`chat_attributes` reads a span's request attributes the same way.
Duck-typed: this module imports no LangChain, since ``observability`` loads no model stack.

**A chat span** (:func:`rag_qa.answering.stream_answer`, one per attempt at a model) is named
``chat {model}`` and carries the operation, the provider, the request model, the temperature
and the maximum tokens it was asked with, the server it was sent to, the time to its first
chunk and its token usage; plus our own: the model's config reference, the attempt, and
whether the hosted fallback answered and why. Never a message's text.

**The names and well-known values come from opentelemetry-semantic-conventions 0.66b1,**
never typed from memory. That release marks every GenAI constant deprecated, "moved to the
OpenTelemetry GenAI semantic conventions repository", which publishes no Python package yet;
the strings are unchanged. Reading a constant, or an enum's member, raises no warning;
calling an enum (``GenAiOperationNameValues("chat")``) raises a ``DeprecationWarning``, which
the deprecation gate would fail on (``tests/test_deprecations.py``). So: members only.
"""

from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

#: ``gen_ai.operation.name`` of every answer: a chat completion.
CHAT: Final = gen_ai.GenAiOperationNameValues.CHAT.value
#: ``gen_ai.provider.name`` for Ollama. No well-known value applies, and then the convention
#: lets an instrumentation use a custom one.
OLLAMA: Final = "ollama"
#: ``gen_ai.provider.name`` for an OpenAI-compatible client, whoever serves it: the API's
#: flavour. The host (Groq, for the hosted fallback) is ``server.address``'s to name, as the
#: convention's note on the attribute allows (S3-7b's spans).
OPENAI: Final = gen_ai.GenAiProviderNameValues.OPENAI.value
#: Providers by the top-level package of the model's class: LangChain's integrations.
_PROVIDERS: Final = {"langchain_ollama": OLLAMA, "langchain_openai": OPENAI}


@dataclass(frozen=True)
class ModelIdentity:
    """A chat model as the GenAI conventions name it."""

    provider: str  # gen_ai.provider.name: "ollama", "openai", or the model's package
    model: str  # gen_ai.request.model: "mistral", "openai/gpt-oss-120b"


def model_identity(llm: Any) -> ModelIdentity:
    """``llm``'s provider and the model it asks for.

    The provider comes from the top-level package of the model's class, and is that package's
    own name when it is neither Ollama's nor OpenAI's integration (a test's fake, say). The
    model is ``ChatOpenAI``'s ``model_name``, else ``ChatOllama``'s ``model``, else the class's
    name.
    """
    package = type(llm).__module__.split(".")[0]
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    return ModelIdentity(
        provider=_PROVIDERS.get(package, package),
        model=str(name) if name else type(llm).__name__,
    )


# --- the chat span (S3-7b) --------------------------------------------------------------------

#: Our own attributes on a chat span: the model's config reference (``Done.model``'s), the
#: attempt at that model (from 1, a retry after a failed connection is 2), whether it is the
#: hosted fallback, and why the fallback was asked.
MODEL_REF: Final = "rag.model_ref"
ATTEMPT: Final = "rag.attempt"
FALLBACK: Final = "rag.fallback"
FALLBACK_REASON: Final = "rag.fallback_reason"
#: The two reasons: the model's breaker refused it, so it was not asked (no span of its own),
#: or its connection failed before a first token (DEC-21).
BREAKER_OPEN: Final = "breaker_open"
CONNECTION_FAILED: Final = "connection_failed"
#: The event that marks the first token of an answer.
FIRST_TOKEN: Final = "rag.first_token"
#: Default ports, for a base URL that names none: ``server.port`` goes with ``server.address``.
_PORTS: Final = {"http": 80, "https": 443}


def chat_span_name(identity: ModelIdentity) -> str:
    """A chat span's name, as the convention has it: the operation, then the request model."""
    return f"{CHAT} {identity.model}"


def chat_attributes(llm: Any, identity: ModelIdentity) -> dict[str, AttributeValue]:
    """The request attributes of a chat span to ``llm``: the operation, the provider and the
    request model; the temperature and the maximum tokens it asks with (``ChatOllama``'s
    ``num_predict``, else ``ChatOpenAI``'s ``max_tokens``, which it sends as
    ``max_completion_tokens``); and the server its base URL names, when it names one."""
    attributes: dict[str, AttributeValue] = {
        gen_ai.GEN_AI_OPERATION_NAME: CHAT,
        gen_ai.GEN_AI_PROVIDER_NAME: identity.provider,
        gen_ai.GEN_AI_REQUEST_MODEL: identity.model,
    }
    temperature = getattr(llm, "temperature", None)
    if isinstance(temperature, int | float) and not isinstance(temperature, bool):
        attributes[gen_ai.GEN_AI_REQUEST_TEMPERATURE] = float(temperature)
    limit = getattr(llm, "num_predict", None)
    if limit is None:
        limit = getattr(llm, "max_tokens", None)
    if isinstance(limit, int) and not isinstance(limit, bool):
        attributes[gen_ai.GEN_AI_REQUEST_MAX_TOKENS] = limit
    url = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
    if isinstance(url, str):
        split = urlsplit(url)
        if split.hostname:
            attributes[SERVER_ADDRESS] = split.hostname
            port = split.port or _PORTS.get(split.scheme)
            if port is not None:
                attributes[SERVER_PORT] = port
    return attributes


def record_first_token(span: Span, seconds: float) -> None:
    """The answer's first token reached ``span``'s attempt ``seconds`` after its request."""
    span.add_event(FIRST_TOKEN)
    span.set_attribute(gen_ai.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK, seconds)


def record_usage(span: Span, input_tokens: int | None, output_tokens: int | None) -> None:
    """The token counts the provider reported for ``span``'s attempt, those it reported."""
    for name, count in (
        (gen_ai.GEN_AI_USAGE_INPUT_TOKENS, input_tokens),
        (gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens),
    ):
        if count is not None:
            span.set_attribute(name, count)
