"""What the GenAI semantic conventions call a chat model (NFR-10, S3-7).

The token metric (:mod:`.metrics`) labels each answer with its provider and the model it
asked for, and S3-7b's chat spans carry the same. :func:`model_identity` reads both from the
model object itself, so the hosted fallback (S3-6) and the tests' fakes need no config
lookup. Duck-typed: this module imports no LangChain, since ``observability`` loads no model
stack.

**The names and well-known values come from opentelemetry-semantic-conventions 0.66b1,**
never typed from memory. That release marks every GenAI constant deprecated, "moved to the
OpenTelemetry GenAI semantic conventions repository", which publishes no Python package yet;
the strings are unchanged. Reading a constant, or an enum's member, raises no warning;
calling an enum (``GenAiOperationNameValues("chat")``) raises a ``DeprecationWarning``, which
the deprecation gate would fail on (``tests/test_deprecations.py``). So: members only.
"""

from dataclasses import dataclass
from typing import Any, Final

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai

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
