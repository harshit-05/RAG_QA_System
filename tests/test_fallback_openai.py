"""The hosted fallback's real client (S3-6, DEC-21), where the ``fallback`` extra is installed.

CI installs no extra, so this whole module is skipped there; it runs after
``uv sync --extra fallback``. A real ``ChatOpenAI``, built by
:func:`rag_qa.components.build_fallback_llm` from the committed ``groq_fallback`` entry, talks
to an httpx2 ``MockTransport`` standing in for the provider. So the tests see what it sends
(the key, the payload), how it streams, how its errors are classified, and that the key
never leaves through a message, an event or a repr.
"""

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import MakeConfig, StreamingFakeChatModel, fake_serving, sse_events, use_fakes

pytest.importorskip("langchain_openai", reason="the fallback extra is not installed")

import httpx2
import openai  # after the skip: the fallback extra brings openai
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from rag_qa import resilience
from rag_qa.answering import Done, Token, stream_answer
from rag_qa.api.app import create_app
from rag_qa.chain import QueryPipeline
from rag_qa.cli import describe_failure
from rag_qa.components import aclose_llm, build_fallback_llm
from rag_qa.config import load_config
from rag_qa.schema import RagConfig
from rag_qa.settings import ServerSettings

KEY = "gsk-sentinel-0123456789abcdef"
REF = "components.llms.groq_fallback"
Handler = Callable[[httpx2.Request], httpx2.Response]


def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def streams(*texts: str) -> Handler:
    """A provider that streams ``texts`` as chat-completion chunks, then ``[DONE]``."""

    def answer(request: httpx2.Request) -> httpx2.Response:
        body = chunk({"role": "assistant", "content": ""})
        body += "".join(chunk({"content": text}) for text in texts)
        body += chunk({}, "stop") + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    return answer


def refuses_the_key(request: httpx2.Request) -> httpx2.Response:
    error = {
        "message": "Invalid API Key",
        "type": "invalid_request_error",
        "code": "invalid_api_key",
    }
    return httpx2.Response(401, json={"error": error})


def unreachable(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("All connection attempts failed", request=request)


class Provider:
    """What the fallback's client sent, and how the fake provider answers."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer: Handler = streams("Spare ", "answer.")

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.answer(request)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Provider:
    """Every fallback built gets its own HTTP client, as it always does, here over a fake
    transport; and its key is set."""
    state = Provider()
    real = openai.DefaultAsyncHttpxClient
    monkeypatch.setattr(
        openai,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: real(transport=httpx2.MockTransport(state), **kwargs),
    )
    monkeypatch.setenv("RAG_FALLBACK_API_KEY", KEY)
    return state


@pytest.fixture
def config(make_config: MakeConfig) -> RagConfig:
    def edit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["pipeline"]["query"]["fallback_llm"] = REF

    return load_config(make_config(edit))


class Refused(StreamingFakeChatModel):
    """A stopped Ollama."""

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> Any:
        raise httpx.ConnectError("All connection attempts failed")
        yield  # an async generator, like every model's stream


def through_the_fallback(spare: Any) -> QueryPipeline:
    prompt = ChatPromptTemplate.from_messages([("human", "{context}\n\nQuestion: {question}")])
    return QueryPipeline(
        retrieve=RunnableLambda(lambda asked: []),
        prompt=prompt,
        llm=Refused(),
        corpus_root=Path("/corpus"),
        llm_ref="components.llms.fake",
        fallback_llm=spare,
        fallback_ref=REF,
    )


def run(pipeline: QueryPipeline) -> list[Any]:
    async def go() -> list[Any]:
        try:
            return [event async for event in stream_answer(pipeline, "What is GLIDER?")]
        finally:
            await pipeline.aclose()

    return asyncio.run(go())


def test_the_fallback_is_built_with_its_key_no_retries_and_its_timeout_phases(
    config: RagConfig, provider: Provider
) -> None:
    spare = build_fallback_llm(config)
    assert spare is not None
    assert spare.max_retries == 0  # type: ignore[attr-defined]
    assert spare.root_async_client.max_retries == 0  # type: ignore[attr-defined]
    assert spare.request_timeout == openai.Timeout(connect=5, read=60, write=30, pool=30)  # type: ignore[attr-defined]
    assert spare.openai_api_key.get_secret_value() == KEY  # type: ignore[attr-defined]
    asyncio.run(aclose_llm(spare))


def test_the_fallback_streams_the_answer_and_sends_what_the_entry_says(
    config: RagConfig, provider: Provider
) -> None:
    events = run(through_the_fallback(build_fallback_llm(config)))
    assert [event.text for event in events if isinstance(event, Token)] == ["Spare ", "answer."]
    assert isinstance(events[-1], Done) and events[-1].model == REF
    (request,) = provider.requests  # one attempt: no SDK retry
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    sent = json.loads(request.content)
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["stream"] is True
    assert sent["temperature"] == 0
    assert sent["max_completion_tokens"] == 1024 and "max_tokens" not in sent
    assert sent["reasoning_effort"] == "low"
    assert sent["include_reasoning"] is False  # Groq's own, through extra_body
    assert "stream_options" not in sent  # a custom base_url: no usage chunk asked for


def test_a_refused_key_is_the_answer_s_error_and_names_the_variable_never_the_key(
    config: RagConfig, provider: Provider, caplog: pytest.LogCaptureFixture
) -> None:
    provider.answer = refuses_the_key
    pipeline = through_the_fallback(build_fallback_llm(config))
    with caplog.at_level(logging.DEBUG), pytest.raises(openai.AuthenticationError) as exc:
        run(pipeline)
    assert resilience.classify(exc.value) == resilience.ANSWERED
    said = describe_failure(exc.value, pipeline.llm)
    assert said.startswith("the hosted fallback failed (")
    assert said.endswith("Check RAG_FALLBACK_API_KEY.")
    for text in (said, str(exc.value), repr(exc.value), caplog.text):
        assert KEY not in text
    assert pipeline.fallback_breaker._failed == 0  # it answered: the request was refused


def test_an_unreachable_provider_is_one_attempt_and_a_failure_on_its_breaker(
    config: RagConfig, provider: Provider
) -> None:
    provider.answer = unreachable
    pipeline = through_the_fallback(build_fallback_llm(config))
    with pytest.raises(openai.APIConnectionError) as exc:
        run(pipeline)
    assert resilience.classify(exc.value) == resilience.CONNECTION
    assert len(provider.requests) == 1  # max_retries 0: the breaker sees every failure
    assert pipeline.fallback_breaker._failed == 1


def test_the_key_leaves_through_no_log_event_or_repr_over_http(
    config: RagConfig, provider: Provider, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    provider.answer = refuses_the_key
    spare = build_fallback_llm(config)
    assert KEY not in repr(spare) and KEY not in str(spare)
    serving = fake_serving(config, Refused())
    serving.pipeline = dataclasses.replace(serving.pipeline, fallback_llm=spare, fallback_ref=REF)
    app = create_app(config, ServerSettings(), serving=serving)
    with caplog.at_level(logging.DEBUG), TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/query", json={"question": "q"})
    name, data = sse_events(response.text)[-1]
    assert name == "error"
    assert data["message"] == "the hosted fallback returned an error"  # fixed text
    assert KEY not in response.text
    assert KEY not in caplog.text


def test_closing_one_fallback_closes_its_own_client_and_no_other(
    config: RagConfig, provider: Provider
) -> None:
    # langchain-openai caches one default client per base URL and timeout, shared by every
    # instance: closing it would break the next. Each fallback owns its own (D9).
    first, second = build_fallback_llm(config), build_fallback_llm(config)
    asyncio.run(aclose_llm(first))
    assert first.root_async_client._client.is_closed  # type: ignore[union-attr]
    assert not second.root_async_client._client.is_closed  # type: ignore[union-attr]
    events = run(through_the_fallback(second))  # the second still answers
    assert isinstance(events[-1], Done)
