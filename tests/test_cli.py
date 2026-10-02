"""rag-query's REPL: it survives failures, exits cleanly, and Ctrl-C stops an answer.

Driven hermetically: a fake pipeline on a scripted chat model, and scripted input, so
no Ollama and no index. The model runs under the REPL's real ``asyncio.Runner``, so a
Ctrl-C is a real SIGINT (``signal.raise_signal``) handled by the Runner's own handler,
exactly as in a terminal (S2-1, DEC-14). The live checks against real Ollama are the
stories' manual steps.
"""

import asyncio
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import MakeConfig, StreamingFakeChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from rag_qa import cli
from rag_qa.chain import QueryPipeline
from rag_qa.cli import EXIT_CANNOT_START, EXIT_OK, repl

CTRL_C = "ctrl-c"  # script: the user presses Ctrl-C once, mid-answer
CTRL_C_TWICE = "ctrl-c twice"  # and again while the answer is unwinding


def _chunk(text: str) -> ChatGenerationChunk:
    return ChatGenerationChunk(message=AIMessageChunk(content=text))


class ScriptedModel(StreamingFakeChatModel):
    """Streams "Partial ", then does the next scripted behaviour: an answer string, an
    exception to raise mid-stream (like a dropped Ollama connection), or a Ctrl-C."""

    script: list[Any] = Field(default_factory=list)
    loops: list[Any] = Field(default_factory=list)  # the event loop of each answer

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._record(messages)
        self.loops.append(asyncio.get_running_loop())
        behaviour = self.script.pop(0)
        self.log.append("start")
        try:
            yield _chunk("Partial ")
            if behaviour in (CTRL_C, CTRL_C_TWICE):
                signal.raise_signal(signal.SIGINT)  # the Runner cancels this answer...
                await asyncio.sleep(30)  # ...and the cancel lands here, as in a read from Ollama
            elif isinstance(behaviour, BaseException):
                raise behaviour
            else:
                yield _chunk(behaviour)
        finally:
            self.log.append("closed")
            if behaviour == CTRL_C_TWICE:
                signal.raise_signal(signal.SIGINT)


def fake_pipeline(model: StreamingFakeChatModel) -> QueryPipeline:
    doc = Document(page_content="c", metadata={"source": "/x/guide.pdf", "page": 0})
    prompt = ChatPromptTemplate.from_messages([("human", "{context}\n\nQuestion: {question}")])
    return QueryPipeline(
        retrieve=RunnableLambda(lambda question: [doc]),
        answer=prompt | model | StrOutputParser(),
        llm=model,
        corpus_root=Path("/x"),
    )


def scripted(*lines: Any) -> Any:
    """An ``input()`` replacement: returns lines in order, raises exceptions given as
    items, then EOFError like the end of piped stdin."""
    queue = list(lines)

    def read(prompt: str) -> str:
        if not queue:
            raise EOFError
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def require_default_sigint_handler() -> None:
    """The Runner installs its SIGINT handler only over Python's default one. Under any
    other (a pytest plugin's, say) the SIGINT these tests raise would not be the
    Runner's to handle, and could abort the whole session: fail clearly instead."""
    handler = signal.getsignal(signal.SIGINT)
    if handler is not signal.default_int_handler:
        pytest.fail(
            f"SIGINT handler is {handler!r}, not signal.default_int_handler, so "
            "asyncio.Runner would not install its own; is a plugin installing one?"
        )


def test_an_answer_streams_then_lists_its_sources(capsys: Any) -> None:
    model = ScriptedModel(script=["the answer."])
    assert repl(fake_pipeline(model), scripted("a question")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Partial the answer." in out
    assert "[1] guide.pdf, p. 1" in out
    assert "Question: a question" in model.prompts[0]


def test_a_failed_answer_keeps_the_session_alive(capsys: Any) -> None:
    # The exact error the live check produced with Ollama stopped mid-session
    # (S1-4 step 3, 2026-10-01): httpx's, raised from under the Ollama client.
    stopped = httpx.ConnectError("[Errno 111] Connection refused")
    model = ScriptedModel(script=[stopped, "second answer"])
    assert repl(fake_pipeline(model), scripted("first?", "second?")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Error: ConnectError: [Errno 111] Connection refused" in out
    assert "second answer" in out
    assert len(model.prompts) == 2


def test_ctrl_c_during_an_answer_stops_it_and_returns_to_the_prompt(capsys: Any) -> None:
    require_default_sigint_handler()
    model = ScriptedModel(script=[CTRL_C, "next answer"])
    assert repl(fake_pipeline(model), scripted("slow question", "next")) == EXIT_OK
    out = capsys.readouterr().out
    assert "answer interrupted" in out
    assert "next answer" in out
    # The interrupted stream closed before the next answer began: nothing ran on.
    assert model.log == ["start", "closed", "start", "closed"]
    assert len(set(map(id, model.loops))) == 1  # one loop for the whole session
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_two_ctrl_cs_during_an_answer_end_the_session(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    require_default_sigint_handler()
    # Closing the clients takes several loop iterations, as httpx's does. That gives the
    # half-unwound answer time to resume with the second KeyboardInterrupt and raise it
    # out of the REPL, unless the session cancels it before running the loop again.
    closed = []

    async def slow_aclose(llm: Any) -> None:
        await asyncio.sleep(0.05)
        closed.append(llm)

    monkeypatch.setattr("rag_qa.chain.aclose_llm", slow_aclose)
    model = ScriptedModel(script=[CTRL_C_TWICE, "never asked"])
    try:
        code = repl(fake_pipeline(model), scripted("slow question", "never asked?"))
    except KeyboardInterrupt:
        # Fail this test rather than let pytest read it as the user aborting the run.
        pytest.fail("the second Ctrl-C escaped the REPL as a KeyboardInterrupt (a traceback)")
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "Exiting..." in out
    assert "answer interrupted" not in out
    assert len(model.prompts) == 1  # the session ended; the second question never ran
    assert closed == [model]  # the clients were still closed, on the session's loop
    (loop,) = model.loops
    assert loop.is_closed()
    assert asyncio.all_tasks(loop) == set()  # the half-unwound answer was finalized
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param([], id="end of piped input"),
        pytest.param([KeyboardInterrupt()], id="ctrl-c at the prompt"),
        pytest.param(["exit"], id="exit"),
        pytest.param(["QUIT"], id="quit, any case"),
    ],
)
def test_every_way_out_exits_cleanly(ending: list, capsys: Any) -> None:
    assert repl(fake_pipeline(ScriptedModel()), scripted(*ending)) == EXIT_OK
    assert "Exiting..." in capsys.readouterr().out


def test_blank_lines_are_not_questions() -> None:
    model = ScriptedModel(script=["answer"])
    repl(fake_pipeline(model), scripted("", "   ", "real question"))
    assert len(model.prompts) == 1
    assert "Question: real question" in model.prompts[0]


def test_system_exit_is_not_swallowed() -> None:
    # The REPL catches Exception, not BaseException: an un-quittable REPL would be a
    # worse bug than the one S1-4 fixes.
    with pytest.raises(SystemExit):
        repl(fake_pipeline(ScriptedModel(script=[SystemExit(3)])), scripted("q"))


def test_the_session_closes_the_models_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []

    async def record(llm: Any) -> None:
        closed.append(llm)

    monkeypatch.setattr("rag_qa.chain.aclose_llm", record)
    model = ScriptedModel()
    repl(fake_pipeline(model), scripted("exit"))
    assert closed == [model]


# --- main(): startup ----------------------------------------------------------------


def _pipeline_must_not_be_built(*args: Any) -> None:
    raise AssertionError("build_query_pipeline was called")


def test_help_does_no_work(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_no_index_yet_says_to_run_rag_ingest(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    monkeypatch.setenv("RAG_VECTOR_STORE_PATH", str(tmp_path / "empty"))
    assert cli.main(["--config", str(make_config())]) == EXIT_CANNOT_START
    assert "Run rag-ingest first" in capsys.readouterr().err


def test_invalid_config_exits_2(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    path = make_config(lambda c: c["pipeline"]["query"].update(llm="components.llms.nope"))
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    assert "no entry 'nope'" in capsys.readouterr().err


def test_main_runs_the_session_with_the_built_pipeline(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "store_exists", lambda path: True)
    model = ScriptedModel(script=["an answer"])
    monkeypatch.setattr(cli, "build_query_pipeline", lambda config: fake_pipeline(model))
    # honoured: looked up at call time
    monkeypatch.setattr("builtins.input", scripted("a question"))
    assert cli.main(["--config", str(make_config())]) == EXIT_OK
    assert "an answer" in capsys.readouterr().out
