"""rag-query's REPL: it survives failures, exits cleanly, and Ctrl-C stops an answer.

Driven hermetically: a fake pipeline on a scripted chat model, and scripted input, so
no Ollama and no index. The model runs under the REPL's real ``asyncio.Runner``, so a
Ctrl-C is a real SIGINT (``signal.raise_signal``) handled by the Runner's own handler,
exactly as in a terminal (S2-1, DEC-14). The live checks against real Ollama are the
stories' manual steps.
"""

import asyncio
import dataclasses
import re
import signal
import socket
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import (
    FakeOllama,
    MakeConfig,
    StreamingFakeChatModel,
    openai_error,
    use_fakes,
    use_fakes_and_reranker,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import Field
from qdrant_client import QdrantClient

from rag_qa import cli, vectorstore
from rag_qa.answering import Done, Token
from rag_qa.chain import QueryPipeline, RetrievalQuery
from rag_qa.cli import EXIT_CANNOT_START, EXIT_OK, HF_CACHE_HINT, describe_failure, repl
from rag_qa.components import OLLAMA_HINT
from rag_qa.resilience import BreakerOpenError, mark_fallback
from rag_qa.vectorstore import QDRANT_HINT, StoreError, StoreUnavailableError

CTRL_C = "ctrl-c"  # script: the user presses Ctrl-C once, mid-answer
CTRL_C_TWICE = "ctrl-c twice"  # and again while the answer is unwinding
CTRL_C_AT_A_CHUNK = "ctrl-c at a chunk"  # once, while a chunk is being processed


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
            elif behaviour == CTRL_C_AT_A_CHUNK:
                # Ctrl-C while the next chunk is being handed on, not while waiting on
                # Ollama: the moment that used to leave the stream open (S2-3 review).
                signal.raise_signal(signal.SIGINT)
                yield _chunk("more ")
                await asyncio.sleep(30)
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
        prompt=prompt,
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


def dead_url() -> str:
    """A local port nothing listens on, so a connection there is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_an_answer_streams_then_lists_its_sources(capsys: Any) -> None:
    model = ScriptedModel(script=["the answer."])
    assert repl(fake_pipeline(model), scripted("a question")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Answer:" in out
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
    assert OLLAMA_HINT not in out  # this model is not Ollama's, so Ollama is not named
    assert "second answer" in out
    assert len(model.prompts) == 2


def test_a_connection_error_from_ollama_names_it_with_the_hint(capsys: Any) -> None:
    # A real ChatOllama on a closed port: what an answer meets once Ollama has stopped.
    # httpx's own text, "All connection attempts failed", does not say what it could not
    # reach (S2-8).
    llm = ChatOllama(model="mistral", base_url=dead_url())
    assert repl(fake_pipeline(llm), scripted("a question")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Error: the connection to Ollama failed (ConnectError: " in out
    assert OLLAMA_HINT in out
    assert "That question was not answered" in out


@pytest.mark.parametrize(
    ("error", "connection"),
    [
        pytest.param(httpx.ConnectError("All connection attempts failed"), True, id="refused"),
        pytest.param(
            httpx.RemoteProtocolError("peer closed connection"), True, id="dropped mid-answer"
        ),
        pytest.param(ConnectionRefusedError(111, "Connection refused"), True, id="builtin"),
        pytest.param(httpx.ConnectTimeout("no route"), True, id="unreachable"),
        pytest.param(httpx.ReadError("connection reset"), True, id="reset"),
        # Ollama is up and slow: "Is Ollama running?" would point the wrong way (S2-8's review).
        pytest.param(httpx.ReadTimeout("timed out"), False, id="slow, not down"),
        pytest.param(httpx.PoolTimeout("no free connection"), False, id="busy client"),
        pytest.param(ValueError("a model error"), False, id="not a connection error"),
    ],
)
def test_only_a_connection_error_from_an_ollama_model_names_ollama(
    error: Exception, connection: bool
) -> None:
    ollama = ChatOllama(model="mistral", base_url=dead_url())
    assert (OLLAMA_HINT in describe_failure(error, ollama)) is connection
    assert describe_failure(error, ScriptedModel()) == f"{type(error).__name__}: {error}"


def test_an_answer_that_fails_before_its_first_token_prints_no_empty_header(
    capsys: Any,
) -> None:
    def broken(question: str) -> list[Document]:
        raise RuntimeError("the index could not be read")

    pipeline = dataclasses.replace(fake_pipeline(ScriptedModel()), retrieve=RunnableLambda(broken))
    assert repl(pipeline, scripted("a question")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Error: RuntimeError: the index could not be read" in out
    assert "Answer:" not in out


def test_an_answer_with_no_text_still_has_its_header(capsys: Any) -> None:
    # One empty chunk: stream_answer drops it, so Done arrives with no Token before it.
    # (No chunk at all is langchain-core's error, "No generation chunks were returned".)
    model = StreamingFakeChatModel(tokens=[""])
    assert repl(fake_pipeline(model), scripted("q")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Answer:" in out
    assert "[1] guide.pdf, p. 1" in out


@pytest.mark.parametrize(("stdin", "stdout"), [(True, True), (True, False), (False, True)])
def test_the_prompt_marks_its_colour_codes_only_where_readline_reads(
    monkeypatch: pytest.MonkeyPatch, stdin: bool, stdout: bool
) -> None:
    # input() reads through readline only when stdin and stdout are both a terminal.
    # There the colour codes must be marked as zero-width, or editing misplaces the
    # cursor; anywhere else input() prints the prompt itself, and a marker is a stray byte.
    terminals = SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: stdin),
        stdout=SimpleNamespace(isatty=lambda: stdout),
    )
    monkeypatch.setattr(cli, "sys", terminals)
    prompt = cli.question_prompt()
    marked = stdin and stdout
    assert ("\001" in prompt) is marked
    assert re.sub("\001[^\002]*\002", "", prompt) == (
        "Question: " if marked else "\033[94mQuestion:\033[0m "
    )
    # libedit drops a marked sequence that ends the prompt (seen in a pty, S2-8): the reset
    # must come before the last visible character.
    assert prompt.endswith(" ")


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


def test_ctrl_c_while_a_chunk_is_processed_leaves_no_stream_open_at_the_prompt(
    capsys: Any,
) -> None:
    # The moment langchain's per-chunk tasks used to lose: the stream to Ollama stayed
    # open, and Ollama kept generating, until the next question ran the loop again
    # (S2-3's second review). Checked where the user sees it: at the next prompt.
    require_default_sigint_handler()
    model = ScriptedModel(script=[CTRL_C_AT_A_CHUNK, "next answer"])
    log_at_prompt: list[list[str]] = []
    read = scripted("slow question", "next")

    def watching(prompt: str) -> str:
        log_at_prompt.append(list(model.log))
        return read(prompt)

    assert repl(fake_pipeline(model), watching) == EXIT_OK
    assert "answer interrupted" in capsys.readouterr().out
    assert log_at_prompt[1] == ["start", "closed"]  # closed before the prompt came back


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


def test_ctrl_c_while_the_clients_close_leaves_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    async def interrupted(llm: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("rag_qa.chain.aclose_llm", interrupted)
    assert repl(fake_pipeline(ScriptedModel()), scripted("exit")) == EXIT_OK
    assert "(closing interrupted)" in capsys.readouterr().out


def test_an_error_while_the_clients_close_still_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # Found in S2-1's second review: before, this escaped repl() as a traceback.
    async def broken(llm: Any) -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr("rag_qa.chain.aclose_llm", broken)
    assert repl(fake_pipeline(ScriptedModel()), scripted("exit")) == EXIT_OK
    assert "could not close the model's connections cleanly: RuntimeError" in (
        capsys.readouterr().out
    )


# --- main(): startup ----------------------------------------------------------------


def _pipeline_must_not_be_built(*args: Any) -> None:
    raise AssertionError("build_query_pipeline was called")


def test_help_does_no_work(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_no_index_yet_says_to_run_rag_ingest(make_config: MakeConfig, capsys: Any) -> None:
    # The real build_query_pipeline: it raises NoIndexError before building any model.
    assert cli.main(["--config", str(make_config())]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "Run rag-ingest first" in err
    assert "v0.3" not in err  # no FAISS folder beside this config


def test_no_index_with_v03_s_folder_still_there_says_it_is_no_longer_read(
    make_config: MakeConfig, tmp_path: Path, capsys: Any
) -> None:
    # Caveat 20, from S3-2: v0.3's FAISS folder beside the config is never read again.
    config = make_config()
    (config.parent / "vectorstore" / "db_faiss.gen-20261005T080811.306241Z-3e556052").mkdir(
        parents=True
    )
    assert cli.main(["--config", str(config)]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "Run rag-ingest first" in err
    assert f"v0.3's FAISS index in {config.parent / 'vectorstore'} is no longer read" in err
    assert 'README, "Upgrading from v0.3"' in err


def test_qdrant_down_is_one_message_with_the_hint_before_anything_else(
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    # Story step 10: a refused connection, through the real client, to a port nothing uses.
    down = QdrantClient(url=dead_url(), check_compatibility=False, timeout=2)
    monkeypatch.setattr(vectorstore, "connect", lambda *args, **kwargs: down)
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    path = probe_config(make_config, fake_ollama.url)
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "Error: cannot use Qdrant at" in err
    assert QDRANT_HINT in err
    assert "Traceback" not in err
    assert fake_ollama.requests == []  # Qdrant first: Ollama was never asked


def test_qdrant_lost_mid_session_keeps_the_session_and_says_so(capsys: Any) -> None:
    error = StoreUnavailableError(f"cannot use Qdrant at http://q (ConnectError). {QDRANT_HINT}")
    assert describe_failure(error, ScriptedModel()) == str(error)


def test_an_index_from_another_embedder_says_to_rebuild(
    fake_rag: Path, make_config: MakeConfig, capsys: Any
) -> None:
    def other_embedder(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["components"]["embedders"]["fake"]["size"] = 16

    assert cli.main(["--config", str(make_config(other_embedder))]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "different embedder" in err
    assert "rag-ingest --rebuild" in err


def test_invalid_config_exits_2(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    path = make_config(lambda c: c["pipeline"]["query"].update(llm="components.llms.nope"))
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    assert "no entry 'nope'" in capsys.readouterr().err


def probe_config(make_config: MakeConfig, base_url: str, *, validate: bool = True) -> Path:
    """The fakes' config, with an Ollama model at ``base_url`` as the query's llm."""

    def edit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["components"]["llms"]["probe"] = {
            "_target_": "langchain_ollama.ChatOllama",
            "model": "mistral",
            "base_url": base_url,
            "validate_model_on_init": validate,
        }
        c["pipeline"]["query"]["llm"] = "components.llms.probe"

    return make_config(edit)


def test_main_runs_the_session_with_the_built_pipeline(
    fake_rag: Path,
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    model = ScriptedModel(script=["an answer"])
    monkeypatch.setattr(cli, "build_query_pipeline", lambda config, **kw: fake_pipeline(model))
    # honoured: looked up at call time
    monkeypatch.setattr("builtins.input", scripted("a question"))
    path = probe_config(make_config, fake_ollama.url)  # over fake_rag's index
    assert cli.main(["--config", str(path)]) == EXIT_OK
    assert "an answer" in capsys.readouterr().out
    # The start-up check asked Ollama, as rag-serve's does: validate_model_on_init, then
    # the ping.
    assert fake_ollama.requests.count("/api/tags") == 2


# --- --source: the sources filter (S3-5, FR-3) ---------------------------------------------


def test_sources_reach_every_answer_once_each_and_the_session_says_so(
    fake_rag: Path,
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    asked: list[RetrievalQuery] = []
    model = ScriptedModel(script=["one", "two"])
    pipeline = dataclasses.replace(
        fake_pipeline(model),
        retrieve=RunnableLambda(lambda query: asked.append(query) or []),
    )
    monkeypatch.setattr(cli, "build_query_pipeline", lambda config, **kw: pipeline)
    monkeypatch.setattr("builtins.input", scripted("first", "second"))
    path = probe_config(make_config, fake_ollama.url)
    argv = ["--config", str(path), "--source", "sub/report.docx", "--source", "notes.txt"]
    assert cli.main([*argv, "--source", "sub/report.docx"]) == EXIT_OK
    sources = ("sub/report.docx", "notes.txt")
    assert asked == [RetrievalQuery("first", sources), RetrievalQuery("second", sources)]
    assert "Answering from sub/report.docx, notes.txt only." in capsys.readouterr().out


def test_a_source_the_index_does_not_hold_stops_the_start_before_ollama_is_asked(
    fake_rag: Path, make_config: MakeConfig, fake_ollama: FakeOllama, capsys: Any
) -> None:
    path = probe_config(make_config, fake_ollama.url)
    argv = ["--config", str(path), "--source", "notes.txt", "--source", "note.txt"]
    assert cli.main([*argv, "--source", "corpus/nope.pdf"]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "--source names documents not in the index:" in err
    assert "'note.txt' (did you mean 'notes.txt'?)" in err
    assert "'corpus/nope.pdf'" in err
    assert "A source is a path relative to the corpus" in err
    assert "'notes.txt' (" not in err  # the one the index holds is not named
    assert fake_ollama.requests == []  # refused from the manifest, before the model


def test_without_sources_a_session_searches_the_whole_corpus(
    fake_rag: Path,
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    asked: list[RetrievalQuery] = []
    pipeline = dataclasses.replace(
        fake_pipeline(ScriptedModel(script=["one"])),
        retrieve=RunnableLambda(lambda query: asked.append(query) or []),
    )
    monkeypatch.setattr(cli, "build_query_pipeline", lambda config, **kw: pipeline)
    monkeypatch.setattr("builtins.input", scripted("first"))
    assert cli.main(["--config", str(probe_config(make_config, fake_ollama.url))]) == EXIT_OK
    assert asked == [RetrievalQuery("first")]
    assert "only." not in capsys.readouterr().out


# --- main(): the start-up guard (S2-8) -------------------------------------------------


@pytest.mark.parametrize(
    ("validate", "reason"),
    [
        (
            True,
            (
                "cannot build the language model components.llms.probe: ConnectionError: "
                "Failed to connect to Ollama"
            ),
        ),
        (False, "the language model components.llms.probe did not answer within 2 s"),
    ],
)
def test_ollama_down_at_start_up_is_one_reason_and_a_hint_not_a_traceback(
    fake_rag: Path,
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    validate: bool,
    reason: str,
) -> None:
    # Before the embedder and the reranker load: they take seconds, and are not needed.
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    path = probe_config(make_config, dead_url(), validate=validate)
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert f"Error: {reason}" in err
    assert OLLAMA_HINT in err
    assert "Traceback" not in err


def test_a_language_model_that_cannot_be_built_stops_the_start_with_its_reason(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    def refuse(config: Any) -> None:
        raise RuntimeError("model 'mistral' not found")

    monkeypatch.setattr("rag_qa.components.build_llm", refuse)  # the llm factory
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    assert cli.main(["--config", str(fake_rag)]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert (
        "Error: cannot build the language model components.llms.fake: RuntimeError: "
        "model 'mistral' not found"
    ) in err
    assert OLLAMA_HINT not in err  # the fake is not an Ollama model


def test_a_missing_index_is_refused_before_ollama_is_asked(
    make_config: MakeConfig, fake_ollama: FakeOllama, capsys: Any
) -> None:
    path = probe_config(make_config, fake_ollama.url)  # the test's store holds no index
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    assert "Run rag-ingest first" in capsys.readouterr().err
    assert fake_ollama.requests == []


def test_a_model_missing_from_the_hf_cache_offline_is_one_reason(
    fake_rag: Path,
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    # S2-4's case: the reranker's model, never downloaded, on a machine now offline. The
    # real build_query_pipeline, with the error transformers raises (seen in S2-8).
    offline = (
        "We couldn't connect to 'https://huggingface.co' to load the files, and couldn't "
        "find them in the cached files."
    )

    def cross_encoder(*args: Any, **kwargs: Any) -> None:
        raise OSError(offline)

    monkeypatch.setattr("rag_qa.rerankers.CrossEncoder", cross_encoder)
    assert cli.main(["--config", str(make_config(use_fakes_and_reranker))]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert f"Error: cannot load the models: OSError: {offline}" in err
    assert HF_CACHE_HINT in err
    assert "Traceback" not in err


def test_an_index_that_cannot_be_read_is_not_blamed_on_the_hf_cache(
    fake_rag: Path,
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    # S2-8's review: the one broad catch around the start used to give an error from the
    # index check "cannot load the models" and the Hugging Face hint, which sent the reader
    # to the wrong place. Since S3-2 the index check reads Qdrant, not files.
    def unreadable(config: Any, **kwargs: Any) -> None:
        raise StoreError("Qdrant at http://q answered 500: segment is corrupt")

    monkeypatch.setattr(cli, "check_index", unreadable)
    assert cli.main(["--config", str(make_config(use_fakes))]) == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "Error: cannot read the index: Qdrant at http://q answered 500" in err
    assert "cannot load the models" not in err
    assert HF_CACHE_HINT not in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "where",
    [
        pytest.param("rag_qa.components.build_llm", id="checking the language model"),
        pytest.param("rag_qa.cli.build_query_pipeline", id="loading the models"),
    ],
)
def test_ctrl_c_while_loading_exits_2_without_a_traceback(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, where: str
) -> None:
    def interrupted(config: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(where, interrupted)
    try:
        status = cli.main(["--config", str(fake_rag)])
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C while loading escaped rag-query as a KeyboardInterrupt (a traceback)")
    assert status == EXIT_CANNOT_START
    assert "Interrupted while loading; the session did not start." in capsys.readouterr().err


def test_ctrl_c_during_the_start_up_ping_exits_2_without_a_traceback(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # A real SIGINT, while preflight's own event loop waits on the server: that loop's
    # Runner turns it into a cancel, then a KeyboardInterrupt out of asyncio.run.
    require_default_sigint_handler()
    closed = []

    async def hanging_ping(llm: Any, timeout_s: float = 2.0) -> bool:
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(30)
        return True

    async def record(llm: Any) -> None:
        closed.append(llm)

    monkeypatch.setattr("rag_qa.components.llm_reachable", hanging_ping)
    monkeypatch.setattr("rag_qa.components.aclose_llm", record)
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    try:
        status = cli.main(["--config", str(fake_rag)])
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C during the ping escaped rag-query as a KeyboardInterrupt")
    assert status == EXIT_CANNOT_START
    assert "Interrupted while loading" in capsys.readouterr().err
    assert len(closed) == 1  # the checked model's clients were still closed
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# --- the hosted fallback (S3-6, DEC-21) ---------------------------------------------------

FALLBACK = "components.llms.groq_fallback"


def fallback_config(make_config: MakeConfig, base_url: str) -> Path:
    """:func:`probe_config`'s, with the real config's hosted fallback set."""

    def edit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["components"]["llms"]["probe"] = {
            "_target_": "langchain_ollama.ChatOllama",
            "model": "mistral",
            "base_url": base_url,
            "validate_model_on_init": True,
        }
        c["pipeline"]["query"]["llm"] = "components.llms.probe"
        c["pipeline"]["query"]["fallback_llm"] = FALLBACK

    return make_config(edit)


def test_with_a_fallback_ollama_down_at_start_up_is_a_warning_and_the_fallback_answers(
    fake_rag: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # DEC-21, design review 1: Ollama down is the fallback's case, so it cannot stop the
    # start. Its model is built without asking Ollama, and the fallback answers.
    spare = StreamingFakeChatModel(tokens=["Spare."])
    monkeypatch.setattr("rag_qa.components.build_fallback_llm", lambda config: spare)
    monkeypatch.setattr("rag_qa.chain.build_fallback_llm", lambda config: spare)
    monkeypatch.setattr("builtins.input", scripted("a question"))
    assert cli.main(["--config", str(fallback_config(make_config, dead_url()))]) == EXIT_OK
    out, err = capsys.readouterr()
    assert (
        f"Warning: the language model components.llms.probe did not answer within 2 s. Until "
        f"it does, the hosted fallback {FALLBACK} answers, and each question and its retrieved "
        f"chunks are sent to https://api.groq.com/openai/v1." in err
    )
    assert OLLAMA_HINT in err
    assert f"When Ollama cannot answer, the hosted fallback {FALLBACK} does" in out
    assert "Spare." in out
    assert f"(answered by the hosted fallback, {FALLBACK})" in out


def test_with_a_fallback_and_no_key_the_session_does_not_start(
    fake_rag: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    path = fallback_config(make_config, dead_url())
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    assert (
        f"Error: the hosted fallback {FALLBACK} needs its API key in RAG_FALLBACK_API_KEY, "
        f"which is not set." in capsys.readouterr().err
    )


def test_with_the_key_but_not_the_extra_the_session_says_how_to_install_it(
    fake_rag: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv("RAG_FALLBACK_API_KEY", "gsk-sentinel-value")
    monkeypatch.setitem(sys.modules, "openai", None)  # as without `uv sync --extra fallback`
    monkeypatch.setattr(cli, "build_query_pipeline", _pipeline_must_not_be_built)
    assert (
        cli.main(["--config", str(fallback_config(make_config, dead_url()))]) == EXIT_CANNOT_START
    )
    err = capsys.readouterr().err
    assert "install it with `uv sync --extra fallback`" in err
    assert "gsk-sentinel-value" not in err


def test_the_line_under_an_answer_names_the_fallback_only_when_it_answered(capsys: Any) -> None:
    async def events(model: str) -> AsyncIterator[Any]:
        yield Token("An answer.")
        yield Done(ttft_ms=1.0, total_ms=2.0, model=model)

    asyncio.run(cli.render(events(FALLBACK), fallback=FALLBACK))
    assert f"(answered by the hosted fallback, {FALLBACK})" in capsys.readouterr().out
    asyncio.run(cli.render(events("components.llms.mistral_ollama"), fallback=FALLBACK))
    assert "answered by" not in capsys.readouterr().out


def test_a_dropped_fallback_stream_is_the_fallback_s_not_ollama_s() -> None:
    # httpx's error, as the fallback's stream raises it: marked by the answer that read it.
    ollama = ChatOllama(model="mistral", base_url=dead_url())
    error = httpx.RemoteProtocolError("peer closed connection")
    unmarked = describe_failure(error, ollama)
    assert unmarked.startswith("the connection to Ollama failed") and OLLAMA_HINT in unmarked
    mark_fallback(error)
    said = describe_failure(error, ollama)
    assert said == "the hosted fallback failed (RemoteProtocolError: peer closed connection)"


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (
            BreakerOpenError("Ollama", 12.0, 3),
            (
                "Ollama is not answering: 3 attempts in a row failed, so it is not asked again "
                f"for 12 s\n{OLLAMA_HINT}"
            ),
        ),
        (
            BreakerOpenError("the hosted fallback", 5.0, 3),
            (
                "the hosted fallback is not answering: 3 attempts in a row failed, so it is "
                "not asked again for 5 s"
            ),
        ),
        (
            openai_error("APIConnectionError")("Connection error."),
            "the hosted fallback failed (APIConnectionError: Connection error.)",
        ),
        (
            openai_error("AuthenticationError", status_code=401)("Invalid API Key"),
            (
                "the hosted fallback failed (AuthenticationError: Invalid API Key)\n"
                "Check RAG_FALLBACK_API_KEY."
            ),
        ),
    ],
    ids=["ollama's breaker", "the fallback's breaker", "fallback unreachable", "key refused"],
)
def test_a_breaker_s_refusal_and_the_fallback_s_errors_say_whose_they_are(
    error: Exception, said: str
) -> None:
    ollama = ChatOllama(model="mistral", base_url=dead_url())
    assert describe_failure(error, ollama) == said
