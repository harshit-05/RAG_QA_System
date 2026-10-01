"""rag-query's REPL survives failures and exits cleanly (S1-4: ISS-06).

Driven hermetically: a fake chain and scripted input, so no Ollama and no index.
The live check (stopping Ollama mid-session) is the story's manual step 3.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import MakeConfig
from langchain_core.documents import Document

from rag_qa import cli
from rag_qa.cli import EXIT_CANNOT_START, EXIT_OK, repl


class FakeChain:
    """Streams like the real chain; each call does the next scripted behaviour."""

    def __init__(self, *behaviours: Any) -> None:
        self.behaviours = list(behaviours)
        self.questions: list[str] = []

    def stream(self, inputs: dict[str, str]) -> Iterator[dict[str, Any]]:
        self.questions.append(inputs["question"])
        behaviour = self.behaviours.pop(0)
        yield {"question": inputs["question"]}
        yield {
            "context": [Document(page_content="c", metadata={"source": "/x/guide.pdf", "page": 0})]
        }
        yield {"answer": "Partial "}
        if isinstance(behaviour, BaseException):
            raise behaviour  # mid-stream, like a dropped Ollama connection
        yield {"answer": behaviour}


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


def test_a_failed_answer_keeps_the_session_alive(capsys: Any) -> None:
    # The exact error the live check produced with Ollama stopped mid-session
    # (S1-4 step 3, 2026-10-01): httpx's, raised from under the Ollama client.
    stopped = httpx.ConnectError("[Errno 111] Connection refused")
    chain = FakeChain(stopped, "second answer")
    assert repl(chain, scripted("first?", "second?")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Error: ConnectError: [Errno 111] Connection refused" in out
    assert "second answer" in out
    assert chain.questions == ["first?", "second?"]


def test_ctrl_c_during_an_answer_returns_to_the_prompt(capsys: Any) -> None:
    chain = FakeChain(KeyboardInterrupt(), "next answer")
    assert repl(chain, scripted("slow question", "next")) == EXIT_OK
    out = capsys.readouterr().out
    assert "answer interrupted" in out
    assert "next answer" in out


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
    assert repl(FakeChain(), scripted(*ending)) == EXIT_OK
    assert "Exiting..." in capsys.readouterr().out


def test_blank_lines_are_not_questions() -> None:
    chain = FakeChain("answer")
    repl(chain, scripted("", "   ", "real question"))
    assert chain.questions == ["real question"]


def test_system_exit_is_not_swallowed() -> None:
    # The REPL catches Exception, not BaseException: an un-quittable REPL would be a
    # worse bug than the one S1-4 fixes.
    with pytest.raises(SystemExit):
        repl(FakeChain(SystemExit(3)), scripted("q"))


# --- main(): startup ----------------------------------------------------------------


def _chain_must_not_be_built(*args: Any) -> None:
    raise AssertionError("build_rag_chain was called")


def test_help_does_no_work(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(cli, "build_rag_chain", _chain_must_not_be_built)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_no_index_yet_says_to_run_rag_ingest(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "build_rag_chain", _chain_must_not_be_built)
    monkeypatch.setenv("RAG_VECTOR_STORE_PATH", str(tmp_path / "empty"))
    assert cli.main(["--config", str(make_config())]) == EXIT_CANNOT_START
    assert "Run rag-ingest first" in capsys.readouterr().err


def test_invalid_config_exits_2(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "build_rag_chain", _chain_must_not_be_built)
    path = make_config(lambda c: c["pipeline"]["query"].update(llm="components.llms.nope"))
    assert cli.main(["--config", str(path)]) == EXIT_CANNOT_START
    assert "no entry 'nope'" in capsys.readouterr().err


def test_main_runs_the_session_with_the_built_chain(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(cli, "store_exists", lambda path: True)
    monkeypatch.setattr(cli, "build_rag_chain", lambda config: FakeChain("an answer"))
    monkeypatch.setattr(
        "builtins.input", scripted("a question")
    )  # honoured: looked up at call time
    assert cli.main(["--config", str(make_config())]) == EXIT_OK
    assert "an answer" in capsys.readouterr().out
