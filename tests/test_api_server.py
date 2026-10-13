"""``rag-serve`` (S2-7, DEC-18): what it refuses before it starts, and how it starts.

uvicorn itself is replaced by a recorder here (``test_api_uvicorn.py`` runs the real one),
so these tests check the command's decisions, not the server.
"""

import logging
import socket
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from conftest import FakeOllama, MakeConfig, use_fakes
from pydantic import ValidationError

from rag_qa.api import server
from rag_qa.components import OLLAMA_HINT
from rag_qa.settings import ServerSettings, describe_settings_error, is_loopback

real_configure_logging = server.configure_logging


@pytest.fixture(autouse=True)
def no_global_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """``rag-serve`` attaches a handler to the package logger for the process's life. In
    a test session that handler would outlive the test's captured stderr."""
    monkeypatch.setattr(server, "configure_logging", lambda: None)


def test_logging_is_configured_once_however_often_main_runs() -> None:
    package = logging.getLogger("rag_qa")
    handlers, level = list(package.handlers), package.level
    try:
        real_configure_logging()
        real_configure_logging()
        added = [h for h in package.handlers if h not in handlers]
        assert len(added) == 1
        assert package.level == logging.INFO
    finally:
        package.handlers[:] = handlers
        package.setLevel(level)


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[uvicorn.Config]:
    """Record the uvicorn configs ``rag-serve`` would run, and run none."""
    configs: list[uvicorn.Config] = []

    def record(self: uvicorn.Server, sockets: Any = None) -> None:
        configs.append(self.config)
        self.started = True

    monkeypatch.setattr(uvicorn.Server, "run", record)
    return configs


def ollama_config(make_config: MakeConfig, base_url: str, model: str = "mistral") -> Path:
    """The fakes' config, with an Ollama model at ``base_url`` that is checked at start-up."""

    def edit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["components"]["llms"]["probe"] = {
            "_target_": "langchain_ollama.ChatOllama",
            "model": model,
            "base_url": base_url,
            "validate_model_on_init": True,
        }
        c["pipeline"]["query"]["llm"] = "components.llms.probe"

    return make_config(edit)


def dead_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_a_non_loopback_host_without_a_token_is_refused_before_the_config_loads(
    capsys: pytest.CaptureFixture[str], ran: list[uvicorn.Config]
) -> None:
    status = server.main(["--host", "0.0.0.0", "--config", "/nonexistent/config.yaml"])
    err = capsys.readouterr().err
    assert status == server.EXIT_CANNOT_START
    assert "refusing to listen on 0.0.0.0 without a token" in err
    assert "Traceback" not in err
    assert ran == []


def test_a_non_loopback_host_from_the_environment_is_refused_too(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, ran: list[uvicorn.Config]
) -> None:
    monkeypatch.setenv("RAG_API_HOST", "192.168.1.10")
    assert server.main([]) == server.EXIT_CANNOT_START
    assert "refusing to listen on 192.168.1.10" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("variable", "value", "problem"),
    [
        ("RAG_API_MAX_CONCURRENT", "0", "RAG_API_MAX_CONCURRENT: Input should be greater than"),
        ("RAG_API_MAX_CONCURRENT", "many", "RAG_API_MAX_CONCURRENT: Input should be a valid"),
        ("RAG_API_PORT", "70000", "RAG_API_PORT: Input should be less than"),
        ("RAG_API_TOKEN", "two words", "RAG_API_TOKEN: must be visible ASCII"),
        ("RAG_API_TOKEN", "sécret", "RAG_API_TOKEN: must be visible ASCII"),
    ],
)
def test_invalid_settings_stop_the_start_and_never_echo_a_value(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    ran: list[uvicorn.Config],
    variable: str,
    value: str,
    problem: str,
) -> None:
    monkeypatch.setenv(variable, value)
    assert server.main([]) == server.EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert problem in err
    assert value not in err  # a token's own value must never reach a terminal or a log
    assert ran == []


def test_a_config_error_stops_the_start(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, ran: list[uvicorn.Config]
) -> None:
    assert server.main(["--config", str(tmp_path / "missing.yaml")]) == server.EXIT_CANNOT_START
    assert "Config file not found" in capsys.readouterr().err
    assert ran == []


def test_ollama_down_at_start_up_is_a_reason_and_a_hint_not_a_traceback(
    capsys: pytest.CaptureFixture[str], make_config: MakeConfig, ran: list[uvicorn.Config]
) -> None:
    path = ollama_config(make_config, dead_url())
    assert server.main(["--config", str(path)]) == server.EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "cannot build the language model components.llms.probe" in err
    assert "Failed to connect to Ollama" in err
    assert OLLAMA_HINT in err
    assert "Traceback" not in err
    assert ran == []


def test_a_model_ollama_lacks_is_named_at_start_up(
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    ran: list[uvicorn.Config],
) -> None:
    path = ollama_config(make_config, fake_ollama.url, model="qwen2:7b")
    assert server.main(["--config", str(path)]) == server.EXIT_CANNOT_START
    assert "Model `qwen2:7b` not found in Ollama" in capsys.readouterr().err
    assert ran == []


def test_with_ollama_up_it_runs_uvicorn_on_the_given_address(
    make_config: MakeConfig, fake_ollama: FakeOllama, ran: list[uvicorn.Config]
) -> None:
    path = ollama_config(make_config, fake_ollama.url)
    assert server.main(["--config", str(path), "--port", "8123"]) == server.EXIT_OK
    (config,) = ran
    assert (config.host, config.port) == ("127.0.0.1", 8123)
    assert config.timeout_graceful_shutdown == server.GRACEFUL_SHUTDOWN_S
    assert fake_ollama.requests.count("/api/tags") == 2  # validate_model_on_init, then the ping


def test_with_a_token_a_non_loopback_host_runs_and_warns_about_cleartext(
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ran: list[uvicorn.Config],
) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "s3cret")
    path = ollama_config(make_config, fake_ollama.url)
    with caplog.at_level(logging.WARNING, logger="rag_qa"):
        assert server.main(["--config", str(path), "--host", "0.0.0.0"]) == server.EXIT_OK
    assert ran[0].host == "0.0.0.0"
    assert "travels in cleartext" in caplog.text
    assert "s3cret" not in caplog.text


@pytest.mark.parametrize(("raised", "started"), [(True, False), (False, False)])
def test_a_server_that_did_not_start_exits_2(
    make_config: MakeConfig,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
    raised: bool,
    started: bool,
) -> None:
    # uvicorn exits when it cannot bind the address; a failed lifespan returns unstarted.
    def run(self: uvicorn.Server, sockets: Any = None) -> None:
        self.started = started
        if raised:
            raise SystemExit(1)

    monkeypatch.setattr(uvicorn.Server, "run", run)
    path = ollama_config(make_config, fake_ollama.url)
    assert server.main(["--config", str(path)]) == server.EXIT_CANNOT_START


@pytest.mark.parametrize("port", ["0", "65536", "http"])
def test_a_port_out_of_range_is_a_usage_error(
    port: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        server.main(["--port", port])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.8.9.10", True),
        ("::1", True),
        ("localhost", True),
        ("LocalHost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("localhost.example.com", False),
        ("my-laptop", False),
        ("", False),
    ],
)
def test_loopback_is_localhost_or_a_loopback_address(host: str, loopback: bool) -> None:
    assert is_loopback(host) is loopback


def test_server_settings_default_to_a_tokenless_loopback_server() -> None:
    settings = ServerSettings()
    assert (settings.token, settings.max_concurrent, settings.host, settings.port) == (
        None, 1, "127.0.0.1", 8000
    )


def test_an_empty_token_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "")
    assert ServerSettings().token is None


def test_a_settings_error_names_the_variable_without_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "pass word")
    with pytest.raises(ValidationError) as exc:
        ServerSettings()
    described = describe_settings_error(exc.value)
    assert described.startswith("RAG_API_TOKEN: must be visible ASCII")
    assert "pass word" not in described


def test_ollama_down_is_caught_even_without_validate_model_on_init(
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    ran: list[uvicorn.Config],
) -> None:
    # The model builds without asking Ollama; the start-up ping asks instead.
    path = ollama_config(make_config, dead_url())
    text = path.read_text().replace("validate_model_on_init: true", "validate_model_on_init: false")
    path.write_text(text)
    assert server.main(["--config", str(path)]) == server.EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "the language model components.llms.probe did not answer within 2 s" in err
    assert OLLAMA_HINT in err
    assert ran == []
