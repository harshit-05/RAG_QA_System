"""Who may ask the API (S3-8, DEC-23): keys, scopes and the rate limits.

:class:`rag_qa.api.auth.KeyRing` first, on its own, with a clock the tests move; then through
the app, with FastAPI's test client, whose requests all come from the address ``testclient``.
The every-route checks (no token: 401; a key without the route's scope: 403) are in
``test_api.py``, beside the token's; a spoofed ``X-Forwarded-For`` over a real socket is in
``test_api_uvicorn.py``.
"""

import hmac
import json
import logging
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from conftest import StreamingFakeChatModel, Traced, fake_serving
from fastapi.testclient import TestClient
from opentelemetry.trace import SpanKind
from pydantic import SecretStr

from rag_qa.api import auth
from rag_qa.api.app import create_app
from rag_qa.api.auth import ApiKey, KeyRing, KeysFileError
from rag_qa.api.state import Serving
from rag_qa.observability import logs, metrics, tracing
from rag_qa.schema import RagConfig
from rag_qa.settings import ServerSettings

TOKEN = "s3cret-token"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def write_keys(path: Path, **scopes: str) -> dict[str, str]:
    """A keys file at ``path`` holding one key per name, its scopes comma-separated; returns
    each key's token. Replaced atomically, as ``rag-keys`` does."""
    tokens = {name: secrets.token_urlsafe(32) for name in scopes}
    keys = [
        ApiKey(name, frozenset(scopes[name].split(",")), auth.digest(token), "2026-10-26T00:00:00Z")
        for name, token in tokens.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(auth.dump(keys), encoding="utf-8")
    os.replace(temporary, path)
    return tokens


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def refusals(reason: str) -> float:
    code = metrics.REFUSAL_CODES[reason]
    return (
        metrics.REGISTRY.get_sample_value("rag_refusals_total", {"code": code, "reason": reason})
        or 0.0
    )


@pytest.fixture
def keys_file(tmp_path: Path) -> Path:
    return tmp_path / "var" / "api-keys.json"


@contextmanager
def serve(
    config: RagConfig,
    *,
    keys_file: Path | None = None,
    token: str | None = None,
    rate_limit: int = 1_000_000,
    serving: Serving | None = None,
    **client: Any,
) -> Iterator[TestClient]:
    settings = ServerSettings(
        keys_file=keys_file,
        token=None if token is None else SecretStr(token),
        rate_limit=rate_limit,
    )
    if serving is None:
        serving = fake_serving(config, StreamingFakeChatModel())
    app = create_app(config, settings, serving=serving)
    with TestClient(app, base_url="http://127.0.0.1", **client) as test_client:
        yield test_client


# --- the keys file ---------------------------------------------------------------------------


def test_a_keys_file_round_trips_and_holds_no_token(keys_file: Path) -> None:
    tokens = write_keys(keys_file, laptop="query,ingest", bot="query")
    text = keys_file.read_text()
    assert all(token not in text for token in tokens.values())
    keys = auth.read_keys(keys_file)
    assert [(k.name, sorted(k.scopes)) for k in keys] == [
        ("laptop", ["ingest", "query"]),
        ("bot", ["query"]),
    ]
    assert auth.parse(auth.dump(keys)) == keys
    assert json.loads(text)["keys"][1]["sha256"] == auth.digest(tokens["bot"]).hex()


def entry(**overrides: Any) -> dict[str, Any]:
    return {"name": "bot", "scopes": ["query"], "sha256": "ab" * 32, "created": "x", **overrides}


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("{not json", "not valid JSON"),
        ('{"version": 2, "keys": []}', "not a version 1 keys file"),
        ('{"version": 1, "keys": [], "more": 1}', "not a version 1 keys file"),
        ('{"version": 1, "keys": {}}', "keys are not a list"),
        (json.dumps({"version": 1, "keys": [entry(token="x")]}), "key 1 is not a name"),
        (json.dumps({"version": 1, "keys": [entry(name="Bot")]}), "no valid name"),
        (json.dumps({"version": 1, "keys": [entry(name="token")]}), "no valid name"),
        (json.dumps({"version": 1, "keys": [entry(scopes=[])]}), "no valid scopes"),
        (json.dumps({"version": 1, "keys": [entry(scopes=["admin"])]}), "no valid scopes"),
        (json.dumps({"version": 1, "keys": [entry(sha256="AB" * 32)]}), "no valid sha256"),
        (json.dumps({"version": 1, "keys": [entry(created=1)]}), "no valid created"),
        (json.dumps({"version": 1, "keys": [entry(), entry()]}), "names a key twice"),
    ],
)
def test_a_keys_file_that_does_not_parse_says_why_without_its_path(
    text: str, problem: str, keys_file: Path
) -> None:
    keys_file.parent.mkdir(parents=True)
    keys_file.write_text(text)
    with pytest.raises(KeysFileError) as raised:
        auth.read_keys(keys_file)
    assert problem in str(raised.value)
    assert str(keys_file.parent) not in str(raised.value)


def test_a_missing_or_unreadable_keys_file_says_so(keys_file: Path) -> None:
    with pytest.raises(KeysFileError, match="does not exist"):
        auth.read_keys(keys_file)
    keys_file.parent.mkdir(parents=True)
    keys_file.mkdir()  # a folder in its place: reading it fails
    with pytest.raises(KeysFileError, match="cannot be read"):
        auth.read_keys(keys_file)


# --- the key ring --------------------------------------------------------------------------------


def test_the_token_is_a_key_named_token_with_every_scope(keys_file: Path) -> None:
    ring = KeyRing(None, SecretStr(TOKEN))
    key = ring.match(TOKEN)
    assert key is not None and (key.name, key.scopes) == ("token", frozenset(auth.SCOPES))
    assert ring.match(TOKEN + "x") is None
    assert ring.configured and ring.problem is None
    assert not KeyRing(None, None).configured


def test_every_key_is_compared_even_after_a_match(
    keys_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No early exit: a wrong token costs what a right one does, the first key's included.
    tokens = write_keys(keys_file, a="query", b="query", c="ingest")
    ring = KeyRing(keys_file, SecretStr(TOKEN))
    compared: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        assert isinstance(a, bytes) and isinstance(b, bytes)  # on bytes, never str
        compared.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    for presented in (TOKEN, tokens["a"], "wrong", "sécret"):
        compared.clear()
        ring.match(presented)
        assert len(compared) == 4, presented  # the token and the file's three


def test_a_change_to_the_file_is_seen_within_5_s_and_no_sooner_than_it_is_looked_for(
    keys_file: Path,
) -> None:
    clock = Clock()
    tokens = write_keys(keys_file, bot="query")
    ring = KeyRing(keys_file, None, clock=clock)
    assert ring.match(tokens["bot"]) is not None
    write_keys(keys_file, other="query")  # bot revoked: a new file, a new inode
    clock.now += 4.9
    assert ring.match(tokens["bot"]) is not None  # not looked at again yet
    clock.now += 0.1
    assert ring.match(tokens["bot"]) is None  # 5 s: looked at, and read again


def test_a_file_edited_in_place_is_read_again_too(keys_file: Path) -> None:
    # Same inode: the size or the mtime moves.
    clock = Clock()
    tokens = write_keys(keys_file, bot="query")
    ring = KeyRing(keys_file, None, clock=clock)
    keys_file.write_text(keys_file.read_text().replace('"query"', '"ingest"'))
    clock.now += 5
    key = ring.match(tokens["bot"])
    assert key is not None and key.scopes == {"ingest"}


def test_a_broken_file_fails_closed_until_it_parses_again(
    keys_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock()
    tokens = write_keys(keys_file, bot="query")
    ring = KeyRing(keys_file, SecretStr(TOKEN), clock=clock)
    keys_file.write_text("{half a file")
    clock.now += 5
    assert ring.match(tokens["bot"]) is None
    assert ring.problem == "the keys file is not valid JSON"
    assert ring.match(TOKEN) is not None  # RAG_API_TOKEN still works
    assert "cannot be used" in caplog.text and str(keys_file) in caplog.text  # the log may
    fixed = write_keys(keys_file, bot="query", other="ingest")
    clock.now += 5
    assert [ring.match(fixed[name]) is not None for name in ("bot", "other")] == [True, True]
    assert ring.problem is None


def test_a_missing_file_at_start_up_holds_no_key_and_is_a_warning(
    keys_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="rag_qa.api.auth")
    ring = KeyRing(keys_file, None)
    assert ring.configured
    assert ring.problem == "the keys file does not exist"
    assert "cannot be used" in caplog.text


# --- through the app: keys and scopes ----------------------------------------------------------


def test_a_key_is_accepted_and_a_wrong_or_missing_one_is_401(
    fake_config: RagConfig, keys_file: Path
) -> None:
    tokens = write_keys(keys_file, bot="query")
    with serve(fake_config, keys_file=keys_file) as client:
        assert (
            client.post(
                "/v1/query", json={"question": "q"}, headers=bearer(tokens["bot"])
            ).status_code
            == 200
        )
        for headers in ({}, bearer("wrong"), {"Authorization": tokens["bot"]}):
            response = client.post("/v1/query", json={"question": "q"}, headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json() == {
                "detail": "a valid API key is required (Authorization: Bearer <token>)"
            }


def test_with_a_keys_file_alone_any_host_is_served(fake_config: RagConfig, keys_file: Path) -> None:
    # "No credential" means no keys file and no token (DEC-23, design review 2): with a keys
    # file alone, a request through a proxy under a public name is served.
    tokens = write_keys(keys_file, ops="ingest")
    with serve(fake_config, keys_file=keys_file) as client:
        for host in ("rag.example.org", "app:8000"):
            headers = {"host": host, **bearer(tokens["ops"])}
            assert client.get("/v1/ingest/x", headers=headers).status_code == 404


def test_a_key_and_the_token_coexist(fake_config: RagConfig, keys_file: Path) -> None:
    tokens = write_keys(keys_file, bot="query")
    with serve(fake_config, keys_file=keys_file, token=TOKEN) as client:
        assert client.get("/v1/ingest/x", headers=bearer(TOKEN)).status_code == 404
        assert client.get("/v1/ingest/x", headers=bearer(tokens["bot"])).status_code == 403


def test_a_broken_keys_file_refuses_keys_with_503_and_health_says_why(
    fake_config: RagConfig, keys_file: Path
) -> None:
    keys_file.parent.mkdir(parents=True)
    keys_file.write_text("[]")
    before = refusals(metrics.KEYS_UNREADABLE)
    with serve(fake_config, keys_file=keys_file, token=TOKEN) as client:
        refused = client.post("/v1/query", json={"question": "q"}, headers=bearer("any-key"))
        assert refused.status_code == 503
        assert refused.headers["retry-after"] == "5"
        assert "keys file cannot be used" in refused.json()["detail"]
        # The token still works, and a request with no key at all is a plain 401.
        assert (
            client.post("/v1/query", json={"question": "q"}, headers=bearer(TOKEN)).status_code
            == 200
        )
        assert client.post("/v1/query", json={"question": "q"}).status_code == 401
        health = client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["status"] == "not ready"
        assert health.json()["keys"] == {
            "ready": False,
            "detail": "the keys file is not a version 1 keys file: every API key is refused "
            "with 503 until it is fixed",
        }
        for response in (refused, health):
            assert str(keys_file.parent) not in response.text  # never the path (DEC-18)
    assert refusals(metrics.KEYS_UNREADABLE) - before == 1


def test_health_reports_a_usable_keys_file_and_none_without_one(
    fake_config: RagConfig, keys_file: Path
) -> None:
    write_keys(keys_file, bot="query")
    with serve(fake_config, keys_file=keys_file) as client:
        keys = client.get("/v1/health").json()["keys"]
    assert keys == {"ready": True, "detail": "the keys file is read"}
    with serve(fake_config, token=TOKEN) as client:
        assert client.get("/v1/health").json()["keys"] is None


def test_wrong_guesses_draw_on_the_failure_bucket_while_the_file_is_broken(
    fake_config: RagConfig, keys_file: Path
) -> None:
    # Else a broken file would answer every wrong guess at the token with an unthrottled 503,
    # and the right one with a 200.
    keys_file.parent.mkdir(parents=True)
    keys_file.write_text("{")
    with serve(fake_config, keys_file=keys_file, token=TOKEN) as client:
        codes = [
            client.get("/v1/ingest/x", headers=bearer(f"guess-{n}")).status_code for n in range(11)
        ]
    assert codes == [503] * 10 + [429]


def test_a_revoked_key_is_refused_within_5_s(fake_config: RagConfig, keys_file: Path) -> None:
    clock = Clock()
    tokens = write_keys(keys_file, bot="query", ops="ingest")
    with serve(fake_config, keys_file=keys_file) as client:
        client.app.state.keys = KeyRing(keys_file, None, clock=clock)  # type: ignore[attr-defined]
        assert client.get("/v1/documents", headers=bearer(tokens["bot"])).status_code != 401
        write_keys(keys_file, ops="ingest")
        clock.now += 5
        assert client.get("/v1/documents", headers=bearer(tokens["bot"])).status_code == 401


def test_the_key_s_name_is_in_the_logs_and_on_the_server_span_and_its_token_in_neither(
    fake_config: RagConfig,
    keys_file: Path,
    traced: Traced,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokens = write_keys(keys_file, laptop="query,ingest")
    monkeypatch.setenv("RAG_LOG_FORMAT", "json")
    logs.configure_logging()
    with serve(fake_config, keys_file=keys_file) as client:
        answer = client.post("/v1/query", json={"question": "q"}, headers=bearer(tokens["laptop"]))
        assert answer.status_code == 200
        assert (
            client.post("/v1/query", json={"question": "q"}, headers=bearer("wrong")).status_code
            == 401
        )
    err = capsys.readouterr().err
    lines = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    answered = [line for line in lines if line["event"].startswith("Answered")]
    assert answered and all(line["api_key"] == "laptop" for line in answered)
    assert tokens["laptop"] not in err and "wrong" not in err
    servers = [s for s in traced.spans() if s.kind is SpanKind.SERVER]
    assert [dict(s.attributes or {}).get(tracing.API_KEY_NAME) for s in servers] == ["laptop", None]
    assert all(tokens["laptop"] not in json.dumps(dict(s.attributes or {})) for s in traced.spans())


# --- through the app: the rate limits -----------------------------------------------------------


def test_the_eleventh_wrong_token_in_a_minute_is_429_and_a_valid_key_is_never_refused_for_it(
    fake_config: RagConfig, keys_file: Path
) -> None:
    tokens = write_keys(keys_file, bot="query")
    before = {r: refusals(r) for r in (metrics.UNAUTHENTICATED, metrics.AUTH_LIMITED)}
    with serve(fake_config, keys_file=keys_file) as client:
        codes = []
        for n in range(11):
            codes.append(
                client.post(
                    "/v1/query", json={"question": "q"}, headers=bearer(f"guess-{n}")
                ).status_code
            )
            # The same address, between the guesses: never refused for them.
            assert (
                client.post(
                    "/v1/query", json={"question": "q"}, headers=bearer(tokens["bot"])
                ).status_code
                == 200
            )
        limited = client.post("/v1/query", json={"question": "q"}, headers=bearer("one more"))
    assert codes == [401] * 10 + [429]
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["retry-after"]) <= 6  # a token back every 6 s
    assert "10 a minute" in limited.json()["detail"]
    assert refusals(metrics.UNAUTHENTICATED) - before[metrics.UNAUTHENTICATED] == 10
    assert refusals(metrics.AUTH_LIMITED) - before[metrics.AUTH_LIMITED] == 2


def test_each_key_has_its_own_bucket(fake_config: RagConfig, keys_file: Path) -> None:
    tokens = write_keys(keys_file, a="ingest", b="ingest")
    before = refusals(metrics.RATE_LIMITED)
    with serve(fake_config, keys_file=keys_file, rate_limit=3) as client:
        codes = [
            client.get("/v1/ingest/x", headers=bearer(tokens["a"])).status_code for _ in range(4)
        ]
        assert client.get("/v1/ingest/x", headers=bearer(tokens["b"])).status_code == 404
        limited = client.get("/v1/ingest/x", headers=bearer(tokens["a"]))
    assert codes == [404, 404, 404, 429]
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) == 20
    assert "this key may make 3 a minute" in limited.json()["detail"]
    assert refusals(metrics.RATE_LIMITED) == before  # no question was refused


def test_without_a_credential_each_address_is_limited(fake_config: RagConfig) -> None:
    with serve(fake_config, rate_limit=2) as client:
        codes = [client.get("/v1/ingest/x").status_code for _ in range(3)]
    assert codes == [404, 404, 429]


def test_the_health_probes_are_never_limited(fake_config: RagConfig, keys_file: Path) -> None:
    # Docker's and Caddy's checks poll them from one address (DEC-23, design review 1).
    write_keys(keys_file, bot="query")
    with serve(fake_config, keys_file=keys_file, rate_limit=1) as client:
        for _ in range(100):
            assert client.get("/v1/health/live").status_code == 200
            assert client.get("/v1/health").status_code != 429
        for n in range(20):  # and failed authentication elsewhere does not touch them
            client.get("/v1/ingest/x", headers=bearer(f"guess-{n}"))
        assert client.get("/v1/health/live").status_code == 200


def test_a_question_refused_for_its_scope_counts_as_a_403(
    fake_config: RagConfig, keys_file: Path
) -> None:
    tokens = write_keys(keys_file, ops="ingest")
    before = refusals(metrics.NO_SCOPE)
    with serve(fake_config, keys_file=keys_file) as client:
        response = client.post("/v1/query", json={"question": "q"}, headers=bearer(tokens["ops"]))
        assert client.get("/v1/documents", headers=bearer(tokens["ops"])).status_code == 403
    assert response.status_code == 403
    assert response.json() == {"detail": "this API key lacks the query scope"}
    assert refusals(metrics.NO_SCOPE) - before == 1  # the question; the document list is none


@pytest.mark.parametrize(("names", "said"), [(["bot"], "1 key."), (["bot", "ops"], "2 keys.")])
def test_reading_the_file_says_how_many_keys_in_number(
    keys_file: Path, caplog: pytest.LogCaptureFixture, names: list[str], said: str
) -> None:
    # Found in S3-8's live run: "1 keys." (the plural slips S2-8 and S3-7 fixed elsewhere).
    caplog.set_level(logging.INFO, logger="rag_qa.api.auth")
    write_keys(keys_file, **dict.fromkeys(names, "query"))
    KeyRing(keys_file, None)
    assert caplog.messages[-1].endswith(f": {said}")
