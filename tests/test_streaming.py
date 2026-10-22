import json
import secrets
import time

import httpx2
import pytest
from flask import Flask
from pytest_mock import MockerFixture

from gfjproxy import handlers
from gfjproxy.handlers import handle_chat_message
from gfjproxy.models import JaiMessage, JaiRequest, JaiResult
from gfjproxy.providers import nvidia
from gfjproxy.routes import proxy as proxy_route
from gfjproxy.utils import ResponseHelper
from gfjproxy.xuiduser import XUID, LocalUserStorage, UserSettings

################################################################################


def sse(*chunks: dict | str) -> bytes:
    lines = []
    for chunk in chunks:
        data = chunk if isinstance(chunk, str) else json.dumps(chunk)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


def delta(content: str | None = None, reasoning: str | None = None) -> dict:
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": d}]}


def use_transport(mocker: MockerFixture, handler) -> list[httpx2.Request]:
    requests = []

    def record(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    client = httpx2.Client(transport=httpx2.MockTransport(record))
    mocker.patch.object(nvidia, "http_client", client)
    return requests


def read_stream(body: str) -> tuple[list[str], str]:
    """Returns the content of every chunk and the whole streamed text."""
    contents = []
    for chunk in body.split("\n\n"):
        if not chunk or chunk == "data: [DONE]":
            continue
        contents.append(
            json.loads(chunk.removeprefix("data: "))["choices"][0]["delta"]["content"]
        )
    return contents, "".join(contents)


################################################################################

# Nvidia provider


def test_nvidia_stream_passes_text_to_sink(mocker: MockerFixture):
    body = sse(
        delta(reasoning="Let me think"),
        delta(content="Hello"),
        delta(content=", world"),
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        "[DONE]",
    )
    requests = use_transport(mocker, lambda _: httpx2.Response(200, content=body))

    pieces = []
    result = nvidia.nvidia_generate_content(
        XUID("john", "smith"),
        "nvapi-key",
        "z-ai/glm-5.3",
        [JaiMessage(content="Hi", role="user")],
        {"stream": pieces.append},
    )

    assert result.status == 200
    assert result.text == "Hello, world"
    assert pieces == ["Hello", ", world"]  # Reasoning is not streamed
    assert result.metadata.token_usage.prompt_tokens == 5

    sent = json.loads(requests[0].content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


def test_nvidia_without_sink_does_not_stream(mocker: MockerFixture):
    reply = {"choices": [{"message": {"content": "Hello"}}]}
    requests = use_transport(mocker, lambda _: httpx2.Response(200, json=reply))

    result = nvidia.nvidia_generate_content(
        XUID("john", "smith"),
        "nvapi-key",
        "z-ai/glm-5.3",
        [JaiMessage(content="Hi", role="user")],
        {},
    )

    assert result.text == "Hello"
    assert json.loads(requests[0].content)["stream"] is False


def test_nvidia_stream_http_error(mocker: MockerFixture):
    error = {"error": {"code": 429, "message": "Too many requests"}}
    use_transport(mocker, lambda _: httpx2.Response(429, json=error))

    result = nvidia.nvidia_generate_content(
        XUID("john", "smith"),
        "nvapi-key",
        "z-ai/glm-5.3",
        [JaiMessage(content="Hi", role="user")],
        {"stream": lambda _: None},
    )

    assert result.status == 429
    assert result.error == "Error from Nvidia NIM (429): Too many requests"


def test_nvidia_stream_error_midway(mocker: MockerFixture):
    body = sse(delta(content="Hel"), {"error": {"message": "Model crashed"}})
    use_transport(mocker, lambda _: httpx2.Response(200, content=body))

    result = nvidia.nvidia_generate_content(
        XUID("john", "smith"),
        "nvapi-key",
        "z-ai/glm-5.3",
        [JaiMessage(content="Hi", role="user")],
        {"stream": lambda _: None},
    )

    assert result.status == 502
    assert result.error == "Error from Nvidia NIM: Model crashed"


@pytest.mark.parametrize("thinking", [False, True])
def test_nvidia_thinking_setting(mocker: MockerFixture, thinking: bool):
    reply = {"choices": [{"message": {"content": "Hello"}}]}
    requests = use_transport(mocker, lambda _: httpx2.Response(200, json=reply))

    nvidia.nvidia_generate_content(
        XUID("john", "smith"),
        "nvapi-key",
        "z-ai/glm-5.3",
        [JaiMessage(content="Hi", role="user")],
        {"thinking": thinking},
    )

    sent = json.loads(requests[0].content)
    assert sent["chat_template_kwargs"] == {"enable_thinking": thinking}


################################################################################

# Chat message handler


def run_handler(
    mocker: MockerFixture,
    last_message: str,
    stream_sink=None,
    text: str = " Hi!",
    quiet_commands: bool = True,
) -> tuple[ResponseHelper, dict]:
    calls = []

    def fake_nvidia(user, api_key, model, messages, settings=None):
        calls.append(dict(settings or {}))
        if sink := (settings or {}).get("stream"):
            sink(text)
        return JaiResult(200, text)

    mocker.patch.dict(handlers.PROVIDER_FUNCS, {"nvidia": fake_nvidia})

    user = UserSettings(LocalUserStorage(), XUID("john", "smith"))
    user.do_show_banner(handlers.BANNER_VERSION)
    jai_req = JaiRequest(
        api_key="nvapi-key",
        models={"nvidia": "z-ai/glm-5.3"},
        messages=[
            JaiMessage(content="You are a bot.", role="system"),
            JaiMessage.parse({"content": last_message, "role": "user"}),
        ],
    )
    jai_req.quiet_commands = quiet_commands

    response = handle_chat_message(user, jai_req, ResponseHelper(), stream_sink)
    return response, calls[0]


def test_handler_streams_stripped_text(mocker: MockerFixture):
    pieces = []
    response, settings = run_handler(mocker, "Hello", pieces.append)

    assert "stream" in settings
    assert pieces == ["Hi!"]  # Leading whitespace is gone, like in the response
    assert response.message == "Hi!"


def test_handler_without_sink_does_not_stream(mocker: MockerFixture):
    _, settings = run_handler(mocker, "Hello")

    assert "stream" not in settings


def test_handler_with_prefill_does_not_stream(mocker: MockerFixture):
    pieces = []
    _, settings = run_handler(mocker, "Hello //prefill this", pieces.append)

    assert "stream" not in settings
    assert pieces == []


def test_handler_nothink(mocker: MockerFixture):
    _, settings = run_handler(mocker, "Hello //nothink this")

    assert settings["thinking"] is False


def test_handler_streams_proxy_messages_first(mocker: MockerFixture):
    """Proxy messages from commands come before the text, so are streamed first."""

    pieces = []
    response, settings = run_handler(
        mocker, "Hello //nothink this", pieces.append, quiet_commands=False
    )

    assert "stream" in settings
    assert "".join(pieces) == response.message
    assert pieces[0].startswith(ResponseHelper.PROXY_TAG_OPEN)


################################################################################

# Proxy route


@pytest.fixture
def app(mocker: MockerFixture):
    mocker.patch.object(proxy_route, "get_cooldown", return_value=0)
    app = Flask(__name__)
    app.register_blueprint(proxy_route.proxy)
    return app


def post_stream(app, handler) -> str:
    proxy_route.handle_chat_message = handler
    response = app.test_client().post(
        "/",
        json={
            "model": "nvidia/z-ai/glm-5.3",
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a bot."},
                {"role": "user", "content": "Hi there"},
            ],
        },
        headers={"Authorization": f"Bearer nvapi-{secrets.token_hex(8)}"},
    )
    assert response.status_code == 200
    return response.get_data(as_text=True)


def test_route_streams_text_as_it_comes(app, mocker: MockerFixture):
    mocker.patch.object(proxy_route, "KEEPALIVE_DELAY", 5)

    def handler(user, jai_req, response, stream_sink=None):
        assert stream_sink is not None
        for piece in ["Once ", "upon ", "a time.\n"]:
            stream_sink(piece)
            time.sleep(0.05)
        response.add_message("Once upon a time.")
        return response.add_proxy_message("Announcement")

    mocker.patch.object(proxy_route, "handle_chat_message", side_effect=handler)
    body = post_stream(app, proxy_route.handle_chat_message)
    contents, text = read_stream(body)

    final = ResponseHelper().add_message("Once upon a time.")
    final.add_proxy_message("Announcement")

    assert contents[:3] == ["Once ", "upon ", "a time.\n"]
    assert text.replace("a time.\n", "a time.", 1) == final.message
    assert body.endswith("data: [DONE]\n\n")


def test_route_appends_error_after_streamed_text(app, mocker: MockerFixture):
    def handler(user, jai_req, response, stream_sink=None):
        stream_sink("Half a sen")
        return response.add_error("Error from Nvidia NIM: Model crashed", 502)

    mocker.patch.object(proxy_route, "handle_chat_message", side_effect=handler)
    _, text = read_stream(post_stream(app, proxy_route.handle_chat_message))

    assert text.startswith("Half a sen\n\n")
    assert "Model crashed" in text


def test_route_keepalive_until_text_arrives(app, mocker: MockerFixture):
    mocker.patch.object(proxy_route, "KEEPALIVE_DELAY", 0.1)
    mocker.patch.object(proxy_route, "KEEPALIVE_INTERVAL", 0.1)

    def handler(user, jai_req, response, stream_sink=None):
        time.sleep(0.35)  # Thinking
        stream_sink("Hello")
        return response.add_message("Hello")

    mocker.patch.object(proxy_route, "handle_chat_message", side_effect=handler)
    contents, text = read_stream(post_stream(app, proxy_route.handle_chat_message))

    assert contents[0] == ""  # Keepalive
    assert "Hello" in contents
    assert text == "Hello"


################################################################################

# Helpers


def test_keepalive_is_an_empty_chunk():
    assert ResponseHelper.STREAM_KEEPALIVE == ResponseHelper.stream_chunk("")


@pytest.mark.parametrize(
    "message, sent, expected",
    [
        ("Hello", "", "Hello"),
        ("Hello world", "Hello", " world"),
        ("Hello", "Hello", ""),
        ("Hello", "Hello \n", ""),
        ("Hello\nproxy", "Hello\n\n", "\nproxy"),
        ("Error 502", "Half", "\n\nError 502"),
    ],
)
def test_unsent(message: str, sent: str, expected: str):
    assert proxy_route._unsent(message, sent) == expected
