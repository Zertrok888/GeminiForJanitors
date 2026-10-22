import json
import secrets
import threading
import time

import pytest
from flask import Flask
from pytest_mock import MockerFixture

from gfjproxy.routes import proxy as proxy_route
from gfjproxy.utils import ResponseHelper

################################################################################


def make_request(text: str, stream: bool = False) -> dict:
    return {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": text},
        ],
        "stream": stream,
    }


@pytest.fixture
def app(mocker: MockerFixture):
    mocker.patch.object(proxy_route, "get_cooldown", return_value=0)
    mocker.patch.object(proxy_route, "WAIT_POLL_INTERVAL", 0.05)
    # Not create_app, it would monkey-patch everything with gevent
    app = Flask(__name__)
    app.register_blueprint(proxy_route.proxy)
    return app


@pytest.fixture
def api_key() -> str:
    return f"test-{secrets.token_hex(8)}"


def mock_slow_handler(mocker: MockerFixture, delay: float, text: str = "Hello"):
    def handler(user, jai_req, response, stream_sink=None):
        time.sleep(delay)
        return response.add_message(text)

    return mocker.patch.object(proxy_route, "handle_chat_message", side_effect=handler)


def post(app, api_key: str, body: dict, results: list | None = None):
    response = app.test_client().post(
        "/", json=body, headers={"Authorization": f"Bearer {api_key}"}
    )
    if results is not None:
        results.append(response)
    return response


def post_in_background(app, api_key: str, body: dict) -> tuple[threading.Thread, list]:
    results = []
    thread = threading.Thread(target=post, args=(app, api_key, body, results))
    thread.start()
    time.sleep(0.2)  # Make sure the request is being processed
    return thread, results


################################################################################


def test_repeated_request_gets_original_result(app, api_key, mocker: MockerFixture):
    """A repeated request waits for the original one instead of failing."""

    handler = mock_slow_handler(mocker, 1.0)
    body = make_request("Hi there")

    thread, results = post_in_background(app, api_key, body)
    repeated = post(app, api_key, body)
    thread.join()

    assert repeated.status_code == 200
    assert repeated.get_json()["choices"][0]["message"]["content"] == "Hello"
    assert results[0].get_data() == repeated.get_data()
    assert handler.call_count == 1


def test_different_request_is_still_rejected(app, api_key, mocker: MockerFixture):
    """A different request while another is being processed is concurrent use."""

    mock_slow_handler(mocker, 1.0)

    thread, results = post_in_background(app, api_key, make_request("Hi there"))
    other = post(app, api_key, make_request("Something else"))
    thread.join()

    assert other.status_code == 403
    assert results[0].status_code == 200


def test_request_after_original_is_processed_again(app, api_key, mocker: MockerFixture):
    """Regenerating a finished message must not reuse the old result."""

    handler = mock_slow_handler(mocker, 0.0)
    body = make_request("Hi there")

    assert post(app, api_key, body).status_code == 200
    assert post(app, api_key, body).status_code == 200
    assert handler.call_count == 2


def test_lock_is_released_after_processing(app, api_key, mocker: MockerFixture):
    """The user can send new requests once the previous one has finished."""

    mock_slow_handler(mocker, 0.0)

    assert post(app, api_key, make_request("One")).status_code == 200
    assert post(app, api_key, make_request("Two")).status_code == 200


def test_quick_stream_has_no_keepalive(app, api_key, mocker: MockerFixture):
    """A fast stream response is sent as before, without keepalive chunks."""

    mock_slow_handler(mocker, 0.0)

    response = post(app, api_key, make_request("Hi there", stream=True))
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert ResponseHelper.STREAM_KEEPALIVE not in body
    assert body.endswith("data: [DONE]\n\n")


def test_slow_stream_sends_keepalive(app, api_key, mocker: MockerFixture):
    """A slow stream response sends empty chunks until the result is ready."""

    mocker.patch.object(proxy_route, "KEEPALIVE_DELAY", 0.1)
    mocker.patch.object(proxy_route, "KEEPALIVE_INTERVAL", 0.1)
    mock_slow_handler(mocker, 0.5)

    response = post(app, api_key, make_request("Hi there", stream=True))
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert body.startswith(ResponseHelper.STREAM_KEEPALIVE)
    assert body.endswith("data: [DONE]\n\n")

    chunks = [c for c in body.split("\n\n") if c and c != "data: [DONE]"]
    contents = [
        json.loads(c.removeprefix("data: "))["choices"][0]["delta"]["content"]
        for c in chunks
    ]
    assert "".join(contents) == "Hello"
    assert len(chunks) > 1


def test_slow_stream_error_is_sent_as_text(app, api_key, mocker: MockerFixture):
    """Once a stream has started, errors can only be sent as stream text."""

    mocker.patch.object(proxy_route, "KEEPALIVE_DELAY", 0.1)
    mocker.patch.object(proxy_route, "KEEPALIVE_INTERVAL", 0.1)

    def handler(user, jai_req, response, stream_sink=None):
        time.sleep(0.3)
        return response.add_error("Something broke", 503)

    mocker.patch.object(proxy_route, "handle_chat_message", side_effect=handler)

    response = post(app, api_key, make_request("Hi there", stream=True))
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Something broke" in body
    assert body.endswith("data: [DONE]\n\n")
