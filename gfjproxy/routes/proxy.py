from collections.abc import Callable
from hashlib import sha256
from queue import Empty, Queue
from secrets import token_hex
from threading import Thread
from time import monotonic, sleep
from traceback import print_exception
from typing import Any

from flask import Blueprint, Response, abort, request

from .._globals import PROCESS_TIMEOUT
from ..cooldown import get_cooldown
from ..handlers import handle_chat_message, handle_proxy_test
from ..logging import xlog, xlogtime
from ..models import JaiRequest
from ..storage import storage
from ..utils import ResponseHelper, comma_split, is_proxy_test
from ..xuid_secret import xuid_secret
from ..xuiduser import XUID, UserSettings

proxy = Blueprint("proxy", __name__)

StreamSink = Callable[[str], None]

# How long a stream response may stay silent before keepalives are sent, and
# how often they are sent afterwards. Quick errors still get a proper status.
KEEPALIVE_DELAY = 15
KEEPALIVE_INTERVAL = 10

# How often a repeated request checks whether the original one has finished
WAIT_POLL_INTERVAL = 1


@proxy.route("/", methods=["POST"])
@proxy.route("/chat/completions", methods=["POST"])
@proxy.route("/quiet/", methods=["POST"])
@proxy.route("/quiet/chat/completions", methods=["POST"])
def handle():
    assert storage is not None  # Make type checkers happy

    request_json = request.get_json(silent=True)
    if not request_json:  # This should never happen.
        abort(400, "Bad Request. Missing or invalid JSON.")
        return  # Some type checkers don't realize abort exits the function

    request_path = request.path

    jai_req = JaiRequest.parse(request_json)
    jai_req.quiet = "/quiet/" in request_path

    proxy_test = is_proxy_test(request_json)

    response = ResponseHelper(use_stream=jai_req.stream, wrap_errors=proxy_test)
    fallback = ResponseHelper(use_stream=jai_req.stream, wrap_errors=proxy_test)

    # JanitorAI provides the user's API key through HTTP Bearer authentication.
    # Google AI cannot be used without an API key and neither can this proxy.

    request_auth = request.headers.get("authorization", "").split(" ", maxsplit=1)
    if len(request_auth) != 2 or request_auth[0].lower() != "bearer":
        return response.build_error("Unauthorized. API key required.", 401)

    api_keys = comma_split(request_auth[1])
    xuid = XUID(api_keys[0], xuid_secret)

    # JanitorAI gives up on slow responses and sends the very same request again.
    # Such a repeat waits for the original request and gets its result instead.

    fingerprint = sha256(request_path.encode() + request.get_data()).hexdigest()

    if not storage.lock(xuid):
        tag = storage.inflight_get(xuid)
        if tag is None or tag.partition(":")[0] != fingerprint:
            xlog(xuid, "User attempted concurrent use")
            return response.build_error(
                "Concurrent use is not allowed. Please wait a moment.", 403
            )

        xlog(xuid, "Repeated request, waiting for the original one")
        return _respond(lambda _: _wait_for_result(xuid, tag, response), fallback)

    user = UserSettings(storage, xuid)

    # Cheap and easy rate limiting

    if (
        (seconds := user.last_seen())
        and (cooldown := get_cooldown())
        and (delay := cooldown - seconds) > 0
    ):
        xlog(user, f"User told to wait {delay} seconds")
        storage.unlock(xuid)
        return response.build_error(f"Please wait {delay} seconds.", 429)

    # Handle user's request

    tag = f"{fingerprint}:{token_hex(8)}"
    storage.inflight_put(xuid, tag)

    def work(stream_sink: StreamSink | None) -> dict:
        try:
            result = _result(
                _process(
                    user,
                    jai_req,
                    response,
                    api_keys,
                    proxy_test,
                    request_path,
                    stream_sink,
                )
            )
            storage.result_put(tag, result)
            return result
        finally:
            storage.inflight_rem(xuid)
            storage.unlock(xuid)

    return _respond(work, fallback)


def _process(
    user: UserSettings,
    jai_req: JaiRequest,
    response: ResponseHelper,
    api_keys: list[str],
    proxy_test: bool,
    request_path: str,
    stream_sink: StreamSink | None,
) -> ResponseHelper:
    assert storage is not None  # Make type checkers happy

    rcounter = user.get_rcounter()
    api_key_index = rcounter % len(api_keys)
    user.inc_rcounter()

    jai_req.api_key = api_keys[api_key_index]
    jai_req.api_key_index = api_key_index
    jai_req.api_key_count = len(api_keys)

    log_details = [
        f"User {user.last_seen_msg()}",
        f"Request #{user.get_rcounter()}",
    ]

    if len(api_keys) > 1:
        log_details.append(f"Key {api_key_index + 1}/{len(api_keys)}")

    ref_time = xlogtime(
        user,
        f"Processing {'stream ' if jai_req.stream else ''}{request_path} ({', '.join(log_details)})",
    )

    try:
        if not jai_req.models:
            response.add_error("Please specify a model.", 400)
        if unknown := jai_req.models.get("unknown"):
            response.add_error(
                (
                    f"Unknown model(s): {unknown}\n"
                    + "Make sure to use OpenRouter syntax `provider/model`.\n"
                    + "Examples: `google/gemini-2.5-flash`, `cerebras/llama3.1-8b`, `deepseek/deepseek-chat`, etc."
                ),
                400,
            )
        elif proxy_test:
            response = handle_proxy_test(user, jai_req, response)
        else:
            response = handle_chat_message(user, jai_req, response, stream_sink)
    except Exception as e:  # ruff: ignore[BLE001]
        response.add_error("Internal Proxy Error", 500)
        print_exception(e)

    if 200 <= response.status <= 299:
        xlogtime(user, "Processing succeeded", ref_time)

        if not proxy_test and (announcement := storage.announcement):
            response.add_proxy_message(f"***\n{announcement}\n***")
    else:
        messages = response.message.split("\n")
        xlogtime(user, f"Processing failed: {messages[0]}", ref_time)
        for message in messages[1:]:
            xlog(user, f"> {message}")

    if user.valid:
        user.save()
    else:
        xlog(user, "Invalid user not saved")

    return response


def _wait_for_result(xuid: XUID, tag: str, response: ResponseHelper) -> dict:
    """Waits for the original request with the given tag and returns its result."""
    assert storage is not None  # Make type checkers happy

    deadline = monotonic() + PROCESS_TIMEOUT + 10

    while monotonic() < deadline:
        if (result := storage.result_get(tag)) is not None:
            xlog(xuid, "Repeated request got the original result")
            return result
        if storage.inflight_get(xuid) != tag:
            # The original request may have finished right after the check above
            if (result := storage.result_get(tag)) is not None:
                xlog(xuid, "Repeated request got the original result")
                return result
            break
        sleep(WAIT_POLL_INTERVAL)

    xlog(xuid, "Repeated request lost the original one")
    return _result(
        response.add_error("The original request was lost. Please try again.", 504)
    )


def _result(response: ResponseHelper) -> dict:
    """Turns a response into plain data that can be shared between requests."""
    built = response.build()
    return {
        "status": built.status_code,
        "content_type": built.content_type,
        "body": built.get_data(as_text=True),
        "stream_body": response.stream_body(),
        "message": response.message,
    }


def _respond(
    work: Callable[[StreamSink | None], dict], fallback: ResponseHelper
) -> Response:
    """Runs the work in the background, so it finishes even if the client goes
    away. A stream response gets the text as the model writes it, if the work
    passes any to the sink, and is kept alive while the work takes long."""

    # Events are ("text", str) for pieces of text and ("done", dict) at the end
    events: Queue[tuple[str, Any]] = Queue()

    def stream_sink(text: str):
        events.put(("text", text))

    def run():
        try:
            result = work(stream_sink if fallback.use_stream else None)
        except Exception as e:  # ruff: ignore[BLE001]
            print_exception(e)
            result = _result(fallback.add_error("Internal Proxy Error", 500))
        events.put(("done", result))

    Thread(target=run, daemon=True).start()

    try:
        event = events.get(timeout=KEEPALIVE_DELAY if fallback.use_stream else None)
    except Empty:
        event = None

    if event is not None and event[0] == "done":
        result = event[1]
        return Response(
            response=[result["body"]],
            status=result["status"],
            content_type=result["content_type"],
        )

    # From here on the status is 200 no matter what, errors become stream text

    def generate(event: tuple[str, Any] | None):
        sent = ""
        while True:
            if event is None:
                yield ResponseHelper.STREAM_KEEPALIVE
            elif event[0] == "text":
                sent += event[1]
                yield ResponseHelper.stream_chunk(event[1])
            else:
                yield ResponseHelper.stream_end(_unsent(event[1]["message"], sent))
                return
            try:
                event = events.get(timeout=KEEPALIVE_INTERVAL)
            except Empty:
                event = None

    return Response(
        response=generate(event),
        status=200,
        content_type="text/event-stream; charset=utf-8",
    )


def _unsent(message: str, sent: str) -> str:
    """Returns the part of the final message that has not been streamed yet."""
    if not sent:
        return message
    # The final message is stripped, so whitespace at the end may be missing
    sent = sent.rstrip()
    if message.startswith(sent):
        return message[len(sent) :]
    # Should not happen, but never leave the user with half a message
    return f"\n\n{message}"
