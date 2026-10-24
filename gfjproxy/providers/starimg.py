from collections.abc import Callable
from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID
from .openai_stream import StreamError, post_stream

STARIMG_URL = "https://ai.starimg.ru/v1/chat/completions"


def starimg_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around AI STAR (ai.starimg.ru), an OpenAI compatible API reseller.

    User paramater is only used for logging. A callable "stream" setting gets
    pieces of the response text as they are generated."""

    if len(messages) == 1 and messages[0].role == "assistant":
        # Claude requires the conversation to start with a user message
        messages.insert(0, JaiMessage(content=".", role="user"))

    starimg_request = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "content": message.content,
                "role": message.role,
            }
            for message in messages
        ],
    }

    stream_sink: Callable[[str], None] | None = None

    for key, value in (settings or {}).items():
        if key == "stream" and callable(value):
            stream_sink = value
            starimg_request["stream"] = True
            starimg_request["stream_options"] = {"include_usage": True}
        elif key == "temperature":
            starimg_request["temperature"] = value
        elif key == "max_tokens":
            starimg_request["max_tokens"] = value
        elif key == "top_p":
            starimg_request["top_p"] = value
        elif key == "frequency_penalty":
            starimg_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            starimg_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        if stream_sink is not None:
            starimg_result = post_stream(
                STARIMG_URL, starimg_request, headers, stream_sink
            )
        else:
            starimg_response = http_client.post(
                STARIMG_URL,
                json=starimg_request,
                headers=headers,
                timeout=PROCESS_TIMEOUT,
            )
            starimg_response.raise_for_status()
            starimg_result = starimg_response.json()
    except httpx2.TimeoutException:
        track_stats("starimg.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from AI STAR"

        try:
            error = e.response.json().get("error")
        except ValueError:
            error = None

        if isinstance(error, dict):
            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
        elif e.response.text:
            message += f": {e.response.text}"

        if e.response.is_client_error:
            track_stats("starimg.failed.client")
        elif e.response.is_server_error:
            track_stats("starimg.failed.server")
        else:
            track_stats("starimg.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except StreamError as e:
        track_stats("starimg.failed.server")
        return JaiResult(502, f"Error from AI STAR: {e}")
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("starimg.failed.exception")
        return JaiResult(502, "Unhanded exception from AI STAR.")

    try:
        text = str(starimg_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := starimg_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        # Rejection?
        xlog(user, f"No result text: {starimg_result!r}")
        track_stats("starimg.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("starimg.succeeded")
    return JaiResult(200, text, metadata=metadata)
