import json
from collections.abc import Callable
from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def _simplify_error_message(message: str) -> str:
    if not message:
        return "[empty error message]"

    # Nvidia sometimes seems to send duplicated error messages in the same string
    # I.e. the first half of the string is equal to the second half, separated by a space
    half_len = len(message) // 2
    if message[half_len] == " " and message[:half_len] == message[half_len + 1 :]:
        return message[:half_len]

    # No simplifications done
    return message


NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class _StreamError(Exception):
    """Error reported by Nvidia NIM in the middle of a stream."""


def _post_stream(
    nvidia_request: dict[str, Any],
    headers: dict[str, str],
    stream_sink: Callable[[str], None],
) -> dict[str, Any]:
    """Streams the response, passing each piece of text to the sink as it comes.
    Returns the whole response shaped like a non-streamed one."""

    content: list[str] = []
    usage = None

    with http_client.stream(
        "POST",
        NVIDIA_URL,
        json=nvidia_request,
        headers=headers,
        timeout=PROCESS_TIMEOUT,
    ) as nvidia_response:
        if nvidia_response.is_error:
            nvidia_response.read()  # Error handling needs the whole body
        nvidia_response.raise_for_status()

        for line in nvidia_response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            if error := chunk.get("error"):
                raise _StreamError(
                    error.get("message", repr(error))
                    if isinstance(error, dict)
                    else str(error)
                )
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                # Reasoning comes as "reasoning_content" and is left out
                if text := (choice.get("delta") or {}).get("content"):
                    content.append(text)
                    stream_sink(text)

    return {
        "choices": [{"message": {"content": "".join(content)}}],
        "usage": usage,
    }


def nvidia_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Nvidia NIM API.

    User paramater is only used for logging. A callable "stream" setting gets
    pieces of the response text as they are generated."""

    if len(messages) == 1 and messages[0].role == "assistant":
        # For some models, "conversation roles must alternate user/assistant/user/assistant/"
        messages.insert(0, JaiMessage(content=".", role="user"))

    nvidia_request = {
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
            nvidia_request["stream"] = True
            nvidia_request["stream_options"] = {"include_usage": True}
        elif key == "temperature":
            nvidia_request["temperature"] = value
        elif key == "max_tokens":
            nvidia_request["max_tokens"] = value
        elif key == "top_k":
            nvidia_request["top_k"] = value
        elif key == "top_p":
            nvidia_request["top_p"] = value
        elif key == "frequency_penalty":
            nvidia_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            nvidia_request["presence_penalty"] = value
        elif key == "thinking":
            # Reasoning models like GLM toggle thinking through their chat template
            nvidia_request["chat_template_kwargs"] = {"enable_thinking": bool(value)}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        if stream_sink is not None:
            nvidia_result = _post_stream(nvidia_request, headers, stream_sink)
        else:
            nvidia_response = http_client.post(
                NVIDIA_URL,
                json=nvidia_request,
                headers=headers,
                timeout=PROCESS_TIMEOUT,
            )
            nvidia_response.raise_for_status()
            nvidia_result = nvidia_response.json()
    except httpx2.TimeoutException:
        track_stats("nvidia.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Nvidia NIM"
        extras = ""

        content_type: str | None
        if content_type := e.response.headers.get("content-type"):
            if content_type.startswith("text/plain"):
                message += f": {e.response.text}"
            elif content_type.startswith("application/json"):
                if isinstance((error := e.response.json().get("error")), dict):
                    if error_code := error.get("code"):
                        message += f" ({error_code})"
                    if error_message := error.get("message"):
                        message += f": {_simplify_error_message(error_message)}"
            elif content_type.startswith("application/problem+json"):
                response_json = e.response.json()

                if title := response_json.get("title"):
                    message += f" ({title})"
                if detail := response_json.get("detail"):
                    message += f": {detail}"
            else:
                xlog(user, f"{message}: {e.response.text!r}")

        if e.response.is_client_error:
            track_stats("nvidia.failed.client")
        elif e.response.is_server_error:
            track_stats("nvidia.failed.server")
        else:
            track_stats("nvidia.failed.unknown")

        return JaiResult(e.response.status_code, message, extras=extras)
    except _StreamError as e:
        track_stats("nvidia.failed.server")
        return JaiResult(
            502, f"Error from Nvidia NIM: {_simplify_error_message(str(e))}"
        )
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("nvidia.failed.exception")
        return JaiResult(502, "Unhanded exception from Nvidia NIM.")

    try:
        text = str(nvidia_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := nvidia_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        # Rejection?
        xlog(user, f"No result text: {nvidia_result!r}")
        track_stats("nvidia.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("nvidia.succeeded")
    return JaiResult(200, text, metadata=metadata)
