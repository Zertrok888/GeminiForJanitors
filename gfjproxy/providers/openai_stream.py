"""Streaming for providers with an OpenAI compatible chat completions API."""

import json
from collections.abc import Callable
from typing import Any

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client


class StreamError(Exception):
    """Error reported by the provider in the middle of a stream."""


def post_stream(
    url: str,
    request: dict[str, Any],
    headers: dict[str, str],
    stream_sink: Callable[[str], None],
) -> dict[str, Any]:
    """Streams the response, passing each piece of text to the sink as it comes.
    Returns the whole response shaped like a non-streamed one.

    HTTP errors are raised as httpx2.HTTPStatusError with the body already read."""

    content: list[str] = []
    usage = None

    with http_client.stream(
        "POST",
        url,
        json=request,
        headers=headers,
        timeout=PROCESS_TIMEOUT,
    ) as response:
        if response.is_error:
            response.read()  # Error handling needs the whole body
        response.raise_for_status()

        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            if error := chunk.get("error"):
                raise StreamError(
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
