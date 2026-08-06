"""SDK-shaped fakes for the Anthropic suites.

Not named ``test_*`` so pytest does not collect it, matching
:mod:`tests.fake_provider`. Shared by :mod:`tests.test_provider_anthropic` and
:mod:`tests.test_provider_claude_code`, which drive the same SDK surface through
two providers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from anthropic import types as sdk

from logpose import Message
from logpose.providers.base import CompletionRequest

__all__ = [
    "FakeClient",
    "FakeMessages",
    "FakeStream",
    "drain",
    "sdk_message",
    "simple_request",
    "text_delta",
    "thinking_delta",
]


class FakeStream:
    """Stands in for ``AsyncMessageStreamManager`` / ``AsyncMessageStream``."""

    def __init__(
        self,
        events: Sequence[Any],
        final: sdk.Message | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._events = list(events)
        self._final = final
        self._error = error

    async def __aenter__(self) -> FakeStream:
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    async def get_final_message(self) -> sdk.Message:
        assert self._final is not None
        return self._final


class FakeMessages:
    """Records the kwargs the provider builds and replays a scripted stream."""

    def __init__(
        self,
        events: Sequence[Any],
        final: sdk.Message | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._final = final
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self._events, self._final, error=self._error)


class FakeClient:
    """An ``AsyncAnthropic``-shaped object exposing ``messages.stream`` and auth.

    ``api_key`` / ``auth_token`` mirror the plain attributes the real client
    carries; the scrubbing tests read them the same way the provider does.
    """

    def __init__(
        self,
        events: Sequence[Any] = (),
        final: sdk.Message | None = None,
        *,
        error: BaseException | None = None,
        api_key: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.messages = FakeMessages(events, final, error=error)
        self.api_key = api_key
        self.auth_token = auth_token


def sdk_message(
    content: Sequence[Any],
    *,
    stop_reason: str | None = "end_turn",
    usage: sdk.Usage | None = None,
) -> sdk.Message:
    """Build a realistic final ``Message`` as the SDK would hand it back."""
    return sdk.Message.model_construct(
        id="msg_01FAKE",
        content=list(content),
        model="claude-opus-5",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=usage or sdk.Usage(input_tokens=0, output_tokens=0),
    )


def text_delta(text: str) -> sdk.RawContentBlockDeltaEvent:
    return sdk.RawContentBlockDeltaEvent(
        type="content_block_delta", index=0, delta=sdk.TextDelta(type="text_delta", text=text)
    )


def thinking_delta(text: str) -> sdk.RawContentBlockDeltaEvent:
    return sdk.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=sdk.ThinkingDelta(type="thinking_delta", thinking=text),
    )

async def drain(provider: Any, req: CompletionRequest) -> list[Any]:
    """Collect every event a provider stream yields."""
    return [event async for event in provider.stream(req)]


def simple_request(**kwargs: Any) -> CompletionRequest:
    """Build a CompletionRequest with sane defaults."""
    params: dict[str, Any] = {
        "messages": [Message.user("hi")],
        "model": "claude-opus-5",
        "max_tokens": 1024,
    }
    params.update(kwargs)
    return CompletionRequest(**params)
