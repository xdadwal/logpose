"""Z.ai Coding Plan Chat Completions with preserved GLM thinking."""

from typing import Any

from logpose.errors import AuthError, LogposeError
from logpose.providers.base import CompletionRequest
from logpose.providers.openai_compat import OpenAICompatProvider

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.3-flash"
EFFORT_LEVELS = ("low", "high", "max")


class ZaiCodingProvider(OpenAICompatProvider):
    """Use a Coding Plan key without inheriting other providers' credentials."""

    name = "zai"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRED_KIND = "api_key"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ) -> None:
        """Configure Coding Plan access.

        Args:
            model: Explicit GLM model identifier.
            api_key: Coding Plan key; otherwise read ZAI_KEY.
            base_url: Coding Plan API root, independent of OPENAI_BASE_URL.
            reasoning_effort: Supported effort, one of low, high, max.
            **kwargs: Transport and token settings for OpenAICompatProvider.
        """
        if reasoning_effort not in EFFORT_LEVELS:
            raise LogposeError("Z.ai effort must be low, high or max")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env="ZAI_KEY",
            preserve_reasoning=True,
            parse_think_tags=False,
            **kwargs,
        )
        self._extra_body = {
            "thinking": {"type": "enabled", "clear_thinking": False},
            "reasoning_effort": reasoning_effort,
            "tool_stream": True,
            **self._extra_body,
        }

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise AuthError("No Z.ai credential configured. Set ZAI_KEY.")
        return super()._headers()

    def _build_body(self, req: CompletionRequest, model: str) -> dict[str, Any]:
        body = super()._build_body(req, model)
        if body.get("reasoning_effort") not in EFFORT_LEVELS:
            raise LogposeError("Z.ai effort must be low, high or max")
        thinking = body.get("thinking")
        if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
            raise LogposeError("Z.ai thinking must remain enabled")
        return body
