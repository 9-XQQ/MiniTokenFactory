
from collections.abc import Mapping
from typing import Any

import httpx

from pydantic import ValidationError

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

from app.providers.base import (
    BaseProvider,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseError,
)


JsonObject = dict[str, Any]


class OpenAICompatibleProvider(BaseProvider):
    """
    通过 OpenAI Compatible API 调用上游模型。

    AsyncClient 由外部创建并注入。
    本 Provider 不负责关闭客户端。
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_mapping: Mapping[str, str],
        api_key: str | None = None,
    ):
        if not base_url.strip():
            raise ValueError("base_url cannot be empty")

        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use HTTP or HTTPS")

        normalized_mapping = {
            internal.strip(): upstream.strip()
            for internal, upstream in model_mapping.items()
        }

        if not normalized_mapping or any(
            not internal or not upstream
            for internal, upstream in normalized_mapping.items()
        ):
            raise ValueError("model_mapping cannot be empty")

        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model_mapping = normalized_mapping
        self._api_key = api_key.strip() if api_key else None

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:

        # 1. 当前版本暂不支持 Streaming
        if request.stream:
            raise NotImplementedError(
                "OpenAICompatibleProvider does not support streaming yet"
            )

        # 2. 将内部模型名称映射到上游模型名称
        upstream_model = self._model_mapping.get(request.model)

        if upstream_model is None:
            raise ValueError(
                f"No upstream model mapping for '{request.model}'"
            )

        # 3. 构造上游请求数据
        payload: JsonObject = {
            "model": upstream_model,
            "messages": [
                message.model_dump()
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        # 4. 配置请求 Header
        headers: dict[str, str] = {}

        if self._api_key:
            headers["Authorization"] = (
                f"Bearer {self._api_key}"
            )

        # 5. 发送 HTTP 请求并处理上游异常
        url = f"{self._base_url}/chat/completions"

        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=headers,
            )

            response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Upstream model request timed out"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise ProviderHTTPError(
                exc.response.status_code
            ) from exc

        except httpx.RequestError as exc:
            raise ProviderConnectionError(
                "Unable to connect to upstream model service"
            ) from exc

        # 6. 解析上游 JSON
        try:
            data = response.json()

        except ValueError as exc:
            raise ProviderResponseError(
                "Upstream returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise ProviderResponseError(
                "Upstream response must be a JSON object"
            )

        # 7. 对外保留 MiniTokenFactory 的模型别名
        data["model"] = request.model

        # 8. 验证并转换成统一响应结构
        try:
            result = ChatCompletionResponse.model_validate(data)

        except ValidationError as exc:
            raise ProviderResponseError(
                "Upstream response does not match the expected schema"
            ) from exc

        return result
