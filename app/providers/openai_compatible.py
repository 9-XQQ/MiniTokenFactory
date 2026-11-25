import json
from collections.abc import AsyncIterator, Mapping

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


    def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """转发上游 OpenAI Compatible SSE 响应。"""

        # 1. 在创建流之前验证请求
        if not request.stream:
            raise ValueError(
                "Streaming request must set stream=True"
            )

        upstream_model = self._model_mapping.get(request.model)

        if upstream_model is None:
            raise ValueError(
                f"No upstream model mapping for '{request.model}'"
            )

        # 2. 构造上游请求
        payload: JsonObject = {
            "model": upstream_model,
            "messages": [
                message.model_dump()
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        headers: dict[str, str] = {}

        if self._api_key:
            headers["Authorization"] = (
                f"Bearer {self._api_key}"
            )

        url = f"{self._base_url}/chat/completions"

        # 3. 真正执行异步网络读取
        async def generate() -> AsyncIterator[str]:

            try:
                async with self._client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as response:

                    response.raise_for_status()

                    data_lines: list[str] = []

                    async for line in response.aiter_lines():

                        # 收集当前 SSE 事件中的 data 字段
                        if line.startswith("data:"):
                            value = line[5:]

                            if value.startswith(" "):
                                value = value[1:]

                            data_lines.append(value)
                            continue

                        # 当前仅处理 OpenAI 兼容的 data 事件
                        if line != "":
                            continue

                        # 空行代表一个 SSE 事件结束
                        if not data_lines:
                            continue

                        event_data = "\n".join(data_lines)
                        data_lines.clear()

                        # 4. 检查流结束标记
                        if event_data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return

                        # 5. 解析上游模型的增量响应
                        try:
                            chunk = json.loads(event_data)
                        except json.JSONDecodeError as exc:
                            raise ProviderResponseError(
                                "Upstream returned invalid SSE JSON"
                            ) from exc

                        if (
                            not isinstance(chunk, dict)
                            or chunk.get("object")
                            != "chat.completion.chunk"
                            or not isinstance(
                                chunk.get("choices"), list
                            )
                        ):
                            raise ProviderResponseError(
                                "Invalid upstream SSE chunk"
                            )

                        # 对外使用 Gateway 注册的模型别名
                        chunk["model"] = request.model

                        # 6. 重新编码后转发给客户端
                        yield (
                            "data: "
                            + json.dumps(
                                chunk,
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )

                    # 上游提前结束，且没有发送 [DONE]
                    raise ProviderResponseError(
                        "Upstream SSE ended without [DONE]"
                    )

            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    "Upstream streaming request timed out"
                ) from exc

            except httpx.HTTPStatusError as exc:
                raise ProviderHTTPError(
                    exc.response.status_code
                ) from exc

            except httpx.RequestError as exc:
                raise ProviderConnectionError(
                    "Upstream streaming connection failed"
                ) from exc

        return generate()

