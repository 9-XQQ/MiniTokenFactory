
from app.providers.base import BaseProvider
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from collections.abc import AsyncIterator

class ModelNotFoundError(Exception):
    """请求的模型不存在。"""
    pass


class ModelRouter:

    def __init__(self):
        # 模型名称 -> Provider 对象
        self._providers: dict[str, BaseProvider] = {}

    def register(
        self,
        model_name: str,
        provider: BaseProvider
    ) -> None:
        """注册模型及其对应的 Provider。"""

        if not model_name:
            raise ValueError("model_name cannot be empty")

        self._providers[model_name] = provider

    def list_models(self) -> list[str]:
        """返回当前注册的模型列表。"""

        return list(self._providers.keys())

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """根据模型名称选择 Provider 并执行调用。"""

        provider = self._providers.get(request.model)

        if provider is None:
            raise ModelNotFoundError(
                f"Model '{request.model}' not found"
            )

        return await provider.chat_completion(request)


    def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """
        根据模型名称选择 Provider，返回流式迭代器。

        输入：
            ChatCompletionRequest

        输出：
            AsyncIterator[str]
        """

        provider = self._providers.get(request.model)

        if provider is None:
            raise ModelNotFoundError(
                f"Model '{request.model}' not found"
            )

        return provider.chat_completion_stream(request)
