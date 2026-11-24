
from abc import ABC, abstractmethod

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class ProviderError(Exception):
    """Provider 调用过程中的基础异常。"""
    pass


class ProviderTimeoutError(ProviderError):
    """上游模型服务请求超时。"""
    pass


class ProviderConnectionError(ProviderError):
    """无法连接上游模型服务。"""
    pass


class ProviderHTTPError(ProviderError):
    """上游返回 HTTP 错误状态码。"""

    def __init__(self, status_code: int):
        self.status_code = status_code

        super().__init__(
            f"Upstream returned HTTP {status_code}"
        )


class ProviderResponseError(ProviderError):
    """上游返回的数据不符合预期。"""
    pass


class BaseProvider(ABC):
    """
    所有模型 Provider 的基础接口。
    """

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        执行一次聊天模型调用。

        Args:
            request: 标准化的聊天请求

        Returns:
            标准化的聊天响应
        """
        raise NotImplementedError
