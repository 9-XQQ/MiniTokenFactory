
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


# ==========================================
# 1. Provider 异常定义
# ==========================================

class ProviderError(Exception):
    """Provider 层异常的基础类。"""
    pass


class ProviderTimeoutError(ProviderError):
    """上游模型请求超时。"""
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
    """上游响应格式异常。"""
    pass


# ==========================================
# 2. Provider 统一接口
# ==========================================

class BaseProvider(ABC):
    """
    所有模型 Provider 的基础接口。

    不同 Provider 必须实现统一的非流式调用方法。
    """

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        执行非流式聊天模型调用。

        Args:
            request: 标准化的聊天请求

        Returns:
            标准化的聊天响应
        """
        raise NotImplementedError

    def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """
        执行流式聊天模型调用。

        Returns:
            异步迭代器，每次产生一个 SSE 事件字符串。
        """

        raise NotImplementedError(
            "This provider does not support streaming"
        )
