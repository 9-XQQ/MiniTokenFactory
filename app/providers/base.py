
from abc import ABC, abstractmethod

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class BaseProvider(ABC):
    """
    所有模型 Provider 的基础接口。

    不同的 Provider 必须实现统一的调用方法。
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
