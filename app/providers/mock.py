
import time
import uuid

from app.providers.base import BaseProvider

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage,
)


class MockProvider(BaseProvider):

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:

        # Day 1 暂不支持流式输出
        if request.stream:
            raise NotImplementedError(
                "MockProvider does not support streaming"
            )

        # 1. 获取最后一条用户输入
        last_message = request.messages[-1].content

        # 2. 模拟模型生成的文本
        reply = f"Mock 模型收到：{last_message}"

        # 3. 构造标准响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=reply
                    ),
                    finish_reason="stop"
                )
            ],
            # 未调用真实 tokenizer，因此暂记为 0
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )

        return response
