
import asyncio
import json
import time
import uuid

from collections.abc import AsyncIterator

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

        if request.stream:
            raise NotImplementedError(
                "Use chat_completion_stream for streaming"
            )

        last_message = request.messages[-1].content

        reply = f"Mock 模型收到：{last_message}"

        return ChatCompletionResponse(
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
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:

        if not request.stream:
            raise ValueError(
                "Streaming request must set stream=True"
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        last_message = request.messages[-1].content
        reply = f"Mock 模型收到：{last_message}"

        def make_event(
            delta: dict,
            finish_reason: str | None = None
        ) -> str:

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason
                    }
                ]
            }

            return (
                "data: "
                + json.dumps(chunk, ensure_ascii=False)
                + "\n\n"
            )

        # 1. 首先发送 assistant 角色信息
        yield make_event({"role": "assistant"})

        # 2. 将回答拆成多个片段逐步发送
        chunk_size = 4

        for i in range(0, len(reply), chunk_size):

            await asyncio.sleep(0.1)

            content = reply[i:i + chunk_size]

            yield make_event({
                "content": content
            })

        # 3. 发送结束标记
        yield make_event(
            delta={},
            finish_reason="stop"
        )

        # 4. 发送 SSE 流结束标记
        yield "data: [DONE]\n\n"
