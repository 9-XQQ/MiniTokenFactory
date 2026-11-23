
from typing import Literal

from pydantic import BaseModel, Field


# 单条对话消息
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# 用户请求
class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)

    messages: list[ChatMessage] = Field(min_length=1)

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0
    )

    max_tokens: int = Field(
        default=256,
        gt=0
    )

    stream: bool = False


# 模型返回的单个选择
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


# Token 使用统计
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# 模型响应
class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# 模型信息
class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "mini-token-factory"


# 模型列表响应
class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
