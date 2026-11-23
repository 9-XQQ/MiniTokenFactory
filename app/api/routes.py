
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelListResponse,
)

from app.providers.mock import MockProvider

from app.services.router import (
    ModelRouter,
    ModelNotFoundError,
)


# 创建 API 路由对象
router = APIRouter()

# 创建模型路由器
model_router = ModelRouter()

# 注册 Mock 模型
model_router.register("mock-chat", MockProvider())


# 1. 查询模型列表
@router.get("/models", response_model=ModelListResponse)
async def list_models():

    models = [
        ModelInfo(id=model_name)
        for model_name in model_router.list_models()
    ]

    return ModelListResponse(data=models)


# 2. 聊天补全接口
@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse
)
async def chat_completions(
    request: ChatCompletionRequest
):

    try:
        response = await model_router.chat_completion(request)
        return response

    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except NotImplementedError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
