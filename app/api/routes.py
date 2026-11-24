
from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelListResponse,
)

from app.providers.base import (
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseError,
)

from app.services.router import ModelNotFoundError


router = APIRouter()


# 1. 查询模型列表
@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request):

    model_router = request.app.state.model_router

    models = [
        ModelInfo(id=model_name)
        for model_name in model_router.list_models()
    ]

    return ModelListResponse(data=models)


# 2. 聊天补全接口
@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
)
async def chat_completions(
    request: Request,
    completion_request: ChatCompletionRequest,
):

    model_router = request.app.state.model_router

    try:

        response = await model_router.chat_completion(
            completion_request
        )

        return response

    except ModelNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except NotImplementedError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ProviderTimeoutError as exc:

        raise HTTPException(
            status_code=504,
            detail=str(exc),
        ) from exc

    except ProviderConnectionError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except ProviderHTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Upstream returned HTTP {exc.status_code}"
            ),
        ) from exc

    except ProviderResponseError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc
