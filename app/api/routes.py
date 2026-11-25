
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

from fastapi.responses import StreamingResponse

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
    semaphore = request.app.state.inference_semaphore

    try:

        # 1. 请求速率限制
        rate_limiter = request.app.state.rate_limiter

        if not await rate_limiter.allow():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
            )

        # 2. 后续继续执行原来的流式/非流式分支

        if completion_request.stream:

            # 流式请求先获取并发名额
            await semaphore.acquire()

            stream = None

            try:
                stream = model_router.chat_completion_stream(
                    completion_request
                )

                # 在响应头发送前检查首个事件
                first_event = await anext(stream)

            except StopAsyncIteration as exc:
                try:
                    if stream is not None:
                        await stream.aclose()
                finally:
                    semaphore.release()

                raise ProviderResponseError(
                    "Upstream returned an empty SSE stream"
                ) from exc

            except BaseException:
                # 预检查失败或被取消时释放资源
                try:
                    if stream is not None:
                        await stream.aclose()
                finally:
                    semaphore.release()

                raise

            async def forward_stream():
                try:
                    yield first_event

                    async for event in stream:
                        yield event

                finally:
                    # 整个流结束后才释放并发名额
                    try:
                        await stream.aclose()
                    finally:
                        semaphore.release()

            return StreamingResponse(
                forward_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # 普通请求：执行期间持有并发名额
        async with semaphore:
            response = await model_router.chat_completion(
                completion_request
            )

        return response

    # 注意：此处接回原来已有的 except 块

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
