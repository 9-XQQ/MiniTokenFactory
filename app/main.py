
from contextlib import asynccontextmanager

import httpx

from fastapi import FastAPI

from app.api.routes import router as api_router

from app.providers.mock import MockProvider

from app.providers.openai_compatible import (
    OpenAICompatibleProvider,
)

from app.services.router import ModelRouter


def create_app(
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    upstream_base_url: str = "http://127.0.0.1:8001/v1",
    upstream_api_key: str | None = None,
) -> FastAPI:
    """
    创建 MiniTokenFactory 应用。

    upstream_transport:
        可选的 HTTP Transport。
        测试时使用 MockTransport，正常运行时使用默认网络 Transport。

    upstream_base_url:
        上游模型服务地址。

    upstream_api_key:
        上游模型服务 API Key。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        # 1. 配置 HTTP 连接池
        limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        )

        # 2. 配置 HTTP 超时
        timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=10.0,
            pool=5.0,
        )

        # 3. 创建共享 HTTP 客户端
        async with httpx.AsyncClient(
            transport=upstream_transport,
            limits=limits,
            timeout=timeout,
        ) as http_client:

            # 4. 创建模型路由器
            model_router = ModelRouter()

            # 5. 注册 MockProvider
            model_router.register(
                "mock-chat",
                MockProvider(),
            )

            # 6. 创建 OpenAI Compatible Provider
            openai_provider = OpenAICompatibleProvider(
                client=http_client,
                base_url=upstream_base_url,
                model_mapping={
                    "qwen-model": (
                        "Qwen/Qwen2.5-1.5B-Instruct"
                    )
                },
                api_key=upstream_api_key,
            )

            # 7. 注册 OpenAI Compatible Provider
            model_router.register(
                "qwen-model",
                openai_provider,
            )

            # 8. 将应用资源保存到 app.state
            app.state.http_client = http_client
            app.state.model_router = model_router

            # 9. 应用开始处理请求
            yield

        # 10. 退出 async with 后，HTTP 客户端自动关闭

    # 创建 FastAPI 应用
    app = FastAPI(
        title="MiniTokenFactory",
        description="A lightweight inference gateway",
        version="0.2.0",
        lifespan=lifespan,
    )

    # 健康检查
    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "service": "MiniTokenFactory",
        }

    # 注册 API 路由
    app.include_router(
        api_router,
        prefix="/v1",
        tags=["Inference"],
    )

    return app


# 正常运行时使用默认网络 Transport
app = create_app()
