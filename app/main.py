
from fastapi import FastAPI

from app.api.routes import router as api_router


# 创建 FastAPI 应用
app = FastAPI(
    title="MiniTokenFactory",
    description="A lightweight inference gateway",
    version="0.1.0",
)


# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "MiniTokenFactory"
    }


# 注册 API 路由
app.include_router(
    api_router,
    prefix="/v1",
    tags=["Inference"]
)
