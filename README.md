# MiniTokenFactory

**一个面向大模型推理服务的轻量级项目，用于学习和实践 Inference Gateway、模型路由、推理后端接入与性能优化。**

目前项目处于 MVP 初期阶段，首先实现统一 API、Provider 抽象和模型路由，后续逐步接入真实推理引擎。

## 1. 项目目标

围绕 Token Factory 的基本架构，逐步实现一个轻量级推理服务系统。

主要学习和实践方向：

- Inference Gateway：统一接收和处理模型推理请求。
- Model Router：根据模型名称选择对应的推理后端。
- Provider：抽象不同模型服务的调用方式。
- OpenAI Compatible API：提供统一的模型调用接口。
- Inference Optimization：研究推理性能、并发和 GPU 资源利用。

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python 3.12 | 核心开发语言 |
| FastAPI | HTTP API 服务 |
| Pydantic | 请求与响应数据校验 |
| Uvicorn | ASGI 服务运行 |
| HTTPX | 异步 HTTP 请求，后续使用 |
| Pytest | 自动化测试 |
| PyTorch | 后续 GPU 与推理实验 |
| CUDA | GPU 计算环境 |

## 3. 项目架构

```text
Client
  |
  v
FastAPI Gateway
  |
  v
Model Router
  |
  v
BaseProvider
  |
  +---- MockProvider
  |
  +---- VLLMProvider (planned)
  |
  v
Inference Engine (planned)
```

当前已经实现 Gateway、Model Router 和 MockProvider 的基础调用链。

## 4. 项目目录

```text
MiniTokenFactory/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── providers/
│   │   ├── base.py
│   │   └── mock.py
│   └── services/
│       └── router.py
├── tests/
│   └── test_api.py
├── benchmark/
├── docs/
├── results/
├── .gitignore
└── README.md
```

## 5. 已实现功能

### 5.1 API Gateway

- `GET /health`：健康检查。
- `GET /v1/models`：获取已注册模型列表。
- `POST /v1/chat/completions`：聊天补全请求。

### 5.2 Provider 抽象

通过 Python ABC 定义统一的 Provider 接口：

- BaseProvider
- MockProvider

### 5.3 Model Router

- 模型注册。
- 根据模型名称查找 Provider。
- 调度 Provider 执行模型调用。
- 处理不存在的模型异常。

### 5.4 自动化测试

当前已完成 7 项 API 测试：

- 健康检查。
- 模型列表。
- 正常聊天请求。
- 不存在的模型。
- 空 messages 参数。
- 不支持流式输出。
- Provider 内部异常。

当前测试结果：7 passed。

## 6. 环境配置

开发环境：

- Windows
- WSL2
- Ubuntu 24.04
- Python 3.12
- VS Code

GPU 实验环境：

- NVIDIA GeForce RTX 3050 Laptop GPU
- PyTorch 2.11.0+cu128
- CUDA 12.8

创建 Python 虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装项目依赖：

```bash
python -m pip install fastapi uvicorn httpx pydantic pytest
```

## 7. 启动服务

在项目根目录执行：

```bash
python -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload
```

访问 API 文档：

http://127.0.0.1:8000/docs

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

查询模型：

```bash
curl http://127.0.0.1:8000/v1/models
```

## 8. 自动化测试

执行全部测试：

```bash
python -m pytest tests -v
```

当前测试结果：

```text
7 passed
```

## 9. 当前功能边界

当前版本为基础 MVP：

- 使用 MockProvider 模拟模型响应。
- 尚未接入真实大模型推理引擎。
- 尚未实现 Streaming。
- Token Usage 暂未接入真实 Tokenizer 统计。
- 尚未实现多模型负载均衡和并发调度。

## 10. 后续开发计划

| 后续 | Continuous Batching、KV Cache、GPU 优化 |
| 后续 | MLIR、算子优化与 AI 编译器基础 |

---

**MiniTokenFactory — Learning and Building LLM Inference Infrastructure.**