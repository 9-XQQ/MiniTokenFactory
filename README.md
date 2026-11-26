# MiniTokenFactory

MiniTokenFactory 是基于 FastAPI 的轻量级 Inference Gateway，面向模型服务接入、异步 HTTP 转发、流式输出、请求控制与性能验证。当前已通过 Ollama 接入 Qwen2.5-1.5B 真实推理，应用代码版本为 `0.2.0`。

项目实现网关和 Provider 适配层，模型计算由 Ollama 及其底层推理后端执行。当前成果不包含自研推理引擎。

## 已实现的工程能力

| 能力 | 当前实现 |
|---|---|
| FastAPI Inference Gateway | 健康检查、模型列表、聊天补全接口 |
| ModelRouter 与 Provider 抽象 | 模型注册、按别名分派；BaseProvider、MockProvider、OpenAICompatibleProvider |
| OpenAI Compatible API | 支持当前 Schema 范围内的 Chat Completions 普通响应与 SSE，不代表完整 API 功能兼容 |
| Ollama 真实推理 | 对外 `qwen-model` 映射到上游 `qwen2.5:1.5b` |
| HTTPX AsyncClient 生命周期 | Lifespan 创建共享客户端和连接池，注入 Provider，应用退出时关闭 |
| SSE Streaming | 上游事件解析、模型别名恢复、首事件预检查、结束标记转发与资源清理 |
| Semaphore 并发控制 | 默认最多 2 个请求同时执行推理/流式转发；流结束后释放名额 |
| Token Bucket Rate Limiting | 进程内请求限流，默认每秒补充 20 个令牌、容量 100，超限返回 429 |
| 自动化测试与 Benchmark | 本地 Pytest 测试及非流式/流式压测脚本；2026-11-26 已有真实推理实验记录 |

实现依据：`app/main.py`、`app/api/`、`app/services/`、`app/providers/`。配置值为当前代码默认值，不表示实测吞吐保证。

## 技术栈和架构

| 技术 | 用途 |
|---|---|
| Python / asyncio | 异步调度、Semaphore、Lock 和异步迭代 |
| FastAPI / Uvicorn | API、ASGI 服务与 Lifespan |
| Pydantic | 请求校验、模型与 JSON 数据转换、非流式响应校验 |
| HTTPX | 共享异步 HTTP 客户端、连接池、超时、上游 JSON/SSE 请求 |
| Ollama + Qwen2.5-1.5B | 实际模型推理，实验权重量化为 Q4_K_M |
| Pytest / HTTPX MockTransport | API、Provider、流式处理、并发及限流测试 |
| NVIDIA GPU / Ollama CUDA 后端 | 底层模型计算及 GPU 配置实验；Gateway 不直接执行 CUDA Kernel |

```text
Client → FastAPI Gateway → ModelRouter → OpenAICompatibleProvider → Ollama → GPU
                                  └→ MockProvider（功能测试）
```

详细说明见 [架构文档](docs/architecture.md)。

## 目录

```text
app/          Gateway、Schema、Router、Provider、限流器
scripts/      benchmark_gateway.py、benchmark_streaming.py
tests/        本地自动化测试（当前被 Git 忽略）
results/      本地原始实验结果（当前内容被 Git 忽略）
experiments/  本地异步/HTTPX 学习实验（当前被 Git 忽略）
docs/         架构、Benchmark、GPU 实验报告
```

## 本地运行与调用

在可访问 Ollama 的 Python 环境中执行；已有开发环境记录为 WSL2 / Ubuntu 24.04 / Python 3.12。仓库目前未提供锁定依赖的安装清单，以下为基础安装方式，不构成精确版本复现保证。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install fastapi uvicorn httpx pydantic pytest
ollama pull qwen2.5:1.5b
# 另行确保 Ollama 服务已启动，默认地址为 http://127.0.0.1:11434
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

接口：`GET /health`、`GET /v1/models`、`POST /v1/chat/completions`；交互文档位于 `http://127.0.0.1:8000/docs`。健康检查仅报告 Gateway 存活，不验证 Ollama 或 GPU 可用性。

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-model","messages":[{"role":"user","content":"你好"}],"max_tokens":96,"stream":false}'
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-model","messages":[{"role":"user","content":"你好"}],"max_tokens":96,"stream":true}'
```

## 测试与实验成果

```bash
python -m pytest tests -v
python scripts/benchmark_gateway.py --count 10 --concurrency 1 2 4
python scripts/benchmark_streaming.py --count 10 --concurrency 1 2 4
```

测试文件覆盖 API、模型别名映射、上游错误、SSE 结束标记、提前关闭 Response、并发上限和令牌桶。已有 GPU 报告记录历史回归结果为 `35 passed, 1 warning`；本次文档整理未重跑测试，该数字不是本次运行结果。使用 MockTransport 的测试验证适配逻辑，真实推理证据需看独立实验记录。

2026-11-26 非流式压测每组 10 次、均成功；客户端并发 1/2/4 的 QPS 分别为 1.2003/1.1977/1.1823，P50 分别为 0.8322/1.6631/3.3693 秒。Gateway 最大并发为 2，Ollama `OLLAMA_NUM_PARALLEL=1`，客户端四并发包含排队，不能解释为 GPU 原生四路并行吞吐。详见 [Benchmark 报告](docs/benchmark_report.md)。

RTX 3050 Laptop GPU（4GB）上已有模型 29/29 层 GPU Offload、Flash Attention 启用记录；KV Cache F16/Q8_0/Q4_0 分配量为 112/59.52/31.51 MiB。Q4_0 出现明显异常输出，不作为推荐配置；Q8_0 仅完成初步输出检查，尚无正式质量评测。详见 [GPU 实验报告](docs/gpu_optimization_report.md)。

## 版本边界

当前未实现自研推理引擎、CUDA Kernel 优化、PagedAttention 或自研 Continuous Batching。Flash Attention、GPU Offload 与 KV Cache 量化属于 Ollama 后端配置验证。Router 尚无动态负载均衡，进程内限流与并发资源也不跨进程共享。

接口仅覆盖当前 Pydantic Schema；Gateway 没有独立 Tokenizer。非流式 usage 使用上游统计，Mock usage 为零占位；SSE Chunk 数不能替代 Token 数。真实客户端断开后 Ollama 是否总能及时停止推理，仍待专项验证。

## Git 忽略与证据可见性

本次核对 `.gitignore` 与 Git 索引：`git ls-files tests results` 无输出，说明两目录当前没有被跟踪文件。

- `tests/`：整个测试目录被忽略，包括现有 `test_api.py`、`test_upstream_streaming.py`、`test_concurrency.py`、限流器测试等。
- `results/*`：现有 Benchmark JSON、day4 JSON、`json_*.txt`、`sse_*.txt` 均被忽略；`!results/.gitkeep` 为例外规则，当前本地未见该占位文件。
- `experiments/`、`*.log`、虚拟环境、缓存和 IDE 文件也被忽略。

因此，文档引用的本地测试和原始结果不会随普通 Git 添加操作进入仓库，新克隆未必能获取这些证据。本次未修改忽略规则、未添加跟踪文件；测试与原始实验记录的后续归档方式需要人工确认。
