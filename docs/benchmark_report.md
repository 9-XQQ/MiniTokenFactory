# MiniTokenFactory Benchmark 报告

实验日期：2026-11-26。本文整理已有记录，本次文档更新未重新压测。

## 数据来源与测试条件

主表逐项取自本地 [benchmark_nonstream_20261126_134121.json](../results/benchmark_nonstream_20261126_134121.json)，created_at 为 2026-11-26T13:40:54.591833。脚本为 [benchmark_gateway.py](../scripts/benchmark_gateway.py)。

| 条件 | 记录值或来源 |
|---|---|
| 接口 | http://127.0.0.1:8000/v1/chat/completions，非流式 |
| 模型 | 对外 qwen-model，上游 qwen2.5:1.5b |
| Prompt | 请介绍大语言模型推理中的Prefill和Decode阶段，并说明它们的主要区别。 |
| temperature / max_tokens | 0.0 / 96 |
| 客户端并发 | 1、2、4，每组 10 次正式请求 |
| 预热 | 当前脚本在正式分组前执行 2 次预热，不计入主表 |
| Gateway 最大并发 | 2；代码默认值及记录中的 gateway_max_concurrent_requests 一致 |
| Gateway 令牌桶 | 每秒补充 20，容量 100；代码与结果记录一致 |
| Ollama 并行序列 | OLLAMA_NUM_PARALLEL=1，来源为已有 GPU 实验报告，JSON 未独立采集该设置 |
| 硬件 / 权重 | RTX 3050 Laptop GPU 4GB / Qwen2.5-1.5B Q4_K_M，来源为已有 GPU 报告 |

Gateway 配置字段由压测脚本写入，并非向运行服务读取的实时配置快照。当前代码与记录一致，但精确复现仍需人工核对实际启动参数。该 JSON 没有记录 KV Cache 类型，不能将主表直接归为 F16、Q8_0 或 Q4_0 性能结果。

## 非流式结果

| 客户端并发 | 请求数 | 成功 / 失败 | 整组耗时（s） | QPS | P50（s） | P95（s） | 输出 Token 总数 | Output Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 / 0 | 8.3311 | 1.2003 | 0.8322 | 0.8431 | 960 | 115.2313 |
| 2 | 10 | 10 / 0 | 8.3492 | 1.1977 | 1.6631 | 1.6919 | 960 | 114.9804 |
| 4 | 10 | 10 / 0 | 8.4584 | 1.1823 | 3.3693 | 3.3957 | 960 | 113.4968 |

主表保留原始汇总值；单请求 latency_s 与 completion_tokens 在原始文件的 groups[].requests 中。P50、P95 仅针对成功请求，由当前非流式脚本使用线性插值计算。仅有每组 10 个样本，P95 不宜外推为稳定的尾延迟保证。

另有同日 [134205 记录](../results/benchmark_nonstream_20261126_134205.json)，属于另一轮实验，未与主表混合或取平均；[134035 记录](../results/benchmark_nonstream_20261126_134035.json) 只有客户端并发 1、请求数 2，不是主表来源。

## 指标口径

| 指标 | 本项目中的含义 | 本次主表是否提供 |
|---|---|---|
| End-to-End Latency | 客户端发送请求到接收完整结果；包括网络、Gateway 和上游排队及推理 | 是，非流式 P50/P95 |
| TTFT | 客户端请求开始到首个非空 delta.content 到达，作为首个生成内容的可观测代理 | 否，非流式无法分离首内容时间 |
| QPS | 成功请求数 / 整组墙钟耗时 | 是，单位 requests/s |
| Output Tokens/s | 成功请求的 completion_tokens 总和 / 整组墙钟耗时 | 是，上游 usage 计数口径 |
| Decode Tokens/s | 单次生成阶段 Token 数 / 该生成阶段耗时，如 Ollama eval_count / (eval_duration / 1e9) | 否，主表未采集生成阶段耗时 |

当前非流式脚本在获取客户端 Semaphore 后开始计单请求时间，以 client.post 返回完整响应时结束；不包含等待客户端 Semaphore 的时间，也不包含后续本地 JSON 解析。Gateway Semaphore 和 Ollama 的等待发生在请求发送后，计入延迟。整组耗时包括客户端调度与等待。因此 End-to-End 指的是这段客户端可观测请求过程，并非每个任务从创建开始的总等待时间。

Output Tokens/s 是整组聚合吞吐，不等于每个请求的 Decode Tokens/s，更不是 GPU Kernel 性能。SSE Chunk、网络字节块、字符数与 Token 数没有一一对应关系；不能根据 Chunk 数推算 Token 吞吐。

## 结果解释

客户端并发从 1 提高到 4 时，QPS 未明显增加，P50/P95 增大。当前 Gateway 最大并发为 2，Ollama NUM_PARALLEL=1，客户端四并发会在 Gateway 和/或上游等待。该结果描述整个请求链路的表现，不能解释为 GPU 原生四路并行吞吐，也不能证明实现了 Continuous Batching。

现有数据不能单独量化 Gateway 开销、Semaphore 等待、Ollama 排队或 GPU 计算各自的占比。需要相同负载下的直连对照与分阶段时间戳才能进一步归因。

## 流式 Benchmark 与自动化验证

[benchmark_streaming.py](../scripts/benchmark_streaming.py) 已实现 TTFT、完整流耗时、QPS、内容 Chunk 数、输出字符数以及 [DONE] 检查。它以首个非空 delta.content 计 TTFT，普通角色事件不算首内容；完整流耗时在收到 [DONE] 后退出流上下文时汇总，包含少量客户端清理开销。该脚本不测 Token 数，不报告 Token/s。其 P95 使用 nearest-rank，P50 使用中位数，与非流式脚本的百分位实现不同。

本地还存在 day4_sse/day4_json 系列 JSON 和 json_*/sse_* 文本，属于其他参数或历史脚本口径；本报告未将其合并到主表，也未将它们冒充当前 streaming 脚本的输出。当前本地未见 benchmark_streaming_*.json，因此这里不发布该脚本的新一轮 TTFT 数字，待验证、归档后补充。

测试文件覆盖普通/流式 API、HTTPX 上游错误、缺少 [DONE]、提前关闭上游 Response、Semaphore 和令牌桶。已有 GPU 报告记载历史回归 `35 passed, 1 warning`；本次未重跑，当前精确通过数量待复验。MockTransport 测试与真实 Ollama 实验应分别看待。

## 复现与待确认项

```bash
python scripts/benchmark_gateway.py --count 10 --concurrency 1 2 4
python scripts/benchmark_streaming.py --count 10 --concurrency 1 2 4
```

复现前需核对模型驻留/预热状态、Ollama 运行版本、KV Cache 类型、Flash Attention、上下文长度、实际并行序列及 Gateway 启动配置。重复实验与正式质量评测尚待完成，本报告不据单轮结果声明稳定性能收益。

本地 tests/ 被整体忽略，results/* 被忽略，只有 results/.gitkeep 有例外规则且当前未见该文件；git ls-files tests results 无输出。上述原始记录链接在当前工作区可读，但普通克隆不能保证获得。本次未修改 .gitignore，需人工决定测试与实验记录的版本控制或外部归档方式。
