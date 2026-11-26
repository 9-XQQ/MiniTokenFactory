# MiniTokenFactory GPU 优化与推理性能实验报告

实验日期：2026-11-26。本文保留已有报告的实验数据，并明确来源和结论边界。本次文档整理未重新运行 GPU 实验。

## 证据来源

本报告原有记录包含 Ollama 后端日志摘录、单次 /api/chat 统计、输出质量观察与历史测试结果。当前工作区可直接复核的原始文件主要为 results/ 下的 Gateway Benchmark 结果；完整 GPU 日志、单次 /api/chat 原始 JSON 和历史 Pytest 完整输出尚未发现独立归档。

因此，下述 GPU 数值作为已有实验报告记录保留，不能表述为本次重新采集或独立复验的结果。后续需人工补齐日志、原始响应及配置快照，使他人可以审计；这些记录的局限不影响保留已观察到的异常输出。

## 环境和配置

| 项目 | 已有记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU，4GB；nvidia-smi 显示 4096 MiB |
| 驱动 / nvidia-smi 显示 CUDA 版本 | 576.83 / 12.9 |
| 推理运行时 | Ollama 0.40.18（历史记录，精确版本待配置快照复核） |
| 模型 / 权重量化 | qwen2.5:1.5b，Qwen2.5-1.5B，Q4_K_M |
| Context / 并行序列 | 4096 / 1 |
| 底层计算 | Ollama 调用的 llama.cpp / llama-server CUDA 后端 |
| GPU Offload | 日志记录 29/29 层卸载到 GPU |
| Flash Attention | 已有日志记录启用，三种 KV Cache 对照均为 ON |

29/29 层 GPU Offload 不代表 GPU 始终满载。nvidia-smi 显示的 CUDA 版本也不应当作 Gateway 编译或调用了对应版本 CUDA Kernel 的证据。

已有实验采用的服务环境变量：

```ini
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_CONTEXT_LENGTH=4096"
Environment="OLLAMA_KV_CACHE_TYPE=<f16|q8_0|q4_0>"
```

这是历史配置说明，不是本次执行变更的指令。环境变量仅说明服务设置，实际生效应结合启动参数及后端日志核对。

## KV Cache 分配与质量观察

| KV Cache | Flash Attention | K Cache（MiB） | V Cache（MiB） | KV Buffer 总量（MiB） | 相对 F16 节省 |
|---|---|---:|---:|---:|---:|
| F16 | ON | 56.00 | 56.00 | 112.00 | 基准 |
| Q8_0 | ON | 29.75 | 29.75 | 59.52 | 46.86% |
| Q4_0 | ON | 15.75 | 15.75 | 31.51 | 71.87% |

数值来自已有报告所记后端日志；总量沿用日志记录，不用已四舍五入的 K/V 分项重新相加。节省比例计算为 (112.00 − 对应总量) / 112.00，仅描述该 Context 和单序列条件下 KV Buffer 分配量，不代表整卡总显存按同一比例下降。

已有日志摘录：

```text
--cache-type-k q8_0
--cache-type-v q8_0
--flash-attn on
CUDA0 KV buffer size = 59.52 MiB
K (q8_0): 29.75 MiB, V (q8_0): 29.75 MiB
```

已有报告还记载 F16 与 Q4_0 的对应 cache-type、flash-attn 和 KV Buffer 日志；完整日志归档待补齐。

- F16：历史样本能够生成较长中文回答，未进行正式质量评测。
- Q8_0：恢复该设置后生成 256 tokens 的可读中文，但对 KV Cache 的解释存在概念错误。仅完成初步输出检查，尚无正式质量评测，不能宣称质量无损，也不能视为已验证推荐配置。
- Q4_0：两次样本分别生成 10 和 8 tokens，均明显异常并以 stop 结束。不能作为推荐配置，不能只按显存占用评价为更优。

现有样本不足以证明 Q4_0 异常完全由 KV Cache 量化单独引起。固定随机性、增加 Prompt 并重复实验后才能进一步归因。

## 单次 /api/chat 性能样本

以下为已有报告保存的单次统计。Decode Tokens/s = eval_count / (eval_duration / 1e9)，衡量生成阶段，区别于 Gateway 聚合 Output Tokens/s。

| 指标 | F16 + Flash Attention | Q8_0 + Flash Attention |
|---|---:|---:|
| Prompt tokens | 40 | 40 |
| Output tokens | 256 | 256 |
| Prompt eval | 0.079100 s | 0.114597 s |
| Decode eval | 2.245100 s | 2.318969 s |
| Decode speed | 约 114.02 tokens/s | 约 110.39 tokens/s |
| Load duration | 2.997400 s | 0.001307 s |
| Total duration | 5.325900 s | 2.444276 s |

两组加载状态不同，Q8_0 的 prompt_eval_cached_count=39，F16 没有相同口径缓存字段；每组只有一个样本。因此不能据此判断 Q8_0 稳定提速或回退，也不能证明 Flash Attention 的独立性能收益。

| Q4_0 样本 | Prompt tokens | Output tokens | Prefill | Decode | Total | 输出 |
|---|---:|---:|---:|---:|---:|---|
| 预热请求 | 40 | 10 | 0.056801 s | 0.100731 s | 2.635004 s | 明显异常 |
| 第二次请求 | 40 | 8 | 0.111551 s | 0.169175 s | 0.291309 s | 明显异常 |

因生成长度短且质量异常，不把 Q4_0 的 Decode 速度用于 F16/Q8_0 性能排名。

## Gateway 非流式 Benchmark

原始依据：[benchmark_nonstream_20261126_134121.json](../results/benchmark_nonstream_20261126_134121.json)。每组 10 次请求，max_tokens=96，temperature=0.0。

| 客户端并发 | 成功 / 失败 | 总耗时（s） | P50（s） | P95（s） | 平均延迟（s） | QPS | 输出 Token 总数 | Output Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 / 0 | 8.3311 | 0.8322 | 0.8431 | 0.8331 | 1.2003 | 960 | 115.2313 |
| 2 | 10 / 0 | 8.3492 | 1.6631 | 1.6919 | 1.5859 | 1.1977 | 960 | 114.9804 |
| 4 | 10 / 0 | 8.4584 | 3.3693 | 3.3957 | 2.8800 | 1.1823 | 960 | 113.4968 |

Gateway 最大并发为 2，Ollama OLLAMA_NUM_PARALLEL=1。客户端并发提高时总吞吐未明显增加，延迟增大，结果包含 Gateway 和上游排队，不能解释为 GPU 原生四路并行吞吐，也不能将变化全部归因于 Semaphore。

聚合 Output Tokens/s 是整组输出 Token 总数除以整组墙钟耗时；单次 Decode Tokens/s 只使用模型生成阶段耗时。非流式 End-to-End Latency 不等于 TTFT；SSE Chunk 数不等于 Token 数。详细计时起止、百分位算法及数据来源见 [Benchmark 报告](benchmark_report.md)。该 JSON 未记录 KV Cache 类型，不将其归入特定 KV 精度对照组。

## Streaming 与历史测试证据

已有报告记录真实 Qwen 非流式及流式接入成功。当前本地 tests/ 可见 API、上游 503、未知模型、SSE 缺少 [DONE]、提前关闭上游 Response、并发及限流相关测试。其中 Qwen Gateway 测试使用 MockTransport，测试名称不能单独证明访问了真实模型；真实推理需结合独立运行记录。

已有报告保留的历史回归结果为：

```text
35 passed, 1 warning
```

原报告说明警告与 Starlette TestClient / HTTPX 弃用相关。本次未重跑测试，完整终端输出待归档，不能据此声称当前工作区已重新通过相同数量的测试。

原报告还记录 curl -N --max-time 2 在 2 秒后以退出码 28 结束，已接收 47,233 bytes，没有 [DONE]，随后 /health 返回 200。这只能说明客户端中断及 Gateway 仍可响应，不能证明 Ollama 推理已同步停止。自动化提前停止测试验证显式 aclose 后 Response 关闭，真实断连后的端到端取消仍待专项验证。

## 工程边界与待验证项

GPU 实验成果是外部推理后端的配置与观测，不包含自研 CUDA Kernel 优化、PagedAttention 或自研 Continuous Batching。当前 Gateway 实现详见 [架构文档](architecture.md)。

仍需确认：

- 补齐完整 GPU 日志、单次 /api/chat JSON、模型及服务配置快照、历史测试输出，复核历史软件版本。
- 固定 Prompt、随机性、模型驻留状态与 Prompt Cache，重复 F16/Q8_0 实验，开展正式质量评测。
- 复验 Q4_0 异常并调查原因；当前不推荐使用。
- 单独对照 Flash Attention 开关，不能用现有单次样本宣称稳定收益。
- 在固定 Gateway 与 Ollama 并发配置下增加重复测试，并验证真实断连后的资源回收与停止推理行为。
- 确认本轮 Benchmark 的实际 KV Cache 类型；确认测试与原始结果的长期归档方式。

tests/、results/* 和 *.log 当前受 .gitignore 影响；tests/results 无 Git 跟踪文件，仅 results/.gitkeep 有例外规则，当前未见该占位文件。本次不修改忽略规则。

## 后续复核命令

以下仅提供人工复核入口，未在本次文档整理中执行：

```bash
ollama --version
ollama show qwen2.5:1.5b
ollama ps
nvidia-smi
systemctl show ollama -p Environment --value
journalctl -u ollama --no-pager
python scripts/benchmark_gateway.py --count 10 --concurrency 1 2 4
python scripts/benchmark_streaming.py --count 10 --concurrency 1 2 4
python -m pytest tests -v
```
