
"""
benchmark_streaming.py

MiniTokenFactory SSE Streaming Benchmark

测量：
1. TTFT：首个非空 delta.content 到达时间
2. Total Latency：完整 SSE 流结束时间
3. QPS：单位时间成功完成的请求数量
4. P50 / P95：延迟分布

注意：
SSE Chunk 不等于 Token，因此不将 Chunk 数量当作 Token 数量。
"""

import argparse
import asyncio
import json
import math
import statistics
import time

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx


@dataclass
class RequestResult:
    request_id: int
    success: bool
    status_code: int | None

    ttft_s: float | None
    total_latency_s: float

    content_chunks: int
    output_chars: int

    finish_reason: str | None
    done_received: bool

    error: str | None


async def iter_sse_data(response: httpx.Response):
    """
    解析 SSE Event 中的 data 字段。

    一个 SSE Event 可能由多行组成，
    因此以空行作为事件分隔符。
    """

    data_lines = []

    async for line in response.aiter_lines():

        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []

            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


async def benchmark_one_request(
    client: httpx.AsyncClient,
    request_id: int,
    semaphore: asyncio.Semaphore,
    args,
    max_tokens: int | None = None,
) -> RequestResult:

    async with semaphore:

        # 请求真正开始发送的时间
        start_time = time.perf_counter()

        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt,
                }
            ],
            "temperature": args.temperature,
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else args.max_tokens
            ),
            "stream": True,
        }

        status_code = None
        ttft = None
        content_chunks = 0
        output_chars = 0

        finish_reason = None
        done_received = False
        error = None

        try:

            async with client.stream(
                "POST",
                args.url,
                json=payload,
            ) as response:

                status_code = response.status_code

                if status_code != 200:
                    body = (
                        await response.aread()
                    ).decode("utf-8", errors="replace")

                    raise RuntimeError(
                        f"HTTP {status_code}: {body[:300]}"
                    )

                async for data in iter_sse_data(response):

                    if data == "[DONE]":
                        done_received = True
                        break

                    if not data:
                        continue

                    event = json.loads(data)

                    if "error" in event:
                        raise RuntimeError(
                            str(event["error"])
                        )

                    choices = event.get("choices", [])

                    for choice in choices:

                        delta = choice.get("delta") or {}

                        content = delta.get("content")

                        if isinstance(content, str) and content:

                            current_time = time.perf_counter()

                            # 第一次收到真正的生成内容
                            if ttft is None:
                                ttft = current_time - start_time

                            content_chunks += 1
                            output_chars += len(content)

                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]

            if not done_received:
                error = "SSE stream ended without [DONE]"

            elif ttft is None:
                error = "No non-empty delta.content received"

        except Exception as exc:

            error = f"{type(exc).__name__}: {exc}"

        total_latency = time.perf_counter() - start_time

        success = (
            error is None
            and status_code == 200
            and done_received
            and ttft is not None
        )

        return RequestResult(
            request_id=request_id,
            success=success,
            status_code=status_code,
            ttft_s=(
                round(ttft, 4)
                if ttft is not None
                else None
            ),
            total_latency_s=round(total_latency, 4),
            content_chunks=content_chunks,
            output_chars=output_chars,
            finish_reason=finish_reason,
            done_received=done_received,
            error=error,
        )


def percentile(values, p):

    if not values:
        return None

    ordered = sorted(values)

    index = max(
        0,
        math.ceil(p * len(ordered)) - 1
    )

    return round(ordered[index], 4)


async def run_benchmark_group(
    client,
    args,
    concurrency,
):

    print(f"\nRunning concurrency={concurrency}...")

    semaphore = asyncio.Semaphore(concurrency)

    start_time = time.perf_counter()

    results = await asyncio.gather(
        *[
            benchmark_one_request(
                client,
                i + 1,
                semaphore,
                args,
            )
            for i in range(args.count)
        ]
    )

    total_time = time.perf_counter() - start_time

    successes = [
        result for result in results
        if result.success
    ]

    failures = [
        result for result in results
        if not result.success
    ]

    ttft_values = [
        result.ttft_s
        for result in successes
        if result.ttft_s is not None
    ]

    latency_values = [
        result.total_latency_s
        for result in successes
    ]

    summary = {
        "client_concurrency": concurrency,
        "request_count": args.count,

        "success_count": len(successes),
        "failure_count": len(failures),

        "total_time_s": round(total_time, 4),

        "ttft_p50_s": (
            round(statistics.median(ttft_values), 4)
            if ttft_values else None
        ),

        "ttft_p95_s": percentile(
            ttft_values, 0.95
        ),

        "latency_p50_s": (
            round(statistics.median(latency_values), 4)
            if latency_values else None
        ),

        "latency_p95_s": percentile(
            latency_values, 0.95
        ),

        "mean_ttft_s": (
            round(statistics.mean(ttft_values), 4)
            if ttft_values else None
        ),

        "mean_latency_s": (
            round(statistics.mean(latency_values), 4)
            if latency_values else None
        ),

        "qps": round(
            len(successes) / total_time,
            4
        ),

        "average_content_chunks": (
            round(
                statistics.mean(
                    [r.content_chunks for r in successes]
                ),
                2,
            )
            if successes else 0
        ),

        "average_output_chars": (
            round(
                statistics.mean(
                    [r.output_chars for r in successes]
                ),
                2,
            )
            if successes else 0
        ),

        "requests": [
            asdict(result)
            for result in results
        ],
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    return summary


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )

    parser.add_argument(
        "--model",
        default="qwen-model",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--prompt",
        default=(
            "请用中文解释大型语言模型推理中的 "
            "Prefill 和 Decode，并给出简单例子。"
        ),
    )

    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be >= 1")

    if any(c < 1 for c in args.concurrency):
        parser.error("concurrency must be >= 1")

    limits = httpx.Limits(
        max_connections=max(args.concurrency) * 2,
        max_keepalive_connections=max(args.concurrency) * 2,
    )

    timeout = httpx.Timeout(
        args.timeout,
        connect=5.0,
        pool=10.0,
    )

    report = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "url": args.url,
            "model": args.model,
            "count": args.count,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "methodology": {
            "ttft": "Request start to first non-empty delta.content",
            "total_latency": "Request start to SSE [DONE]",
            "qps": "Successful requests / group wall time",
            "p95": "Nearest-rank percentile",
            "token_count": "Not measured; SSE chunks are not tokens",
        },
        "groups": [],
    }

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
    ) as client:

        # 预热模型，避免冷启动影响正式数据
        print("Warming up model...")

        warmup = await benchmark_one_request(
            client,
            0,
            asyncio.Semaphore(1),
            args,
            max_tokens=min(args.max_tokens, 16),
        )

        if not warmup.success:
            raise RuntimeError(
                f"Warmup failed: {warmup.error}"
            )

        for concurrency in args.concurrency:

            summary = await run_benchmark_group(
                client,
                args,
                concurrency,
            )

            report["groups"].append(summary)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        output_dir
        / f"benchmark_streaming_{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
