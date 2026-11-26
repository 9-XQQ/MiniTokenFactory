
import argparse
import asyncio
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx


PROMPT = "请介绍大语言模型推理中的Prefill和Decode阶段，并说明它们的主要区别。"


def percentile(values: list[float], p: float) -> float:
    """采用线性插值计算百分位数。"""
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0

    position = (len(sorted_values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    request_id: int,
) -> dict:
    """发送一次非流式请求，记录耗时和结果。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "max_tokens": 96,
        "stream": False,
    }

    start = time.perf_counter()

    try:
        response = await client.post(url, json=payload)
        latency = time.perf_counter() - start

        if response.status_code != 200:
            return {
                "request_id": request_id,
                "success": False,
                "status_code": response.status_code,
                "latency_s": latency,
                "error": response.text[:300],
            }

        data = response.json()
        tokens = data.get("usage", {}).get(
            "completion_tokens"
        )

        if not isinstance(tokens, int):
            raise ValueError("Missing completion_tokens")

        return {
            "request_id": request_id,
            "success": True,
            "status_code": 200,
            "latency_s": latency,
            "completion_tokens": tokens,
        }

    except (httpx.RequestError, ValueError) as exc:
        return {
            "request_id": request_id,
            "success": False,
            "latency_s": time.perf_counter() - start,
            "error": str(exc),
        }


async def run_group(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    concurrency: int,
    request_count: int,
) -> dict:
    """限制客户端并发，完成一组测试。"""
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(request_id: int) -> dict:
        async with semaphore:
            return await send_request(
                client, url, model, request_id
            )

    start = time.perf_counter()

    results = await asyncio.gather(
        *(worker(i) for i in range(request_count))
    )

    total_time = time.perf_counter() - start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency_s"] for r in successful]
    output_tokens = sum(
        r["completion_tokens"] for r in successful
    )

    summary = {
        "client_concurrency": concurrency,
        "request_count": request_count,
        "success_count": len(successful),
        "failure_count": len(failed),
        "total_time_s": round(total_time, 4),
        "p50_latency_s": round(
            percentile(latencies, 50), 4
        ),
        "p95_latency_s": round(
            percentile(latencies, 95), 4
        ),
        "mean_latency_s": round(
            statistics.mean(latencies), 4
        ) if latencies else None,
        "qps": round(
            len(successful) / total_time, 4
        ),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(
            output_tokens / total_time, 4
        ),
        "requests": results,
    }

    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--model", default="qwen-model")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--concurrency", type=int, nargs="+",
        default=[1, 2, 4],
    )
    args = parser.parse_args()

    if args.count < 1 or any(c < 1 for c in args.concurrency):
        parser.error("count and concurrency must be >= 1")

    # 客户端超时与 Gateway 上游超时分开配置
    timeout = httpx.Timeout(
        connect=5.0, read=120.0, write=10.0, pool=10.0
    )

    report = {
        "created_at": datetime.now().isoformat(),
        "url": args.url,
        "model": args.model,
        "prompt": PROMPT,
        "temperature": 0.0,
        "max_tokens": 96,
        "gateway_max_concurrent_requests": 2,
        "gateway_rate_per_second": 20.0,
        "gateway_rate_capacity": 100,
        "groups": [],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        print("Warming up model...")
        for i in range(2):
            result = await send_request(
                client, args.url, args.model, -(i + 1)
            )
            if not result["success"]:
                raise RuntimeError(
                    f"Warmup request failed: {result}"
                )

        for concurrency in args.concurrency:
            print(f"\nRunning concurrency={concurrency}...")
            summary = await run_group(
                client, args.url, args.model,
                concurrency, args.count,
            )

            report["groups"].append(summary)
            print(json.dumps(
                {k: v for k, v in summary.items() if k != "requests"},
                ensure_ascii=False, indent=2,
            ))

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    filename = (
        "benchmark_nonstream_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output_path = output_dir / filename
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
