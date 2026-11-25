
import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter

import httpx


def percentile(values: list[float], p: float) -> float:
    """最近秩百分位数。"""
    values = sorted(values)
    index = max(0, math.ceil(len(values) * p) - 1)
    return values[index]


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    mode: str,
) -> dict:
    payload = {
        "model": "mock-chat",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "stream": mode == "sse",
    }

    async with semaphore:
        started = time.perf_counter()
        status = 0
        completed = False
        first_content_ms = None

        try:
            if mode == "json":
                response = await client.post(
                    url,
                    json=payload,
                )
                status = response.status_code

                if status == 200:
                    data = response.json()
                    completed = (
                        data.get("object") == "chat.completion"
                    )

            else:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                ) as response:
                    status = response.status_code

                    if status == 200:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue

                            data = line[6:]

                            if data == "[DONE]":
                                completed = True
                                break

                            chunk = json.loads(data)

                            choices = chunk.get("choices", [])
                            if choices:
                                content = choices[0].get(
                                    "delta", {}
                                ).get("content")

                                if (
                                    content
                                    and first_content_ms is None
                                ):
                                    first_content_ms = (
                                        time.perf_counter()
                                        - started
                                    ) * 1000

        except (httpx.HTTPError, ValueError, KeyError):
            completed = False

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        return {
            "status": status,
            "ok": status == 200 and completed,
            "latency_ms": latency_ms,
            "first_content_ms": first_content_ms,
        }


async def benchmark(args):
    semaphore = asyncio.Semaphore(args.concurrency)

    timeout = httpx.Timeout(30.0, connect=5.0)

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max(20, args.concurrency),
        ),
    ) as client:

        started = time.perf_counter()

        results = await asyncio.gather(
            *(
                send_request(
                    client,
                    semaphore,
                    args.url,
                    args.mode,
                )
                for _ in range(args.requests)
            )
        )

        duration = time.perf_counter() - started

    successful = [
        result for result in results if result["ok"]
    ]

    latencies = [
        result["latency_ms"] for result in successful
    ]

    first_content = [
        result["first_content_ms"]
        for result in successful
        if result["first_content_ms"] is not None
    ]

    print("\n=== MiniTokenFactory Benchmark ===")
    print(f"模式: {args.mode}")
    print(f"请求数: {args.requests}")
    print(f"客户端并发数: {args.concurrency}")
    print(f"总耗时: {duration:.3f} s")
    print(f"完成吞吐: {len(results) / duration:.2f} RPS")
    print(f"成功数: {len(successful)}")
    print(
        f"成功率: "
        f"{len(successful) / len(results) * 100:.2f}%"
    )

    status_counts = Counter(
        result["status"] for result in results
    )
    print(f"HTTP 状态分布: {dict(status_counts)}")

    if latencies:
        print(
            f"P50: {statistics.median(latencies):.2f} ms"
        )
        print(
            f"P95: {percentile(latencies, 0.95):.2f} ms"
        )

    if first_content:
        print(
            "首个内容片段 P50: "
            f"{statistics.median(first_content):.2f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--mode",
        choices=["json", "sse"],
        default="json",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error(
            "requests and concurrency must be >= 1"
        )

    asyncio.run(benchmark(args))
