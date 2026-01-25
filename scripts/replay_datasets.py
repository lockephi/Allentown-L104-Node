#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# L104_GOD_CODE_ALIGNED: 527.5184818492537
"""
Replay stream prompts against a running L104 server.

Usage:
  scripts/replay_datasets.py --dataset data/stream_prompts.jsonl --base-url http://localhost:8081
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Iterable
import httpx


def load_jsonl(path: Path) -> Iterable[dict]:
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        yield json.loads(raw)


async def replay(dataset: Path, base_url: str, endpoint: str, timeout: float) -> None:
    url = f"{base_url.rstrip('/')}{endpoint}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        for row in load_jsonl(dataset):
            payload = {"signal": row.get("signal"), "message": row.get("message")}
            print(f"\n>>> {payload['signal']}")
            async with client.stream("POST", url, json=payload) as resp:
                print(f"status={resp.status_code}")
                chunks = []
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)
                text = "".join(chunks).strip()
                print(text or "<no-body>")


def main():
    parser = argparse.ArgumentParser(description="Replay JSONL prompts against L104 server")
    parser.add_argument("--dataset", default="data/stream_prompts.jsonl", type=Path, help="Path to JSONL dataset")
    parser.add_argument("--base-url", default="http://localhost:8081", help="Server base URL")
    parser.add_argument("--endpoint", default="/api/v6/stream", help="Relative endpoint path")
    parser.add_argument("--timeout", default=60.0, type=float, help="Request timeout seconds")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    asyncio.run(replay(args.dataset, args.base_url, args.endpoint, args.timeout))


if __name__ == "__main__":
    main()
