#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def api_request(url: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout: int = 300) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[HTTP {e.code}] {url} -> {err_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[Connection Error] {url} -> {e.reason}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /generate endpoint and tokens_per_second")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the worker manager")
    parser.add_argument("--worker", default="worker1", help="Worker ID (e.g., worker1, worker2, worker3, judge)")
    parser.add_argument("--model", default=None, help="Optional GGUF filename to load before testing")
    parser.add_argument("--kv", default=None, help="Optional KV cache quantization override (e.g., q8_0, q4_0, f16)")
    parser.add_argument("--prompt", default="Explain the difference between mutex and semaphore in 5 concise bullet points.", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark iterations")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # 1. Check node health and active workers
    status = api_request(f"{base_url}/health")
    active_workers = status.get("active_workers", {})
    print(f"=== Node: {status.get('node_role')} | GPU Offload: {status.get('gpu_offload_supported')} | Peak RSS: {status.get('process_peak_rss_mb')} MB ===")

    # 2. Load model if requested or if worker is not loaded yet
    if args.model:
        print(f"Loading model '{args.model}' on '{args.worker}'...")
        load_payload: Dict[str, Any] = {
            "worker_id": args.worker,
            "model_name": args.model,
            "n_gpu_layers": -1,
            "n_ctx": 8192,
        }
        if args.kv:
            load_payload["type_k"] = args.kv
            load_payload["type_v"] = args.kv

        load_res = api_request(f"{base_url}/api/v1/system/load_model", method="POST", payload=load_payload, timeout=600)
        print(f"-> {load_res.get('message')}")
        status = api_request(f"{base_url}/health")
        active_workers = status.get("active_workers", {})

    if args.worker not in active_workers:
        print(f"Error: Worker '{args.worker}' has no loaded model. Pass --model <filename.gguf> to load one.", file=sys.stderr)
        sys.exit(1)

    worker_meta = active_workers[args.worker]
    kv_info = worker_meta.get("kv_cache", {})
    print(
        f"Active Model : {worker_meta.get('model_name')} "
        f"(Ctx: {worker_meta.get('n_ctx')}, KV: {kv_info.get('type_k')}/{kv_info.get('type_v')}, "
        f"FA: {worker_meta.get('flash_attn')})"
    )
    print("-" * 75)

    # 3. Run generation benchmark
    gen_payload = {
        "prompt": args.prompt,
        "system_prompt": "You are a precise technical assistant.",
        "max_tokens": args.max_tokens,
        "temperature": args.temp,
    }

    tps_results = []
    for i in range(1, args.runs + 1):
        client_start = time.monotonic()
        res = api_request(f"{base_url}/api/v1/{args.worker}/generate", method="POST", payload=gen_payload)
        client_rtt = round(time.monotonic() - client_start, 3)

        server_time = res.get("generation_time_seconds", 0.0)
        tps = res.get("tokens_per_second", 0.0)
        p_tok = res.get("prompt_tokens", 0)
        c_tok = res.get("completion_tokens", 0)
        net_overhead_ms = round((client_rtt - server_time) * 1000, 1)

        tps_results.append(tps)
        print(
            f"[Run {i}/{args.runs}] "
            f"Speed: {tps:6.2f} tok/s | "
            f"Server: {server_time:6.3f}s | "
            f"Client RTT: {client_rtt:6.3f}s (Net: +{net_overhead_ms}ms) | "
            f"Tokens: {p_tok} in / {c_tok} out | "
            f"Stop: {res.get('finish_reason')}"
        )

        if i == args.runs:
            print("-" * 75)
            print("Sample output:\n" + res.get("generated_text", ""))

    if len(tps_results) > 1:
        avg_tps = round(sum(tps_results) / len(tps_results), 2)
        print("-" * 75)
        print(f"Average Speed ({args.runs} runs): {avg_tps} tokens/s")


if __name__ == "__main__":
    main()
