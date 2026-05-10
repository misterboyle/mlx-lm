#!/usr/bin/env python3
"""Parallel decode stress test for MoE server.

Tests throughput, latency, and memory at different concurrency levels.
Simulates subagent dispatch patterns (short prompts, ~50-200 tokens).

Usage:
    python3 moe-parallel-stress.py --concurrency 4   # test 4 parallel requests
    python3 moe-parallel-stress.py --concurrency 8   # test 8 parallel requests
    python3 moe-parallel-stress.py --all             # test 4,6,8,12,16
"""

import argparse
import json
import subprocess
import time
import urllib.request
import urllib.error
import statistics
import sys

BASE_URL = "http://127.0.0.1:30083/v1/chat/completions"
MODEL = "/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"

# Simulate different subagent workload sizes
WORKLOADS = {
    "tiny": {  # ~50 tokens - quick tool call
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Run: glob src/**/*.py"}
        ],
        "max_tokens": 100
    },
    "short": {  # ~150 tokens - explore subagent
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Explore the codebase. Find all files related to cache management. Report file paths and key function names. Be concise."}
        ],
        "max_tokens": 500
    },
    "medium": {  # ~300 tokens - analysis subagent
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Analyze the server architecture. Find the main request handling code, API endpoints, and generation loop. Report the architecture, key classes, and file paths. Include a brief summary of the request flow from HTTP to token generation."}
        ],
        "max_tokens": 1000
    }
}


def send_request(payload, timeout=120):
    """Send a single request and return (latency_ms, total_tokens, cached_tokens, success)."""
    data = json.dumps(payload).encode()
    start = time.time()
    try:
        req = urllib.request.Request(
            BASE_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            elapsed_ms = (time.time() - start) * 1000
            usage = result.get("usage", {})
            total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            return elapsed_ms, total, cached, True
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return elapsed_ms, 0, 0, False


def run_batch(workload_name, n_requests, concurrency):
    """Run n_requests with given concurrency using threading."""
    import concurrent.futures
    
    workload = WORKLOADS[workload_name]
    payloads = []
    for i in range(n_requests):
        p = json.loads(json.dumps(workload))  # deep copy
        p["messages"] = json.loads(json.dumps(workload["messages"]))
        p["messages"].append({"role": "user", "content": f"Task {i}: {workload['messages'][-1]['content']}"})
        payloads.append(p)
    
    latencies = []
    tokens = []
    cached = []
    successes = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, p) for p in payloads]
        for f in concurrent.futures.as_completed(futures):
            elapsed, total, cache, ok = f.result()
            latencies.append(elapsed)
            tokens.append(total)
            cached.append(cache)
            if ok:
                successes += 1
    
    return {
        "latencies": latencies,
        "tokens": tokens,
        "cached": cached,
        "successes": successes,
        "total": n_requests
    }


def get_memory():
    """Get current RSS and wired memory for the MoE server process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "mlx_lm.server"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        if not pids[0]:
            return None
        
        total_rss = 0
        for pid in pids:
            pid = pid.strip()
            if not pid:
                continue
            try:
                out = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", pid],
                    stderr=subprocess.DEVNULL
                ).strip()
                total_rss += int(out)
            except:
                pass
        return total_rss / 1024  # Convert KB to MB
    except:
        return None


def analyze_results(results, workload_name, concurrency):
    """Analyze and print results."""
    lats = results["latencies"]
    tokens = results["tokens"]
    cached = results["cached"]
    successes = results["successes"]
    total = results["total"]
    
    print(f"\n{'='*60}")
    print(f"Workload: {workload_name} | Concurrency: {concurrency}")
    print(f"{'='*60}")
    print(f"  Success rate: {successes}/{total} ({successes/total*100:.0f}%)")
    print(f"  Throughput:   {total/(sum(lats)/1000):.2f} req/s")
    print(f"  Total tokens: {sum(tokens):,}")
    print(f"  Total cached: {sum(cached):,} ({sum(cached)/sum(tokens)*100:.0f}% if tokens > 0)")
    print()
    print(f"  Latency:")
    print(f"    Mean:   {statistics.mean(lats):>8.0f} ms")
    print(f"    Median: {statistics.median(lats):>8.0f} ms")
    print(f"    P50:    {sorted(lats)[int(len(lats)*0.5)]:>8.0f} ms")
    print(f"    P95:    {sorted(lats)[int(len(lats)*0.95)]:>8.0f} ms")
    print(f"    P99:    {sorted(lats)[int(len(lats)*0.99)]:>8.0f} ms")
    print(f"    Min:    {min(lats):>8.0f} ms")
    print(f"    Max:    {max(lats):>8.0f} ms")
    print(f"    StdDev: {statistics.stdev(lats) if len(lats) > 1 else 0:>8.0f} ms")
    
    mem = get_memory()
    if mem:
        print(f"\n  Server RSS: {mem:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="MoE parallel decode stress test")
    parser.add_argument("--concurrency", type=int, help="Test specific concurrency level")
    parser.add_argument("--all", action="store_true", help="Test all concurrency levels")
    parser.add_argument("--workload", choices=["tiny", "short", "medium"], default="short",
                        help="Workload size (default: short)")
    parser.add_argument("--n-requests", type=int, default=20, help="Number of requests per test")
    args = parser.parse_args()
    
    # Verify server is up
    try:
        urllib.request.urlopen("http://127.0.0.1:30083/health", timeout=5)
    except:
        print("ERROR: MoE server not responding on :30083")
        sys.exit(1)
    
    print("MoE Parallel Decode Stress Test")
    print(f"Model: {MODEL.split('/')[-1]}")
    print(f"Workload: {args.workload}")
    print(f"Requests per test: {args.n_requests}")
    
    if args.all:
        concurrency_levels = [4, 6, 8, 12, 16]
    elif args.concurrency:
        concurrency_levels = [args.concurrency]
    else:
        concurrency_levels = [8]  # default
    
    for concurrency in concurrency_levels:
        print(f"\n{'#'*60}")
        print(f"# Testing concurrency={concurrency}...")
        print(f"{'#'*60}")
        
        # Warmup
        print("  Warmup...")
        send_request({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        })
        
        # Run test
        print(f"  Running {args.n_requests} requests at concurrency={concurrency}...")
        results = run_batch(args.workload, args.n_requests, concurrency)
        analyze_results(results, args.workload, concurrency)
        
        # Brief pause between tests
        if concurrency_levels.index(concurrency) < len(concurrency_levels) - 1:
            print("\n  Pausing 5s before next test...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("Test complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
