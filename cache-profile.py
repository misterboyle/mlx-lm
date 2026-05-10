#!/usr/bin/env python3
"""
Standalone cache profiler for the MoE server.

Designed to be run by a human (not an agent) so it can safely
restart the server for clean baselines.

Experiments:
  1. cache_hit_rate   - Identical prompts → measure hit rate
  2. y_value_analysis - Correlate log Y-values with known prompt sizes
  3. cache_growth     - Unique prompts → track cache growth curve
  4. eviction         - Push past caps → observe eviction behavior
  5. concurrent       - Two sessions interleaved → interference

Usage:
  python3 cache-profile.py <experiment> [options]

  Examples:
    python3 cache-profile.py cache_hit_rate
    python3 cache-profile.py all --rounds 10
    python3 cache-profile.py eviction --model /path/to/model
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.path.expanduser("~/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit")
DEFAULT_PORT = 30083
DEFAULT_LOG = os.path.expanduser("~/.localllm/moe.log")
DEFAULT_MAX_TOKENS = 50
DEFAULT_ROUNDS = 5


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class CacheSnapshot:
    """Parsed from 'Prompt Cache:' log lines."""
    total_sequences: int = 0
    total_gb: float = 0.0
    assistant_seqs: int = 0
    assistant_gb: float = 0.0
    user_seqs: int = 0
    user_gb: float = 0.0
    system_seqs: int = 0
    system_gb: float = 0.0

    @classmethod
    def from_log_line(cls, line: str) -> "CacheSnapshot | None":
        m = re.match(r"Prompt Cache: (\d+) sequences, ([\d.]+) GB", line)
        if not m:
            return None
        snap = cls(total_sequences=int(m.group(1)), total_gb=float(m.group(2)))
        for role in ("assistant", "user", "system"):
            rm = re.search(rf"- {role}: (\d+) sequences, ([\d.]+) GB", line)
            if rm:
                setattr(snap, f"{role}_seqs", int(rm.group(1)))
                setattr(snap, f"{role}_gb", float(rm.group(2)))
        return snap

    def __str__(self):
        return (f"Cache[{self.total_sequences}seq/{self.total_gb:.1f}GB] "
                f"A:{self.assistant_seqs} U:{self.user_seqs} S:{self.system_seqs}")


def parse_cache_log(log_path: str, lines_back: int = 200) -> list[CacheSnapshot]:
    snapshots = []
    try:
        with open(log_path) as f:
            for line in f.readlines()[-lines_back:]:
                snap = CacheSnapshot.from_log_line(line.strip())
                if snap:
                    snapshots.append(snap)
    except FileNotFoundError:
        pass
    return snapshots


def get_latest_cache(log_path: str) -> CacheSnapshot | None:
    snaps = parse_cache_log(log_path)
    return snaps[-1] if snaps else None


def api_call(model: str, messages: list, max_tokens: int, port: int) -> dict | None:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        proc = subprocess.run(
            ["curl", "-s", f"http://localhost:{port}/v1/chat/completions",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        return data.get("usage", {})
    except Exception:
        return None


def restart_server(port: int, log_path: str, timeout: int = 120) -> bool:
    """Restart the MoE server and wait for it to be ready."""
    print(f"\n[RESTART] Stopping MoE server...")
    subprocess.run(["pkill", "-f", "start-server.sh moe"],
                   capture_output=True)
    time.sleep(3)

    print(f"[RESTART] Starting MoE server...")
    subprocess.Popen(
        ["bash", os.path.expanduser("~/localllm/start-server.sh"), "moe"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            proc = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                capture_output=True, timeout=5,
            )
            if proc.returncode == 0:
                time.sleep(5)  # let cache settle
                print(f"[RESTART] Server ready. Cache: {get_latest_cache(log_path)}")
                return True
        except Exception:
            pass
        time.sleep(2)

    print("[RESTART] Timed out waiting for server")
    return False


# ─── Experiments ────────────────────────────────────────────────────────────

def experiment_cache_hit_rate(args):
    print("=" * 60)
    print("EXPERIMENT 1: Cache Hit Rate")
    print("=" * 60)

    prompts = [
        ("tiny", [{"role": "user", "content": "hi"}]),
        ("short", [{"role": "user", "content": "What is the capital of France?"}]),
        ("medium", [{"role": "user", "content": (
            "Write a detailed explanation of how transformer models work, "
            "including attention mechanisms, positional encoding, and "
            "feed-forward layers. Be thorough."
        )}]),
        ("long", [{"role": "user", "content": (
            "Explain the complete history of computing from Babbage's "
            "analytical engine to modern large language models. Cover "
            "key milestones, breakthroughs, and paradigm shifts. "
            "Include at least 15 distinct historical periods or inventions."
        )}]),
    ]

    for name, messages in prompts:
        print(f"\n--- {name} ({len(messages[0]['content'])} chars) ---")
        results = []
        for i in range(args.rounds + 1):
            usage = api_call(args.model, messages, args.max_tokens, args.port)
            if not usage:
                print(f"  Call {i}: FAILED")
                continue
            results.append(usage)
            print(f"  Call {i}: prompt={usage['prompt_tokens']}tok, "
                  f"completion={usage['completion_tokens']}tok, "
                  f"total={usage['total_tokens']}tok")

        # Analyze
        if len(results) >= 2:
            cold = results[0]
            warm_avg = sum(r['prompt_tokens'] for r in results[1:]) / len(results[1:])
            if cold['prompt_tokens'] > 0:
                reduction = (1 - warm_avg / cold['prompt_tokens']) * 100
                print(f"  Hit rate: {reduction:.0f}% token reduction (cold={cold['prompt_tokens']} → warm={warm_avg:.0f})")

    print("\n" + "=" * 60)


def experiment_y_value_analysis(args):
    print("=" * 60)
    print("EXPERIMENT 5: Y-Value Analysis")
    print("=" * 60)

    # Build known cache entries
    print("\nBuilding cache with known prompts...")
    cache_entries = [
        ("system", [{"role": "system", "content": "You are a helpful assistant."}]),
        ("user_short", [{"role": "user", "content": "Hello."}]),
    ]
    for name, msgs in cache_entries:
        usage = api_call(args.model, msgs, args.max_tokens, args.port)
        if usage:
            print(f"  Cached: {name} → {usage['prompt_tokens']} tokens")

    # Partial overlap: system is cached, user is new
    print("\n--- Partial overlap (system cached, user new) ---")
    cache_before = get_latest_cache(args.log)
    print(f"  Cache before: {cache_before}")

    partial = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"},
    ]
    usage = api_call(args.model, partial, args.max_tokens, args.port)
    cache_after = get_latest_cache(args.log)

    if usage:
        print(f"  Result: prompt={usage['prompt_tokens']}tok")
        print(f"  Cache after: {cache_after}")
        print(f"  If system is cached, prompt_tokens < total_prompt_tokens")

    # Full overlap: exact same prompt
    print("\n--- Full overlap (exact same prompt) ---")
    cache_before = get_latest_cache(args.log)
    usage = api_call(args.model, partial, args.max_tokens, args.port)
    cache_after = get_latest_cache(args.log)

    if usage:
        print(f"  Result: prompt={usage['prompt_tokens']}tok")
        print(f"  Cache: {cache_before} → {cache_after}")

    print("\n" + "=" * 60)


def experiment_cache_growth(args):
    print("=" * 60)
    print("EXPERIMENT 3: Cache Growth Curve")
    print("=" * 60)

    sizes = [1, 5, 10, 25, 50, 100]
    print(f"\nSending {len(sizes)} unique prompts of increasing size...\n")

    for i, size_kb in enumerate(sizes):
        filler = "The quick brown fox jumps over the lazy dog. " * (size_kb * 15)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"[Session {i+1}] {filler[:size_kb*1000]}"},
        ]
        cache_before = get_latest_cache(args.log)
        usage = api_call(args.model, messages, args.max_tokens, args.port)
        cache_after = get_latest_cache(args.log)

        if usage and cache_after:
            seq_delta = cache_after.total_sequences - (cache_before.total_sequences if cache_before else 0)
            gb_delta = cache_after.total_gb - (cache_before.total_gb if cache_before else 0)
            print(f"[{i+1}/{len(sizes)}] {size_kb:3d}KB → prompt={usage['prompt_tokens']}tok, "
                  f"cache={cache_after.total_sequences}seq/{cache_after.total_gb:.1f}GB "
                  f"(+{seq_delta}seq, +{gb_delta:.1f}GB)")

    print("\n" + "=" * 60)


def experiment_eviction(args):
    print("=" * 60)
    print("EXPERIMENT 4: Eviction Under Pressure")
    print("=" * 60)

    num = 60  # exceeds promptCacheSize=50
    print(f"\nSending {num} unique short prompts (cap=50 sequences)...\n")

    for i in range(num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Unique prompt #{i+1}. Session identifier: {i+1}."},
        ]
        api_call(args.model, messages, args.max_tokens, args.port)
        cache = get_latest_cache(args.log)

        if cache and (i + 1) % 5 == 0:
            print(f"  [{i+1:3d}] {cache}")

    print("\n" + "=" * 60)


def experiment_concurrent(args):
    print("=" * 60)
    print("EXPERIMENT 4: Concurrent Sessions")
    print("=" * 60)

    session_a = [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": "Solve 23 * 47 step by step."},
    ]
    session_b = [
        {"role": "system", "content": "You are a writing assistant."},
        {"role": "user", "content": "Write a haiku about algorithms."},
    ]

    print(f"\nRunning {args.rounds} alternating rounds (A, B, A, B, ...)...\n")
    for i in range(args.rounds):
        for name, msgs in [("A", session_a), ("B", session_b)]:
            cache_before = get_latest_cache(args.log)
            usage = api_call(args.model, msgs, args.max_tokens, args.port)
            cache_after = get_latest_cache(args.log)

            if usage and cache_after:
                print(f"  R{i+1}.{name}: prompt={usage['prompt_tokens']}tok, "
                      f"cache={cache_after.total_sequences}seq/{cache_after.total_gb:.1f}GB")

    print("\n" + "=" * 60)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Server Cache Profiler")
    parser.add_argument("experiment", choices=[
        "cache_hit_rate", "y_value_analysis", "cache_growth",
        "eviction", "concurrent", "all"
    ])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--restart", action="store_true",
                        help="Restart server before running (clears cache)")

    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Check server
    try:
        proc = subprocess.run(
            ["curl", "-sf", f"http://localhost:{args.port}/health"],
            capture_output=True, timeout=5,
        )
        if proc.returncode != 0:
            print(f"ERROR: Server not on port {args.port}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Optional restart
    if args.restart:
        if not restart_server(args.port, args.log):
            sys.exit(1)

    experiments = {
        "cache_hit_rate": experiment_cache_hit_rate,
        "y_value_analysis": experiment_y_value_analysis,
        "cache_growth": experiment_cache_growth,
        "eviction": experiment_eviction,
        "concurrent": experiment_concurrent,
    }

    if args.experiment == "all":
        for name, fn in experiments.items():
            fn(args)
            print()
    else:
        experiments[args.experiment](args)


if __name__ == "__main__":
    main()
