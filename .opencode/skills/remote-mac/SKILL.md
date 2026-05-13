---
name: remote-mac
description: SSH into remote Mac for end-to-end testing with mlx-lm server
---

# Skill: remote-mac

## Purpose

SSH into the dedicated MacBook for interactive testing with real models and servers.

## Iterative Workflow

**Always follow this cycle when working on remote Mac testing:**

```
1. DEVELOP locally — write code, run tests (python -m unittest discover tests/ -v)
2. COMMIT locally — git add -A && git commit -m "clear message"
3. PUSH to GitHub — git push origin feature/turboquant-kv-cache
4. SSH to remote — ssh michael@172.16.49.25
5. PULL on remote — cd ~/mlx-lm-turbo && git pull origin feature/turboquant-kv-cache
6. TEST integration — launch server directly + opencode
7. CHECK logs — tail -f /tmp/mlx_lm_server.log | grep -E "TURBO|KV-QUANT|KV Cache:"
```

**Critical reminders:**
- **Never skip commit + push** — if it's not committed and pushed, the remote can't pull it
- **Always pull before testing** — the remote may have stale code
- **Always kill existing servers before starting a new one** — `pkill -f mlx_lm.server`
- **Check git log on remote** after pull to confirm you have the right commits

## Connection Details

```bash
ssh michael@172.16.49.25
# Key-based auth (no password)
```

## Machine Specs

- 48GB RAM, 16 cores, macOS
- Repo at `~/mlx-lm-turbo`
- venv already set up
- Opencode at `/opt/homebrew/bin/opencode`

## Server Lifecycle Management

**Always kill existing servers before starting a new one** to prevent OOM crashes.

### Start server

```bash
ssh michael@172.16.49.25
cd ~/mlx-lm-turbo
pkill -f mlx_lm.server
sleep 2
PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 0.0.0.0 --port 8080 \
  --turbo-kv-bits 3 \
  > /tmp/mlx_lm_server.log 2>&1 &
```

Replace `--turbo-kv-bits 3` with any server args you need (e.g., `--single-mode`, `--turbo-fp16-layers 2`).

### Stop server

```bash
ssh michael@172.16.49.25
pkill -f mlx_lm.server
sleep 2
lsof -ti :8080 || echo "Port 8080 is free"
```

### Check server status

```bash
ssh michael@172.16.49.25
lsof -ti :8080 && echo "Server running" || echo "Server not running"
tail -f /tmp/mlx_lm_server.log
```

## Testing Protocol

### Step 1: Curl test FIRST (always)

**Before running opencode, verify the server responds with curl.** This avoids hanging on opencode when the server is broken.

```bash
ssh michael@172.16.49.25
curl -s --max-time 30 http://172.16.49.25:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

Expected: JSON response with `choices[0].message.content`. If curl fails, **do not run opencode** — fix the server first.

### Step 2: Opencode test (only if curl passes)

**Default timeout: 30s for first-turn tests.** Opencode can hang indefinitely on broken servers — always use timeouts.

```bash
ssh michael@172.16.49.25
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run "hello"
```

Expected: Model responds with a greeting. If this hangs past 30s, the server is likely stuck — check logs.

### Step 3: Session continuity (the real test)

```bash
# First message
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run "what is your name?"

# Continue the session (this tests KV cache persistence)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "what did I just ask you?"
```

Expected: Model remembers the previous message. The `-c` flag continues the session, growing the conversation context.

### Step 4: Multi-turn conversation

```bash
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run "summarize our conversation so far in one sentence"
```

Expected: Model summarizes the full conversation, proving context is being maintained.

### Step 5: TurboQuant with opencode

```bash
# Start server with TurboQuant
ssh michael@172.16.49.25
cd ~/mlx-lm-turbo
pkill -f mlx_lm.server
sleep 2
PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 0.0.0.0 --port 8080 \
  --turbo-kv-bits 3 \
  > /tmp/mlx_lm_server.log 2>&1 &

# Test with opencode
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run "hello"
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "what did I just say?"
```

Expected: Same as above, but with TurboQuant compression active. Check logs for `[TURBO]` messages.

## Verifying TurboQuant is Active

```bash
ssh michael@172.16.49.25
tail -f /tmp/mlx_lm_server.log | grep -E "TURBO|KV-QUANT|make_prompt_cache"
```

Look for:
- `turbo_kv_bits=3` in the logs
- `TurboQuantKVCache` in the cache types
- `[TURBO]` messages

## Checking Prompt Cache Growth

```bash
ssh michael@172.16.49.25
tail -f /tmp/mlx_lm_server.log | grep "Prompt Cache:"
```

Watch the cache grow as you add messages with `-c`. The rotating cache should trim old entries when it hits the limit.

## Common Workflows

### 1. Test with curl

```bash
curl http://172.16.49.25:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### 2. Check server logs

```bash
ssh michael@172.16.49.25
tail -f /tmp/mlx_lm_server.log
```

### 3. Run tests

```bash
ssh michael@172.16.49.25
cd ~/mlx-lm-turbo
source venv/bin/activate
python -m unittest tests.test_turboquant -v
```

## Known Issues & Debugging

### Debugging with TQ_DEBUG

When adding debug logging for TurboQuant issues:
- **Clear the log file before restarting:** `> /tmp/mlx_lm_server.log` — stale log data from previous runs confuses debugging
- **Export the env var in the shell:** `export TQ_DEBUG=1` before starting the server, not just `TQ_DEBUG=1` in the command — the env var may not propagate to the server process
- **Log at every state change point:** Not just the start of `update_and_fetch`, but also when `_prefix_len` is modified, and in `trim()`, `filter()`, and `__init__`
- **Include the cache id:** `id(self)` in log lines to distinguish between different cache objects
- **Add try/except around the error point:** To capture full state when the error occurs

### KV Cache stats logging silently fails

The `KV Cache:` logging in `server.py` silently fails (caught by `except Exception: pass`). This is a known gap — the prompt cache object may not have `nbytes` or `stats_by_type()` attributes. Not critical for functionality but useful for debugging.

### Opencode hangs

If opencode hangs past 30s:
1. Kill the opencode process (Ctrl+C)
2. Check server logs: `tail -50 /tmp/mlx_lm_server.log`
3. Look for `Exception` or `Traceback` in logs
4. If server is stuck, restart it: `pkill -f mlx_lm.server; sleep 2; PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server ...`

### BrokenPipeError in logs

Normal — it's opencode closing the connection after receiving its response. Not an error.

### Nested mlx_lm/mlx_lm/ folder on remote

There may be a nested `mlx_lm/mlx_lm/` folder on the remote Mac. This is harmless — the editable install points to the correct location. It's confusing but doesn't affect testing.

### Quality testing with opencode

When comparing response quality across different configurations (e.g., baseline vs TurboQuant):
- **Use opencode, not curl** — curl with short responses (200 tokens) doesn't reveal quality differences
- **Use longer generations** — at least 500+ tokens to see meaningful differences
- **Use diverse prompts** — creative writing, reasoning, code generation, etc.
- **Run multiple trials** — single samples can be misleading due to temperature randomness
- **Compare the same prompt** across configurations to isolate the effect

## Notes

- The remote machine has the same repo as local but may be on a different branch
- Always activate venv before running anything
- Use `nohup` for long-running servers
- Server logs go to `/tmp/mlx_lm_server.log`
- Port 8080 is the default server port
- **Always kill existing servers before starting a new one** — `pkill -f mlx_lm.server`
- Opencode is at `/opt/homebrew/bin/opencode`
- The `-c` flag in opencode continues the session (grows context)
- **Always test with curl first, opencode second**
- **Default timeout for opencode: 30s**
- **Default timeout for curl: 30s**
- BrokenPipeError in logs is normal — it's opencode closing the connection after receiving its response
