---
name: model-quality-test
description: Standardized multi-turn quality testing for LLM inference configurations (baseline vs TurboQuant vs other quantization)
---

# Skill: Model Quality Testing

## Purpose

Standardized multi-turn quality testing for LLM inference configurations (baseline vs TurboQuant vs other quantization). Tests story generation, context retention, code reasoning, and tool-calling capability.

## Prerequisites

- Remote Mac with opencode installed (`/opt/homebrew/bin/opencode`)
- MLX LM server running on port 8080
- Model path: `/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit`
- Server logs at `/tmp/mlx_lm_server.log`

## Test Procedure

### Step 1: Start Server

```bash
ssh michael@172.16.49.25
cd ~/mlx-lm-turbo
pkill -f mlx_lm.server
sleep 2
PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 0.0.0.0 --port 8080 \
  [EXTRA_ARGS] \
  > /tmp/mlx_lm_server.log 2>&1 &
```

For TurboQuant: add `--turbo-kv-bits 3` or `--turbo-kv-bits 4`

### Step 2A: Context Test (Turns 1-4) — Session A

**⚠️ CRITICAL: Restart the server before Session A.**
Kill any existing server, clear logs, start a fresh server. This ensures the KV cache is empty.

```bash
# Kill old server, clear logs, start fresh
pkill -f mlx_lm.server
sleep 2
> /tmp/mlx_lm_server.log
PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 0.0.0.0 --port 8080 \
  [EXTRA_ARGS] \
  > /tmp/mlx_lm_server.log 2>&1 &
sleep 25
```

**⚠️ CRITICAL: Run turns sequentially, one at a time. Do NOT run them concurrently.**
Each turn must complete fully before starting the next. Running them in parallel will cause all turns to share the same cache state, producing invalid results.

**This is one self-contained session.** Turn 1 starts with a clean slate (no `-c`). Turns 2-4 continue from turn 1 (use `-c`).

```bash
# Turn 1: Story creation (CLEAN SLATE — no -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run 'Tell me a detailed story about a detective solving a case in Paris. Include at least 3 specific clues, 2 suspects, and a red herring. Make it at least 150 words.' > /tmp/test_turn1.txt 2>&1

# Turn 2: Early recall (CONTINUES from turn 1 — use -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c 'What were the three clues the detective found? List them explicitly.' > /tmp/test_turn2.txt 2>&1

# Turn 3: Mid recall (CONTINUES from turn 2 — use -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c 'Who was the red herring, and how did the detective realize it?' > /tmp/test_turn3.txt 2>&1

# Turn 4: Full synthesis (CONTINUES from turn 3 — use -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c 'Now summarize the entire case in 3 sentences, naming the real culprit and how the clues proved it.' > /tmp/test_turn4.txt 2>&1
```

### Step 2B: Coding Task Test (Turns 5-8) — Session B

**⚠️ CRITICAL: Restart the server between Session A and Session B.**
The server must be completely restarted to clear its KV cache. Otherwise Session B will inherit accumulated context from Session A, producing invalid results.

```bash
# Kill the old server
pkill -f mlx_lm.server
sleep 2

# Clear logs
> /tmp/mlx_lm_server.log

# Start fresh server (same config as before)
source venv/bin/activate
nohup python -m mlx_lm.server --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit --host 0.0.0.0 --port 8080 [EXTRA_ARGS] > /tmp/mlx_lm_server.log 2>&1 &

# Wait for server to be ready
sleep 25
```

**⚠️ CRITICAL: Run turns sequentially, one at a time. Do NOT run them concurrently.**
Each turn must complete fully before starting the next. Running them in parallel will cause all turns to share the same cache state, producing invalid results.

**This is a completely separate session from Session A.** Turn 5 starts with a clean slate (no `-c`). Turns 6-8 continue from turn 5 (use `-c`). Do NOT use `-c` for turn 5 — it must not inherit context from turns 1-4.

```bash
# Turn 5: Code navigation (CLEAN SLATE — no -c, separate session from turns 1-4)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run 'Look at the mlx_lm/models directory. Find the TurboQuant cache implementation and explain how the quantize-on-write lifecycle works. Specifically: how are the norms stored, and when does quantization happen?' > /tmp/test_turn5.txt 2>&1

# Turn 6: Kernel understanding (CONTINUES from turn 5 — use -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "Now look at the Metal kernel in turboquant_metal.py. How does the fused dequantize+attention kernel avoid materializing the full fp16 cache? What is the bit unpacking logic?" > /tmp/test_turn6.txt 2>&1

# Turn 7: Tradeoff analysis (CONTINUES from turn 6 — use -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "Based on what you have seen, what is the main tradeoff of TurboQuant vs a standard KV cache? Think about memory, speed, and quality." > /tmp/test_turn7.txt 2>&1

# Turn 8: Context recall (CONTINUES from turn 7 — use -c)
/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "Earlier I asked you about the quantize-on-write lifecycle. Can you tell me again how the norms are stored and what p_dim means?" > /tmp/test_turn8.txt 2>&1
```

### Step 3: Capture Memory Stats

Use the `/metrics` endpoint to get proper memory profiling data. This is more reliable than parsing log output.

```bash
ssh michael@172.16.49.25
curl -s http://172.16.49.25:8080/metrics | python3 -m json.tool
```

This returns:
- `mlx.active_memory_bytes` — total MLX memory in use
- `mlx.cache_memory_bytes` — MLX internal cache
- `mlx.peak_memory_bytes` — peak memory since startup
- `prompt_cache.total_sequences` — number of cached sequences
- `prompt_cache.total_bytes` — total cache size in bytes
- `prompt_cache.by_type` — breakdown by cache type (assistant, user, system)
- `model.loaded` — whether model is loaded
- `model.model_size_bytes` — model weight size
- `model.draft_model_loaded` — whether draft model is loaded
- `model.draft_model_size_bytes` — draft model size

Record these after each turn. Compare across configurations to verify:
1. **Cache compression** — TurboQuant should show smaller `total_bytes` than baseline
2. **Memory efficiency** — `active_memory_bytes` should be lower with TurboQuant
3. **Cache growth** — track how `total_sequences` and `total_bytes` grow across turns
4. **Model size** — verify model is loaded and size matches expectations

For a quick comparison across configs, run this after each test:

```bash
ssh michael@172.16.49.25
echo "=== $(date) ==="
curl -s http://172.16.49.25:8080/metrics | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f'Active: {m[\"mlx\"][\"active_memory_bytes\"] / 1e9:.2f} GB')
print(f'Peak:   {m[\"mlx\"][\"peak_memory_bytes\"] / 1e9:.2f} GB')
print(f'Spike:  {(m[\"mlx\"][\"peak_memory_bytes\"] - m[\"mlx\"][\"active_memory_bytes\"]) / 1e9:.2f} GB')
print(f'Cache:  {m[\"prompt_cache\"][\"total_sequences\"]} seq, {m[\"prompt_cache\"][\"total_bytes\"] / 1e6:.1f} MB')
for k, v in m['prompt_cache']['by_type'].items():
    if v['sequences'] > 0:
        print(f'  {k}: {v[\"sequences\"]} seq, {v[\"bytes\"] / 1e6:.1f} MB')
"
```

### Step 4: Capture Token Counts

```bash
grep 'Prompt processing complete' /tmp/mlx_lm_server.log
```

The denominator in `Prompt processing complete: N tokens` gives the total token count for the prompt. This tells us how much context is being accumulated.

### Step 5: Verify Outputs

Check each turn file for:
1. **Response quality** — meaningful text, not empty or truncated
2. **Tool-calling** — turns 5-8 should show `Glob`/`Read` tool calls, NOT `ls` hallucinations
3. **Context retention** — turn 8 should reference details from turn 5 (norms, p_dim)
4. **Story consistency** — turns 2-4 should reference same clues/suspects/red herring from turn 1
5. **Response length** — compare character counts across configs to ensure comparable output sizes

## Expected Baseline Results

### Context Test (Turns 1-4) — Session A

| Turn | Test | Expected |
|------|------|----------|
| 1 | Story creation | Clean story with 3 clues, 2 suspects, 1 red herring |
| 2 | Early recall | Lists all 3 clues correctly |
| 3 | Mid recall | Identifies red herring + reasoning |
| 4 | Full synthesis | 3 sentences, names culprit, links clues |

### Coding Task Test (Turns 5-8) — Session B

| Turn | Test | Expected |
|------|------|----------|
| 5 | Code nav | Detailed explanation of norms storage + quantization timing |
| 6 | Kernel understanding | Explains fused kernel + bit unpacking logic |
| 7 | Tradeoff analysis | Structured answer covering memory/speed/quality |
| 8 | Context recall | Recalls norms + p_dim from turn 5 |

## Red Flags

- **Empty responses** — model failed to generate
- **`ls` hallucinations** — model tries to call unavailable `ls` tool (indicates broken tool-calling)
- **Repeated text** — model stuck in loop
- **Generic responses** — "I don't have access to files" (model can't reason about codebase)
- **Context loss** — turn 8 doesn't reference turn 5 details
- **Cache size mismatch** — significantly smaller than expected (may indicate truncated responses)
- **Token count mismatch** — denominator in progress bar differs significantly between configs (indicates different context accumulation)

## Comparison

Run baseline first, then run same test with TurboQuant enabled. Compare:
1. **Cache compression** — use `/metrics` to verify `prompt_cache.total_bytes` is smaller with TurboQuant
2. **Memory efficiency** — compare `mlx.active_memory_bytes` across configs
3. **Response quality** — side-by-side
4. **Tool-calling capability** — no `ls` hallucinations
5. **Context retention** — turn 8 references turn 5
6. **Response lengths** — character counts should be similar

## Notes

- Use opencode, not curl — opencode provides tool-calling interface
- Use `-c` flag for continuation (grows context within a session)
- **NO `-c` for turn 1 and turn 5** — these start fresh sessions
- **Use `-c` for turns 2-4** (continue Session A) and **turns 6-8** (continue Session B)
- **Session A (turns 1-4) and Session B (turns 5-8) are completely separate** — turn 5 must NOT inherit context from turns 1-4
- **Restart the server between Session A and Session B** — kill the old server, clear logs, start a fresh server. This clears the KV cache so Session B starts clean.
- **Restart the server between config tests** (baseline → TQ3 → TQ4) — same procedure. Each config test must start with a clean server.
- **ALL turns must be run sequentially, one at a time.** Never run multiple turns concurrently — each turn must complete before the next one starts. Parallel execution shares cache state and produces invalid results.
- **Session A (turns 1-4) is sufficient for comparing KV cache compression.** Use the `/metrics` endpoint to verify `prompt_cache.total_bytes` is smaller with TurboQuant.
- **Session B (turns 5-8) is only needed if you see tool call loops.** Tool call loops can happen even with fp16 KV cache, so they're not a definitive indicator of a TurboQuant regression. If Session B works on the first try, great. If it loops, retry a few times.
- Server must be running before each test
- Clear server logs between tests if needed: `> /tmp/mlx_lm_server.log`
- Each turn takes ~30-60 seconds to complete
- **Turns 5-8 are the critical test** — this is where TQ divergence manifests
- Use `curl -s http://localhost:8080/metrics | python3 -m json.tool` to check memory stats at any point
