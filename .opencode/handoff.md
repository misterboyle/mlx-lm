# Session Handoff

## Current State
- **Branch:** `hq-554-baseline` (3 commits ahead of `upstream/feature/turboquant-kv-cache`)
- **Latest commit:** `ece5cb1` - fix: use self.cli_args instead of cli_args
- **Remote Mac:** `172.16.49.25` (needs to pull latest branch)
- **Model:** `/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit`
- **Server port:** 8080

## What Was Done
1. **hq-554 (baseline test):** Confirmed upstream errors on Qwen3.6 with `--turbo-kv-bits 3` in single mode. Error: `ValueError: [TurboQuant] Incompatible cache type in layer 0: ArraysCache`
2. **hq-ni0 (fix):** Implemented per-layer filtering in `make_prompt_cache()` to replace blanket rejection. Also added force single mode when `--turbo-kv-bits` is set.
3. **hq-ni0 (verification):** Tested on remote Mac with opencode Session A (story generation). Results:
   - Turn 1: ✅ Story creation works (1284 bytes)
   - Turn 2: ✅ Early recall works (334 bytes)
   - Turn 3: ✅ Mid recall works (349 bytes)
   - Turn 4: ⚠️ Empty response (81 bytes) - likely TurboQuant 3-bit quality issue with long context (11011 tokens)

## What Needs to Be Done Next
**Run Scenario B (coding task test) on the latest commit:**

1. **Pull latest code on remote Mac:**
   ```bash
   ssh michael@172.16.49.25
   cd ~/mlx-lm-turbo
   git fetch origin hq-554-baseline:hq-554-baseline
   git checkout hq-554-baseline
   ```

2. **Restart server fresh:**
   ```bash
   pkill -f mlx_lm.server
   sleep 2
   > /tmp/mlx_lm_server.log
   source venv/bin/activate
   nohup python -m mlx_lm.server --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit --host 0.0.0.0 --port 8080 --turbo-kv-bits 3 > /tmp/mlx_lm_server.log 2>&1 &
   sleep 25
   ```

3. **Run Scenario B turns sequentially (one at a time, no parallel):**
   - **Turn 5 (CLEAN SLATE):** `opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run 'Look at the mlx_lm/models directory. Find the TurboQuant cache implementation and explain how the quantize-on-write lifecycle works. Specifically: how are the norms stored, and when does quantization happen?'`
   - **Turn 6 (CONTINUE):** `opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "Now look at the Metal kernel in turboquant_metal.py. How does the fused dequantize+attention kernel avoid materializing the full fp16 cache? What is the bit unpacking logic?"`
   - **Turn 7 (CONTINUE):** `opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "Based on what you have seen, what is the main tradeoff of TurboQuant vs a standard KV cache? Think about memory, speed, and quality."`
   - **Turn 8 (CONTINUE):** `opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "Earlier I asked you about the quantize-on-write lifecycle. Can you tell me again how the norms are stored and what p_dim means?"`

4. **Check results:**
   - Verify each turn produces meaningful output (not empty)
   - Check for tool-calling capability (should see Glob/Read tool calls, NOT ls hallucinations)
   - Verify context retention (turn 8 should reference details from turn 5)
   - Check server logs for errors

5. **Update hq-ni0 with results**

## Relevant Context
- The fix forces single mode when `--turbo-kv-bits` is set (because `TurboQuantKVCache` lacks `merge/filter/extract/extend`)
- Session A (story) worked for turns 1-3, failed on turn 4 (empty response at 11011 tokens)
- Session B (coding) tests tool-calling capability and context retention on codebase questions
- Server logs at `/tmp/mlx_lm_server.log`
- Opencode is at `/opt/homebrew/bin/opencode`
- venv at `~/mlx-lm-turbo/venv`

## Beads Status
- **hq-9uw** (epic): 5/8 complete (62%)
- **hq-ni0**: In progress (fix implemented, Session A tested, Session B pending)
- **hq-554**: Closed (baseline test complete)
- **hq-9qj**: Open (batch methods - not needed for single mode)
- **hq-0f9**: Open (fused FA kernel - future optimization)
- **hq-0ir**: Open (symmetric K quantization - future optimization)
