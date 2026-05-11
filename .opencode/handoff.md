# Handoff: TurboQuant KV Cache for Qwen3.6 MoE

## Session Identity: Vidad (CoS)

## Completed

### This session
- **Initialized beads database** in `mlx-lm-turbo` repo
- **Rebased onto upstream** `feature/turboquant-kv-cache` — picked up 4 upstream commits (MoE expert offloading, quantization config remaps)
- **Implemented attention sinks (hq-9mm)** — `min_tokens_before_quant` parameter keeps first N tokens in fp16, quantizes the rest
  - Modified `TurboQuantKVCache` and `BatchTurboQuantKVCache`
  - Added CLI arg `--turbo-min-tokens-before-quant` (default 128)
  - Added `make_prompt_cache` parameter passing
  - 5 new tests in `TestAttentionSinks` class
  - All 78 turboquant tests pass
- **Added KV cache stats logging** — logs memory usage after each request so we can track TurboQuant savings in real time
- **Created remote-mac skill** at `~/.config/opencode/skills/remote-mac/SKILL.md`
- **Created helper script** `scripts/safe_server.sh` — kills existing server before starting new one, checks memory pressure, waits for readiness
- **End-to-end testing** on remote Mac (172.16.49.25):
  - SSH key-based auth, opencode at `/opt/homebrew/bin/opencode`
  - Opencode `run` and `run -c` both work with TurboQuant active
  - Session continuity confirmed (model remembers conversation across `-c` invocations)
  - **Quality observation**: responses noticeably poorer with TurboQuant active (matches prior findings)

### Previous sessions (committed)
- Quantize-on-write lifecycle (hq-c19) — `update_and_fetch` uses `fused_quantize`
- Hybrid attention awareness (hq-vvj) — per-layer cache type detection for Qwen3.6 MoE

## In Progress

### Bug fix just committed
- **BatchTurboQuantKVCache prefix assignment bug** (just fixed, just pushed)
  - When `S < quant_start`, `actual_prefix` is 0 but `quant_start > 0`, causing broadcast error
  - Fix: check `actual_prefix > 0` instead of `quant_start > 0`
  - **Need to verify the fix works** — server crashed on first opencode test after pull

### Pending verification
- Pull the fix on remote Mac and restart server
- Test opencode again to confirm the fix works
- Check KV cache stats in logs to confirm memory savings are visible

## Pending

### Remaining beads
- **hq-0f9** — Promote fused FA kernel for packed TQ3 K/V reads (P1)
  - Single Metal kernel reads packed TQ3 directly, never materializes full fp16
  - Key to matching llama.cpp memory profile
  - Depends on hq-c19 (done)
- **hq-0ir** — Wire symmetric K quantization via prerot_fused_qk_scores (P2)
  - K quantized at 3-bit symmetric via rotated domain dot products
  - Depends on hq-0f9

### Future
- Compare TurboQuant quality vs baseline (already observed degradation)
- Consider if fused FA kernel (hq-0f9) would help quality or just memory

## Blockers

- **None** — just need to verify the bug fix works

## Notes

### Remote Mac
- **Host:** `172.16.49.25`
- **Auth:** Key-based (no password)
- **Specs:** 48GB RAM, 16 cores, macOS
- **Repo:** `~/mlx-lm-turbo`
- **venv:** already set up
- **Opencode:** `/opt/homebrew/bin/opencode`
- **Server logs:** `/tmp/mlx_lm_server.log`

### Testing workflow
1. `ssh michael@172.16.49.25`
2. `cd ~/mlx-lm-turbo && git pull origin feature/turboquant-kv-cache`
3. `bash scripts/safe_server.sh 8080 /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit --turbo-kv-bits 3`
4. Test: `/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run "hello"`
5. Continue: `/opt/homebrew/bin/opencode -m mlx-moe//Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit run -c "what did I just say?"`
6. Check logs: `tail -f /tmp/mlx_lm_server.log | grep "KV Cache:"`

### Key files
- `mlx_lm/models/turboquant_cache.py` — core cache classes (modified for attention sinks)
- `mlx_lm/models/cache.py` — `make_prompt_cache` (passes `min_tokens_before_quant`)
- `mlx_lm/generate.py` — CLI args
- `mlx_lm/server.py` — CLI args + KV cache stats logging
- `tests/test_turboquant.py` — 78 tests (5 new for attention sinks)
- `scripts/safe_server.sh` — helper script for remote Mac

### Beads status
```
🌲 hq-9uw: TurboQuant KV cache compression for Qwen3.6 MoE on MLX [P0] (in_progress)
    ├── hq-0f9: Promote fused FA kernel for packed TQ3 K/V reads [P1] (open)
    ├── hq-0ir: Wire symmetric K quantization via prerot_fused_qk_scores [P2] (open)
    ├── hq-9mm: Integrate attention sinks -- fp16 prefix for first N tokens [P1] (✓ closed)
    ├── hq-c19: Implement quantize-on-write KV cache lifecycle [P0] (✓ closed)
    └── hq-vvj: Add hybrid attention awareness for Qwen3.6 MoE [P0] (✓ closed)
```

### Quality observation
- TurboQuant with `--turbo-kv-bits 3` produces noticeably poorer responses than baseline
- This matches prior findings — quantization degrades quality
- The fused FA kernel (hq-0f9) may help, or we may need to accept the tradeoff
