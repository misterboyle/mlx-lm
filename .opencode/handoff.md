# Handoff: TurboQuant KV Cache for Qwen3.6 MoE

## Session Identity: Vidad (CoS)

## Completed

### This session
- **Fixed BatchTurboQuantKVCache prefix bug** (3 fixes)
  1. **Store prefix at beginning of buffer** — prefix tokens were stored at `self._k_prefix[..., self._idx : self._idx + actual_prefix, :]` but `self._idx` is the global token count (2048), not the position within the prefix buffer. Fixed to use `self._prefix_len` as the position.
  2. **Filter prefix storage** — after batch filtering, the prefix storage wasn't filtered, causing shape mismatch `(2,2,128,256)` vs `(1,2,128,256)`. Added `self._k_prefix = self._k_prefix[batch_indices]` to filter.
  3. **Reset `_prefix_len` in `trim()` and `filter()`** — these methods reset `_idx` but not `_prefix_len`, causing stale state.
- **Opencode test passes** — first message and continuation both work correctly
- **All 78 unit tests pass**
- **Bug bead created and closed**: hq-9mm.1 — "Bug: BatchTurboQuantKVCache prefix storage shape mismatch after batch filtering"

### Previous sessions (committed)
- Quantize-on-write lifecycle (hq-c19) — `update_and_fetch` uses `fused_quantize`
- Hybrid attention awareness (hq-vvj) — per-layer cache type detection for Qwen3.6 MoE
- Attention sinks (hq-9mm) — `min_tokens_before_quant` parameter keeps first N tokens in fp16

## In Progress

### Quality comparison testing (started but not completed)
- Started comparing response quality across:
  - Baseline (no TurboQuant)
  - TurboQuant 4-bit
  - TurboQuant 3-bit
- Need to run proper opencode tests with longer generations (200 tokens was too short)
- Use opencode, not curl, for meaningful quality assessment

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
- Complete quality comparison testing (opencode with longer generations)
- Compare TurboQuant quality vs baseline (already observed degradation)
- Consider if fused FA kernel (hq-0f9) would help quality or just memory

## Blockers

- **None** — the prefix bug is fixed

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
- `mlx_lm/models/turboquant_cache.py` — core cache classes (modified for attention sinks + prefix bug fixes)
- `mlx_lm/models/cache.py` — `make_prompt_cache` (passes `min_tokens_before_quant`)
- `mlx_lm/generate.py` — CLI args
- `mlx_lm/server.py` — CLI args + KV cache stats logging
- `tests/test_turboquant.py` — 78 tests (5 new for attention sinks)
- `scripts/safe_server.sh` — helper script for remote Mac testing

### Beads status
```
🌲 hq-9uw: TurboQuant KV cache compression for Qwen3.6 MoE on MLX [P0] (in_progress)
    ├── hq-0f9: Promote fused FA kernel for packed TQ3 K/V reads [P1] (open)
    ├── hq-0ir: Wire symmetric K quantization via prerot_fused_qk_scores [P2] (open)
    ├── hq-9mm: Integrate attention sinks -- fp16 prefix for first N tokens [P1] (✓ closed)
    │   └── hq-9mm.1: Bug: BatchTurboQuantKVCache prefix storage shape mismatch [P1] (✓ closed)
    ├── hq-c19: Implement quantize-on-write KV cache lifecycle [P0] (✓ closed)
    └── hq-vvj: Add hybrid attention awareness for Qwen3.6 MoE [P0] (✓ closed)
```

### Quality observation
- TurboQuant with `--turbo-kv-bits 3` produces noticeably poorer responses than baseline
- This matches prior findings — quantization degrades quality
- The fused FA kernel (hq-0f9) may help, or we may need to accept the tradeoff
