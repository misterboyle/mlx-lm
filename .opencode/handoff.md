# Session Handoff

## Current State
- **Branch:** `hq-554-baseline` (5 commits ahead of `upstream/feature/turboquant-kv-cache`)
- **Latest commit:** `645360a` refactor: replace TurboQuant auto single-mode with explicit `--single-mode` flag
- **Remote Mac:** `172.16.49.25` (needs to pull latest branch)
- **Model:** `/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit`
- **Server port:** 8080

## What Was Done
1. **hq-ni0 (fix):** Implemented per-layer filtering in `make_prompt_cache()` for hybrid models. Added `--single-mode` flag (replaces auto single-mode from `--turbo-kv-bits`). Fixed `mx.Dtype` deepcopy bug in `TurboQuantKVCache` (stored as string `_dtype_str`).
2. **hq-ni0 (verification):** Tested on remote Mac with Qwen3.6:
   - Scenario A (story): Turns 1-3 ✅, Turn 4 empty at 11K tokens (quality issue)
   - Scenario B (coding, single-mode): All 4 turns ✅, no deepcopy errors, tool-calling works
   - Clean comparison: Standard KV 3.43 GB → TurboQuant 1.82 GB (**~47% memory savings**)

## What Needs to Be Done Next
**Pull upstream turboquant changes and implement batch mode support:**

1. **Pull latest from upstream:**
   ```bash
   ssh michael@172.16.49.25
   cd ~/mlx-lm-turbo
   git fetch upstream feature/turboquant-kv-cache
   git merge upstream/feature/turboquant-kv-cache
   # Resolve any conflicts
   ```

2. **Review upstream changes** — check what's new in the turboquant branch since we diverged:
   - New kernels? Algorithm changes?
   - Any new cache types or batch methods?
   - Check if upstream has added `merge/filter/extract/extend` to TurboQuantKVCache

3. **Implement batch mode for TurboQuant:**
   - The blocker was `TurboQuantKVCache` lacking `merge/filter/extract/extend` methods
   - `BatchKVCache` requires these for combining multiple sequences
   - Options:
     a) Implement batch methods on `TurboQuantKVCache` (complex — requires handling packed storage)
     b) Create `BatchTurboQuantKVCache` wrapper (easier, follows existing pattern)
     c) Keep single-mode only for TurboQuant (simpler, but loses caching benefit)
   - The `--single-mode` flag is now separate from `--turbo-kv-bits`, so batch mode can be enabled independently

4. **Test batch mode:**
   - Verify cache merging works correctly
   - Verify dequantization is correct after merge/extract
   - Run Scenario B with batch mode enabled (no `--single-mode`)
   - Compare cache hit rates vs single-mode

## Relevant Context
- The `--single-mode` flag is now the explicit way to force single-sequence generation
- TurboQuant no longer auto-forces single mode — batch mode support is the next step
- The deepcopy fix (`_dtype_str`) is needed regardless of batch mode
- Server logs at `/tmp/mlx_lm_server_tq.log` (or similar)
- Opencode at `/opt/homebrew/bin/opencode`
- venv at `~/mlx-lm-turbo/venv`

## Beads Status
- **hq-9uw** (epic): 6/8 complete (75%) — hq-ni0 now closed
- **hq-9qj**: Open — Add batch methods to TurboQuantKVCache (merge, filter, extract, extend)
- **hq-0f9**: Open — Promote fused FA kernel for packed TQ3 K/V reads
- **hq-0ir**: Open — Wire symmetric K quantization via prerot_fused_qk_scores
- **hq-ni0**: ✅ Closed (fix implemented and verified)
- **hq-554**: ✅ Closed (baseline test complete)

## Known Issues
- Turn 4 empty response at 11K tokens with TurboQuant 3-bit (quality issue, not a bug)
- `test_evaluate` fails due to missing `lm_eval` module (pre-existing, unrelated)
