# Handoff: hq-9qj.8 — Lifecycle Tests & SSM Bug Discovery

## Completed

- **`hq-9qj.8`**: Improved BatchTurboQuantKVCache test coverage based on reference patterns.
- **New test file**: `tests/test_turboquant_server_lifecycle.py` (377 lines, 4 tests).
- **Model**: Qwen3.6-35B-A3B-UD-MLX-4bit (hybrid SSM+Transformer).

### Test Results

| Test | Result | Notes |
|------|--------|-------|
| `test_save_load_inference` | **PASS** | Save→load outputs match within bfloat16 precision (`rtol=0, atol=1e-0`). |
| `test_model_save_load_inference` | **PASS** | Tokens match exactly, logits match within tolerance. |
| `test_multi_step_decode_after_merge` | **ERROR** | SSM layer shape mismatch on multi-token decode. |
| `test_full_server_lifecycle` | **ERROR** | Same SSM layer shape mismatch after filter/trim. |

## Bugs Found

### 1. SSM Layer Shape Mismatch (Critical)

**Symptom:**
```
ValueError: [concatenate] All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are (2,3,8192), (1,3,8192), and the concatenation axis is 1.
```

**Location:**
`mlx_lm/models/qwen3_5.py` line 158 in `linear_attn` (qwen3_5.py:158).

**Context:**
- Happens during **multi-token decode** with **batched caches** (B=2).
- Single-token decode works (existing tests pass).
- Happens in the SSM layer (`linear_attn`), not the attention layers.
- The shape mismatch is `(2,3,8192)` vs `(1,3,8192)` — likely `conv_state` vs `qkv` concatenation.

**Impact:**
- Prevents multi-token decode from working with batched caches.
- This is a real production bug that happens in the server during continuous batching.

## What to Do Next

1. **Fix SSM shape mismatch**:
   - Investigate `qwen3_5.py` line 158.
   - The issue is likely that `conv_state` and `qkv` have different batch dimensions when processing multiple tokens at once.
   - Ensure the SSM layer handles batched inputs correctly for multi-token sequences.

2. **Smoke test fixes**:
   - Run `tests.test_turboquant_server_lifecycle` to verify the fix.
   - Run `tests.test_turboquant` to ensure no regressions.
   - Verify single-token decode still works.

3. **Continue `hq-9qj`**:
   - Once the SSM bug is fixed, re-run the failing tests.
   - If they pass, move on to `hq-9qj.7` (Model quality testing).

## Notes

- **Tolerance**: Tests use `rtol=0, atol=1e-0` for logits comparison. This accounts for bfloat16 precision floor (~0.01-0.1) and multiple comparisons across 248k logits. An error of 1.0 in a logit corresponds to a factor of `e` (~2.7) in probability.
- **Model**: Qwen3.6 hybrid has SSM layers (ArraysCache) and attention layers (TurboQuantKVCache). Only attention layers participate in merge/filter/extend/trim operations.
- **Server flow**: `from_state` is NOT used in the normal server flow. The LRU cache stores live Python objects and uses `copy.deepcopy` when fetching. `from_state` is only for disk persistence (`load_prompt_cache`), which is tested separately.
