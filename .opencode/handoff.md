# Handoff: hq-9qj.10 — batch+TurboQuant output quality regression

## Branch State
- **hq-9qj-batch**: Contains regression test + two fixes committed

## Completed

### Regression test locked down
`test_batch_turbo_quant_matches_single_mode` captures the bug:
- FP16 single vs batch: similarity = **1.00** (batching itself is fine)
- Single-mode TQ vs batch+TQ: similarity = **0.34** (was 0.20, improved after fixes)
- Threshold: batch+TQ must match single-mode+TQ to same degree as FP16 (≥0.99)

### Two bugs found and fixed in `turboquant_cache.py`

**Bug 1: `_get_k_boundaries()` / `_get_v_boundaries()` returned centroids, not boundaries**
- `fused_quantize` expects boundaries (midpoints between codebook values), not centroids (codebook values)
- `merge()` stored `first._k_q.centroids` in `batch_cache._k_centroids`
- `_get_k_boundaries()` returned `self._k_centroids` directly instead of computing `(centroids[:-1] + centroids[1:]) / 2.0`
- **Fix**: Changed both methods to compute boundaries from centroids
- **Impact**: Small improvement (0.20 → 0.25)

**Bug 2: Norms written as scalars instead of reshaped arrays**
- `BatchTurboQuantKVCache.update_and_fetch` line 648: `self.k_norms[i, :, pad:end] = k_nrm[0]`
- `k_nrm` has shape `(H*S,)` from `fused_quantize`, so `k_nrm[0]` is a scalar
- Single mode does `k_nrm.reshape(B, H, S)` (line 211)
- Same bug on line 664 for `v_nrm[0]`
- **Fix**: Changed to `k_nrm.reshape(H, S)` and `v_nrm.reshape(H, S)`
- **Impact**: Small improvement (0.25 → 0.34)

## Current State

**After both fixes:**
- KV values after prefill: IDENTICAL (diff=0.000000) ✓
- KV values after decode: STILL DIFFERENT (K diff=14.875, V diff=8.500) ✗
- Logits diverge in FIRST decode step (max diff ~1.4) ✗
- Similarity improved from 0.20 → 0.34 but still failing

## What was ruled out

1. **KV quantization/dequantization**: Identical after prefill (diff=0.000000)
2. **RoPE offset type**: Qwen3.6 tolerates array offsets (diff=0.0)
3. **Attention mask**: Both return None for N=1 decode step
4. **Quantizer params**: Centroids and boundaries now match after fix

## What remains (next steps)

The KV values after prefill are identical, but after a decode step they diverge massively. The bug is in `BatchTurboQuantKVCache.update_and_fetch` during the decode path.

**Hypothesis**: The norms are still wrong after the reshape fix. Check:
1. `k_nrm` shape from `fused_quantize` — is it `(H*S,)` or `(B*H*S,)`? The batched cache processes one batch entry at a time in the loop, so `key_slice` is `(1, H, S, k_dim)`, which reshapes to `(H*S,)`. After `fused_quantize`, `k_nrm` should be `(H*S,)`. The reshape to `(H, S)` should be correct.
2. But wait — `fused_quantize` is called on `key_slice.reshape(-1, k_dim)` which is `(H*S,)`. The output `k_nrm` is `(H*S,)`. So `k_nrm.reshape(H, S)` should be correct.
3. **Check**: Is `k_nrm` actually `(H*S,)` or something else? Print the shape.
4. **Check**: Is the issue in `_quantize_key_packed` or `_quantize_value_packed`? These call `fused_quantize` on `keys.reshape(-1, k_dim)`. For a single batch entry, `keys` is `(1, H, S, k_dim)`, so `reshape(-1, k_dim)` is `(H*S,)`. The output `k_nrm` is `(H*S,)`.
5. **Check**: Is there a mismatch in how `k_packed` is written vs how `k_norms` is written? `k_packed` uses `k_pk[0]` (squeezes batch dim), `k_norms` uses `k_nrm.reshape(H, S)`.
6. **Check**: Is the issue in `_fetch_all`? It reads from `start: start+length` positions and returns data at `:length`. For a single-entry batch with `left_padding=[0]`, `start=0`, `length=offset`. This should be correct.

**Concrete next steps:**
1. Add print statements to `BatchTurboQuantKVCache.update_and_fetch` to log `k_nrm.shape` and `k_nrm` values
2. Compare with single mode's `k_nrm.reshape(B, H, S)` — are the values identical?
3. If norms are correct, check if the issue is in how `k_packed` is written (line 647: `self.k_packed[i, :, pad:end, :] = k_pk[0]`)
4. If both are correct, check `_fetch_all` — does it read from the right positions?
5. Consider: the batched cache processes entries in a loop (one at a time), while single mode processes all entries at once. Is there a subtle difference in how `fused_quantize` handles the input shape?

## Related beads
- hq-fka.3: Rope offset type bug (separate, not blocking)
- hq-9qj.8: Lifecycle tests (pass with Qwen3.6)
