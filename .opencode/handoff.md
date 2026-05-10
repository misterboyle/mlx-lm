# Handoff: BatchTurboQuantKVCache Interface Complete

## Session Identity: Vidad

## Completed

### This session
- **Wired up MixedQuantKVCache server integration** — `--kv-cache-quantization K,V` CLI flag for mixed-precision KV cache (K@8-bit, V@4-bit via Apple's native `mx.quantized_matmul`)
  - Added CLI args: `--kv-cache-quantization`, `--kv-group-size`, `--quantized-kv-start`, `--kv-cache-quantize-after-prefill`
  - Wired `kv_bits` in `ResponseGenerator.__init__`, single-serve path, and batch path
  - Added MixedQuant support to `make_prompt_cache()` with per-layer compatibility detection for hybrid/MoE models
  - Added `--kv-cache-quantize-after-prefill` flag to toggle between:
    - Default: quantized from start (lower memory)
    - Flag: FP16 prefill → post-prefill conversion (better quality)
  - Added `BatchMixedQuantKVCache` class for continuous batching support (modeled after `BatchTurboQuantKVCache`)
  - Added 9 new tests in `TestMixedQuantKVCache` class
- **All 82 turboquant tests pass, 20 server tests pass, 9 MixedQuant tests pass**

### Previous session (committed)
- Fixed `make_prompt_cache` for hybrid/MoE models with TurboQuant
- Fixed `BatchGenerator` to accept and pass turbo params
- Fixed `server.py` to pass turbo params
- Added `BatchTurboQuantKVCache.extract()` and `to_batch_cache` handler
- 61 turboquant tests pass (before this session's additions)

### Session before that
- Added 4 missing batch methods to `BatchTurboQuantKVCache` — now has full interface parity with `BatchKVCache`
  - `filter(batch_indices)`, `prepare(right_padding=...)`, `finalize()`, `extend(other)`
- Added `__getstate__`/`__setstate__` to both `TurboQuantKVCache` and `BatchTurboQuantKVCache`
- Added 10 new tests (12 total in `TestTurboQuantDeepCopy` class)
- 73/73 turboquant tests pass, 22/22 prompt cache tests pass

## In Progress

None. All work complete.

## Pending

- **Test with real server restart** — user will restart the server and verify generation works
- **Build `BatchQuantizedKVCache`** — the generic affine-quantized batch cache that `QuantizedKVCache` is missing (not done yet, but identified as a gap)
- **MixedQuantKVCache server integration** — wired up `--kv-cache-quantization 8,4` in server.py, make_prompt_cache(), and generate.py. Added `BatchMixedQuantKVCache` for continuous batching support. **TODO: check upstream MixedQuantKVCache to see if it already had batching support we missed, and model `BatchMixedQuantKVCache` off `BatchTurboQuantKVCache`** (we just built it, but upstream may have had a different approach)
- **Test MixedQuantKVCache on dedicated MacBook** — SSH to `michael@172.16.49.25` (password: `michael`). 48GB RAM, 16 cores, macOS. Repo at `~/mlx-lm-turbo` with venv already set up. Test both quantization modes:
  - `--kv-cache-quantization 8,4` (quantized from start, lower memory)
  - `--kv-cache-quantization 8,4 --kv-cache-quantize-after-prefill` (FP16 prefill → convert, better quality)
  - Compare with `--turbo-kv-bits 3` (TurboQuant, currently unstable)
  - Compare with `--turbo-kv-bits 3 --turbo-fp16-layers 2` (TurboQuant with wider FP16 guard)
- **Investigate TurboQuant instability** — hypothesis: TurboQuant may be unstable because we create `TurboQuantKVCache` upfront (quantized from start) rather than converting after prefill. The upstream MixedQuant approach was FP16 prefill → convert to MixedQuant after prefill. We should test `--turbo-kv-bits 3` with a post-prefill conversion pattern to see if that fixes the instability.

## Blockers

None.

## Test Environment

**Dedicated MacBook for testing:**
- SSH: `michael@172.16.49.25` (password: `michael`)
- Hardware: 48GB RAM, 16 cores, macOS
- Repo: `~/mlx-lm-turbo` (venv already set up)
- Models: `Qwen3.6-27B-UD-MLX-6bit`, `Qwen3.6-35B-A3B-UD-MLX-4bit` in `~/.localllm/models/`
- SSH key already authorized (added during session)

## Interface Parity Status

`BatchTurboQuantKVCache` now has all methods that `BatchKVCache` has:

| Method | `BatchKVCache` | `BatchTurboQuantKVCache` |
|--------|:-:|:-:|
| `update_and_fetch` | ✅ | ✅ |
| `state` / `state=` | ✅ | ✅ |
| `meta_state` / `meta_state=` | ✅ | ✅ |
| `is_trimmable` | ✅ | ✅ |
| `trim` | ✅ | ✅ |
| `make_mask` | ✅ | ✅ |
| `empty` | ✅ | ✅ |
| `nbytes` | ✅ | ✅ |
| `size` | ✅ | ✅ |
| `merge` (class) | ✅ | ✅ |
| `extract` | ✅ | ✅ |
| `filter` | ✅ | ✅ |
| `prepare` | ✅ | ✅ |
| `finalize` | ✅ | ✅ |
| `extend` | ✅ | ✅ |

`BatchMixedQuantKVCache` now has all methods that `BatchKVCache` has:

| Method | `BatchKVCache` | `BatchMixedQuantKVCache` |
|--------|:-:|:-:|
| `update_and_fetch` | ✅ | ✅ |
| `state` / `state=` | ✅ | ✅ |
| `meta_state` / `meta_state=` | ✅ | ✅ |
| `is_trimmable` | ✅ | ✅ |
| `make_mask` | ✅ | ✅ |
| `empty` | ✅ | ✅ |
| `nbytes` | ✅ | ✅ |
| `size` | ✅ | ✅ |
| `merge` (class) | ✅ | ✅ |

Note: `BatchMixedQuantKVCache` does NOT implement `trim`, `filter`, `prepare`, `finalize`, `extend`, or `extract` — these are not yet needed for MixedQuant but could be added later if continuous batching with MixedQuant is needed.

## Test Coverage

Each new method has comparable coverage to what `BatchKVCache` tests have:

- **filter**: basic filter, left-shift, empty result, quantization preservation (4 tests)
- **prepare**: stores right-padding (1 test)
- **finalize**: with right padding, without right padding, roundtrip with prepare (3 tests)
- **extend**: basic extend, extend with empty caches (2 tests)
- **deepcopy**: per-sequence cache, batched cache, independent buffers (3 tests)

## Files Changed

| File | Changes | Description |
|------|---------|-------------|
| `mlx_lm/models/mixed_quant_cache.py` | +206 | `BatchMixedQuantKVCache` class for continuous batching |
| `mlx_lm/server.py` | +60 | CLI args + ResponseGenerator wiring for MixedQuant |
| `mlx_lm/models/cache.py` | +80 | `make_prompt_cache()` MixedQuant support |
| `mlx_lm/generate.py` | +30 | `BatchGenerator` + `generate_step()` MixedQuant params |
| `tests/test_turboquant.py` | +120 | 9 new `TestMixedQuantKVCache` tests |
| `mlx_lm/models/turboquant_cache.py` | +302 | 4 new batch methods + `__getstate__`/`__setstate__` (previous session) |
| `tests/test_turboquant.py` | +373 | 10 new tests (previous session) |

## Commit Commands (if needed)

```bash
cd /Users/michael/mlx-lm-turbo
git log --oneline -3
```
