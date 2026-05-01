# Server Development

This skill covers conventions for working on the mlx-lm server with TurboQuant integration.

## Server Architecture

The server is in `mlx_lm/server.py` (~1500 lines). Key classes:

- `ResponseGenerator` — handles generation loop, cache management
- `APIHandler` — HTTP request handling, OpenAI-compatible API
- `ModelProvider` — model loading, cache initialization

## TurboQuant Server Integration

The server supports KV cache quantization through these mechanisms:

1. **`_maybe_quantize_cache`** — converts FP16 cache to quantized after prefill
2. **`_maybe_dequantize_cache`** — reverses quantization for serialization
3. **Disk cache** — persists prompt caches to disk via `DiskBackedPromptCache`

CLI flags:
- `--kv-cache-quantization K,V` — quantize KV cache (e.g. `8,4`)
- `--quantized-kv-start N` — only quantize caches with at least N tokens
- `--prompt-cache-dir PATH` — persist prompt caches to disk
- `--turbo-kv-bits N` — TurboQuant bit width
- `--turbo-fp16-layers N` — keep first N layers in FP16

## Common Server Bugs

1. **Cache state serialization** — ensure `state` and `meta_state` properties roundtrip correctly
2. **GQA models** — verify `n_rep` is passed correctly to quantized attention
3. **MoE models** — handle `CacheList` for models with multiple expert caches
4. **Disk cache** — handle empty arrays (MoE sub-caches) via `empty.json`

## Testing Server Changes

1. Run `python -m unittest tests.test_server -v` for server-specific tests
2. Run `python -m unittest tests.test_turboquant -v` for TurboQuant tests
3. Test with a real model: `mlx_lm.server --model <model> --kv-cache-quantization 8,4`

## Upstream Sync

When syncing with upstream mlx-lm:
1. Key integration points: `cache.py`, `base.py`, `generate.py`, `server.py`
2. Check for API changes in `mlx_lm.models.cache` and `mlx_lm.models.base`
3. The `feature/turboquant-kv-cache` branch diverges from upstream `main`
