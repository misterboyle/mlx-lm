# Project Learnings

mlx-lm-turbo specific learnings captured from implementation and experimentation.

---

## KV Cache Memory Behavior / 2026-05-08

### Context
Investigated prompt cache behavior with Qwen3.6-27B-8bit on Apple Silicon (M5 Max, 128GB unified memory) using TurboQuant 3-bit KV compression. Tested with two concurrent conversations (A: ~90K tokens, B: ~12K tokens).

### Key Findings

**1. Byte cap is post-batch trim, not hard limit**
- `--prompt-cache-bytes` is NOT enforced per-insert
- The cache grows freely during compute, then gets trimmed AFTER each batch via `trim_to(n_bytes=total - active)`
- This causes oscillation: cache grows during compute → trimmed after batch → repeat
- Without the byte cap, cache grows monotonically (only bounded by `--prompt-cache-size`)

**2. Log reports compressed sizes, but they're still large**
- `TurboQuantKVCache.nbytes` sums compressed buffer sizes (uint32 bit-packed + float32 norms)
- The log IS reporting compressed sizes
- Qwen3.6-27B has 16 full attention layers (scale with KV) + 48 linear attention layers (fixed overhead)
- Only the 16 full attention layers contribute to per-token cache cost
- 3-bit TurboQuant compresses those 16 layers, but 47 GB cache for ~100K tokens suggests ~470 KB/token
- **TODO**: Verify per-token cache cost calculation - may be missing overhead or miscounting layers

**3. Metal memory is elastic, not static**
- Process RSS stays flat (~27 GB) regardless of cache size
- System wired memory spikes during compute (80-100 GB), drops at rest (4-80 GB)
- Metal unpins GPU buffers when idle, re-pins during compute
- The cache is LOGICAL, not PHYSICAL — entries exist but memory is reclaimed

**4. Sequence count vs byte cap tradeoff**
- With 20 GB byte cap: cache oscillated 16-30 GB, sequences jumped 4-12, constant eviction
- Without byte cap: cache hit 12 sequences and grew to 47 GB, no byte-driven eviction
- The byte cap was doing real work: evicting enough entries to keep cache bounded
- 20 GB cap ≈ 39K tokens cached (too small for 90K conversations)

**5. True memory bound with 20 GB cap**
- The cap was effective but too aggressive for long conversations
- Cache would temporarily exceed 20 GB during compute, then get trimmed
- This caused frequent cache misses on long conversations

### Recommendations

- **Raise byte cap to 40-50 GB** to accommodate both conversations (90K + 12K ≈ 100K tokens × 512 KB/token ≈ 51 GB)
- **Monitor wired memory** (not RSS) for real GPU memory usage
- **Consider per-type caps** (assistant/user/system) for finer control
- **Document the elastic Metal memory behavior** — cache is logical, not physical

### Open Questions

- What's the optimal byte cap for 128GB unified memory?
- Should we implement per-type byte caps?
- How does TurboQuant compression scale with different bit widths (1-4 bits)?
- Can we improve the post-batch trim to be more predictive?
