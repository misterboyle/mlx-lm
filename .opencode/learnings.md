# Project Learnings

## Git Workflow (2025-05-13)

Established clean branch model after slopy branch tracking caused confusion:

- **Working branches** (`dev`, `hq-<bead-id>`) always track `origin/` — never upstream/official remotes directly
- **Read-only mirrors** (`upstream/feature/turboquant-kv-cache`, `official/main`) are local branches that track their respective remotes for inspection only
- Feature branches are named after beads: `hq-554-baseline`, `hq-9qj`, etc.
- `dev` is the main development branch; `main` is the release branch
- Rebase upstream changes onto `dev`, not `main`
- Releases: merge `dev` into `main` with `--no-ff`, tag with semver

## Memory Profiling with /metrics (2026-05-13)

### Key Findings

- **Peak memory is cumulative** — `mlx.peak_memory_bytes` is a running maximum since startup, never decreases. Use it to track whether peaks are growing across sessions, not to see "current peak."
- **Transient spikes are real** — ~4 GB above steady-state during generation (dequant buffers, attention temp storage, output buffers). Captured via `peak - active` at end of session.
- **Active memory drops after generation** — temporary buffers are freed, so active memory returns to steady-state. This is normal.

### Metrics Capture Protocol

Always capture these at each step:
```bash
curl -s http://172.16.49.25:8080/metrics | python3 -c "
import json,sys; m=json.load(sys.stdin)
a=m['mlx']['active_memory_bytes']; p=m['mlx']['peak_memory_bytes']
print(f'Active: {a/1e9:.2f} GB')
print(f'Peak:   {p/1e9:.2f} GB')
print(f'Spike:  {(p-a)/1e9:.2f} GB')
"
```

### TurboQuant vs Uncompressed (2026-05-13)

- **Cache compression: ~2.3x** — TQ3 cache is 2.3x smaller than uncompressed at same conversation depth
- **Peak memory: similar** — TQ3 and uncompressed have nearly identical peak memory (~27.7 GB vs 27.4 GB)
- **Transient spikes: similar** — TQ3 spike 4.11 GB, uncompressed 4.41 GB (within 0.3 GB)
- **TQ3 saves RAM/disk cache space, not GPU memory** — the KV cache is stored in RAM, not GPU memory. Peak GPU memory is driven by transient allocations during generation (dequant buffers, attention temp storage), which are similar for both configs.
- **The real benefit of TQ3:** allows larger contexts to fit in available RAM before hitting disk swap. For 50GB+ caches, this matters.

### What Reduces Memory

- **Transient spike reduction:** hq-0f9 (fused FA kernel) should reduce the ~4 GB spike by eliminating temporary fp16 dequant buffers during attention computation
- **Steady-state reduction:** cache eviction, fewer cached sequences, smaller model
- **hq-0f9 won't reduce steady-state active memory** — it only reduces the spike above steady-state
