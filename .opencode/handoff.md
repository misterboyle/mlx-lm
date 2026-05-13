# Handoff: Memory profiling complete, hq-lq1.5 comparison done

## Completed

- **hq-lq1.1** (Memory profiling) — CLOSED. Both Session A and B tested with 3-bit TurboQuant.
- **hq-lq1.5** (TQ3 vs uncompressed comparison) — Done. Comparison table in task notes.
- **hq-fka-1** (Debug logging) — Merged to dev. Memory logging in generate.py and signal handlers in server.py.
- **hq-server-ready-log** — Merged to dev. Server logs "Server is ready to accept connections."

## Open Tasks (in hq-lq1 epic)

| Task | Description |
|------|-------------|
| **hq-lq1.2** | Model quality testing in batch mode with TurboQuant (depends on batch mode support) |
| **hq-lq1.3** | Memory profiling with fused FA kernel (hq-0f9) |
| **hq-lq1.4** | High-frequency metrics polling for transient spike detection |

## Key Findings from Comparison

- **Cache compression: ~2.3x** — TQ3 cache is 2.3x smaller at turn 8 (2676 MB vs 6172 MB)
- **Peak memory: similar** — TQ3 peaks at 27.73 GB, uncompressed at 27.44 GB (within 0.3 GB)
- **Transient spikes: similar** — TQ3 spike 4.11 GB, uncompressed 4.41 GB (within 0.3 GB)
- **Context retention: both work** — both correctly recall details from turn 5

**Bottom line:** TurboQuant saves disk/RAM cache space but doesn't reduce GPU peak memory. The benefit is allowing larger contexts to fit in available RAM before hitting disk swap.

## What to Do Next

1. Pick up hq-lq1.2, hq-lq1.3, or hq-lq1.4
2. hq-lq1.4 (high-freq polling) can be done independently — run the metrics endpoint every 100-500ms during generation to capture spike shape
3. hq-lq1.3 depends on hq-0f9 (fused FA kernel) being implemented first
4. hq-lq1.2 depends on batch mode TurboQuant support

## Server Config (if needed)

```bash
ssh michael@172.16.49.25
pkill -f mlx_lm.server
sleep 2
cd ~/mlx-lm-turbo && git pull origin dev
> /tmp/mlx_lm_server.log
source venv/bin/activate
PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 0.0.0.0 --port 8080 \
  --turbo-kv-bits 3 --single-mode \
  >> /tmp/mlx_lm_server.log 2>&1 &
```

## Key Files

- `.opencode/skills/model-quality-test/SKILL.md` — Updated metrics capture (includes peak_memory)
- `mlx_lm/server.py` — Added readiness log, signal handlers, memory logging
- `mlx_lm/generate.py` — Added memory logging at key points
