# Handoff: Memory profiling complete, epic hq-lq1 has 3 pending tasks

## Completed

- **hq-lq1.1** (Memory profiling) — CLOSED. Both Session A (story) and Session B (coding) tested with 3-bit TurboQuant.
  - Peak memory: 27.92 GB (4.2 GB transient spike above steady-state 23.68 GB)
  - Cache growth linear, no duplicate inserts
  - Context retention verified across all turns
- **hq-server-ready-log** — Merged to dev. Server now logs "Server is ready to accept connections."
- **model-quality-test skill** — Updated capture script to include `peak_memory_bytes` and spike calculation

## Open Tasks (in hq-lq1 epic)

| Task | Description |
|------|-------------|
| **hq-lq1.2** | Model quality testing in batch mode with TurboQuant (depends on batch mode support) |
| **hq-lq1.3** | Memory profiling with fused FA kernel (hq-0f9) |
| **hq-lq1.4** | High-frequency metrics polling for transient spike detection |

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

- `.opencode/skills/model-quality-test/SKILL.md` — Updated metrics capture
- `mlx_lm/server.py` — Added readiness log message
