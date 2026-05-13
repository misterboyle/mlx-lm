# Handoff: Memory Profiling Task

## Completed

- Created epic `hq-lq1` (System characterization) and task `hq-lq1.1` (Memory profiling)
- Task properly linked as child of epic via `--parent` flag
- Updated task with server configuration details
- Model quality test skill updated to use `/metrics` endpoint instead of log parsing

## Server Configuration

- **TurboQuant**: 3-bit KV cache
- **FP16 layers**: 1 (default)
- **Mode**: single-mode
- **Model**: Qwen3.6-35B-A3B-UD-MLX-4bit

## Server Start Command

```bash
pkill -f mlx_lm.server
sleep 2
PYTHONUNBUFFERED=1 nohup python -m mlx_lm.server \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 0.0.0.0 --port 8080 \
  --turbo-kv-bits 3 \
  --single-mode \
  > /tmp/mlx_lm_server.log 2>&1 &
```

## What to Do Next

1. **Claim the task**: `bd update hq-lq1.1 --claim`
2. **Start the server** on the remote Mac with the config above
3. **Run a multi-turn conversation** (4-8 turns) using opencode
4. **Capture `/metrics` output** after each turn:
   ```bash
   curl -s http://172.16.49.25:8080/metrics | python3 -m json.tool
   ```
5. **Record memory data** after each turn:
   - `mlx.active_memory_bytes` — total MLX memory
   - `prompt_cache.total_bytes` — cache size
   - `prompt_cache.total_sequences` — number of cached sequences
   - `prompt_cache.by_type` — breakdown by type
6. **Verify cache growth is linear** — no duplicate inserts from the segment cache bug
7. **Document results** in the task notes

## Key Files

- `.opencode/skills/model-quality-test/SKILL.md` — Updated to use `/metrics` endpoint
- `mlx_lm/server.py` — Fixed segment cache duplicate insert bug (lines 1082-1094)

## Notes

- The segment cache bug was fixed: full cache save moved outside the segment loop to prevent duplicate inserts
- The `/metrics` endpoint is working and returns structured JSON
- Use opencode with `-c` flag for continuation turns
- NO `-c` for first turn (clean slate)
