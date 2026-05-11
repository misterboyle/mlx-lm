# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

- The default branch is `feature/turboquant-kv-cache` (30 commits with all TurboQuant work). `main` only has 2 commits and is not the active branch.
- Prefer automation: execute requested actions without confirmation unless blocked by missing info or safety/irreversibility.

## Development Setup

Use a Python 3.14+ venv in the repo root:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -e ".[test]"
```

The venv directory (`venv/`) is gitignored. Always activate before running tests or the server.

## Style Guide

### General Principles

- No docstrings on private helpers (`_method`). Docstrings only on public classes and functions.
- No type annotations on private helper parameters — only on public APIs.
- Use `mx` as the alias for `mlx.core` (universal across all modules).
- Use `nn` as the alias for `mlx.nn`.
- Keep Metal kernel strings as raw strings (`"""..."""`) with minimal comments inside.
- Prefer early returns over nested conditionals.
- Use `float | None` for optional floats, not `Optional[float]`.
- Use lowercase tuple types (`tuple[mx.array, mx.array]`), not `Tuple[...]`.
- Code must pass `black` and `isort` formatting (enforced by pre-commit hooks).

### Imports

Order (strict, observed across all modules):
1. `__future__` (if needed, e.g. `from __future__ import annotations`)
2. stdlib (sorted: `argparse`, `copy`, `json`, `math`, `os`, `time`, etc.)
3. `import mlx.core as mx`
4. `import mlx.nn as nn` (when needed)
5. `from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten`
6. third-party (`numpy`, `transformers`, `huggingface_hub`, `requests`)
7. local (`mlx_lm.*`), using relative imports within packages (`from .cache import ...`)

### Naming

- `snake_case` for functions and variables.
- `PascalCase` for classes.
- Prefix private helpers with `_` (functions, classes, module-level singletons).
- Dimension variables: `B, H, S, D` for batch/heads/sequence/dimension in cache code.
- Attention dims: `n_heads`, `n_q_heads`, `n_kv_heads`, `n_rep`.
- Packed storage: `k_packed`, `v_packed`, `k_norms`, `v_norms`, `p_dim`, `vpw`.
- Pre-rotated query: `q_rot`.

### Metal Kernels

- Kernel source lives in Python strings passed to `mx.fast.metal_kernel()`.
- Use `template` parameters for compile-time types: `template=[("T", mx.float32)]`.
- Kernels that don't use template types pass `template=[]`.
- Keep kernels self-contained — no cross-kernel dependencies.
- Lazy singleton pattern: module-level `_kernel = None`, initialized on first call.
- Thread layout: `grid=(n_vecs * dim, 1, 1)`, `threadgroup=(dim, 1, 1)` — one threadgroup per vector.
- Common patterns: WHT butterfly (`while (h < dim)`), bit unpacking (`word >> (pos * bits) & mask`), `simd_sum` reductions, GQA via `kv_head = head / n_rep`.
- When adding a new kernel, add a corresponding test in `tests/test_turboquant.py`.

## Testing

- Run `python -m unittest discover tests/ -v` from the repo root (venv must be activated).
- TurboQuant tests are in `tests/test_turboquant.py` (668 lines, 8 test classes).
- Tests use `unittest.TestCase` — NOT pytest-style.
- Test model: `mlx-community/Qwen1.5-0.5B-Chat-4bit` (loaded once in `setUpClass`).
- Tests print diagnostic output with `print()` for debugging; this is fine.
- Correctness is verified against reference implementations:
  - WHT invertibility: `atol=1e-5`
  - Cosine similarity: `> 0.85` for 3-bit quantization
  - Compression ratio: `> 3.0x` for 3-bit
  - Roundtrip: `mx.array_equal` after pack/unpack
- Always `mx.eval()` before comparing results (MLX is lazy).
- When adding a feature, add at least one test that verifies correctness against a reference.

## Formatting

- Run `pre-commit run --all-files` to check formatting (black + isort).
- Can also run manually: `black file.py` and `isort --profile=black file.py`.
- CI runs pre-commit hooks on pull requests.

## Running Server

- Server entry point: `mlx_lm.server` (console script: `mlx_lm.server`).
- TurboQuant server flags:
  - `--kv-cache-quantization K,V` — quantize KV cache (e.g. `8,4`)
  - `--quantized-kv-start N` — only quantize caches with at least N tokens
  - `--prompt-cache-dir PATH` — persist prompt caches to disk
  - `--turbo-kv-bits N` — TurboQuant bit width
  - `--turbo-fp16-layers N` — keep first N layers in FP16
- Server tests in `tests/test_server.py` use `DummyModelProvider` mock + `http.server.HTTPServer`.

## Package

- Installed editable: `pip install -e .`.
- Hard dependencies: `mlx>=0.30.4`, `numpy`, `transformers>=5.0.0`, `sentencepiece`, `protobuf`, `pyyaml`, `jinja2`.
- Python 3.8+ required.
- 17 console scripts (see setup.py entry_points).

## TurboQuant Integration

The TurboQuant-specific files in this repo:

| File | Purpose |
|------|---------|
| `mlx_lm/models/turboquant_cache.py` | TurboQuantKVCache — PolarQuant KV compression |
| `mlx_lm/models/turboquant_rotation.py` | Walsh-Hadamard Transform + random diagonal |
| `mlx_lm/models/turboquant_metal.py` | Fused Metal quantize/dequantize kernels |
| `mlx_lm/models/turboquant_packing.py` | Bit-packing indices into uint32 words |
| `mlx_lm/models/turboquant_kernels.py` | Packed dequant + fused Q@K^T Metal kernels |
| `mlx_lm/models/mixed_quant_cache.py` | MixedQuantKVCache — K@8-bit, V@4-bit |
| `mlx_lm/disk_cache.py` | DiskBackedPromptCache — disk-persisted LRU |

Modified upstream files:

| File | Changes |
|------|---------|
| `mlx_lm/models/cache.py` | Added `KVCache.to_turbo_quantized()`, TurboQuant/MixedQuant to allowlists |
| `mlx_lm/models/base.py` | Added `mixed_quantized_scaled_dot_product_attention()`, auto-routing |
| `mlx_lm/generate.py` | Added `--turbo-kv-bits`, `--turbo-fp16-layers` CLI args |
| `mlx_lm/server.py` | Added KV quantization, disk cache, CLI args |

The `turboquant-mlx` library is the upstream source for the Metal kernels and quantization algorithms. Changes to kernels should be coordinated with that repo.
