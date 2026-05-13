# Project Learnings

## Git Workflow (2025-05-13)

Established clean branch model after slopy branch tracking caused confusion:

- **Working branches** (`dev`, `hq-<bead-id>`) always track `origin/` — never upstream/official remotes directly
- **Read-only mirrors** (`upstream/feature/turboquant-kv-cache`, `official/main`) are local branches that track their respective remotes for inspection only
- Feature branches are named after beads: `hq-554-baseline`, `hq-9qj`, etc.
- `dev` is the main development branch; `main` is the release branch
- Rebase upstream changes onto `dev`, not `main`
- Releases: merge `dev` into `main` with `--no-ff`, tag with semver
