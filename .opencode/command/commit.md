---
description: git commit with conventional message
---

Commit changes following repo conventions.

## Conventions

- Prefix commit messages with scope: `server:`, `cache:`, `kernel:`, `model:`, `test:`, `docs:`, `fix:`
- Explain WHY from a user perspective, not WHAT was changed
- Be specific — no generic messages like "improved performance"
- If there are merge conflicts, DO NOT fix them — notify the user

## Steps

1. Run `git status` and `git diff` to see all changes
2. Run `git log --oneline -5` to match recent message style
3. Stage relevant files
4. Commit with a scoped message
5. Run `git status` to verify
