---
description: run the full test suite
---

Run tests and report results.

```bash
python -m unittest discover tests/ -v
```

All tests should pass. If any fail, report the failures and suggest fixes.

For TurboQuant-specific tests only:

```bash
python -m unittest tests.test_turboquant -v
```
