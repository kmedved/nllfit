# Contributing

## Development setup

```bash
pip install -e .[dev]
pytest
ruff check .
ruff format .
mypy hetero_nll
```

## Design principles

- Keep distributional outputs explicit: mu/var/log_var.
- Prefer numerically-stable defaults (variance clipping, eps floors).
- Prefer data-leakage-safe training recipes by default (OOF residuals).
- Keep optional dependencies optional (`glum`, `lightgbm`).
