# Validation Gates

Required checks after behavior changes:
- `poetry run ruff check momentum/ tests/ mobile/main.py`
- `poetry run mypy momentum/`
- `poetry run pytest tests/ -v`

When user-visible behavior/rationale changes:
- Update `README.md` and `SCIENCE.md`
- Keep root/mobile documentation copies aligned when shared behavior is described
