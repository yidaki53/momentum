# Core Rules

- Never commit or push changes.
- Use Poetry-first Python commands (`poetry run ...`) for linting, typing, tests, scripts, and app entry points.
- Keep architecture boundaries clear:
  - Domain logic in `momentum/domain/`.
  - UI/presentation helpers in `momentum/ui/`.
  - Persistence/query mapping in `momentum/db.py`.
  - Shared domain models in `momentum/models.py`.
- Preserve backward compatibility where feasible (for example, thin re-export shims for moved modules).
