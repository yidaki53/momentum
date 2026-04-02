# Momentum agent instructions

## Hard rule
- Never commit or push changes.

## Command execution
- Use Poetry-first Python commands (`poetry run ...`) for linting, typing, tests, scripts, and app entry points.

## Instruction assets index
- Core rules: `.github/instructions/01-core-rules.md`
- Architecture and Cython learnings: `.github/instructions/02-architecture-and-performance.md`
- Release and versioning policy: `.github/instructions/03-release-and-versioning.md`
- Android release safety: `.github/instructions/04-android-release-safety.md`
- Validation gates: `.github/instructions/05-validation-gates.md`
- CI and runtime operations: `.github/instructions/06-ci-and-runtime.md`
- Runtime diagnostics playbook: `.github/instructions/07-runtime-diagnostics.md`

## Prompt assets index
- Feature workflow: `.github/prompts/feature-workflow.prompt.md`
- Cross-surface parity: `.github/prompts/cross-surface-parity.prompt.md`

## GitHub workflow notes
- For this repository, use the `yidaki53` GitHub account for `gh` and git-authenticated GitHub operations.
- If a push updates `.github/workflows/*`, ensure the active token has `workflow` scope before pushing.
- If push is rejected with OAuth workflow-scope errors, refresh token scope before retrying.
