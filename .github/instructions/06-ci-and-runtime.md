# CI And Runtime Operations

## Workflow reliability patterns
- Prefer explicit, minimal shell for secrets decoding and validation in CI steps.
- Add preflight checks close to the failing boundary:
  - decode/format validation for secret material,
  - crypto-level validation for signing assets,
  - clear non-zero exits with actionable messages.
- Keep action versions current to stay ahead of runtime deprecations.

## Secret handling patterns
- Use file-based secret ingestion where possible to avoid newline and padding corruption.
- Validate base64 payload length and decode success before downstream steps.
- Rotate secrets when provenance or integrity is uncertain.

## Runtime verification after CI green
- Run lightweight startup checks in addition to build checks.
- Prioritize user-visible flows likely to regress first:
  - first launch/home render,
  - settings toggles and persistence,
  - assessment results rendering.
- If optional subsystems fail (charts/media), degrade gracefully with text fallback instead of crashing.
