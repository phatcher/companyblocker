# CLI Conventions

## System Targeting

- All pipeline scripts must accept `--systems` as the only targeting flag.
- Do not add `--system` aliases in new scripts.
- `--systems` must support both formats through existing normalization logic:
  - Space-separated values (preferred), for example: `--systems gb fr ie`
  - Comma-separated values, for example: `--systems gb,fr,ie`

## Progress Output

- Multi-system operations should emit `[info]` progress messages at start, per-system, and completion.

## Consistency Rule

- When adding or modifying scripts, keep argument names and behaviour aligned with existing pipeline CLIs.
- Update README examples in the same change when CLI behaviour changes.

## Metadata-Driven Development

- Prefer metadata-first designs for system-specific behaviour.
- Model system differences in catalog metadata (for example under `src/acquisition/catalog/systems/*.json`) and resolve them through the registry/planning flow.
- Do not add new hardcoded per-system behaviour in code when metadata can represent the same intent.
- If metadata support does not yet exist and a code-based fallback is proposed, get explicit user approval before implementing the code fallback.
