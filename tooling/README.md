# Tooling

`tooling/` holds the scripts that serve the repository itself rather than any stage of the company-matching pipeline: the gates that enforce a repo-wide invariant and the generators that keep checked-in artifacts current. A script belongs here when it reads source, git state or build artifacts, never `data/`, and no pipeline area owns it; a gate that checks one area's work stays with that area, and `scripts/README.md`'s "Repo gates" section lists those.

## The tools

| Script | Purpose |
| --- | --- |
| `check_package_docs.py` | Verify each `packages/*/README.md` is self-contained and documents its public API. Pre-push hook. |
| `baseline.py` | Compare a ratchet test's measurement against its `src/tests/baselines/<name>.txt`: fail on a new offender, warn on a stale line. |
| `check_direct_tests.py` | List the modules no test imports directly, per `docs/TESTSTYLE.md`'s direct-test rule; `src/tests/tooling/scripts/test_direct_tests.py` pins the count. |
| `check_test_tier_markers.py` | List every test that matches a broad-tier behavioural signature but carries no `@pytest.mark.integration` or `@pytest.mark.performance`, per `docs/TESTSTYLE.md`'s unit-tier rule; `src/tests/tooling/scripts/test_check_test_tier_markers.py` pins the count against `src/tests/baselines/test_tier_markers.txt`. Doubles as the collection plugin that produces the list: a run adding `-p check_test_tier_markers` lets the ratchet read that run's own collection instead of starting a second one, and one without it falls back to the subprocess, correct and slower. It marks and deselects nothing. |
| `test_tier_classifier.py` | The rule `check_test_tier_markers.py` applies: given a test's source and its fixtures', say whether it starts a subprocess, materialises a layer or reads the real corpus. A library, not a command. |
| `design_artifact_status.py` | Report whether the pyscn and graphify design artifacts are stale against current `HEAD`. Pre-push hook. |
| `update_design_artifacts.py` | Regenerate the pyscn and graphify design artifacts. |
| `export_diagrams.py` | Export every diagram in the repo to a PNG under `docs/diagrams/png/`. |
| `sync_palette.py` | Write the diagram palette and fonts from `docs/diagrams/house-style.css` into the PlantUML and Structurizr include files, which neither toolchain reads from CSS; `export_diagrams.py` runs it before every export. |
| `start_structurizr.py` | Run the local Structurizr Lite workspace with Docker Compose. |
| `_tooling_common.py` | What the tools share: the repository-root derivation, walking to the nearest `pyproject.toml`, kept here so no tool imports the code it polices. A library, not a command. |

## Imports

`tooling/` is a source root, declared in `pytest.ini`'s `pythonpath` and in `tach.toml`'s `source_roots`, and it has no `__init__.py`. Each script is a top-level module, so a tool importing a sibling writes `from test_tier_classifier import classify_callable`, and a test writes the same, since `pytest.ini`'s `pythonpath` already holds `tooling/`. It is deliberately not nested under `scripts/`, which is a package: a source root inside a package makes every module reachable two ways, which is the ambiguity `tach.toml`'s comment records as having broken test selection repo-wide.

## Tests

An area's tests live under `src/tests/<area>/`, so this area's live under `src/tests/tooling/`, and a script's test under `src/tests/<area>/scripts/` where the area owns the entry point the script runs. `docs/TESTSTYLE.md` states the rule and is the authority. `src/tests/tooling/scripts/` also holds the tests of the repo gate and shared tooling scripts and any test whose subject is consistency across several areas' scripts, since those drive no other area. A script test's path states its owner rather than leaving it to be re-derived; there is no shared script-test directory.

## Running a tool

Every Python script here runs through the shared venv:

```
uv run --no-sync python tooling/check_direct_tests.py --help
```

`--no-sync` is not optional in a worktree: a plain `uv run` re-syncs the shared `.venv` and repoints every editable install at whichever directory invoked it.
