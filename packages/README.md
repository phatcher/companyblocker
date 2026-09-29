# packages/

Each subdirectory here is a standalone, independently-versionable library: its own `pyproject.toml`, no dependency on this repo's `src/`, consumable from an unrelated project. `company_cleanse`, `company_tokenize`, `company_vectorize`, `company_classify`, `company_perturbation` and `company_resolvers` are the six today. Each has its own `README.md` describing what it does; this file covers the one rule that binds all of them.

## Input contract

A package function that reads real, pre-existing files takes one of three things, and never a fourth:

- **A root path**, when the package owns deciding what lives under it (a directory the package itself wrote, for instance).
- **A root path plus a file pattern**, when the caller names which files under the root it means (an exact filename, or a glob).
- **An already-resolved file list**, when a caller that knows its own storage layout has done the selecting and hands over the result.

What it must never do is guess a directory's shape on its own, whether by defaulting to "every file under this root" with no pattern given, or by importing this monorepo's `src/workspace` package to ask it. Both make the package's correctness depend on a workspace layout that does not exist once the package is used standalone. `company_tokenize.TokenizeFilesRequest.source_files` is the already-resolved-list shape this repo's own pipeline uses (`scripts/pipeline_runner.py` resolves it via `src/workspace` and passes the result in); `TokenizeFilesRequest.input_file` is the root-plus-pattern shape for a caller that only knows a filename or glob under `cleansed_dir`.

`tooling/check_package_docs.py` enforces this: it fails the gate if any file under a package's own `src/` imports `workspace`, or hand-builds a `jurisdiction_code=`-style partition path. Both are `src/workspace`-owned facts about this monorepo's on-disk layout, not something a standalone package may reproduce or depend on.
