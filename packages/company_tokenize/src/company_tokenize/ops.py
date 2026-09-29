"""Tokenize named parquet files, with one tokenizer, a dual pair or several in one pass.

These helpers do not scan a directory. A caller gives either `source_files`, paths it
has already resolved, or `input_file`, a filename or glob matched under
`cleansed_dir`; passing neither raises. Which files under a root are the input is a
fact about the workspace that owns them, so it stays the caller's.
"""

from __future__ import annotations

import gc
import shutil
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .manifest import validate_dual_tokenizer_artifacts
from .tokenization import tokenize_name_dataframe_with_request
from .tokenization_params import (
    TokenizationRequest,
    TokenizerCalculationSpec,
    coerce_tokenization_request,
    compile_tokenization_request,
)

TOKENIZED_OUTPUT_COLUMNS = (
    "name_cleansed",
    "country_tokens",
    "global_tokens",
)


@dataclass(frozen=True)
class TokenizeFilesRequest:
    cleansed_dir: Path
    tokenized_dir: Path
    tokenization: TokenizationRequest
    input_file: str | None = None
    source_files: tuple[Path, ...] | None = None
    """Inputs to tokenize, already resolved by the caller.

    Which files under a root count as a system's real data is a fact about
    the workspace that owns them, not about tokenization -- in this
    repository `src/workspace` answers it, and this package must stay
    consumable outside that repository. So a caller that knows its layout
    resolves and passes the list here; a caller that doesn't must instead
    give `input_file`, a filename or glob pattern applied under
    `cleansed_dir`. One of the two is required (see `packages/README.md`);
    this package never defaults to "every parquet under the root."
    """


def _validate_dual_tokenizer_paths(
    *,
    country_tokenizer_path: Path,
    global_tokenizer_path: Path,
    country_trainer: str,
    global_trainer: str,
) -> None:
    if not country_tokenizer_path.exists():
        raise FileNotFoundError(
            f"Country tokenizer not found: {country_tokenizer_path}. "
            "Provide a country tokenizer before running dual token mode."
        )
    if not global_tokenizer_path.exists():
        raise FileNotFoundError(
            f"Global tokenizer not found: {global_tokenizer_path}. "
            "Provide a global tokenizer before running dual token mode."
        )
    # Best-effort: only validates when both sides carry a run manifest (see
    # `validate_dual_tokenizer_artifacts`); artifacts without one proceed
    # unchecked, as they always have.
    validate_dual_tokenizer_artifacts(
        country_tokenizer_path=country_tokenizer_path,
        global_tokenizer_path=global_tokenizer_path,
        country_trainer=country_trainer,
        global_trainer=global_trainer,
    )


def _validate_multi_tokenizer_specs(
    *,
    tokenizer_specs: list[TokenizerCalculationSpec] | list[Mapping[str, object]],
) -> None:
    request = coerce_tokenization_request(tokenizer_specs=tokenizer_specs)
    compiled = compile_tokenization_request(request)

    for spec in compiled.tokenizer_calculations:
        if spec.tokenizer_path is None:
            raise ValueError("Each tokenizer spec must include tokenizer_path.")
        if not spec.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {spec.tokenizer_path}. Train or provide a tokenizer first."
            )


def tokenize_name_with_request(request: TokenizeFilesRequest) -> int:
    return _run_tokenization_from_cleansed(
        cleansed_dir=request.cleansed_dir,
        tokenized_dir=request.tokenized_dir,
        name_col=request.tokenization.name_col,
        input_file=request.input_file,
        source_files=request.source_files,
        transform_fn=lambda df: tokenize_name_dataframe_with_request(
            df,
            request=request.tokenization,
        ),
    )


def tokenize_name(
    cleansed_dir: Path,
    tokenized_dir: Path,
    tokenizer_path: Path | None = None,
    trainer: str = "wordpiece",
    name_col: str = "name_cleansed",
    token_col: str = "name_tokens",  # nosec B107 - a dataframe column name, not a credential
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str | None = None,
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
    input_file: str | None = None,
    source_files: tuple[Path, ...] | None = None,
) -> int:
    if tokenizer_path is not None and not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Train or provide a tokenizer first."
        )

    request = TokenizeFilesRequest(
        cleansed_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        tokenization=coerce_tokenization_request(
            tokenizer_specs=[
                TokenizerCalculationSpec(
                    token_col=token_col,
                    tokenizer_path=tokenizer_path,
                    trainer=trainer,
                )
            ],
            name_col=name_col,
            noise_words=noise_words,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            preprocess_profile=preprocess_profile,
        ),
        input_file=input_file,
        source_files=source_files,
    )
    return tokenize_name_with_request(request)


def _build_dual_tokenization_request(
    *,
    name_col: str,
    country_token_col: str,
    country_tokenizer_path: Path,
    country_trainer: str,
    global_token_col: str,
    global_tokenizer_path: Path,
    global_trainer: str,
    noise_words: set[str] | list[str] | tuple[str, ...] | None,
    noise_words_path: str | Path | None,
    noise_words_profile: str,
    noise_words_set_kind: str,
    trimmed_name_col: str | None,
    preprocess_profile: str | None,
) -> TokenizationRequest:
    return coerce_tokenization_request(
        tokenizer_specs=[
            TokenizerCalculationSpec(
                token_col=country_token_col,
                tokenizer_path=country_tokenizer_path,
                trainer=country_trainer,
            ),
            TokenizerCalculationSpec(
                token_col=global_token_col,
                tokenizer_path=global_tokenizer_path,
                trainer=global_trainer,
            ),
        ],
        name_col=name_col,
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        trimmed_name_col=trimmed_name_col,
        preprocess_profile=preprocess_profile,
    )


def tokenize_name_dual(
    cleansed_dir: Path,
    tokenized_dir: Path,
    *,
    country_tokenizer_path: Path,
    global_tokenizer_path: Path,
    country_trainer: str = "wordpiece",
    global_trainer: str = "wordpiece",
    name_col: str = "name_cleansed",
    country_token_col: str = "country_tokens",
    global_token_col: str = "global_tokens",
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str | None = None,
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
    input_file: str | None = None,
    source_files: tuple[Path, ...] | None = None,
) -> int:
    _validate_dual_tokenizer_paths(
        country_tokenizer_path=country_tokenizer_path,
        global_tokenizer_path=global_tokenizer_path,
        country_trainer=country_trainer,
        global_trainer=global_trainer,
    )

    request = TokenizeFilesRequest(
        cleansed_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        tokenization=_build_dual_tokenization_request(
            name_col=name_col,
            country_token_col=country_token_col,
            country_tokenizer_path=country_tokenizer_path,
            country_trainer=country_trainer,
            global_token_col=global_token_col,
            global_tokenizer_path=global_tokenizer_path,
            global_trainer=global_trainer,
            noise_words=noise_words,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            preprocess_profile=preprocess_profile,
        ),
        input_file=input_file,
        source_files=source_files,
    )
    return tokenize_name_with_request(request)


def tokenize_name_multi(
    cleansed_dir: Path,
    tokenized_dir: Path,
    *,
    tokenizer_specs: list[TokenizerCalculationSpec] | list[Mapping[str, object]],
    name_col: str = "name_cleansed",
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str | None = None,
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
    input_file: str | None = None,
    source_files: tuple[Path, ...] | None = None,
) -> int:
    _validate_multi_tokenizer_specs(tokenizer_specs=tokenizer_specs)

    request = TokenizeFilesRequest(
        cleansed_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        tokenization=coerce_tokenization_request(
            tokenizer_specs=tokenizer_specs,
            name_col=name_col,
            noise_words=noise_words,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            preprocess_profile=preprocess_profile,
        ),
        input_file=input_file,
        source_files=source_files,
    )
    return tokenize_name_with_request(request)


def _run_tokenization_from_cleansed(
    *,
    cleansed_dir: Path,
    tokenized_dir: Path,
    name_col: str,
    input_file: str | None,
    transform_fn: Callable[[pl.DataFrame], pl.DataFrame],
    source_files: tuple[Path, ...] | None = None,
) -> int:
    eligible_files, skipped_files = _resolve_tokenizable_files(
        cleansed_dir=cleansed_dir,
        name_col=name_col,
        input_file=input_file,
        source_files=source_files,
    )
    return _run_tokenization_for_files(
        eligible_files=eligible_files,
        source_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        name_col=name_col,
        skipped_files=skipped_files,
        transform_fn=transform_fn,
    )


def _run_tokenization_for_files(
    *,
    eligible_files: list[Path],
    source_dir: Path,
    tokenized_dir: Path,
    name_col: str,
    skipped_files: int,
    transform_fn: Callable[[pl.DataFrame], pl.DataFrame],
) -> int:
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = tokenized_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    chunk_throughputs: list[float] = []

    # Mirror each input's own relative path rather than deciding an output
    # shape. Sniffing the input for `jurisdiction_code=*` and branching on it
    # meant this stage could disagree with the layer it reads about that
    # layer's shape -- and when the sniff missed, it silently fell through to
    # a flat layout rather than failing.
    staged_to_final: list[tuple[Path, Path]] = []

    for file_path in eligible_files:
        t_chunk_start = time.perf_counter()
        base_name = file_path.name
        output_path = tokenized_dir / file_path.relative_to(source_dir)
        staged_path = chunks_dir / output_path.relative_to(tokenized_dir)
        staged_path.parent.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(file_path)
        df_out = transform_fn(df)
        df_out.write_parquet(staged_path, compression="snappy")
        staged_to_final.append((staged_path, output_path))

        chunk_rows = len(df_out)
        elapsed_s = max(time.perf_counter() - t_chunk_start, 1e-9)
        rows_per_sec = chunk_rows / elapsed_s
        chunk_throughputs.append(rows_per_sec)
        print(
            f"  {base_name}: {chunk_rows:,} rows in {elapsed_s:.2f}s "
            f"({rows_per_sec:,.0f} rows/s) -> {output_path}"
        )
        total_rows += chunk_rows

        del df, df_out
        gc.collect()

    for staged_path, output_path in staged_to_final:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.replace(output_path)
    shutil.rmtree(chunks_dir, ignore_errors=True)

    _print_throughput_summary(chunk_throughputs)
    print(
        f"\nPass 2 complete -> {len(eligible_files)} file(s) written to {tokenized_dir} "
        f"(skipped {skipped_files} file(s) without '{name_col}')."
    )
    return total_rows


def _resolve_tokenizable_files(
    *,
    cleansed_dir: Path,
    name_col: str,
    input_file: str | None,
    source_files: tuple[Path, ...] | None = None,
) -> tuple[list[Path], int]:
    """Decide which parquet files under `cleansed_dir` to tokenize.

    This package's input contract (see `packages/README.md`) is a root path, a
    file pattern, or an already-resolved file list -- never a workspace
    layout. `source_files` is the already-resolved-list shape and wins where
    a caller supplied one; `input_file` is the root-plus-pattern shape,
    matched under `cleansed_dir` (a plain filename or a glob pattern). One of
    the two is required: this package does not guess which files under a
    root are its input by defaulting to "every parquet file present," since
    that default is itself a claim about the root's layout (that nothing
    else -- a staging directory, a sibling family holding a different row
    shape -- shares it), a claim only the caller resolving its own workspace
    can make correctly.

    Files lacking `name_col` are skipped and counted rather than failing the
    run, since a root may legitimately hold companion data of other shapes.
    """
    if source_files is not None:
        candidate_files = sorted(source_files)
        if input_file is not None:
            candidate_files = [
                path for path in candidate_files if path.name == input_file
            ]
        if not candidate_files:
            raise FileNotFoundError(
                f"No supplied source file matched {input_file!r}"
                if input_file is not None
                else "No source files were supplied to tokenize."
            )
    elif input_file is not None:
        matches = sorted(
            path for path in cleansed_dir.rglob(input_file) if path.is_file()
        )
        if not matches:
            raise FileNotFoundError(
                f"Input parquet file not found under {cleansed_dir}: {input_file}"
            )
        candidate_files = matches
    else:
        raise ValueError(
            "No input given: pass `source_files` (an already-resolved file "
            "list) or `input_file` (a filename/glob pattern under "
            f"{cleansed_dir}). This package does not resolve which files "
            "under a root count as its input on its own -- see "
            "`packages/README.md`."
        )

    if not candidate_files:
        raise FileNotFoundError(
            f"No cleansed parquet files found in {cleansed_dir}. Provide tokenizable parquet input first."
        )

    eligible_files: list[Path] = []
    skipped_files = 0
    for file_path in candidate_files:
        schema_names = set(pl.read_parquet_schema(file_path).keys())
        if name_col in schema_names:
            eligible_files.append(file_path)
        else:
            skipped_files += 1
    if not eligible_files:
        raise FileNotFoundError(
            f"No parquet file under {cleansed_dir} carries the '{name_col}' column; "
            f"{skipped_files} file(s) were skipped for lacking it."
        )
    return eligible_files, skipped_files


def _print_throughput_summary(chunk_throughputs: list[float]) -> None:
    if not chunk_throughputs:
        return

    print(
        "Pass 2 throughput summary -> "
        f"min: {min(chunk_throughputs):,.0f} rows/s, "
        f"median: {statistics.median(chunk_throughputs):,.0f} rows/s, "
        f"max: {max(chunk_throughputs):,.0f} rows/s"
    )
