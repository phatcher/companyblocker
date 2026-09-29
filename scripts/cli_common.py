"""What every script here shares: its arguments, its dry run, and how it reports.

- **The shallow dry run.** `add_dry_run_arg` adds `--dry-run`; the entry point calls `report_dry_run` or `report_resolved_settings` immediately before the call it exists to make, and stops. Nothing below the entry point learns it is in a dry run, so it reports a decision rather than opening a second path through the work.
- **What a run would overwrite.** A script that writes shared storage describes each output as a `PlannedOutput` (clear, extend or migrate) and calls `report_output_plan`, which reads only those paths to say what each holds now.
- **A bad argument.** Every entry point's footer is `raise SystemExit(run_reporting_argument_errors(main))`: an error in `ARGUMENT_ERRORS` is one line on stderr and exit code 2, as argparse reports a missing flag, and anything else propagates as a traceback.
- **Declared settings.** A setting is declared once, as data, in its owner's `_cli_helper` module, and a script maps it onto a flag or an `--additional-args` key; the comment block above `declared_settings` gives the shape.
- **Naming a run's inputs.** A person names natural facts, `--source gleif --target ie`, `--perturbed <profile>`, never a URI or a layout; `source_system_from_args` turns them into a reference pinned to what is on disk. A version, date or seed has a bare form and a role-qualified one (`RoleSetting`); left out, a version or date means the latest on disk and a seed the only one there.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from acquisition.pipeline_selection import all_target_code_lists
from workspace.kind_layout import Kind
from workspace.reference import (
    AmbiguousReferenceError,
    InvalidReferenceError,
    Reference,
    ReferenceNotFoundError,
    Side,
    layout_for,
    reference,
    require_reference,
    resolve_latest,
)
from workspace.repository import repository_root
from workspace.roots import (
    ARTIFACT_DIR_NAME,
    DATA_DIR_NAME,
    ENV_DATA_DIR,
    ENV_OUTPUT_DIR,
    ENV_TEMP_DIR,
    TEMP_DIR_NAME,
    WorkspaceRoots,
    resolve_workspace_roots,
)
from workspace.run_inputs import (
    is_perturbed_source,
    perturbed_parts,
    profile_request,
    source_request,
)

_RUN_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_TRUE_BOOL_LITERALS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_BOOL_LITERALS = frozenset({"0", "false", "no", "n", "off"})


def parse_run_date(value: str) -> str:
    """Parse and normalize run-date CLI input to YYYY-MM-DD."""
    candidate = value.strip()
    match = _RUN_DATE_PATTERN.fullmatch(candidate)
    if match is None:
        raise argparse.ArgumentTypeError("Run date must be in YYYY-MM-DD format.")

    year, month, day = (int(part) for part in match.groups())
    try:
        parsed = date(year, month, day)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    return parsed.isoformat()


def resolve_existing_path_and_rows(
    *, source: str | Path, rows: int
) -> tuple[Path, int]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if rows < 1:
        raise ValueError("--rows must be >= 1")
    return source_path, rows


def parse_stage_key_value_args(
    additional_args: list[list[str]] | None,
    *,
    invalid_message: str,
    empty_message: str,
    parse_group: Callable[[list[str]], dict[str, str]] | None = None,
) -> dict[str, dict[str, object]]:
    options: dict[str, dict[str, object]] = {}
    if not additional_args:
        return options

    for group in additional_args:
        parsed_group = (
            parse_group(group)
            if parse_group is not None
            else _parse_stage_arg_group_default(
                group,
                invalid_message=invalid_message,
                empty_message=empty_message,
            )
        )
        for stage_key, raw_value in parsed_group.items():
            if "." not in stage_key:
                raise ValueError(
                    invalid_message.format(token=f"{stage_key}={raw_value}")
                )

            stage_name, option_name = stage_key.split(".", 1)
            stage_name = stage_name.strip().lower()
            option_name = option_name.strip().lower()
            if not stage_name or not option_name:
                raise ValueError(empty_message.format(token=f"{stage_key}={raw_value}"))

            options.setdefault(stage_name, {})[option_name] = coerce_scalar_value(
                raw_value
            )

    return options


def parse_key_value_pairs(
    tokens: list[str],
    *,
    invalid_message: str,
    empty_message: str,
    require_dot: bool = False,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in tokens:
        normalized_token = token.strip()
        if not normalized_token:
            continue
        if "=" not in normalized_token or (require_dot and "." not in normalized_token):
            raise ValueError(invalid_message.format(token=normalized_token))

        key, value = normalized_token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise ValueError(empty_message.format(token=normalized_token))
        parsed[key] = value

    return parsed


def _parse_stage_arg_group_default(
    tokens: list[str],
    *,
    invalid_message: str,
    empty_message: str,
) -> dict[str, str]:
    return parse_key_value_pairs(
        tokens,
        invalid_message=invalid_message,
        empty_message=empty_message,
        require_dot=True,
    )


def supported_code_lists() -> tuple[list[str], list[str]]:
    return all_target_code_lists()


def supported_codes_epilog() -> str:
    supported_countries, supported_systems = supported_code_lists()
    return (
        "Country codes in --systems all: "
        + ", ".join(supported_countries)
        + "\n"
        + "Non-country systems in --systems all: "
        + ", ".join(supported_systems)
    )


def add_systems_arg(
    parser: argparse.ArgumentParser,
    *,
    action: str | None = None,
    detail: str | None = None,
) -> None:
    subject = "One or more system codes"
    if action:
        subject += f" to {action}"

    if detail is None:
        detail = (
            "Countries are valid system codes (for example 'gb'), and non-country "
            "systems are handled the same way. Use 'all' to target every system the "
            "catalog flags with all_systems_target."
        )

    help_text = subject + "."
    if detail:
        help_text += f" {detail}"
    help_text += " Comma-separated values are accepted."

    parser.add_argument(
        "--systems",
        nargs="+",
        required=True,
        help=help_text,
    )


def add_run_date_arg(
    parser: argparse.ArgumentParser,
    *,
    subject: str = "Run date",
    detail: str | None = None,
    fallback: str | None = None,
    default_behavior: str | None = None,
) -> None:
    if fallback is None and default_behavior is not None:
        fallback_by_behavior = {
            "today": "Defaults to today's date.",
            "latest_supported_snapshots": "If omitted, latest available snapshots are used where supported.",
            "latest_source_snapshot": "If omitted, the latest available source snapshot for that system is used.",
            "latest_canonical_folder": "If omitted, the latest canonical folder is used.",
        }
        fallback = fallback_by_behavior.get(default_behavior)

    help_text = f"{subject} in YYYY-MM-DD format."
    if detail:
        help_text += f" {detail}"
    if fallback:
        help_text += f" {fallback}"

    parser.add_argument(
        "--date", dest="run_date", default=None, type=parse_run_date, help=help_text
    )


def add_root_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """The single anchor `add_workspace_roots_args` replaces. A script still
    carrying it is one its area's script-wiring item has not yet moved."""
    parser.add_argument("--root", default=".", help=help_text)


def add_workspace_roots_args(parser: argparse.ArgumentParser) -> None:
    """Add `--data-dir`, `--output-dir` and `--temp-dir`, the three overridable
    workspace roots; `resolve_workspace_roots_from_args` turns them into the
    resolved value every layout function takes.

    Each flag overrides its environment variable, which overrides the default
    under the checkout, and a relative value resolves against the checkout,
    never the working directory.
    """
    for flag, variable, default_name, meaning in (
        (
            "--data-dir",
            ENV_DATA_DIR,
            DATA_DIR_NAME,
            "the data root every layer lives under",
        ),
        (
            "--output-dir",
            ENV_OUTPUT_DIR,
            ARTIFACT_DIR_NAME,
            "the artifacts root every run's output lives under",
        ),
        ("--temp-dir", ENV_TEMP_DIR, TEMP_DIR_NAME, "the root for disposable scratch"),
    ):
        parser.add_argument(
            flag,
            default=None,
            help=(
                f"Override {meaning}. Defaults to {variable} when set, else "
                f"{default_name}/ under the checkout; a relative path resolves "
                "against the checkout."
            ),
        )


def add_external_destination_arg(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    help_text: str,
    required: bool = False,
) -> None:
    """Add a flag naming a destination the workspace does not own: another
    repository, or this checkout's tracked `docs/`. It defaults to `None`, so a
    script resolves its default from the roots once they are known.

    The one other function allowed to add a location flag besides
    `add_workspace_roots_args`, so a script cannot use it to reach into a
    workspace root by hand without saying so at the call.
    """
    parser.add_argument(
        flag, type=Path, default=None, required=required, help=help_text
    )


def resolve_workspace_roots_from_args(args: argparse.Namespace) -> WorkspaceRoots:
    """The resolved roots for a script that added `add_workspace_roots_args`.

    The checkout is the one this file sits in, so the roots follow the code
    that runs rather than the directory it was launched from.
    """
    return resolve_workspace_roots(
        repository_root(Path(__file__)),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        environ=os.environ,
    )


_ROOT_FLAGS = ("--data-dir", "--output-dir", "--temp-dir")


def _reference_flag(prefix: str | None, name: str) -> str:
    stem = name.replace("_", "-")
    return f"--{prefix}-{stem}" if prefix else f"--{stem}"


def _reference_dest(prefix: str | None, name: str) -> str:
    return f"{prefix.replace('-', '_')}_{name}" if prefix else name


def add_reference_args(
    parser: argparse.ArgumentParser,
    kind: Kind,
    side: Side,
    *,
    prefix: str | None = None,
) -> None:
    """Add one flag per segment of `kind`'s `side`, the only way a script names one.

    A segment is `--<segment>`, or `--<prefix>-<segment>` where a script takes
    two references of one kind; an optional segment is a switch named for its
    word (`--draft`). Each value is checked against its segment as it is
    parsed. `require_reference_from_args` builds the reference and checks it
    exists.
    """
    layout = layout_for(kind, side)
    for segment in layout.segments:
        flag = _reference_flag(prefix, segment.literal or segment.name)
        dest = _reference_dest(prefix, segment.name)
        where = f"{layout.scheme}://"
        if segment.literal is not None:
            parser.add_argument(
                flag,
                dest=dest,
                action="store_true",
                default=False,
                help=f"Select the {segment.literal} {segment.name} of {where}.",
            )
            continue

        def admitted(value: str, segment=segment) -> str:
            if not segment.admits(value):
                raise argparse.ArgumentTypeError(
                    f"{value!r} is not a valid {segment.name}"
                )
            return value

        parser.add_argument(
            flag,
            dest=dest,
            default=None,
            type=admitted,
            help=f"The {segment.name.replace('_', ' ')} of {where}.",
        )


def reference_from_args(
    args: argparse.Namespace, kind: Kind, side: Side, *, prefix: str | None = None
) -> Reference:
    """The reference, or selection, the flags `add_reference_args` added name."""
    values: dict[str, str] = {}
    for segment in layout_for(kind, side).segments:
        given = getattr(args, _reference_dest(prefix, segment.name))
        if segment.literal is not None:
            if given:
                values[segment.name] = segment.literal
        elif given is not None:
            values[segment.name] = given
    return reference(kind, side, **values)


def reference_flags(ref: Reference, *, prefix: str | None = None) -> list[str]:
    """The flags that name `ref`; the inverse of `reference_from_args`."""
    layout = ref.layout
    flags: list[str] = []
    for name, value in ref.values:
        segment = layout.segment(name)
        if segment.literal is not None:
            flags.append(_reference_flag(prefix, segment.literal))
        else:
            flags.extend([_reference_flag(prefix, name), value])
    return flags


def require_reference_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    kind: Kind,
    side: Side,
    *,
    prefix: str | None = None,
) -> Reference:
    """The one existing reference the flags name, checked before any work.

    A missing segment, a selection matching nothing, or one matching several
    ends the script through `parser.error`, naming what does exist.
    """
    try:
        return require_reference(
            roots, reference_from_args(args, kind, side, prefix=prefix)
        )
    except (
        InvalidReferenceError,
        ReferenceNotFoundError,
        AmbiguousReferenceError,
    ) as error:
        parser.error(str(error))


class RoleSettingError(ValueError):
    """A version, date or seed flag that names no one role's setting."""


@dataclass(frozen=True)
class RoleSetting:
    """A version, a date or a seed, and the role it plays in the command.

    Each has a qualified flag naming the role, `--source-date`,
    `--perturbed-seed`, never a system's name or a layer's, and the bare obvious
    word, `--date`, `--seed`. A `role` of None is the script's own setting, whose
    flag is the bare word itself. `shared` says a bare flag may set this role
    together with others; a seed is never shared, since the same number is never
    meant for two roles.
    """

    name: str
    role: str | None
    help: str
    type: Callable[[str], object] = str
    shared: bool = True

    @property
    def bare_flag(self) -> str:
        return f"--{self.name}"

    @property
    def flag(self) -> str:
        return self.bare_flag if self.role is None else f"--{self.role}-{self.name}"

    @property
    def dest(self) -> str:
        return self.name if self.role is None else f"{self.role}_{self.name}"


def _bare_dest(name: str) -> str:
    return f"bare_{name}"


def add_role_settings(
    parser: argparse.ArgumentParser, settings: Iterable[RoleSetting]
) -> None:
    """Add each setting's qualified flag and, once per name, the bare one."""
    settings = list(settings)
    for setting in settings:
        others = [s for s in settings if s.name == setting.name and s is not setting]
        bare = "" if setting.role is None else f" Also {setting.bare_flag}"
        if bare and others:
            bare += ", where no other role it could mean is left open"
        parser.add_argument(
            setting.flag,
            dest=setting.dest,
            default=None,
            type=setting.type,
            help=f"{setting.help}{bare + '.' if bare else ''}",
        )
    for name in dict.fromkeys(setting.name for setting in settings):
        group = [setting for setting in settings if setting.name == name]
        if any(setting.role is None for setting in group):
            continue
        parser.add_argument(
            f"--{name}",
            dest=_bare_dest(name),
            default=None,
            type=group[0].type,
            help=f"Short for {', '.join(setting.flag for setting in group)}.",
        )


def resolve_role_settings(
    args: argparse.Namespace,
    settings: Iterable[RoleSetting],
    *,
    accepts: Callable[[RoleSetting, object], bool] | None = None,
) -> dict[str, object]:
    """Each setting's value by its dest, the bare flag given to every role that
    takes one.

    A qualified flag wins for its role. Where one role is left the bare flag is
    simply that role's. Where several are, each must be able to take the value,
    which `accepts` decides, and a seed, never shared, goes to none of them; both
    are refused naming the qualified spellings.
    """
    settings = list(settings)
    values = {setting.dest: getattr(args, setting.dest, None) for setting in settings}
    for name in dict.fromkeys(setting.name for setting in settings):
        bare = getattr(args, _bare_dest(name), None)
        group = [setting for setting in settings if setting.name == name]
        if bare is None or any(setting.role is None for setting in group):
            continue
        open_roles = [setting for setting in group if values[setting.dest] is None]
        qualified = ", ".join(setting.flag for setting in open_roles)
        if len(open_roles) > 1 and not all(setting.shared for setting in open_roles):
            raise RoleSettingError(
                f"--{name} {bare} could mean several roles and is never shared "
                f"between them; give {qualified}."
            )
        refused = [
            setting
            for setting in open_roles
            if len(open_roles) > 1
            and accepts is not None
            and not accepts(setting, bare)
        ]
        if refused:
            raise RoleSettingError(
                f"--{name} {bare} does not suit {', '.join(s.role or name for s in refused)}; "
                f"give {qualified}."
            )
        for setting in open_roles:
            values[setting.dest] = bare
    return values


PERTURBED_SOURCE_SETTINGS = (
    RoleSetting(
        "version",
        "perturbed",
        "The profile version a perturbed source was made under; the latest if omitted.",
    ),
    RoleSetting(
        "seed",
        "perturbed",
        "The seed a perturbed source was made under; needed only where several exist.",
        type=int,
        shared=False,
    ),
)


def add_perturbed_source_args(
    parser: argparse.ArgumentParser, *, own_seed: bool = False
) -> None:
    """`--perturbed <profile>`, reading the source as perturbed under that
    profile, with its version and seed. A script with a seed of its own keeps the
    bare `--seed` for that one, so only `--perturbed-seed` is added."""
    parser.add_argument(
        "--perturbed",
        default=None,
        metavar="PROFILE",
        help="Read the source as perturbed under this profile. Implies truth in source_uri.",
    )
    if not own_seed:
        add_role_settings(parser, PERTURBED_SOURCE_SETTINGS)
        return
    for setting in PERTURBED_SOURCE_SETTINGS:
        parser.add_argument(
            setting.flag,
            dest=setting.dest,
            default=None,
            type=setting.type,
            help=setting.help,
        )


def perturbed_source_from_args(
    args: argparse.Namespace, roots: WorkspaceRoots, *, source: str
) -> Reference | None:
    """The perturbed dataset the flags name, pinned to what is on disk, or None
    where the source is read as it is."""
    values = resolve_role_settings(args, PERTURBED_SOURCE_SETTINGS)
    if args.perturbed is None:
        given = [name for name, value in values.items() if value is not None]
        if given:
            raise RoleSettingError(
                f"{given} belong to --perturbed, which is not given."
            )
        return None
    return resolve_latest(
        roots,
        source_request(
            source=source,
            perturbed=args.perturbed,
            version=values["perturbed_version"],  # type: ignore[arg-type]
            seed=values["perturbed_seed"],  # type: ignore[arg-type]
        ),
    )


def source_system_from_args(args: argparse.Namespace, roots: WorkspaceRoots) -> str:
    """The source system string the run's configuration still takes: the system
    code, or for `--perturbed` the dataset the flags pin. Nobody types the
    perturbed spelling, so one given as `--source` is refused."""
    source = str(args.source_system).strip().lower()
    if ":" in source:
        raise RoleSettingError(
            f"--source {source} spells a perturbed dataset; give the system and "
            "--perturbed <profile>, with --version and --seed only where needed."
        )
    dataset = perturbed_source_from_args(args, roots, source=source)
    if dataset is None:
        return source
    return dataset.uri


def source_flags(source_system: str) -> list[str]:
    """The flags naming a source system string, for a script launching another."""
    if not is_perturbed_source(source_system):
        return ["--source", source_system]
    source, profile, version, seed = perturbed_parts(source_system)
    return [
        "--source", source, "--perturbed", profile,
        "--perturbed-version", version, "--perturbed-seed", str(seed),
    ]  # fmt: skip


def add_perturb_args(parser: argparse.ArgumentParser) -> None:
    """`--perturb <profile>` for the script that makes a perturbed dataset, a
    verb where `--perturbed` reads one, with the version and whether it is a draft."""
    parser.add_argument(
        "--perturb", required=True, metavar="PROFILE", help="The profile to apply."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="The profile version; the latest promoted if omitted.",
    )
    parser.add_argument(
        "--draft", action="store_true", default=False, help="Apply the version's draft."
    )


def profile_from_args(args: argparse.Namespace, roots: WorkspaceRoots) -> Reference:
    """The profile `--perturb` names, pinned to a version that exists."""
    return resolve_latest(
        roots,
        profile_request(perturb=args.perturb, version=args.version, draft=args.draft),
    )


def script_invocation(argv: list[str]) -> tuple[str, ...]:
    """What a record keeps of a run's command line: the script's name and its
    flags, without the root flags, so re-running it reproduces the run under
    whatever roots are current."""
    script, *arguments = argv
    kept: list[str] = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument in _ROOT_FLAGS:
            skip_value = True
            continue
        if argument.split("=", 1)[0] in _ROOT_FLAGS:
            continue
        kept.append(argument)
    return (Path(script).name, *kept)


def add_chunk_size_arg(
    parser: argparse.ArgumentParser, *, help_text: str, default: int = 100_000
) -> None:
    parser.add_argument("--chunk-size", type=int, default=default, help=help_text)


def add_company_col_arg(
    parser: argparse.ArgumentParser, *, default: str = "name"
) -> None:
    parser.add_argument(
        "--company-col",
        default=default,
        help="Company name column for cleanse input. Defaults to canonical column 'name'.",
    )


def add_char_whitelist_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--char-whitelist",
        default=r"[^a-z0-9\s!&]",
        help="Regex describing characters to remove during cleansing.",
    )


def add_input_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional parquet filename to process for downstream stages "
            "(for example 'sample.parquet')."
        ),
    )


def add_name_col_arg(
    parser: argparse.ArgumentParser, *, default: str = "name_cleansed"
) -> None:
    parser.add_argument(
        "--name-col",
        default=default,
        help="Cleansed name column used for tokenization.",
    )


def add_token_col_arg(
    parser: argparse.ArgumentParser, *, default: str = "name_tokens"
) -> None:
    parser.add_argument(
        "--token-col",
        default=default,
        help="Output token column name for tokenization.",
    )


def add_token_mode_arg(
    parser: argparse.ArgumentParser, *, default: str = "single"
) -> None:
    parser.add_argument(
        "--token-mode",
        choices=["single", "dual"],
        default=default,
        help="Tokenization mode: single writes one output token column, dual writes country/global token columns.",
    )


def add_country_token_col_arg(
    parser: argparse.ArgumentParser, *, default: str = "country_tokens"
) -> None:
    parser.add_argument(
        "--country-token-col",
        default=default,
        help="Output country token column name for dual mode.",
    )


def add_global_token_col_arg(
    parser: argparse.ArgumentParser, *, default: str = "global_tokens"
) -> None:
    parser.add_argument(
        "--global-token-col",
        default=default,
        help="Output global token column name for dual mode.",
    )


def add_tokenizer_path_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Explicit tokenizer path for tokenization. Overrides scope/profile selection.",
    )


def add_tokenizer_scope_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer-scope",
        choices=["country", "global"],
        default="country",
        help="Tokenizer resolution scope for the tokenize stage.",
    )


def add_tokenizer_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer-profile",
        default="default",
        help="Global tokenizer profile when --tokenizer-scope global.",
    )


def add_tokenizer_backend_args(
    parser: argparse.ArgumentParser,
    *,
    trainer_choices: list[str],
    default_trainer: str = "wordpiece",
) -> None:
    parser.add_argument(
        "--tokenizer",
        choices=trainer_choices,
        default=default_trainer,
        help="Tokenizer backend.",
    )
    parser.add_argument(
        "--encoding",
        dest="tokenizer_encoding",
        choices=["unigram", "bpe"],
        default="bpe",
        help="SentencePiece encoding when --tokenizer sentencepiece.",
    )
    parser.add_argument(
        "--sp-character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage when --tokenizer sentencepiece.",
    )
    parser.add_argument(
        "--sp-byte-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SentencePiece byte fallback when --tokenizer sentencepiece.",
    )


def add_tokenizer_artifact_output_args(
    parser: argparse.ArgumentParser,
    *,
    default_corpus_filename: str,
) -> None:
    parser.add_argument(
        "--corpus-filename",
        default=default_corpus_filename,
        help=(
            "Corpus file name used under the tokenizer working tree. "
            f"Default: {default_corpus_filename}."
        ),
    )


def add_optimize_mode_args(
    parser: argparse.ArgumentParser,
    *,
    base_defaults: dict[str, float | int],
    default_vocab_grid: str,
    default_min_freq_grid: str,
    expansion_strategy_choices: list[str],
    default_expansion_strategy: str,
    optimize_preset_choices: list[str],
) -> None:
    parser.add_argument(
        "--optimize-preset",
        choices=sorted(optimize_preset_choices),
        default=None,
        help=(
            "Named bundle of threshold and grid-search defaults, "
            "budget and quality bar chosen together instead of a dozen "
            "individual flags. Not the same as --profile (the global "
            "tokenizer profile name). Any flag also passed explicitly "
            "still wins over the preset's own value. Default: chosen "
            "from --tokenizer, unchanged from this mechanism's pre-preset "
            "behavior. Available: " + ", ".join(sorted(optimize_preset_choices))
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=base_defaults["validation_fraction"],
    )
    parser.add_argument(
        "--unk-rate-threshold", type=float, default=base_defaults["unk_rate_threshold"]
    )
    parser.add_argument(
        "--fertility-target", type=float, default=base_defaults["fertility_target"]
    )
    parser.add_argument(
        "--fertility-tolerance",
        type=float,
        default=base_defaults["fertility_tolerance"],
        help="Maximum allowed absolute distance from --fertility-target.",
    )
    parser.add_argument(
        "--fertility-min", type=float, default=base_defaults["fertility_min"]
    )
    parser.add_argument(
        "--fertility-max", type=float, default=base_defaults["fertility_max"]
    )
    parser.add_argument(
        "--token-count-median-max",
        type=float,
        default=base_defaults["token_count_median_max"],
    )
    parser.add_argument(
        "--token-count-p95-max",
        type=float,
        default=base_defaults["token_count_p95_max"],
    )
    parser.add_argument("--eligibility-pass-rate", type=float, default=0.80)
    parser.add_argument(
        "--vocab-size-distance-tolerance",
        type=float,
        default=0.01,
        help=(
            "Within eligible candidates, choose the smallest effective vocab size among pairs whose median "
            "fertility distance is within this tolerance of the best (lowest) median distance."
        ),
    )
    parser.add_argument("--vocab-sizes", default=default_vocab_grid)
    parser.add_argument("--min-frequencies", default=default_min_freq_grid)
    parser.add_argument(
        "--elbow-min-points",
        type=int,
        default=3,
        help="Minimum evaluated vocab points per seed/min-frequency pair before elbow cutoff is considered.",
    )
    parser.add_argument(
        "--elbow-min-improvement",
        type=float,
        default=0.002,
        help=(
            "Minimum absolute fertility-distance improvement required to treat a larger vocab candidate as meaningful. "
            "Smaller gains are treated as post-elbow behavior."
        ),
    )
    parser.add_argument(
        "--elbow-extra-steps",
        type=int,
        default=2,
        help="Number of additional larger-vocab steps to evaluate after elbow crossing before cutoff.",
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel optimize candidates to run at once. Default: logical CPU count minus one, with a floor of 1.",
    )
    parser.add_argument(
        "--worker-memory-floor-gb",
        type=float,
        default=base_defaults["worker_memory_floor_gb"],
        help=(
            "Conservative starting per-candidate memory estimate (GB) used before "
            "any real worker RSS samples exist for this system. Only acts as a "
            "floor -- real history/observed samples take priority once available."
        ),
    )
    parser.add_argument(
        "--memory-headroom-fraction",
        type=float,
        default=base_defaults["memory_headroom_fraction"],
        help="Fraction of currently-available system memory held back as a safety margin when sizing concurrent waves.",
    )
    parser.add_argument(
        "--wave-chunk-size",
        type=int,
        default=base_defaults["wave_chunk_size"],
        help=(
            "Jobs a wave's workers process before that wave's process pool is torn "
            "down and recycled. Bounds per-worker memory accumulation (confirmed "
            "an allocator artifact, not fixable with gc.collect()) independent of "
            "how much memory is available."
        ),
    )
    parser.add_argument(
        "--max-parallel-waves",
        type=int,
        default=None,
        help="Cap on concurrently-running waves. Default: the resolved --max-workers ceiling.",
    )
    parser.add_argument(
        "--disable-adaptive-concurrency",
        action="store_true",
        default=False,
        help=(
            "Fall back to a single fixed-size worker pool for the whole run "
            "(today's pre-Phase-2 behavior) instead of memory-aware wave "
            "scheduling. No wave_budget/concurrency diagnostics in this mode."
        ),
    )
    parser.add_argument(
        "--candidate-timeout-seconds",
        type=float,
        default=base_defaults["candidate_timeout_seconds"],
        help=(
            "Per-candidate future.result() budget. A candidate that "
            "exceeds it is treated as a terminal, non-retryable hang: its pool "
            "is torn down and recreated (fixed-pool mode) or simply not "
            "waited on (adaptive wave mode), and the sweep continues with the "
            "remaining candidates. A conservative placeholder, not yet "
            "anchored to a real measurement -- see "
            "training.optimize_retry.DEFAULT_CANDIDATE_TIMEOUT_SECONDS."
        ),
    )
    parser.add_argument("--diagnostic-top-n", type=int, default=20)
    parser.add_argument(
        "--pilot-seed-count",
        type=int,
        default=1,
        help="Number of initial seeds used for vocab-first pilot screening.",
    )
    parser.add_argument(
        "--vocab-frontier-top-k",
        type=int,
        default=4,
        help="Minimum number of vocab points per min-frequency to keep after pilot screening.",
    )
    parser.add_argument(
        "--vocab-frontier-distance-tolerance",
        type=float,
        default=base_defaults["vocab_frontier_distance_tolerance"],
        help="Keep vocab points whose pilot median fertility distance is within this tolerance of the best.",
    )
    parser.add_argument(
        "--expansion-strategy",
        choices=expansion_strategy_choices,
        default=default_expansion_strategy,
        help=(
            "Arm allocation/elimination strategy used during the expansion "
            "phase after pilot screening. Currently only 'heuristic' (today's "
            "fixed one-job-per-seed allocation with pass-rate/fertility "
            "pruning) is available."
        ),
    )
    for flag, help_text in (
        (
            "--pilot-early-stop",
            (
                "Stop sending a pilot line (one seed and min_frequency) to larger "
                "vocab points once its train-split fertility is below the target "
                "or has stopped moving. One further point runs after that, and the "
                "auto point always runs. false runs the full grid."
            ),
        ),
        (
            "--skip-equivalent-candidates",
            (
                "Do not train a pilot or expansion candidate another candidate "
                "already fixes; it takes that candidate's result. That is every "
                "lower min_frequency at a vocab whose highest min_frequency "
                "reached its full vocabulary, and every larger vocab on a "
                "min_frequency that has reached its plateau. false trains them all."
            ),
        ),
        (
            "--refining-phase",
            (
                "Run the third phase after pilot and expansion, which searches "
                "inside the winning bracket holding min_frequency at its winning "
                "value. false skips it."
            ),
        ),
    ):

        def _bool_value(value: str, flag: str = flag) -> bool:
            return parse_optional_bool(value, argument_name=flag)

        parser.add_argument(
            flag,
            type=_bool_value,
            default=True,
            metavar="{true,false}",
            help=f"{help_text} (default: true)",
        )
    parser.add_argument(
        "--refine-vocab-step",
        type=int,
        default=None,
        help=(
            "Vocab spacing for the refining phase. Default: derived from the "
            "winning combo's measured seed noise and the fertility slope "
            "between its bracket points (see "
            "src/training/optimize_search_methodology.md Part 8) instead of "
            "a fixed value."
        ),
    )
    parser.add_argument(
        "--refine-vocab-step-minimum",
        type=int,
        default=100,
        help=(
            "Floor under the noise-derived refining step -- a flat "
            "fertility slope falls back to this value. Ignored "
            "when --refine-vocab-step is given explicitly."
        ),
    )
    parser.add_argument(
        "--delete-rejected-models",
        type=lambda value: parse_optional_bool(
            value, argument_name="--delete-rejected-models"
        ),
        nargs="?",
        const=True,
        default=True,
        help=(
            "Delete rejected optimize candidate model outputs under artifacts/tokenizers/.../optimize/models. "
            "Use --delete-rejected-models false to keep them for analysis."
        ),
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        default=False,
        help="Finalize optimize output by merging shards, selecting the best pair, and retraining the winner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Print optimize scheduling plan (including no-pruning upper-bound jobs and duplicate-output checks) "
            "without training or evaluating candidates."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Force overwrite optimize artifacts. Without --force, optimize reuses fresh corpus/candidates and skips "
            "already-valid model outputs when canary and corpus freshness checks pass."
        ),
    )

    def _reoptimize_value(value: str) -> bool:
        return parse_optional_bool(value, argument_name="--reoptimize")

    parser.add_argument(
        "--reoptimize",
        type=_reoptimize_value,
        default=False,
        metavar="{true,false}",
        help=(
            "true reselects and refines under the current code even when the "
            "finalized result looks fresh, reusing every trained candidate; "
            "--force retrains them instead. (default: false)"
        ),
    )


def add_tokenizer_training_core_args(
    parser: argparse.ArgumentParser,
    *,
    trainer_choices: list[str],
    default_corpus_filename: str,
    systems_detail: str,
    default_profile: str = "auto",
    default_trainer: str = "wordpiece",
    default_vocab_size: int = 40_000,
    default_min_frequency: int = 1,
    default_name_col: str = "name_cleansed",
) -> None:
    add_systems_arg(parser, detail=systems_detail)
    parser.add_argument(
        "--profile",
        default=default_profile,
        help=f"Global tokenizer profile name (default: {default_profile}).",
    )
    add_tokenizer_backend_args(
        parser, trainer_choices=trainer_choices, default_trainer=default_trainer
    )
    add_root_arg(parser, help_text="Project root containing data/ and artifacts/.")
    parser.add_argument("--vocab-size", type=int, default=default_vocab_size)
    parser.add_argument("--min-frequency", type=int, default=default_min_frequency)
    add_name_col_arg(parser, default=default_name_col)
    add_tokenizer_artifact_output_args(
        parser, default_corpus_filename=default_corpus_filename
    )
    add_optimize_wipe_arg(parser)


def add_dry_run_arg(
    parser: argparse.ArgumentParser, *, help_text: str | None = None
) -> None:
    """Add a shallow ``--dry-run`` flag.

    Shallow means the flag is read once, at the entry point, after arguments
    are resolved and immediately before the call the script exists to make.
    Nothing below that call learns it is in a dry run: no flag threads down
    into the module doing the real work. A script that opts in checks
    ``args.dry_run`` itself, calls `report_dry_run` with whatever it has
    already resolved, and returns before making that call -- it does not
    gain a second, dry-run-aware code path through the work itself.
    """
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=help_text
        or (
            "Resolve arguments and report the call this run would make, "
            "without making it."
        ),
    )


def report_dry_run(script_name: str, **resolved: object) -> None:
    """Print a shallow dry-run report and nothing else.

    Call once arguments are resolved and the script has reached the call it
    exists to make. Prints the resolved values that decided what would be
    called, then stops -- the caller returns immediately afterwards rather
    than making that call.
    """
    print(f"[dry-run] {script_name}: would proceed with:")
    for key, value in resolved.items():
        print(f"  {key}={value!r}")


OUTPUT_CLEAR = "clear"
OUTPUT_EXTEND = "extend"
OUTPUT_MIGRATE = "migrate"
OUTPUT_EFFECTS = (OUTPUT_CLEAR, OUTPUT_EXTEND, OUTPUT_MIGRATE)


@dataclass(frozen=True)
class PlannedOutput:
    """One output path a run would touch, and what the run would do to it.

    `effect` is `OUTPUT_CLEAR` (what is there is removed and rewritten),
    `OUTPUT_EXTEND` (what is there is kept and added to) or `OUTPUT_MIGRATE`
    (what is there is moved to `into`). `pattern` is the glob counted as what
    the path holds; `holds` replaces that count with a description of the
    caller's own, for a layer whose unit is not a file. An `optional` path is
    left out of the report when it does not exist, for a legacy location whose
    absence is the normal case.
    """

    path: Path
    effect: str
    pattern: str = "**/*"
    into: Path | None = None
    note: str = ""
    holds: Callable[[Path], str] | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if self.effect not in OUTPUT_EFFECTS:
            raise ValueError(
                f"output effect {self.effect!r} is not one of {OUTPUT_EFFECTS}"
            )
        if (self.effect == OUTPUT_MIGRATE) != (self.into is not None):
            raise ValueError("a migrated output, and only one, names where it goes")


def _describe_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    size = size_bytes / 1024
    for unit in ("KiB", "MiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def describe_output_contents(path: Path, *, pattern: str = "**/*") -> str:
    """What `path` holds, read from disk: absent, empty, or a count and size of
    the files matching `pattern` beneath a directory."""
    if not path.exists():
        return "does not exist"
    if path.is_file():
        return f"1 file(s), {_describe_size(path.stat().st_size)}"
    files = [candidate for candidate in path.glob(pattern) if candidate.is_file()]
    if not files:
        return "exists, empty"
    total = sum(candidate.stat().st_size for candidate in files)
    return f"{len(files)} file(s), {_describe_size(total)}"


def report_output_plan(prefix: str, outputs: Iterable[PlannedOutput]) -> None:
    """Print, for each output a run would touch, what it would do to that path
    and what the path holds now.

    Deep in what it reports and shallow in how: the entry point resolves its
    own output paths and calls this before the call it exists to make, and
    nothing is read beyond those paths' own contents.
    """
    for output in outputs:
        exists = output.path.exists()
        if output.optional and not exists:
            continue
        if not exists:
            holds = "does not exist"
        elif output.holds is not None:
            holds = output.holds(output.path)
        else:
            holds = describe_output_contents(output.path, pattern=output.pattern)
        destination = f" into {output.into}" if output.into is not None else ""
        note = f" -- {output.note}" if output.note else ""
        print(
            f"{prefix} would {output.effect} {output.path}{destination} [{holds}]{note}"
        )


def add_optimize_wipe_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--optimize-wipe",
        action="store_true",
        default=False,
        help=(
            "Acknowledge that resampling the training corpus would orphan an "
            "existing --mode optimize sweep that depends on its current bytes "
            "(for this trainer or any other trainer sharing the same corpus) -- "
            "the sweep itself won't be deleted, but it becomes unreachable from "
            "any new sweep, which will start over from scratch instead of "
            "extending it. Required whenever that would happen; otherwise the "
            "run refuses and names the at-risk sweep(s)."
        ),
    )


def parse_bool_literal(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    lowered = str(value).strip().lower()
    if lowered in _TRUE_BOOL_LITERALS:
        return True
    if lowered in _FALSE_BOOL_LITERALS:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_optional_bool(value: str | bool | None, *, argument_name: str) -> bool:
    if value is None:
        return True

    try:
        parsed = parse_bool_literal(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected boolean value for {argument_name} (for example true/false)."
        ) from exc
    if parsed is not None:
        return parsed
    raise argparse.ArgumentTypeError(
        f"Expected boolean value for {argument_name} (for example true/false)."
    )


def coerce_scalar_value(value: str) -> object:
    try:
        parsed_bool = parse_bool_literal(value)
    except ValueError:
        parsed_bool = None
    if parsed_bool is not None:
        return parsed_bool
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# Declared settings.
#
# A setting is declared once, as plain data, by the owner of the code that reads
# it, in that owner's `_cli_helper` module: a package or an area. A script maps
# the declared settings it takes onto its own surface, a flag or a prefixed
# `--additional-args` key, optionally with a default of its own. Everything a
# caller sees about a setting -- its flag, the default in its help, whether it
# applies to the chosen strategy, where its value came from -- is derived here
# from those two pieces, so an owner never builds a parser and a script never
# restates what an owner knows.
#
# A declaration is a mapping with:
#   name     the setting's identifier, the config field it fills where one exists
#   type     "str", "int", "float" or "bool"
#   default  the owner's default
#   help     one line saying what the setting does
#   applies  optional; dimension -> the values it applies under, a setting
#            applying only when every named dimension matches. The dimensions
#            are `SETTING_DIMENSIONS`; an empty or absent mapping applies always.
#   choices  optional; the values a flag accepts
#   nargs    optional; argparse's nargs for a list-valued flag
#   option   optional; the `backend_options` key a backend setting fills. Every
#            such setting that applies to a run is recorded with it, given or
#            defaulted (`backend_options_from_settings`), so a changed default
#            never hands a new run the identity of one made under the old.
#
# A script's mapping is setting name -> {"flag": "--x"} or {"key": "stage.x"},
# plus an optional "default" overriding the owner's and, for a flag, an optional
# "required" and a "note" appended to the owner's help for what the setting
# means to this script.

SettingDeclaration = Mapping[str, object]
SettingSurface = Mapping[str, object]

SETTING_DIMENSIONS = ("representation", "text_view", "backend")
SETTING_TYPES = ("str", "int", "float", "bool")

SOURCE_GIVEN = "given"
SOURCE_DEFAULT = "default"
SOURCE_SCRIPT_DEFAULT = "script default"

_GIVEN_SETTINGS_ATTR = "_given_settings"
_ARGPARSE_TYPES: dict[str, Callable[[str], object]] = {
    "str": str,
    "int": int,
    "float": float,
}


def declared_settings(
    *groups: tuple[SettingDeclaration, ...],
) -> dict[str, SettingDeclaration]:
    """Every declaration from `groups`, by name, validated.

    Raises `ValueError` for a declaration missing a required field, naming an
    unknown type or applicability dimension, or sharing its name or its
    backend option with another: two owners declaring one name would leave a
    script unable to say whose it takes, and two settings filling one option
    would leave a run unable to say which it recorded. Raises `TypeError` if a
    declaration's `applies` is not a mapping.
    """
    declarations: dict[str, SettingDeclaration] = {}
    options: dict[str, str] = {}
    for group in groups:
        for declaration in group:
            missing = {"name", "type", "default", "help"} - set(declaration)
            if missing:
                raise ValueError(
                    f"setting declaration {declaration!r} is missing {sorted(missing)}"
                )
            name = str(declaration["name"])
            if declaration["type"] not in SETTING_TYPES:
                raise ValueError(
                    f"setting {name!r} has type {declaration['type']!r}; "
                    f"expected one of {SETTING_TYPES}"
                )
            applies = declaration.get("applies") or {}
            if not isinstance(applies, Mapping):
                raise TypeError(f"setting {name!r} applies must be a mapping")
            unknown = set(applies) - set(SETTING_DIMENSIONS)
            if unknown:
                raise ValueError(
                    f"setting {name!r} applies names unknown dimensions "
                    f"{sorted(unknown)}; expected {SETTING_DIMENSIONS}"
                )
            if name in declarations:
                raise ValueError(f"setting {name!r} is declared twice")
            option = declaration.get("option")
            if option is not None:
                if str(option) in options:
                    raise ValueError(
                        f"settings {options[str(option)]!r} and {name!r} both fill "
                        f"backend option {option!r}"
                    )
                options[str(option)] = name
            declarations[name] = declaration
    return declarations


def backend_option_surface(
    declarations: Mapping[str, SettingDeclaration],
) -> dict[str, dict[str, object]]:
    """A script mapping putting every backend setting on its
    `backend.<option>` key."""
    return {
        name: {"key": f"backend.{declaration['option']}"}
        for name, declaration in declarations.items()
        if declaration.get("option") is not None
    }


def backend_options_from_settings(
    declarations: Mapping[str, SettingDeclaration],
    context: Mapping[str, object],
    values: Mapping[str, object],
) -> dict[str, object]:
    """Every backend setting that applies under `context`, by its option key:
    its value in `values` where named there, its declared default otherwise.

    This is what a run records as its backend's settings, so every setting
    that shapes a run's result is part of its identity whether given or not.
    """
    return {
        str(declaration["option"]): (
            values[name] if name in values else declaration["default"]
        )
        for name, declaration in declarations.items()
        if declaration.get("option") is not None
        and setting_applies(declaration, context)
    }


def setting_applies(
    declaration: SettingDeclaration, context: Mapping[str, object]
) -> bool:
    """Whether `declaration` applies under `context`, dimension -> value.

    A dimension `context` does not name is not a reason to hide a setting, so a
    script sweeping several representations still reports the settings each
    of them takes.
    """
    applies = declaration.get("applies") or {}
    if not isinstance(applies, Mapping):
        return True
    for dimension, allowed in applies.items():
        current = context.get(dimension)
        if current is None:
            continue
        if current not in tuple(allowed):
            return False
    return True


class _GivenStoreAction(argparse.Action):
    """Store a flag's value and record that the caller gave it."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        _mark_given(namespace, self.dest)


def setting_was_given(args: argparse.Namespace, name: str) -> bool:
    """Whether the caller passed the declared flag for `name`, rather than
    leaving it at a default."""
    return name in getattr(args, _GIVEN_SETTINGS_ATTR, set())


def _mark_given(namespace: argparse.Namespace, dest: str) -> None:
    given = getattr(namespace, _GIVEN_SETTINGS_ATTR, None)
    if given is None:
        given = set()
        setattr(namespace, _GIVEN_SETTINGS_ATTR, given)
    given.add(dest)


def _surface_declaration(
    declarations: Mapping[str, SettingDeclaration],
    name: str,
    surface: SettingSurface,
) -> SettingDeclaration:
    if name not in declarations:
        raise ValueError(f"script maps setting {name!r}, which no owner declares")
    if ("flag" in surface) == ("key" in surface):
        raise ValueError(
            f"setting {name!r} must be mapped to exactly one of a flag or a key"
        )
    return declarations[name]


def _effective_default(
    declaration: SettingDeclaration, surface: SettingSurface
) -> object:
    return surface["default"] if "default" in surface else declaration["default"]


def _describe_default(value: object) -> str:
    if value is None:
        return "unset"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _coerce_setting(declaration: SettingDeclaration, value: object) -> object:
    """A prefixed key's value as its declared type, refusing one it cannot be.

    `parse_stage_key_value_args` has already turned the token into a bool, int,
    float or string, and turns `1`/`0` into booleans, so a numeric setting
    accepts a boolean as its integer value.
    """
    if value is None:
        return None
    name = declaration["name"]
    kind = declaration["type"]
    try:
        if kind == "bool":
            return parse_bool_literal(value)
        if kind == "int":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(value)
            return int(value) if isinstance(value, (int, float)) else int(str(value))
        if kind == "float":
            if isinstance(value, (bool, int, float)):
                return float(value)
            return float(str(value))
        return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"setting {name!r} expects {kind}, got {value!r}") from exc


def add_declared_arguments(
    parser: argparse.ArgumentParser,
    declarations: Mapping[str, SettingDeclaration],
    mapping: Mapping[str, SettingSurface],
) -> None:
    """Add a flag for every setting `mapping` puts on a flag, each naming its
    default in its help. Settings mapped to a prefixed key add nothing here;
    the script's `--additional-args` carries them."""
    for name, surface in mapping.items():
        declaration = _surface_declaration(declarations, name, surface)
        if "flag" not in surface:
            continue
        default = _effective_default(declaration, surface)
        note = f" {surface['note']}" if surface.get("note") else ""
        # A required flag has no default a caller could fall back on.
        suffix = (
            " (required)"
            if surface.get("required")
            else f" (default: {_describe_default(default)})"
        )
        help_text = f"{declaration['help']}{note}{suffix}".replace("%", "%%")
        if declaration["type"] == "bool":
            flag = str(surface["flag"])

            def _bool_value(value: str, flag: str = flag) -> bool:
                return parse_optional_bool(value, argument_name=flag)

            parser.add_argument(
                flag,
                dest=name,
                default=default,
                action=_GivenStoreAction,
                type=_bool_value,
                metavar="{true,false}",
                help=help_text,
            )
            continue
        choices: Any = declaration.get("choices")
        nargs: Any = declaration.get("nargs")
        parser.add_argument(
            str(surface["flag"]),
            dest=name,
            default=default,
            action=_GivenStoreAction,
            type=_ARGPARSE_TYPES[str(declaration["type"])],
            choices=tuple(choices) if choices is not None else None,
            nargs=nargs,
            required=bool(surface.get("required", False)),
            help=help_text,
        )


def declared_keys_help(
    declarations: Mapping[str, SettingDeclaration],
    mapping: Mapping[str, SettingSurface],
) -> str:
    """The prefixed keys `mapping` accepts, each with its default, for the help
    of the script's `--additional-args`."""
    entries = []
    for name, surface in mapping.items():
        declaration = _surface_declaration(declarations, name, surface)
        if "key" in surface:
            default = _describe_default(_effective_default(declaration, surface))
            entries.append(f"{surface['key']} (default: {default})")
    return ", ".join(entries).replace("%", "%%")


def resolve_declared_settings(
    args: argparse.Namespace,
    declarations: Mapping[str, SettingDeclaration],
    mapping: Mapping[str, SettingSurface],
    *,
    additional: Mapping[str, Mapping[str, object]] | None = None,
    context: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Every setting `mapping` takes, resolved: its `name`, `value`, `surface`,
    `source` (`SOURCE_GIVEN`, `SOURCE_DEFAULT` or `SOURCE_SCRIPT_DEFAULT`) and
    whether it `applies` under `context`.

    `additional` is `parse_stage_key_value_args`'s output for the prefixed keys.
    A setting that does not apply is still resolved, since its value still
    reaches the configuration it fills; `applies` decides only what is
    reported.
    """
    given: set[str] = getattr(args, _GIVEN_SETTINGS_ATTR, set())
    stage_options = additional or {}
    resolved: list[dict[str, object]] = []
    for name, surface in mapping.items():
        declaration = _surface_declaration(declarations, name, surface)
        default = _effective_default(declaration, surface)
        default_source = (
            SOURCE_SCRIPT_DEFAULT if "default" in surface else SOURCE_DEFAULT
        )
        if "flag" in surface:
            value = getattr(args, name)
            source = SOURCE_GIVEN if name in given else default_source
            surface_name = str(surface["flag"])
        else:
            surface_name = str(surface["key"])
            stage, _, option = surface_name.lower().partition(".")
            options = stage_options.get(stage, {})
            if option in options:
                value = _coerce_setting(declaration, options[option])
                source = SOURCE_GIVEN
            else:
                value = default
                source = default_source
        resolved.append(
            {
                "name": name,
                "value": value,
                "surface": surface_name,
                "source": source,
                "applies": setting_applies(declaration, context or {}),
            }
        )
    return resolved


def resolved_setting_values(resolved: list[dict[str, object]]) -> dict[str, object]:
    """Setting name -> value, applicable or not, for building a configuration."""
    return {str(setting["name"]): setting["value"] for setting in resolved}


def applicable_settings_record(
    resolved: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """The applicable settings as a JSON-ready record, name -> value, source and
    surface, for what a finished run writes about how it was configured."""
    return {
        str(setting["name"]): {
            "value": setting["value"],
            "source": setting["source"],
            "surface": setting["surface"],
        }
        for setting in resolved
        if setting["applies"]
    }


def report_resolved_settings(
    script_name: str, resolved: list[dict[str, object]], **extra: object
) -> None:
    """Print a dry-run report of the applicable settings, each with its value
    and where it came from, followed by any `extra` resolved values."""
    print(f"[dry-run] {script_name}: would proceed with:")
    for setting in resolved:
        if setting["applies"]:
            print(f"  {setting['name']}={setting['value']!r} ({setting['source']})")
    for key, value in extra.items():
        print(f"  {key}={value!r}")


ARGUMENT_ERRORS: tuple[type[Exception], ...] = (
    FileExistsError,
    FileNotFoundError,
    LookupError,
    ValueError,
)
"""The errors that tell the caller to change what they passed: a run that
already exists for the date given, an input, label or promoted pointer that
does not exist for the system named, a code no catalog knows, or flags that
cannot go together. `LookupError` is here for `ReferenceNotFoundError`,
`AmbiguousReferenceError`, `PointerError` and `NothingPromotedError`, none
of which is a `ValueError`."""

ARGUMENT_ERROR_EXIT_CODE = 2
"""The exit code argparse uses for a bad command line."""


def argument_error_message(error: BaseException) -> str:
    """`str(error)`, except for a `KeyError`, whose `str` is the missing key's
    repr rather than the message it was raised with."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def run_reporting_argument_errors(main: Callable[[], int]) -> int:
    """Run a script's `main`, reporting an `ARGUMENT_ERRORS` member the way
    argparse reports a missing flag, one line on stderr and exit code 2,
    instead of a traceback.

    Sits at the `__main__` boundary, so a test that calls `main()` directly
    still sees the raise. Any other exception propagates: a traceback is the
    right report for a failure that is not the command line's.
    """
    try:
        return main()
    except ARGUMENT_ERRORS as error:
        program = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "script"
        print(f"{program}: error: {argument_error_message(error)}", file=sys.stderr)
        return ARGUMENT_ERROR_EXIT_CODE
