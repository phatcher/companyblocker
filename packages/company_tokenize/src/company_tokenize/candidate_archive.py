"""An archive of tokenizers kept for comparison, beside a scope's working tokenizer.

The archive is an `archive/` directory in the scope's working folder, indexed by
`index.json`. Plain training archives the candidate it stored as `naive` and an
optimize promotion the one it stored as `promoted`, each with the stored candidate's
reference; an optimize candidate is copied out of its run before a later run can
purge it; anything else worth keeping is archived by hand. Which candidate is
promoted is read from the pointer, never from the archive.

`resolve_archive_label()` and `resolve_archived_tokenizer_path()` turn a label into the
archived model file, so a workload can run against an archived candidate; an unknown
or ambiguous label raises with the real labels listed rather than falling back to
the operational tokenizer. `compare_archive_entries()` compares two entries' metrics
and gives a verdict (improved, worse or a trade-off) with per-metric deltas.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .optimize import evaluate_tokenizer_metrics
from .paths import resolve_tokenizer_paths, to_portable_path_str
from .training import (
    compute_corpus_content_hash,
    resolve_candidate_model_suffix_for_trainer,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_dir,
    resolve_optimize_sweep_paths,
)

ARCHIVE_DIRNAME = "archive"
INDEX_FILENAME = "index.json"

# What an archived tokenizer *is*, so a reader can tell training experiment
# from a stored candidate without inferring it from file layout:
#   - "promoted": the candidate the optimizer's promotion stored and pointed
#     `promoted` at. Its entry carries that candidate's `reference`.
#   - "naive": the candidate plain training stored and pointed `naive` at,
#     with its `reference`.
#   - "optimize_candidate": one point from an elbow-optimize sweep, copied
#     out of its sweep leaf before a later, differently-configured sweep
#     purges it.
#   - "manual": any other tokenizer worth keeping for comparison.
# Which candidate is promoted is the pointer's to say, never the archive's.
ArchiveEntryKind = str  # "promoted" | "naive" | "optimize_candidate" | "manual"


def resolve_archive_dir(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
) -> Path:
    # Takes an already-resolved `tokenizer_root` rather than deriving
    # `artifacts/tokenizers` from a project root itself -- this package stays
    # dependency-free of `workspace` (`tach.toml`'s packages-may-not-depend-on-
    # workspace rule), so the caller resolves that root (typically via
    # `workspace.artifact_layout.tokenizer_artifact_root`) and hands it in,
    # the same shape `paths.resolve_tokenizer_paths` already takes.
    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_root, scope=scope, system=system, profile=profile
    )
    return tokenizer_paths.directory / ARCHIVE_DIRNAME


def resolve_candidate_model_path(
    *,
    optimize_models_dir: Path,
    trainer: str,
    seed: int,
    vocab_requested: int,
    min_frequency: int,
) -> Path:
    """Build an optimize candidate's model path from its identifying keys.

    Mirrors the naming scheme `optimize_execution.build_optimize_candidate_job`
    uses when writing candidates (`vocab_requested == -1` -> the literal
    `"auto"` tag, not the resolved vocab size).
    """
    vocab_tag = "auto" if vocab_requested == -1 else str(vocab_requested)
    suffix = resolve_candidate_model_suffix_for_trainer(trainer=trainer)
    return (
        optimize_models_dir
        / f"{trainer}_seed_{seed}_v{vocab_tag}_mf{min_frequency}.{suffix}"
    )


def read_archive_index(archive_dir: Path) -> list[dict[str, Any]]:
    index_path = archive_dir / INDEX_FILENAME
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    return list(entries) if isinstance(entries, list) else []


def write_archive_index(archive_dir: Path, entries: list[dict[str, Any]]) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    index_path = archive_dir / INDEX_FILENAME
    index_path.write_text(
        json.dumps({"entries": entries}, indent=2, default=str), encoding="utf-8"
    )


# Matches the auto-generated label shapes from before archive labels dropped
# their embedded timestamp. The trainer-variant segment is optional because
# it was only added partway through the same session that introduced these
# labels in the first place -- the earliest entries look like
# `trained_20260826T165855Z-882590ed` or `promoted_opt-20260826T165953Z-7b5e2741`
# (no variant at all), later ones like
# `trained_sentencepiece_20260826T215510Z-37abd895` (with one). Never matches
# a manually-chosen descriptive label like `v25000_mf1_train_split_fix_winner`,
# since none of those contain a `YYYYMMDDTHHMMSSZ`-shaped substring.
_LEGACY_TIMESTAMPED_LABEL_RE = re.compile(
    r"^(?P<prefix>promoted|trained)_(?:(?P<variant>.+?)_)?(?:opt-)?"
    r"\d{8}T\d{6}Z-(?P<hex>[0-9a-f]{8})$"
)


def migrate_legacy_archive_labels(archive_dir: Path) -> list[tuple[str, str]]:
    """Rename archive entries from the old timestamp label format to the current one.

    Renames in place, preserving each entry's original random hash suffix, so nothing is
    lost. Idempotent: entries already in the new format, or manually labeled, are left
    untouched, so this is safe to re-run.
    """
    entries = read_archive_index(archive_dir)
    renamed: list[tuple[str, str]] = []
    for entry in entries:
        old_label = entry.get("label")
        if not isinstance(old_label, str):
            continue
        match = _LEGACY_TIMESTAMPED_LABEL_RE.match(old_label)
        if match is None:
            continue
        variant = match["variant"]
        if variant is None:
            # Earliest label scheme predates embedding the trainer in the
            # label at all -- recover it from the entry's own trainer field
            # rather than leaving the migrated label variant-less.
            entry_trainer = entry.get("trainer")
            variant = (
                entry_trainer
                if isinstance(entry_trainer, str) and entry_trainer
                else "unknown"
            )
        new_label = f"{match['prefix']}_{variant}_{match['hex']}"
        if new_label == old_label:
            continue

        for old_path in archive_dir.glob(f"{old_label}.*"):
            new_path = old_path.with_name(
                old_path.name.replace(old_label, new_label, 1)
            )
            if new_path.exists():
                raise FileExistsError(
                    "Migration target already exists, refusing to overwrite: "
                    f"{new_path}"
                )
            old_path.rename(new_path)

        entry["label"] = new_label
        metadata_path = archive_dir / f"{new_label}.metadata.json"
        if metadata_path.exists():
            metadata_path.write_text(
                json.dumps(entry, indent=2, default=str), encoding="utf-8"
            )
        renamed.append((old_label, new_label))

    if renamed:
        write_archive_index(archive_dir, entries)

    return renamed


def attach_report_to_archive_entry(
    *,
    archive_dir: Path,
    label: str,
    report_markdown: str,
    image_files: dict[str, Path] | None = None,
    report_filename: str | None = None,
) -> Path:
    """Attach a generated presentation report to an archived entry.

    Uses the same label-keyed file family the archive already has for
    `<label>.metadata.json`/`<label>.summary.json`, extended rather than replaced, per
    this repo's precedent of one archive mechanism rather than a parallel one.

    `image_files` maps a destination filename (as referenced by
    `report_markdown`'s image links) to its current source path; each is
    copied into `archive_dir` so the report's relative image links keep
    working wherever the archive directory is copied or moved, independent of
    the location the image was originally rendered to -- often the
    `optimize/` scratch directory, which a later, differently-configured
    sweep purges.

    Records the report's filename on the matching index.json entry
    (`report_path`) so a reader can tell a report exists without probing the
    filesystem. Raises KeyError if `label` has no existing index entry --
    call this *after* archiving the tokenizer/candidate itself, not before.
    """
    entries = read_archive_index(archive_dir)
    matching = [entry for entry in entries if entry.get("label") == label]
    if not matching:
        raise KeyError(f"No archived entry labeled '{label}' in {archive_dir}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    for dest_name, source_path in (image_files or {}).items():
        shutil.copy2(source_path, archive_dir / dest_name)

    filename = report_filename or f"{label}.report.md"
    report_path = archive_dir / filename
    report_path.write_text(report_markdown, encoding="utf-8")

    matching[0]["report_path"] = filename
    write_archive_index(archive_dir, entries)
    return report_path


def resolve_archive_label(
    *,
    archive_dir: Path,
    trainer: str | None = None,
    label: str | None = None,
    latest: bool = False,
) -> tuple[str, str]:
    """Resolve a `--label`/`--latest` CLI choice against real archive entries.

    Returns `(label, trainer)`, so a caller can script against the archive without first
    inspecting `index.json` or already knowing a full generated label such as
    `promoted_wordpiece_7b5e2741`.

    Exactly one of `label`/`latest` must be given. `trainer` is an optional
    narrowing filter, not a requirement: entries from every trainer are
    considered when omitted, and the matched entry's own `trainer` field is
    returned, so a caller doesn't need to know it up front either.

    - `latest=True`: the single most recently archived entry (by
      `archived_utc`) among the (optionally trainer-filtered) entries, of any
      kind. It is not the promoted one, which only the pointer names.
    - `label`: an exact label match wins outright; otherwise a
      case-insensitive substring match (matches a trailing hash slug like
      `7b5e274` against `promoted_wordpiece_7b5e2741` just as well as a
      leading one).

    Raises `ValueError` if both or neither of `label`/`latest` are given, if
    no entries match the (optional) trainer filter, or if `label` resolves
    to zero or more than one entry -- listing the real candidate labels
    either way, and suggesting `trainer=` as a disambiguator when the
    ambiguity spans more than one trainer, rather than a bare `KeyError`.
    """
    if (label is None) == (not latest):
        raise ValueError("Provide exactly one of label= or latest=True.")

    entries = read_archive_index(archive_dir)
    if trainer is not None:
        entries = [entry for entry in entries if entry.get("trainer") == trainer]
    if not entries:
        scope = f"trainer={trainer!r} " if trainer is not None else ""
        raise ValueError(f"No archived entries {scope}in {archive_dir}.")

    if latest:
        newest = max(entries, key=lambda entry: str(entry.get("archived_utc", "")))
        return str(newest["label"]), str(newest["trainer"])

    assert label is not None  # nosec B101 - narrowed by the exactly-one check above
    exact = [entry for entry in entries if entry.get("label") == label]
    if exact:
        return label, str(exact[0]["trainer"])

    needle = label.lower()
    matches = [
        entry for entry in entries if needle in str(entry.get("label", "")).lower()
    ]
    if len(matches) == 1:
        return str(matches[0]["label"]), str(matches[0]["trainer"])

    available = ", ".join(sorted(str(entry.get("label")) for entry in entries))
    if not matches:
        raise ValueError(
            f"No archived entry matches {label!r} in {archive_dir}. "
            f"Available: {available}."
        )
    ambiguous = ", ".join(sorted(str(entry.get("label")) for entry in matches))
    trainers_involved = sorted({str(entry.get("trainer")) for entry in matches})
    hint = (
        f" Narrow with trainer= ({', '.join(trainers_involved)})."
        if trainer is None and len(trainers_involved) > 1
        else " Be more specific."
    )
    raise ValueError(
        f"{label!r} matches more than one archived entry: {ambiguous}.{hint}"
    )


@dataclass(frozen=True)
class ArchivedTokenizerSelection:
    """One archived entry resolved to a model file a workload can run against."""

    label: str
    trainer: str
    model_path: Path
    archive_dir: Path


def resolve_archived_tokenizer_path(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    label: str,
    trainer: str | None = None,
) -> ArchivedTokenizerSelection:
    """Resolve a named archived candidate to its on-disk model file.

    The opt-in counterpart to `resolve_tokenizer_paths`, which only ever
    resolves a scope's working tree, `artifacts/tokenizers/work/<scope>/`. A workload that passes no label keeps
    resolving that operational tokenizer unchanged; one that passes a label
    runs against that archived entry instead, which is what makes two
    candidates comparable end to end rather than only by their own training
    metrics.

    `label` is resolved through `resolve_archive_label`, so a trailing hash
    slug works as well as a full label, and an unknown or ambiguous one
    raises `ValueError` listing the archive's real labels -- never a silent
    fall back to the operational tokenizer. `trainer`, when given, narrows
    the search to that trainer's entries; the matched entry's own trainer
    decides the model file's extension (`.model` for sentencepiece, `.json`
    otherwise), so a caller doesn't need `resolve_tokenizer_path_for_trainer`
    on top of this.

    Read-only: nothing here writes to the archive or its index, so the
    project-relative posix paths stored in `index.json` are untouched. The
    returned `model_path` is a runtime path for the current machine and is
    deliberately absolute when `tokenizer_root` is.
    """
    archive_dir = resolve_archive_dir(
        tokenizer_root=tokenizer_root, scope=scope, system=system, profile=profile
    )
    resolved_label, entry_trainer = resolve_archive_label(
        archive_dir=archive_dir, trainer=trainer, label=label
    )
    suffix = resolve_candidate_model_suffix_for_trainer(trainer=entry_trainer)
    model_path = archive_dir / f"{resolved_label}.{suffix}"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Archived entry '{resolved_label}' is indexed in {archive_dir}, but its "
            f"model file is missing: {model_path}"
        )
    return ArchivedTokenizerSelection(
        label=resolved_label,
        trainer=entry_trainer,
        model_path=model_path,
        archive_dir=archive_dir,
    )


def _add_archive_entry(*, archive_dir: Path, entry: dict[str, object]) -> None:
    """Insert or replace one index entry by label."""
    entries = read_archive_index(archive_dir)
    entries = [item for item in entries if item.get("label") != entry.get("label")]
    entries.append(entry)
    write_archive_index(archive_dir, entries)


def _archive_tokenizer_file(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None,
    profile: str,
    trainer: str,
    source_tokenizer_path: Path,
    label: str,
    kind: ArchiveEntryKind,
    fertility_target: float,
    notes: str | None,
    identity: dict[str, object],
    precomputed_metrics: dict[str, float] | None = None,
    summary: dict[str, object] | None = None,
) -> Path:
    if not source_tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {source_tokenizer_path}")

    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_root, scope=scope, system=system, profile=profile
    )
    archive_dir = resolve_archive_dir(
        tokenizer_root=tokenizer_root, scope=scope, system=system, profile=profile
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    suffix = resolve_candidate_model_suffix_for_trainer(trainer=trainer)
    dest_model_path = archive_dir / f"{label}.{suffix}"
    shutil.copy2(source_tokenizer_path, dest_model_path)

    # Metrics are computed fresh, uniformly, against the tokenizer's own
    # training corpus for every entry regardless of where it came from --
    # deliberately not reused from optimize's run log (validation-split
    # metrics) or from a prior promotion's summary (which may not exist for
    # a naive baseline never run through optimize). This is what makes
    # entries actually comparable to each other. A caller that already
    # computed metrics on this exact file against this exact corpus (e.g.
    # finalize_optimize_target, right before promoting) can pass them in via
    # precomputed_metrics to skip paying for a second full-corpus pass.
    if precomputed_metrics is not None:
        metrics = precomputed_metrics
    else:
        metrics, _, _, _ = evaluate_tokenizer_metrics(
            corpus_path=tokenizer_paths.corpus_path,
            tokenizer_path=dest_model_path,
            trainer=trainer,
            fertility_target=fertility_target,
            diagnostic_top_n=1,
        )

    entry: dict[str, object] = {
        "label": label,
        "kind": kind,
        "archived_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trainer": trainer,
        "scope": scope,
        "system": system,
        "profile": profile,
        # Portable relative to `tokenizer_root`, the one directory this
        # package is handed: where that root sits is its caller's business.
        "source_tokenizer_path": to_portable_path_str(
            source_tokenizer_path, project_root=tokenizer_root
        ),
        "notes": notes,
        "metrics": metrics,
        **identity,
    }
    _add_archive_entry(archive_dir=archive_dir, entry=entry)

    metadata_path = archive_dir / f"{label}.metadata.json"
    metadata_path.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")

    if summary is not None:
        summary_path = archive_dir / f"{label}.summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    return dest_model_path


def archive_optimize_candidate(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    trainer: str = "wordpiece",
    tokenizer_encoding: str | None = None,
    seed: int,
    vocab_requested: int,
    min_frequency: int,
    label: str,
    fertility_target: float = 1.25,
    notes: str | None = None,
) -> Path:
    """Copy one optimize candidate out of its sweep leaf before archiving it.

    The leaf is keyed by (corpus, trainer, model-variant, grid): see
    `resolve_optimize_sweep_paths`. A bpe and a unigram sweep for the same trainer never
    share a leaf, so archiving one never risks a differently-configured sweep purging it
    out from under you; this only fails if the candidate
    genuinely was never trained under this exact (seed, vocab, min_frequency)
    for this trainer/model-variant.
    """
    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_root,
        scope=scope,
        system=system,
        profile=profile,
    )
    optimize_dir = resolve_optimize_dir(
        scope_directory=tokenizer_paths.directory,
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"No optimize sweep recorded for trainer={trainer} "
            f"tokenizer_encoding={tokenizer_encoding or 'bpe'} at {pointer_path} -- run a "
            "sweep first, or check --tokenizer/--encoding match what was swept."
        )
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    corpus_content_hash = compute_corpus_content_hash(tokenizer_paths.corpus_path)
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=corpus_content_hash,
        grid_hash=str(pointer_payload["grid_hash"]),
    )
    source_path = resolve_candidate_model_path(
        optimize_models_dir=sweep_paths.models_dir,
        trainer=trainer,
        seed=seed,
        vocab_requested=vocab_requested,
        min_frequency=min_frequency,
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {source_path}. The sweep for "
            f"trainer={trainer} tokenizer_encoding={tokenizer_encoding or 'bpe'} is recorded, "
            "but this exact seed/vocab/min_frequency combination wasn't trained in "
            "it -- double check the identifying keys, or that the corpus hasn't "
            "changed since this sweep ran (a resampled corpus is a different sweep "
            "leaf entirely)."
        )

    return _archive_tokenizer_file(
        tokenizer_root=tokenizer_root,
        scope=scope,
        system=system,
        profile=profile,
        trainer=trainer,
        source_tokenizer_path=source_path,
        label=label,
        kind="optimize_candidate",
        fertility_target=fertility_target,
        notes=notes,
        identity={
            "seed": seed,
            "vocab_size_requested": vocab_requested,
            "min_frequency": min_frequency,
        },
    )


def archive_promoted_tokenizer(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    trainer: str = "wordpiece",
    source_tokenizer_path: Path,
    label: str,
    reference: str,
    fertility_target: float = 1.25,
    notes: str | None = None,
    precomputed_metrics: dict[str, float] | None = None,
    summary: dict[str, object] | None = None,
) -> Path:
    """Archive the candidate the optimizer's promotion stored under `reference`.

    The entry carries `reference`, so a reader holding the reference the
    pointer names for `promoted` finds this entry's metrics and summary.

    `precomputed_metrics`, when given, skips a redundant full-corpus
    `evaluate_tokenizer_metrics` pass -- the optimize path already has these
    from its own retrain-and-evaluate step. `summary`, when given, is
    written verbatim as `<label>.summary.json` alongside the model and
    `.metadata.json`, so the evidence for *why* a sweep winner was chosen
    (candidate_stats, elbow diagnostics) survives as durably as the model
    file itself, not just as a single mutable `optimize_summary.json` the
    next `--finalize` will overwrite.
    """
    return _archive_tokenizer_file(
        tokenizer_root=tokenizer_root,
        scope=scope,
        system=system,
        profile=profile,
        trainer=trainer,
        source_tokenizer_path=source_tokenizer_path,
        label=label,
        kind="promoted",
        fertility_target=fertility_target,
        notes=notes,
        identity={"reference": reference},
        precomputed_metrics=precomputed_metrics,
        summary=summary,
    )


def archive_naive_tokenizer(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    trainer: str = "wordpiece",
    source_tokenizer_path: Path,
    label: str,
    reference: str,
    fertility_target: float = 1.25,
    notes: str | None = None,
) -> Path:
    """Archive the candidate plain training stored under `reference`.

    Plain training is a first guess at the settings, not a choice among
    candidates, so its entry is kind `naive` and never reads as promoted.
    """
    return _archive_tokenizer_file(
        tokenizer_root=tokenizer_root,
        scope=scope,
        system=system,
        profile=profile,
        trainer=trainer,
        source_tokenizer_path=source_tokenizer_path,
        label=label,
        kind="naive",
        fertility_target=fertility_target,
        notes=notes,
        identity={"reference": reference},
    )


def archive_trained_tokenizer(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    trainer: str = "wordpiece",
    source_tokenizer_path: Path,
    label: str,
    fertility_target: float = 1.25,
    notes: str | None = None,
) -> Path:
    """Archive any other tokenizer worth keeping for comparison.

    For example a naive fixed-vocab baseline trained outside the optimize sweep.
    """
    return _archive_tokenizer_file(
        tokenizer_root=tokenizer_root,
        scope=scope,
        system=system,
        profile=profile,
        trainer=trainer,
        source_tokenizer_path=source_tokenizer_path,
        label=label,
        kind="manual",
        fertility_target=fertility_target,
        notes=notes,
        identity={},
    )


# Lower is better, and scored toward the verdict. Deliberately excludes
# `fertility` and `token_count_mean`: fertility is supposed to sit close to
# a *target*, not be minimized, and `fertility_distance` already is that
# deviation -- scoring raw `fertility` too double-counts the same signal
# with the wrong sign whenever fertility moves toward the target from below.
# token_count_mean is the same kind of context number, not an independent
# cost. Both are still reported for context, just not tallied.
_SCORED_METRIC_KEYS: tuple[str, ...] = (
    "fertility_distance",
    "unk_rate",
    "single_char_token_pct",
)
_CONTEXT_METRIC_KEYS: tuple[str, ...] = ("fertility", "token_count_mean")


def compare_archive_entries(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    label_a: str,
    label_b: str,
) -> dict[str, object]:
    """Compare two archived entries' metrics and give a plain-language read.

    Answers "does the new one actually improve things": lower is better for
    the *scored* metrics (fertility_distance, unk_rate, single_char_token_pct
    -- genuine cost/deviation measures), so B is called the winner only if
    it's better-or-equal on every scored metric, worse if the reverse, and a
    tradeoff otherwise. `fertility` and `token_count_mean` are reported for
    context but not scored -- fertility is supposed to sit close to a
    target, not be minimized, and `fertility_distance` already is that
    deviation; scoring both double-counts the same signal.
    """
    archive_dir = resolve_archive_dir(
        tokenizer_root=tokenizer_root, scope=scope, system=system, profile=profile
    )
    entries_by_label = {
        str(entry.get("label")): entry for entry in read_archive_index(archive_dir)
    }
    if label_a not in entries_by_label:
        raise KeyError(f"No archived entry labeled '{label_a}' in {archive_dir}")
    if label_b not in entries_by_label:
        raise KeyError(f"No archived entry labeled '{label_b}' in {archive_dir}")

    entry_a = entries_by_label[label_a]
    entry_b = entries_by_label[label_b]
    metrics_a = entry_a.get("metrics", {}) or {}
    metrics_b = entry_b.get("metrics", {}) or {}

    deltas: dict[str, dict[str, float | None]] = {}
    better_count = 0
    worse_count = 0
    for key in (*_SCORED_METRIC_KEYS, *_CONTEXT_METRIC_KEYS):
        value_a = metrics_a.get(key)
        value_b = metrics_b.get(key)
        if not isinstance(value_a, (int, float)) or not isinstance(
            value_b, (int, float)
        ):
            deltas[key] = {"a": value_a, "b": value_b, "delta": None}
            continue
        delta = float(value_b) - float(value_a)
        deltas[key] = {"a": float(value_a), "b": float(value_b), "delta": delta}
        if key not in _SCORED_METRIC_KEYS:
            continue
        if delta < 0:
            better_count += 1
        elif delta > 0:
            worse_count += 1

    vocab_a = entry_a.get("vocab_size_requested")
    vocab_b = entry_b.get("vocab_size_requested")
    if better_count > 0 and worse_count == 0:
        verdict = f"'{label_b}' improves on '{label_a}' across every compared metric."
    elif worse_count > 0 and better_count == 0:
        verdict = f"'{label_b}' is worse than '{label_a}' across every compared metric."
    else:
        verdict = (
            f"'{label_b}' vs '{label_a}' is a tradeoff: better on {better_count} "
            f"metric(s), worse on {worse_count} -- check the per-metric deltas."
        )
    if isinstance(vocab_a, (int, float)) and isinstance(vocab_b, (int, float)):
        vocab_delta_pct = (
            (float(vocab_b) - float(vocab_a)) / float(vocab_a) * 100.0
            if vocab_a
            else None
        )
        if vocab_delta_pct is not None:
            verdict += f" Vocab size change: {vocab_delta_pct:+.1f}%."

    return {
        "label_a": label_a,
        "label_b": label_b,
        "kind_a": entry_a.get("kind"),
        "kind_b": entry_b.get("kind"),
        "vocab_size_requested_a": vocab_a,
        "vocab_size_requested_b": vocab_b,
        "min_frequency_a": entry_a.get("min_frequency"),
        "min_frequency_b": entry_b.get("min_frequency"),
        "metric_deltas": deltas,
        "better_count": better_count,
        "worse_count": worse_count,
        "verdict": verdict,
    }
