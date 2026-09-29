from pathlib import Path

import pytest

from blocking.run_layout import (
    BlockingPairing,
    BlockingRunLocation,
    BlockingSourceKind,
    ComparisonArtefact,
    RunArtefact,
    audit_reference_for,
    blocking_pairing,
    blocking_source_placement,
    iter_blocking_comparisons,
    iter_blocking_runs,
    read_run_location,
    resolve_audit_location,
    resolve_combined_comparison_location,
    resolve_comparison_location,
    resolve_measurement_report_path,
    resolve_pairing_dir,
    resolve_run_location,
    select_blocking_runs,
)
from tests.production_records import write_record
from workspace.identity import RunKeys
from workspace.kind_layout import Kind, data_directory
from workspace.roots import WorkspaceRoots, default_workspace_roots

_ROOTS = default_workspace_roots(Path("/root"))

_KEYS = RunKeys(
    settings="ssss",
    source_population="aaaa",
    target_population="bbbb",
    truth="tttt",
    index="iiii",
)


def _location(
    *,
    roots: WorkspaceRoots = _ROOTS,
    source_system: str = "gleif",
    target_system: str = "gb",
    representation: str = "tfidf",
    keys: RunKeys = _KEYS,
):
    """One resolved run location, varying whatever the caller names."""
    return resolve_run_location(
        roots,
        source_system=source_system,
        target_system=target_system,
        representation=representation,
        keys=keys,
    )


def _write_run(
    roots: WorkspaceRoots,
    *,
    source_system: str = "gleif",
    target_system: str = "gb",
    representation: str = "tfidf",
    keys: RunKeys = _KEYS,
    finished: bool = True,
) -> Path:
    """A run on disk, with the record that marks it finished unless the
    caller asks for one that never completed."""
    location = _location(
        roots=roots,
        source_system=source_system,
        target_system=target_system,
        representation=representation,
        keys=keys,
    )
    location.directory.mkdir(parents=True, exist_ok=True)
    if finished:
        write_record(roots, location.directory)
    return location.directory


def test_resolve_audit_location_sits_beside_the_run_under_its_own_key() -> None:
    """The audit is filed under `blocking`'s `audit` leaf, keyed on the same
    key segment as the run rather than a settings digest of its own, and
    resolving it twice gives the same directory."""
    location = _location()

    first = resolve_audit_location(_ROOTS, location)
    second = resolve_audit_location(_ROOTS, location)

    assert first.directory == second.directory
    assert first.directory != location.directory
    assert first.directory.name == location.directory.name
    relative = first.directory.relative_to(_ROOTS.artifacts).parts
    assert relative[:2] == ("blocking", "audit")
    run_relative = location.directory.relative_to(_ROOTS.artifacts).parts
    assert relative[:2] != run_relative[:2]


def test_audit_reference_for_refuses_a_non_plain_source_kind() -> None:
    location = _location()
    names_location = BlockingRunLocation(
        directory=location.directory,
        pairing=BlockingPairing(
            target_system="gb",
            source_kind=BlockingSourceKind.NAMES,
            source_segment="gleif",
        ),
        representation="tfidf",
    )

    with pytest.raises(ValueError, match="not supported"):
        audit_reference_for(names_location)


def test_blocking_source_placement_files_a_system_under_its_own_data() -> None:
    assert blocking_source_placement("gleif") == (BlockingSourceKind.DATA, "gleif")


def test_blocking_source_placement_flattens_a_perturbed_selector_to_one_segment() -> (
    None
):
    """One segment, underscore-joined, so a perturbed source sits at the same
    depth as a plain one and a glob across kinds lines up; the selector's
    colons never reach a directory name, and no part can hold the separator."""
    assert blocking_source_placement("perturbed://ie/en-lite/v1/42") == (
        BlockingSourceKind.PERTURBED,
        "ie_en-lite_v1_42",
    )


def test_pairing_label_is_the_tree_s_own_three_segments() -> None:
    pairing = blocking_pairing(source_system="gleif", target_system="gb")

    assert pairing.label == "gb/data/gleif"


def test_resolve_comparison_location_sits_beside_the_pairing_runs() -> None:
    pairing_dir = resolve_pairing_dir(_ROOTS, source_system="gleif", target_system="gb")

    location = resolve_comparison_location(
        _ROOTS, source_system="gleif", target_system="gb"
    )

    assert location.directory == pairing_dir / "comparison"
    assert location.pairing is not None
    assert location.pairing.label == "gb/data/gleif"


def test_combined_comparison_sits_outside_the_tree_it_combines() -> None:
    """Beside blocking's data directory, never inside it, so concatenating
    every pairing's report can never read a previous combination of itself."""
    combined = resolve_combined_comparison_location(_ROOTS)

    assert combined.pairing is None
    assert data_directory(_ROOTS, Kind.BLOCKING) not in combined.directory.parents


def test_run_location_keys_beneath_the_pairing_and_representation() -> None:
    """Target first, then the source's kind and its own segment, then the
    representation, then the key: every run against one target sits under
    that target, and the ladder's tiers sit beside each other under it."""
    run_dir = _location().directory

    assert run_dir.parent.name == "tfidf"
    assert (
        run_dir.parent.parent
        == resolve_pairing_dir(
            _ROOTS, source_system="gleif", target_system="gb"
        ).resolve()
    )
    assert (
        run_dir.parent.parent.relative_to(
            data_directory(_ROOTS, Kind.BLOCKING).resolve()
        )
        == Path("gb") / "data" / "gleif"
    )


def test_run_location_is_stable_for_the_same_keys() -> None:
    assert _location().directory == _location().directory


def test_run_location_separates_runs_whose_keys_differ() -> None:
    other = RunKeys(
        settings=_KEYS.settings,
        source_population="cccc",
        target_population=_KEYS.target_population,
        truth=_KEYS.truth,
        index=_KEYS.index,
    )

    assert _location().directory != _location(keys=other).directory


def test_run_location_files_a_perturbed_source_at_the_same_depth() -> None:
    """A perturbed set scored against the system it was made from sits under
    that system, under the `perturbed` kind, one segment for the selector, so
    it is the same number of levels deep as a plain run against the same
    target and one glob reads both."""
    run_dir = _location(
        source_system="perturbed://ie/p1/v1/1", target_system="ie"
    ).directory

    blocking_data = data_directory(_ROOTS, Kind.BLOCKING).resolve()
    assert run_dir.parent.parent.relative_to(blocking_data) == (
        Path("ie") / "perturbed" / "ie_p1_v1_1"
    )
    assert len(run_dir.relative_to(blocking_data).parts) == len(
        _location().directory.relative_to(blocking_data).parts
    )


def test_every_artefact_a_run_holds_is_named_beneath_its_own_directory() -> None:
    location = _location()

    for artefact in RunArtefact:
        path = location.path(artefact)
        assert path.is_relative_to(location.directory)
        assert path != location.directory


def test_read_run_location_recovers_what_resolving_one_produced() -> None:
    """A script handed a run directory reads back the pairing and
    representation the writer resolved it under, rather than re-deriving them
    from the path itself."""
    resolved = _location(source_system="perturbed://ie/p1/v1/1", target_system="ie")

    read_back = read_run_location(_ROOTS, resolved.directory)

    assert read_back == resolved


def test_read_run_location_rejects_a_directory_outside_the_tree(
    tmp_path: Path,
) -> None:
    roots = default_workspace_roots(tmp_path)
    stray = tmp_path / "somewhere" / "else" / "a" / "b" / "c"
    stray.mkdir(parents=True)

    with pytest.raises(ValueError, match="not a blocking run directory"):
        read_run_location(roots, stray)


def test_iter_blocking_runs_hands_back_each_run_with_its_pairing(
    tmp_path: Path,
) -> None:
    roots = default_workspace_roots(tmp_path)
    _write_run(roots)
    _write_run(roots, source_system="perturbed://ie/p1/v1/1", target_system="ie")

    found = sorted(iter_blocking_runs(roots), key=lambda run: run.pairing.label)

    assert [run.pairing.label for run in found] == [
        "gb/data/gleif",
        "ie/perturbed/ie_p1_v1_1",
    ]
    assert {run.representation for run in found} == {"tfidf"}


def test_discovery_hands_back_exactly_what_a_writer_resolved(tmp_path: Path) -> None:
    """The writer's location and the reader's are the same object's worth of
    facts, artefact paths included, so a change to the tree or to a file name
    cannot move one without the other."""
    roots = default_workspace_roots(tmp_path)
    written = _location(roots=roots)
    _write_run(roots)

    (found,) = list(iter_blocking_runs(roots))

    assert found == written
    assert [found.path(artefact) for artefact in RunArtefact] == [
        written.path(artefact) for artefact in RunArtefact
    ]


def test_iter_blocking_runs_skips_a_run_still_being_written(tmp_path: Path) -> None:
    """The summary is written last, so a directory without one is a run in
    progress rather than a finished one to report on."""
    roots = default_workspace_roots(tmp_path)
    _write_run(roots, finished=False)

    assert list(iter_blocking_runs(roots)) == []


def test_iter_blocking_comparisons_finds_a_pairing_s_report_only(
    tmp_path: Path,
) -> None:
    roots = default_workspace_roots(tmp_path)
    _write_run(roots)
    pairing_report = resolve_comparison_location(
        roots, source_system="gleif", target_system="gb"
    )
    pairing_report.directory.mkdir(parents=True, exist_ok=True)
    pairing_report.path(ComparisonArtefact.REPORT).write_text("", encoding="utf-8")
    combined = resolve_combined_comparison_location(roots)
    combined.directory.mkdir(parents=True, exist_ok=True)
    combined.path(ComparisonArtefact.REPORT).write_text("", encoding="utf-8")

    found = list(iter_blocking_comparisons(roots))

    assert [report.directory for report in found] == [pairing_report.directory]


def test_iter_blocking_runs_reports_nothing_before_a_run_has_been_written(
    tmp_path: Path,
) -> None:
    assert list(iter_blocking_runs(default_workspace_roots(tmp_path))) == []


def test_select_blocking_runs_narrows_by_pairing_representation_and_key(
    tmp_path: Path,
) -> None:
    roots = default_workspace_roots(tmp_path)
    other_keys = RunKeys(
        settings="other",
        source_population=_KEYS.source_population,
        target_population=_KEYS.target_population,
        truth=_KEYS.truth,
        index=_KEYS.index,
    )
    first = _write_run(roots)
    second = _write_run(roots, keys=other_keys)
    _write_run(roots, representation="wordpiece")
    _write_run(roots, target_system="ie")

    pairing = select_blocking_runs(roots, source_system="gleif", target_system="gb")
    tfidf = select_blocking_runs(
        roots, source_system="gleif", target_system="gb", representation="tfidf"
    )
    keyed = select_blocking_runs(
        roots, source_system="gleif", target_system="gb", run_key=second.name
    )

    assert len(pairing) == 3
    assert sorted(run.directory for run in tfidf) == sorted([first, second])
    assert [run.directory for run in keyed] == [second]


def test_measurement_report_is_keyed_by_its_settings_beside_the_run_tree() -> None:
    report = resolve_measurement_report_path(
        _ROOTS, measurement="short_name_recall", settings={"seed": 42}
    )
    resampled = resolve_measurement_report_path(
        _ROOTS, measurement="short_name_recall", settings={"seed": 7}
    )

    assert report != resampled
    assert report == resolve_measurement_report_path(
        _ROOTS, measurement="short_name_recall", settings={"seed": 42}
    )
    assert report.parent.parent.name == "short_name_recall"
    assert data_directory(_ROOTS, Kind.BLOCKING) not in report.parents
