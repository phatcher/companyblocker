from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from acquisition import pipeline, pipeline_resources, pipeline_selection
from acquisition.downloader import DEFAULT_USER_AGENT
from acquisition.models import ResearchPlan
from acquisition.registry import get_system_plan
from workspace.data_layout import data_root
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def test_build_paths_and_normalize_helpers(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    paths = pipeline.build_paths(workspace_roots, "gb")
    assert paths.data_root == data_root(workspace_roots)
    assert paths.acquire_dir == layer_fixture_dir("gb", layer="acquire")
    assert paths.prepare_dir == layer_fixture_dir("gb", layer="prepare")
    assert paths.source_dir == layer_fixture_dir("gb", layer="source")
    assert paths.canonical_dir == layer_fixture_dir("gb", layer="canonical")

    assert pipeline.normalize_systems("GB, ie") == ["gb", "ie"]
    assert pipeline.normalize_systems(["GB", "ie,nl"]) == ["gb", "ie", "nl"]

    with pytest.raises(ValueError, match="At least one system code"):
        pipeline.normalize_systems("  ,  ")


def test_supported_system_codes_reads_the_all_flag_not_the_status(monkeypatch):
    """`all` membership comes from each plan's `all_systems_target`, so a
    `live` system flagged out stays out and a `supported` one flagged in comes
    along. Reading `status` instead was wrong in both directions."""

    def plan(status: str, *, in_all: bool) -> SimpleNamespace:
        return SimpleNamespace(status=status, all_systems_target=in_all)

    monkeypatch.setattr(
        pipeline_selection,
        "COUNTRY_REGISTRY",
        {
            "gb": plan("live", in_all=True),
            "dk": plan("supported", in_all=True),
            "nl": plan("research_required", in_all=False),
        },
    )
    monkeypatch.setattr(
        pipeline_selection,
        "SYSTEM_REGISTRY",
        {
            "gleif": plan("live", in_all=True),
            "perturbed": plan("live", in_all=False),
        },
    )

    assert pipeline_selection.all_target_system_codes() == ["dk", "gb", "gleif"]
    assert pipeline_selection.all_target_code_lists() == (["dk", "gb"], ["gleif"])
    assert pipeline._supported_system_codes() == ["dk", "gb", "gleif"]


def test_normalize_systems_expands_all_from_the_committed_catalog():
    """The real catalog's answer, end to end: `perturbed` (live, but generated
    rather than acquired) is out and `wikidata` is in."""
    expanded = pipeline.normalize_systems("all")

    assert expanded == ["fr", "gb", "gleif", "ie", "offeneregister", "wikidata"]


def test_normalize_systems_expands_all_via_its_resolver(mocker):
    supported_system_codes = mocker.patch.object(
        pipeline,
        "_supported_system_codes",
        return_value=["fr", "gb", "gleif"],
    )

    assert pipeline.normalize_systems("all") == ["fr", "gb", "gleif"]
    assert pipeline.normalize_systems(["all", "zz"]) == ["fr", "gb", "gleif", "zz"]
    assert pipeline.normalize_systems("all", expand_all=False) == ["all"]
    assert supported_system_codes.call_count == 2


def test_resolve_system_selection_for_all_and_global(mocker):
    supported_system_codes = mocker.patch.object(
        pipeline,
        "_supported_system_codes",
        return_value=["fr", "gb"],
    )

    all_selection = pipeline.resolve_system_selection(
        ["all"],
        add_global_for_all=True,
        global_expands_to_all=True,
    )
    assert all_selection.requested == ["all"]
    assert all_selection.concrete == ["fr", "gb"]
    assert all_selection.include_global_target is True

    global_selection = pipeline.resolve_system_selection(
        ["global"],
        add_global_for_all=True,
        global_expands_to_all=True,
    )
    assert global_selection.requested == ["global"]
    assert global_selection.concrete == ["fr", "gb"]
    assert global_selection.include_global_target is True
    assert supported_system_codes.call_count == 2


def test_validate_requested_systems_rejects_unknown_code_but_allows_keywords():
    """`all`/`global` are selection keywords, checked against nothing; every
    other code goes through the injected lookup."""
    calls: list[str] = []

    def fake_get_system_plan(code: str):
        calls.append(code)
        if code == "glief":
            raise KeyError("Unknown system 'glief'. Known values: gb, ie")
        return SimpleNamespace(code=code)

    pipeline_selection.validate_requested_systems(
        ["all", "global", "gb"], get_system_plan_fn=fake_get_system_plan
    )
    assert calls == ["gb"]

    with pytest.raises(ValueError, match="Unknown system 'glief'"):
        pipeline_selection.validate_requested_systems(
            ["glief"], get_system_plan_fn=fake_get_system_plan
        )


def test_validate_requested_systems_against_the_committed_catalog():
    """End to end against the real registry: a typo names itself and every
    known code; a real code and the `all`/`global` keywords pass."""
    pipeline.validate_requested_systems(["gb", "all", "global"])

    with pytest.raises(ValueError, match="Unknown system 'glief'\\. Known values:"):
        pipeline.validate_requested_systems(["glief"])


def test_parse_iso_date_cases():
    assert pipeline._parse_iso_date(None) is None
    assert pipeline._parse_iso_date("") is None
    assert pipeline._parse_iso_date("2026-06-20T12:00:00Z") == date(2026, 6, 20)
    assert pipeline._parse_iso_date("not-a-date") is None


def test_latest_acquired_source_file_handles_missing_and_valid_directories(
    tmp_path: Path,
    layer_fixture_dir,
):
    resource = SimpleNamespace(resolve_file_name=lambda run_date: "bulk.json.gz")

    missing_date, missing_path = pipeline_resources._latest_acquired_source_file(
        layer_fixture_dir("gb", layer="acquire"),
        resource=resource,
        parse_iso_date_fn=pipeline._parse_iso_date,
    )
    assert missing_date is None
    assert missing_path is None

    acquire_root = layer_fixture_dir("gb", layer="acquire")
    first_dir = acquire_root / "2026-06-01"
    first_dir.mkdir(parents=True, exist_ok=True)
    (first_dir / "bulk.json.gz").write_bytes(b"first")

    second_dir = acquire_root / "2026-06-02"
    second_dir.mkdir(parents=True, exist_ok=True)
    (second_dir / "bulk.json.gz").write_bytes(b"second")

    latest_date, latest_path = pipeline_resources._latest_acquired_source_file(
        acquire_root,
        resource=resource,
        parse_iso_date_fn=pipeline._parse_iso_date,
    )
    assert latest_date == date(2026, 6, 2)
    assert latest_path == second_dir / "bulk.json.gz"


def test_run_acquisition_dry_run(tmp_path: Path, workspace_roots: WorkspaceRoots):
    results = pipeline.run_acquisition(
        ["gb"], roots=workspace_roots, run_date="2026-06-15", dry_run=True
    )
    assert len(results) == 1
    assert results[0].status == "dry_run"
    assert "would download" in results[0].message


def test_run_acquisition_supported_status_runs_via_dry_run_without_allow_research(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
):
    """`supported` (dk's real catalog status) must clear the runnable-status
    gate on its own, exactly like `live` -- no `allow_research` flag required.

    `dk` is the subject because its real catalog status is `supported`; any
    other still-`supported` catalog system exercises the same gate."""
    assert get_system_plan("dk").status == "supported"

    results = pipeline.run_acquisition(
        ["dk"], roots=workspace_roots, run_date="2026-06-15", dry_run=True
    )
    assert len(results) == 1
    assert results[0].status == "dry_run"


def test_run_acquisition_unsupported_plan_skip_and_raise(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    supported_plan = get_system_plan("gb")
    unsupported_plan = replace(
        supported_plan,
        code="xx",
        display_name="Unsupported",
        status="research_required",
        all_systems_target=False,
        notes="Needs research",
        research=ResearchPlan(),
    )

    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=unsupported_plan
    )

    skipped = pipeline.run_acquisition(
        ["xx"], roots=workspace_roots, run_date="2026-06-15", skip_unsupported=True
    )
    assert skipped[0].status == "research_required"

    with pytest.raises(RuntimeError, match="Needs research"):
        pipeline.run_acquisition(
            ["xx"], roots=workspace_roots, run_date="2026-06-15", skip_unsupported=False
        )
    assert get_system_plan_mock.call_count == 2


def test_run_acquisition_blocked_plan_rejected_even_with_allow_research(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    """`blocked` is never folded into the runnable-status set the way
    `research_required` is -- `allow_research=True` must not let a blocked
    plan through, even paired with a research policy that would otherwise
    permit the `acquire` stage."""
    base_plan = get_system_plan("gb")
    blocked_plan = replace(
        base_plan,
        code="xx",
        display_name="Blocked",
        status="blocked",
        all_systems_target=False,
        notes="No access",
        research=ResearchPlan(allow_research_runtime=True, allowed_stages=("acquire",)),
    )

    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=blocked_plan
    )

    skipped = pipeline.run_acquisition(
        ["xx"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        skip_unsupported=True,
    )
    assert skipped[0].status == "blocked"
    assert "No access" in skipped[0].message

    with pytest.raises(RuntimeError, match="No access"):
        pipeline.run_acquisition(
            ["xx"],
            roots=workspace_roots,
            run_date="2026-06-15",
            allow_research=True,
            skip_unsupported=False,
        )
    assert get_system_plan_mock.call_count == 2


def test_run_acquisition_allow_research_runs_resource_download(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}
    base_plan = get_system_plan("gb")
    research_plan = replace(
        base_plan,
        code="dbpedia",
        status="research_required",
        all_systems_target=False,
        notes="Research run",
        research=ResearchPlan(allow_research_runtime=True, allowed_stages=("acquire",)),
    )

    def fake_download_resource(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT, chunk_size=1024 * 1024
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"dbpedia-bytes")
        captured["downloaded"] = True
        return "abcd"

    download_resource = mocker.Mock(side_effect=fake_download_resource)
    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=research_plan
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    skipped = pipeline.run_acquisition(
        ["dbpedia"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )
    assert skipped[0].status == "research_required"

    results = pipeline.run_acquisition(
        ["dbpedia"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        adapter_handlers=adapter_handlers,
    )

    assert results[0].status == "downloaded"
    assert captured.get("downloaded") is True
    download_resource.assert_called_once()
    assert get_system_plan_mock.call_count == 2


def test_run_acquisition_wires_wikidata_bulk_download(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}
    base_plan = get_system_plan("gb")
    wikidata_plan = replace(
        base_plan,
        code="wikidata",
        status="research_required",
        all_systems_target=False,
        notes="Wikidata acquisition run",
        resources=(
            replace(
                base_plan.resources[0],
                name="wikidata_entities_latest_all",
                url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz",
                file_format="jsonl",
                file_name="wikidata-all.json.gz",
                file_name_template=None,
                snapshot_date_mode="run_date",
                adapter="direct_download",
            ),
        ),
        research=ResearchPlan(allow_research_runtime=True, allowed_stages=("acquire",)),
    )

    def fake_download_resource(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT, chunk_size=1024 * 1024
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bz2-placeholder")
        captured["downloaded"] = (
            resource.name,
            destination.name,
            user_agent,
            chunk_size,
        )
        return "wikidata-hash"

    def fake_extract_wikidata_company_projection(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        scan_mode="full",
        cleanup_intermediate=True,
    ):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            '{"id":"Q1","label_en":"Alpha"}\n', encoding="utf-8"
        )
        captured["projected"] = (source_path.name, destination_path.name)
        return 1

    download_resource = mocker.Mock(side_effect=fake_download_resource)
    extract_wikidata_company_projection_two_pass = mocker.patch.object(
        pipeline,
        "extract_wikidata_company_projection_two_pass",
        side_effect=fake_extract_wikidata_company_projection,
    )
    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=wikidata_plan
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    skipped = pipeline.run_acquisition(
        ["wikidata"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )
    assert skipped[0].status == "research_required"

    results = pipeline.run_acquisition(
        ["wikidata"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        adapter_handlers=adapter_handlers,
    )

    assert results[0].status == "downloaded"
    downloaded = captured.get("downloaded")
    assert isinstance(downloaded, tuple)
    assert downloaded[0] == "wikidata_entities_latest_all"
    assert downloaded[1] == "wikidata-all.json.gz"
    assert captured.get("projected") == (
        "wikidata-all.json.gz",
        "wikidata-companies.jsonl",
    )
    download_resource.assert_called_once()
    extract_wikidata_company_projection_two_pass.assert_called_once()
    assert get_system_plan_mock.call_count == 2


def test_run_acquisition_wikidata_reuses_existing_source_file(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}
    base_plan = get_system_plan("gb")
    wikidata_plan = replace(
        base_plan,
        code="wikidata",
        status="research_required",
        all_systems_target=False,
        notes="Wikidata acquisition run",
        resources=(
            replace(
                base_plan.resources[0],
                name="wikidata_entities_latest_all",
                url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz",
                file_format="jsonl",
                file_name="wikidata-all.json.gz",
                file_name_template=None,
                snapshot_date_mode="run_date",
                adapter="direct_download",
            ),
        ),
        research=ResearchPlan(allow_research_runtime=True, allowed_stages=("acquire",)),
    )

    existing_source = (
        layer_fixture_dir("wikidata", layer="acquire")
        / "2026-06-15"
        / "wikidata-all.json.gz"
    )
    existing_source.parent.mkdir(parents=True, exist_ok=True)
    existing_source.write_bytes(b"already-downloaded-90gb-placeholder")

    def fail_download(*args, **kwargs):
        raise AssertionError(
            "download_resource should not be called when source already exists"
        )

    def fake_extract_wikidata_company_projection(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        scan_mode="full",
        cleanup_intermediate=True,
    ):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            '{"id":"Q1","label_en":"Alpha"}\n', encoding="utf-8"
        )
        captured["projected"] = (source_path.name, destination_path.name)
        return 1

    download_resource = mocker.Mock(side_effect=fail_download)
    extract_wikidata_company_projection_two_pass = mocker.patch.object(
        pipeline,
        "extract_wikidata_company_projection_two_pass",
        side_effect=fake_extract_wikidata_company_projection,
    )
    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=wikidata_plan
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    results = pipeline.run_acquisition(
        ["wikidata"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        adapter_handlers=adapter_handlers,
    )

    assert results[0].status == "downloaded"
    assert captured.get("projected") == (
        "wikidata-all.json.gz",
        "wikidata-companies.jsonl",
    )
    download_resource.assert_not_called()
    extract_wikidata_company_projection_two_pass.assert_called_once()
    get_system_plan_mock.assert_called_once_with("wikidata")


def test_run_acquisition_wikidata_freshness_skip_reuses_projection_for_acquisition_snapshot(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    base_plan = get_system_plan("gb")
    wikidata_plan = replace(
        base_plan,
        code="wikidata",
        status="research_required",
        all_systems_target=False,
        notes="Wikidata acquisition run",
        resources=(
            replace(
                base_plan.resources[0],
                name="wikidata_entities_latest_all",
                url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz",
                file_format="jsonl",
                file_name="wikidata-all.json.gz",
                file_name_template=None,
                snapshot_date_mode="run_date",
                refresh_if_older_than_days=90,
                adapter="direct_download",
            ),
        ),
        research=ResearchPlan(allow_research_runtime=True, allowed_stages=("acquire",)),
    )

    existing_source = (
        layer_fixture_dir("wikidata", layer="acquire")
        / "2026-06-14"
        / "wikidata-all.json.gz"
    )
    existing_source.parent.mkdir(parents=True, exist_ok=True)
    existing_source.write_bytes(b"already-downloaded-90gb-placeholder")

    def fail_download(*args, **kwargs):
        raise AssertionError(
            "download_resource should not be called when freshness gate fails"
        )

    existing_projection = (
        layer_fixture_dir("wikidata", layer="prepare")
        / "2026-06-14"
        / "wikidata-companies.jsonl"
    )
    existing_projection.parent.mkdir(parents=True, exist_ok=True)
    existing_projection.write_text('{"id":"Q1","label_en":"Alpha"}\n', encoding="utf-8")

    def fail_projection(*args, **kwargs):
        raise AssertionError(
            "projection should not be rebuilt when acquisition-snapshot projection already exists"
        )

    download_resource = mocker.Mock(side_effect=fail_download)
    extract_wikidata_company_projection_two_pass = mocker.patch.object(
        pipeline,
        "extract_wikidata_company_projection_two_pass",
        side_effect=fail_projection,
    )
    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=wikidata_plan
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    results = pipeline.run_acquisition(
        ["wikidata"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        adapter_handlers=adapter_handlers,
    )

    assert results[0].status == "freshness_skipped"
    assert existing_projection.exists()
    assert not (
        layer_fixture_dir("wikidata", layer="prepare")
        / "2026-06-15"
        / "wikidata-companies.jsonl"
    ).exists()
    download_resource.assert_not_called()
    extract_wikidata_company_projection_two_pass.assert_not_called()
    get_system_plan_mock.assert_called_once_with("wikidata")


def test_run_acquisition_wikidata_freshness_skip_reuses_existing_chunked_prepare_output(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    base_plan = get_system_plan("gb")
    wikidata_plan = replace(
        base_plan,
        code="wikidata",
        status="research_required",
        all_systems_target=False,
        notes="Wikidata acquisition run",
        resources=(
            replace(
                base_plan.resources[0],
                name="wikidata_entities_latest_all",
                url="https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz",
                file_format="jsonl",
                file_name="wikidata-all.json.gz",
                file_name_template=None,
                snapshot_date_mode="run_date",
                refresh_if_older_than_days=90,
                adapter="direct_download",
            ),
        ),
        research=ResearchPlan(allow_research_runtime=True, allowed_stages=("acquire",)),
    )

    existing_source = (
        layer_fixture_dir("wikidata", layer="acquire")
        / "2026-06-14"
        / "wikidata-all.json.gz"
    )
    existing_source.parent.mkdir(parents=True, exist_ok=True)
    existing_source.write_bytes(b"already-downloaded-90gb-placeholder")

    projected_chunk_file = (
        layer_fixture_dir("wikidata", layer="prepare")
        / "2026-06-14"
        / "wikidata-companies.chunks"
        / "000000-000999"
        / "wikidata-companies-part-000001.jsonl"
    )
    projected_chunk_file.parent.mkdir(parents=True, exist_ok=True)
    projected_chunk_file.write_text(
        '{"id":"Q1","label_en":"Alpha"}\n', encoding="utf-8"
    )

    download_resource = mocker.Mock(
        side_effect=AssertionError("download should not run")
    )
    extract_wikidata_company_projection_two_pass = mocker.patch.object(
        pipeline,
        "extract_wikidata_company_projection_two_pass",
        side_effect=AssertionError(
            "projection should not rerun when chunked output exists"
        ),
    )
    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=wikidata_plan
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    results = pipeline.run_acquisition(
        ["wikidata"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        adapter_handlers=adapter_handlers,
    )

    assert results[0].status == "freshness_skipped"
    download_resource.assert_not_called()
    extract_wikidata_company_projection_two_pass.assert_not_called()
    get_system_plan_mock.assert_called_once_with("wikidata")


def test_run_acquisition_allow_research_respects_manifest_execution_policy(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    base_plan = get_system_plan("gb")
    denied_plan = replace(
        base_plan,
        code="dbpedia",
        status="research_required",
        all_systems_target=False,
        notes="Research run",
        research=ResearchPlan(allow_research_runtime=False, allowed_stages=()),
    )

    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=denied_plan
    )

    skipped = pipeline.run_acquisition(
        ["dbpedia"],
        roots=workspace_roots,
        run_date="2026-06-15",
        allow_research=True,
        skip_unsupported=True,
    )
    assert skipped[0].status == "research_required"
    assert "execution policy denies stage 'acquire'" in skipped[0].message

    with pytest.raises(RuntimeError, match="execution policy denies stage 'acquire'"):
        pipeline.run_acquisition(
            ["dbpedia"],
            roots=workspace_roots,
            run_date="2026-06-15",
            allow_research=True,
            skip_unsupported=False,
        )
    assert get_system_plan_mock.call_count == 2


def test_run_acquisition_unsupported_adapter_raises(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    supported_plan = get_system_plan("gb")
    broken_plan = replace(
        supported_plan,
        resources=(replace(supported_plan.resources[0], adapter="bogus"),),
    )

    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=broken_plan
    )

    with pytest.raises(RuntimeError, match="Unsupported adapter 'bogus'"):
        pipeline.run_acquisition(["gb"], roots=workspace_roots, run_date="2026-06-15")
    get_system_plan_mock.assert_called_once_with("gb")


def test_run_acquisition_records_extracted_files_for_zip_resources(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    supported_plan = get_system_plan("gb")
    extracting_plan = replace(
        supported_plan,
        resources=(replace(supported_plan.resources[0], extract=True),),
    )

    def fake_download_resource(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT, chunk_size=1024 * 1024
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"zip-bytes")
        return "ziphash"

    def fake_extract_zip_archive(source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        extracted = destination / "inside.csv"
        extracted.write_text("id,name\n1,Example\n", encoding="utf-8")
        return [extracted]

    def fake_sha256_file(path):
        return f"hash:{path.name}"

    download_resource = mocker.Mock(side_effect=fake_download_resource)
    extract_zip_archive = mocker.patch.object(
        pipeline, "extract_zip_archive", side_effect=fake_extract_zip_archive
    )
    sha256_file = mocker.patch.object(
        pipeline, "sha256_file", side_effect=fake_sha256_file
    )
    get_system_plan_mock = mocker.patch.object(
        pipeline, "get_system_plan", return_value=extracting_plan
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    results = pipeline.run_acquisition(
        ["gb"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )

    assert results[0].status == "downloaded"
    download_resource.assert_called_once()
    extract_zip_archive.assert_called_once()
    assert sha256_file.call_count == 1
    get_system_plan_mock.assert_called_once_with("gb")


def test_run_acquisition_wires_france_bulk_download(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}

    def fake_download_resource(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT, chunk_size=1024 * 1024
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"france-bulk-data")
        captured["resource_name"] = resource.name
        captured["destination"] = destination
        captured["user_agent"] = user_agent
        return "deadbeef"

    download_resource = mocker.Mock(side_effect=fake_download_resource)
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    results = pipeline.run_acquisition(
        ["fr"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )

    assert len(results) == 1
    assert results[0].country == "fr"
    assert results[0].status == "downloaded"
    assert results[0].artifact_count == 1
    assert captured["resource_name"] == "stock_unite_legale_parquet"

    expected_source = (
        layer_fixture_dir("fr", layer="acquire")
        / "2026-06-01"
        / "stockunitelegale.parquet"
    )
    assert captured["destination"] == expected_source
    assert expected_source.exists()
    download_resource.assert_called_once()
    assert not (
        layer_fixture_dir("manifest", layer="acquisition_manifest.parquet")
    ).exists()


def test_run_acquisition_wires_united_kingdom_bulk_download(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}

    def fake_download_resource(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT, chunk_size=1024 * 1024
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"uk-bulk-data")
        captured["resource_name"] = resource.name
        captured["destination"] = destination
        captured["user_agent"] = user_agent
        return "cafebabe"

    download_resource = mocker.Mock(side_effect=fake_download_resource)
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    results = pipeline.run_acquisition(
        ["gb"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )

    assert len(results) == 1
    assert results[0].country == "gb"
    assert results[0].status == "downloaded"
    assert results[0].artifact_count == 1
    assert captured["resource_name"] == "basic_company_data_as_one_file"
    assert captured["user_agent"] == DEFAULT_USER_AGENT

    expected_source = (
        layer_fixture_dir("gb", layer="acquire")
        / "2026-06-01"
        / "BasicCompanyDataAsOneFile-2026-06-01.zip"
    )
    assert captured["destination"] == expected_source
    assert expected_source.exists()
    download_resource.assert_called_once()
    assert not (
        layer_fixture_dir("manifest", layer="acquisition_manifest.parquet")
    ).exists()


def test_run_acquisition_skips_when_freshness_threshold_not_met(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    def fake_download_resource(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT, chunk_size=1024 * 1024
    ):
        raise AssertionError(
            "download_resource should not be called when freshness gate fails"
        )

    download_resource = mocker.Mock(side_effect=fake_download_resource)
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "direct_download": download_resource,
    }

    existing_source = (
        layer_fixture_dir("ie", layer="acquire") / "2026-06-01" / "companies.csv.zip"
    )
    existing_source.parent.mkdir(parents=True, exist_ok=True)
    existing_source.write_bytes(b"abc")

    results = pipeline.run_acquisition(
        ["ie"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )

    assert len(results) == 1
    assert results[0].country == "ie"
    assert results[0].status == "freshness_skipped"
    assert "freshness policy" in results[0].message
    assert not (layer_fixture_dir("ie", layer="acquire") / "2026-06-15").exists()
    assert not (layer_fixture_dir("ie", layer="prepare") / "2026-06-15").exists()

    download_resource.assert_not_called()
    assert not (
        layer_fixture_dir("manifest", layer="acquisition_manifest.parquet")
    ).exists()


def test_source_resource_resolves_template_from_run_date():
    resource = get_system_plan("gb").resources[0]
    assert (
        resource.resolve_file_name("2026-06-15")
        == "BasicCompanyDataAsOneFile-2026-06-01.zip"
    )
    assert (
        resource.resolve_download_url("2026-06-15")
        == "https://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2026-06-01.zip"
    )


def test_run_acquisition_wires_finland_api_snapshot_adapter(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}

    def fake_download_api_json_snapshot(
        resource, destination, *, user_agent=DEFAULT_USER_AGENT
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            '{"businessId":"1234567-8","name":"Example Oy"}\n', encoding="utf-8"
        )
        captured["resource_name"] = resource.name
        captured["destination"] = destination
        captured["user_agent"] = user_agent
        return "feedface"

    download_api_json_snapshot = mocker.Mock(
        side_effect=fake_download_api_json_snapshot
    )
    adapter_handlers = {
        **pipeline.ADAPTER_HANDLERS,
        "api_json_snapshot": download_api_json_snapshot,
    }

    results = pipeline.run_acquisition(
        ["fi"],
        roots=workspace_roots,
        run_date="2026-06-15",
        adapter_handlers=adapter_handlers,
    )

    assert len(results) == 1
    assert results[0].country == "fi"
    assert results[0].status == "downloaded"
    assert results[0].artifact_count == 1
    assert captured["resource_name"] == "prh_open_data_companies"
    assert captured["user_agent"] == DEFAULT_USER_AGENT

    expected_source = (
        layer_fixture_dir("fi", layer="acquire")
        / "2026-06-15"
        / "fi-prh-companies-2026-06-15.jsonl"
    )
    assert captured["destination"] == expected_source
    assert expected_source.exists()
    download_api_json_snapshot.assert_called_once()
    assert not (
        layer_fixture_dir("manifest", layer="acquisition_manifest.parquet")
    ).exists()


def test_resolve_adapter_handler_supports_dbpedia_databus_latest():
    handler = pipeline._resolve_adapter_handler("dbpedia_databus_latest")
    assert handler is pipeline.ADAPTER_HANDLERS["dbpedia_databus_latest"]
