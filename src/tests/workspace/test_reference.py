"""Direct tests for `workspace.reference`.

What is pinned is the contract a script's entry point relies on: a reference
round-trips between its values, its URI and its location; a URI and a path may
order and group segments differently yet never let two references share or nest
a location; and a selection either names one existing thing or fails naming
what does exist.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

from acquisition.plan_registry import SYSTEM_REGISTRY
from workspace import reference as reference_module
from workspace.derived_uri import NAME_SCHEME, PERTURBED_SCHEME
from workspace.kind_layout import Kind, data_directory
from workspace.reference import (
    BLOCKING_PLAIN,
    RESERVED_WORDS,
    SLUG,
    VERSION,
    AmbiguousReferenceError,
    Fixed,
    InvalidReferenceError,
    Layout,
    LayoutError,
    Part,
    Reference,
    ReferenceNotFoundError,
    Segment,
    Side,
    check_side_roots,
    locate,
    one_of,
    parse_reference,
    reference,
    reference_at,
    require_reference,
    resolve_latest,
    run_source,
    select_references,
    selection_directory,
    side_root,
)
from workspace.roots import WorkspaceRoots


def _profile(name: str, version: str, *, draft: bool = False) -> Reference:
    stage = {"stage": "draft"} if draft else {}
    return reference(
        Kind.PERTURBATION, Side.PROFILE, name=name, version=version, **stage
    )


def _run(source: str, representation: str, key: str, **tail: str) -> Reference:
    return reference(
        Kind.BLOCKING,
        Side.DATA,
        source=source,
        target="gb",
        representation=representation,
        key=key,
        **tail,
    )


_PERTURBED = {
    "derivation": "perturbed",
    "profile": "en-lite",
    "version": "v1",
    "seed": "42",
}


def _make(roots: WorkspaceRoots, *refs: Reference) -> None:
    for ref in refs:
        locate(roots, ref).mkdir(parents=True)


@pytest.mark.parametrize(
    ("ref", "uri", "relative"),
    [
        (
            _profile("en-lite", "v2"),
            "perturbation://en-lite/v2",
            ("en-lite", "v2"),
        ),
        (
            _profile("en-lite", "v2", draft=True),
            "perturbation://en-lite/v2/draft",
            ("en-lite", "draft", "v2"),
        ),
        (
            reference(
                Kind.PERTURBATION,
                Side.DATA,
                source="ie",
                profile="en-lite",
                version="v1",
                seed="42",
            ),
            "perturbed://ie/en-lite/v1/42",
            ("ie", "en-lite", "v1", "42"),
        ),
        (
            reference(
                Kind.PERTURBATION,
                Side.DATA,
                source="gb",
                dataset="names",
                profile="en-lite",
                version="v1",
                seed="42",
            ),
            "perturbed://gb/names/en-lite/v1/42",
            ("gb", "names", "en-lite", "v1", "42"),
        ),
        (
            _run("gleif", "tfidf", "0123456789ab"),
            "blocking://gleif/gb/tfidf/0123456789ab",
            ("gb", "data", "gleif", "tfidf", "0123456789ab"),
        ),
        (
            _run("gb", "tfidf", "0123456789ab", dataset="names"),
            "blocking://gb/gb/tfidf/names/0123456789ab",
            ("gb", "names", "gb", "tfidf", "0123456789ab"),
        ),
        (
            _run("ie", "tfidf", "0123456789ab", **_PERTURBED),
            "blocking://ie/gb/tfidf/perturbed/en-lite/v1/42/0123456789ab",
            ("gb", "perturbed", "ie_en-lite_v1_42", "tfidf", "0123456789ab"),
        ),
        (
            _run("gb", "tfidf", "0123456789ab", dataset="names", **_PERTURBED),
            "blocking://gb/gb/tfidf/names/perturbed/en-lite/v1/42/0123456789ab",
            ("gb", "perturbed-names", "gb_en-lite_v1_42", "tfidf", "0123456789ab"),
        ),
        (
            reference(
                Kind.BLOCKING,
                Side.DATA,
                source="benchmark",
                target="amazon-google",
                representation="tfidf",
                key="0123456789ab",
            ),
            "blocking://benchmark/amazon-google/tfidf/0123456789ab",
            ("amazon-google", "benchmark", "tfidf", "0123456789ab"),
        ),
    ],
)
def test_a_reference_round_trips_between_values_uri_and_location(
    workspace_roots: WorkspaceRoots, ref: Reference, uri: str, relative: tuple[str, ...]
):
    location = locate(workspace_roots, ref)

    assert ref.uri == uri
    assert parse_reference(uri) == ref
    assert location == side_root(workspace_roots, ref.kind, ref.side).joinpath(
        *relative
    )
    assert reference_at(workspace_roots, location) == ref


def test_a_draft_sits_beside_its_promoted_version_not_inside_it(
    workspace_roots: WorkspaceRoots,
):
    promoted = locate(workspace_roots, _profile("en-lite", "v2"))
    draft = locate(workspace_roots, _profile("en-lite", "v2", draft=True))

    assert promoted not in draft.parents
    assert draft not in promoted.parents
    assert _profile("en-lite", "v2").immutable
    assert not _profile("en-lite", "v2", draft=True).immutable


def test_references_that_differ_near_a_boundary_never_share_or_nest_a_location(
    workspace_roots: WorkspaceRoots,
):
    """Every real registration, fed values chosen to collide if a segment's
    value could be read as its neighbour's or an optional stage as a value."""
    refs = [
        _profile("draft", "v2"),
        _profile("v2", "v3"),
        _profile("v2", "v3", draft=True),
        _profile("en-lite", "v2"),
        _profile("en-lite", "v2", draft=True),
        _run("ie", "tfidf", "ab", **_PERTURBED),
        _run("ie-en", "tfidf", "ab", **{**_PERTURBED, "profile": "lite"}),
        _run("ie", "tfidf", "ab", dataset="names", **_PERTURBED),
        _run("ie", "tfidf", "ab", dataset="names"),
        _run("ie-en-lite-v1-42", "tfidf", "ab"),
        reference(
            Kind.PERTURBATION,
            Side.DATA,
            source="ie",
            profile="en",
            version="v1",
            seed="42",
        ),
        reference(
            Kind.PERTURBATION,
            Side.DATA,
            source="ie-en",
            profile="v1",
            version="v4",
            seed="2",
        ),
    ]
    locations = {ref: locate(workspace_roots, ref) for ref in refs}

    assert len(set(locations.values())) == len(refs)
    for a, b in combinations(locations.values(), 2):
        assert a not in b.parents and b not in a.parents
    for ref, location in locations.items():
        assert reference_at(workspace_roots, location) == ref


def test_registered_sides_share_no_root(workspace_roots: WorkspaceRoots):
    check_side_roots(workspace_roots)


def test_two_sides_sharing_a_root_are_refused(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    monkeypatch.setitem(reference_module._SIDE_ROOTS, Side.PROFILE, data_directory)

    with pytest.raises(LayoutError, match="share a location"):
        check_side_roots(workspace_roots)


def test_no_kind_is_a_row_scheme():
    """A reference's scheme is a concern's name; a row's is a system code or a
    record kind, and the two must never be read as each other."""
    row_schemes = set(SYSTEM_REGISTRY) | {
        NAME_SCHEME,
        PERTURBED_SCHEME,
    }

    assert not {kind.value for kind in Kind} & row_schemes


def test_a_layout_leaving_a_segment_out_of_its_path_is_refused():
    with pytest.raises(LayoutError, match="out of its path"):
        Layout(
            scheme="probe",
            kind=Kind.BLOCKING,
            side=Side.PROFILE,
            segments=(Segment("name", SLUG), Segment("version", VERSION)),
            path=(Part(("name",)),),
            immutable=True,
        )


def test_a_join_whose_separator_a_segment_admits_is_refused():
    with pytest.raises(LayoutError, match="cannot be split back"):
        Layout(
            scheme="probe",
            kind=Kind.BLOCKING,
            side=Side.PROFILE,
            segments=(Segment("system", SLUG), Segment("name", SLUG)),
            path=(Part(("system", "name"), separator="-"),),
            immutable=True,
        )


def test_a_layout_nesting_one_shape_inside_another_is_refused():
    """The stage last in the path would put a draft inside its promoted version."""
    with pytest.raises(LayoutError, match="coincide or nest"):
        Layout(
            scheme="probe",
            kind=Kind.BLOCKING,
            side=Side.PROFILE,
            segments=(
                Segment("name", SLUG),
                Segment("version", VERSION),
                Segment("stage", literal="draft"),
            ),
            path=(Part(("name",)), Part(("version",)), Part(("stage",))),
            immutable=True,
        )


def test_a_stage_an_open_segment_could_be_read_as_is_refused():
    """A stage first could be read as a profile named `draft`."""
    with pytest.raises(LayoutError, match="coincide or nest"):
        Layout(
            scheme="probe",
            kind=Kind.BLOCKING,
            side=Side.PROFILE,
            segments=(
                Segment("name", SLUG),
                Segment("version", VERSION),
                Segment("stage", literal="draft"),
            ),
            path=(Part(("stage",)), Part(("name",)), Part(("version",))),
            immutable=True,
        )


def test_a_join_and_fixed_names_order_the_path_independently_of_the_uri(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    layout = Layout(
        scheme="probe",
        kind=Kind.BLOCKING,
        side=Side.PROFILE,
        segments=(
            Segment("system", SLUG),
            Segment("name", SLUG),
            Segment("version", VERSION),
        ),
        path=(
            Part(("version",)),
            Fixed("by-name"),
            Part(("system", "name"), separator="_"),
        ),
        immutable=True,
    )
    monkeypatch.setitem(
        reference_module._LAYOUTS, (Kind.BLOCKING, Side.PROFILE), [layout]
    )
    near = reference(
        Kind.BLOCKING, Side.PROFILE, system="ie-en", name="lite", version="v1"
    )
    far = reference(
        Kind.BLOCKING, Side.PROFILE, system="ie", name="en-lite", version="v1"
    )

    location = locate(workspace_roots, near)

    assert near.uri == "probe://ie-en/lite/v1"
    assert location.relative_to(
        side_root(workspace_roots, Kind.BLOCKING, Side.PROFILE)
    ) == Path("v1", "by-name", "ie-en_lite")
    assert locate(workspace_roots, far) != location
    assert reference_at(workspace_roots, location) == near
    _make(workspace_roots, near, far)
    assert select_references(
        workspace_roots, reference(Kind.BLOCKING, Side.PROFILE, system="ie")
    ) == [far]


@pytest.mark.parametrize(
    "uri",
    [
        "gleif://data/gb",
        "blocking://gleif/gb/tfidf/not-a-key",
        "blocking://GB",
        "perturbation://en-lite/v2/final",
        "perturbation://en-lite/draft",
        "no-separator",
    ],
)
def test_a_uri_naming_no_registered_reference_is_refused(uri: str):
    with pytest.raises(InvalidReferenceError):
        parse_reference(uri)


def test_a_selection_may_leave_out_any_segment_and_its_uri_says_which():
    every_v1 = reference(Kind.PERTURBATION, Side.PROFILE, version="v1")
    against_gb = reference(Kind.BLOCKING, Side.DATA, target="gb")

    assert every_v1.uri == "perturbation://*/v1"
    assert against_gb.uri == "blocking://*/gb"
    assert parse_reference(every_v1.uri) == every_v1
    assert parse_reference(against_gb.uri) == against_gb
    assert not against_gb.complete


def test_every_run_against_one_target_is_a_selection(workspace_roots: WorkspaceRoots):
    plain = _run("gleif", "tfidf", "aaaaaaaaaaaa")
    names = _run("gb", "tfidf", "bbbbbbbbbbbb", dataset="names")
    perturbed = _run("ie", "tfidf", "cccccccccccc", **_PERTURBED)
    _make(workspace_roots, plain, names, perturbed)

    against_gb = reference(Kind.BLOCKING, Side.DATA, target="gb")

    assert select_references(workspace_roots, against_gb) == sorted(
        [plain, names, perturbed], key=lambda ref: ref.uri
    )
    assert select_references(workspace_roots, against_gb, layout=BLOCKING_PLAIN) == [
        plain
    ]
    assert select_references(workspace_roots, plain) == [plain]


@pytest.mark.parametrize(
    ("run", "source"),
    [
        (_run("gleif", "tfidf", "ab"), "data://gleif/matched"),
        (_run("gb", "tfidf", "ab", dataset="names"), "data://gb/names"),
        (_run("ie", "tfidf", "ab", **_PERTURBED), "perturbed://ie/en-lite/v1/42"),
        (
            _run("gb", "tfidf", "ab", dataset="names", **_PERTURBED),
            "perturbed://gb/names/en-lite/v1/42",
        ),
    ],
)
def test_a_run_decomposes_to_its_source_its_target_and_its_representation(
    run: Reference, source: str
):
    assert run_source(run).uri == source
    assert run.fields["target"] == "gb"
    assert run.fields["representation"] == "tfidf"


@pytest.mark.parametrize("word", RESERVED_WORDS)
def test_a_reserved_word_names_no_system_profile_or_representation(word: str):
    with pytest.raises(InvalidReferenceError):
        _run("gleif", word, "ab")
    with pytest.raises(InvalidReferenceError):
        reference(Kind.PERTURBATION, Side.PROFILE, name=word, version="v1")
    with pytest.raises(InvalidReferenceError):
        reference(Kind.BLOCKING, Side.DATA, source="gleif", target=word)


def test_a_location_outside_every_layout_names_no_reference(
    workspace_roots: WorkspaceRoots,
):
    base = side_root(workspace_roots, Kind.BLOCKING, Side.DATA)

    with pytest.raises(InvalidReferenceError):
        reference_at(workspace_roots, base / "gb" / "comparison")


def test_a_partial_reference_selects_every_existing_reference_beneath_it(
    workspace_roots: WorkspaceRoots,
):
    tfidf = _run("gleif", "tfidf", "aaaaaaaaaaaa")
    sbert = _run("gleif", "sbert", "bbbbbbbbbbbb")
    other = _run("wikidata", "tfidf", "cccccccccccc")
    _make(workspace_roots, tfidf, sbert, other)
    (locate(workspace_roots, tfidf).parent.parent / "comparison").mkdir()

    selected = select_references(
        workspace_roots,
        reference(Kind.BLOCKING, Side.DATA, source="gleif", target="gb"),
        layout=BLOCKING_PLAIN,
    )

    assert selected == [sbert, tfidf]


def test_a_selection_directory_is_the_part_of_the_path_the_selection_fixes(
    workspace_roots: WorkspaceRoots,
):
    base = side_root(workspace_roots, Kind.BLOCKING, Side.DATA)
    pairing = reference(Kind.BLOCKING, Side.DATA, source="gleif", target="gb")

    assert (
        selection_directory(workspace_roots, pairing, layout=BLOCKING_PLAIN)
        == base / "gb" / "data" / "gleif"
    )
    assert (
        selection_directory(workspace_roots, reference(Kind.BLOCKING, Side.DATA))
        == base
    )
    assert selection_directory(workspace_roots, _run("gleif", "tfidf", "ab")) == locate(
        workspace_roots, _run("gleif", "tfidf", "ab")
    )


def test_a_selection_never_finds_a_draft_it_did_not_name(
    workspace_roots: WorkspaceRoots,
):
    _make(
        workspace_roots,
        _profile("en-lite", "v1"),
        _profile("en-lite", "v2", draft=True),
    )

    assert select_references(
        workspace_roots, reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite")
    ) == [_profile("en-lite", "v1")]


def test_requiring_a_missing_reference_names_what_exists(
    workspace_roots: WorkspaceRoots,
):
    _make(workspace_roots, _profile("en-lite", "v1"))

    with pytest.raises(ReferenceNotFoundError, match="perturbation://en-lite/v1"):
        require_reference(workspace_roots, _profile("en-lite", "v2"))


def test_requiring_a_reference_nothing_exists_beside_says_so(
    workspace_roots: WorkspaceRoots,
):
    with pytest.raises(ReferenceNotFoundError, match="nothing exists"):
        require_reference(workspace_roots, _profile("en-lite", "v2"))


def test_requiring_one_of_several_matches_names_them(workspace_roots: WorkspaceRoots):
    _make(workspace_roots, _profile("en-lite", "v1"), _profile("en-lite", "v2"))

    with pytest.raises(AmbiguousReferenceError, match="en-lite/v1.*en-lite/v2"):
        require_reference(
            workspace_roots, reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite")
        )


def test_requiring_an_existing_reference_returns_it(workspace_roots: WorkspaceRoots):
    _make(workspace_roots, _profile("en-lite", "v1"))

    assert require_reference(workspace_roots, _profile("en-lite", "v1")) == _profile(
        "en-lite", "v1"
    )


@pytest.mark.parametrize(
    ("uri", "parts"),
    [
        (
            "data://gb/canonical/2026-06-01",
            ("gb", "canonical", "2026-06-01", "primary"),
        ),
        ("data://gb/names/2026-06-01", ("gb", "canonical", "2026-06-01", "names")),
        ("data://gleif/matched", ("gleif", "matched", "current", "primary")),
    ],
)
def test_each_dataset_locates_where_it_sits_today_and_reads_back(
    workspace_roots: WorkspaceRoots, uri: str, parts: tuple[str, ...]
):
    dataset = parse_reference(uri)

    assert dataset.uri == uri
    assert locate(workspace_roots, dataset) == workspace_roots.data.joinpath(*parts)
    assert reference_at(workspace_roots, locate(workspace_roots, dataset)) == dataset


def test_a_request_naming_no_version_resolves_to_the_latest_and_has_no_location(
    workspace_roots: WorkspaceRoots,
):
    for date in ("2026-01-05", "2026-06-01"):
        _make(workspace_roots, parse_reference(f"data://gb/canonical/{date}"))
    request = parse_reference("data://gb/canonical")

    assert not request.complete
    with pytest.raises(InvalidReferenceError):
        locate(workspace_roots, request)
    assert (
        resolve_latest(workspace_roots, request).uri == "data://gb/canonical/2026-06-01"
    )
    with pytest.raises(ReferenceNotFoundError):
        resolve_latest(workspace_roots, parse_reference("data://ie/names"))
    with pytest.raises(InvalidReferenceError):
        resolve_latest(workspace_roots, parse_reference("data://gb"))


def test_two_layouts_of_one_side_may_not_share_or_nest_a_location():
    nested = Layout(
        scheme="data",
        kind=Kind.DATA,
        side=Side.DATA,
        segments=(Segment("system", SLUG), Segment("dataset", one_of("matched"))),
        path=(Part(("system",)), Part(("dataset",)), Fixed("current")),
        immutable=False,
    )

    with pytest.raises(LayoutError, match="coincide or nest"):
        reference_module._register(nested)


def test_a_default_a_segment_cannot_hold_is_refused():
    with pytest.raises(LayoutError, match="default it cannot hold"):
        Layout(
            scheme="probe",
            kind=Kind.BLOCKING,
            side=Side.PROFILE,
            segments=(
                Segment("name", SLUG),
                Segment("version", VERSION, default="latest"),
            ),
            path=(Part(("name",)), Part(("version",))),
            immutable=True,
        )


def test_a_tokenizer_candidate_locates_by_scope_trainer_and_key(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """The layout says where a promoted candidate's directory sits and reads
    it back; the key is `company_tokenize`'s, so only its shape is checked."""
    candidate = reference(
        Kind.TOKENIZER,
        Side.DATA,
        scope="ie",
        tokenizer_id="sentencepiece_unigram",
        key="0a1b2c3d_seed_7_v24000_mf2",
    )

    assert candidate.complete and candidate.immutable
    assert candidate.uri == (
        "tokenizer://ie/sentencepiece_unigram/0a1b2c3d_seed_7_v24000_mf2"
    )
    assert parse_reference(candidate.uri) == candidate
    assert locate(workspace_roots, candidate) == (
        tmp_path
        / "artifacts"
        / "tokenizers"
        / "data"
        / "ie"
        / "sentencepiece_unigram"
        / "0a1b2c3d_seed_7_v24000_mf2"
    )
    assert reference(
        Kind.TOKENIZER, Side.DATA, scope="ie", tokenizer_id="wordpiece"
    ).uri == ("tokenizer://ie/wordpiece")
