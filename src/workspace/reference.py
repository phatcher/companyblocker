"""Typed references: what a script names instead of a path, and where each lives.

A reference names a thing by its type and parameters, never by its location.
Its compact form is a URI whose scheme says what the thing is and whose segments
are the parameters that scheme's layout registers, in its registered order:

    perturbation://en-lite/v2/draft
    perturbed://ie/en-lite/v1/42
    blocking://gb/data/gleif/tfidf/0123456789ab

A layout belongs to a `kind_layout.Kind` and a side, `profile` for what is
authored or `data` for what is produced, which together choose the tree its
locations sit in. Neither is a word of the URI: a kind with both sides registers
a scheme for each, named as the two things they are.

**Every layout is registered here and nowhere else.** This module owns each
kind's URI and the location built from it; an area builds a reference and asks
where it is, and never registers a layout or composes a location of its own.
A new kind or side is registered here before any area can build one, which is
what keeps a location stable once things have been written to it.

**The URI and the location are separate mappings.** Each side registers its
URI segments and, independently, a path template beneath the side's root
(`kind_layout.profile_directory` or `data_directory`): fixed names, single
segments and joins of several segments, in any order. What makes that freedom
safe is checked when a layout is built, not left to a test of each caller:

- every segment reaches the path, so two references differing anywhere name two
  locations;
- a join's separator holds a character no joined segment admits, so the join
  splits back into exactly its segments;
- a reference is recovered from its path alone, with no file read;
- no complete location lies inside or on another, across every shape a side's
  optional segments give it.

That last rule is why a draft profile sits at `en-lite/draft/v2/`, beside the
promoted `en-lite/v2/` rather than inside it, while its URI keeps the stage
last. No two sides may share a root or nest one root inside another;
`check_side_roots` states that for a set of resolved roots.

**A partial reference is a selection.** Giving some of a layout's required
segments names every complete reference holding them, and `select_references`
finds those that exist. Its URI writes `*` for a segment left out before one
that is given, `blocking://*/ie` being every run against one target. An optional segment is only ever
present by being named, so a selection never finds a draft it did not ask for.
`require_reference` is the existence check an entry point makes before any
work: it fails naming what does exist when a selection matches nothing, and
naming the matches when it matches more than one.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations, product
from pathlib import Path

from .kind_layout import Kind, data_directory, profile_directory
from .roots import WorkspaceRoots

URI_SEPARATOR = "://"

SELECT_ANY = "*"
"""What a selection's URI writes for a segment it leaves out."""

MAX_REPORTED_REFERENCES = 20
"""How many references an error lists before summarizing the rest."""


class Side(StrEnum):
    """Which of a kind's two trees a reference lives in."""

    PROFILE = "profile"
    DATA = "data"


_SIDE_ROOTS: dict[Side, Callable[[WorkspaceRoots, Kind], Path]] = {
    Side.PROFILE: profile_directory,
    Side.DATA: data_directory,
}


class InvalidReferenceError(ValueError):
    """A URI, a segment value or a location that names no registered reference."""


class LayoutError(ValueError):
    """A layout whose path template is not unique or clashes with itself."""


class ReferenceNotFoundError(LookupError):
    """A selection that matches nothing on disk."""


class AmbiguousReferenceError(LookupError):
    """A selection that matches several references where one is needed."""


@dataclass(frozen=True)
class SegmentType:
    """What values one segment admits: a full-match pattern over a character set.

    The character set is stated rather than read out of the pattern, because
    whether a join's separator can occur inside a value has to be decided when
    a layout is built, and a regular expression cannot be asked that.
    """

    pattern: str
    charset: frozenset[str]
    words: tuple[str, ...] | None = None
    """The closed list of values this type admits, where it has one."""

    def admits(self, value: str) -> bool:
        return (
            value not in ("", ".", "..")
            and set(value) <= self.charset
            and re.fullmatch(self.pattern, value) is not None
        )


_LOWER_ALNUM = frozenset(string.ascii_lowercase + string.digits)

SLUG = SegmentType(r"[a-z0-9]+(?:-[a-z0-9]+)*", _LOWER_ALNUM | {"-"})
"""Lowercase kebab case: a system code, a profile name."""

VERSION = SegmentType(r"v[1-9][0-9]*", _LOWER_ALNUM)
"""A promoted version, a plain `v<n>` sequence."""

INTEGER = SegmentType(r"-?[0-9]+", frozenset(string.digits) | {"-"})

HEX = SegmentType(r"[0-9a-f]+", frozenset("0123456789abcdef"))

IDENTIFIER = SegmentType(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", _LOWER_ALNUM | {"-", "_"})
"""A method or representation name, which may use underscores."""


RESERVED_WORDS: tuple[str, ...] = ("names", "perturbed", "benchmark")
"""Markers a URI is read by, so no system, profile or representation may be
called one."""


def _unreserved(segment_type: SegmentType) -> SegmentType:
    refusal = "|".join(re.escape(word) for word in RESERVED_WORDS)
    return SegmentType(
        f"(?!(?:{refusal})$)(?:{segment_type.pattern})", segment_type.charset
    )


NAME = _unreserved(SLUG)
"""A system code or a profile name: kebab case, and no reserved word."""

REPRESENTATION = _unreserved(IDENTIFIER)
"""A representation name, which may use underscores, and no reserved word."""


def one_of(*values: str) -> SegmentType:
    """A segment admitting exactly `values`."""
    return SegmentType(
        "|".join(re.escape(value) for value in values),
        frozenset("".join(values)),
        words=tuple(values),
    )


@dataclass(frozen=True)
class Segment:
    """One named parameter of a reference.

    A segment with a `literal` is optional: it is either absent or that exact
    word, and it follows every required segment in the URI. A present optional
    segment makes the reference mutable, the way a draft is. A segment with a
    `default` takes that value when it is not given and every segment before it
    is, so a request that leaves it out means the default rather than every
    value. A `latest` segment is a version whose values sort: a request that
    leaves it out means the greatest that exists, which `resolve_latest` finds.
    Any other segment a request leaves out has to be the only one that exists.
    `identity` says whether the segment counts towards a run's identity.
    """

    name: str
    domain: SegmentType | None = None
    literal: str | None = None
    default: str | None = None
    identity: bool = True
    latest: bool = False

    @property
    def optional(self) -> bool:
        return self.literal is not None

    def admits(self, value: str) -> bool:
        if self.literal is not None:
            return value == self.literal
        assert self.domain is not None  # nosec B101 - type narrowing on an invariant the branch above holds
        return self.domain.admits(value)

    @property
    def charset(self) -> frozenset[str]:
        if self.literal is not None:
            return frozenset(self.literal)
        assert self.domain is not None  # nosec B101 - type narrowing on an invariant the branch above holds
        return self.domain.charset


@dataclass(frozen=True)
class Fixed:
    """A path part that is always the same name."""

    name: str


@dataclass(frozen=True)
class Part:
    """A path part rendered from one segment, or from several joined by `separator`."""

    segments: tuple[str, ...]
    separator: str = ""


PathPart = Fixed | Part

_UNSAFE_PATH_CHARACTERS = frozenset('/\\:*?"<>|')


@dataclass(frozen=True)
class Layout:
    """One side of one kind: its scheme, its URI segments and its path template.

    The scheme names what the thing is and is how a URI finds its layout; the
    side chooses the tree its location sits in and appears in no URI.
    """

    scheme: str
    kind: Kind
    side: Side
    segments: tuple[Segment, ...]
    path: tuple[PathPart, ...]
    immutable: bool
    _by_name: Mapping[str, Segment] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_by_name", {segment.name: segment for segment in self.segments}
        )
        _check_layout(self)

    @property
    def required(self) -> tuple[Segment, ...]:
        return tuple(segment for segment in self.segments if not segment.optional)

    @property
    def optional(self) -> tuple[Segment, ...]:
        return tuple(segment for segment in self.segments if segment.optional)

    def segment(self, name: str) -> Segment:
        return self._by_name[name]

    def shape(self, present: frozenset[str]) -> tuple[PathPart, ...]:
        """The path parts a location has when exactly the optional segments
        named in `present` are given."""
        return tuple(
            part
            for part in self.path
            if not (
                isinstance(part, Part)
                and self.segment(part.segments[0]).optional
                and part.segments[0] not in present
            )
        )

    def shapes(self) -> Iterator[tuple[frozenset[str], tuple[PathPart, ...]]]:
        names = [segment.name for segment in self.optional]
        for size in range(len(names) + 1):
            for chosen in combinations(names, size):
                present = frozenset(chosen)
                yield present, self.shape(present)


@dataclass(frozen=True)
class Reference:
    """A kind, a side and a leading run of that side's segment values.

    Built through `reference` or `parse_reference`, which validate every value;
    complete when every required segment is given.
    """

    kind: Kind
    side: Side
    values: tuple[tuple[str, str], ...]

    @property
    def layout(self) -> Layout:
        """The one layout these values name. A side registering several is
        told apart by their fixed words, so a selection too short to hold one
        names no single layout."""
        layouts = _named_layouts(self.kind, self.side, self.fields)
        if len(layouts) != 1:
            raise InvalidReferenceError(
                f"{render_reference(self)} names {len(layouts)} layouts of "
                f"{self.kind.value}, not one."
            )
        return layouts[0]

    @property
    def fields(self) -> dict[str, str]:
        return dict(self.values)

    @property
    def complete(self) -> bool:
        return len(_fitting(self.kind, self.side, self.fields)) == 1

    @property
    def immutable(self) -> bool:
        """Whether this reference names one content forever: its side is
        immutable and no optional segment, such as a draft stage, is present."""
        given = self.fields
        return self.layout.immutable and not any(
            segment.name in given for segment in self.layout.optional
        )

    @property
    def uri(self) -> str:
        return render_reference(self)

    def __str__(self) -> str:
        return self.uri


def _check_layout(layout: Layout) -> None:
    """Refuse a layout whose locations are not unique or clash."""
    where = _label(layout)
    names = [segment.name for segment in layout.segments]
    if len(set(names)) != len(names):
        raise LayoutError(f"{where} names a segment twice: {names}.")
    seen_optional = False
    for segment in layout.segments:
        if (segment.domain is None) == (segment.literal is None):
            raise LayoutError(
                f"{where} segment {segment.name!r} needs exactly one of a domain or a literal."
            )
        if segment.default is not None and (
            segment.optional or not segment.admits(segment.default)
        ):
            raise LayoutError(
                f"{where} segment {segment.name!r} has a default it cannot hold."
            )
        if segment.latest and segment.optional:
            raise LayoutError(
                f"{where} segment {segment.name!r} is optional and `latest`."
            )
        if segment.optional:
            seen_optional = True
        elif seen_optional:
            raise LayoutError(
                f"{where} segment {segment.name!r} is required but follows an optional one."
            )
        if segment.charset & _UNSAFE_PATH_CHARACTERS:
            raise LayoutError(
                f"{where} segment {segment.name!r} admits a character a path cannot hold."
            )

    placed: list[str] = []
    for part in layout.path:
        if isinstance(part, Fixed):
            if not part.name or set(part.name) & _UNSAFE_PATH_CHARACTERS:
                raise LayoutError(f"{where} has an unusable fixed name {part.name!r}.")
            continue
        unknown = [name for name in part.segments if name not in layout._by_name]
        if unknown:
            raise LayoutError(f"{where} path names unregistered segments {unknown}.")
        placed.extend(part.segments)
        if len(part.segments) > 1:
            _check_join(layout, part, where)
    missing = [name for name in names if name not in placed]
    if missing:
        raise LayoutError(
            f"{where} leaves {missing} out of its path, so references differing "
            "only there would share a location."
        )
    doubled = sorted({name for name in placed if placed.count(name) > 1})
    if doubled:
        raise LayoutError(f"{where} places {doubled} in its path more than once.")

    shapes = list(layout.shapes())
    for (present_a, shape_a), (present_b, shape_b) in combinations(shapes, 2):
        shorter, longer = sorted((shape_a, shape_b), key=len)
        if all(
            _parts_may_coincide(layout, a, b)
            for a, b in zip(shorter, longer, strict=False)
        ):
            raise LayoutError(
                f"{where}: a location with optional segments {sorted(present_a)} and "
                f"one with {sorted(present_b)} can coincide or nest, so one could be "
                "read as the other."
            )


def _check_join(layout: Layout, part: Part, where: str) -> None:
    if not part.separator or set(part.separator) & _UNSAFE_PATH_CHARACTERS:
        raise LayoutError(f"{where} joins {part.segments} with an unusable separator.")
    for name in part.segments:
        segment = layout.segment(name)
        if segment.optional:
            raise LayoutError(f"{where} joins optional segment {name!r}.")
        if set(part.separator) <= segment.charset:
            raise LayoutError(
                f"{where} joins {part.segments} with {part.separator!r}, which "
                f"segment {name!r} admits, so the join cannot be split back."
            )


def _part_charset(layout: Layout, part: PathPart) -> frozenset[str]:
    if isinstance(part, Fixed):
        return frozenset(part.name)
    charset = frozenset(part.separator)
    for name in part.segments:
        charset |= layout.segment(name).charset
    return charset


def _parts_may_coincide(
    layout: Layout, a: PathPart, b: PathPart, *, other: Layout | None = None
) -> bool:
    """Whether two path parts could render the same directory name.

    `a` is a part of `layout` and `b` of `other`, the same layout unless given.
    Decided exactly where a fixed name, a literal or a closed list of words is
    involved, and conservatively (assumed possible) between two open segment
    types unless one's rendering needs a character the other cannot hold.
    """
    other = other or layout
    if isinstance(a, Fixed) and isinstance(b, Fixed):
        return a.name == b.name
    if isinstance(a, Fixed):
        assert isinstance(b, Part)  # nosec B101 - type narrowing on an invariant the branch above holds
        return _parse_part(other, b, a.name) is not None
    if isinstance(b, Fixed):
        return _parse_part(layout, a, b.name) is not None
    for one, one_layout, two, two_layout in (
        (a, layout, b, other),
        (b, other, a, layout),
    ):
        if len(one.segments) == 1:
            words = _segment_words(one_layout.segment(one.segments[0]))
            if words is not None:
                return any(
                    _parse_part(two_layout, two, word) is not None for word in words
                )
    if a.separator and not set(a.separator) <= _part_charset(other, b):
        return False
    return not (b.separator and not set(b.separator) <= _part_charset(layout, a))


def _segment_words(segment: Segment) -> tuple[str, ...] | None:
    """The exact words a segment renders, where it is a literal or admits a
    closed list; None where it is open."""
    if segment.literal is not None:
        return (segment.literal,)
    assert segment.domain is not None  # nosec B101 - type narrowing on an invariant the branch above holds
    return segment.domain.words


def _parse_part(layout: Layout, part: Part, name: str) -> dict[str, str] | None:
    """The segment values one directory name renders, or None if it renders none."""
    pieces = name.split(part.separator) if part.separator else [name]
    if len(pieces) != len(part.segments):
        return None
    values: dict[str, str] = {}
    for segment_name, piece in zip(part.segments, pieces, strict=True):
        if not layout.segment(segment_name).admits(piece):
            return None
        values[segment_name] = piece
    return values


def _render_part(part: PathPart, values: Mapping[str, str]) -> str:
    if isinstance(part, Fixed):
        return part.name
    return part.separator.join(values[name] for name in part.segments)


_LAYOUTS: dict[tuple[Kind, Side], list[Layout]] = {}
_SCHEMES: dict[str, list[Layout]] = {}


def _register(layout: Layout) -> Layout:
    """Add a checked layout to the registry.

    A side registers one scheme and may register several layouts under it, each
    a flat list of segments told from the others by a fixed word, provided no
    location of one can coincide with or nest inside a location of another.
    """
    key = (layout.kind, layout.side)
    for other in _SCHEMES.get(layout.scheme, []):
        if (other.kind, other.side) != key:
            raise LayoutError(
                f"{_label(layout)} is already registered to another side."
            )
    for sibling in _LAYOUTS.get(key, []):
        if sibling.scheme != layout.scheme:
            raise LayoutError(
                f"{layout.kind.value}'s {layout.side.value} side is {_label(sibling)}, "
                f"so it cannot also register {_label(layout)}."
            )
        _check_layouts_apart(sibling, layout)
    _LAYOUTS.setdefault(key, []).append(layout)
    _SCHEMES.setdefault(layout.scheme, []).append(layout)
    return layout


def _check_layouts_apart(one: Layout, two: Layout) -> None:
    """Refuse two layouts of one side whose locations can coincide or nest."""
    for (_, shape_a), (_, shape_b) in product(one.shapes(), two.shapes()):
        if all(
            _parts_may_coincide(one, a, b, other=two)
            for a, b in zip(shape_a, shape_b, strict=False)
        ):
            raise LayoutError(
                f"{_label(one)} registers two layouts whose locations can coincide or "
                f"nest: {[s.name for s in one.segments]} and {[s.name for s in two.segments]}."
            )


def _candidates(kind: Kind, side: Side, values: Mapping[str, str]) -> list[Layout]:
    """The side's layouts that hold every given segment and admit its value."""
    return [
        layout
        for layout in _LAYOUTS.get((kind, side), [])
        if all(
            name in layout._by_name and layout.segment(name).admits(value)
            for name, value in values.items()
        )
    ]


def _fitting(kind: Kind, side: Side, values: Mapping[str, str]) -> list[Layout]:
    """The layouts these values complete: every required segment is given."""
    return [
        layout
        for layout in _candidates(kind, side, values)
        if all(segment.name in values for segment in layout.required)
    ]


def _named_layouts(kind: Kind, side: Side, values: Mapping[str, str]) -> list[Layout]:
    """The layouts a reference's values name: the one they complete, else every
    one that could hold them."""
    return _fitting(kind, side, values) or _candidates(kind, side, values)


def layout_for(kind: Kind, side: Side) -> Layout:
    """A side's base layout: the first it registers, the one with no marker."""
    layouts = _LAYOUTS.get((kind, side), [])
    if layouts:
        return layouts[0]
    raise InvalidReferenceError(
        f"{kind.value} has no registered {side.value} side; "
        f"registered: {sorted({_label(layout) for layout in registered_layouts()})}."
    )


def registered_layouts() -> tuple[Layout, ...]:
    return tuple(layout for layouts in _LAYOUTS.values() for layout in layouts)


def _label(layout: Layout) -> str:
    return f"{layout.scheme}{URI_SEPARATOR}"


def reference(kind: Kind, side: Side, **values: str) -> Reference:
    """Build a reference from named segment values, validating each.

    All of a layout's required segments make a complete reference; fewer make a
    selection. Where the side registers several layouts the values pick among
    them, and a selection too short to pick one is ordered by the first.
    """
    given = {name: str(value) for name, value in values.items()}
    layouts = _LAYOUTS.get((kind, side), [])
    if not layouts:
        layout_for(kind, side)
    layout = (_named_layouts(kind, side, given) or layouts)[0]
    where = _label(layout)
    unknown = sorted(set(given) - {segment.name for segment in layout.segments})
    if unknown:
        raise InvalidReferenceError(
            f"{where} has no segment {unknown}; its segments are "
            f"{[segment.name for segment in layout.segments]}."
        )
    ordered: list[tuple[str, str]] = []
    gap = False
    for segment in layout.segments:
        if segment.name not in given:
            if segment.default is not None and not gap:
                ordered.append((segment.name, segment.default))
            elif not segment.optional:
                gap = True
            continue
        value = given[segment.name]
        if not segment.admits(value):
            raise InvalidReferenceError(
                f"{where} segment {segment.name!r} does not admit {value!r}."
            )
        ordered.append((segment.name, value))
    return Reference(kind=kind, side=side, values=tuple(ordered))


def parse_reference(uri: str) -> Reference:
    """The reference a URI renders; the inverse of `render_reference`."""
    if URI_SEPARATOR not in uri:
        raise InvalidReferenceError(f"{uri!r} has no '{URI_SEPARATOR}'.")
    scheme, rest = uri.split(URI_SEPARATOR, 1)
    layouts = _SCHEMES.get(scheme)
    if not layouts:
        raise InvalidReferenceError(
            f"{uri!r} has scheme {scheme!r}, which names no registered layout; "
            f"schemes are {sorted(_SCHEMES)}."
        )
    tokens = rest.split("/") if rest else []
    failure: InvalidReferenceError | None = None
    for layout in layouts:
        try:
            return _parse_tokens(uri, layout, list(tokens))
        except InvalidReferenceError as error:
            failure = failure or error
    assert failure is not None  # nosec B101 - type narrowing on an invariant the branch above holds
    raise failure


def _parse_tokens(uri: str, layout: Layout, tokens: list[str]) -> Reference:
    required = layout.required
    if len(tokens) > len(layout.segments):
        raise InvalidReferenceError(
            f"{uri!r} has more segments than {layout.scheme} holds."
        )
    values = {
        segment.name: token
        for segment, token in zip(required, tokens[: len(required)], strict=False)
        if token != SELECT_ANY
    }
    remaining = tokens[len(required) :]
    for segment in layout.optional:
        if remaining and remaining[0] == segment.literal:
            values[segment.name] = remaining.pop(0)
    if remaining:
        raise InvalidReferenceError(
            f"{uri!r} ends in {remaining}, which no optional segment of "
            f"{_label(layout)} names."
        )
    for name, value in values.items():
        if not layout.segment(name).admits(value):
            raise InvalidReferenceError(
                f"{_label(layout)} segment {name!r} does not admit {value!r}."
            )
    return reference(layout.kind, layout.side, **values)


def render_reference(ref: Reference) -> str:
    """A reference's URI: its side's scheme, then its values in registered order,
    with `*` for a required segment left out before one that is given."""
    layouts = _LAYOUTS[(ref.kind, ref.side)]
    given = ref.fields
    layout = (_named_layouts(ref.kind, ref.side, given) or layouts)[0]
    required = [segment.name for segment in layout.required]
    last = max((i for i, name in enumerate(required) if name in given), default=-1)
    tokens = [given.get(name, SELECT_ANY) for name in required[: last + 1]]
    tokens += [
        given[segment.name] for segment in layout.optional if segment.name in given
    ]
    return URI_SEPARATOR.join((layouts[0].scheme, "/".join(tokens)))


def side_root(roots: WorkspaceRoots, kind: Kind, side: Side) -> Path:
    """The directory every location of one side sits beneath."""
    return _SIDE_ROOTS[side](roots, kind)


def locate(roots: WorkspaceRoots, ref: Reference) -> Path:
    """Where a complete reference's directory is, whether or not it exists."""
    if not ref.complete:
        raise InvalidReferenceError(
            f"{ref.uri} is a selection, not one location; select it instead."
        )
    layout = ref.layout
    values = ref.fields
    shape = layout.shape(
        frozenset(name for name in values if layout.segment(name).optional)
    )
    return side_root(roots, ref.kind, ref.side).joinpath(
        *(_render_part(part, values) for part in shape)
    )


def selection_directory(
    roots: WorkspaceRoots, selection: Reference, *, layout: Layout | None = None
) -> Path:
    """The deepest directory every reference `selection` names sits beneath:
    the side's root joined with each leading path part the selection fixes.

    For a caller that keeps something of its own beside a group of references,
    such as a report over every run of one pairing, without composing the
    group's directory itself.
    """
    values = selection.fields
    directory = side_root(roots, selection.kind, selection.side)
    if layout is None:
        layouts = _named_layouts(selection.kind, selection.side, values)
        if len(layouts) != 1:
            return directory
        layout = layouts[0]
    present = frozenset(name for name in values if layout.segment(name).optional)
    for part in layout.shape(present):
        if isinstance(part, Part) and not all(name in values for name in part.segments):
            break
        directory = directory / _render_part(part, values)
    return directory


def reference_at(roots: WorkspaceRoots, location: Path) -> Reference:
    """The reference whose location `location` is; the inverse of `locate`.

    Reads nothing but the path.
    """
    target = Path(location).resolve()
    for layout in registered_layouts():
        base = side_root(roots, layout.kind, layout.side).resolve()
        if target == base or base not in target.parents:
            continue
        names = target.relative_to(base).parts
        for _, shape in layout.shapes():
            values = _parse_names(layout, shape, names)
            if values is not None:
                return reference(layout.kind, layout.side, **values)
    raise InvalidReferenceError(f"{target} is the location of no registered reference.")


def _parse_names(
    layout: Layout, shape: tuple[PathPart, ...], names: tuple[str, ...]
) -> dict[str, str] | None:
    if len(names) != len(shape):
        return None
    values: dict[str, str] = {}
    for part, name in zip(shape, names, strict=True):
        if isinstance(part, Fixed):
            if name != part.name:
                return None
            continue
        parsed = _parse_part(layout, part, name)
        if parsed is None:
            return None
        values.update(parsed)
    return values


def select_references(
    roots: WorkspaceRoots, selection: Reference, *, layout: Layout | None = None
) -> list[Reference]:
    """Every complete reference under `selection` whose directory exists, in
    URI order. A complete reference selects itself, and `layout` keeps a
    selection several layouts could hold to one of them."""
    known = selection.fields
    base = side_root(roots, selection.kind, selection.side)
    found: list[Reference] = []
    layouts = (
        [layout] if layout else _named_layouts(selection.kind, selection.side, known)
    )
    for one in layouts:
        present = frozenset(name for name in known if one.segment(name).optional)
        found.extend(
            reference(selection.kind, selection.side, **values)
            for values in _walk(one, one.shape(present), base, dict(known))
        )
    return sorted(found, key=lambda ref: ref.uri)


def _walk(
    layout: Layout, shape: tuple[PathPart, ...], directory: Path, values: dict[str, str]
) -> Iterator[dict[str, str]]:
    if not shape:
        yield values
        return
    part, rest = shape[0], shape[1:]
    if isinstance(part, Fixed) or all(name in values for name in part.segments):
        child = directory / _render_part(part, values)
        if child.is_dir():
            yield from _walk(layout, rest, child, values)
        return
    if not directory.is_dir():
        return
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_part(layout, part, child.name)
        if parsed is None or any(
            values.get(name, value) != value for name, value in parsed.items()
        ):
            continue
        yield from _walk(layout, rest, child, {**values, **parsed})


def require_reference(roots: WorkspaceRoots, selection: Reference) -> Reference:
    """The one existing reference `selection` names, checked before any work.

    Raises `ReferenceNotFoundError` naming what exists under the longest part
    of the selection that matches anything, and `AmbiguousReferenceError`
    naming the matches when there are several.
    """
    matches = select_references(roots, selection)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousReferenceError(
            f"{selection.uri} matches {len(matches)} references; name one of "
            f"{_listing(matches)}."
        )
    for broader in _broadenings(selection):
        existing = select_references(roots, broader)
        if existing:
            raise ReferenceNotFoundError(
                f"{selection.uri} does not exist; under {broader.uri} there is "
                f"{_listing(existing)}."
            )
    raise ReferenceNotFoundError(
        f"{selection.uri} does not exist, and nothing exists under "
        f"{selection.uri.split(URI_SEPARATOR)[0]}{URI_SEPARATOR}."
    )


def resolve_latest(roots: WorkspaceRoots, request: Reference) -> Reference:
    """The one reference a request names once everything it left out is pinned.

    A complete reference is itself. A `latest` segment left out means the
    greatest version that exists; any other segment left out, a seed say, has no
    order for a latest to mean anything, so it must be the only one on disk and
    several are refused by name. What is returned is what a record then carries:
    nothing locates, keys or records the request itself.
    """
    if request.complete:
        return request
    # A request holding no marker means the plain thing, `entities` being
    # implied and never written: the one layout whose every marker is given.
    plain = [
        layout
        for layout in _named_layouts(request.kind, request.side, request.fields)
        if all(
            segment.name in request.fields
            for segment in layout.required
            if _segment_words(segment) is not None
        )
    ]
    if len(plain) != 1:
        raise InvalidReferenceError(
            f"{request.uri} is a selection over {len(plain)} layouts, not a request for one thing."
        )
    layout = plain[0]
    found = select_references(roots, request, layout=layout)
    if not found:
        raise ReferenceNotFoundError(f"{request.uri} names nothing on disk.")
    for segment in layout.required:
        if segment.latest and segment.name not in request.fields:
            greatest = max(_version_order(ref.fields[segment.name]) for ref in found)
            found = [
                ref
                for ref in found
                if _version_order(ref.fields[segment.name]) == greatest
            ]
    if len(found) > 1:
        raise AmbiguousReferenceError(
            f"{request.uri} matches {len(found)} references with no order between "
            f"them; name one of {_listing(found)}."
        )
    return found[0]


def _version_order(value: str) -> tuple[int, str]:
    """Sorts `v10` after `v9` and a date by its digits: shorter first, then by text."""
    return len(value), value


def _broadenings(selection: Reference) -> Iterator[Reference]:
    """`selection` with its trailing values dropped one at a time."""
    values = selection.values
    for length in range(len(values) - 1, -1, -1):
        yield Reference(selection.kind, selection.side, values[:length])


def _listing(refs: Iterable[Reference]) -> str:
    uris = [ref.uri for ref in refs]
    shown = ", ".join(uris[:MAX_REPORTED_REFERENCES])
    extra = len(uris) - MAX_REPORTED_REFERENCES
    return f"{shown} and {extra} more" if extra > 0 else shown


def check_side_roots(roots: WorkspaceRoots) -> None:
    """Refuse registered sides whose roots coincide or nest under `roots`."""
    bases = {
        _label(layout): side_root(roots, layout.kind, layout.side).resolve()
        for layout in registered_layouts()
    }
    for (label_a, base_a), (label_b, base_b) in combinations(bases.items(), 2):
        if base_a == base_b or base_a in base_b.parents or base_b in base_a.parents:
            raise LayoutError(
                f"{label_a} at {base_a} and {label_b} at {base_b} share a location."
            )


PERTURBATION_PROFILE = _register(
    Layout(
        scheme="perturbation",
        kind=Kind.PERTURBATION,
        side=Side.PROFILE,
        segments=(
            Segment("name", NAME),
            Segment("version", VERSION, latest=True),
            Segment("stage", literal="draft"),
        ),
        # The stage sits between name and version, never after the version,
        # so a draft of v2 is beside the promoted v2 rather than inside it.
        path=(Part(("name",)), Part(("stage",)), Part(("version",))),
        immutable=True,
    )
)

NAMES_MARKER = Segment("dataset", one_of("names"))
"""The one dataset word: a system's name variants in place of its entities."""

PERTURBED_MARKER = Segment("derivation", one_of("perturbed"))

_PERTURBATION = (
    Segment("profile", NAME),
    Segment("version", VERSION, latest=True),
    Segment("seed", INTEGER),
)

PERTURBED_ENTITIES = _register(
    Layout(
        scheme="perturbed",
        kind=Kind.PERTURBATION,
        side=Side.DATA,
        segments=(Segment("source", NAME), *_PERTURBATION),
        path=(
            Part(("source",)),
            Part(("profile",)),
            Part(("version",)),
            Part(("seed",)),
        ),
        # Regenerated in place under one name, so its content is digested.
        immutable=False,
    )
)

PERTURBED_NAMES = _register(
    Layout(
        scheme="perturbed",
        kind=Kind.PERTURBATION,
        side=Side.DATA,
        segments=(Segment("source", NAME), NAMES_MARKER, *_PERTURBATION),
        path=(
            Part(("source",)),
            Part(("dataset",)),
            Part(("profile",)),
            Part(("version",)),
            Part(("seed",)),
        ),
        immutable=False,
    )
)

# A run is read by position, the source system, the target system and the
# representation, then whatever sets this source apart from the system's plain
# rows, then the key. Each kind of source is its own flat layout, and the tree
# still files a run target first, under a word for the kind of source, so
# everything scored against one target sits together.
_RUN_HEAD = (
    Segment("source", NAME),
    Segment("target", NAME),
    Segment("representation", REPRESENTATION),
)
_RUN_KEY = Segment("key", HEX)
_RUN_TAIL = (Part(("representation",)), Part(("key",)))
_PERTURBED_SOURCE = Part(("source", "profile", "version", "seed"), separator="_")
"""A perturbed source's parts as one directory, joined by a character no
kebab-case segment holds, so every kind of source sits at one depth."""

BLOCKING_PLAIN = _register(
    Layout(
        scheme="blocking",
        kind=Kind.BLOCKING,
        side=Side.DATA,
        segments=(*_RUN_HEAD, _RUN_KEY),
        path=(Part(("target",)), Fixed("data"), Part(("source",)), *_RUN_TAIL),
        # The key is the digest of everything the run consumed.
        immutable=True,
    )
)

BLOCKING_NAMES = _register(
    Layout(
        scheme="blocking",
        kind=Kind.BLOCKING,
        side=Side.DATA,
        segments=(*_RUN_HEAD, NAMES_MARKER, _RUN_KEY),
        path=(Part(("target",)), Part(("dataset",)), Part(("source",)), *_RUN_TAIL),
        immutable=True,
    )
)

BLOCKING_PERTURBED = _register(
    Layout(
        scheme="blocking",
        kind=Kind.BLOCKING,
        side=Side.DATA,
        segments=(*_RUN_HEAD, PERTURBED_MARKER, *_PERTURBATION, _RUN_KEY),
        path=(Part(("target",)), Part(("derivation",)), _PERTURBED_SOURCE, *_RUN_TAIL),
        immutable=True,
    )
)

BLOCKING_PERTURBED_NAMES = _register(
    Layout(
        scheme="blocking",
        kind=Kind.BLOCKING,
        side=Side.DATA,
        segments=(*_RUN_HEAD, NAMES_MARKER, PERTURBED_MARKER, *_PERTURBATION, _RUN_KEY),
        path=(
            Part(("target",)),
            Part(("derivation", "dataset"), separator="-"),
            _PERTURBED_SOURCE,
            *_RUN_TAIL,
        ),
        immutable=True,
    )
)

BLOCKING_BENCHMARK = _register(
    Layout(
        scheme="blocking",
        kind=Kind.BLOCKING,
        side=Side.DATA,
        # The one head that is not two systems: `benchmark` stands where the
        # source would and the benchmark's name where the target would, since
        # its two tables only ever meet each other.
        segments=(
            Segment("source", one_of("benchmark")),
            Segment("target", NAME),
            Segment("representation", REPRESENTATION),
            _RUN_KEY,
        ),
        path=(Part(("target",)), Part(("source",)), *_RUN_TAIL),
        immutable=True,
    )
)

# One audit per run, filed beside it under `blocking`'s own `audit`
# leaf (`kind_layout.Kind.BLOCKING_AUDIT`) rather than inside the run's
# directory, keyed on the same `key` segment as the run it audits so a
# second audit of one run resolves to the location the first wrote. Its own
# scheme, since a `(kind, side)` pair registers exactly one -- see
# `_register`'s docstring -- and `Kind.BLOCKING`'s `Side.DATA` is already
# `"blocking"`. Only a plain (`BlockingSourceKind.DATA`) run is covered
# today; `blocking.run_layout.audit_reference_for` refuses any other kind of
# source rather than guess at an unregistered shape.
BLOCKING_AUDIT_PLAIN = _register(
    Layout(
        scheme="blocking-audit",
        kind=Kind.BLOCKING_AUDIT,
        side=Side.DATA,
        segments=(*_RUN_HEAD, _RUN_KEY),
        path=(Part(("target",)), Part(("source",)), *_RUN_TAIL),
        immutable=True,
    )
)

DATE = SegmentType(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", frozenset(string.digits) | {"-"})
"""A dated snapshot's version: its acquisition date, which sorts."""

CANONICAL_DATASET = "canonical"
NAMES_DATASET = "names"
MATCHED_DATASET = "matched"

# A system registers three datasets, each a flat layout told apart by its
# dataset word, and each locating where it sits today: the path holds the fixed
# parts the URI leaves out. `names` is a dataset in its own right and a peer of
# `canonical`, though its files sit in the same dated snapshot. Acquire, shard,
# prepare and source are acquisition's own business and get no reference.
DATA_CANONICAL = _register(
    Layout(
        scheme="data",
        kind=Kind.DATA,
        side=Side.DATA,
        segments=(
            Segment("system", SLUG),
            Segment("dataset", one_of(CANONICAL_DATASET)),
            Segment("version", DATE, latest=True),
        ),
        path=(
            Part(("system",)),
            Part(("dataset",)),
            Part(("version",)),
            Fixed("primary"),
        ),
        # A snapshot is regenerated in place under its date until it is sealed,
        # so its content is digested wherever it is consumed.
        immutable=False,
    )
)

DATA_NAMES = _register(
    Layout(
        scheme="data",
        kind=Kind.DATA,
        side=Side.DATA,
        segments=(
            Segment("system", SLUG),
            Segment("dataset", one_of(NAMES_DATASET)),
            Segment("version", DATE, latest=True),
        ),
        path=(
            Part(("system",)),
            Fixed(CANONICAL_DATASET),
            Part(("version",)),
            Part(("dataset",)),
        ),
        immutable=False,
    )
)

DATA_MATCHED = _register(
    Layout(
        scheme="data",
        kind=Kind.DATA,
        side=Side.DATA,
        segments=(
            Segment("system", SLUG),
            Segment("dataset", one_of(MATCHED_DATASET)),
        ),
        path=(
            Part(("system",)),
            Part(("dataset",)),
            Fixed("current"),
            Fixed("primary"),
        ),
        # Regenerated in place under one name and no date, so its content is
        # digested wherever it is consumed.
        immutable=False,
    )
)

TOKENIZER_CANDIDATE = _register(
    Layout(
        scheme="tokenizer",
        kind=Kind.TOKENIZER,
        side=Side.DATA,
        segments=(
            Segment("scope", NAME),
            Segment("tokenizer_id", IDENTIFIER),
            Segment("key", IDENTIFIER),
        ),
        path=(Part(("scope",)), Part(("tokenizer_id",)), Part(("key",))),
        # The key is `company_tokenize`'s own derivation, which names what the
        # candidate was trained from; the files inside are that package's too.
        immutable=True,
    )
)


def run_source(run: Reference) -> Reference:
    """The source a run's URI names, rebuilt from its head and tail with no
    lookup: a plain source is the system's labelled rows, `matched`; `names` is
    the system's names, a request for the latest; and a perturbed source is the
    perturbed dataset itself."""
    fields = run.fields
    if run.kind is not Kind.BLOCKING or "source" not in fields:
        raise InvalidReferenceError(f"{run.uri} names no run's source.")
    if fields["source"] in RESERVED_WORDS:
        raise InvalidReferenceError(
            f"{run.uri} is a benchmark run, whose tables are no dataset."
        )
    names = {"dataset": fields["dataset"]} if "dataset" in fields else {}
    if "derivation" in fields:
        return reference(
            Kind.PERTURBATION,
            Side.DATA,
            source=fields["source"],
            profile=fields["profile"],
            version=fields["version"],
            seed=fields["seed"],
            **names,
        )
    return reference(
        Kind.DATA,
        Side.DATA,
        system=fields["source"],
        dataset=names.get("dataset", MATCHED_DATASET),
    )


__all__ = [
    "BLOCKING_AUDIT_PLAIN",
    "BLOCKING_BENCHMARK",
    "BLOCKING_NAMES",
    "BLOCKING_PERTURBED",
    "BLOCKING_PERTURBED_NAMES",
    "BLOCKING_PLAIN",
    "CANONICAL_DATASET",
    "TOKENIZER_CANDIDATE",
    "DATA_CANONICAL",
    "DATA_MATCHED",
    "DATA_NAMES",
    "DATE",
    "MATCHED_DATASET",
    "NAMES_DATASET",
    "HEX",
    "IDENTIFIER",
    "INTEGER",
    "MAX_REPORTED_REFERENCES",
    "NAME",
    "NAMES_MARKER",
    "PERTURBED_ENTITIES",
    "PERTURBED_MARKER",
    "PERTURBED_NAMES",
    "PERTURBATION_PROFILE",
    "REPRESENTATION",
    "RESERVED_WORDS",
    "SELECT_ANY",
    "SLUG",
    "URI_SEPARATOR",
    "VERSION",
    "AmbiguousReferenceError",
    "Fixed",
    "InvalidReferenceError",
    "Layout",
    "LayoutError",
    "Part",
    "PathPart",
    "Reference",
    "ReferenceNotFoundError",
    "Segment",
    "SegmentType",
    "Side",
    "check_side_roots",
    "layout_for",
    "locate",
    "one_of",
    "parse_reference",
    "reference",
    "reference_at",
    "registered_layouts",
    "render_reference",
    "require_reference",
    "resolve_latest",
    "run_source",
    "select_references",
    "selection_directory",
    "side_root",
]
