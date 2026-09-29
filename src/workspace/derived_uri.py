"""One composer and one parser for the URI of a derived row and its dataset.

A derived URI is its derivation and the source URI it was derived from, ordered
from the general to the specific: the scheme, the system, the dataset word where
the rows are not the system's entities, the derivation's parameters, and last
the row's own identity.

    gb://123                                         an entity, the underived case
    name://gb/123/<hash>                             a name variant of it
    perturbed://ie/en-lite/v1/42/123                 a perturbed copy of an entity
    perturbed://gb/names/en-lite/v1/42/123/<hash>    a perturbed name variant

Every such URI decomposes to where the row came from with no lookup
(`DerivedUri.source`), and none says where anything is stored. A derived dataset
is the same URI less the row's own identity (`DerivedUri.dataset`), so a row and
its dataset share this one composer and parser:

    perturbed://ie/en-lite/v1/42          ie's entities under a profile and seed
    perturbed://gb/names/en-lite/v1/42    gb's names under the same

`names` is the one dataset word and `entities` is implied and never written.
Nothing is told apart by counting segments alone: the `names` marker says whether
a name hash follows the entity id.

An entity's own URI is not re-spelled: its local id follows the scheme as it
always has. Inside a derived URI the id is one segment, so a `/` in it is
percent-encoded, and `%` with it so the encoding reads back.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from .kind_layout import Kind
from .reference import (
    HEX,
    NAME,
    PERTURBED_ENTITIES,
    PERTURBED_NAMES,
    URI_SEPARATOR,
    VERSION,
    InvalidReferenceError,
    Reference,
    Side,
    parse_reference,
    reference,
)

NAME_SCHEME = "name"
PERTURBED_SCHEME = "perturbed"
DERIVED_SCHEMES: frozenset[str] = frozenset({NAME_SCHEME, PERTURBED_SCHEME})

NAMES_DATASET = "names"
"""The one dataset word: a system's name variants in place of its entities."""

NAME_HASH_LENGTH = 16

_REFERENCE_SCHEMES: frozenset[str] = frozenset(kind.value for kind in Kind)
"""A reference's scheme names a concern, so it is never read as a system code."""

_ENCODED = {"%": "%25", "/": "%2F"}
_DECODED = {encoded.lower(): raw for raw, encoded in _ENCODED.items()}
_ENCODED_PATTERN = re.compile("|".join(_DECODED), re.IGNORECASE)


def name_hash(*, name_type: str, value: str) -> str:
    """The hash segment of a name variant's URI, over its name type and value.

    Each field is hashed length-prefixed rather than joined, since a company
    name holds any separator a join could use. The entity is not hashed in: the
    URI's own path already says whose name it is.
    """
    hasher = hashlib.sha256()
    for field in (name_type, value):
        encoded = field.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()[:NAME_HASH_LENGTH]


@dataclass(frozen=True)
class Perturbation:
    """What a perturbed dataset was made with: the profile, its version and the seed."""

    profile: str
    version: str
    seed: int

    def __post_init__(self) -> None:
        if not NAME.admits(self.profile):
            raise InvalidReferenceError(f"{self.profile!r} is no profile name.")
        if not VERSION.admits(self.version):
            raise InvalidReferenceError(f"{self.version!r} is no profile version.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise InvalidReferenceError(f"{self.seed!r} is no seed.")


@dataclass(frozen=True)
class DerivedUri:
    """A row's identity, or with no `entity_id` the dataset such rows belong to."""

    system: str
    names: bool = False
    perturbation: Perturbation | None = None
    entity_id: str | None = None
    name_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            not NAME.admits(self.system)
            or self.system in DERIVED_SCHEMES
            or self.system in _REFERENCE_SCHEMES
        ):
            raise InvalidReferenceError(f"{self.system!r} is no system code.")
        if self.entity_id is not None and not self.entity_id:
            raise InvalidReferenceError("An entity id is empty.")
        hashed = self.names and self.entity_id is not None
        if hashed != (self.name_hash is not None):
            raise InvalidReferenceError(
                "A name hash belongs to exactly a name row: `names` with an entity id."
            )
        if self.name_hash is not None and not (
            HEX.admits(self.name_hash) and len(self.name_hash) == NAME_HASH_LENGTH
        ):
            raise InvalidReferenceError(f"{self.name_hash!r} is no name hash.")

    @property
    def is_row(self) -> bool:
        return self.entity_id is not None

    @property
    def derived(self) -> bool:
        """Whether this is anything but an entity or a system's entities."""
        return self.names or self.perturbation is not None

    @property
    def source(self) -> DerivedUri | None:
        """What this was derived from, one step back; None for the underived case.

        A perturbed row gives the row that was perturbed, a name variant its
        entity, and a perturbed dataset the dataset that was perturbed.
        """
        if self.perturbation is not None:
            return replace(self, perturbation=None)
        if self.names:
            return replace(self, names=False, name_hash=None)
        return None

    @property
    def dataset(self) -> DerivedUri:
        """This URI less the row's own identity."""
        return replace(self, entity_id=None, name_hash=None)

    @property
    def dataset_reference(self) -> Reference:
        """The registered reference of the perturbed dataset this belongs to."""
        if self.perturbation is None:
            raise InvalidReferenceError(
                f"{self.system}'s {'names' if self.names else 'entities'} are a stored "
                "dataset, not a derived one, and have no derived URI."
            )
        return reference(
            Kind.PERTURBATION,
            Side.DATA,
            source=self.system,
            profile=self.perturbation.profile,
            version=self.perturbation.version,
            seed=str(self.perturbation.seed),
            **({"dataset": NAMES_DATASET} if self.names else {}),
        )

    @property
    def uri(self) -> str:
        return compose_uri(self)

    def __str__(self) -> str:
        return self.uri


def name_variant_uri(*, source_uri: str, name_type: str, value: str) -> str:
    """A name variant's URI, from the entity it is a name of and the name itself:
    the entity's own URI supplies the path, and the hash covers the type and value."""
    entity = parse_uri(source_uri)
    if entity.derived or not entity.is_row:
        raise InvalidReferenceError(f"{source_uri!r} is no entity to hold a name.")
    return replace(
        entity, names=True, name_hash=name_hash(name_type=name_type, value=value)
    ).uri


def perturbed_uri(*, source_uri: str, perturbation: Perturbation) -> str:
    """A perturbed row's URI, from the row that was perturbed, an entity or a
    name variant, and what it was perturbed with."""
    source = parse_uri(source_uri)
    if source.perturbation is not None or not source.is_row:
        raise InvalidReferenceError(f"{source_uri!r} is no row to perturb.")
    return replace(source, perturbation=perturbation).uri


def source_uri_of(uri: str) -> str | None:
    """The URI a derived row came from, one step back, or None for an entity."""
    source = parse_uri(uri).source
    return None if source is None else source.uri


def _encode_id(entity_id: str) -> str:
    return "".join(_ENCODED.get(character, character) for character in entity_id)


def _decode_id(segment: str) -> str:
    return _ENCODED_PATTERN.sub(lambda match: _DECODED[match.group().lower()], segment)


def compose_uri(derived: DerivedUri) -> str:
    """The URI of a row or a derived dataset; the inverse of `parse_uri`."""
    if derived.perturbation is None:
        if derived.entity_id is None:
            return derived.dataset_reference.uri
        if not derived.names:
            return f"{derived.system}{URI_SEPARATOR}{derived.entity_id}"
        segments = [
            derived.system,
            _encode_id(derived.entity_id),
            derived.name_hash or "",
        ]
        return f"{NAME_SCHEME}{URI_SEPARATOR}{'/'.join(segments)}"

    identity = [] if derived.entity_id is None else [_encode_id(derived.entity_id)]
    if derived.name_hash is not None:
        identity.append(derived.name_hash)
    return "/".join([derived.dataset_reference.uri, *identity])


def parse_uri(uri: str) -> DerivedUri:
    """The row or derived dataset a URI names; the inverse of `compose_uri`."""
    if URI_SEPARATOR not in uri:
        raise InvalidReferenceError(f"{uri!r} has no '{URI_SEPARATOR}'.")
    scheme, rest = uri.split(URI_SEPARATOR, 1)
    if not rest:
        raise InvalidReferenceError(f"{uri!r} names nothing after its scheme.")
    if scheme not in DERIVED_SCHEMES:
        return DerivedUri(system=scheme, entity_id=rest)

    tokens = rest.split("/")
    if not all(tokens):
        raise InvalidReferenceError(f"{uri!r} has an empty segment.")
    if scheme == NAME_SCHEME:
        if len(tokens) != 3:
            raise InvalidReferenceError(
                f"{uri!r} is not '{NAME_SCHEME}{URI_SEPARATOR}<system>/<id>/<hash>'."
            )
        system, entity_id, hashed = tokens
        return DerivedUri(
            system=system, names=True, entity_id=_decode_id(entity_id), name_hash=hashed
        )

    # The dataset part is the registered layout's to parse; what follows it is
    # the row's own identity, an id and for a name its hash.
    for layout, identity_length in (
        (None, 0),
        (PERTURBED_ENTITIES, 1),
        (PERTURBED_NAMES, 2),
    ):
        head = tokens[: len(tokens) - identity_length]
        try:
            dataset = parse_reference(f"{scheme}{URI_SEPARATOR}{'/'.join(head)}")
        except InvalidReferenceError:
            continue
        if not dataset.complete or (
            layout is not None and dataset.layout is not layout
        ):
            continue
        fields = dataset.fields
        identity = tokens[len(head) :]
        return DerivedUri(
            system=fields["source"],
            names="dataset" in fields,
            perturbation=Perturbation(
                profile=fields["profile"],
                version=fields["version"],
                seed=int(fields["seed"]),
            ),
            entity_id=_decode_id(identity[0]) if identity else None,
            name_hash=identity[1] if len(identity) == 2 else None,
        )
    raise InvalidReferenceError(
        f"{uri!r} is not '{PERTURBED_SCHEME}{URI_SEPARATOR}<system>[/{NAMES_DATASET}]"
        "/<profile>/<version>/<seed>' with, for a row, its id and for a name its hash."
    )


__all__ = [
    "DERIVED_SCHEMES",
    "NAMES_DATASET",
    "NAME_HASH_LENGTH",
    "NAME_SCHEME",
    "PERTURBED_SCHEME",
    "DerivedUri",
    "Perturbation",
    "compose_uri",
    "name_hash",
    "name_variant_uri",
    "parse_uri",
    "perturbed_uri",
    "source_uri_of",
]
