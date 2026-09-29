"""Reading one authored profile from a file the caller supplies.

`profile_schema.py` is the schema and stays free of I/O; this module is the file
half, and the only place in the package that touches disk. The package holds no
location of its own, ships no profiles and keeps no registry: where a profile
lives, and which profiles exist, belong to the repository using this package.

One invariant is enforced here: a profile's declared `profile_id` matches the
name the caller asked for. Without it the same profile answers to two names, and
a run recorded under one cannot be replayed from the name it recorded, which is
the one thing naming a profile is for.
"""

from __future__ import annotations

import json
from pathlib import Path

from .profile_schema import PerturbationProfile, parse_profile


class ProfileIdMismatchError(ValueError):
    """A profile's declared `profile_id` disagrees with the name it was read as."""


def read_profile(path: Path, *, profile_id: str) -> PerturbationProfile:
    """Parse the profile at `path`, which must declare `profile_id`."""
    path = Path(path)
    profile = parse_profile(json.loads(path.read_text(encoding="utf-8")))
    if profile.profile_id != profile_id:
        raise ProfileIdMismatchError(
            f"{path} declares profile_id {profile.profile_id!r}, so it cannot be "
            f"read as {profile_id!r}: a profile must answer to one name only."
        )
    return profile
