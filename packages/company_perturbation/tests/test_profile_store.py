"""Direct tests for `company_perturbation.profile_store`.

The package reads one profile from a path it is given and holds nothing about
where profiles live. What it does own is the invariant that a profile answers to
one name only.
"""

from __future__ import annotations

import json

import pytest
from company_perturbation.profile_schema import PerturbationProfile
from company_perturbation.profile_store import ProfileIdMismatchError, read_profile


def _write(path, declared_id: str = "light-noise"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "profile_id": declared_id,
                "scenarios": [
                    {
                        "scenario_id": "one-typo",
                        "chain": [{"operator_id": "typo.keyboard_substitution"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_profile_read_is_the_one_the_file_holds(tmp_path):
    path = _write(tmp_path / "light-noise" / "v1" / "profile.json")

    profile = read_profile(path, profile_id="light-noise")

    assert isinstance(profile, PerturbationProfile)
    assert [scenario.scenario_id for scenario in profile.scenarios] == ["one-typo"]


def test_a_profile_that_disagrees_with_the_name_it_is_read_as_is_an_error(tmp_path):
    """Otherwise one profile answers to two names and a run cannot be replayed."""
    path = _write(tmp_path / "profile.json", declared_id="heavy-noise")

    with pytest.raises(ProfileIdMismatchError, match="must answer to one name only"):
        read_profile(path, profile_id="light-noise")


def test_reading_a_file_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_profile(tmp_path / "profile.json", profile_id="light-noise")
