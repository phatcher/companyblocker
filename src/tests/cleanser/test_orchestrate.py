import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from company_cleanse.normalize import normalize_company_type_value
from company_cleanse.pipeline import (
    CANONICAL_COMPANY_TYPE_STEP,
    DERIVE_ACRONYM_FIELD_STEP,
    ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
    ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    GENERATE_CLEANSED_COMPANY_NAME_STEP,
    STEP_ENGINE_POLARS,
    STEP_ENGINE_UDF,
)
from company_cleanse.rules import (
    _build_company_type_mapping,
    get_company_type_rules,
    get_company_type_rules_for_country,
)

from acquisition import cleanser_orchestrate
from acquisition.cleanser_orchestrate import name_cleanse
from acquisition.registry import get_system_company_type_mapping

cases_path = Path(__file__).with_name("cleanser_cases.json")
CLEANSER_CASES = json.loads(cases_path.read_text(encoding="utf-8"))

_ALL_SWITCHABLE_STEPS = (
    CANONICAL_COMPANY_TYPE_STEP,
    GENERATE_CLEANSED_COMPANY_NAME_STEP,
    DERIVE_ACRONYM_FIELD_STEP,
    ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
    ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
)


def _step_engines_for_mode(step_engine_mode: str) -> dict[str, str]:
    engine = STEP_ENGINE_UDF if step_engine_mode == "udf" else STEP_ENGINE_POLARS
    return {step_name: engine for step_name in _ALL_SWITCHABLE_STEPS}


@pytest.fixture(params=["udf", "polars"], ids=["engines_udf", "engines_polars"])
def step_engine_mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.mark.parametrize(
    "case",
    CLEANSER_CASES,
    ids=lambda case: case["input"],
)
def test_expected_cleansing_cases(case, step_engine_mode: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": [case["input"]],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules()
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        expected = case["expected"]
        for key in [
            "short_name",
            "quoted_name",
            "company_type",
            "name_cleansed",
            "acronym",
        ]:
            actual_value = row[key]
            expected_value = expected.get(key)
            if isinstance(actual_value, str) and isinstance(expected_value, str):
                assert actual_value == expected_value.lower()
            else:
                assert actual_value == expected_value


def test_conflicting_company_type_rules_fail_fast():
    conflicting_rules = [
        ("Aktiengesellschaft", "AG"),
        ("Aktiengesellschaft", "SA"),
    ]
    canonical_types = {"AG", "SA"}

    with pytest.raises(ValueError, match="Conflicting company type mappings detected"):
        _build_company_type_mapping(conflicting_rules, canonical_types)


def test_source_company_type_column_mapping_takes_precedence(step_engine_mode: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["EXAMPLE FUND"],
                "CompanyCategory": ["Investment Company with Variable Capital"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Investment Company with Variable Capital": "Investment Company",
            },
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == "investment company"


def test_name_extracted_type_takes_precedence_over_source_type(step_engine_mode: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["!Abridge Tax Ltd"],
                "CompanyCategory": ["Public Limited Company"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Public Limited Company": "Public Limited Company",
            },
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == "ltd"
        assert "!abridge tax ltd" in row["name_cleansed"]
        assert " p l c" not in row["name_cleansed"]


def test_source_company_type_fallback_can_classify_llp():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["ABRIDGE TAX"],
                "CompanyCategory": ["Limited Liability Partnership"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Limited Liability Partnership": "Limited Liability Partnership",
            },
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == "llp"
        assert row["name_cleansed"] == "abridge tax llp"


@pytest.mark.parametrize(
    "input_name, expected_cleansed",
    [
        ("(LTD) ALPHA CARE LTD", "ltd alpha care ltd"),
        ("(CIC) COMMUNITY CARE", "cic community care"),
    ],
)
def test_non_acronym_leading_short_name_is_preserved_in_cleansed_name(
    input_name: str,
    expected_cleansed: str,
    step_engine_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": [input_name],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["name_cleansed"] == expected_cleansed


def test_source_company_type_known_target_appends_suffix():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["DEEP CLEAN"],
                "CompanyCategory": ["Private Limited Company"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Private Limited Company": "Limited",
            },
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == "ltd"
        assert row["name_cleansed"] == "deep clean ltd"


def test_quoted_name_short_name_override_uses_exact_company_stem(step_engine_mode: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ['"TRIPLE D" LTD'],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["quoted_name"] == "triple d"
        assert row["short_name"] == "triple d"
        assert row["name_cleansed"] == "triple d ltd"


def test_quoted_name_trims_trailing_company_type_suffix(step_engine_mode: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ['"ABC LIMITED" LTD'],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["quoted_name"] == "abc"
        assert row["short_name"] == "abc"
        assert row["name_cleansed"] == "abc ltd"


def test_single_character_quoted_name_is_preserved_in_cleansed_name(
    step_engine_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ['"K" LINE SHIPPING LTD'],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["quoted_name"] == "k"
        assert row["short_name"] == "k line shipping"
        assert row["name_cleansed"] == "k line shipping ltd"


def test_source_company_type_unknown_target_does_not_append_suffix(
    step_engine_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["DEEP CLEAN"],
                "CompanyCategory": ["Overseas entity"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Overseas entity": "Overseas entity",
            },
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == "overseas entity"
        assert row["name_cleansed"] == "deep clean"


def test_fr_source_mapping_non_extension_keeps_company_type_without_suffix(
    step_engine_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "fr"
        )
        source_company_type_mapping = get_system_company_type_mapping("fr")

        candidate = None
        for source_value, mapped_value in sorted(source_company_type_mapping.items()):
            normalized = normalize_company_type_value(mapped_value, transliterate=True)
            if normalized and normalized not in company_type_mapping:
                candidate = (source_value, mapped_value)
                break

        assert candidate is not None
        source_value, mapped_value = candidate

        source = pl.DataFrame(
            {
                "CompanyName": ["EXEMPLE FONDS"],
                "CompanyCategory": [source_value],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["ET"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping=source_company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == mapped_value.lower()
        assert row["name_cleansed"] == "exemple fonds"
        assert row["company_type_source"] == "source_mapping"


def test_fr_source_mapping_resolving_type_appends_suffix():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["IMMOBILIERE DES JARDINS"],
                "CompanyCategory": ["6540"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "fr"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["ET"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            source_company_type_col="CompanyCategory",
            source_company_type_mapping=get_system_company_type_mapping("fr"),
        )
        assert rows == 1

        actual = pl.read_parquet(output_dir / "chunks" / "cases.parquet")
        row = actual.row(0, named=True)
        assert row["company_type"] == "sci"
        assert row["name_cleansed"] == "immobiliere des jardins sci"
        assert row["company_type_source"] == "source_mapping"


def test_name_cleanse_writes_empty_cleanser_when_blank_cleansed_rows_exist(
    step_engine_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["???", "ALPHA LTD"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 2

        empty_output = output_dir / "empty_cleanser.parquet"
        assert empty_output.exists()
        empties = pl.read_parquet(empty_output)
        assert empties.height == 1


def test_name_cleanse_removes_stale_empty_cleanser_when_no_blank_rows(
    step_engine_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        stale_empty_output = output_dir / "empty_cleanser.parquet"
        pl.DataFrame({"CompanyName": ["STALE"]}).write_parquet(stale_empty_output)

        source = pl.DataFrame(
            {
                "CompanyName": ["ALPHA LTD", "BETA LTD"],
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
        assert rows == 2

        assert not stale_empty_output.exists()


def test_name_cleanse_empty_sidecar_step_fails_on_missing_cleansed_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_engine_mode: str,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"CompanyName": ["ALPHA LTD"]}).write_parquet(
        input_dir / "cases.parquet"
    )

    def _write_without_cleansed_column(_lf, output_path: str, **_kwargs) -> int:
        pl.DataFrame({"CompanyName": ["ALPHA LTD"]}).write_parquet(output_path)
        return 1

    monkeypatch.setattr(
        cleanser_orchestrate, "_write_cleanse_output", _write_without_cleansed_column
    )

    company_type_regex, company_type_mapping = get_company_type_rules_for_country("gb")
    with pytest.raises(ValueError, match="name_cleansed"):
        name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col="CompanyName",
            and_tokens=["AND"],
            char_whitelist=r"[^a-z0-9\s!&]",
            company_type_mapping=company_type_mapping,
            step_engines=_step_engines_for_mode(step_engine_mode),
        )
