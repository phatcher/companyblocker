"""A fixture corpus shaped to exercise the contracts between pipeline stages.

Every other test in the repository writes the input its one stage expects.
That leaves the handoffs untested: a stage can change the shape of what it
emits and every test still passes, because no test reads a real upstream
output. This corpus exists so a test can drive the real stages in order and
let each one read only what its predecessor actually wrote.

**What the corpus is.** Two systems' Acquire-layer payloads, in the formats
their real plans declare -- a GLEIF LEI-CDF XML inside a zip, and a
Companies House CSV inside a zip -- written under `tmp_path`. Nothing below
Acquire is materialized: the stages produce it.

**Why these two systems.** They are the pair the production match flow
already runs (`gleif` as the multi-jurisdiction source, `gb` as the national
register it resolves against), so the chain a test drives is the one
operators drive rather than a synthetic arrangement. Between them they carry
what the contracts need:

  - `gleif` spans two jurisdictions, so its canonical layer is genuinely
    Hive-partitioned and the flat-versus-partitioned distinction is real
    rather than a single-partition special case;
  - both carry name variants (GLEIF `OtherEntityNames`, Companies House
    `PreviousName_N.CompanyName`), so the `names/` family is populated and
    the sidecar handoff is exercised;
  - one entity is present in both systems under the same registration
    number and differing legal names ("ACME HOLDINGS LIMITED" against
    "ACME HOLDINGS LTD"), so Match produces a real `match_uri` for a pair
    that only agrees once cleansed -- ground truth a blocking run can score
    against, and an entity whose cleansed form differs from its raw name.

**Size.** Five entities in total. The contracts are about shape, not volume,
and every stage here is exercised by the shape of its input rather than the
amount of it.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from workspace.data_layout import layer_directory
from workspace.roots import WorkspaceRoots, default_workspace_roots

CORPUS_RUN_DATE = "2026-06-01"
"""The snapshot date every system in the corpus is acquired under.

A month start, because `gb`'s plan declares `snapshot_date_mode: month_start`
and resolves any run date back to one, while `gleif`'s uses the run date as
given. Only a month start makes both land on the same snapshot directory,
which the stages need if one run date is to drive them both.
"""

SOURCE_SYSTEM = "gleif"
TARGET_SYSTEM = "gb"

SOURCE_JURISDICTIONS = ("gb", "ie")
"""The jurisdictions `gleif`'s canonical layer partitions into."""

SHARED_JURISDICTION = "gb"
"""The one jurisdiction both systems hold, so Match and the blocking run
have a partition in common to work over."""

_GLEIF_LEI_CDF_XML = """<?xml version="1.0" encoding="UTF-8"?>
<lei:LEIData xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
    <lei:LEIRecords>
        <lei:LEIRecord>
            <lei:LEI>PIPE0GB000000000001</lei:LEI>
            <lei:Entity>
                <lei:LegalName>ACME HOLDINGS LTD</lei:LegalName>
                <lei:OtherEntityNames>
                    <lei:OtherEntityName type="PREVIOUS_LEGAL_NAME">ACME HOLDINGS PLC</lei:OtherEntityName>
                </lei:OtherEntityNames>
                <lei:LegalJurisdiction>GB</lei:LegalJurisdiction>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
                <lei:RegistrationAuthority>
                    <lei:RegistrationAuthorityID>RA000585</lei:RegistrationAuthorityID>
                    <lei:RegistrationAuthorityEntityID>00000001</lei:RegistrationAuthorityEntityID>
                </lei:RegistrationAuthority>
                <lei:LegalAddress>
                    <lei:FirstAddressLine>1 High Street</lei:FirstAddressLine>
                    <lei:City>London</lei:City>
                    <lei:Country>GB</lei:Country>
                </lei:LegalAddress>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2015-03-10T00:00:00Z</lei:InitialRegistrationDate>
            </lei:Registration>
        </lei:LEIRecord>
        <lei:LEIRecord>
            <lei:LEI>PIPE0GB000000000003</lei:LEI>
            <lei:Entity>
                <lei:LegalName>ZETA UNRELATED LIMITED</lei:LegalName>
                <lei:LegalJurisdiction>GB</lei:LegalJurisdiction>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
                <lei:RegistrationAuthority>
                    <lei:RegistrationAuthorityID>RA000585</lei:RegistrationAuthorityID>
                    <lei:RegistrationAuthorityEntityID>00009999</lei:RegistrationAuthorityEntityID>
                </lei:RegistrationAuthority>
            </lei:Entity>
        </lei:LEIRecord>
        <lei:LEIRecord>
            <lei:LEI>PIPE0IE000000000002</lei:LEI>
            <lei:Entity>
                <lei:LegalName>BETA TRADING LIMITED</lei:LegalName>
                <lei:LegalJurisdiction>IE</lei:LegalJurisdiction>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
                <lei:RegistrationAuthority>
                    <lei:RegistrationAuthorityID>RA000590</lei:RegistrationAuthorityID>
                    <lei:RegistrationAuthorityEntityID>222222</lei:RegistrationAuthorityEntityID>
                </lei:RegistrationAuthority>
            </lei:Entity>
        </lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""

_GB_BASIC_COMPANY_DATA_CSV = (
    "CompanyName,CompanyNumber,CompanyCategory,CompanyStatus,IncorporationDate,"
    "URI,PreviousName_1.CompanyName\n"
    "ACME HOLDINGS LIMITED,00000001,Private Limited Company,Active,2015-03-10,"
    "https://find-and-update.company-information.service.gov.uk/company/00000001,"
    "ACME HOLDINGS PLC\n"
    "OMEGA WIDGETS PLC,00000002,Public Limited Company,Active,2001-01-05,"
    "https://find-and-update.company-information.service.gov.uk/company/00000002,\n"
)


@dataclass(frozen=True)
class PipelineCorpus:
    """A materialized corpus and the handful of facts a test needs to drive it.

    Holds only what the fixture decided and a test cannot re-derive without
    repeating that decision: where the corpus is, which snapshot date it was
    written under, and which systems and jurisdictions it covers. Everything
    else -- what any stage wrote, and where -- is read back through the
    stages' own resolvers, which is the point of the exercise.
    """

    root: Path
    run_date: str
    source_system: str
    target_system: str
    source_jurisdictions: tuple[str, ...]
    shared_jurisdiction: str

    @property
    def roots(self) -> WorkspaceRoots:
        """The corpus's root as the resolved roots every layout function and
        stage takes; `root` stays the `Path` the fixture paths are composed from."""
        return default_workspace_roots(self.root)

    def acquire_dir(self, system: str) -> Path:
        return self.root / "data" / system / "acquire" / self.run_date

    def source_root(self, system: str) -> Path:
        return self.root / "data" / system / "source" / self.run_date

    def canonical_dir(self, system: str) -> Path:
        return self.root / "data" / system / "canonical" / self.run_date

    def layer_dir(self, system: str, layer: str) -> Path:
        return layer_directory(self.root / "data" / system, layer)


def write_pipeline_corpus(root: Path) -> PipelineCorpus:
    """Materialize the corpus's Acquire-layer payloads under `root`.

    Separate from the fixture so a test needing a variant -- a second
    snapshot, a system removed -- can call it directly rather than having the
    fixture grow parameters for every such case.
    """
    corpus = PipelineCorpus(
        root=root,
        run_date=CORPUS_RUN_DATE,
        source_system=SOURCE_SYSTEM,
        target_system=TARGET_SYSTEM,
        source_jurisdictions=SOURCE_JURISDICTIONS,
        shared_jurisdiction=SHARED_JURISDICTION,
    )

    gleif_dir = corpus.acquire_dir(SOURCE_SYSTEM)
    gleif_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        gleif_dir / f"gleif-lei-cdf-{CORPUS_RUN_DATE}.zip", "w"
    ) as archive:
        archive.writestr("gleif-lei-cdf.xml", _GLEIF_LEI_CDF_XML)

    gb_dir = corpus.acquire_dir(TARGET_SYSTEM)
    gb_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        gb_dir / f"BasicCompanyDataAsOneFile-{CORPUS_RUN_DATE}.zip", "w"
    ) as archive:
        archive.writestr("BasicCompanyDataAsOneFile.csv", _GB_BASIC_COMPANY_DATA_CSV)

    return corpus
