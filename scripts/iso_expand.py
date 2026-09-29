#!/usr/bin/env python3
"""Per-country ISO 20275 coverage: how many active legal forms are unmapped, and how many cannot be mapped for want of an abbreviation.

`generate_iso20275_additions.py` reads the same two files but proposes canonical values for
the forms it can, so a country with no proposals there may still have unmapped forms here.
"""

import json
import sys

with open(
    "packages/company_cleanse/src/company_cleanse/resources/company_type_rules.json",
    encoding="utf-8",
) as f:
    rules = json.load(f)
with open(
    "packages/company_cleanse/src/company_cleanse/resources/entity_legal_forms_iso20275.json",
    encoding="utf-8",
) as f:
    iso_data = json.load(f)

rule_countries = sorted(
    {r.get("country", "").upper() for r in rules if r.get("country")}
)
iso_by_country: dict[str, list[dict]] = {}
for entry in iso_data.values():
    cc = entry.get("country_code", "").upper()
    if cc:
        iso_by_country.setdefault(cc, []).append(entry)

overlap = sorted([c for c in rule_countries if c in iso_by_country])

existing_sources: dict[str, set[str]] = {}
for r in rules:
    cc = r.get("country", "").upper()
    existing_sources.setdefault(cc, set()).add(r.get("source", "").strip().lower())


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


target_cc = sys.argv[1].upper() if len(sys.argv) > 1 else None

if not target_cc:
    # Print tier summary only
    print(
        f"{'CC':4} {'new':>4} {'non-ascii':>9} {'multi-abbr':>10} {'no-abbr':>7} {'tier':>4}"
    )
    print("-" * 45)
    for cc in overlap:
        forms = iso_by_country[cc]
        existing = existing_sources.get(cc, set())
        new_forms = [
            form
            for form in forms
            if form.get("entity_legal_form_name", "").strip().lower() not in existing
            and form.get("status", "") == "ACTV"
        ]
        non_ascii = sum(
            1
            for form in new_forms
            if not is_ascii(form.get("entity_legal_form_name", ""))
        )
        multi_abbr = sum(
            1 for form in new_forms if ";" in (form.get("abbreviations_local") or "")
        )
        no_abbr = sum(
            1
            for form in new_forms
            if not (form.get("abbreviations_local") or "").strip()
            and not (form.get("abbreviations_transliterated") or "").strip()
        )
        tier = (
            1
            if non_ascii == 0 and multi_abbr == 0
            else (3 if non_ascii == len(new_forms) else 2)
        )
        print(
            f"{cc:4} {len(new_forms):>4} {non_ascii:>9} {multi_abbr:>10} {no_abbr:>7} {tier:>4}"
        )
else:
    # Detailed output for one country
    cc = target_cc
    forms = iso_by_country.get(cc, [])
    existing = existing_sources.get(cc, set())
    new_forms = [
        form
        for form in forms
        if form.get("entity_legal_form_name", "").strip().lower() not in existing
        and form.get("status", "") == "ACTV"
    ]
    new_forms.sort(key=lambda x: x.get("entity_legal_form_name", ""))

    print(f"\n{'=' * 70}")
    print(f"COUNTRY: {cc}  —  {len(new_forms)} new active forms to consider")
    print(f"{'=' * 70}")
    print(f"{'NAME (native)':45} {'ABBR_LOCAL':20} {'ABBR_TRANS':15}")
    print("-" * 80)
    for form in new_forms:
        name = form.get("entity_legal_form_name", "")
        abbr = (form.get("abbreviations_local") or "").strip()
        abbr_t = (form.get("abbreviations_transliterated") or "").strip()
        flags = []
        if not is_ascii(name):
            flags.append("NON-ASCII")
        if ";" in abbr:
            flags.append("MULTI-ABBR")
        if not abbr and not abbr_t:
            flags.append("NO-ABBR")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        print(f"  {name[:43]:45} {abbr[:18]:20} {abbr_t[:13]:15}{flag_str}")

    print(f"\nExisting rules for {cc}:")
    for r in sorted(
        [r for r in rules if r.get("country", "").upper() == cc],
        key=lambda x: x["source"],
    ):
        print(f"  {r['source']:45} -> {r['canonical']}")
