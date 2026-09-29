import tempfile
import unittest
from pathlib import Path

import polars as pl

from acquisition import private_analysis
from acquisition.cleanser_ops import analyze_private_last_tokens_by_country
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import default_workspace_roots


class PrivateAnalysisTests(unittest.TestCase):
    def test_resolve_artifact_paths_defaults_and_override(self):
        default_base, default_parquet, default_txt, default_md = (
            private_analysis._resolve_artifact_paths(None)
        )
        self.assertEqual(default_base, private_analysis.DEFAULT_OUTPUT_DIR)
        self.assertEqual(default_parquet.name, "private_country_analysis.parquet")
        self.assertEqual(default_txt.name, "private_country_analysis.txt")
        self.assertEqual(default_md.name, "private_country_analysis.md")

        custom = Path("custom/output")
        custom_base, *_ = private_analysis._resolve_artifact_paths(custom)
        self.assertEqual(custom_base, custom)

    def test_helper_normalization_utilities(self):
        self.assertEqual(
            private_analysis._canonicalize_token("Société Anónima"), "SOCIETE_ANONIMA"
        )
        self.assertEqual(private_analysis._canonicalize_token("!!!"), "UNKNOWN")
        self.assertEqual(
            private_analysis._normalize_for_match("  Public   Limited Company  "),
            "public limited company",
        )
        self.assertEqual(
            private_analysis._split_rule_tokens("Public Limited Company"),
            {"public", "limited", "company"},
        )
        self.assertEqual(
            private_analysis._normalize_upper("  alpha   beta "), "ALPHA BETA"
        )
        self.assertEqual(private_analysis._token_alnum_upper("S.R.L."), "SRL")
        self.assertEqual(
            private_analysis._initialism(["with", "limited", "liability"]), "WLL"
        )

    def test_build_existing_rule_signatures(self):
        exact, tokens = private_analysis._build_existing_rule_signatures(
            [
                ("Public Limited Company", "PLC", "gb"),
                ("Limited Liability Partnership", "LLP", "gb"),
            ]
        )
        self.assertIn("public limited company", exact)
        self.assertIn("PLC", exact)
        self.assertIn("public", tokens)
        self.assertIn("llp", tokens)

    def test_commentary_notes_and_noise_detection(self):
        duplicate_notes = private_analysis._commentary_notes("GROUP", 25, True)
        self.assertTrue(any("already covered" in note for note in duplicate_notes))
        self.assertTrue(any("High-frequency" in note for note in duplicate_notes))
        self.assertTrue(
            any("Generic business term" in note for note in duplicate_notes)
        )

        new_notes = private_analysis._commentary_notes("AB", 5, False)
        self.assertTrue(
            any("Candidate missing legal-form token" in note for note in new_notes)
        )
        self.assertTrue(any("Very short token" in note for note in new_notes))
        self.assertTrue(any("Medium-frequency" in note for note in new_notes))

        self.assertTrue(private_analysis._is_likely_noise("GB", "UK"))
        self.assertTrue(private_analysis._is_likely_noise("Belgium", "Belgium"))
        self.assertTrue(private_analysis._is_likely_noise("GB", "AB"))
        self.assertFalse(private_analysis._is_likely_noise("GB", "Partnership"))

    def test_build_dynamic_long_form_evidence_short_circuits(self):
        empty_candidates = pl.DataFrame(
            {"country": [], "last_token": [], "quantity": []}
        )
        self.assertEqual(
            private_analysis._build_dynamic_long_form_evidence(
                empty_candidates, Path("missing.parquet")
            ),
            {},
        )

    def test_build_dynamic_long_form_evidence_missing_columns_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            records_file = Path(tmp_dir) / "private_records.parquet"
            pl.DataFrame({"other": ["x"]}).write_parquet(records_file)
            candidates = pl.DataFrame(
                {"country": ["GB"], "last_token": ["WLL"], "quantity": [10]}
            )

            result = private_analysis._build_dynamic_long_form_evidence(
                candidates, records_file
            )
            self.assertEqual(result, {})

    def test_build_dynamic_long_form_evidence_infers_phrase(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            records_file = Path(tmp_dir) / "private_records.parquet"
            pl.DataFrame(
                {
                    "CountryOfOrigin": ["GB", "GB", "GB"],
                    "company_type": ["PRIVATE", "PRIVATE", "PRIVATE"],
                    "name_cleansed": [
                        "with limited liability wll",
                        "with limited liability wll",
                        "wrong phrase wll",
                    ],
                }
            ).write_parquet(records_file)
            candidates = pl.DataFrame(
                {"country": ["GB"], "last_token": ["WLL"], "quantity": [10]}
            )

            result = private_analysis._build_dynamic_long_form_evidence(
                candidates, records_file
            )
            self.assertEqual(result[("GB", "WLL")], ("WITH LIMITED LIABILITY", 2, 3))

    def test_derive_mapping_with_and_without_evidence(self):
        evidence = {("GB", "WLL"): ("WITH LIMITED LIABILITY", 2, 3)}
        self.assertEqual(
            private_analysis._derive_mapping("GB", "WLL", evidence),
            (
                "With Limited Liability",
                "Inferred from private records: 'WITH LIMITED LIABILITY' + WLL (2/3 supporting names).",
            ),
        )
        self.assertEqual(
            private_analysis._derive_mapping("GB", "PLC", {}), ("PLC", None)
        )

    def test_private_last_token_counts_by_country(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleansed_dir = root / "cleansed"
            output_path = root / "analysis" / "private_last_tokens.parquet"
            cleansed_dir.mkdir(parents=True, exist_ok=True)

            df_1 = pl.DataFrame(
                {
                    "CountryOfOrigin": ["Belgium", "Belgium", "Andorra", "Andorra"],
                    "company_type": ["PRIVATE", "PRIVATE", "PRIVATE", "LTD"],
                    "name_cleansed": [
                        "BEKINA BOOTS NV",
                        "ACME NV",
                        "ALFA SL",
                        "OMEGA LTD",
                    ],
                }
            )
            df_2 = pl.DataFrame(
                {
                    "CountryOfOrigin": ["Belgium", "Belgium", "Andorra", None],
                    "company_type": ["PRIVATE", "LTD", "PRIVATE", "PRIVATE"],
                    "name_cleansed": [
                        "BETA NV",
                        "PUBLIC LTD",
                        "BETA SL",
                        "NO COUNTRY TRUST",
                    ],
                }
            )

            df_1.write_parquet(cleansed_dir / "part-1.parquet")
            df_2.write_parquet(cleansed_dir / "part-2.parquet")

            result = analyze_private_last_tokens_by_country(
                cleansed_dir=cleansed_dir,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            actual = {
                (row["country"], row["last_token"]): row["quantity"]
                for row in result.to_dicts()
            }
            expected = {
                ("Andorra", "sl"): 2,
                ("Belgium", "nv"): 3,
                ("UNKNOWN", "trust"): 1,
            }
            self.assertEqual(actual, expected)

    def test_top_n_per_country_filter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleansed_dir = root / "cleansed"
            cleansed_dir.mkdir(parents=True, exist_ok=True)

            df = pl.DataFrame(
                {
                    "CountryOfOrigin": ["Belgium"] * 5,
                    "company_type": ["PRIVATE"] * 5,
                    "name_cleansed": [
                        "A NV",
                        "B NV",
                        "C NV",
                        "D BV",
                        "E BV",
                    ],
                }
            )
            df.write_parquet(cleansed_dir / "part.parquet")

            result = analyze_private_last_tokens_by_country(
                cleansed_dir=cleansed_dir,
                top_n_per_country=1,
            )

            rows = result.to_dicts()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["country"], "Belgium")
            self.assertEqual(rows[0]["last_token"], "nv")
            self.assertEqual(rows[0]["quantity"], 3)

    def test_private_analysis_uses_supplied_cleansed_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleansed_dir = system_layer_dir(
                default_workspace_roots(root), "ie", layer=CLEANSED_LAYER_NAME
            )
            cleansed_dir.mkdir(parents=True, exist_ok=True)

            analysis = pl.DataFrame(
                {
                    "country": ["IRELAND", "IRELAND"],
                    "last_token": ["ltd", "limited"],
                    "quantity": [8, 3],
                }
            )
            analysis.write_parquet(cleansed_dir / "private_country_analysis.parquet")

            private_records = pl.DataFrame(
                {
                    "CountryOfOrigin": ["Ireland"],
                    "company_type": ["PRIVATE"],
                    "name_cleansed": ["EXAMPLE LIMITED"],
                }
            )
            private_records.write_parquet(cleansed_dir / "private_records.parquet")

            high_freq = private_analysis.analyze_and_report(
                min_frequency=5, cleansed_dir=cleansed_dir
            )
            template_path = private_analysis.generate_proposed_rule_additions(
                min_frequency=5,
                cleansed_dir=cleansed_dir,
            )

            self.assertEqual(template_path.parent, cleansed_dir)
            self.assertTrue((cleansed_dir / "private_country_analysis.txt").exists())
            self.assertTrue((cleansed_dir / "private_country_analysis.md").exists())
            self.assertEqual(high_freq.height, 1)
            self.assertEqual(high_freq["last_token"].to_list(), ["ltd"])

    def test_proposed_rules_flag_tokens_already_covered_by_existing_rules(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleansed_dir = system_layer_dir(
                default_workspace_roots(root), "gb", layer=CLEANSED_LAYER_NAME
            )
            cleansed_dir.mkdir(parents=True, exist_ok=True)

            analysis = pl.DataFrame(
                {
                    "country": ["GB", "GB"],
                    "last_token": ["Grouping", "Partnership"],
                    "quantity": [74, 12],
                }
            )
            analysis.write_parquet(cleansed_dir / "private_country_analysis.parquet")

            template_path = private_analysis.generate_proposed_rule_additions(
                min_frequency=5,
                high_priority_threshold=20,
                cleansed_dir=cleansed_dir,
            )

            proposal_text = template_path.read_text(encoding="utf-8")
            self.assertIn("DUPLICATE?", proposal_text)
            self.assertNotIn("- **Status**: NEW", proposal_text)

    def test_proposed_rules_infer_only_when_uncovered(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleansed_dir = system_layer_dir(
                default_workspace_roots(root), "gb", layer=CLEANSED_LAYER_NAME
            )
            cleansed_dir.mkdir(parents=True, exist_ok=True)

            analysis = pl.DataFrame(
                {
                    "country": ["GB"],
                    "last_token": ["wll"],
                    "quantity": [38],
                }
            )
            analysis.write_parquet(cleansed_dir / "private_country_analysis.parquet")

            private_records = pl.DataFrame(
                {
                    "CountryOfOrigin": ["GB", "GB"],
                    "company_type": ["PRIVATE", "PRIVATE"],
                    "name_cleansed": [
                        "acme with limited liability",
                        "baker with limited liability",
                    ],
                }
            )
            private_records.write_parquet(cleansed_dir / "private_records.parquet")

            template_path = private_analysis.generate_proposed_rule_additions(
                min_frequency=5,
                high_priority_threshold=20,
                cleansed_dir=cleansed_dir,
            )

            proposal_text = template_path.read_text(encoding="utf-8")
            self.assertIn('"wll", "WLL"', proposal_text)
            self.assertIn("DUPLICATE?", proposal_text)
            self.assertNotIn("Inferred from private records", proposal_text)

    def test_analyze_and_report_raises_when_analysis_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cleansed_dir = system_layer_dir(
                default_workspace_roots(Path(tmp_dir)), "gb", layer=CLEANSED_LAYER_NAME
            )
            with self.assertRaises(FileNotFoundError):
                private_analysis.analyze_and_report(cleansed_dir=cleansed_dir)

    def test_generate_proposed_rule_additions_raises_when_analysis_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cleansed_dir = system_layer_dir(
                default_workspace_roots(Path(tmp_dir)), "gb", layer=CLEANSED_LAYER_NAME
            )
            with self.assertRaises(FileNotFoundError):
                private_analysis.generate_proposed_rule_additions(
                    cleansed_dir=cleansed_dir
                )

    def test_generate_proposed_rule_additions_handles_empty_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cleansed_dir = system_layer_dir(
                default_workspace_roots(Path(tmp_dir)), "gb", layer=CLEANSED_LAYER_NAME
            )
            cleansed_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "country": ["GB", "GB"],
                    "last_token": ["UK", "AB"],
                    "quantity": [10, 7],
                }
            ).write_parquet(cleansed_dir / "private_country_analysis.parquet")

            template_path = private_analysis.generate_proposed_rule_additions(
                min_frequency=5,
                high_priority_threshold=20,
                cleansed_dir=cleansed_dir,
            )

            proposal_text = template_path.read_text(encoding="utf-8")
            self.assertIn("No candidates.", proposal_text)

    def test_generate_resolved_rule_file_and_resolve_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleansed_dir = system_layer_dir(
                default_workspace_roots(root), "gb", layer=CLEANSED_LAYER_NAME
            )
            cleansed_dir.mkdir(parents=True, exist_ok=True)
            template_path = cleansed_dir / "proposed_rule_template.md"
            template_path.write_text("template body", encoding="utf-8")

            resolved_path = private_analysis.generate_resolved_rule_file(
                cleansed_dir=cleansed_dir
            )
            self.assertTrue(resolved_path.exists())
            self.assertEqual(resolved_path.read_text(encoding="utf-8"), "template body")

            wrapper_path = private_analysis.resolve_proposed_rule_template(
                cleansed_dir=cleansed_dir
            )
            self.assertEqual(wrapper_path.read_text(encoding="utf-8"), "template body")

    def test_generate_resolved_rule_file_raises_when_template_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cleansed_dir = system_layer_dir(
                default_workspace_roots(Path(tmp_dir)), "gb", layer=CLEANSED_LAYER_NAME
            )
            with self.assertRaises(FileNotFoundError):
                private_analysis.generate_resolved_rule_file(cleansed_dir=cleansed_dir)


if __name__ == "__main__":
    unittest.main()
