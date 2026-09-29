from __future__ import annotations

from collections.abc import Mapping

import polars as pl


class OffeneregisterShardTransformer:
    @staticmethod
    def _normalize_to_dict_list(value: object) -> list[dict[str, object]]:
        if isinstance(value, pl.Series):
            value = value.to_list()

        if value is None:
            return []
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            return [dict(value)]
        return []

    @classmethod
    def _extract_previous_names(cls, value: object) -> list[str] | None:
        items = cls._normalize_to_dict_list(value)
        names = [
            name.strip()
            for item in items
            for name in [item.get("company_name")]
            if isinstance(name, str) and name.strip()
        ]
        return names or None

    @classmethod
    def _extract_subsequent_registration_identifiers(
        cls, value: object
    ) -> list[str] | None:
        items = cls._normalize_to_dict_list(value)
        identifiers: list[str] = []

        for item in items:
            subsequent_entity = item.get("subsequent_entity")
            if not isinstance(subsequent_entity, dict):
                continue
            entity_properties = subsequent_entity.get("entity_properties")
            if not isinstance(entity_properties, dict):
                continue

            company_number = entity_properties.get("company_number")
            if not isinstance(company_number, str) or not company_number.strip():
                continue

            jurisdiction_code = entity_properties.get("jurisdiction_code")
            if isinstance(jurisdiction_code, str) and jurisdiction_code.strip():
                identifiers.append(f"{jurisdiction_code.lower()}:{company_number}")
            else:
                identifiers.append(company_number)

        return identifiers or None

    def flatten_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        transformed = frame

        if "previous_names" in transformed.columns:
            transformed = transformed.with_columns(
                pl.col("previous_names")
                .map_elements(
                    self._extract_previous_names, return_dtype=pl.List(pl.Utf8)
                )
                .alias("previous_names_list")
            )

        if "subsequent_registrations" in transformed.columns:
            transformed = transformed.with_columns(
                pl.col("subsequent_registrations")
                .map_elements(
                    self._extract_subsequent_registration_identifiers,
                    return_dtype=pl.List(pl.Utf8),
                )
                .alias("subsequent_registration_identifiers")
            )

        dtype = transformed.schema.get("all_attributes")
        if isinstance(dtype, pl.Struct):
            register_metadata = pl.col("all_attributes")

            top_level_fields = {field.name: field.dtype for field in dtype.fields}
            additional_data_dtype = top_level_fields.get("additional_data")
            additional_data_fields = (
                {field.name for field in additional_data_dtype.fields}
                if isinstance(additional_data_dtype, pl.Struct)
                else set()
            )

            def _string_field(name: str) -> pl.Expr:
                if name in top_level_fields:
                    return register_metadata.struct.field(name).cast(pl.Utf8)
                return pl.lit(None, dtype=pl.Utf8)

            def _bool_additional_data_field(name: str) -> pl.Expr:
                if (
                    "additional_data" in top_level_fields
                    and name in additional_data_fields
                ):
                    return (
                        register_metadata.struct.field("additional_data")
                        .struct.field(name)
                        .cast(pl.Boolean)
                    )
                return pl.lit(None, dtype=pl.Boolean)

            register_art_expr = _string_field("_registerArt")
            register_number_expr = _string_field("_registerNummer")
            transformed = transformed.with_columns(
                [
                    register_art_expr.alias("register_art"),
                    register_art_expr.alias("company_type"),
                    register_number_expr.alias("register_number"),
                    _string_field("federal_state").alias("federal_state"),
                    _string_field("former_registrar").alias("former_registrar"),
                    _string_field("native_company_number").alias(
                        "native_company_number"
                    ),
                    _string_field("registered_office").alias("registered_office"),
                    _string_field("registrar").alias("registrar"),
                    _bool_additional_data_field("AD").alias("register_flag_ad"),
                    _bool_additional_data_field("CD").alias("register_flag_cd"),
                    _bool_additional_data_field("DK").alias("register_flag_dk"),
                    _bool_additional_data_field("HD").alias("register_flag_hd"),
                    _bool_additional_data_field("SI").alias("register_flag_si"),
                    _bool_additional_data_field("UT").alias("register_flag_ut"),
                    _bool_additional_data_field("VÖ").alias("register_flag_vo"),
                ]
            )

        drop_columns = [
            column
            for column in [
                "all_attributes",
                "officers",
                "previous_names",
                "subsequent_registrations",
            ]
            if column in transformed.columns
        ]
        if drop_columns:
            transformed = transformed.drop(drop_columns)

        return transformed


_OFFENEREGISTER_TRANSFORMER = OffeneregisterShardTransformer()


def flatten_offeneregister_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return _OFFENEREGISTER_TRANSFORMER.flatten_frame(frame)
