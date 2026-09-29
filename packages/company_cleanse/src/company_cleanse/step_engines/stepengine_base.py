from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from . import stepengine_noop_ops


class CleanseStepEngine(ABC):
    """Step-engine contract used by the cleansing pipeline."""

    @abstractmethod
    def extract_canonical_company_type(
        self,
        lf: pl.LazyFrame,
        company_type_mapping: dict[str, str],
    ) -> pl.LazyFrame:
        raise NotImplementedError

    @abstractmethod
    def generate_cleansed_company_name(
        self,
        lf: pl.LazyFrame,
        and_tokens: tuple[str, ...] | None,
        char_whitelist: str,
        normalization_operations: tuple[str, ...] = (),
    ) -> pl.LazyFrame:
        raise NotImplementedError

    @abstractmethod
    def derive_acronym_field(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError

    @abstractmethod
    def ensure_non_acronym_short_name(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError

    @abstractmethod
    def ensure_quoted_name_in_cleansed(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError


class DelegatingCleanseStepEngine(CleanseStepEngine):
    def __init__(self, ops) -> None:
        self._ops = ops

    def extract_canonical_company_type(
        self,
        lf: pl.LazyFrame,
        company_type_mapping: dict[str, str],
    ) -> pl.LazyFrame:
        return self._ops.extract_canonical_company_type(lf, company_type_mapping)

    def generate_cleansed_company_name(
        self,
        lf: pl.LazyFrame,
        and_tokens: tuple[str, ...] | None,
        char_whitelist: str,
        normalization_operations: tuple[str, ...] = (),
    ) -> pl.LazyFrame:
        return self._ops.generate_cleansed_company_name(
            lf,
            and_tokens,
            char_whitelist,
            normalization_operations,
        )

    def derive_acronym_field(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return self._ops.derive_acronym_field(lf)

    def ensure_non_acronym_short_name(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return self._ops.ensure_non_acronym_short_name(lf)

    def ensure_quoted_name_in_cleansed(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return self._ops.ensure_quoted_name_in_cleansed(lf)


class NoopCleanseStepEngine(DelegatingCleanseStepEngine):
    """No-op step engine used to isolate non-step-engine costs."""

    def __init__(self, ops=stepengine_noop_ops) -> None:
        super().__init__(ops)
