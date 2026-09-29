from __future__ import annotations

from . import stepengine_polars_ops
from .stepengine_base import DelegatingCleanseStepEngine


class PolarsCleanseStepEngine(DelegatingCleanseStepEngine):
    def __init__(self, ops=stepengine_polars_ops) -> None:
        super().__init__(ops)
