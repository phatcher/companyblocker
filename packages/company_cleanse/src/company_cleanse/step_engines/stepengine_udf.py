from __future__ import annotations

from . import stepengine_udf_ops
from .stepengine_base import DelegatingCleanseStepEngine


class UdfCleanseStepEngine(DelegatingCleanseStepEngine):
    def __init__(self, ops=stepengine_udf_ops) -> None:
        super().__init__(ops)
