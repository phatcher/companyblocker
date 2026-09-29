from __future__ import annotations

import polars as pl


def build_connected_components(
    edges: pl.DataFrame,
    *,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
) -> pl.DataFrame:
    if source_id_col not in edges.columns or target_id_col not in edges.columns:
        raise ValueError(
            f"edges must contain columns '{source_id_col}' and '{target_id_col}'."
        )

    if edges.height == 0:
        return pl.DataFrame(
            {
                "cluster_id": [],
                "node_id": [],
            },
            schema={
                "cluster_id": pl.Utf8,
                "node_id": pl.Utf8,
            },
        )

    parent: dict[str, str] = {}

    def _find(node: str) -> str:
        if node not in parent:
            parent[node] = node
            return node
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def _union(left: str, right: str) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for row in edges.select(
        pl.col(source_id_col).cast(pl.Utf8), pl.col(target_id_col).cast(pl.Utf8)
    ).iter_rows():
        source_id, target_id = row
        if source_id is None or target_id is None:
            continue
        _union(str(source_id), str(target_id))

    rows: list[dict[str, str]] = []
    for node in sorted(parent.keys()):
        root = _find(node)
        rows.append({"cluster_id": f"cc:{root}", "node_id": node})

    return pl.DataFrame(rows, schema={"cluster_id": pl.Utf8, "node_id": pl.Utf8})
