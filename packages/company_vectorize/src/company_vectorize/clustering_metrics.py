from __future__ import annotations

import polars as pl


def compute_directional_coverage(
    *,
    source_nodes: pl.DataFrame,
    target_nodes: pl.DataFrame,
    clusters: pl.DataFrame,
    node_id_col: str = "node_id",
    cluster_id_col: str = "cluster_id",
    system_col: str = "system",
) -> pl.DataFrame:
    for name, frame in (
        ("source_nodes", source_nodes),
        ("target_nodes", target_nodes),
        ("clusters", clusters),
    ):
        if frame is None:
            raise ValueError(f"{name} cannot be None")

    if (
        node_id_col not in source_nodes.columns
        or node_id_col not in target_nodes.columns
    ):
        raise ValueError(f"source_nodes and target_nodes must contain '{node_id_col}'.")
    if node_id_col not in clusters.columns or cluster_id_col not in clusters.columns:
        raise ValueError(
            f"clusters must contain '{node_id_col}' and '{cluster_id_col}'."
        )

    source_only = source_nodes.select(
        pl.col(node_id_col).cast(pl.Utf8).alias("node_id"),
    ).unique()
    target_only = target_nodes.select(
        pl.col(node_id_col).cast(pl.Utf8).alias("node_id"),
    ).unique()

    source_records = int(source_only.height)
    if source_records == 0:
        return pl.DataFrame(
            {
                "source_records": [0],
                "source_records_with_cluster": [0],
                "source_records_clustered_with_target": [0],
                "directional_coverage_ratio": [0.0],
            },
            schema={
                "source_records": pl.Int64,
                "source_records_with_cluster": pl.Int64,
                "source_records_clustered_with_target": pl.Int64,
                "directional_coverage_ratio": pl.Float64,
            },
        )

    clusters_norm = clusters.select(
        pl.col(cluster_id_col).cast(pl.Utf8).alias("cluster_id"),
        pl.col(node_id_col).cast(pl.Utf8).alias("node_id"),
    )
    source_joined = source_only.join(clusters_norm, on="node_id", how="left")

    source_records_with_cluster = int(
        source_joined.filter(pl.col("cluster_id").is_not_null()).height
    )
    if target_only.height == 0 or clusters_norm.height == 0:
        clustered_with_target = 0
    else:
        target_clusters = (
            target_only.join(clusters_norm, on="node_id", how="inner")
            .select("cluster_id")
            .unique()
        )
        clustered_with_target = int(
            source_joined.join(target_clusters, on="cluster_id", how="inner")
            .select("node_id")
            .unique()
            .height
        )

    ratio = (
        float(clustered_with_target) / float(source_records)
        if source_records > 0
        else 0.0
    )
    return pl.DataFrame(
        {
            "source_records": [source_records],
            "source_records_with_cluster": [source_records_with_cluster],
            "source_records_clustered_with_target": [clustered_with_target],
            "directional_coverage_ratio": [ratio],
        },
        schema={
            "source_records": pl.Int64,
            "source_records_with_cluster": pl.Int64,
            "source_records_clustered_with_target": pl.Int64,
            "directional_coverage_ratio": pl.Float64,
        },
    )


def compute_cluster_shape_metrics(
    *,
    clusters: pl.DataFrame,
    cluster_id_col: str = "cluster_id",
) -> pl.DataFrame:
    if cluster_id_col not in clusters.columns:
        raise ValueError(f"clusters must contain '{cluster_id_col}'.")

    if clusters.height == 0:
        return pl.DataFrame(
            {
                "total_clusters": [0],
                "singleton_clusters": [0],
                "singleton_ratio": [0.0],
                "mean_cluster_size": [0.0],
                "p95_cluster_size": [0.0],
                "max_cluster_size": [0],
            },
            schema={
                "total_clusters": pl.Int64,
                "singleton_clusters": pl.Int64,
                "singleton_ratio": pl.Float64,
                "mean_cluster_size": pl.Float64,
                "p95_cluster_size": pl.Float64,
                "max_cluster_size": pl.Int64,
            },
        )

    size_df = clusters.group_by(cluster_id_col).len().rename({"len": "cluster_size"})
    total_clusters = int(size_df.height)
    singleton_clusters = int(size_df.filter(pl.col("cluster_size") == 1).height)
    singleton_ratio = (
        float(singleton_clusters) / float(total_clusters) if total_clusters > 0 else 0.0
    )
    mean_cluster_size = (
        float(size_df.select(pl.col("cluster_size").mean()).item())
        if total_clusters > 0
        else 0.0
    )
    p95_value = (
        float(
            size_df.select(
                pl.col("cluster_size").quantile(0.95, interpolation="nearest")
            ).item()
        )
        if total_clusters > 0
        else 0.0
    )
    max_cluster_size = (
        int(size_df.select(pl.col("cluster_size").max()).item())
        if total_clusters > 0
        else 0
    )

    return pl.DataFrame(
        {
            "total_clusters": [total_clusters],
            "singleton_clusters": [singleton_clusters],
            "singleton_ratio": [singleton_ratio],
            "mean_cluster_size": [mean_cluster_size],
            "p95_cluster_size": [p95_value],
            "max_cluster_size": [max_cluster_size],
        },
        schema={
            "total_clusters": pl.Int64,
            "singleton_clusters": pl.Int64,
            "singleton_ratio": pl.Float64,
            "mean_cluster_size": pl.Float64,
            "p95_cluster_size": pl.Float64,
            "max_cluster_size": pl.Int64,
        },
    )
