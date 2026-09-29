from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts import measure_gleif_jurisdiction_disparity as module
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def test_measure_jurisdiction_disparity_reads_family_split_primary_directory(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    """Real cleansed/ output lives under a `primary/` family directory, not
    directly at the layer's own top level -- the old hand-built
    "jurisdiction_code=*/*.parquet" glob at cleansed_dir's own top level
    could not see it.
    """
    partition_dir = (
        system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=gb"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "jurisdiction_code": ["gb"],
            "name_cleansed": ["acme"],
        }
    ).write_parquet(partition_dir / "part-00001.parquet")

    mocker.patch.object(
        module,
        "load_tokenizer_encoder",
        return_value=(lambda name: name.split(), "[UNK]"),
    )

    buckets = module.measure_jurisdiction_disparity(
        roots=workspace_roots,
        system="gleif",
        tokenizer_path=tmp_path / "tokenizer.json",
        trainer="wordpiece",
        name_col="name_cleansed",
    )

    assert set(buckets) == {"gb"}
    assert buckets["gb"].rows == 1
