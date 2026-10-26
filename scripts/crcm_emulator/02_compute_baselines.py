"""
Compute baselines for the CRCM emulator from a YAML configuration.

Run::

    python scripts/crcm_emulator/02_compute_baselines.py \
        --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``

Notes
-----
A snakemake parallel version can be found in `scripts/meta_workflows/crcm_emulator/`
"""

import argparse
from pathlib import Path

from resoterre.pipelines.crcm_emulator.cmip6_to_zarr_workflow import get_chunk_indices
from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def main(config_path: Path) -> None:
    """
    Compute baselines for the CRCM emulator from preprocessed GCM and CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    """
    config = crcm_emulator_parse_config(config_path)
    baseline = InterpolationBaseline(config, initialize_zarr=True)
    for simulation in config.preprocessing.simulations:
        print(f"Processing simulation: {simulation}")
        if config.preprocessing.gcm.start_datetime is None:
            raise ValueError("GCM start_datetime must be specified in the configuration.")
        if config.preprocessing.gcm.end_datetime is None:
            raise ValueError("GCM end_datetime must be specified in the configuration.")
        chunk_indices = get_chunk_indices(
            gcm=simulation[0],
            start_datetime=config.preprocessing.gcm.start_datetime,
            end_datetime=config.preprocessing.gcm.end_datetime,
            chunk_size=config.preprocessing.chunk_size,
            chunks_per_task=config.preprocessing.chunks_per_task,
        )
        for i, chunk_index_tuple in enumerate(chunk_indices):
            chunk_idx_start, chunk_idx_end = chunk_index_tuple
            print(f"Processing chunk {i + 1}/{len(chunk_indices)}: start={chunk_idx_start}, end={chunk_idx_end}")
            baseline.run(gcm_simulation=simulation, chunk_idx_start=chunk_idx_start, chunk_idx_end=chunk_idx_end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    args = parser.parse_args()
    main(args.config)
