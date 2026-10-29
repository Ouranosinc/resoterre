"""
Compute baselines for the CRCM emulator from a YAML configuration.

Run::

    python scripts/crcm_emulator/02_compute_baselines.py \
        --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``

Simulations whose baseline is already computed for the validation and test periods are skipped (use ``--force`` to
recompute). The baseline is then evaluated on ``data.test_periods`` with ``inference.test_metrics`` and the results are
written to ``<paths.baselines_out_dirs[interpolation]>/interpolation_baseline_test_results.json``; the validation
periods are scored the same way and saved to ``interpolation_baseline_validation_results.json``.

Notes
-----
A snakemake parallel version can be found in `scripts/meta_workflows/crcm_emulator/`
"""

import argparse
import json
from pathlib import Path

from resoterre.pipelines.crcm_emulator.cmip6_to_zarr_workflow import get_chunk_indices
from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def main(config_path: Path, force: bool = False) -> None:
    """
    Compute baselines for the CRCM emulator from preprocessed GCM and CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    force : bool
        If True, recompute the baseline even if it was previously computed.
    """
    config = crcm_emulator_parse_config(config_path)
    baseline = InterpolationBaseline(config, initialize_zarr=False)
    if force:
        simulations = config.preprocessing.simulations
    else:
        simulations = [x for x in config.preprocessing.simulations if not baseline.is_computed(x)]
        for simulation in config.preprocessing.simulations:
            if simulation not in simulations:
                print(f"Skipping simulation (baseline already computed): {simulation}")
    baseline.initialize_zarr(simulations)
    for simulation in simulations:
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

    for split in ("validation", "test"):
        if not getattr(config.data, f"{split}_periods"):
            print(f"data.{split}_periods is empty, skipping {split} evaluation.")
            continue
        results = baseline.evaluate(config.inference.test_metrics, split=split)
        print(f"Baseline {split} metrics:")
        for name, value in results.items():
            print(f"{name}: {value}")
        output_path = Path(
            config.paths.baselines_out_dirs["interpolation"], f"interpolation_baseline_{split}_results.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"metrics": results}, indent=2))
        print(f"Saved baseline {split} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    parser.add_argument("--force", action="store_true", help="Recompute baselines even if already computed")
    args = parser.parse_args()
    main(args.config, force=args.force)
