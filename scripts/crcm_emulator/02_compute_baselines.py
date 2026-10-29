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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from resoterre.pipelines.crcm_emulator.cmip6_to_zarr_workflow import get_chunk_indices
from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def run_chunk(config_path: Path, simulation: list[str], chunk_idx_start: int, chunk_idx_end: int) -> None:
    """
    Compute the baseline of one chunk of one simulation (worker entry point).

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    simulation : list[str]
        List containing the GCM name, emission scenario, and ensemble member.
    chunk_idx_start : int
        Start index of the chunk.
    chunk_idx_end : int
        End index of the chunk.
    """
    InterpolationBaseline(crcm_emulator_parse_config(config_path)).run(
        gcm_simulation=simulation, chunk_idx_start=chunk_idx_start, chunk_idx_end=chunk_idx_end
    )


def main(config_path: Path, force: bool = False, workers: int = 1) -> None:
    """
    Compute baselines for the CRCM emulator from preprocessed GCM and CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    force : bool
        If True, recompute the baseline even if it was previously computed.
    workers : int
        Number of processes computing chunks in parallel. Chunks write disjoint regions of the zarr, so this mirrors
        the snakemake workflow, where the zarr is initialized once and every chunk is an independent job.
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
    if config.preprocessing.gcm.start_datetime is None:
        raise ValueError("GCM start_datetime must be specified in the configuration.")
    if config.preprocessing.gcm.end_datetime is None:
        raise ValueError("GCM end_datetime must be specified in the configuration.")
    tasks: list[tuple[list[str], int, int]] = []
    for simulation in simulations:
        chunk_indices = get_chunk_indices(
            gcm=simulation[0],
            start_datetime=config.preprocessing.gcm.start_datetime,
            end_datetime=config.preprocessing.gcm.end_datetime,
            chunk_size=config.preprocessing.chunk_size,
            chunks_per_task=config.preprocessing.chunks_per_task,
        )
        tasks.extend((simulation, chunk_idx_start, chunk_idx_end) for chunk_idx_start, chunk_idx_end in chunk_indices)
    print(f"Computing {len(tasks)} chunks with {workers} worker(s)")
    if workers <= 1:
        for i, task in enumerate(tasks):
            print(f"Processing chunk {i + 1}/{len(tasks)}: {task}")
            run_chunk(config_path, *task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(run_chunk, config_path, *task) for task in tasks]
            for i, future in enumerate(futures):
                future.result()  # re-raises worker errors
                print(f"Finished chunk {i + 1}/{len(tasks)}: {tasks[i]}")

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
    parser.add_argument("--workers", type=int, default=1, help="Number of processes computing chunks in parallel")
    args = parser.parse_args()
    main(args.config, force=args.force, workers=args.workers)
