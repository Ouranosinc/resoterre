"""
Evaluate the CRCM emulator baselines from a YAML configuration.

Run::

    python scripts/crcm_emulator/02_score_baselines.py \
        --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``

The baseline zarr datasets must already be computed, e.g. with the snakemake workflow
``scripts/meta_workflows/crcm_emulator/rules/03_baseline.smk``. The baseline is evaluated on ``data.test_periods`` with
``inference.test_metrics`` and the results are written to
``<paths.baselines_out_dirs[interpolation]>/interpolation_baseline_test_results.json``; the validation periods are
scored the same way and saved to ``interpolation_baseline_validation_results.json``. If found, training reuses
these files (``training.log_interpolation_baseline``) instead of scoring the baseline again.
"""

import argparse
from pathlib import Path

from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def main(config_path: Path) -> None:
    """
    Evaluate the precomputed baselines for the CRCM emulator against the preprocessed CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    """
    config = crcm_emulator_parse_config(config_path)
    baseline = InterpolationBaseline(config)
    for split in ("validation", "test"):
        if not getattr(config.data, f"{split}_periods"):
            print(f"data.{split}_periods is empty, skipping {split} evaluation.")
            continue
        results = baseline.evaluate(config.inference.test_metrics, split=split)
        print(f"Baseline {split} metrics:")
        for name, value in results.items():
            print(f"{name}: {value}")
        output_path = baseline.save_results(results, split)
        print(f"Saved baseline {split} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    args = parser.parse_args()
    main(args.config)
