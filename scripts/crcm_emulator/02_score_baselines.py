"""
Evaluate the CRCM emulator baselines from a YAML configuration.

Run::

    python scripts/crcm_emulator/02_score_baselines.py \
        --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml

The baseline zarr datasets must already be computed, e.g. with the snakemake workflow
``scripts/meta_workflows/crcm_emulator/rules/03_baseline.smk``. The baseline is evaluated on ``data.validation_periods``
with ``inference.evaluation_metrics``. With ``--test``, it is also evaluated on ``data.test_periods`` with the same
metrics, as for the trained model (``04_inference.py``). The results are written to
``<paths.baselines_out_dirs[interpolation]>/interpolation_baseline_<split>_results.json``. If found, training
reuses these files (``training.log_interpolation_baseline``) instead of scoring the baseline again.
"""

import argparse
from pathlib import Path

from resoterre.pipelines.crcm_emulator.crcm_emulator_baselines import InterpolationBaseline, single_valued_results
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def main(config_path: Path, test: bool = False) -> None:
    """
    Evaluate the precomputed baselines for the CRCM emulator against the preprocessed CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to a CRCM emulator YAML configuration.
    test : bool
        Also evaluate the baseline on the test periods.
    """
    config = crcm_emulator_parse_config(config_path)
    baseline = InterpolationBaseline(config)
    metric_names = config.inference.evaluation_metrics
    for split in ("validation", "test") if test else ("validation",):
        if not getattr(config.data, f"{split}_periods"):
            print(f"data.{split}_periods is empty, skipping {split} evaluation.")
            continue
        results = baseline.evaluate(metric_names, split=split)
        output_path = baseline.save_results(results, split)
        print(f"Baseline {split} metrics:")
        for name, value in single_valued_results(results).items():
            print(f"{name}: {value}")
        print(f"Saved baseline {split} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    parser.add_argument("--test", action="store_true", help="Also evaluate the baseline on the test periods")
    args = parser.parse_args()
    main(args.config, args.test)
