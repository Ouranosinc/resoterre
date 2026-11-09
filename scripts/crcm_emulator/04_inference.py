"""
Evaluate a trained CRCM emulator checkpoint on the validation data, and optionally on the test data.

Run ``python scripts/crcm_emulator/04_inference.py --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``

The checkpoint is evaluated on ``data.validation_periods`` with ``inference.evaluation_metrics``. With ``--test``, it is
also evaluated on ``data.test_periods`` with the same metrics. The results are saved to
``<experiment_name>_validation_results.json`` and ``<experiment_name>_test_results.json`` in ``paths.output``.
"""

import argparse
import json
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch.utils.data as td

from resoterre.pipelines.crcm_emulator.crcm_emulator_training import CRCMEmulatorTrainingFromConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.metrics import build_metrics


def evaluate_and_save(
    runner: CRCMEmulatorTrainingFromConfig,
    split: str,
    data_loaders: dict[str, td.DataLoader[Any]],
    metric_names: list[str],
    output_directory: Path,
) -> None:
    """
    Evaluate the loaded checkpoint on the data loaders of a split, print the metrics and save them as JSON.

    Parameters
    ----------
    runner : CRCMEmulatorTrainingFromConfig
        Trainer with the checkpoint loaded.
    split : str
        Either "test" or "validation", used in the output file name.
    data_loaders : dict[str, td.DataLoader]
        Data loaders by key of the periods of the split.
    metric_names : list[str]
        Names of the extra metrics to compute, in addition to the losses.
    output_directory : Path
        Directory of the output file ``<experiment_name>_<split>_results.json``.
    """
    results = runner.evaluate_by_key(data_loaders, build_metrics(metric_names), desc=split.capitalize())
    single_valued_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}

    output_path = Path(output_directory, f"{runner.config.experiment_name}_{split}_results.json")
    with output_path.open("w") as f:
        json.dump(
            {
                "epoch": runner.epoch_counter,
                "nb_of_samples": {key: len(cast(Sized, loader.dataset)) for key, loader in data_loaders.items()},
                "metrics": single_valued_results,
            },
            f,
            indent=4,
        )
    print(json.dumps(single_valued_results, indent=4))
    print(f"{split.capitalize()} results saved to {output_path}")


def main(config_path: Path, checkpoint: str = "latest", test: bool = False) -> None:
    """
    Evaluate a trained CRCM emulator on the validation periods, and optionally on the test periods.

    Parameters
    ----------
    config_path : Path
        Path to the CRCM emulator YAML configuration used for training.
    checkpoint : str
        Checkpoint to evaluate, "latest" or "best".
    test : bool
        Also evaluate the checkpoint on the test periods.
    """
    config = crcm_emulator_parse_config(config_path)
    output_directory = config.paths.output
    if output_directory is None:
        raise ValueError("Set paths.output to the directory containing the trained checkpoints.")
    if config.preprocessing.gcm.path is None or config.preprocessing.crcm.path is None:
        raise ValueError("Set preprocessing.gcm.path and preprocessing.crcm.path in the configuration.")
    # Disable Comet ML logging, nothing is being trained.
    config = config.model_copy(update={"logger": None})

    runner = CRCMEmulatorTrainingFromConfig(config)
    if not runner.validation_data_loaders:
        raise ValueError("No validation samples match data.validation_periods and the selected variables.")
    test_data_loaders = runner.build_test_data_loaders() if test else {}

    runner.load_state(output_directory, tag=checkpoint)
    runner.to_device(config.training.device)
    evaluate_and_save(
        runner, "validation", runner.validation_data_loaders, config.inference.evaluation_metrics, output_directory
    )
    if test:
        evaluate_and_save(runner, "test", test_data_loaders, config.inference.evaluation_metrics, output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    parser.add_argument("--checkpoint", choices=["latest", "best"], default="best", help="Checkpoint to evaluate")
    parser.add_argument("--test", action="store_true", help="Also evaluate the checkpoint on the test periods")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.test)
