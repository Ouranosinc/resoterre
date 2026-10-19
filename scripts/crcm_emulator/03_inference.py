"""
Evaluate a trained CRCM emulator checkpoint on the test data.

Run ``python scripts/crcm_emulator/03_inference.py --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils import data as td

from resoterre.data_management.timeseries import MultiTimeseries
from resoterre.hybrid_data_loaders.crcm_emulator_data_loader import CRCMEmulatorDataset
from resoterre.pipelines.crcm_emulator.crcm_emulator_training import CRCMEmulatorTrainingFromConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config


def main(config_path: Path, checkpoint: str = "latest") -> None:
    """
    Evaluate a trained CRCM emulator on the test periods of preprocessed GCM and CRCM Zarr data.

    Parameters
    ----------
    config_path : Path
        Path to the CRCM emulator YAML configuration used for training.
    checkpoint : str
        Checkpoint to evaluate, "latest" or "best".
    """
    config = crcm_emulator_parse_config(config_path)
    if config.paths.output is None:
        raise ValueError("Set paths.output to the directory containing the trained checkpoints.")
    if config.preprocessing.gcm.path is None or config.preprocessing.crcm.path is None:
        raise ValueError("Set preprocessing.gcm.path and preprocessing.crcm.path in the configuration.")
    if not config.data.test_periods:
        raise ValueError("Set data.test_periods in the configuration.")
    # Disable Comet ML logging, nothing is being trained.
    config = config.model_copy(update={"logger": None})
    output_directory = config.paths.output

    runner = CRCMEmulatorTrainingFromConfig(config)
    test_dataset = CRCMEmulatorDataset(
        path_gcm_preprocessing=config.preprocessing.gcm.path,
        path_crcm_preprocessing=config.preprocessing.crcm.path,
        simulations=config.preprocessing.simulations,
        gcm_variables=config.data.gcm_variables,
        crcm_variables=config.data.crcm_variables,
        time_periods=config.data.test_periods,
        max_open_datasets=config.data.max_open_datasets,
    )
    if len(test_dataset) == 0:
        raise ValueError("No test samples match test_periods and the selected variables.")
    test_data_loader = td.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
        multiprocessing_context="spawn",
        pin_memory=True,
    )

    runner.load_state(output_directory, tag=checkpoint)
    runner.to_device(config.training.device)
    runner.set_eval_mode()
    runner.validation_metrics = MultiTimeseries()
    with torch.no_grad():
        for item in test_data_loader:
            runner.validation_step(item)
    runner.flush_validation_metrics()

    results = {name: float(np.mean(series.values)) for name, series in runner.validation_metrics.items()}
    output_path = Path(output_directory, f"{config.experiment_name}_test_results.json")
    with output_path.open("w") as f:
        json.dump({"epoch": runner.epoch_counter, "nb_of_samples": len(test_dataset), "metrics": results}, f, indent=4)
    print(json.dumps(results, indent=4))
    print(f"Test results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="CRCM emulator YAML configuration")
    parser.add_argument("--checkpoint", choices=["latest", "best"], default="best", help="Checkpoint to evaluate")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
