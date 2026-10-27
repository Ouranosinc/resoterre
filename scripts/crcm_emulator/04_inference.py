"""
Evaluate a trained CRCM emulator checkpoint on the test data.

Run ``python scripts/crcm_emulator/04_inference.py --config configs/crcm_emulator/crcm_emulator_cnrm_ssp245.yaml``
"""

import argparse
import json
from pathlib import Path

from resoterre.pipelines.crcm_emulator.crcm_emulator_training import CRCMEmulatorTrainingFromConfig
from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import crcm_emulator_parse_config
from resoterre.pipelines.crcm_emulator.metrics import build_metrics


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
    # Disable Comet ML logging, nothing is being trained.
    config = config.model_copy(update={"logger": None})
    output_directory = config.paths.output

    runner = CRCMEmulatorTrainingFromConfig(config)
    test_dataset, test_data_loader = runner.build_test_data_loader()

    runner.load_state(output_directory, tag=checkpoint)
    runner.to_device(config.training.device)
    results = runner.evaluate(test_data_loader, build_metrics(config.inference.test_metrics), desc="Test")

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
