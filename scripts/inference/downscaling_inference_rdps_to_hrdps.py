"""Script for performing inference to downscale RDPS data to HRDPS resolution using a pre-trained model."""

import argparse

from resoterre.experiments.rdps_to_hrdps_workflow import inference_from_preprocessed_data
from resoterre.io_utils import override_config_paths


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference script to downscale RDPS data to HRDPS resolution using a pre-trained model."
    )
    parser.add_argument("config", type=str, help="Path to the configuration file for inference.")
    parser.add_argument(
        "--preprocess_batch",
        type=str,
        default=None,
        help="Path to the preprocessed batch file (overrides path_preprocessed_batch in config).",
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default=None,
        help="Path to the directory containing the pre-trained model (overrides path_models in config).",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Path to the directory where inference results will be saved (overrides path_output in config).",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to execute the inference process."""
    args = parse_args()
    overrides = {
        "path_preprocessed_batch": args.preprocess_batch,
        "path_models": args.path_models,
        "path_output": args.path_output,
    }
    config = override_config_paths(args.config, overrides)
    inference_from_preprocessed_data(config=config)


if __name__ == "__main__":
    main()
