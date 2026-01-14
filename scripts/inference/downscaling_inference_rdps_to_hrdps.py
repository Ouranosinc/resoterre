"""Script for performing inference to downscale RDPS data to HRDPS resolution using a pre-trained model."""

import argparse

from resoterre.experiments.rdps_to_hrdps_workflow import inference_from_preprocessed_data


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
    return parser.parse_args()


def main() -> None:
    """Main function to execute the inference process."""
    args = parse_args()
    inference_from_preprocessed_data(config=args.config)


if __name__ == "__main__":
    main()
