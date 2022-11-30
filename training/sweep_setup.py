import argparse
from pathlib import Path

import wandb
import yaml

wb_api = wandb.Api()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--project",
        help="The project to log the sweep results to.",
    )
    parser.add_argument(
        "--config",
        help="The configuration path to set up the sweep with.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    sweep_id = wandb.sweep(sweep=config, project=args.project)


if __name__ == "__main__":
    main()
