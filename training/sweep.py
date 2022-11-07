import argparse
import os
from pathlib import Path

import torch
import wandb
import yaml

wb_api = wandb.Api()
DEFAULT_PROJECT = "admirer-training"
DEFAULT_SWEEP_CONFIG = "training/simple-overfit-sweep.yaml"
DEFAULT_GPU_LIST = [gpu for gpu in range(torch.cuda.device_count())]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="If specified, this is the project to log the sweep results to. Otherwise, they will be logged to {DEFAULT_PROJECT}.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_SWEEP_CONFIG,
        help="If specified, this is the configuration path to set up the sweep with. Otherwise, it will be set up with {DEFAULT_SWEEP_CONFIG}.",
    )
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPU_LIST,
        help="If specified, this is the list of gpus to use as specified by their system ids. Otherwise, all the system gpus will be used.",
    )
    parser.add_argument(
        "--count",
        default=None,
        help="If specified, this is the number of trials to run per W&B agent. Otherwise, unless optimizing with grid search, it will run indefinitely until terminated.",
    )
    args = parser.parse_args()
    count = int(args.count) if args.count else None

    config = yaml.safe_load(Path(args.config).read_text())
    sweep_id = wandb.sweep(sweep=config, project=args.project)
    for gpu in args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if count:
            wandb.agent(sweep_id, entity=wb_api.default_entity, project=args.project, count=count)
        else:
            wandb.agent(sweep_id, entity=wb_api.default_entity, project=args.project)


if __name__ == "__main__":
    main()
