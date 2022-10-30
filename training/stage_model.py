# Imports
import argparse
import os
from pathlib import Path
from typing import Any

import wandb
from wandb import Artifact
from wandb.sdk.wandb_run import Run
from dotenv import load_dotenv

# Variables
PROJECT = "admirer"
JOB_TYPE = "stage"
STAGED_MODEL_NAME = "answer"
STAGED_MODEL_TYPE = "prod-ready"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path("question_answer/")

PROD_STAGING_ROOT = PROJECT_ROOT / DIRECTORY / Path("artifacts")
PROD_PATHS = ["coco_annotations", "coco_clip_new", "transformers", "onnx"]

# Load environtal variables from .env file
load_dotenv()

api = wandb.Api()
DEFAULT_ENTITY = api.default_entity


def get_logging_run(artifact: Artifact) -> Run:
    """Get the W&B run that logged the artifact"""
    api_run = artifact.logged_by()
    return api_run


def print_info(artifact: Artifact, run=None) -> None:
    """Prints info about the artifact and the run"""
    run = get_logging_run(artifact)

    full_artifact_name = f"{artifact.entity}/{artifact.project}/{artifact.name}"
    print(f"Using artifact {full_artifact_name}")
    artifact_url_prefix = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}"
    artifact_url_suffix = f"{artifact.name.replace(':', '/')}"
    print(f"View at URL: {artifact_url_prefix}/{artifact_url_suffix}")

    print(f"Logged by {run.name} -- {run.project}/{run.entity}/{run.id}")
    print(f"View at URL: {run.url}")


def download_artifact(artifact_path: str) -> Artifact:
    """Downloads the artifact at artifact_path to the target directory."""
    if wandb.run is not None:  # if we are inside a W&B run, track that we used this artifact
        artifact: Artifact = wandb.use_artifact(artifact_path)
    else:  # otherwise, just download the artifact via the API
        artifact: Artifact = api.artifact(artifact_path)
    artifact.download(root=PROD_STAGING_ROOT)

    return artifact


def setup(fetch, override) -> Any:
    # If overriding, remove local files; otherwise, make sure local files do not match all W&B files
    if fetch:
        if override:
            if os.path.exists(PROD_STAGING_ROOT):
                os.remove(PROD_STAGING_ROOT)
        else:
            if os.path.exists(PROD_STAGING_ROOT):
                return


def upload_staged_model(staged_at: Artifact) -> None:
    """Uploads a staged arfifact to W&B"""
    staged_at.add_dir(PROD_STAGING_ROOT)
    wandb.log_artifact(staged_at)


def main(args):
    if args.fetch:
        staged_files = f"{DEFAULT_ENTITY}/{PROJECT}/{STAGED_MODEL_NAME}:latest"
        artifact = download_artifact(staged_files)
        print_info(artifact)
        return

    with wandb.init(job_type=JOB_TYPE, project=PROJECT):
        staged_at = wandb.Artifact(STAGED_MODEL_NAME, type=STAGED_MODEL_TYPE)
        upload_staged_model(staged_at)


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"If provided, download the latest version of artifact files to {PROD_STAGING_ROOT}.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help=f"If provided, override local files at {PROD_STAGING_ROOT} with downloaded files.",
    )
    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    setup(args.fetch, args.override)
    main(args)
