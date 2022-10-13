# Imports
import argparse
import os
from pathlib import Path
import shutil

import wandb


# Variables
PROJECT = "admirer"
JOB_TYPE = "stage"
STAGED_MODEL_NAME = "answer"
STAGED_MODEL_TYPE = "prod-ready"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path("question_answer/")

PROD_STAGING_ROOT = PROJECT_ROOT / DIRECTORY / Path("models")
PROD_PATHS = ["coco_annotations", "coco_clip_new", "transformers"]

api = wandb.Api()
DEFAULT_ENTITY = api.default_entity


def get_logging_run(artifact):
    api_run = artifact.logged_by()
    return api_run


def print_info(artifact, run=None):
    run = get_logging_run(artifact)

    full_artifact_name = f"{artifact.entity}/{artifact.project}/{artifact.name}"
    print(f"Using artifact {full_artifact_name}")
    artifact_url_prefix = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}"
    artifact_url_suffix = f"{artifact.name.replace(':', '/')}"
    print(f"View at URL: {artifact_url_prefix}/{artifact_url_suffix}")

    print(f"Logged by {run.name} -- {run.project}/{run.entity}/{run.id}")
    print(f"View at URL: {run.url}")


def download_artifact(artifact_path):
    """Downloads the artifact at artifact_path to the target directory."""
    if wandb.run is not None:  # if we are inside a W&B run, track that we used this artifact
        artifact = wandb.use_artifact(artifact_path)
    else:  # otherwise, just download the artifact via the API
        artifact = api.artifact(artifact_path)
    artifact.download(root=PROJECT_ROOT / DIRECTORY)

    return artifact


def setup(fetch, override):
    # If overriding, remove local files; otherwise, make sure local files do not match all W&B files
    if fetch:
        if override:
            if os.path.exists(PROD_STAGING_ROOT):
                os.remove(PROD_STAGING_ROOT)
        else:
            if os.path.exists(PROD_STAGING_ROOT):
                return


def upload_staged_model(staged_at):
    os.mkdir(PROD_STAGING_ROOT)
    for path in PROD_PATHS:
        shutil.move(PROJECT_ROOT / DIRECTORY / path, PROD_STAGING_ROOT / path)
    staged_at.add_dir(PROD_STAGING_ROOT)
    wandb.log_artifact(staged_at)
    for path in PROD_PATHS:
        shutil.move(PROD_STAGING_ROOT / path, PROJECT_ROOT / DIRECTORY / path)
    os.rmdir(PROD_STAGING_ROOT)


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
        help=f"If provided, download the latest version of artifact files to {PROJECT / DIRECTORY}.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help=f"If provided, override local files at {PROJECT / DIRECTORY} with downloaded files.",
    )
    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    setup(args.fetch, args.override)
    main(args)
