# Imports
import argparse
import os
from pathlib import Path

import wandb


# Variables
PROJECT = "admirer"
JOB_TYPE = "stage"
STAGED_MODEL_NAME = "answer"
STAGED_MODEL_TYPE = "prod-ready"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path("question_answer/")
PROD_STAGING_ROOT_1 = Path("coco_annotations/")
CAPTIONS = "captions_train2014.json"
ANNOTATIONS = "mscoco_train2014_annotations.json"
QUESTIONS = "OpenEnded_mscoco_train2014_questions.json"
PROD_STAGING_ROOT_2 = Path("coco_clip_new/")
IMG_FEAT = "coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy"
QUES_FEAT = "coco_clip_vitb16_train2014_okvqa_question.npy"
IDXS = "okvqa_qa_line2sample_idx_train2014.json"

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
    files = []
    for file in [CAPTIONS, ANNOTATIONS, QUESTIONS]:
        files.append(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1 / file)
    for file in [IMG_FEAT, QUES_FEAT, IDXS]:
        files.append(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2 / file)

    # If overriding, remove local files; otherwise, make sure local files do not match all W&B files
    if fetch:
        count = 0
        for file in files:
            if override:
                if os.path.exists(file):
                    os.remove(file)
            else:
                if os.path.exists(file):
                    count += 1
        if count == 6:
            return


def fix_paths(override):
    # Put files into subdirectories to match format
    if not os.path.exists(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1):
        os.mkdir(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1)
    if not os.path.exists(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2):
        os.mkdir(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2)
    for file in [CAPTIONS, ANNOTATIONS, QUESTIONS]:
        if override:
            os.rename(PROJECT_ROOT / DIRECTORY / file, PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1 / file)
        else:
            if os.path.exists(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1 / file):
                os.remove(PROJECT_ROOT / DIRECTORY / file)
            else:
                os.rename(PROJECT_ROOT / DIRECTORY / file, PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1 / file)
    for file in [IMG_FEAT, QUES_FEAT, IDXS]:
        if override:
            os.rename(PROJECT_ROOT / DIRECTORY / file, PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2 / file)
        else:
            if os.path.exists(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2 / file):
                os.remove(PROJECT_ROOT / DIRECTORY / file)
            else:
                os.rename(PROJECT_ROOT / DIRECTORY / file, PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2 / file)


def upload_staged_model(staged_at):
    for file in [CAPTIONS, ANNOTATIONS, QUESTIONS]:
        staged_at.add_file(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_1 / file)
    for file in [IMG_FEAT, QUES_FEAT, IDXS]:
        staged_at.add_file(PROJECT_ROOT / DIRECTORY / PROD_STAGING_ROOT_2 / file)
    wandb.log_artifact(staged_at)


def main(args):
    if args.fetch:
        staged_files = f"{DEFAULT_ENTITY}/{PROJECT}/{STAGED_MODEL_NAME}:latest"
        artifact = download_artifact(staged_files)
        print_info(artifact)
        fix_paths(args.override)
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
