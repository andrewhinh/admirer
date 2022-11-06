"""Stages a model for use in production.

If based on a checkpoint, the model is saved locally and uploaded to W&B.

If based on a model that is already uploaded, the model file is downloaded locally.

For details on how the W&B artifacts backing the checkpoints and models are handled,
see the documenation for stage_model.find_artifact.
"""
# Imports
import argparse
from pathlib import Path
import tempfile

from dotenv import load_dotenv
import torch
import wandb
from wandb import Artifact
from wandb.sdk.wandb_run import Run

from question_answer.lit_models import GPT2
from training.util import setup_data_and_model_from_args


# Variables
# these names are all set by the pl.loggers.WandbLogger
MODEL_CHECKPOINT_TYPE = "model"
BEST_CHECKPOINT_ALIAS = "best"
MODEL_CHECKPOINT_PATH = "model.ckpt"
LOG_DIR = Path("training") / "logs"

STAGED_MODEL_TYPE = "prod-ready"
STAGED_MODEL_FILENAME = "model.pt"  # standard nomenclature; pytorch_model.bin is also used

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path("question_answer/")

LITMODEL_CLASS = GPT2

api = wandb.Api()

DEFAULT_ENTITY = api.default_entity
DEFAULT_FROM_PROJECT = "admirer-training"
DEFAULT_TO_PROJECT = "admirer-training"
DEFAULT_STAGED_MODEL_NAME = "answer"

PROD_STAGING_ROOT = PROJECT_ROOT / DIRECTORY / Path("artifacts")
PROD_PATHS = ["coco_annotations", "coco_clip_new", "transformers", "onnx"]

load_dotenv()


def main(args):
    prod_staging_directory = PROD_STAGING_ROOT / args.staged_model_name
    prod_staging_directory.mkdir(exist_ok=True, parents=True)
    # if we're just fetching an already compiled model
    if args.fetch:
        staged_files = f"{DEFAULT_ENTITY}/{args.from_project}/{args.staged_model_name}:latest"
        artifact = download_artifact(staged_files, prod_staging_directory)
        print_info(artifact)
        return  # and we're done

    # otherwise, we'll need to download the weights, compile the model, and save it
    with wandb.init(
        job_type="stage", project=args.to_project, dir=LOG_DIR
    ):  # log staging to W&B so prod and training are connected
        # find the model checkpoint and retrieve its artifact name and an api handle
        if args.run:
            ckpt_at, ckpt_api = find_artifact(
                project=args.from_project, type=MODEL_CHECKPOINT_TYPE, alias=args.ckpt_alias, run=args.run
            )

            # get the run that produced that checkpoint
            logging_run = get_logging_run(ckpt_api)
            print_info(ckpt_api, logging_run)
            metadata = get_checkpoint_metadata(logging_run, ckpt_api)

            with tempfile.TemporaryDirectory() as tmp_dir:
                # download the checkpoint to a temporary directory
                download_artifact(ckpt_at, tmp_dir)
                # reload the model from that checkpoint
                model = load_model_from_checkpoint(metadata, directory=tmp_dir)
                # save the model to .pt in the staging directory
                save_model_to_pt(model, directory=prod_staging_directory / "transformers" / "trained_caption")

        # create an artifact for the staged, deployable model
        staged_at = wandb.Artifact(args.staged_model_name, type=STAGED_MODEL_TYPE)
        # upload the staged model so it can be downloaded elsewhere
        upload_staged_model(staged_at, from_directory=prod_staging_directory)


def find_artifact(project: str, type: str, alias: str, run=None):
    """Finds the artifact of a given type with a given alias under the entity and project.

    Parameters
    ----------
    project
        The project to find the artifact from.
    type
        The name of the type of the artifact.
    alias : str
        The alias for this artifact. This alias must be unique within the
        provided type for the run, if provided, or for the project,
        if the run is not provided.
    run : str
        Optionally, the run in which the artifact is located.

    Returns
    -------
    Tuple[path, artifact]
        An identifying path and an API handle for a matching artifact.
    """
    if run is not None:
        path = _find_artifact_run(project=project, type=type, run=run, alias=alias)
    else:
        path = _find_artifact_project(project=project, type=type, alias=alias)
    return path, api.artifact(path)


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


def get_checkpoint_metadata(run, checkpoint):
    config = run.config
    out = {"config": config}
    try:
        ckpt_filename = checkpoint.metadata["original_filename"]
        out["original_filename"] = ckpt_filename
        metric_key = checkpoint.metadata["ModelCheckpoint"]["monitor"]
        metric_score = checkpoint.metadata["score"]
        out[metric_key] = metric_score
    except KeyError:
        pass
    return out


def download_artifact(artifact_path: str, target_directory: Path) -> Artifact:
    """Downloads the artifact at artifact_path to the target directory."""
    if wandb.run is not None:  # if we are inside a W&B run, track that we used this artifact
        artifact: Artifact = wandb.use_artifact(artifact_path)
    else:  # otherwise, just download the artifact via the API
        artifact: Artifact = api.artifact(artifact_path)
    artifact.download(root=target_directory)

    return artifact


def load_model_from_checkpoint(ckpt_metadata, directory):
    config = ckpt_metadata["config"]
    args = argparse.Namespace(**config)

    _, model = setup_data_and_model_from_args(args)

    # load LightningModule from checkpoint
    pth = Path(directory) / MODEL_CHECKPOINT_PATH
    lit_model = LITMODEL_CLASS.load_from_checkpoint(
        checkpoint_path=pth, args=args, model=model.vit2gpt2, tokenizer=model.gpt2_tokenizer, strict=False
    )
    lit_model.eval()

    return lit_model


def save_model_to_pt(model, directory):
    path = Path(directory) / STAGED_MODEL_FILENAME
    torch.save(model.state_dict(), path)


def upload_staged_model(staged_at: Artifact, from_directory: Path) -> None:
    """Uploads a staged arfifact to W&B"""
    staged_at.add_dir(from_directory)
    wandb.log_artifact(staged_at)


def _find_artifact_run(project, type, run, alias):
    run_name = f"{DEFAULT_ENTITY}/{project}/{run}"
    api_run = api.run(run_name)
    artifacts = api_run.logged_artifacts()

    match = [art for art in artifacts if alias in art.aliases and art.type == type]
    if not match:
        raise ValueError(f"No artifact with alias {alias} found at {run_name} of type {type}")
    if len(match) > 1:
        raise ValueError(f"Multiple artifacts ({len(match)}) with alias {alias} found at {run_name} of type {type}")
    return f"{DEFAULT_ENTITY}/{project}/{match[0].name}"


def _find_artifact_project(project, type, alias):
    project_name = f"{DEFAULT_ENTITY}/{project}"
    api_project = api.project(project, entity=DEFAULT_ENTITY)
    api_artifact_types = api_project.artifacts_types()
    # loop through all artifact types in this project
    for artifact_type in api_artifact_types:
        if artifact_type.name != type:
            continue  # skipping those that don't match type
        collections = artifact_type.collections()
        # loop through all artifacts and their versions
        for collection in collections:
            versions = collection.versions()
            for version in versions:
                if alias in version.aliases:  # looking for the first one that matches the alias
                    return f"{project_name}/{version.name}"
        raise ValueError(f"Artifact with alias {alias} not found in type {type} in {project_name}")
    raise ValueError(f"Artifact type {type} not found. {project_name} could be private or not exist.")


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"If provided, download the latest version of artifact files to {PROD_STAGING_ROOT}.",
    )
    parser.add_argument(
        "--from_project",
        type=str,
        default=DEFAULT_FROM_PROJECT,
        help=f"Project from which to download the checkpoint. Default is {DEFAULT_FROM_PROJECT}",
    )
    parser.add_argument(
        "--to_project",
        type=str,
        default=DEFAULT_TO_PROJECT,
        help=f"Project to which to upload the compiled model. Default is {DEFAULT_TO_PROJECT}.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=f"Optionally, the name of a run to check for an artifact of type {MODEL_CHECKPOINT_TYPE} that has the provided CKPT_ALIAS. Default is None.",
    )
    parser.add_argument(
        "--ckpt_alias",
        type=str,
        default=BEST_CHECKPOINT_ALIAS,
        help=f"Alias that identifies which model checkpoint should be staged.The artifact's alias can be set manually or programmatically elsewhere. Default is '{BEST_CHECKPOINT_ALIAS}'.",
    )
    parser.add_argument(
        "--staged_model_name",
        type=str,
        default=DEFAULT_STAGED_MODEL_NAME,
        help=f"Name to give the staged model artifact. Default is '{DEFAULT_STAGED_MODEL_NAME}'.",
    )
    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)
