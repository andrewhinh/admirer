# Imports
import argparse
import os
import subprocess


# Code execution
def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ecr_repo_name",
        type=str,
        default="admirer-frontend",
        help="Name of the AWS ECR repo to deploy to",
    )
    parser.add_argument(
        "--pull_image",
        action="store_true",
        help="Whether to pull the image from the AWS ECR repo",
    )
    parser.add_argument(
        "--dockerfile_path",
        type=str,
        default="api_serverless/Dockerfile",
        help="Path to the Dockerfile to build",
    )
    parser.add_argument(
        "--update_lambda_func",
        action="store_true",
        help="Whether to update the AWS Lambda function",
    )
    return parser


def main(args):
    # Specify ECR repo name
    os.environ["LAMBDA_NAME"] = args.ecr_repo_name

    # Get image URI
    proc = subprocess.run(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--query",
            "Account",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    aws_account_id = proc.stdout
    proc = subprocess.run(
        [
            "aws",
            "configure",
            "get",
            "region",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    aws_region = proc.stdout
    os.environ["AWS_REGION"] = aws_region.strip("\n")
    os.environ["AWS_ACCOUNT_ID"] = aws_account_id.replace('"', "").strip("\n")
    os.environ["ECR_URI"] = ".".join(
        [os.environ["AWS_ACCOUNT_ID"], "dkr", "ecr", os.environ["AWS_REGION"], "amazonaws.com"]
    )
    os.environ["IMAGE_URI"] = "/".join([os.environ["ECR_URI"], os.environ["LAMBDA_NAME"]])

    # Whether to pull or push the image
    if args.pull_image:
        # Pull the image from the container registry
        subprocess.run(["docker", "pull", os.environ["IMAGE_URI"] + ":latest"])
    else:
        # Build container image
        subprocess.run(
            [
                "docker",
                "build",
                "--no-cache",
                "-t",
                os.environ["LAMBDA_NAME"],
                ".",
                "--file",
                args.dockerfile_path,
            ]
        )

        # Upload to the container registry
        subprocess.run(["docker", "tag", os.environ["LAMBDA_NAME"] + ":latest", os.environ["IMAGE_URI"] + ":latest"])
        subprocess.run(["docker", "push", os.environ["IMAGE_URI"] + ":latest"])

    # Whether to update the AWS Lambda function
    if args.update_lambda_func:
        # Update the AWS Lambda function accordingly
        proc = subprocess.run(
            [
                "aws",
                "lambda",
                "update-function-code",
                "--function-name",
                os.environ["LAMBDA_NAME"],
                "--image-uri",
                os.environ["IMAGE_URI"] + ":latest",
            ],
        )


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
