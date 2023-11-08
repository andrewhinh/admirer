# Imports
import os
import subprocess


def main():
    # Specify ECR repo name
    os.environ["LAMBDA_NAME"] = "admirer-backend"

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
            "./api_serverless/Dockerfile",
        ]
    )

    # Upload to the container registry
    subprocess.run(["docker", "tag", os.environ["LAMBDA_NAME"] + ":latest", os.environ["IMAGE_URI"] + ":latest"])
    subprocess.run(["docker", "push", os.environ["IMAGE_URI"] + ":latest"])

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
    main()
