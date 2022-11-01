# Imports
import os
import subprocess


# Build container image
os.environ["LAMBDA_NAME"] = "admirer-backend"
subprocess.run(["cd", ".."], shell=True)
subprocess.run(
    ["docker", "build", "--no-cache", "-t", os.environ["LAMBDA_NAME"], ".", "--file", "api_serverless/Dockerfile"]
)
subprocess.run(["cd", "backend_setup"], shell=True)


# Upload to the container registry
proc = subprocess.run(["aws", "sts", "get-caller-identity", "--query", "Account"], stdout=subprocess.PIPE, text=True)
aws_account_id = proc.stdout
proc = subprocess.run(["aws", "configure", "get", "region"], stdout=subprocess.PIPE, text=True)
aws_region = proc.stdout
os.environ["AWS_REGION"] = aws_region.strip("\n")
os.environ["AWS_ACCOUNT_ID"] = aws_account_id.replace('"', "").strip("\n")
os.environ["ECR_URI"] = ".".join(
    [os.environ["AWS_ACCOUNT_ID"], "dkr", "ecr", os.environ["AWS_REGION"], "amazonaws.com"]
)

subprocess.run(
    [
        "aws",
        "ecr",
        "get-login-password",
        "--region",
        os.environ["AWS_REGION"],
        "|",
        "docker",
        "login",
        "--username",
        "AWS",
        "--password-stdin",
        os.environ["ECR_URI"],
    ],
    shell=True,
)
subprocess.run(
    [
        "aws",
        "ecr",
        "create-repository",
        "--repository-name",
        os.environ["LAMBDA_NAME"],
        "--image-scanning-configuration",
        "scanOnPush=true",
        "--image-tag-mutability",
        "MUTABLE",
        "|",
        "jq",
        "-C",
    ],
    shell=True,
)

os.environ["IMAGE_URI"] = "/".join([os.environ["ECR_URI"], os.environ["LAMBDA_NAME"]])
subprocess.run(["docker", "tag", os.environ["LAMBDA_NAME"] + ":latest", os.environ["IMAGE_URI"] + ":latest"])
subprocess.run(["docker", "push", os.environ["IMAGE_URI"] + ":latest"])
